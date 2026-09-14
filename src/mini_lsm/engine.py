"""存储引擎主入口 —— 把 WAL、MemTable、SSTable、Compaction 串成可用的 KV 存储。

阶段 3 的能力边界:
    文件不再只增不减了 —— L0 攒够就归并到 L1,超预算再往 L2 压,
    压到最底层时墓碑被真正清掉。查询也从"挨个翻 L0"变成
    "L0 挨个试 + L1 以上二分定位"。

    还差两样:
      - **没有 Bloom Filter**:查一个不存在的 key,每一层还是要真的去读一次
        文件才能确定"不在里面"(阶段 4 解决)
      - **没有 MVCC**:读写用一把大锁串行化,也没有快照隔离(阶段 5 解决)

写路径(必须严格保持这个顺序):
    1. 先把变更追加到 WAL        ← 落盘,这是持久性的来源
    2. 再写入内存表              ← 快速生效
    3. 内存表满了就刷成 SSTable  ← 释放内存
    4. 文件攒多了就 compaction   ← 控制读放大
    顺序不能反。如果先改内存再写日志,那么"日志还没写完就崩溃"
    的情况下,内存里的改动会丢失,而调用方已经收到"成功"了。

读路径(从新到旧,先命中先返回):
    内存表
      → L0 从新到旧逐个试(键范围重叠,没法二分)
      → L1、L2…… 二分定位到唯一可能包含它的文件(层内不重叠)
    墓碑会**终止**查找 —— 它表示"这个键在此刻被删了",
    所以不需要、也不能再去更旧的文件里找。

崩溃安全的总原则:
    **manifest 是唯一的"有效文件集合"**。任何改动都遵循
    "先把新数据落盘 → 再原子替换 manifest → 最后删旧数据"。
    崩在任何一步,重启后要么是旧的一套、要么是新的一套,不存在中间态。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .compaction import (
    DEFAULT_L0_TRIGGER,
    DEFAULT_LEVEL_BUDGET,
    DEFAULT_LEVEL_FACTOR,
    CompactionTask,
    pick_task,
    plan_full_compaction,
)
from .errors import ClosedError, CorruptionError, InvalidArgumentError
from .iterator import MergingIterator
from .manifest import MANIFEST_FILENAME, FileMeta, Manifest
from .memtable import DEFAULT_CAPACITY_BYTES, MemTable
from .record import RecordType
from .sstable import (
    DEFAULT_BLOCK_SIZE,
    SUFFIX as SSTABLE_SUFFIX,
    TMP_SUFFIX as SSTABLE_TMP_SUFFIX,
    SSTableReader,
    SSTableWriter,
    parse_file_id,
    sstable_filename,
)
from .wal import ReplayResult, WAL

#: 默认的 WAL 文件名
WAL_FILENAME = "wal.log"

#: 默认层数(L0、L1、L2)
DEFAULT_NUM_LEVELS = 3

#: compaction 产出的单个文件的目标大小
DEFAULT_TARGET_FILE_SIZE = 2 * 1024 * 1024

#: maybe_compact 的轮数上界。策略正确时几轮就收敛,这个上界只是防死循环。
DEFAULT_MAX_COMPACTION_ROUNDS = 16


def to_bytes(value: object, name: str) -> bytes:
    """把用户输入统一转成 bytes。

    允许传 str(按 UTF-8 编码),这样交互式使用时不必到处写 b""。
    """
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise InvalidArgumentError(
        f"{name} 必须是 str 或 bytes,实际是 {type(value).__name__}"
    )


@dataclass
class EngineStats:
    """引擎当前状态的快照,用于监控和调试。"""

    memtable_entries: int = 0
    memtable_size: int = 0
    memtable_capacity: int = 0
    tombstones: int = 0
    wal_size: int = 0
    flush_pending: bool = False
    recovered_records: int = 0
    recovery_truncated: bool = False
    recovery_reason: str | None = None
    sstable_count: int = 0
    sstable_entries: int = 0
    sstable_bytes: int = 0
    flushes: int = 0
    compactions: int = 0
    level_files: list[int] = field(default_factory=list)
    level_bytes: list[int] = field(default_factory=list)

    @property
    def memtable_usage_ratio(self) -> float:
        if self.memtable_capacity <= 0:
            return 0.0
        return self.memtable_size / self.memtable_capacity

    def __str__(self) -> str:
        lines = [
            f"内存表: {self.memtable_entries} 条,"
            f" {self.memtable_size} / {self.memtable_capacity} 字节"
            f" ({self.memtable_usage_ratio:.1%})",
            f"墓碑:   {self.tombstones} 个",
            f"WAL:    {self.wal_size} 字节",
            f"待刷盘: {'是' if self.flush_pending else '否'}",
            f"SSTable: {self.sstable_count} 个文件,"
            f" {self.sstable_entries} 条,{self.sstable_bytes} 字节",
            f"刷盘/归并: {self.flushes} 次 / {self.compactions} 次",
        ]
        if self.level_files:
            layout = "  ".join(
                f"L{index}={files}个/{size}B"
                for index, (files, size) in enumerate(
                    zip(self.level_files, self.level_bytes)
                )
            )
            lines.append(f"分层:   {layout}")
        if self.recovered_records:
            note = "(含截断)" if self.recovery_truncated else ""
            lines.append(f"启动恢复: 重放 {self.recovered_records} 条记录{note}")
            if self.recovery_reason:
                lines.append(f"          {self.recovery_reason}")
        return "\n".join(lines)


class LSMEngine:
    """基于 LSM 思想的键值存储引擎。

    用法::

        with LSMEngine("./data") as db:
            db.put("name", "alice")
            print(db.get("name"))     # b'alice'
            db.delete("name")
            print(db.get("name"))     # None
    """

    def __init__(
        self,
        data_dir: str | Path,
        memtable_capacity: int = DEFAULT_CAPACITY_BYTES,
        wal_sync_on_write: bool = False,
        sstable_block_size: int = DEFAULT_BLOCK_SIZE,
        auto_flush: bool = True,
        num_levels: int = DEFAULT_NUM_LEVELS,
        l0_compaction_trigger: int = DEFAULT_L0_TRIGGER,
        level_size_budget: int = DEFAULT_LEVEL_BUDGET,
        level_size_factor: int = DEFAULT_LEVEL_FACTOR,
        target_file_size: int = DEFAULT_TARGET_FILE_SIZE,
        auto_compact: bool = True,
    ) -> None:
        """
        参数:
            data_dir:            数据目录,不存在会自动创建
            memtable_capacity:   内存表容量上限(字节),超过就自动刷盘
            wal_sync_on_write:   每次写入是否立即 fsync。
                                 False 能扛住进程崩溃;True 才能扛住断电。
            sstable_block_size:  SSTable 数据块的目标大小(字节)
            auto_flush:          内存表满了是否自动刷成 SSTable
            num_levels:          层数(至少 2)
            l0_compaction_trigger: L0 文件数达到多少就触发 compaction
            level_size_budget:   L1 的容量预算(字节)
            level_size_factor:   层容量增长系数
            target_file_size:    compaction 产出文件的目标大小(字节)
            auto_compact:        是否在刷盘后自动触发 compaction
        """
        if num_levels < 2:
            raise InvalidArgumentError("num_levels 至少为 2")
        if l0_compaction_trigger < 1:
            raise InvalidArgumentError("l0_compaction_trigger 至少为 1")

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.wal_path = self.data_dir / WAL_FILENAME

        self._lock = threading.RLock()
        self._closed = False
        self._memtable = MemTable(memtable_capacity)
        self._sstable_block_size = sstable_block_size
        self._auto_flush = auto_flush
        self._auto_compact = auto_compact
        self._num_levels = num_levels
        self._l0_trigger = l0_compaction_trigger
        self._level_budget = level_size_budget
        self._level_factor = level_size_factor
        self._target_file_size = target_file_size

        self._flush_count = 0
        self._compaction_count = 0
        self._readers: dict[int, SSTableReader] = {}
        self._next_file_id = 1

        # 先把磁盘上已有的 SSTable 挂上来(它们是"更旧"的数据),
        # 再重放 WAL 重建内存表(更新的数据)。读路径按这个新旧顺序找。
        self._load_state()

        self._wal = WAL(self.wal_path, sync_on_write=wal_sync_on_write)
        self._recovery: ReplayResult = self._recover()

    # ------------------------------------------------------------ 启动

    def _load_state(self) -> None:
        """从 manifest 恢复"当前有效的文件集合",并打开它们。"""
        manifest_path = self.data_dir / MANIFEST_FILENAME

        # 崩溃留下的半截文件:临时数据文件和临时 manifest 都清掉
        for stale in self.data_dir.glob(f"*{SSTABLE_TMP_SUFFIX}"):
            stale.unlink(missing_ok=True)
        manifest_path.with_name(manifest_path.name + ".tmp").unlink(missing_ok=True)

        self._manifest = Manifest(manifest_path, self._num_levels)

        if manifest_path.exists():
            self._manifest.load()
            # manifest 描述的是实际的数据布局,以它为准
            self._num_levels = self._manifest.num_levels
        else:
            self._migrate_from_directory_scan()

        self._verify_files_exist()
        self._remove_orphans()
        self._open_readers()

        self._next_file_id = max(
            self._manifest.next_file_id,
            max(self._manifest.file_ids(), default=0) + 1,
        )

    def _migrate_from_directory_scan(self) -> None:
        """目录里没有 manifest 时,把已有的 ``.sst`` 全部当作 L0。

        这是阶段 2 的数据目录格式 —— 那时文件只增不减,扫目录就足以恢复。
        这样老目录不用转换就能直接打开。
        """
        for path in self.data_dir.glob(f"*{SSTABLE_SUFFIX}"):
            file_id = parse_file_id(path)
            if file_id is None:
                continue        # 名字不是我们的格式,不碰
            with SSTableReader(path, file_id) as reader:
                if reader.entry_count == 0:
                    continue    # 空表不该存在,忽略
                self._manifest.levels[0].append(
                    FileMeta(
                        file_id=file_id,
                        level=0,
                        smallest=reader.first_key,
                        largest=reader.last_key,
                        entry_count=reader.entry_count,
                        size=reader.size,
                    )
                )

        self._manifest.levels[0].sort(key=lambda meta: meta.file_id, reverse=True)
        if not self._manifest.levels[0]:
            # 空目录:不写 manifest,保持"未初始化"的状态。
            #
            # 如果这里写了空 manifest,就等于把目录"占"下了 ——
            # 之后有人往这个目录里放一批 .sst(比如从阶段 2 的目录拷过来),
            # 下次启动会因为 manifest 已存在而跳过扫描,把它们当成孤儿删掉。
            # 不写就没有这个问题:文件数一直为 0 的目录,每次启动都重扫一遍。
            return

        self._next_file_id = (
            max(meta.file_id for meta in self._manifest.levels[0]) + 1
        )
        self._save_manifest()

    def _verify_files_exist(self) -> None:
        missing = [
            meta.file_id
            for meta in self._manifest.files()
            if not (self.data_dir / sstable_filename(meta.file_id)).exists()
        ]
        if missing:
            raise CorruptionError(
                f"manifest 引用的数据文件不见了:{missing}。"
                f"数据目录可能被外部改动过 —— 继续启动会静默丢数据"
            )

    def _remove_orphans(self) -> None:
        """删掉 manifest 没有引用的 ``.sst``。

        compaction 的顺序是"先写新文件 → 再改 manifest → 最后删旧文件"。
        崩在中间就会留下没人引用的文件。**它们是垃圾不是数据** ——
        因为此刻 manifest 指向的那一套已经是完整的了。
        """
        live = self._manifest.file_ids()
        for path in self.data_dir.glob(f"*{SSTABLE_SUFFIX}"):
            file_id = parse_file_id(path)
            if file_id is None or file_id in live:
                continue
            path.unlink(missing_ok=True)

    def _open_readers(self) -> None:
        """打开所有被引用的文件。

        宁可启动时就发现损坏,也不要等到某次查询才炸 ——
        那样"查不到"和"数据坏了"就没法区分了。
        """
        for meta in self._manifest.files():
            path = self.data_dir / sstable_filename(meta.file_id)
            try:
                self._readers[meta.file_id] = SSTableReader(path, meta.file_id)
            except CorruptionError as exc:
                raise CorruptionError(
                    f"SSTable 损坏,拒绝以可能丢数据的方式启动:{path}"
                ) from exc

    def _recover(self) -> ReplayResult:
        """重放 WAL,重建内存表。

        ``WAL.recover`` 会自动截断末尾的残骸(崩溃时写了一半的记录),
        所以这一步结束后,日志里的每个字节都是有效的。
        """
        result = self._wal.recover()

        for rec in result.records:
            if rec.rec_type is RecordType.PUT:
                self._memtable.put(rec.key, rec.value)
            elif rec.rec_type is RecordType.DELETE:
                self._memtable.delete(rec.key)

        return result

    def _save_manifest(self) -> None:
        """原子地把 manifest 落盘,并把目录项也刷一下。

        调用方必须保证被引用的数据文件**已经先 fsync 过了**。
        """
        self._manifest.next_file_id = self._next_file_id
        self._manifest.save()
        self._sync_dir()

    # ------------------------------------------------------------ 写入

    def put(self, key: object, value: object) -> None:
        """写入一个键值对。

        先写 WAL(持久化),再改内存表。返回即代表数据已经"不会因
        进程崩溃而丢失"(前提是 ``wal_sync_on_write`` 为 True 时才能
        扛住断电,否则只扛得住进程崩溃)。
        """
        kb = to_bytes(key, "key")
        vb = to_bytes(value, "value")

        with self._lock:
            self._ensure_open()
            self._wal.append(RecordType.PUT, kb, vb)
            self._memtable.put(kb, vb)
            self._after_write()

    def delete(self, key: object) -> None:
        """删除一个键。

        实际动作是写入一个墓碑。键在逻辑上立即不可见,但物理数据
        要等到 compaction 压到最底层才会被真正清除。
        """
        kb = to_bytes(key, "key")

        with self._lock:
            self._ensure_open()
            self._wal.append(RecordType.DELETE, kb)
            self._memtable.delete(kb)
            self._after_write()

    def _after_write(self) -> None:
        """写入之后的维护动作。"""
        if self._auto_flush and self._memtable.is_full:
            self.flush()

    def put_many(self, items: dict | list[tuple]) -> int:
        """批量写入,返回写入条数。

        注意:这不是一个原子事务 —— 中途失败会留下部分写入。
        真正的原子性要等阶段 5 的 MVCC。
        """
        pairs = list(items.items()) if isinstance(items, dict) else list(items)
        for key, value in pairs:
            self.put(key, value)
        return len(pairs)

    # ------------------------------------------------------------ 刷盘

    def flush(self) -> int:
        """把内存表刷成一个 SSTable,返回写入的条目数(0 表示没东西可刷)。

        **步骤顺序是这个方法唯一重要的东西**:

            1. 把内存表快照写成 SSTable 并 fsync   ← 数据先落到新地方
            2. 原子改名成正式文件
            3. 更新 manifest(L0 多了一个文件)      ← 从此它是"有效的一套"
            4. 换一个空的内存表
            5. 截断 WAL                            ← 最后才丢弃旧地方

        任何一步之后崩溃都是安全的:只要 WAL 还在,重放一遍就能重建内存表。
        重放是幂等的 —— 同样的 put 应用两次,结果一样。

        墓碑也会被写进 SSTable。不能在这里丢掉它们:
        更旧的 SSTable 里可能还有这个键的值,墓碑是唯一能压住它的东西。
        """
        with self._lock:
            self._ensure_open()

            if len(self._memtable) == 0:
                return 0

            entries = list(self._memtable.items())      # 含墓碑
            file_id = self._next_file_id
            final_path = self.data_dir / sstable_filename(file_id)
            tmp_path = self.data_dir / f"sst-{file_id:06d}{SSTABLE_TMP_SUFFIX}"

            writer = SSTableWriter(tmp_path, block_size=self._sstable_block_size)
            try:
                for key, value in entries:
                    writer.add(key, value)
                writer.finish()
            except Exception:
                writer.abort()
                tmp_path.unlink(missing_ok=True)
                raise

            # 原子改名:要么看到完整的 .sst,要么看不到
            os.replace(tmp_path, final_path)
            self._sync_dir()

            try:
                reader = SSTableReader(final_path, file_id)
            except CorruptionError:
                final_path.unlink(missing_ok=True)
                raise

            meta = FileMeta(
                file_id=file_id,
                level=0,
                smallest=reader.first_key,
                largest=reader.last_key,
                entry_count=reader.entry_count,
                size=reader.size,
            )

            # manifest 先指向新文件 —— 它已经 fsync 过了,可以安全被引用
            self._manifest.add(meta)
            self._next_file_id = file_id + 1
            self._save_manifest()

            self._readers[file_id] = reader
            self._memtable = MemTable(self._memtable.capacity_bytes)
            self._flush_count += 1

            self._wal.truncate()

            if self._auto_compact:
                self.maybe_compact()

            return len(entries)

    def _sync_dir(self) -> None:
        """把目录项刷到磁盘,确保改名结果持久化。

        Windows 不允许对目录调用 ``os.open``/``fsync``,所以这里静默跳过 ——
        在 Windows 上,改名本身是元数据操作,NTFS 会自己保证一致性。
        """
        try:
            fd = os.open(self.data_dir, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    # ------------------------------------------------------------ Compaction

    def maybe_compact(self) -> int:
        """按需做若干轮 compaction,返回实际执行的轮数。

        循环有上界。策略正确时几轮就收敛:一轮 L0 compaction 会把 L0 清空;
        往下一层压会让上一层变小。最底层不触发任务(见 ``pick_task`` 的
        循环上界),所以不存在"越压越多"的死循环。
        """
        with self._lock:
            self._ensure_open()
            rounds = 0
            while rounds < DEFAULT_MAX_COMPACTION_ROUNDS:
                task = pick_task(
                    self._manifest,
                    l0_trigger=self._l0_trigger,
                    level_budget_base=self._level_budget,
                    level_size_factor=self._level_factor,
                )
                if task is None:
                    break
                self._run_compaction(task)
                rounds += 1
            return rounds

    def compact(self) -> bool:
        """做**一轮** compaction。返回是否真的做了。

        想手动控制节奏时用它;想"该做就做"用 ``maybe_compact()``。
        """
        with self._lock:
            self._ensure_open()
            task = pick_task(
                self._manifest,
                l0_trigger=self._l0_trigger,
                level_budget_base=self._level_budget,
                level_size_factor=self._level_factor,
            )
            if task is None:
                return False
            self._run_compaction(task)
            return True

    def compact_all(self) -> int:
        """把所有层的数据一路压到最底层,返回参与归并的文件数。

        这一步会**真正清掉墓碑**(压到最底层后不可能再有更旧的数据),
        并把文件数压到最少。手动整理和测试时用。
        """
        with self._lock:
            self._ensure_open()
            task = plan_full_compaction(self._manifest)
            if task is None:
                return 0
            return self._run_compaction(task)

    def _run_compaction(self, task: CompactionTask) -> int:
        """执行一次 compaction。

        顺序:
            1. 归并输入,写成若干新文件并 fsync
            2. 原子更新 manifest —— 从此"有效的一套"就是新文件
            3. 换掉内存里的 reader
            4. 删旧文件 —— 到这里才安全,manifest 已经不引用它们了
        """
        added = self._merge_and_write(
            task.inputs, task.target_level, task.drop_tombstones
        )
        self._apply_compaction(task.inputs, added)
        self._compaction_count += 1
        return len(task.inputs)

    def _merge_and_write(
        self, inputs: list[FileMeta], target_level: int, drop_tombstones: bool
    ) -> list[FileMeta]:
        """归并输入文件并切成若干个新文件。

        ``inputs`` 必须**按新到旧**排列 —— 归并时同 key 取谁完全由它决定。
        只写文件,不碰 manifest、不删旧文件。
        """
        sources = [self._readers[meta.file_id].iter_entries() for meta in inputs]
        merged = MergingIterator(sources)

        added: list[FileMeta] = []
        writer: SSTableWriter | None = None
        tmp_path: Path | None = None
        file_id = 0

        try:
            for key, value in merged:
                # 压到最底层时,墓碑已经没有需要压住的对象了
                if value is None and drop_tombstones:
                    continue

                if writer is None:
                    file_id = self._next_file_id
                    self._next_file_id += 1
                    tmp_path = self.data_dir / f"sst-{file_id:06d}{SSTABLE_TMP_SUFFIX}"
                    writer = SSTableWriter(
                        tmp_path, block_size=self._sstable_block_size
                    )

                writer.add(key, value)

                if writer.bytes_written >= self._target_file_size:
                    added.append(
                        self._finish_output(writer, file_id, tmp_path, target_level)
                    )
                    writer = None
                    tmp_path = None

            if writer is not None:
                added.append(
                    self._finish_output(writer, file_id, tmp_path, target_level)
                )
                writer = None
        except Exception:
            # 出错就把这次已经写出的新文件全部清掉,不留下半成品
            if writer is not None:
                writer.abort()
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            for meta in added:
                (self.data_dir / sstable_filename(meta.file_id)).unlink(
                    missing_ok=True
                )
            raise

        return added

    def _finish_output(
        self, writer: SSTableWriter, file_id: int, tmp_path: Path, level: int
    ) -> FileMeta:
        """收尾一个输出文件:落盘、fsync、原子改名成正式文件。"""
        meta = writer.finish()
        final_path = self.data_dir / sstable_filename(file_id)
        os.replace(tmp_path, final_path)
        self._sync_dir()
        return FileMeta(
            file_id=file_id,
            level=level,
            smallest=meta.first_key,
            largest=meta.last_key,
            entry_count=meta.entry_count,
            size=meta.size,
        )

    def _apply_compaction(
        self, removed: list[FileMeta], added: list[FileMeta]
    ) -> None:
        """把一次 compaction 的结果落到磁盘和内存状态上。

        顺序不能变:新文件此刻**已经** fsync 过了(在 ``_merge_and_write``
        里做的),所以 manifest 可以先指向它们;而删旧文件必须放在
        manifest 落盘**之后**。
        """
        self._manifest.replace(removed, added)
        self._save_manifest()

        for meta in removed:
            reader = self._readers.pop(meta.file_id, None)
            if reader is not None:
                reader.close()
        for meta in added:
            path = self.data_dir / sstable_filename(meta.file_id)
            self._readers[meta.file_id] = SSTableReader(path, meta.file_id)

        # manifest 已经不再引用这些文件了,现在删才安全
        for meta in removed:
            (self.data_dir / sstable_filename(meta.file_id)).unlink(missing_ok=True)

    # ------------------------------------------------------------ 读取

    def get(self, key: object) -> bytes | None:
        """查询键。返回 ``None`` 表示不存在或已被删除。"""
        kb = to_bytes(key, "key")

        with self._lock:
            self._ensure_open()

            # 内存表最新,先查它
            found, value = self._memtable.get_entry(kb)
            if found:
                return value        # 可能是 None(墓碑)

            # L0:键范围互相重叠,只能从新到旧挨个试
            for meta in self._manifest.levels[0]:
                found, value = self._readers[meta.file_id].get(kb)
                if found:
                    return value    # 墓碑同样在这里终止查找

            # L1 及以上:层内不重叠,二分就能定位到唯一可能包含它的文件
            for level in range(1, self._num_levels):
                meta = self._manifest.find_file(level, kb)
                if meta is None:
                    continue
                found, value = self._readers[meta.file_id].get(kb)
                if found:
                    return value
            return None

    def get_str(self, key: object, encoding: str = "utf-8") -> str | None:
        """查询并解码成字符串,方便交互式使用。"""
        raw = self.get(key)
        return None if raw is None else raw.decode(encoding, errors="replace")

    def contains(self, key: object) -> bool:
        """键是否存在(墓碑算不存在)。"""
        return self.get(key) is not None

    def scan(
        self,
        start: object | None = None,
        end: object | None = None,
    ) -> Iterator[tuple[bytes, bytes]]:
        """按 key 升序扫描区间 ``[start, end)``。

        两端都可以省略:省略 start 表示从头开始,省略 end 表示扫到末尾。
        墓碑会被自动跳过。

        实现是把内存表和所有层的所有文件一起交给归并迭代器 ——
        对上层来说,"多个来源"和"一个有序表"没有区别。
        阶段 5 会把它换成惰性流式版本;现在的结果会先物化成列表,
        所以扫描一个很大的库会占不少内存。
        """
        with self._lock:
            self._ensure_open()
            # 在锁内先把结果物化成列表,避免迭代过程中被写入干扰
            lo = to_bytes(start, "start") if start is not None else b""
            hi = to_bytes(end, "end") if end is not None else None

            sources: list[Iterator[tuple[bytes, bytes | None]]] = [
                self._memtable.items()
            ]
            for meta in self._manifest.files():     # 已是"新到旧"
                sources.append(self._readers[meta.file_id].iter_entries())

            snapshot: list[tuple[bytes, bytes]] = []
            for key, value in MergingIterator(sources):
                if key < lo:
                    continue
                if hi is not None and key >= hi:
                    break
                if value is None:      # 墓碑,跳过
                    continue
                snapshot.append((key, value))

        return iter(snapshot)

    def keys(self) -> Iterator[bytes]:
        """所有存活键,升序。"""
        return (k for k, _ in self.scan())

    # ------------------------------------------------------------ 状态

    def stats(self) -> EngineStats:
        """返回当前状态快照。"""
        with self._lock:
            files = self._manifest.files()
            return EngineStats(
                memtable_entries=len(self._memtable),
                memtable_size=self._memtable.approximate_size,
                memtable_capacity=self._memtable.capacity_bytes,
                tombstones=self._memtable.tombstone_count,
                wal_size=self._wal.size,
                flush_pending=self._memtable.is_full,
                recovered_records=self._recovery.record_count,
                recovery_truncated=self._recovery.truncated,
                recovery_reason=self._recovery.reason,
                sstable_count=len(files),
                sstable_entries=sum(meta.entry_count for meta in files),
                sstable_bytes=sum(meta.size for meta in files),
                flushes=self._flush_count,
                compactions=self._compaction_count,
                level_files=[len(level) for level in self._manifest.levels],
                level_bytes=[
                    self._manifest.level_bytes(index)
                    for index in range(self._num_levels)
                ],
            )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def recovery_result(self) -> ReplayResult:
        """启动时那次恢复的完整结果,便于测试和诊断。"""
        return self._recovery

    @property
    def sstables(self) -> list[SSTableReader]:
        """当前挂载的 SSTable,**最新的在前**。"""
        return [self._readers[meta.file_id] for meta in self._manifest.files()]

    @property
    def manifest(self) -> Manifest:
        """当前版本清单。测试和诊断用。"""
        return self._manifest

    @property
    def num_levels(self) -> int:
        return self._num_levels

    # ------------------------------------------------------------ 生命周期

    def sync(self) -> None:
        """强制把 WAL 刷到磁盘。想扛断电就调用它。

        注意:内存表**不会**因为 sync 而落盘 —— 它的持久性由 WAL 保证。
        """
        with self._lock:
            self._ensure_open()
            self._wal.sync()

    def close(self) -> None:
        """关闭引擎(会先 fsync,确保已确认的写入真正落盘)。

        **不会**顺手把内存表刷成 SSTable —— 那样每次关闭都会留下一个
        很小的文件。内存表的数据由 WAL 保证,下次启动重放即可。
        """
        with self._lock:
            if self._closed:
                return
            self._wal.close()
            for reader in self._readers.values():
                reader.close()
            self._readers.clear()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedError()

    def __enter__(self) -> "LSMEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        state = "closed" if self._closed else "open"
        layout = "/".join(str(len(level)) for level in self._manifest.levels)
        return (
            f"<LSMEngine {self.data_dir} entries={len(self._memtable)} "
            f"levels={layout} {state}>"
        )
