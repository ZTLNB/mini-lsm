"""存储引擎主入口 —— 把 WAL、MemTable、SSTable 串成可用的 KV 存储。

阶段 2 的能力边界:
    数据现在能超过内存了 —— 内存表攒满就刷成 SSTable,重启时**只重放
    上次刷盘之后的那段 WAL**,不用把历史数据重放一遍。
    但还缺两样东西:
      - **没有 compaction**:删掉的键只是留下墓碑,SSTable 只会越攒越多,
        读一个 key 要挨个查(阶段 3 解决)
      - **没有 Bloom Filter**:查一个不存在的 key 要把每个 SSTable 都翻一遍
        (阶段 4 解决)

写路径(必须严格保持这个顺序):
    1. 先把变更追加到 WAL        ← 落盘,这是持久性的来源
    2. 再写入内存表              ← 快速生效
    3. 内存表满了就刷成 SSTable  ← 释放内存
    顺序不能反。如果先改内存再写日志,那么"日志还没写完就崩溃"
    的情况下,内存里的改动会丢失,而调用方已经收到"成功"了。

读路径(从新到旧,先命中先返回):
    内存表 → L0 最新的 SSTable → ... → L0 最旧的 SSTable
    墓碑会**终止**查找 —— 它表示"这个键在此刻被删了",
    所以不需要、也不能再去更旧的文件里找。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .errors import ClosedError, CorruptionError, InvalidArgumentError
from .iterator import MergingIterator
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
            f" {self.sstable_entries} 条,{self.sstable_bytes} 字节"
            f" (已刷盘 {self.flushes} 次)",
        ]
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
    ) -> None:
        """
        参数:
            data_dir:           数据目录,不存在会自动创建
            memtable_capacity:  内存表容量上限(字节),超过就自动刷盘
            wal_sync_on_write:  每次写入是否立即 fsync。
                                False 能扛住进程崩溃;True 才能扛住断电。
            sstable_block_size: SSTable 数据块的目标大小(字节)
            auto_flush:         内存表满了是否自动刷成 SSTable
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.wal_path = self.data_dir / WAL_FILENAME

        self._lock = threading.RLock()
        self._closed = False
        self._memtable = MemTable(memtable_capacity)
        self._sstable_block_size = sstable_block_size
        self._auto_flush = auto_flush
        self._flush_count = 0

        # 先把磁盘上已有的 SSTable 挂上来(它们是"更旧"的数据),
        # 再重放 WAL 重建内存表(更新的数据)。读路径按这个新旧顺序找。
        self._sstables: list[SSTableReader] = self._load_sstables()
        self._next_table_id = self._compute_next_table_id()

        # 先打开 WAL,再恢复 —— 恢复过程需要读日志
        self._wal = WAL(self.wal_path, sync_on_write=wal_sync_on_write)
        self._recovery: ReplayResult = self._recover()

    # ------------------------------------------------------------ 启动

    def _load_sstables(self) -> list[SSTableReader]:
        """扫描数据目录,打开所有 SSTable,**最新的排在前面**。

        为什么靠扫目录而不是维护一个清单?
            因为写入用的是"临时文件 + 原子改名",所以只要文件名对得上,
            文件内容就一定是完整的。这让启动逻辑简单到几乎没有出错空间。
            等到阶段 3 有了 compaction(需要成批原子地增删文件),
            才会引入 manifest。
        """
        # 崩溃可能留下没改完名的临时文件,直接清掉
        for stale in self.data_dir.glob(f"*{SSTABLE_TMP_SUFFIX}"):
            stale.unlink(missing_ok=True)

        readers: list[SSTableReader] = []
        for path in self.data_dir.glob(f"*{SSTABLE_SUFFIX}"):
            file_id = parse_file_id(path)
            if file_id is None:
                continue        # 不是我们的文件,不碰
            try:
                readers.append(SSTableReader(path, file_id))
            except CorruptionError as exc:
                raise CorruptionError(
                    f"SSTable 损坏,拒绝以可能丢数据的方式启动:{path}"
                ) from exc

        # 编号越大越新
        readers.sort(key=lambda reader: reader.file_id, reverse=True)
        return readers

    def _compute_next_table_id(self) -> int:
        if not self._sstables:
            return 1
        return max(reader.file_id for reader in self._sstables) + 1

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
            if self._auto_flush and self._memtable.is_full:
                self.flush()

    def delete(self, key: object) -> None:
        """删除一个键。

        实际动作是写入一个墓碑。键在逻辑上立即不可见,但物理数据
        要等到后续 compaction 才会被真正清除。
        """
        kb = to_bytes(key, "key")

        with self._lock:
            self._ensure_open()
            self._wal.append(RecordType.DELETE, kb)
            self._memtable.delete(kb)
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
            3. 把新表插到 L0 最前面(内存里)
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
            file_id = self._next_table_id
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

            # 到这里新表已经稳稳落盘,可以安全地丢掉内存里的旧副本
            self._sstables.insert(0, reader)
            self._memtable = MemTable(self._memtable.capacity_bytes)
            self._next_table_id += 1
            self._flush_count += 1

            self._wal.truncate()

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

            # 再按从新到旧查 SSTable
            for table in self._sstables:
                found, value = table.get(kb)
                if found:
                    return value    # 墓碑同样在这里终止查找
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

        实现是把内存表和所有 SSTable 一起交给归并迭代器 ——
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
            sources.extend(table.iter_entries() for table in self._sstables)

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
                sstable_count=len(self._sstables),
                sstable_entries=sum(t.entry_count for t in self._sstables),
                sstable_bytes=sum(t.size for t in self._sstables),
                flushes=self._flush_count,
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
        return list(self._sstables)

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
            for table in self._sstables:
                table.close()
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
        return (
            f"<LSMEngine {self.data_dir} entries={len(self._memtable)} "
            f"sstables={len(self._sstables)} {state}>"
        )
