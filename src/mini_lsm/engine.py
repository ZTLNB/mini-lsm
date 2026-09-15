"""存储引擎主入口 —— 把 WAL、MemTable、SSTable、Compaction 串成可用的 KV 存储。

阶段 5 的能力边界:
    读路径补上了最后两块:**一致视图**和**流式扫描**。

    - ``snapshot()`` 拿到某一时刻的只读视图,之后的所有写入对它不可见,
      且**跨多次 get/scan 都成立**(不是"每次查都看一眼当前状态")。
    - ``scan()`` 不再把结果物化成列表 —— 它返回一个惰性迭代器,
      内存占用只和"块大小 × 来源个数"有关,和结果集大小无关。
      而且迭代过程中**不持有引擎锁**,所以边扫边写不会被卡住。

    实现细节见 ``snapshot.py``。核心是:SSTable 不可变 + manifest 每次
    替换就是一个新版本,于是"某一刻的一致状态"可以精确表示成
    "内存表副本 + 层结构副本 + 一批被 pin 住的文件"。

    仍然没有的东西:没有 per-key 版本链(所以不支持"读历史某个时间点"),
    没有事务。

---
阶段 6:并发写(见下)

    阶段 5 结束时,读已经不阻塞写了,但**写和写之间、写和 compaction
    之间仍然互斥** —— 因为 ``put``/``flush``/``compaction`` 共用同一把
    可重入锁,而 ``flush`` 和 ``compaction`` 内部有大量磁盘 I/O。
    一把锁把"改一个字节的内存"和"写几 MB 的文件"保护在同一个临界区里,
    代价完全不成比例。

    这一版把一把大锁拆成三把,并引入**不可变内存表**:

        _maintenance_lock  串行化刷盘与归并(同一时刻只有一个维护操作)
        _append_lock       WAL 追加 + 内存表写入 + 内存表轮转
        _state_lock        结构状态(manifest、readers、快照、计数器)

    加锁顺序固定为 ``maintenance → append → state``,任何地方都不许反向。

    关键变化是**刷盘时只换指针,不搬数据**:

        内存表写满
          → 在 _append_lock 里把指针换掉(微秒级):
              满的那张变成"不可变内存表",新建一张空表接住后续写入
          → 释放 _append_lock,让其他写者继续往新表里写
          → 在锁外把不可变表慢慢写成 SSTable
          → 最后才更新 manifest、摘掉不可变表、重写 WAL

    于是:写入只在"换指针"那一瞬间被挡一下,不再陪着刷盘一起等磁盘。
    归并同理 —— 归并的慢 I/O 全程在 ``_state_lock`` 之外,所以
    **读也不会被归并卡住**(阶段 5 时读是要等的)。

    代价与新增的约束:

    - 读路径多了一层:活动内存表 → **不可变内存表** → L0 → L1……
      漏掉中间那一层,刷盘期间就会"偶发查不到刚写的数据"。
    - WAL 不能简单截断了。刷盘期间日志里同时有"刚被刷走的表"和
      "正在写的表"两段记录,截断会丢后者。所以改成 ``WAL.rewrite``:
      用当前内存表的内容整体替换日志,顺带去重(见 ``wal.py``)。
    - 崩溃恢复不需要改:日志里多出来的那一段(属于已刷盘的表)重放是
      幂等的,值本来就一致。半截的 ``wal.log.tmp`` 启动时清掉即可。

    参考:``_rotate_locked`` / ``_flush_immutable_locked`` /
    ``_rewrite_or_truncate_wal_locked``。

写路径(必须严格保持这个顺序):
    1. 先把变更追加到 WAL        ← 落盘,这是持久性的来源
    2. 再写入内存表              ← 快速生效
    3. 内存表满了就刷成 SSTable  ← 释放内存
    4. 文件攒多了就 compaction   ← 控制读放大
    顺序不能反。如果先改内存再写日志,那么"日志还没写完就崩溃"
    的情况下,内存里的改动会丢失,而调用方已经收到"成功"了。

    并发写入时,"1 和 2 必须是一个原子步骤"这个要求变得更要紧:
    如果两个线程各自追加了日志、却按相反的顺序改了内存表,那么重启重放
    日志得到的顺序就和崩溃前内存里的顺序不一致 —— 同一个键的最终值可能
    对不上。``_append_lock`` 保护的正是这一对操作。

读路径(从新到旧,先命中先返回):
    活动内存表
      → 不可变内存表(只在刷盘期间存在)
      → L0 从新到旧逐个试(键范围重叠,没法二分)
      → L1、L2…… 二分定位到唯一可能包含它的文件(层内不重叠)
    墓碑会**终止**查找 —— 它表示"这个键在此刻被删了",
    所以不需要、也不能再去更旧的文件里找。

    这条路径由 ``snapshot.search_levels`` 实现,实时读和快照读**共用同一份**。
    两边各写一份的话,迟早有一边忘了检查布隆过滤器、或者忘了
    "L0 要挨个试",而这类分叉的表现是"偶发查不到数据",极难定位。

崩溃安全的总原则:
    **manifest 是唯一的"有效文件集合"**。任何改动都遵循
    "先把新数据落盘 → 再原子替换 manifest → 最后删旧数据"。
    崩在任何一步,重启后要么是旧的一套、要么是新的一套,不存在中间态。

    快照给这条原则加了一个附加条件:**被快照 pin 住的文件不能删**,
    只能推迟到最后一个引用它的快照关闭之后再删(见 ``_pending_delete``)。

    并发写入再给刷盘加了一个附加条件:**不可变内存表只有在 manifest
    已经指向它的 SSTable 之后才能摘掉**。反过来的话,那张表既不在内存里、
    也不在 manifest 里,它的数据就只靠 WAL 兜着 —— 而 WAL 可能已经被重写了。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .block_cache import DEFAULT_CACHE_BYTES, BlockCache, CacheStats
from .bloom import DEFAULT_BITS_PER_KEY
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
from .snapshot import ScanCursor, Snapshot, search_levels
from .util import to_bytes
from .wal import ReplayResult, WAL

#: 默认的 WAL 文件名
WAL_FILENAME = "wal.log"

#: 默认层数(L0、L1、L2)
DEFAULT_NUM_LEVELS = 3

#: compaction 产出的单个文件的目标大小
DEFAULT_TARGET_FILE_SIZE = 2 * 1024 * 1024

#: maybe_compact 的轮数上界。策略正确时几轮就收敛,这个上界只是防死循环。
DEFAULT_MAX_COMPACTION_ROUNDS = 16

#: 内存表涨到容量的多少倍时,写入必须排队等刷盘(背压)。
#:
#: 平时写入**不排队** —— 抢不到维护权就直接返回,数据交给正在刷的那个人。
#: 但如果内存表已经涨到两倍容量还没人把它刷出去,说明写入速度已经超过
#: 刷盘速度了,这时候必须真的等一等,否则内存会被撑爆。
MEMTABLE_STALL_FACTOR = 2


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

    immutable_entries: int = 0
    """不可变内存表里的条目数。**只在刷盘期间大于 0**。

    它长期不为 0 说明有个刷盘卡住了 —— 那期间所有写入都压在这一层里,
    而它占的内存是活动内存表的整整一倍。
    """
    immutable_bytes: int = 0
    """不可变内存表的估算字节数。"""
    wal_rewrites: int = 0
    """WAL 被整体重写的次数。

    只在"刷盘后日志里还留着新内存表的记录"时才会发生(见 ``WAL.rewrite``);
    单线程顺序写入时内存表总是被刷空,所以这个数会一直是 0,走的是更便宜的截断。
    """

    bloom_rejections: int = 0
    """布隆过滤器挡下了多少次文件读 —— 也就是省掉了多少次磁盘 I/O。"""
    bloom_corrupt_files: int = 0
    """过滤器块损坏、已降级成"没有过滤器"的文件数。"""
    cache: CacheStats = field(default_factory=CacheStats)

    snapshots: int = 0
    """当前还活着的快照数(每个 ``scan()`` 迭代器算一个)。"""
    pinned_files: int = 0
    """被活快照 pin 住、因此**不能删**的文件数。"""
    pending_delete_files: int = 0
    """已经离开 manifest、但因为还被快照引用而推迟删除的文件数。

    它长期不为 0 说明有快照忘了关 —— 那批磁盘空间一直回收不了。
    """

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
        ]
        if self.immutable_entries or self.immutable_bytes:
            lines.append(
                f"不可变: {self.immutable_entries} 条,"
                f" {self.immutable_bytes} 字节  ← 正在刷盘,写入不受它影响"
            )
        lines += [
            f"墓碑:   {self.tombstones} 个",
            f"WAL:    {self.wal_size} 字节"
            + (f"(整体重写 {self.wal_rewrites} 次)"
               if self.wal_rewrites else ""),
            f"待刷盘: {'是' if self.flush_pending else '否'}",
            f"SSTable: {self.sstable_count} 个文件,"
            f" {self.sstable_entries} 条,{self.sstable_bytes} 字节",
            f"刷盘/归并: {self.flushes} 次 / {self.compactions} 次",
            f"过滤器: 挡下 {self.bloom_rejections} 次文件读"
            + (f"(有 {self.bloom_corrupt_files} 个过滤器损坏)"
               if self.bloom_corrupt_files else ""),
            str(self.cache),
        ]
        if self.snapshots or self.pending_delete_files:
            lines.append(
                f"快照:   {self.snapshots} 个活着,"
                f" pin 住 {self.pinned_files} 个文件,"
                f" 待删 {self.pending_delete_files} 个"
            )
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
        bloom_bits_per_key: int = DEFAULT_BITS_PER_KEY,
        block_cache_size: int = DEFAULT_CACHE_BYTES,
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
            bloom_bits_per_key:  每个 key 分给布隆过滤器多少 bit。
                                 10 对应约 1% 假阳性率;设 0 可以关掉过滤器。
            block_cache_size:    块缓存容量(字节),0 表示关闭缓存
        """
        if num_levels < 2:
            raise InvalidArgumentError("num_levels 至少为 2")
        if l0_compaction_trigger < 1:
            raise InvalidArgumentError("l0_compaction_trigger 至少为 1")
        if bloom_bits_per_key < 0:
            raise InvalidArgumentError("bloom_bits_per_key 不能为负数")
        if block_cache_size < 0:
            raise InvalidArgumentError("block_cache_size 不能为负数")

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.wal_path = self.data_dir / WAL_FILENAME

        # 三把锁,职责严格分开。加锁顺序固定为
        #     _maintenance_lock → _append_lock → _state_lock
        # 任何地方都不许反向获取,否则就是死锁。
        #
        # 为什么不继续用一把大锁:刷盘和归并内部有大量磁盘 I/O,
        # 而写入只是"追加几字节 + 改一个 dict 项"。让这两件事抢同一把锁,
        # 等于每次刷盘都把整个库停掉。
        self._maintenance_lock = threading.RLock()
        """串行化刷盘与归并 —— 同一时刻只允许一个维护操作在跑。

        它们都会改 manifest,而 manifest 是"整体原子替换"的:
        两个维护操作同时改,后写的那次会把前一次的改动整个覆盖掉,
        文件就从清单里凭空消失。所以这里必须串行。
        """
        self._append_lock = threading.RLock()
        """保护 WAL 追加 + 内存表写入 + 内存表轮转这一组操作。

        这三件事必须**成组原子**:日志里的顺序就是崩溃后重放出来的顺序,
        而内存表里的顺序决定了"同一个键谁赢"。两者不一致,
        重启前后的数据就会对不上(详见模块开头"写路径"一节)。
        """
        self._state_lock = threading.RLock()
        """保护结构状态:manifest、readers、快照表、待删清单、计数器。

        它**绝不在磁盘 I/O 期间持有** —— 这正是"归并不卡读"的原因。
        """

        self._closed = False
        self._memtable = MemTable(memtable_capacity)
        #: 正在刷盘的那张内存表。正常情况下是 None;
        #: 只有"满了、指针已经换掉、但还没写成 SSTable"这段时间才非空。
        self._immutable: MemTable | None = None
        self._sstable_block_size = sstable_block_size
        self._auto_flush = auto_flush
        self._auto_compact = auto_compact
        self._num_levels = num_levels
        self._l0_trigger = l0_compaction_trigger
        self._level_budget = level_size_budget
        self._level_factor = level_size_factor
        self._target_file_size = target_file_size
        self._bloom_bits_per_key = bloom_bits_per_key

        self._flush_count = 0
        self._compaction_count = 0
        self._bloom_rejections = 0
        self._wal_rewrite_count = 0
        self._readers: dict[int, SSTableReader] = {}
        self._next_file_id = 1

        # 活着的快照:snapshot_id → Snapshot。
        # 引擎**强引用**快照,快照**弱引用**引擎 —— 单向强引用,避免循环,
        # 快照的释放时机完全由调用方(或 GC)决定,不会被引擎拖着不放。
        self._snapshots: dict[int, Snapshot] = {}
        self._next_snapshot_id = 1

        # 已经离开 manifest、但还被某个快照引用的文件。
        # file_id → (meta, reader)。reader **不能提前关** —— 快照还要用它读。
        self._pending_delete: dict[
            int, tuple[FileMeta, SSTableReader | None]
        ] = {}

        # 所有 reader 共用同一份块缓存 —— 跨文件的读才有机会互相命中。
        # (每个文件各配一个缓存的话,缓存总量会随文件数线性膨胀)
        self._block_cache = BlockCache(block_cache_size)

        # 先把磁盘上已有的 SSTable 挂上来(它们是"更旧"的数据),
        # 再重放 WAL 重建内存表(更新的数据)。读路径按这个新旧顺序找。
        self._load_state()

        self._wal = WAL(self.wal_path, sync_on_write=wal_sync_on_write)
        self._recovery: ReplayResult = self._recover()

    # ------------------------------------------------------------ 启动

    def _load_state(self) -> None:
        """从 manifest 恢复"当前有效的文件集合",并打开它们。"""
        manifest_path = self.data_dir / MANIFEST_FILENAME

        # 崩溃留下的半截文件:临时数据文件、临时 manifest、临时 WAL 都清掉。
        #
        # 半截的 wal.log.tmp 可以放心删:它只在 "WAL.rewrite 写完临时文件、
        # 但还没来得及原子改名" 这个窗口里存在。那时正式的 wal.log 仍然完好
        # (它比临时文件更旧、更长),重放它得到的状态是对的 —— 见 WAL.rewrite。
        for stale in self.data_dir.glob(f"*{SSTABLE_TMP_SUFFIX}"):
            stale.unlink(missing_ok=True)
        manifest_path.with_name(manifest_path.name + ".tmp").unlink(missing_ok=True)
        self.wal_path.with_name(self.wal_path.name + ".tmp").unlink(missing_ok=True)

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
            with SSTableReader(path, file_id, cache=self._block_cache) as reader:
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
                self._readers[meta.file_id] = SSTableReader(
                    path, meta.file_id, cache=self._block_cache
                )
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

        **多个线程可以同时调用它。** 互相之间只在"追加日志 + 改内存表"
        这一小段上排队,不会陪着别人的刷盘一起等磁盘。
        """
        kb = to_bytes(key, "key")
        vb = to_bytes(value, "value")
        self._write_record(RecordType.PUT, kb, vb)

    def delete(self, key: object) -> None:
        """删除一个键。

        实际动作是写入一个墓碑。键在逻辑上立即不可见,但物理数据
        要等到 compaction 压到最底层才会被真正清除。
        """
        kb = to_bytes(key, "key")
        self._write_record(RecordType.DELETE, kb)

    def _write_record(
        self, rec_type: RecordType, kb: bytes, vb: bytes = b""
    ) -> None:
        """写入路径的唯一入口:``put`` 和 ``delete`` 都走这里。

        ``_append_lock`` 只圈住"追加日志 + 改内存表"这两步,因为
        它们必须是一个原子步骤(理由见模块开头的"写路径"一节)。

        ⚠️ 触发刷盘的判断**必须在释放 _append_lock 之后**才处理。
        刷盘要写几 MB 的文件,如果占着 ``_append_lock`` 去写,
        其他写者就只能干等 —— 那和不拆锁没有任何区别。
        """
        with self._append_lock:
            self._ensure_open()
            self._wal.append(rec_type, kb, vb)
            if rec_type is RecordType.DELETE:
                self._memtable.delete(kb)
            else:
                self._memtable.put(kb, vb)
            full = self._memtable.is_full

        if full and self._auto_flush:
            self._after_write()

    def _after_write(self) -> None:
        """写入之后的维护动作:内存表满了就把它刷出去。

        ⚠️ **抢不到维护权就直接返回,不排队**。这是这里最容易被写错的地方。

        排队看起来更"安全",实际上是白等:这一刻内存表满了,但我们的数据
        已经在 WAL 和内存表里了 —— 它不会丢,也不需要立刻变成 SSTable。
        而正在刷的那个人刷完之后会顺手再看一眼内存表(见 ``_drain_locked``),
        该刷的会一起刷掉。

        为什么非要这么抠:``_drain_locked`` 里除了刷盘还会跑 ``maybe_compact``,
        而它最多可以跑 ``DEFAULT_MAX_COMPACTION_ROUNDS`` 轮。让一堆写者
        排在后面等一场完整的刷盘 + 归并,尾延迟会难看到离谱
        (实测:16 次维护里有 11 次是在白等,合计 2.7 秒)。

        唯一的例外是内存表已经涨到 ``MEMTABLE_STALL_FACTOR`` 倍容量 ——
        那说明写入比刷盘快,再不等就要把内存撑爆,这时候才真的排队(背压)。
        """
        if self._maintenance_lock.acquire(blocking=False):
            try:
                if not self._closed:
                    self._drain_locked(force=False)
            finally:
                self._maintenance_lock.release()
            return

        memtable = self._memtable
        if memtable.approximate_size >= (
            MEMTABLE_STALL_FACTOR * memtable.capacity_bytes
        ):
            with self._maintenance_lock:
                if not self._closed:
                    self._drain_locked(force=False)

    def put_many(self, items: dict | list[tuple]) -> int:
        """批量写入,返回写入条数。

        ⚠️ **这不是一个原子事务** —— 它只是 `put` 的一个循环。
        中途失败(比如某个 value 类型不对)会留下**部分写入**,
        而且没有任何办法回滚。

        别指望阶段 5 解决了这个:阶段 5 加的是**读**的一致视图
        (快照),不是**写**的原子性。真正的批量原子写需要 WAL 里
        带事务边界标记 + 提交记录,是另一件事 —— 这个项目没做。

        另外它也不是为了"快":每次 `put` 都要单独过一遍锁和 WAL 追加。
        真要做批量导入,直接循环调用 `put` 效果一样。
        """
        pairs = list(items.items()) if isinstance(items, dict) else list(items)
        for key, value in pairs:
            self.put(key, value)
        return len(pairs)

    # ------------------------------------------------------------ 刷盘

    def flush(self) -> int:
        """把内存表刷成一个 SSTable,返回写入的条目数(0 表示没东西可刷)。

        这是**显式**刷盘:不管内存表满没满都刷,把当前所有数据都落成文件。

        **步骤顺序是这个方法唯一重要的东西**:

            1. 把内存表轮转掉(换指针)           ← 快,写者立刻可以继续
            2. 把不可变表写成 SSTable 并 fsync   ← 数据先落到新地方(慢,不持锁)
            3. 原子改名成正式文件
            4. 更新 manifest(L0 多了一个文件)    ← 从此它是"有效的一套"
            5. 摘掉不可变表
            6. 重写/截断 WAL                     ← 最后才丢弃旧地方

        第 1 步是阶段 6 才有的。阶段 5 里"写 SSTable"和"换内存表"是一起做的,
        而且全程占着锁,所以刷盘期间谁也别想写。现在数据在写文件之前就已经
        从内存表里搬走了(搬进不可变表),写者可以立刻往新表里写。

        任何一步之后崩溃都是安全的:只要 WAL 还在,重放一遍就能重建内存表。
        重放是幂等的 —— 同样的 put 应用两次,结果一样。

        墓碑也会被写进 SSTable。不能在这里丢掉它们:
        更旧的 SSTable 里可能还有这个键的值,墓碑是唯一能压住它的东西。
        """
        with self._maintenance_lock:
            self._ensure_open()
            return self._drain_locked(force=True)

    def _drain_locked(self, force: bool) -> int:
        """把该刷的内存表都刷掉,返回写入的条目总数。**必须持有 _maintenance_lock**。

        ``force=True`` 是显式 ``flush()`` 的语义:当前内存表也一起刷。
        ``force=False`` 是自动维护的语义:**只刷满的**。这一条很重要 ——
        自动路径如果也强制刷,那么每次内存表刚满、并发写入又刚好填了一点,
        就会不停地把半满的表也刷出去,文件数会失控。
        """
        total = 0

        # 先把积压的不可变表刷掉。它比当前内存表旧,必须先落地 ——
        # 否则后刷的(更新的)数据会排在它前面,读的时候旧值会盖住新值。
        while self._immutable is not None:
            total += self._flush_immutable_locked()

        if force or self._memtable.is_full:
            if len(self._memtable) > 0:
                self._rotate_locked()
                total += self._flush_immutable_locked()

        if total and self._auto_compact:
            self.maybe_compact()

        return total

    def _rotate_locked(self) -> None:
        """把当前内存表换成一张空的,旧的那张变成不可变表。

        **这是整个并发设计的关键一步**:它只做两次指针赋值,不碰磁盘,
        所以耗时可忽略。写者最多在这里被挡一下,不用等刷盘。

        为什么必须在 ``_append_lock`` 里做:轮转改变了"日志后半段对应哪张表"。
        如果换指针的时候有线程正追加到一半,日志的顺序和内存表的顺序
        就对不上了 —— 而崩溃恢复正是按日志顺序重放的。
        """
        with self._append_lock:
            with self._state_lock:
                old = self._memtable
                self._immutable = old
                self._memtable = MemTable(old.capacity_bytes)

    def _flush_immutable_locked(self) -> int:
        """把不可变内存表写成 SSTable,返回条目数。**必须持有 _maintenance_lock**。

        这个方法里**没有一处长时间持锁** —— 这是它存在的全部意义。
        每次需要碰共享状态时,都只短暂地取一下 ``_state_lock``:

            取条目、分配 file_id   → 短暂持锁
            写文件、fsync、改名     → 不持任何锁   ← 慢的地方
            更新 manifest、摘表     → 短暂持锁
            重写/截断 WAL           → 持 _append_lock(有界)
        """
        with self._state_lock:
            imm = self._immutable
            if imm is None:
                return 0
            entries = list(imm.items())         # 含墓碑
            file_id = self._alloc_file_id()

        if not entries:
            # 轮转时不会换一张空表进来,所以这里理论上到不了。
            # 真到了就摘掉它,否则上面的 while 会卡住。
            with self._state_lock:
                self._immutable = None
            return 0

        final_path = self.data_dir / sstable_filename(file_id)
        tmp_path = self.data_dir / f"sst-{file_id:06d}{SSTABLE_TMP_SUFFIX}"

        writer = SSTableWriter(
            tmp_path,
            block_size=self._sstable_block_size,
            bloom_bits_per_key=self._bloom_bits_per_key,
        )
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
            reader = SSTableReader(final_path, file_id, cache=self._block_cache)
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

        with self._state_lock:
            # manifest 先指向新文件 —— 它已经 fsync 过了,可以安全被引用
            self._manifest.add(meta)
            self._save_manifest()

            self._readers[file_id] = reader
            self._flush_count += 1

            # ⚠️ 摘掉不可变表必须放在 manifest 更新**之后**。
            # 反过来的话,会出现一个瞬间:数据既不在内存里、也不在 manifest 里。
            # 如果恰好在那一刻崩溃,而 WAL 又已经被重写过,数据就真丢了。
            self._immutable = None

        self._rewrite_or_truncate_wal_locked()
        return len(entries)

    def _rewrite_or_truncate_wal_locked(self) -> None:
        """刷盘之后,把日志里"已经进了 SSTable 的那一段"丢掉。

        **能截断就直接截断**:当前内存表是空的 —— 说明日志里全是刚刷走的那张表,
        数据已经在 SSTable 里了,清掉即可。这是单线程顺序写入下的常态,
        也是唯一一条不需要额外 I/O 的路径(所以这个项目里它的开销是 0)。

        **否则整体重写**:并发写入时日志里还夹着新内存表的记录,截断会丢数据。
        这时用当前内存表的内容整体替换日志 —— 顺带把重复的键去掉了
        (内存表里每个键只有一条,而日志里同一个键可能追加过很多次)。

        ⚠️ 重写期间必须独占 ``_append_lock``,理由见 ``WAL.rewrite``。
        代价有界:最多是内存表容量那么多字节。
        """
        with self._append_lock:
            with self._state_lock:
                items = list(self._memtable.items())

            if not items:
                self._wal.truncate()
                return

            self._wal.rewrite(items)
            self._sync_dir()

            with self._state_lock:
                self._wal_rewrite_count += 1

    def _alloc_file_id(self) -> int:
        """分配一个文件 id。**必须在 _state_lock 里调用**。

        并发下这里必须是原子的:两个维护操作同时抢到同一个 id,
        后写的那个会把前一个的文件覆盖掉,而 manifest 里两条记录指向同一个文件。
        (``_maintenance_lock`` 已经保证了同一时刻只有一个维护操作,
        但把 id 分配收进锁里更不容易出错,代价也可以忽略。)
        """
        file_id = self._next_file_id
        self._next_file_id += 1
        return file_id

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

        ``_maintenance_lock`` 是**可重入**的,所以 ``flush()`` 内部可以直接
        调它 —— 不需要"先放锁再取锁"那套绕法,也不会真的嵌套两次维护。
        """
        with self._maintenance_lock:
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
                self._run_compaction_locked(task)
                rounds += 1
            return rounds

    def compact(self) -> bool:
        """做**一轮** compaction。返回是否真的做了。

        想手动控制节奏时用它;想"该做就做"用 ``maybe_compact()``。
        """
        with self._maintenance_lock:
            self._ensure_open()
            task = pick_task(
                self._manifest,
                l0_trigger=self._l0_trigger,
                level_budget_base=self._level_budget,
                level_size_factor=self._level_factor,
            )
            if task is None:
                return False
            self._run_compaction_locked(task)
            return True

    def compact_all(self) -> int:
        """把所有层的数据一路压到最底层,返回参与归并的文件数。

        这一步会**真正清掉墓碑**(压到最底层后不可能再有更旧的数据),
        并把文件数压到最少。手动整理和测试时用。
        """
        with self._maintenance_lock:
            self._ensure_open()
            task = plan_full_compaction(self._manifest)
            if task is None:
                return 0
            return self._run_compaction_locked(task)

    def _run_compaction_locked(self, task: CompactionTask) -> int:
        """执行一次 compaction。**必须持有 _maintenance_lock**。

        顺序:
            1. 归并输入,写成若干新文件并 fsync   ← 慢 I/O,不持 _state_lock
            2. 原子更新 manifest —— 从此"有效的一套"就是新文件
            3. 换掉内存里的 reader
            4. 删旧文件 —— 到这里才安全,manifest 已经不引用它们了

        为什么第 1 步可以不持 ``_state_lock``:
            输入文件是不可变的(SSTable 写完之后从不修改),而且
            ``_maintenance_lock`` 保证了没有别的维护操作会来删它们。
            所以**读路径在这期间可以照常跑** —— 阶段 5 时两者是要互相等的,
            归并一个几 MB 的库会把所有读卡住全程。

        输入文件的 reader 也不会被谁关掉:能关 reader 的只有
        ``_apply_compaction`` 和快照释放,而前者同样要 ``_maintenance_lock``。
        """
        added = self._merge_and_write(
            task.inputs, task.target_level, task.drop_tombstones
        )
        with self._state_lock:
            self._apply_compaction(task.inputs, added)
            self._compaction_count += 1
        return len(task.inputs)

    def _merge_and_write(
        self, inputs: list[FileMeta], target_level: int, drop_tombstones: bool
    ) -> list[FileMeta]:
        """归并输入文件并切成若干个新文件。

        ``inputs`` 必须**按新到旧**排列 —— 归并时同 key 取谁完全由它决定。
        只写文件,不碰 manifest、不删旧文件。

        整个方法跑在 ``_state_lock`` 之外(只在取 reader 和分配 file_id 时
        短暂进去一下),所以归并期间读不会被卡住。
        """
        with self._state_lock:
            sources = [
                self._readers[meta.file_id].iter_entries() for meta in inputs
            ]
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
                    with self._state_lock:
                        file_id = self._alloc_file_id()
                    tmp_path = self.data_dir / f"sst-{file_id:06d}{SSTABLE_TMP_SUFFIX}"
                    writer = SSTableWriter(
                        tmp_path,
                        block_size=self._sstable_block_size,
                        bloom_bits_per_key=self._bloom_bits_per_key,
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

        快照带来一个额外的岔路:如果某个旧文件还被活着的快照引用,
        就**既不能关 reader 也不能删文件**,只能登记进 ``_pending_delete``,
        等最后一个引用它的快照关掉再清理。
        """
        self._manifest.replace(removed, added)
        self._save_manifest()

        pinned = self._pinned_file_ids()

        for meta in removed:
            reader = self._readers.pop(meta.file_id, None)
            if meta.file_id in pinned:
                # 快照还要读它。reader 留着(快照自己持有一份引用),
                # 缓存里的块也留着 —— 快照扫描时正好还能命中。
                self._pending_delete[meta.file_id] = (meta, reader)
                continue

            if reader is not None:
                reader.close()
            # 顺手把它在缓存里的块清掉。不清也不会读到脏数据(file_id
            # 不复用),但这些块再也不会被访问,留着纯粹是占内存。
            self._block_cache.evict_file(meta.file_id)
            # manifest 已经不再引用它了,现在删才安全
            (self.data_dir / sstable_filename(meta.file_id)).unlink(
                missing_ok=True
            )

        for meta in added:
            path = self.data_dir / sstable_filename(meta.file_id)
            self._readers[meta.file_id] = SSTableReader(
                path, meta.file_id, cache=self._block_cache
            )

    # ------------------------------------------------------------ 快照

    def snapshot(self) -> Snapshot:
        """创建一份一致只读视图。

        返回的 ``Snapshot`` 会**pin 住它用到的所有文件** —— 之后即使
        compaction 想删它们,也只能推迟。所以**一定要关掉**::

            with db.snapshot() as snap:
                print(snap.get("k"))

        要扫描区间直接用 ``db.scan()``(它内部就是这么做的),
        不需要自己开快照。
        """
        with self._state_lock:
            self._ensure_open()

            snap = Snapshot(
                engine=self,
                snapshot_id=self._next_snapshot_id,
                # 内存表此刻的有序副本(活动表 + 不可变表)。
                # 建列表之后就各自独立了,之后的 put/delete 改的是原 dict,
                # 碰不到这一份。
                mem_items=self._merged_mem_items_locked(),
                # 传 manifest.levels 本身 —— Snapshot 内部会逐层复制。
                levels=self._manifest.levels,
                readers=self._readers,
            )
            self._snapshots[snap.snapshot_id] = snap
            self._next_snapshot_id += 1
            return snap

    def _merged_mem_items_locked(self) -> list[tuple[bytes, bytes | None]]:
        """活动内存表 + 不可变内存表的合并视图,按 key 升序。**需持有 _state_lock**。

        为什么要合并:不可变表**比所有 SSTable 都新** —— 它只是还没来得及
        写成文件而已。快照只认这一份列表,漏掉不可变表,刷盘期间创建的快照
        就会看不到那批刚写进去的数据(表现为"偶发少了几条",极难复现)。

        同一个键两边都有时**活动表优先**,因为它更晚写。
        """
        active = list(self._memtable.items())
        imm = self._immutable
        if imm is None or len(imm) == 0:
            return active

        # 不可变表更旧,先铺底;活动表覆盖上去
        merged: dict[bytes, bytes | None] = dict(imm.items())
        merged.update(active)
        return sorted(merged.items())

    def _pinned_file_ids(self) -> set[int]:
        """所有活快照引用到的文件 id 的并集。

        只在 compaction 和快照释放时算 —— 这两处都不是热路径,
        所以直接遍历所有快照就够了,不必额外维护引用计数。
        """
        pinned: set[int] = set()
        for snap in self._snapshots.values():
            pinned |= snap.file_ids
        return pinned

    def _release_snapshot(self, snap: Snapshot) -> None:
        """快照关闭时回调:解除 pin,顺手回收已经没人引用的待删文件。

        注意这里**不碰** ``self._readers`` —— 待删文件的 reader 早就被
        从里面摘掉了,它现在只活在 ``_pending_delete`` 和快照手里。
        """
        with self._state_lock:
            self._snapshots.pop(snap.snapshot_id, None)
            if not self._pending_delete:
                return
            still_pinned = self._pinned_file_ids()
            for file_id in list(self._pending_delete):
                if file_id in still_pinned:
                    continue        # 还有别的快照在看它,继续等
                _meta, reader = self._pending_delete.pop(file_id)
                if reader is not None:
                    reader.close()
                self._block_cache.evict_file(file_id)
                (self.data_dir / sstable_filename(file_id)).unlink(
                    missing_ok=True
                )

    def _drain_pending_delete(self) -> None:
        """把所有待删文件真的删掉。**只在引擎关闭时调用**。

        引擎要关了,快照也一并失效,所以这里不需要再管 pin。
        """
        for file_id, (_meta, reader) in self._pending_delete.items():
            if reader is not None:
                reader.close()
            (self.data_dir / sstable_filename(file_id)).unlink(missing_ok=True)
        self._pending_delete.clear()

    # ------------------------------------------------------------ 读取

    def get_entry(self, key: object) -> tuple[bool, bytes | None]:
        """返回 ``(是否存在, 值)``,和 ``MemTable.get_entry`` 的口径一致。

        ``(True, None)`` 表示墓碑(写过又删了),``(False, None)`` 表示
        从未出现过。``get()`` 把两者都折叠成 ``None``;需要区分时用这个。

        真正"翻层"的逻辑在 ``snapshot.search_levels`` 里 ——
        和快照读共用同一份,免得两边慢慢走岔。
        """
        kb = to_bytes(key, "key")

        with self._state_lock:
            self._ensure_open()

            # 内存表最新,先查它
            found, value = self._memtable.get_entry(kb)
            if found:
                return (True, value)    # 可能是 None(墓碑)

            # 其次是不可变内存表。它只在刷盘期间存在,但那一刻它比所有
            # SSTable 都新 —— 漏掉这一层,刷盘期间就会"偶发查不到刚写的数据"。
            imm = self._immutable
            if imm is not None:
                found, value = imm.get_entry(kb)
                if found:
                    return (True, value)

            found, value, rejections = search_levels(
                self._manifest.levels, self._readers, kb
            )
            self._bloom_rejections += rejections
            return (found, value)

    def get(self, key: object) -> bytes | None:
        """查询键。返回 ``None`` 表示不存在或已被删除。

        每一层都先问布隆过滤器"这个文件里一定没有这个 key 吗":
        答"一定没有"就跳过,**一次磁盘读都省了**。这是阶段 4 的主要收益 ——
        查不存在的键不再需要把每层都真读一遍。
        """
        found, value = self.get_entry(key)
        return value if found else None

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
    ) -> ScanCursor:
        """按 key 升序**流式**扫描区间 ``[start, end)``。

        两端都可以省略:省略 start 表示从头开始,省略 end 表示扫到末尾。
        墓碑会被自动跳过。

        和阶段 4 的三个区别,都是这个返回值带来的:

        1. **惰性**:读一个块才解析一个块。扫一个比内存还大的库也不会 OOM。
        2. **不阻塞写**:迭代期间**不持有引擎锁**,靠 pin 住文件保证数据还在。
        3. **一致**:一调用就定格。``it = db.scan(); db.put(...)`` 之后再迭代,
           看到的仍然是调用那一刻的状态 —— 而不是"扫到哪算哪"的混合视图。

        返回的迭代器持有快照,扫完自动释放;中途 ``break`` 掉则由对象回收兜底。
        想更精确地控制生命周期,用 ``with db.snapshot()`` 自己管::

            with db.snapshot() as snap:
                for key, value in snap.scan("a", "z"):
                    ...
        """
        snap = self.snapshot()
        return ScanCursor(snap, snap.scan(start, end))

    def keys(
        self,
        start: object | None = None,
        end: object | None = None,
    ) -> Iterator[bytes]:
        """区间内所有存活键,升序。"""
        return (key for key, _ in self.scan(start, end))

    # ------------------------------------------------------------ 状态

    def stats(self) -> EngineStats:
        """返回当前状态快照。"""
        with self._state_lock:
            files = self._manifest.files()
            imm = self._immutable
            return EngineStats(
                memtable_entries=len(self._memtable),
                memtable_size=self._memtable.approximate_size,
                memtable_capacity=self._memtable.capacity_bytes,
                tombstones=self._memtable.tombstone_count,
                wal_size=self._wal.size,
                flush_pending=self._memtable.is_full,
                immutable_entries=0 if imm is None else len(imm),
                immutable_bytes=0 if imm is None else imm.approximate_size,
                wal_rewrites=self._wal_rewrite_count,
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
                bloom_rejections=self._bloom_rejections,
                bloom_corrupt_files=sum(
                    1 for reader in self._readers.values() if reader.bloom_corrupt
                ),
                cache=self._block_cache.stats(),
                snapshots=len(self._snapshots),
                pinned_files=len(self._pinned_file_ids()),
                pending_delete_files=len(self._pending_delete),
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

    @property
    def block_cache(self) -> BlockCache:
        """引擎共用的块缓存。测试和诊断用。"""
        return self._block_cache

    @property
    def bloom_bits_per_key(self) -> int:
        """写新文件时给布隆过滤器分配的 bits/key(0 表示不写过滤器)。"""
        return self._bloom_bits_per_key

    @property
    def snapshot_count(self) -> int:
        """当前活着的快照数。每个未耗尽的 ``scan()`` 迭代器算一个。"""
        with self._state_lock:
            return len(self._snapshots)

    @property
    def pending_delete_files(self) -> list[int]:
        """已经离开 manifest、但还在等快照释放的文件 id,升序。

        它长期不空就说明有快照忘了关 —— 那批磁盘空间回收不了。
        """
        with self._state_lock:
            return sorted(self._pending_delete)

    @property
    def pinned_file_ids(self) -> list[int]:
        """当前被快照 pin 住的文件 id,升序。"""
        with self._state_lock:
            return sorted(self._pinned_file_ids())

    @property
    def immutable_entries(self) -> int:
        """正在刷盘的那张不可变内存表里有多少条(没有则为 0)。

        它**长期不为 0** 说明刷盘卡住了 —— 那期间写入都堆在活动内存表里,
        而内存占用是平时的两倍。
        """
        with self._state_lock:
            imm = self._immutable
            return 0 if imm is None else len(imm)

    # ------------------------------------------------------------ 生命周期

    def sync(self) -> None:
        """强制把 WAL 刷到磁盘。想扛断电就调用它。

        注意:内存表**不会**因为 sync 而落盘 —— 它的持久性由 WAL 保证。

        这里用 ``_append_lock`` 而不是 ``_state_lock``:它保护的是写路径,
        和"结构状态"无关;而且 fsync 是 I/O,不该占着 ``_state_lock`` 做,
        否则一次 sync 会把所有读也一起挡住。
        """
        with self._append_lock:
            self._ensure_open()
            self._wal.sync()

    def close(self) -> None:
        """关闭引擎(会先 fsync,确保已确认的写入真正落盘)。

        **不会**顺手把内存表刷成 SSTable —— 那样每次关闭都会留下一个
        很小的文件。内存表的数据由 WAL 保证,下次启动重放即可。

        已经发出去的快照会一并失效(再读会抛 ``ClosedError``),
        它们 pin 住的待删文件也在这里真正删掉 —— 引擎都关了,
        没有"还在读"的可能了。

        三层锁按 ``maintenance → append → state`` 的顺序一次性全拿下来,
        意义是:**等所有在飞的维护操作和写入都结束**,再动手清理。
        特别是 ``_append_lock`` —— 拿到它就说明没有线程正追加到一半,
        此后 ``_closed`` 置位,后续写入一律被 ``_ensure_open`` 挡掉。
        """
        with self._maintenance_lock:
            with self._append_lock:
                with self._state_lock:
                    if self._closed:
                        return

                    # 先失效快照。注意用 _invalidate 而不是 close:后者会回调
                    # _release_snapshot,而这里正持着锁,重入容易出岔子。
                    for snap in list(self._snapshots.values()):
                        snap._invalidate()
                    self._snapshots.clear()

                    self._wal.close()
                    for reader in self._readers.values():
                        reader.close()
                    self._readers.clear()

                    self._drain_pending_delete()
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
