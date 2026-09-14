"""Compaction 的策略层。

Compaction 要解决阶段 2 留下的两个问题:

    1. **文件只增不减** —— 删掉的键只留下墓碑,SSTable 越攒越多。
       demo 里 8 KiB 内存表写 3000 条记录就刷出了 49 个文件。
    2. **读放大** —— L0 的文件键范围互相重叠,查一个 key 最坏要把
       L0 全部翻一遍。

做法是**分层归并**:

    L0 文件数到阈值 ──► 整个 L0 + L1 里键范围重叠的文件一起归并 ──► 写回 L1
    L1 超过容量预算 ──► 挑一个 L1 文件 + L2 里重叠的文件归并 ──► 写回 L2
    压到最底层时,墓碑可以安全丢掉

两个必须说清楚的正确性要点:

**为什么必须带上目标层里"重叠"的文件?**
    归并是"同 key 只保留最新版本"。如果只把 L0 的文件写进 L1,
    L1 里原有的同 key 数据就会和新的并存 —— 层内一旦重叠,
    "二分找到唯一可能包含 key 的文件"这个前提就崩了,查询会静默出错。
    所以重叠的文件必须一起参与归并。

**为什么只有压到最底层才能丢墓碑?**
    墓碑的作用是压住**更旧**的数据。只要还有更深的层,那层里就可能有
    同一个键的旧值,墓碑一丢,旧值立刻复活。压到最底层意味着
    "不可能再有更旧的数据了",此时墓碑才完成了它的使命。

这个模块只负责**决定该做什么**(策略)。真正写文件、改 manifest、
删旧文件是引擎的事(机制)。分开的好处是策略能单独测,不用碰磁盘。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .manifest import FileMeta, Manifest

__all__ = [
    "CompactionTask",
    "level_budget",
    "overlapping_files",
    "pick_task",
    "plan_full_compaction",
]

#: L0 文件数达到这个值就触发 compaction
DEFAULT_L0_TRIGGER = 4

#: L1 的容量预算,之后每层乘以下面的系数
DEFAULT_LEVEL_BUDGET = 1024 * 1024

#: 层容量增长系数。越往下数据越旧越冷,单层容量可以更大。
DEFAULT_LEVEL_FACTOR = 4


def level_budget(
    level: int, base: int = DEFAULT_LEVEL_BUDGET,
    factor: int = DEFAULT_LEVEL_FACTOR,
) -> int:
    """第 ``level`` 层的容量预算(字节)。

    L0 不按字节限制(它按文件数触发),所以返回 0。
    L1 是 ``base``,L2 是 ``base * factor``,以此类推。
    """
    if level <= 0:
        return 0
    return base * (factor ** (level - 1))


def overlapping_files(
    files: list[FileMeta], smallest: bytes, largest: bytes
) -> list[FileMeta]:
    """挑出键范围与 ``[smallest, largest]`` 有交集的文件。

    层内文件按 ``smallest`` 升序,理论上可以二分收窄范围。这里直接线性过滤 ——
    compaction 是低频操作,而且每层的文件数通常不多。
    真到几千个文件的规模,这里该换成 ``bisect``。
    """
    return [meta for meta in files if meta.overlaps(smallest, largest)]


@dataclass
class CompactionTask:
    """一次 compaction 要做什么。"""

    level: int
    """源层号(L0 为 0)。"""

    target_level: int
    """输出层号。"""

    source_files: list[FileMeta] = field(default_factory=list)
    """源层里要参与归并的文件,**按新到旧排列**。"""

    target_files: list[FileMeta] = field(default_factory=list)
    """目标层里键范围重叠、必须一起归并的文件,**比 source_files 更旧**。"""

    drop_tombstones: bool = False
    """是否可以丢掉墓碑。只有输出层是最底层时才为 True。"""

    @property
    def inputs(self) -> list[FileMeta]:
        """全部输入文件,按**新到旧**排列。

        这个顺序直接决定归并时同 key 取谁 —— 排错了会读到旧值。
        """
        return self.source_files + self.target_files

    @property
    def input_bytes(self) -> int:
        return sum(meta.size for meta in self.inputs)

    @property
    def input_entries(self) -> int:
        return sum(meta.entry_count for meta in self.inputs)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"<Compaction L{self.level}→L{self.target_level} "
            f"{len(self.source_files)}+{len(self.target_files)} 个文件, "
            f"{self.input_bytes} 字节"
            f"{', 丢墓碑' if self.drop_tombstones else ''}>"
        )


def pick_task(
    manifest: Manifest,
    *,
    l0_trigger: int = DEFAULT_L0_TRIGGER,
    level_budget_base: int = DEFAULT_LEVEL_BUDGET,
    level_size_factor: int = DEFAULT_LEVEL_FACTOR,
) -> CompactionTask | None:
    """挑一个最该做的 compaction;返回 ``None`` 表示暂时不需要。

    优先级:**先管 L0**。L0 的文件键范围重叠,每一次查询都要挨个试,
    它对读性能的伤害是直接的、乘性的。
    """
    max_level = manifest.max_level

    # ---- L0 文件太多
    l0 = manifest.levels[0]
    if len(l0) >= l0_trigger:
        smallest = min(meta.smallest for meta in l0)
        largest = max(meta.largest for meta in l0)
        overlap = overlapping_files(manifest.levels[1], smallest, largest)
        return CompactionTask(
            level=0,
            target_level=1,
            source_files=list(l0),          # levels[0] 已经是"新到旧"
            target_files=overlap,
            drop_tombstones=(1 == max_level),
        )

    # ---- 某一层超了预算
    # 注意循环上界是 max_level(不含):最底层没有"下一层"可以压,
    # 所以不会触发。这同时保证了 maybe_compact 的循环一定会终止。
    for level in range(1, max_level):
        budget = level_budget(level, level_budget_base, level_size_factor)
        if manifest.level_bytes(level) <= budget:
            continue

        # 挑"与下一层重叠最少"的文件 —— 要重写的数据量最小。
        # 这是最朴素的一条启发式;真实引擎还会考虑文件年龄、删除比例等。
        next_level = manifest.levels[level + 1]
        candidate = min(
            manifest.levels[level],
            key=lambda meta: (
                len(overlapping_files(next_level, meta.smallest, meta.largest)),
                meta.file_id,
            ),
        )
        overlap = overlapping_files(next_level, candidate.smallest, candidate.largest)
        return CompactionTask(
            level=level,
            target_level=level + 1,
            source_files=[candidate],
            target_files=overlap,
            drop_tombstones=(level + 1 == max_level),
        )

    return None


def plan_full_compaction(manifest: Manifest) -> CompactionTask | None:
    """把**所有层**的数据一路压到最底层。手动整理或测试时用。

    这一次会把墓碑全部清掉 —— 压到最底层之后不可能再有更旧的数据,
    墓碑已经没有需要压住的对象了。
    """
    files = manifest.files()
    if not files:
        return None
    if len(files) == 1 and files[0].level == manifest.max_level:
        return None         # 已经是最理想的状态

    return CompactionTask(
        level=0,
        target_level=manifest.max_level,
        source_files=list(files),       # files() 已是"新到旧"
        target_files=[],
        drop_tombstones=True,
    )
