"""版本清单(Manifest)—— 记录"当前有效的是哪一套 SSTable"。

为什么阶段 2 不需要它,阶段 3 就必须有:
    阶段 2 的文件只增不减,而且文件名和内容一一对应。所以"扫目录"就能
    恢复,简单到几乎没有出错空间。
    但 compaction 会**成批地增删文件**。崩溃完全可能发生在
    "新文件已经写好、旧文件还没删掉"的中间态 —— 这时目录里新旧两套文件
    同时存在,光看目录无法判断该用哪一套。选错就是丢数据。

Manifest 的解法是**整体原子替换**:
    把"当前有效的全部文件"写进一个临时文件,fsync,然后 ``os.replace``。
    崩溃后要么看到旧的一套,要么看到新的一套,不存在中间态。

和真实引擎的差别:
    RocksDB 的 MANIFEST 是**追加式的版本编辑日志**(每次 compaction 只追加
    一条增量记录)。这里每次重写整个文件 —— 因为状态只有几 KB,
    重写的代价可以忽略,而"整体替换"的正确性显然更容易论证。
    等到文件数上万、manifest 变成几 MB 时,才值得换成追加式。
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

from .errors import CorruptionError

__all__ = [
    "MANIFEST_FILENAME",
    "FileMeta",
    "Manifest",
    "find_file_in_level",
    "files_overlapping",
]

#: manifest 文件名(故意不带 .sst 后缀,免得被当成数据文件扫到)
MANIFEST_FILENAME = "MANIFEST.json"

#: 格式版本。将来改格式时靠它判断能不能读。
MANIFEST_VERSION = 1


def _encode_key(key: bytes) -> str:
    """key 是任意字节,JSON 存不下 —— 用 base64 包一层。"""
    return base64.b64encode(key).decode("ascii")


def _decode_key(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


@dataclass(frozen=True)
class FileMeta:
    """一个 SSTable 文件的元信息。

    除了"它在哪一层",还记着**键范围** —— 这是 compaction 规划的依据:
    要合并 L1 的某个文件时,必须把 L2 里键范围与之重叠的文件一起带上,
    否则合并结果里会出现重复的 key。
    """

    file_id: int
    level: int
    smallest: bytes
    largest: bytes
    entry_count: int
    size: int

    @property
    def key_range(self) -> tuple[bytes, bytes]:
        return (self.smallest, self.largest)

    def contains(self, key: bytes) -> bool:
        return self.smallest <= key <= self.largest

    def overlaps(self, smallest: bytes, largest: bytes) -> bool:
        """键范围是否与 ``[smallest, largest]`` 有交集(闭区间)。"""
        return self.smallest <= largest and self.largest >= smallest

    def to_dict(self) -> dict:
        return {
            "file_id": self.file_id,
            "level": self.level,
            "smallest": _encode_key(self.smallest),
            "largest": _encode_key(self.largest),
            "entry_count": self.entry_count,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileMeta":
        return cls(
            file_id=int(data["file_id"]),
            level=int(data["level"]),
            smallest=_decode_key(data["smallest"]),
            largest=_decode_key(data["largest"]),
            entry_count=int(data["entry_count"]),
            size=int(data["size"]),
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"FileMeta(id={self.file_id}, L{self.level}, "
            f"{self.smallest!r}..{self.largest!r}, {self.entry_count} 条)"
        )


def find_file_in_level(files: list[FileMeta], key: bytes) -> FileMeta | None:
    """在**非重叠**层里二分找出唯一可能包含 ``key`` 的文件。

    ⚠️ 为什么把它抽成模块级函数,而不是只留在 ``Manifest`` 上:

        阶段 5 的快照要在**自己捕获的那一份层结构**上做同样的查找 ——
        因为查找期间真实的 manifest 可能已经被 compaction 改掉了。
        如果快照另写一份二分,两份实现迟早会分叉,而分叉的表现是
        "偶发查不到数据",极难定位。共用一份就不存在这个问题。

    前提(调用方必须保证):``files`` 按 ``smallest`` **升序**且互不重叠。
    L0 不满足这个前提,不能用它。
    """
    if not files or key < files[0].smallest:
        return None

    lo, hi = 0, len(files) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if files[mid].smallest <= key:
            lo = mid
        else:
            hi = mid - 1

    candidate = files[lo]
    return candidate if key <= candidate.largest else None


def files_overlapping(
    files: list[FileMeta], start: bytes, end: bytes
) -> list[FileMeta]:
    """挑出与 ``[start, end)`` 有交集的文件。

    范围扫描用它剪枝:一个文件的最大键还不到 ``start``、或者最小键已经
    到了 ``end``,它就整个不在区间里 —— 连打开都不必。

    注意区间是**左闭右开**的,所以判断用的是 ``largest < start`` 和
    ``smallest >= end``:恰好等于 ``end`` 的文件要排除掉。
    """
    return [
        meta for meta in files
        if meta.largest >= start and meta.smallest < end
    ]


class Manifest:
    """当前有效的 SSTable 集合。

    层序约定(**这是整层的契约**):

    - **L0**:由内存表刷盘产生。键范围**允许重叠**,靠 ``file_id`` 从大到小
      表示从新到旧 —— 查一个 key 要挨个试。
    - **L1 及以上**:键范围**互不重叠**,按 ``smallest`` 升序排列。
      于是查一个 key 只需要二分找到唯一可能包含它的那个文件。
      这是 compaction 换来的最大收益。
    """

    def __init__(self, path: str | Path, num_levels: int = 3) -> None:
        if num_levels < 2:
            raise ValueError("num_levels 至少为 2(否则没有地方可以往下压)")
        self.path = Path(path)
        self.num_levels = num_levels
        self.levels: list[list[FileMeta]] = [[] for _ in range(num_levels)]
        self.next_file_id = 1

    # ------------------------------------------------------------ 读写

    @property
    def max_level(self) -> int:
        """最底层的层号。只有压到这一层才能安全地丢掉墓碑。"""
        return self.num_levels - 1

    def files(self) -> list[FileMeta]:
        """所有文件,按"从新到旧"排列(L0 新的在前,然后 L1、L2……)。"""
        return [meta for level in self.levels for meta in level]

    def file_ids(self) -> set[int]:
        return {meta.file_id for meta in self.files()}

    def level_bytes(self, level: int) -> int:
        return sum(meta.size for meta in self.levels[level])

    def level_entries(self, level: int) -> int:
        return sum(meta.entry_count for meta in self.levels[level])

    def find_file(self, level: int, key: bytes) -> FileMeta | None:
        """在非重叠层里找出唯一可能包含 key 的文件。

        L0 用不了这个 —— 它内部允许重叠,只能挨个查。
        """
        return find_file_in_level(self.levels[level], key)

    # ------------------------------------------------------------ 增删

    def add(self, meta: FileMeta) -> None:
        self.levels[meta.level].append(meta)
        self._sort_level(meta.level)

    def replace(self, removed: list[FileMeta], added: list[FileMeta]) -> None:
        """一次 compaction 的结果:去掉 ``removed``,加入 ``added``。

        先全部删除再加入 —— 这样"某个文件既是输入又是输出"也不会出错。
        """
        removed_ids = {meta.file_id for meta in removed}
        for level in self.levels:
            level[:] = [meta for meta in level if meta.file_id not in removed_ids]
        for meta in added:
            self.levels[meta.level].append(meta)
        for index in range(self.num_levels):
            self._sort_level(index)

    def _sort_level(self, level: int) -> None:
        if level == 0:
            # L0 允许重叠,顺序表达的是"新旧",不是键序
            self.levels[level].sort(key=lambda meta: meta.file_id, reverse=True)
        else:
            self.levels[level].sort(key=lambda meta: meta.smallest)

    # ------------------------------------------------------------ 持久化

    def to_dict(self) -> dict:
        return {
            "version": MANIFEST_VERSION,
            "num_levels": self.num_levels,
            "next_file_id": self.next_file_id,
            "levels": [[meta.to_dict() for meta in level] for level in self.levels],
        }

    def save(self) -> None:
        """原子地落盘。

        先写 ``MANIFEST.json.tmp`` 并 fsync,再 ``os.replace``。
        调用方必须保证:**被引用的数据文件已经先落盘了** ——
        manifest 一旦指向它们,它们就必须真的在。
        """
        payload = json.dumps(self.to_dict(), ensure_ascii=False,
                             sort_keys=True).encode("utf-8")
        tmp = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with open(tmp, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    def load(self) -> None:
        """从磁盘读取,替换当前内容。"""
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError as exc:
            raise CorruptionError(f"manifest 不存在:{self.path}") from exc

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CorruptionError(f"manifest 不是合法 JSON:{self.path}") from exc

        version = data.get("version")
        if version != MANIFEST_VERSION:
            raise CorruptionError(
                f"manifest 版本不支持:期望 {MANIFEST_VERSION},实际 {version!r}"
            )

        num_levels = int(data.get("num_levels", self.num_levels))
        if num_levels < 2:
            raise CorruptionError(f"manifest 里的 num_levels 非法:{num_levels}")

        self.num_levels = num_levels
        self.next_file_id = int(data.get("next_file_id", 1))
        self.levels = [[] for _ in range(num_levels)]

        for index, level in enumerate(data.get("levels", [])):
            if index >= num_levels:
                raise CorruptionError(
                    f"manifest 里有超出层数的数据:第 {index} 层(共 {num_levels} 层)"
                )
            for item in level:
                self.levels[index].append(FileMeta.from_dict(item))

        for index in range(num_levels):
            self._sort_level(index)

    # ------------------------------------------------------------ 校验

    def check_invariants(self) -> None:
        """检查层序契约有没有被破坏。测试和诊断用。

        这些不变量一旦破了,查询会**静默**返回错误结果 ——
        所以宁可在这里大声报错。
        """
        seen: dict[int, int] = {}
        for index, level in enumerate(self.levels):
            for meta in level:
                if meta.file_id in seen:
                    raise CorruptionError(
                        f"文件 {meta.file_id} 同时出现在 L{seen[meta.file_id]} 和 "
                        f"L{index}"
                    )
                seen[meta.file_id] = index

                if meta.smallest > meta.largest:
                    raise CorruptionError(
                        f"文件 {meta.file_id} 的键范围反了:"
                        f"{meta.smallest!r} > {meta.largest!r}"
                    )
                if meta.entry_count <= 0:
                    raise CorruptionError(f"文件 {meta.file_id} 的条目数为 0")
                if meta.level != index:
                    raise CorruptionError(
                        f"文件 {meta.file_id} 记的层号是 L{meta.level},"
                        f"实际挂在 L{index}"
                    )

        # L0 按 file_id 从大到小(新到旧)
        l0 = self.levels[0]
        if l0 != sorted(l0, key=lambda meta: meta.file_id, reverse=True):
            raise CorruptionError("L0 没有按 file_id 从大到小排列")

        # L1 及以上必须互不重叠,且按 smallest 升序
        for index in range(1, self.num_levels):
            files = self.levels[index]
            if files != sorted(files, key=lambda meta: meta.smallest):
                raise CorruptionError(f"L{index} 没有按 smallest 升序排列")
            for left, right in zip(files, files[1:]):
                if left.largest >= right.smallest:
                    raise CorruptionError(
                        f"L{index} 里文件 {left.file_id} 和 {right.file_id} "
                        f"的键范围重叠:{left.largest!r} >= {right.smallest!r}"
                    )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        counts = ", ".join(f"L{i}={len(level)}" for i, level in enumerate(self.levels))
        return f"<Manifest {counts} next_id={self.next_file_id}>"
