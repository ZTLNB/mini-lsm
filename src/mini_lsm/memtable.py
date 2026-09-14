"""内存表(MemTable)—— 可变的有序映射。

它是 LSM 里唯一"可写"的数据结构。所有写入都先落到这里,
积累到一定体积后再整体刷成一个不可变的 SSTable。

为什么强调"有序":
    阶段 2 刷盘时,我们希望写出的 SSTable 内部是有序的 ——
    这样查询才能用二分查找,范围扫描才能顺序读。如果内存表
    本身无序,刷盘时还得额外排序一次。
    这里用 dict 存储 + 遍历时排序,兼顾了 O(1) 查找和有序输出。

墓碑(tombstone):
    删除不能真的把键从表里抹掉 —— 因为更旧的版本可能躺在
    某个 SSTable 里。所以删除是写入一个"墓碑"标记,表示
    "这个键在此刻被删除了"。查询时遇到墓碑,就当作"不存在",
    并且可以确定不需要再往更旧的数据里找。
    这里用 ``None`` 作为墓碑的值。
"""

from __future__ import annotations

from typing import Iterator

#: 默认容量 4 MiB。超过后引擎会把它刷成 SSTable。
DEFAULT_CAPACITY_BYTES = 4 * 1024 * 1024

#: 每个键值对在 Python dict 里的额外开销估算(哈希槽、对象头等)。
#: 不精确,但足够用来判断"该刷盘了"。
ENTRY_OVERHEAD = 64


class MemTable:
    """内存中的有序键值表。

    墓碑用 ``None`` 表示。因此 ``get`` 返回 ``None`` 有两种含义:
    键不存在,或者键已被删除 —— 对调用方而言这两者是同一件事。
    需要区分时请用 ``items()`` 或 ``get_entry()``。
    """

    __slots__ = ("_data", "_size", "capacity_bytes")

    def __init__(self, capacity_bytes: int = DEFAULT_CAPACITY_BYTES) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes 必须为正数")
        self._data: dict[bytes, bytes | None] = {}
        self._size = 0
        self.capacity_bytes = capacity_bytes

    # ------------------------------------------------------------ 写入

    def put(self, key: bytes, value: bytes) -> None:
        """写入或覆盖一个键。"""
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError("key 必须是 bytes")
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError("value 必须是 bytes")

        key = bytes(key)
        value = bytes(value)
        self._adjust_size(key, self._data.get(key, _MISSING), value)
        self._data[key] = value

    def delete(self, key: bytes) -> None:
        """写入墓碑,标记该键已被删除。"""
        if not isinstance(key, (bytes, bytearray)):
            raise TypeError("key 必须是 bytes")

        key = bytes(key)
        self._adjust_size(key, self._data.get(key, _MISSING), None)
        self._data[key] = None

    def _adjust_size(self, key: bytes, old: object, new: bytes | None) -> None:
        """维护内存占用估算。覆盖写入时要把旧值的大小减掉。"""
        if old is _MISSING:
            # 新键:加上 key + value + 开销
            self._size += len(key) + ENTRY_OVERHEAD
        else:
            # 已存在的键:减掉旧 value 的大小
            self._size -= len(old) if isinstance(old, bytes) else 0

        self._size += len(new) if new is not None else 0

    # ------------------------------------------------------------ 读取

    def get(self, key: bytes) -> bytes | None:
        """查询键。

        返回 ``None`` 表示"不存在或已删除" —— 对调用方是同一件事。
        """
        return self._data.get(bytes(key))

    def get_entry(self, key: bytes) -> tuple[bool, bytes | None]:
        """返回 ``(是否存在, 值)``。

        需要区分"没写过"和"写过又删了"时用这个:
        ``(True, None)`` 表示墓碑,``(False, None)`` 表示从未出现。
        """
        key = bytes(key)
        if key not in self._data:
            return (False, None)
        return (True, self._data[key])

    def __contains__(self, key: bytes) -> bool:
        """注意:墓碑也返回 True —— 因为它确实"存在"于表中。"""
        return bytes(key) in self._data

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------ 遍历

    def items(self) -> Iterator[tuple[bytes, bytes | None]]:
        """按 key 字典序升序产出 ``(key, value)``。

        墓碑的 value 为 ``None``。刷盘时需要保留墓碑,所以这里不跳过。
        """
        for key in sorted(self._data):
            yield key, self._data[key]

    def live_items(self) -> Iterator[tuple[bytes, bytes]]:
        """只产出真实存在的键值对(跳过墓碑)。用于统计和调试。"""
        for key, value in self.items():
            if value is not None:
                yield key, value

    def keys(self) -> Iterator[bytes]:
        """按升序产出所有 key(含墓碑)。"""
        return iter(sorted(self._data))

    def range_items(self, start: bytes, end: bytes) -> Iterator[tuple[bytes, bytes | None]]:
        """产出 ``[start, end)`` 区间内的键值对,按升序。

        阶段 5 的范围扫描会用到;现在先提供出来,顺便让有序性可见。
        """
        for key, value in self.items():
            if key < start:
                continue
            if key >= end:
                return
            yield key, value

    # ------------------------------------------------------------ 状态

    @property
    def approximate_size(self) -> int:
        """估算的内存占用(字节)。"""
        return self._size

    @property
    def is_full(self) -> bool:
        """是否已达到容量上限,该刷盘了。"""
        return self._size >= self.capacity_bytes

    @property
    def tombstone_count(self) -> int:
        """墓碑数量。墓碑太多说明该 compaction 了。"""
        return sum(1 for v in self._data.values() if v is None)

    def clear(self) -> None:
        """清空。**仅在刷盘成功后调用** —— 否则会丢数据。"""
        self._data.clear()
        self._size = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"MemTable(entries={len(self._data)}, "
            f"size={self._size}/{self.capacity_bytes} bytes, "
            f"tombstones={self.tombstone_count})"
        )


class _Missing:
    """哨兵:区分"键不存在"与"值为 None(墓碑)"。"""

    def __repr__(self) -> str:  # pragma: no cover
        return "<MISSING>"


_MISSING = _Missing()
