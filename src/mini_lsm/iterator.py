"""多来源归并迭代器。

一个 key 可能同时存在于内存表和好几个 SSTable 里。归并迭代器让
"多个来源"对上层看起来像"一个有序表" —— 扫描和 compaction 都靠它,
不用各自写一遍合并逻辑。

约定(**这三条是整层的契约**):
    1. 每个来源产出 ``(key, value)``,按 key **升序**
    2. ``value is None`` 表示墓碑
    3. **来源的先后顺序就是新旧顺序**:下标 0 最新。
       同一个 key 出现在多个来源时,只产出最新的那个,其余丢弃。

墓碑也会被产出 —— 过滤是调用方的事。这是刻意的:
面向用户的扫描要跳过墓碑,但 compaction 必须看到它们,
否则"删除"这个信息就丢了,旧值会从更底层的文件里复活。
"""

from __future__ import annotations

import heapq
from typing import Iterable, Iterator

__all__ = ["MergingIterator"]


class MergingIterator(Iterator[tuple[bytes, bytes | None]]):
    """把多个各自有序的来源合并成一条有序流。

    用法::

        it = MergingIterator([memtable.items(), sst_newest.iter_entries(),
                              sst_older.iter_entries()])
        for key, value in it:
            ...        # key 升序;同 key 只出现一次(最新版本)

    实现是标准 k 路归并:每个来源先取出队首压进小顶堆,弹出后立刻补下一条。
    堆元素是 ``(key, source_index, seq, value)`` ——
    ``source_index`` 保证同 key 时**新的来源先弹出**,
    ``seq`` 是个全局自增序号,纯粹为了杜绝"key 和 source_index 都相等时
    去比较 value"的情况(``bytes`` 和 ``None`` 没法比大小,会抛 TypeError)。
    """

    __slots__ = ("_sources", "_iters", "_heap", "_seq", "_last_key", "_exhausted")

    def __init__(
        self,
        sources: Iterable[Iterable[tuple[bytes, bytes | None]]],
    ) -> None:
        self._sources = list(sources)
        self._iters = [iter(source) for source in self._sources]
        self._heap: list[tuple[bytes, int, int, bytes | None]] = []
        self._seq = 0
        #: 上一个产出的 key,用来丢弃同 key 的旧版本
        self._last_key: bytes | None = None
        self._exhausted = False

        for index, iterator in enumerate(self._iters):
            self._advance(index, iterator)

    def _advance(self, index: int, iterator: Iterator) -> None:
        """从第 index 个来源取下一条,压入堆。取完了就什么都不做。"""
        try:
            key, value = next(iterator)
        except StopIteration:
            return
        self._seq += 1
        heapq.heappush(self._heap, (key, index, self._seq, value))

    def __iter__(self) -> "MergingIterator":
        return self

    def __next__(self) -> tuple[bytes, bytes | None]:
        while self._heap:
            key, index, _seq, value = heapq.heappop(self._heap)
            # 立刻补上该来源的下一条,保持堆里每个来源最多一个待选元素
            self._advance(index, self._iters[index])

            if self._last_key is not None and key == self._last_key:
                continue        # 同 key 的旧版本,已经被更新的来源产出过了
            self._last_key = key
            return key, value

        raise StopIteration

    # ------------------------------------------------------------ 便捷方法

    def live(self) -> Iterator[tuple[bytes, bytes]]:
        """只产出真实存在的键值对(跳过墓碑)。

        注意它返回的是**新的迭代器**,原迭代器仍可用。
        """
        for key, value in self:
            if value is not None:
                yield key, value

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"MergingIterator(sources={len(self._sources)})"
