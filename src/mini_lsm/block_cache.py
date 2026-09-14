"""块缓存(Block Cache)—— 避免同一块被反复读盘和反复解析。

要解决的问题:
    每次 ``get()`` 都要把一个数据块从文件里读出来、校验 CRC、再把里面的
    记录逐条解析出来。查询热点数据时,同一个块会被反复读、反复解析 ——
    而 SSTable 是**不可变**的,解析结果永远不会变,完全可以缓存。

    纯 Python 里"解析"这一步尤其贵(逐条解 struct),所以缓存的是
    **解析后的记录列表**而不是原始字节 —— 命中时能省掉整条解析路径。

为什么是 LRU:
    缓存容量有限,必须决定淘汰谁。LRU(淘汰最久未使用的)对
    "热点集中"的访问模式效果最好,而且用 ``OrderedDict`` 实现
    就是 O(1),不需要自己写双向链表。

⚠️ 一个容易忽略的点:**顺序全扫不应该填缓存**。
    compaction 会把整个文件从头读到尾,每个块只读一次、之后再也不碰。
    如果这些块都塞进缓存,它们会把真正的热点数据全部挤出去 ——
    这就是所谓的"缓存污染"。所以读取接口带一个 ``use_cache`` 开关,
    全量扫描时关掉它。见 ``sstable.SSTableReader.iter_entries``。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable

__all__ = ["DEFAULT_CACHE_BYTES", "CACHE_ENTRY_OVERHEAD", "CacheStats", "BlockCache"]

#: 默认缓存 8 MiB。够放 2000 个 4 KiB 的块。
DEFAULT_CACHE_BYTES = 8 * 1024 * 1024

#: 每条记录在缓存里的额外开销估算(Python 对象头 + 元组 + 列表槽位)。
#: 不精确 —— 只用来判断"该淘汰了",和 MemTable 的容量口径保持一致。
CACHE_ENTRY_OVERHEAD = 64


@dataclass
class CacheStats:
    """缓存的工作情况快照。"""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    oversized: int = 0
    """因为单块比整个缓存还大、索性没缓存的次数。"""
    entries: int = 0
    bytes: int = 0
    capacity_bytes: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.lookups
        return self.hits / total if total else 0.0

    @property
    def usage_ratio(self) -> float:
        if self.capacity_bytes <= 0:
            return 0.0
        return self.bytes / self.capacity_bytes

    def __str__(self) -> str:
        if self.capacity_bytes <= 0:
            return "块缓存: 已禁用"
        return (
            f"块缓存: 命中 {self.hits} / {self.lookups}"
            f" ({self.hit_rate:.1%}),淘汰 {self.evictions} 次,"
            f" {self.entries} 块 / {self.bytes} 字节"
            f" (容量 {self.capacity_bytes},占用 {self.usage_ratio:.1%})"
        )


def estimate_entries_size(entries: Iterable[tuple]) -> int:
    """估算一批记录占多少内存。

    只统计 key 和 value 的字节数加上一个固定的每条约开销。
    Python 对象的真实开销难以精确计算,但用于淘汰决策足够了。
    """
    total = 0
    for item in entries:
        _rec_type, key, value = item
        total += len(key) + len(value) + CACHE_ENTRY_OVERHEAD
    return total


class BlockCache:
    """按 ``(file_id, block_index)`` 索引的 LRU 块缓存。

    键里带 ``file_id`` 是必须的:compaction 之后新文件会拿到**新的**
    file_id(编号单调递增,绝不复用),所以不同文件里的"第 3 块"
    不会互相串味。
    """

    __slots__ = ("_capacity", "_data", "_bytes", "_hits", "_misses",
                 "_evictions", "_oversized", "_lock")

    def __init__(self, capacity_bytes: int = DEFAULT_CACHE_BYTES) -> None:
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes 不能为负数")
        self._capacity = capacity_bytes
        # OrderedDict 的迭代顺序就是"最近使用顺序":队尾最新,队首最旧。
        # 值存 (大小, 记录列表) —— 把大小一起记下来,
        # 淘汰时就不用把记录列表重新算一遍,整个过程是 O(1)。
        self._data: OrderedDict[tuple[int, int], tuple[int, list[tuple]]] = (
            OrderedDict()
        )
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._oversized = 0
        # 引擎自己有一把大锁,但缓存加锁让它能独立、安全地被复用。
        # 这里不会回调引擎,所以不存在锁顺序反转的风险。
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 查询

    def get(self, file_id: int, block_index: int) -> list[tuple] | None:
        """取一块。命中时把它挪到"最近使用"的一端。"""
        key = (file_id, block_index)
        with self._lock:
            if self._capacity <= 0:
                self._misses += 1
                return None
            found = self._data.get(key)
            if found is None:
                self._misses += 1
                return None
            self._data.move_to_end(key)
            self._hits += 1
            return found[1]

    def put(self, file_id: int, block_index: int, entries: list[tuple]) -> None:
        """放一块进去,必要时淘汰最久未使用的那些。"""
        if self._capacity <= 0 or not entries:
            return

        size = estimate_entries_size(entries)
        if size > self._capacity:
            # 单块比整个缓存还大。硬塞进去会把缓存清空一次,
            # 那还不如不缓存 —— 记一笔,让调用方能从统计里看出来。
            self._oversized += 1
            return

        key = (file_id, block_index)
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._bytes -= old[0]

            self._data[key] = (size, entries)
            self._bytes += size

            while self._bytes > self._capacity and self._data:
                _, (evicted_size, _evicted) = self._data.popitem(last=False)
                self._bytes -= evicted_size
                self._evictions += 1

    def evict_file(self, file_id: int) -> int:
        """扔掉某个文件的全部缓存块,返回清掉的块数。

        compaction 删掉文件之后要调用它 —— 那些块再也不会被读到,
        留着只会白占内存。(file_id 不复用,所以不清理也不会读到脏数据,
        但内存是实打实浪费的。)
        """
        with self._lock:
            victims = [key for key in self._data if key[0] == file_id]
            for key in victims:
                self._bytes -= self._data.pop(key)[0]
            return len(victims)

    def clear(self) -> None:
        """清空内容(统计计数保留)。"""
        with self._lock:
            self._data.clear()
            self._bytes = 0

    # ------------------------------------------------------------ 状态

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def enabled(self) -> bool:
        return self._capacity > 0

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._data

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                oversized=self._oversized,
                entries=len(self._data),
                bytes=self._bytes,
                capacity_bytes=self._capacity,
            )

    def reset_stats(self) -> None:
        """只清统计,不动缓存内容。用来做"这一段操作的命中率是多少"。"""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._oversized = 0

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        state = f"{len(self._data)} 块 / {self._bytes} 字节" if self.enabled else "已禁用"
        return f"<BlockCache {state} 容量={self._capacity}>"
