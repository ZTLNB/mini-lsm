"""快照读 + 流式范围扫描。

阶段 4 之后还剩两个问题:

    1. **``scan()`` 要把结果全部物化成列表** —— 扫一个比内存还大的库,
       内存会跟着一起涨。而且它是在**锁内**物化的,扫多久就把写阻塞多久。
    2. **读没有一致视图** —— 遍历过程中如果发生 compaction 或刷盘,
       同一个 key 可能被看到两次、也可能被跳过,取决于读到哪一步。

这一版用**快照(snapshot)**一起解决。

---

## 快照是怎么做到"一致视图"的

先明确一件事:这里的多版本**不是** per-key 的版本链(那需要给每条记录
编号,格式要大改),而是利用一个已经存在的事实 ——

    **SSTable 不可变,而 manifest 的每一次替换就是一个新的"版本"。**

于是"某一时刻的一致状态"可以精确地表示成三个东西:

    快照 = 内存表的有序副本 + 那一刻的层结构 + 那一刻的文件集合

- **内存表副本**:``sorted(dict.items())`` 出来的新列表,元素是不可变元组。
  之后的 put/delete 改的是原 dict,碰不到这份副本。
- **层结构副本**:``FileMeta`` 是 frozen dataclass,复制一层列表就够了。
  之后 compaction 改的是 manifest 自己的列表。
- **文件集合**:这批文件的 id。**它们必须活着** —— 这就是"pin"(见下)。

有了这三样,快照看到的东西就**永远不会变**了:

    之后写入     → 落进新的内存表,副本不受影响
    之后刷盘     → 产生新文件,快照的层结构里没有它
    之后 compaction → 产生新文件 + 删旧文件,但快照 pin 住的旧文件还在

所以快照读天然满足:**创建之后的所有写入对它不可见**。

## 代价:文件要被"pin"住

compaction 会删掉旧文件。如果某个快照还在读它,就不能删 ——
在 Windows 上更直接:文件句柄还开着的时候 ``unlink`` 会失败。

所以引擎维护一个"待删清单":被快照引用的文件推迟删除,
等最后一个引用它的快照关掉之后再删。**这意味着长时间不关的快照会占住磁盘**。

## 为什么 scan 是流式的

``scan()`` 返回的是**生成器**,每读一个块才解析一个块。所以:

- 内存占用与结果集大小**无关**,只与块大小和来源个数有关
- 调用方 ``break`` 掉就能提前结束,后面的块一个都不会读

对比一下:之前是"锁内物化成 list 再返回",内存和阻塞时间都正比于结果集。

⚠️ 一个容易忽略的后果:**流式迭代期间不能一直占着锁**,否则"读不阻塞写"
就无从谈起。所以快照迭代全程**不持有引擎锁** —— 它靠 pin 住文件来保证
数据还在,而不是靠锁。
"""

from __future__ import annotations

import weakref
from bisect import bisect_left
from typing import Iterable, Iterator

from .errors import ClosedError
from .iterator import MergingIterator
from .manifest import FileMeta, find_file_in_level, files_overlapping
from .util import to_bytes

__all__ = ["Snapshot", "ScanCursor", "search_levels"]


def search_levels(
    levels: Iterable[Iterable[FileMeta]],
    readers: dict,
    key: bytes,
) -> tuple[bool, bytes | None, int]:
    """在"层结构 + 文件读取器"上按从新到旧找 key。

    返回 ``(是否找到, 值, 被布隆过滤器挡下的次数)``。

    ⚠️ 抽成函数是为了让**引擎的实时读**和**快照读**走同一份逻辑。
    两边各写一份的话,迟早会有一边忘了检查布隆过滤器、或者忘了
    "L0 要挨个试"——而这类分叉的表现是"偶发查不到数据",极难定位。

    ``levels`` 只需要是个"层的序列",所以既能传 ``manifest.levels``
    (实时),也能传快照自己捕获的那一份。
    """
    level_list = list(levels)
    rejections = 0

    # L0:键范围互相重叠,只能从新到旧挨个试
    for meta in level_list[0]:
        reader = readers[meta.file_id]
        if not reader.might_contain(key):
            rejections += 1
            continue
        found, value = reader.get(key)
        if found:
            return (True, value, rejections)

    # L1 及以上:层内不重叠,二分就能定位到唯一可能包含它的文件
    for index in range(1, len(level_list)):
        meta = find_file_in_level(level_list[index], key)
        if meta is None:
            continue
        reader = readers[meta.file_id]
        if not reader.might_contain(key):
            rejections += 1
            continue
        found, value = reader.get(key)
        if found:
            return (True, value, rejections)

    return (False, None, rejections)


class Snapshot:
    """某一时刻的一致只读视图。

    请通过 ``engine.snapshot()`` 创建,并用 ``with`` 保证释放::

        with db.snapshot() as snap:
            snap.get("k")
            for key, value in snap.scan("a", "z"):
                ...

    ⚠️ **一定要关掉**。快照会把用到的文件 pin 住,不关就等于那批磁盘空间
    一直回收不了。用 ``with`` 是最省心的方式。
    """

    __slots__ = (
        "_engine_ref", "_snapshot_id",
        "_mem_keys", "_mem_values",
        "_levels", "_readers", "_file_ids",
        "_closed", "_rejections", "_scans",
    )

    def __init__(
        self,
        engine,
        snapshot_id: int,
        mem_items: list[tuple[bytes, bytes | None]],
        levels: list[list[FileMeta]],
        readers: dict,
    ) -> None:
        # 只持弱引用:引擎 → 快照(强引用,登记在 _snapshots 里),
        # 快照 → 引擎用弱引用,这样两者之间没有循环,释放时机可预测。
        self._engine_ref = weakref.ref(engine)
        self._snapshot_id = snapshot_id

        # 内存表拆成"键列表 + 值列表":键列表专门用来 bisect 定位范围起点,
        # 不必为了找一个起点把整个列表走一遍。
        self._mem_keys = [key for key, _ in mem_items]
        self._mem_values = [value for _, value in mem_items]

        # 层结构**复制一份**。不复制的话,之后 compaction 往
        # manifest.levels 里增删,快照就会跟着变 —— 一致视图立刻失效。
        self._levels = [list(level) for level in levels]
        self._file_ids = frozenset(
            meta.file_id for level in self._levels for meta in level
        )

        # ⚠️ reader 也要**抄一份引用到自己名下**,不能直接存引擎那个 dict。
        # 引擎 compaction 时会从自己的 dict 里 ``pop`` 掉旧文件,存同一份
        # 的话快照会跟着"丢文件",读到一半突然 KeyError —— 而这恰恰是
        # 快照最该扛住的场景(边扫边 compaction)。
        self._readers = {fid: readers[fid] for fid in self._file_ids}

        self._closed = False
        self._rejections = 0
        self._scans = 0

    # ------------------------------------------------------------ 生命周期

    def close(self) -> None:
        """释放快照。重复调用安全。

        真正的清理交给引擎:把登记的 pin 去掉,顺手回收那些"已经没有
        任何快照引用"的待删文件。
        """
        if self._closed:
            return
        self._closed = True
        self._mem_keys = []
        self._mem_values = []
        self._levels = []
        self._file_ids = frozenset()
        engine = self._engine_ref()
        if engine is not None:
            engine._release_snapshot(self)

    def _invalidate(self) -> None:
        """引擎关闭时调用:只标记失效,**不回调引擎**(避免重入)。"""
        self._closed = True
        self._mem_keys = []
        self._mem_values = []
        self._levels = []
        self._file_ids = frozenset()

    def _ensure_usable(self) -> None:
        if self._closed:
            raise ClosedError("这个快照已经关闭")
        engine = self._engine_ref()
        if engine is None or engine.closed:
            self._closed = True
            raise ClosedError("引擎已关闭,快照随之失效")

    def __enter__(self) -> "Snapshot":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ 读取

    def get_entry(self, key: object) -> tuple[bool, bytes | None]:
        """返回 ``(是否存在, 值)``。

        ``(True, None)`` 表示墓碑,``(False, None)`` 表示从未出现过 ——
        和 ``MemTable.get_entry`` 的口径一致。
        """
        self._ensure_usable()
        kb = to_bytes(key, "key")

        # 内存表最新,先查它。键列表有序,所以是二分而不是遍历。
        index = bisect_left(self._mem_keys, kb)
        if index < len(self._mem_keys) and self._mem_keys[index] == kb:
            return (True, self._mem_values[index])

        found, value, rejections = search_levels(
            self._levels, self._readers, kb
        )
        self._rejections += rejections
        return (found, value)

    def get(self, key: object) -> bytes | None:
        """查一个键。``None`` 表示"不存在或已删除"。

        两者对调用方是同一件事;要区分就用 :meth:`get_entry`。
        """
        found, value = self.get_entry(key)
        return value if found else None

    def get_str(self, key: object, encoding: str = "utf-8") -> str | None:
        raw = self.get(key)
        return None if raw is None else raw.decode(encoding, errors="replace")

    def contains(self, key: object) -> bool:
        return self.get(key) is not None

    def scan(
        self,
        start: object | None = None,
        end: object | None = None,
    ) -> Iterator[tuple[bytes, bytes]]:
        """流式产出 ``[start, end)`` 区间内**存活**的键值对,按 key 升序。

        返回的是生成器 —— 读一块才解析一块,``break`` 掉就不读了。

        墓碑在这里被**过滤掉**(它们的作用是压住旧版本,不该露给用户);
        但它们在归并过程中是必须看到的,否则被删的键会从更旧的文件里复活。
        """
        self._ensure_usable()
        self._scans += 1

        lo = to_bytes(start, "start") if start is not None else None
        hi = to_bytes(end, "end") if end is not None else None

        for key, value in MergingIterator(self._sources(lo, hi)):
            if value is None:
                continue            # 墓碑:压住旧版本,但不产出
            yield key, value

    def keys(self, start: object | None = None,
             end: object | None = None) -> Iterator[bytes]:
        """只产出 key。"""
        return (key for key, _ in self.scan(start, end))

    def items(self, start: object | None = None,
              end: object | None = None) -> Iterator[tuple[bytes, bytes]]:
        """``scan`` 的别名,和 dict 的用法对齐。"""
        return self.scan(start, end)

    def _sources(
        self, lo: bytes | None, hi: bytes | None
    ) -> list[Iterator[tuple[bytes, bytes | None]]]:
        """组装归并来源,**新的在前**。

        顺序就是新旧顺序,``MergingIterator`` 靠它决定同一个 key 取谁 ——
        排错了会读到旧值,所以这里绝不能"随手排一下"。

        文件按范围剪枝:与 ``[lo, hi)`` 没有交集的文件根本不打开。
        窄区间扫描时这一条能省掉绝大部分文件。
        """
        sources: list[Iterator[tuple[bytes, bytes | None]]] = [
            self._memtable_source(lo, hi)
        ]
        for level in self._levels:
            for meta in level:
                if lo is not None or hi is not None:
                    if not _overlaps(meta, lo, hi):
                        continue
                sources.append(
                    self._readers[meta.file_id].iter_entries(
                        start=lo, end=hi, fill_cache=False
                    )
                )
        return sources

    def _memtable_source(
        self, lo: bytes | None, hi: bytes | None
    ) -> Iterator[tuple[bytes, bytes | None]]:
        """内存表副本在 ``[lo, hi)`` 内的那一段。

        用 bisect 直接在键列表上切出区间,不用从第一个键开始走。
        这正是 ``_mem_keys`` / ``_mem_values`` 拆成两个列表的原因 ——
        ``bisect`` 需要一个**独立的、有序的键序列**,拿 ``[(k, v), ...]``
        去二分得自己写比较函数,而元组比较会在 key 相同时去比 value,
        碰到 ``bytes`` 和 ``None`` 混在一起就抛 ``TypeError``。

        ⚠️ 边界是这个方法最容易写错的地方(start 比第一个键还小、
        end 落在两个键中间、start == end、反向区间)。
        ``MemTable.range_items()`` 是同一件事的**朴素 O(n) 版本**,
        正确性一眼可见 —— 它被留作预言机,由
        ``tests/test_snapshot.py::TestMemtableSliceOracle`` 拿随机区间
        和这里做比对。改这段之前先跑那个测试。
        """
        keys = self._mem_keys
        begin = bisect_left(keys, lo) if lo is not None else 0
        stop = bisect_left(keys, hi) if hi is not None else len(keys)
        values = self._mem_values
        for index in range(begin, stop):
            yield keys[index], values[index]

    # ------------------------------------------------------------ 元信息

    @property
    def snapshot_id(self) -> int:
        return self._snapshot_id

    @property
    def file_ids(self) -> frozenset[int]:
        """这个快照 pin 住的文件。引擎靠它决定哪些文件能删。"""
        return self._file_ids

    @property
    def memtable_entries(self) -> int:
        """快照里内存表副本的条目数(含墓碑)。"""
        return len(self._mem_keys)

    @property
    def sstable_count(self) -> int:
        return sum(len(level) for level in self._levels)

    @property
    def bloom_rejections(self) -> int:
        """这个快照上的查询被布隆过滤器挡下了多少次文件读。"""
        return self._rejections

    @property
    def scan_count(self) -> int:
        """这个快照上开过多少次 scan。"""
        return self._scans

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        if self._closed:
            return f"<Snapshot #{self._snapshot_id} closed>"
        return (
            f"<Snapshot #{self._snapshot_id} "
            f"memtable={len(self._mem_keys)} files={self.sstable_count}>"
        )


def _overlaps(meta: FileMeta, lo: bytes | None, hi: bytes | None) -> bool:
    """文件的键范围与 ``[lo, hi)`` 有没有交集(两端可为 None)。"""
    if lo is not None and meta.largest < lo:
        return False
    if hi is not None and meta.smallest >= hi:
        return False
    return True


class ScanCursor(Iterator[tuple[bytes, bytes]]):
    """``engine.scan()`` 的返回值:一个**持有快照**的迭代器。

    为什么需要它:

        ``engine.scan()`` 必须一调用就定格(否则
        ``it = db.scan(); db.put(...)`` 之后再迭代就会看到新写入,
        这不是调用方期望的语义)。所以快照在 ``scan()`` 里**立刻**创建。

        但快照需要释放。把它交给迭代器:迭代到底自动释放,
        中途 ``break`` 掉则在对象被回收时释放。

    需要更精确的生命周期控制时,直接用 ``with db.snapshot()``。
    """

    __slots__ = ("_snapshot", "_iterator", "_closed")

    def __init__(self, snapshot: Snapshot, iterator: Iterator) -> None:
        self._snapshot = snapshot
        self._iterator = iterator
        self._closed = False

    def __iter__(self) -> "ScanCursor":
        return self

    def __next__(self) -> tuple[bytes, bytes]:
        if self._closed:
            raise StopIteration
        try:
            return next(self._iterator)
        except StopIteration:
            self.close()
            raise

    def close(self) -> None:
        """提前结束迭代并释放快照。重复调用安全。"""
        if self._closed:
            return
        self._closed = True
        # 生成器要先关掉,否则它引用的那些块迭代器不会立刻释放
        close = getattr(self._iterator, "close", None)
        if close is not None:
            close()
        self._snapshot.close()

    @property
    def snapshot(self) -> Snapshot:
        return self._snapshot

    def __del__(self) -> None:  # pragma: no cover - 依赖 GC 时机
        # 调用方可能中途 break 掉,不走到 StopIteration。这时靠对象回收兜底,
        # 否则快照会把文件一直 pin 住(磁盘回收不了)。
        try:
            self.close()
        except Exception:
            # 解释器退出阶段什么都可能失败,这里绝不能把异常抛出去
            pass

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        state = "closed" if self._closed else "open"
        return f"<ScanCursor {state} snapshot=#{self._snapshot.snapshot_id}>"
