"""SSTable(Sorted String Table)—— 不可变的有序磁盘表。

这是 LSM 里**唯一存到磁盘上的数据格式**。内存表攒满之后整体刷成一个
SSTable,之后这个文件只读、永不修改。所有复杂度都来自这一点:
既然文件不可变,就不存在"就地更新"和"写一半被别人读到"的问题。

文件布局(格式版本 2)::

    +---------------------------+
    | 数据块 0                  |  多条 entry,按 key 升序
    +---------------------------+  block := [len:4][crc32:4] + entries
    | 数据块 1                  |
    +---------------------------+
    | ...                       |
    +---------------------------+
    | 索引块                    |  稀疏索引:每个数据块的**起始 key** → (偏移, 长度)
    +---------------------------+  同样是 [len:4][crc32:4] + entries
    | 过滤器块                  |  布隆过滤器,回答"这个 key 一定不在吗"
    +---------------------------+  同样是 [len:4][crc32:4] + payload
    | footer(固定 48 字节)      |  索引偏移、索引长度、条目数、
    +---------------------------+  过滤器偏移、过滤器长度、magic

entry 的字节布局和 WAL 的 payload **完全一致**:
``rec_type(1) + key_len(4) + key + value_len(4) + value``。
两者共用 ``encode_payload`` / ``decode_payload``,少一套编解码就少一处会写错的地方。

磁盘格式的演进(v1 → v2):
    阶段 2/3 的文件是 **v1** —— 32 字节 footer,没有过滤器块。
    阶段 4 加了布隆过滤器,footer 变成 48 字节,magic 也从 ``MINILSM1``
    换成 ``MINILSM2``。

    换 magic 是**为了能同时读两种格式**:读文件时先看末尾 8 字节,
    是 ``MINILSM2`` 就按 48 字节解析,是 ``MINILSM1`` 就按 32 字节解析。
    于是 v1 的老文件不需要任何转换就能直接打开 —— 这是磁盘格式演进
    必须守住的一条:加字段可以,但不能让老数据读不出来。

    对老文件来说 ``might_contain()`` 只能返回 True(没有过滤器可用),
    也就是退化成阶段 3 的行为 —— 慢一点,但结果依然正确。

三个关键设计:

**为什么是"稀疏"索引?**
    每 4 KiB 一个索引项,而不是每个 key 一个。索引常驻内存,必须小 ——
    1 亿条记录如果逐条建索引,光索引就要好几 GB。稀疏索引让索引大小
    正比于"块的个数"而不是"记录的个数",代价是查一个 key 需要
    在块内顺序扫一遍(块只有 4 KiB,这点开销可以忽略)。

**为什么 CRC 按块算而不是按条算?**
    SSTable 的读取粒度就是块 —— 要么整块读进来用,要么根本不读。
    按块校验和按条校验能发现的问题一样多,但校验开销只有 1/N。

**为什么要有布隆过滤器?**
    稀疏索引只能回答"key 落在哪个块",前提是**已经决定要读这个文件**。
    而查一个不存在的 key 时,最该省掉的恰恰是"读文件"这个动作本身。
    布隆过滤器用每个 key 约 1 字节的代价,换掉绝大部分无谓的读。

为什么用"每块的起始 key"而不是"结束 key"?
    两种都能用,起始 key 更直观。代价是"整个表的最大 key"没法从索引直接
    拿到(需要读最后一个块),所以它是**惰性**计算的并缓存起来。
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator

from .block_cache import BlockCache
from .bloom import DEFAULT_BITS_PER_KEY, BloomBuilder, BloomFilter
from .errors import CorruptionError, InvalidArgumentError
from .record import (
    RecordType,
    encode_payload,
    frame,
    iter_payload,
    unframe,
)

__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "FOOTER_SIZE",
    "FOOTER_SIZE_V1",
    "FOOTER_MAGIC",
    "FOOTER_MAGIC_V1",
    "FOOTER_MAGIC_V2",
    "FORMAT_VERSION",
    "SUFFIX",
    "TMP_SUFFIX",
    "SSTableMeta",
    "SSTableReader",
    "SSTableWriter",
    "sstable_filename",
    "parse_file_id",
    "read_all",
]

#: 数据块的目标大小。超过它就切下一块。
DEFAULT_BLOCK_SIZE = 4096

#: 当前写出的格式版本。
FORMAT_VERSION = 2

#: v1 的 magic(阶段 2/3 写的文件)。保留它是为了还能读老数据。
FOOTER_MAGIC_V1 = b"MINILSM1"

#: v2 的 magic。新文件一律用这个。
FOOTER_MAGIC_V2 = b"MINILSM2"

#: 新文件用的 magic。
FOOTER_MAGIC = FOOTER_MAGIC_V2

#: v1 footer 固定 32 字节:
#: 索引偏移(8) + 索引长度(8) + 条目数(8) + magic(8)
_FOOTER_V1 = struct.Struct(">QQQ8s")
FOOTER_SIZE_V1 = _FOOTER_V1.size

#: v2 footer 固定 48 字节:
#: 索引偏移(8) + 索引长度(8) + 条目数(8) + 过滤器偏移(8) + 过滤器长度(8) + magic(8)
_FOOTER_V2 = struct.Struct(">QQQQQ8s")
FOOTER_SIZE = _FOOTER_V2.size

#: magic 在 footer 的最末尾,固定 8 字节 —— 先读它判版本,再按版本解析整个 footer
_MAGIC_SIZE = 8

#: 索引项:起始 key 长度(4) + 块偏移(8) + 块长度(8) + 起始 key
_INDEX_ENTRY = struct.Struct(">IQQ")

_FILENAME_RE = re.compile(r"^sst-(\d+)\.sst$")

#: 文件名模板与后缀
SUFFIX = ".sst"
TMP_SUFFIX = ".sst.tmp"


def sstable_filename(file_id: int) -> str:
    """按编号生成 SSTable 文件名。编号补零,这样按文件名排序就是按新旧排序。"""
    return f"sst-{file_id:06d}{SUFFIX}"


def parse_file_id(path: str | Path) -> int | None:
    """从文件名里解析出编号。名字不符合规范时返回 ``None``。"""
    match = _FILENAME_RE.match(Path(path).name)
    return int(match.group(1)) if match else None


@dataclass
class SSTableMeta:
    """一个刚写完的 SSTable 的元信息。"""

    file_id: int
    path: Path
    entry_count: int
    size: int
    first_key: bytes | None
    last_key: bytes | None

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"SSTableMeta(id={self.file_id}, entries={self.entry_count}, "
            f"size={self.size})"
        )


@dataclass
class _IndexEntry:
    """稀疏索引里的一项:某个数据块的起始 key 和它在文件里的位置。"""

    first_key: bytes
    offset: int
    length: int

    def encode(self) -> bytes:
        return (
            _INDEX_ENTRY.pack(len(self.first_key), self.offset, self.length)
            + self.first_key
        )


def _decode_index(payload: bytes, offset: int) -> list[_IndexEntry]:
    """解析索引块。"""
    entries: list[_IndexEntry] = []
    cursor = 0
    total = len(payload)
    while cursor < total:
        if total < cursor + _INDEX_ENTRY.size:
            raise CorruptionError("索引项头部不完整", offset)
        key_len, block_offset, block_len = _INDEX_ENTRY.unpack_from(payload, cursor)
        cursor += _INDEX_ENTRY.size
        if total < cursor + key_len:
            raise CorruptionError("索引项里的 key 数据不完整", offset)
        first_key = payload[cursor : cursor + key_len]
        cursor += key_len
        entries.append(_IndexEntry(first_key, block_offset, block_len))
    return entries


# ---------------------------------------------------------------- 写入


class SSTableWriter:
    """把一个**已排好序**的键值流写成 SSTable 文件。

    用法::

        writer = SSTableWriter("sst-000001.sst")
        for key, value in sorted_entries:      # key 必须严格升序
            writer.add(key, value)             # value 传 None 表示墓碑
        meta = writer.finish()

    写完的路径是**临时文件**,由调用方负责 ``os.replace`` 成正式文件 ——
    这样崩溃时不会留下一个"看起来完整但其实写了一半"的 .sst。
    """

    __slots__ = (
        "_path", "_block_size", "_out", "_buf", "_buf_first_key",
        "_index", "_count", "_last_key", "_first_key", "_finished", "_bytes",
        "_bloom",
    )

    def __init__(
        self,
        path: str | Path,
        block_size: int = DEFAULT_BLOCK_SIZE,
        bloom_bits_per_key: int = DEFAULT_BITS_PER_KEY,
    ) -> None:
        if block_size <= 0:
            raise InvalidArgumentError("block_size 必须为正数")
        if bloom_bits_per_key < 0:
            raise InvalidArgumentError("bloom_bits_per_key 不能为负数")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._block_size = block_size
        self._out: BinaryIO = open(self._path, "wb")
        self._buf = bytearray()
        self._buf_first_key: bytes | None = None
        self._index: list[_IndexEntry] = []
        self._count = 0
        self._last_key: bytes | None = None
        self._first_key: bytes | None = None
        self._bytes = 0
        self._finished = False
        # 过滤器**边写边建**:每来一个 key 只留两个 8 字节哈希,
        # 因为位图的位数要等 key 总数确定后才能算出来(见 bloom.BloomBuilder)。
        # 传 0 表示不写过滤器 —— 用来对照"没有过滤器时慢多少"。
        self._bloom = (
            BloomBuilder(bloom_bits_per_key) if bloom_bits_per_key > 0 else None
        )

    # ------------------------------------------------------------ 写入

    def add(self, key: bytes, value: bytes | None) -> None:
        """追加一条记录。``value`` 为 ``None`` 表示墓碑。

        key 必须**严格升序**。这不是可有可无的校验:SSTable 的二分查找
        完全依赖有序性,一旦乱序,查不到数据是静默发生的 ——
        宁可在这里大声报错。
        """
        if self._finished:
            raise InvalidArgumentError("这个 SSTable 已经写完了,不能再 add")

        if self._last_key is not None and key <= self._last_key:
            raise InvalidArgumentError(
                f"key 必须严格升序:收到 {key!r},但上一条是 {self._last_key!r}"
            )

        rec_type = RecordType.DELETE if value is None else RecordType.PUT
        self._buf += encode_payload(rec_type, key, value if value is not None else b"")

        # 墓碑也要进过滤器。它占着一个 key 位置,查询时同样需要
        # "确定不在"这个判断 —— 漏掉的话,查被删的键会多读文件。
        if self._bloom is not None:
            self._bloom.add(key)

        if self._buf_first_key is None:
            self._buf_first_key = key
        if self._first_key is None:
            self._first_key = key
        self._last_key = key
        self._count += 1

        if len(self._buf) >= self._block_size:
            self._flush_block()

    def _flush_block(self) -> None:
        """把当前缓冲区落成一块,并往索引里记一项。"""
        if not self._buf:
            return
        framed = frame(bytes(self._buf))
        offset = self._bytes
        self._out.write(framed)
        self._bytes += len(framed)
        assert self._buf_first_key is not None
        self._index.append(_IndexEntry(self._buf_first_key, offset, len(framed)))
        self._buf.clear()
        self._buf_first_key = None

    # ------------------------------------------------------------ 状态

    @property
    def bytes_written(self) -> int:
        """当前已经写出的字节数,含还在缓冲区、尚未落成块的部分。

        compaction 用它判断"这个文件够大了,可以收尾换下一个"。
        """
        return self._bytes + len(self._buf)

    @property
    def entry_count(self) -> int:
        """已经 add 进来的条目数。"""
        return self._count

    def finish(self) -> SSTableMeta:
        """收尾:写出索引块、过滤器块和 footer,fsync 后关闭。返回元信息。

        写入顺序固定为 ``数据块… → 索引块 → 过滤器块 → footer``。
        footer 最后写,而且它一落盘文件就算完整 —— 因为读的时候
        是从 footer 入手的,footer 不在,前面写的东西就当没写。
        """
        if self._finished:
            raise InvalidArgumentError("这个 SSTable 已经写完了")

        self._flush_block()

        index_offset = self._bytes
        index_bytes = b"".join(entry.encode() for entry in self._index)
        framed_index = frame(index_bytes)
        self._out.write(framed_index)
        self._bytes += len(framed_index)

        # 过滤器块。也用 frame() 包一层,顺便白拿一个 CRC32 ——
        # 过滤器坏了虽然不致命(降级成"没有过滤器"),但能发现总是好的。
        # 关掉过滤器时干脆不写这一块,footer 里记 (0, 0)。
        if self._bloom is not None:
            filter_offset = self._bytes
            framed_filter = frame(self._bloom.build().to_bytes())
            self._out.write(framed_filter)
            self._bytes += len(framed_filter)
            filter_len = len(framed_filter)
        else:
            filter_offset = 0
            filter_len = 0

        self._out.write(
            _FOOTER_V2.pack(
                index_offset, len(framed_index), self._count,
                filter_offset, filter_len, FOOTER_MAGIC_V2,
            )
        )
        self._bytes += FOOTER_SIZE

        self._out.flush()
        os.fsync(self._out.fileno())
        self._out.close()
        self._finished = True

        return SSTableMeta(
            file_id=parse_file_id(self._path) or 0,
            path=self._path,
            entry_count=self._count,
            size=self._bytes,
            first_key=self._first_key,
            last_key=self._last_key,
        )

    def abort(self) -> None:
        """放弃写入并关掉句柄(临时文件留给调用方删)。"""
        if self._finished:
            return
        try:
            self._out.close()
        finally:
            self._finished = True

    def __enter__(self) -> "SSTableWriter":
        return self

    def __exit__(self, *exc) -> None:
        if not self._finished:
            self.abort()


# ---------------------------------------------------------------- 读取


class SSTableReader:
    """只读地打开一个 SSTable。

    打开时把**索引整个读进内存**(它很小),数据块按需读 ——
    这正是稀疏索引的意义。过滤器块也一并读进来,它更小。

    同一个 ``cache`` 可以被很多个 reader 共用 —— 引擎就是这么做的,
    这样跨文件的读都能命中同一份缓存。
    """

    __slots__ = (
        "_path", "_fh", "_size", "_index_offset", "_index_len",
        "_count", "_index", "file_id", "_last_key", "_closed",
        "_filter_offset", "_filter_len", "_version", "_bloom",
        "_bloom_corrupt", "_cache",
    )

    def __init__(
        self,
        path: str | Path,
        file_id: int | None = None,
        cache: BlockCache | None = None,
    ) -> None:
        self._path = Path(path)
        self._fh: BinaryIO = open(self._path, "rb")
        self._size = self._path.stat().st_size
        self._closed = False
        self._last_key: bytes | None = None
        self._index: list[_IndexEntry] = []
        self._bloom: BloomFilter | None = None
        self._bloom_corrupt = False
        self._cache = cache
        self._version = 0
        self._filter_offset = 0
        self._filter_len = 0

        parsed = parse_file_id(self._path)
        self.file_id = file_id if file_id is not None else (
            parsed if parsed is not None else -1
        )

        try:
            self._read_footer()
            self._read_index()
            self._read_filter()
        except Exception:
            self._fh.close()
            self._closed = True
            raise

    # ------------------------------------------------------------ 加载

    def _read_footer(self) -> None:
        """解析 footer。

        **先读末尾 8 字节的 magic 判版本,再按版本解析整个 footer** ——
        因为 v1 和 v2 的 footer 长度不同,不知道版本就没法知道该从
        哪里开始读。magic 放在最末尾正是为了这一点。
        """
        if self._size < FOOTER_SIZE_V1:
            raise CorruptionError(
                f"文件太小,连 footer 都放不下({self._size} 字节):{self._path}"
            )

        self._fh.seek(self._size - _MAGIC_SIZE)
        magic = self._fh.read(_MAGIC_SIZE)

        if magic == FOOTER_MAGIC_V2:
            footer_size = FOOTER_SIZE
            self._version = 2
        elif magic == FOOTER_MAGIC_V1:
            footer_size = FOOTER_SIZE_V1
            self._version = 1
        else:
            raise CorruptionError(
                f"footer magic 不匹配(期望 {FOOTER_MAGIC_V2!r} 或 "
                f"{FOOTER_MAGIC_V1!r},实际 {magic!r}),"
                f"这可能不是 SSTable:{self._path}"
            )

        if self._size < footer_size:
            raise CorruptionError(
                f"文件太小,放不下 {footer_size} 字节的 footer({self._size} 字节)"
            )

        self._fh.seek(self._size - footer_size)
        raw = self._fh.read(footer_size)
        footer_start = self._size - footer_size

        if self._version == 2:
            (index_offset, index_len, count,
             filter_offset, filter_len, _magic) = _FOOTER_V2.unpack(raw)
            self._filter_offset = filter_offset
            self._filter_len = filter_len
        else:
            index_offset, index_len, count, _magic = _FOOTER_V1.unpack(raw)
            self._filter_offset = 0
            self._filter_len = 0

        self._validate_layout(index_offset, index_len, footer_start)

        self._index_offset = index_offset
        self._index_len = index_len
        self._count = count

    def _validate_layout(self, index_offset: int, index_len: int,
                         footer_start: int) -> None:
        """核对各块的位置能不能拼成完整的文件。

        布局约定(见模块开头):``数据块… | 索引块 | 过滤器块 | footer``。
        三者的位置必须严丝合缝,否则说明文件被改动过 —— 宁可现在报错,
        也不要等到某次查询读出莫名其妙的结果。
        """
        index_end = index_offset + index_len
        if index_offset < 0 or index_len < 0 or index_end > footer_start:
            raise CorruptionError(
                f"索引块越界:索引结束于 {index_end},但 footer 起始于 {footer_start}",
                index_offset,
            )

        if self._version == 2 and self._filter_len > 0:
            if index_end != self._filter_offset:
                raise CorruptionError(
                    f"索引块和过滤器块之间有空隙或重叠:索引结束于 {index_end},"
                    f"过滤器起始于 {self._filter_offset}"
                )
            if self._filter_offset + self._filter_len != footer_start:
                raise CorruptionError(
                    f"过滤器块结束位置和 footer 对不上:"
                    f"{self._filter_offset + self._filter_len} != {footer_start}"
                )
        elif index_end != footer_start:
            raise CorruptionError(
                f"索引位置和文件大小对不上:索引结束于 {index_end},"
                f"但 footer 起始于 {footer_start}"
            )

    def _read_index(self) -> None:
        self._fh.seek(self._index_offset)
        framed = self._fh.read(self._index_len)
        if len(framed) < self._index_len:
            raise CorruptionError("索引块读不完整", self._index_offset)
        self._index = _decode_index(unframe(framed, self._index_offset),
                                    self._index_offset)

    def _read_filter(self) -> None:
        """读布隆过滤器。

        ⚠️ 这里的容错策略和数据块**故意不同**:

            数据块损坏 → 抛错,让引擎拒绝启动(读不出数据就是丢数据)
            过滤器损坏 → 记一笔,降级成"没有过滤器",继续打开

        因为过滤器只影响**快慢**,不影响**对错**。为了一个坏掉的
        优化结构而让用户完全打不开数据,是得不偿失的。
        降级之后 ``might_contain()`` 恒返回 True,行为退回阶段 3,
        结果依然正确,只是多读几次文件。
        """
        if self._version < 2 or self._filter_len == 0:
            return
        try:
            self._fh.seek(self._filter_offset)
            framed = self._fh.read(self._filter_len)
            if len(framed) < self._filter_len:
                raise CorruptionError("过滤器块读不完整", self._filter_offset)
            self._bloom = BloomFilter.from_bytes(
                unframe(framed, self._filter_offset)
            )
        except CorruptionError:
            # 降级,而不是让整个文件打不开
            self._bloom = None
            self._bloom_corrupt = True

    # ------------------------------------------------------------ 读取

    def _read_block(self, offset: int, length: int) -> list[tuple[RecordType, bytes, bytes]]:
        """读一块并拆成多条记录。

        ⚠️ **长度必须先校验再读**。这里的 ``length`` 最终来自磁盘上的索引,
        而损坏的数据可以让它变成一个天文数字 —— 直接 ``read(1 << 40)``
        会先尝试分配 1 TiB 内存,进程当场就没了,根本轮不到后面的
        "读到的字节数不够" 检查。先核对边界,再申请内存。
        """
        if offset < 0 or length < 0 or offset + length > self._size:
            raise CorruptionError(
                f"数据块越界:偏移 {offset}、长度 {length},"
                f"但文件只有 {self._size} 字节",
                offset,
            )

        self._fh.seek(offset)
        framed = self._fh.read(length)
        if len(framed) < length:
            raise CorruptionError(
                f"数据块读不完整:期望 {length} 字节,实际 {len(framed)} 字节", offset
            )
        payload = unframe(framed, offset)
        return list(iter_payload(payload, offset))

    def _find_block(self, key: bytes) -> int:
        """二分查找 key 可能落在哪个块里。

        返回"起始 key ≤ 目标 key 的最后一个块"的下标;若目标比第一块的
        起始 key 还小,返回 -1(说明整个表里都没有)。
        """
        if not self._index:
            return -1
        if key < self._index[0].first_key:
            return -1

        lo, hi = 0, len(self._index) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._index[mid].first_key <= key:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _read_block_at(self, index: int, use_cache: bool = True) -> list[tuple]:
        """按块下标读一块,优先走缓存。

        缓存键用**块下标**而不是文件偏移:同一块在文件里的偏移是固定的,
        但用下标更直观,而且和索引项一一对应。

        ``use_cache=False`` 用于顺序全扫 —— 那些块只会被读一次,
        塞进 LRU 会把真正的热点挤出去(缓存污染)。
        """
        if use_cache and self._cache is not None:
            cached = self._cache.get(self.file_id, index)
            if cached is not None:
                return cached

        entry = self._index[index]
        block = self._read_block(entry.offset, entry.length)

        if use_cache and self._cache is not None:
            self._cache.put(self.file_id, index, block)
        return block

    def might_contain(self, key: bytes) -> bool:
        """``False`` 表示这个文件**一定**没有这个 key,调用方可以跳过它。

        没有过滤器时(v1 老文件、或过滤器损坏降级)恒返回 ``True`` ——
        "不知道"就老老实实去读,绝不能瞎猜"不在"。
        """
        if self._bloom is None:
            return True
        return self._bloom.might_contain(key)

    def get(self, key: bytes) -> tuple[bool, bytes | None]:
        """查一个 key,返回 ``(是否找到, 值)``。

        找到墓碑时返回 ``(True, None)`` —— 这个区分很重要:
        它告诉调用方"**不必**再往更旧的文件里找了",否则被删掉的键
        会从底层 SSTable 里复活。
        """
        if self._count == 0:
            return (False, None)

        # 便宜的检查放在前面:key 超出这个文件的键范围,直接说不在。
        # (这个范围判断是免费的,因为 last_key 已经从索引里缓存好了)
        last = self.last_key
        if last is not None and key > last:
            return (False, None)

        index = self._find_block(key)
        if index < 0:
            return (False, None)

        for rec_type, found_key, value in self._read_block_at(index):
            if found_key == key:
                return (True, None if rec_type is RecordType.DELETE else value)
            if found_key > key:
                break       # 块内也是升序,后面不可能再有
        return (False, None)

    def iter_entries(self, fill_cache: bool = False) -> Iterator[tuple[bytes, bytes | None]]:
        """按 key 升序产出 ``(key, value)``;墓碑的 value 为 ``None``。

        ``fill_cache`` 默认是 ``False``:这个接口的主要用途是 compaction
        和全量扫描,每个块只读一次。让它填缓存只会把热点数据挤出去。
        """
        for index in range(len(self._index)):
            for rec_type, key, value in self._read_block_at(index, fill_cache):
                yield key, (None if rec_type is RecordType.DELETE else value)

    # ------------------------------------------------------------ 元信息

    @property
    def path(self) -> Path:
        return self._path

    @property
    def size(self) -> int:
        """文件字节数。"""
        return self._size

    @property
    def entry_count(self) -> int:
        """条目数(含墓碑)。"""
        return self._count

    @property
    def block_count(self) -> int:
        """数据块个数,等于索引项个数。"""
        return len(self._index)

    @property
    def first_key(self) -> bytes | None:
        """最小的 key。索引里直接就有,不需要读数据块。"""
        return self._index[0].first_key if self._index else None

    @property
    def last_key(self) -> bytes | None:
        """最大的 key。

        索引存的是每块的**起始** key,所以拿不到表尾 —— 需要读最后一个块。
        只在第一次访问时读一次,之后缓存。
        """
        if self._last_key is None and self._index:
            block = self._read_block_at(len(self._index) - 1)
            if block:
                self._last_key = block[-1][1]
        return self._last_key

    # ------------------------------------------------------------ 过滤器

    @property
    def version(self) -> int:
        """文件的格式版本(1 = 阶段 2/3,2 = 阶段 4 起)。"""
        return self._version

    @property
    def has_bloom_filter(self) -> bool:
        """这个文件里有没有可用的布隆过滤器。"""
        return self._bloom is not None

    @property
    def bloom_filter(self) -> BloomFilter | None:
        return self._bloom

    @property
    def bloom_corrupt(self) -> bool:
        """过滤器块损坏、已经降级成"没有过滤器"了吗。"""
        return self._bloom_corrupt

    @property
    def filter_size(self) -> int:
        """过滤器块在文件里占多少字节(含 8 字节头部)。"""
        return self._filter_len

    def __len__(self) -> int:
        return self._count

    def close(self) -> None:
        if self._closed:
            return
        self._fh.close()
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> "SSTableReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        state = "closed" if self._closed else "open"
        bloom = "有过滤器" if self.has_bloom_filter else "无过滤器"
        return (
            f"<SSTableReader {self._path.name} v{self._version} entries={self._count} "
            f"blocks={self.block_count} {bloom} {state}>"
        )


def read_all(path: str | Path) -> list[tuple[bytes, bytes | None]]:
    """把整个 SSTable 读成列表。诊断和测试用。"""
    with SSTableReader(path) as reader:
        return list(reader.iter_entries())
