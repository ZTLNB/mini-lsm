"""SSTable(Sorted String Table)—— 不可变的有序磁盘表。

这是 LSM 里**唯一存到磁盘上的数据格式**。内存表攒满之后整体刷成一个
SSTable,之后这个文件只读、永不修改。所有复杂度都来自这一点:
既然文件不可变,就不存在"就地更新"和"写一半被别人读到"的问题。

文件布局::

    +---------------------------+
    | 数据块 0                  |  多条 entry,按 key 升序
    +---------------------------+  block := [len:4][crc32:4] + entries
    | 数据块 1                  |
    +---------------------------+
    | ...                       |
    +---------------------------+
    | 索引块                    |  稀疏索引:每个数据块的**起始 key** → (偏移, 长度)
    +---------------------------+  同样是 [len:4][crc32:4] + entries
    | footer(固定 32 字节)      |  索引偏移、索引长度、条目数、magic
    +---------------------------+

entry 的字节布局和 WAL 的 payload **完全一致**:
``rec_type(1) + key_len(4) + key + value_len(4) + value``。
两者共用 ``encode_payload`` / ``decode_payload``,少一套编解码就少一处会写错的地方。

两个关键设计:

**为什么是"稀疏"索引?**
    每 4 KiB 一个索引项,而不是每个 key 一个。索引常驻内存,必须小 ——
    1 亿条记录如果逐条建索引,光索引就要好几 GB。稀疏索引让索引大小
    正比于"块的个数"而不是"记录的个数",代价是查一个 key 需要
    在块内顺序扫一遍(块只有 4 KiB,这点开销可以忽略)。

**为什么 CRC 按块算而不是按条算?**
    SSTable 的读取粒度就是块 —— 要么整块读进来用,要么根本不读。
    按块校验和按条校验能发现的问题一样多,但校验开销只有 1/N。

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
    "FOOTER_MAGIC",
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

#: footer 的 magic。读文件时先看它,能立刻认出"这根本不是 SSTable"。
FOOTER_MAGIC = b"MINILSM1"

#: footer 固定 32 字节,位于文件末尾:
#: 索引偏移(8) + 索引长度(8) + 条目数(8) + magic(8)
_FOOTER = struct.Struct(">QQQ8s")
FOOTER_SIZE = _FOOTER.size

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
    )

    def __init__(self, path: str | Path, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        if block_size <= 0:
            raise InvalidArgumentError("block_size 必须为正数")
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
        """收尾:写出索引块和 footer,fsync 后关闭。返回元信息。"""
        if self._finished:
            raise InvalidArgumentError("这个 SSTable 已经写完了")

        self._flush_block()

        index_offset = self._bytes
        index_bytes = b"".join(entry.encode() for entry in self._index)
        framed_index = frame(index_bytes)
        self._out.write(framed_index)
        self._bytes += len(framed_index)

        self._out.write(
            _FOOTER.pack(index_offset, len(framed_index), self._count, FOOTER_MAGIC)
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
    这正是稀疏索引的意义。
    """

    __slots__ = (
        "_path", "_fh", "_size", "_index_offset", "_index_len",
        "_count", "_index", "file_id", "_last_key", "_closed",
    )

    def __init__(self, path: str | Path, file_id: int | None = None) -> None:
        self._path = Path(path)
        self._fh: BinaryIO = open(self._path, "rb")
        self._size = self._path.stat().st_size
        self._closed = False
        self._last_key: bytes | None = None
        self._index: list[_IndexEntry] = []

        parsed = parse_file_id(self._path)
        self.file_id = file_id if file_id is not None else (
            parsed if parsed is not None else -1
        )

        try:
            self._read_footer()
            self._read_index()
        except Exception:
            self._fh.close()
            self._closed = True
            raise

    # ------------------------------------------------------------ 加载

    def _read_footer(self) -> None:
        if self._size < FOOTER_SIZE:
            raise CorruptionError(
                f"文件太小,连 footer 都放不下({self._size} 字节):{self._path}"
            )

        self._fh.seek(self._size - FOOTER_SIZE)
        raw = self._fh.read(FOOTER_SIZE)
        index_offset, index_len, count, magic = _FOOTER.unpack(raw)

        if magic != FOOTER_MAGIC:
            raise CorruptionError(
                f"footer magic 不匹配(期望 {FOOTER_MAGIC!r},实际 {magic!r}),"
                f"这可能不是 SSTable:{self._path}"
            )

        expected_index_end = self._size - FOOTER_SIZE
        if index_offset + index_len != expected_index_end:
            raise CorruptionError(
                f"索引位置和文件大小对不上:索引结束于 "
                f"{index_offset + index_len},但 footer 起始于 {expected_index_end}"
            )

        self._index_offset = index_offset
        self._index_len = index_len
        self._count = count

    def _read_index(self) -> None:
        self._fh.seek(self._index_offset)
        framed = self._fh.read(self._index_len)
        if len(framed) < self._index_len:
            raise CorruptionError("索引块读不完整", self._index_offset)
        self._index = _decode_index(unframe(framed, self._index_offset),
                                    self._index_offset)

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

    def get(self, key: bytes) -> tuple[bool, bytes | None]:
        """查一个 key,返回 ``(是否找到, 值)``。

        找到墓碑时返回 ``(True, None)`` —— 这个区分很重要:
        它告诉调用方"**不必**再往更旧的文件里找了",否则被删掉的键
        会从底层 SSTable 里复活。
        """
        if self._count == 0:
            return (False, None)

        last = self.last_key
        if last is not None and key > last:
            return (False, None)

        index = self._find_block(key)
        if index < 0:
            return (False, None)

        entry = self._index[index]
        for rec_type, found_key, value in self._read_block(entry.offset, entry.length):
            if found_key == key:
                return (True, None if rec_type is RecordType.DELETE else value)
            if found_key > key:
                break       # 块内也是升序,后面不可能再有
        return (False, None)

    def iter_entries(self) -> Iterator[tuple[bytes, bytes | None]]:
        """按 key 升序产出 ``(key, value)``;墓碑的 value 为 ``None``。"""
        for entry in self._index:
            for rec_type, key, value in self._read_block(entry.offset, entry.length):
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
            entry = self._index[-1]
            block = self._read_block(entry.offset, entry.length)
            if block:
                self._last_key = block[-1][1]
        return self._last_key

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
        return (
            f"<SSTableReader {self._path.name} entries={self._count} "
            f"blocks={self.block_count} {state}>"
        )


def read_all(path: str | Path) -> list[tuple[bytes, bytes | None]]:
    """把整个 SSTable 读成列表。诊断和测试用。"""
    with SSTableReader(path) as reader:
        return list(reader.iter_entries())
