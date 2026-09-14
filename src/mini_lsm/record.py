"""WAL 记录的二进制编解码。

这是整个存储引擎的**地基** —— 磁盘格式一旦定错,后面所有东西都要返工。
所以这里把格式写清楚,并且用 CRC 保护每一个字节。

记录格式(全部大端序 big-endian):

    偏移      长度        字段
    ------    --------    ------------------------------------------
    0         4           payload_len   —— payload 的字节数
    4         4           crc32         —— payload 的 CRC32 校验和
    8         1           rec_type      —— 1=PUT, 2=DELETE
    9         4           key_len
    13        key_len     key
    ...       4           value_len
    ...       value_len   value
    ------    --------    ------------------------------------------
    合计      8 + payload_len

    payload = rec_type(1) + key_len(4) + key + value_len(4) + value

为什么把 payload_len 放在最前面?
    读取时先拿到长度,才知道后面要读多少字节。这样即使文件被截断,
    也能立刻判断出"这条记录不完整",而不是读到一半才发现。

为什么 crc 覆盖整个 payload 而不是分开校验?
    简单且够用。分开校验能定位到具体字段,但 WAL 的场景下
    "这条记录坏了" 已经足够 —— 处理方式都是截断到上一条好记录。

和 SSTable 的关系:
    SSTable 的数据块里存的是**同样布局的 payload 拼接**。
    两者共用 ``encode_payload`` / ``decode_payload``,只是外层容器不同 ——
    WAL 一条记录一个头部,SSTable 一整块一个头部(块级校验)。
    少一套编解码就少一处会写错的地方。
"""

from __future__ import annotations

import struct
import zlib
from enum import IntEnum
from typing import BinaryIO, Iterator

from .errors import ChecksumMismatchError, TruncatedRecordError

# 大端序结构体
_HEADER = struct.Struct(">II")       # payload_len, crc32
_U32 = struct.Struct(">I")           # 通用的 4 字节无符号整数

#: 记录头部(不含 payload)的固定大小
HEADER_SIZE = _HEADER.size

#: 一条 payload 里除 key/value 数据之外的固定开销:
#: rec_type(1) + key_len(4) + value_len(4)
PAYLOAD_OVERHEAD = 1 + 4 + 4


def payload_size(key_len: int, value_len: int) -> int:
    """一条 payload 占用的总字节数。

    顺序解析拼接在一起的 payload 时需要它来推进游标 ——
    ``decode_payload`` 只返回解出来的内容,不返回"消耗了多少字节"。
    """
    return PAYLOAD_OVERHEAD + key_len + value_len


class RecordType(IntEnum):
    """WAL 记录类型。"""

    PUT = 1
    DELETE = 2

    @classmethod
    def from_byte(cls, value: int) -> "RecordType":
        try:
            return cls(value)
        except ValueError as exc:
            raise TruncatedRecordError(
                f"未知的记录类型字节:{value}"
            ) from exc


class Record:
    """一条已解码的 WAL 记录。"""

    __slots__ = ("rec_type", "key", "value", "offset", "size")

    def __init__(
        self,
        rec_type: RecordType,
        key: bytes,
        value: bytes = b"",
        offset: int = 0,
        size: int = 0,
    ) -> None:
        self.rec_type = rec_type
        self.key = key
        self.value = value
        self.offset = offset
        #: 该记录在文件里占用的总字节数(含头部),用于计算有效数据边界
        self.size = size

    @property
    def is_delete(self) -> bool:
        return self.rec_type is RecordType.DELETE

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        preview = self.value[:20]
        return (
            f"Record({self.rec_type.name}, key={self.key!r}, "
            f"value={preview!r}{'...' if len(self.value) > 20 else ''})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Record):
            return NotImplemented
        return (
            self.rec_type == other.rec_type
            and self.key == other.key
            and self.value == other.value
        )


# ---------------------------------------------------------------- 编码


def encode_payload(rec_type: RecordType, key: bytes, value: bytes = b"") -> bytes:
    """把 (类型, key, value) 编码成 payload(**不含**长度/CRC 头部)。

    WAL 记录和 SSTable 数据块共用这个布局:
        payload = rec_type(1) + key_len(4) + key + value_len(4) + value
    """
    return (
        bytes([int(rec_type)])
        + _U32.pack(len(key))
        + key
        + _U32.pack(len(value))
        + value
    )


def frame(payload: bytes) -> bytes:
    """给一段字节加上 ``[长度][CRC32]`` 头部。

    这是 WAL 单条记录的容器,也是 SSTable 单个数据块的容器。
    """
    return _HEADER.pack(len(payload), zlib.crc32(payload)) + payload


def unframe(framed: bytes, offset: int = 0) -> bytes:
    """拆掉 ``[长度][CRC32]`` 头部并校验,返回 payload。

    WAL 是**流式**读取(一条接一条,用 ``iter_records``);
    SSTable 是**按偏移随机**读取(一次读一整块)。两者共用同一套头部格式,
    但需要不同的读取入口 —— 这个函数服务后者。
    """
    if len(framed) < HEADER_SIZE:
        raise TruncatedRecordError(
            f"块头不完整:期望 {HEADER_SIZE} 字节,实际 {len(framed)} 字节", offset
        )

    payload_len, stored_crc = _HEADER.unpack_from(framed, 0)
    payload = framed[HEADER_SIZE:]

    if len(payload) < payload_len:
        raise TruncatedRecordError(
            f"块数据不完整:期望 {payload_len} 字节,实际 {len(payload)} 字节", offset
        )

    payload = payload[:payload_len]
    actual_crc = zlib.crc32(payload)
    if actual_crc != stored_crc:
        raise ChecksumMismatchError(stored_crc, actual_crc, offset)

    return payload


def encode_record(rec_type: RecordType, key: bytes, value: bytes = b"") -> bytes:
    """把一条记录编码成字节串,可直接追加写入文件。

    参数:
        rec_type: PUT 或 DELETE
        key:      键(字节串)
        value:    值;DELETE 记录应传空字节串
    """
    return frame(encode_payload(rec_type, key, value))


# ---------------------------------------------------------------- 解码


def decode_payload_from(
    payload: bytes, cursor: int = 0, offset: int = 0
) -> tuple[RecordType, bytes, bytes]:
    """从 ``payload`` 的 ``cursor`` 位置解出一条记录。

    调用方需自行保证 cursor 处确实有一条完整记录
    (由 ``iter_payload`` / ``iter_records`` 负责)。
    """
    total = len(payload)
    if cursor >= total:
        raise TruncatedRecordError("payload 已到末尾,没有可解析的记录", offset)

    rec_type = RecordType.from_byte(payload[cursor])
    cursor += 1

    if total < cursor + 4:
        raise TruncatedRecordError("payload 缺少 key_len 字段", offset)
    key_len = _U32.unpack_from(payload, cursor)[0]
    cursor += 4

    if total < cursor + key_len:
        raise TruncatedRecordError("payload 中的 key 数据不完整", offset)
    key = payload[cursor : cursor + key_len]
    cursor += key_len

    if total < cursor + 4:
        raise TruncatedRecordError("payload 缺少 value_len 字段", offset)
    value_len = _U32.unpack_from(payload, cursor)[0]
    cursor += 4

    if total < cursor + value_len:
        raise TruncatedRecordError("payload 中的 value 数据不完整", offset)
    value = payload[cursor : cursor + value_len]

    return rec_type, key, value


def decode_payload(payload: bytes, offset: int = 0) -> tuple[RecordType, bytes, bytes]:
    """解析**整段就是一条记录**的 payload,返回 (类型, key, value)。"""
    return decode_payload_from(payload, 0, offset)


def iter_payload(
    payload: bytes, offset: int = 0
) -> Iterator[tuple[RecordType, bytes, bytes]]:
    """按顺序拆开一段**拼接了多条记录**的 payload。

    SSTable 的一个数据块正是这样一段字节。用游标推进而不是反复切片,
    避免在大块上退化成 O(n²) 的内存拷贝。
    """
    cursor = 0
    total = len(payload)
    while cursor < total:
        rec_type, key, value = decode_payload_from(payload, cursor, offset)
        cursor += payload_size(len(key), len(value))
        yield rec_type, key, value


def iter_records(fh: BinaryIO, verify_crc: bool = True) -> Iterator[Record]:
    """从文件对象顺序迭代记录。

    行为约定(**这是崩溃恢复正确性的关键**):

    - 正常读到文件末尾 → 迭代自然结束,不抛异常
    - 记录不完整(崩溃写了一半)→ 抛 ``TruncatedRecordError``
    - CRC 校验失败(数据损坏)→ 抛 ``ChecksumMismatchError``

    调用方应捕获 ``CorruptionError`` 并**截断到上一条好记录**,
    而不是把异常抛给用户 —— 崩溃残留是 LSM 的预期内情况。

    参数:
        verify_crc: 设为 False 可跳过校验(仅在诊断工具中使用)
    """
    offset = 0
    while True:
        header = fh.read(HEADER_SIZE)

        # 干净的文件末尾:一个字节都不剩
        if not header:
            return

        # 头部读不满 —— 崩溃时正写到一半
        if len(header) < HEADER_SIZE:
            raise TruncatedRecordError(
                f"记录头不完整:期望 {HEADER_SIZE} 字节,实际 {len(header)} 字节",
                offset,
            )

        payload_len, stored_crc = _HEADER.unpack(header)

        payload = fh.read(payload_len)
        if len(payload) < payload_len:
            raise TruncatedRecordError(
                f"payload 不完整:期望 {payload_len} 字节,实际 {len(payload)} 字节",
                offset,
            )

        if verify_crc:
            actual_crc = zlib.crc32(payload)
            if actual_crc != stored_crc:
                raise ChecksumMismatchError(stored_crc, actual_crc, offset)

        rec_type, key, value = decode_payload(payload, offset)
        yield Record(rec_type, key, value, offset, HEADER_SIZE + payload_len)

        offset += HEADER_SIZE + payload_len
