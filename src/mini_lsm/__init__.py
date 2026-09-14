"""mini-lsm —— 从零实现的 LSM-Tree 存储引擎。

已完成:
    阶段 1  WAL 预写日志 + MemTable 内存表 + 崩溃恢复
    阶段 2  SSTable 刷盘 + 多来源读路径(数据量不再受内存限制)

快速开始::

    from mini_lsm import LSMEngine

    with LSMEngine("./mydata") as db:
        db.put("name", "alice")
        db.put("city", "shanghai")
        print(db.get_str("name"))      # alice

        db.delete("city")
        print(db.get("city"))          # None

        for key, value in db.scan(b"a", b"z"):
            print(key, value)

数据目录里会有 ``wal.log`` 和若干 ``sst-*.sst``。
内存表写满后自动刷成 SSTable,所以数据量可以远超内存。
"""

from .engine import WAL_FILENAME, EngineStats, LSMEngine, to_bytes
from .errors import (
    ChecksumMismatchError,
    ClosedError,
    CorruptionError,
    InvalidArgumentError,
    LSMError,
    TruncatedRecordError,
)
from .iterator import MergingIterator
from .memtable import DEFAULT_CAPACITY_BYTES, MemTable
from .record import (
    HEADER_SIZE,
    PAYLOAD_OVERHEAD,
    Record,
    RecordType,
    decode_payload,
    encode_payload,
    encode_record,
    frame,
    iter_payload,
    iter_records,
    payload_size,
    unframe,
)
from .sstable import (
    DEFAULT_BLOCK_SIZE,
    FOOTER_SIZE,
    SSTableMeta,
    SSTableReader,
    SSTableWriter,
    parse_file_id,
    read_all,
    sstable_filename,
)
from .wal import ReplayResult, WAL, read_records

__version__ = "0.2.0"

__all__ = [
    # 主入口
    "LSMEngine",
    "EngineStats",
    "to_bytes",
    "WAL_FILENAME",
    # 内存侧
    "MemTable",
    "DEFAULT_CAPACITY_BYTES",
    # 磁盘侧
    "SSTableWriter",
    "SSTableReader",
    "SSTableMeta",
    "sstable_filename",
    "parse_file_id",
    "read_all",
    "DEFAULT_BLOCK_SIZE",
    "FOOTER_SIZE",
    # 日志
    "WAL",
    "ReplayResult",
    "read_records",
    # 归并
    "MergingIterator",
    # 记录层
    "Record",
    "RecordType",
    "encode_record",
    "encode_payload",
    "decode_payload",
    "iter_records",
    "iter_payload",
    "frame",
    "unframe",
    "payload_size",
    "HEADER_SIZE",
    "PAYLOAD_OVERHEAD",
    # 异常
    "LSMError",
    "CorruptionError",
    "TruncatedRecordError",
    "ChecksumMismatchError",
    "ClosedError",
    "InvalidArgumentError",
]
