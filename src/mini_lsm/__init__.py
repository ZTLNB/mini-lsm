"""mini-lsm —— 从零实现的 LSM-Tree 存储引擎。

阶段 1 已实现:WAL 预写日志 + MemTable 内存表 + 崩溃恢复。

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
"""

from .engine import EngineStats, LSMEngine, to_bytes
from .errors import (
    ChecksumMismatchError,
    ClosedError,
    CorruptionError,
    InvalidArgumentError,
    LSMError,
    TruncatedRecordError,
)
from .memtable import DEFAULT_CAPACITY_BYTES, MemTable
from .record import HEADER_SIZE, Record, RecordType, encode_record, iter_records
from .wal import ReplayResult, WAL, read_records

__version__ = "0.1.0"

__all__ = [
    # 主入口
    "LSMEngine",
    "EngineStats",
    "to_bytes",
    # 组件
    "MemTable",
    "WAL",
    "ReplayResult",
    "read_records",
    # 记录层
    "Record",
    "RecordType",
    "encode_record",
    "iter_records",
    "HEADER_SIZE",
    # 异常
    "LSMError",
    "CorruptionError",
    "TruncatedRecordError",
    "ChecksumMismatchError",
    "ClosedError",
    "InvalidArgumentError",
    # 常量
    "DEFAULT_CAPACITY_BYTES",
]
