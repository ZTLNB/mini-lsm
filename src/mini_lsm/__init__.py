"""mini-lsm —— 从零实现的 LSM-Tree 存储引擎。

已完成:
    阶段 1  WAL 预写日志 + MemTable 内存表 + 崩溃恢复
    阶段 2  SSTable 刷盘 + 多来源读路径(数据量不再受内存限制)
    阶段 3  Manifest + 分层 Compaction(控制读放大,清理墓碑)
    阶段 4  Bloom Filter + Block Cache(消掉无谓的磁盘读)
    阶段 5  快照读 + 流式范围扫描(一致视图,内存不随结果集增长)

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

        # 一致视图:创建之后的写入对它不可见
        with db.snapshot() as snap:
            db.put("name", "bob")      # 快照看不到这次写入
            print(snap.get_str("name"))  # alice

数据目录里会有 ``wal.log``、``MANIFEST.json`` 和若干 ``sst-*.sst``。
内存表写满后自动刷成 SSTable,文件攒多了自动 compaction,
所以数据量可以远超内存,查询也不会随文件数线性变慢。
"""

from .block_cache import DEFAULT_CACHE_BYTES, BlockCache, CacheStats
from .bloom import (
    DEFAULT_BITS_PER_KEY,
    BloomBuilder,
    BloomFilter,
    estimate_false_positive_rate,
)
from .compaction import (
    CompactionTask,
    level_budget,
    overlapping_files,
    pick_task,
    plan_full_compaction,
)
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
from .manifest import MANIFEST_FILENAME, FileMeta, Manifest, find_file_in_level
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
from .snapshot import ScanCursor, Snapshot, search_levels
from .sstable import (
    DEFAULT_BLOCK_SIZE,
    FOOTER_MAGIC,
    FOOTER_MAGIC_V1,
    FOOTER_MAGIC_V2,
    FOOTER_SIZE,
    FOOTER_SIZE_V1,
    FORMAT_VERSION,
    SSTableMeta,
    SSTableReader,
    SSTableWriter,
    parse_file_id,
    read_all,
    sstable_filename,
)
from .wal import ReplayResult, WAL, read_records
__version__ = "0.5.0"

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
    "FOOTER_SIZE_V1",
    "FOOTER_MAGIC",
    "FOOTER_MAGIC_V1",
    "FOOTER_MAGIC_V2",
    "FORMAT_VERSION",
    # 布隆过滤器
    "BloomFilter",
    "BloomBuilder",
    "DEFAULT_BITS_PER_KEY",
    "estimate_false_positive_rate",
    # 块缓存
    "BlockCache",
    "CacheStats",
    "DEFAULT_CACHE_BYTES",
    # 版本清单
    "Manifest",
    "FileMeta",
    "MANIFEST_FILENAME",
    "find_file_in_level",
    # 快照与流式扫描
    "Snapshot",
    "ScanCursor",
    "search_levels",
    # Compaction
    "CompactionTask",
    "pick_task",
    "plan_full_compaction",
    "overlapping_files",
    "level_budget",
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
