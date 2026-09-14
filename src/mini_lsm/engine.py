"""存储引擎主入口 —— 把 WAL 和 MemTable 串成可用的 KV 存储。

阶段 1 的能力边界(说清楚,免得误解):
    这是一个**带崩溃恢复能力的内存 KV 存储**。所有数据同时存在于
    内存表和 WAL 中,进程崩溃后能完整恢复。
    但它**还没有 SSTable**,所以数据不能超过内存容量,重启时也要
    把整个 WAL 重放一遍。这两个限制会在阶段 2 引入磁盘文件后解除。

写路径(必须严格保持这个顺序):
    1. 先把变更追加到 WAL        ← 落盘,这是持久性的来源
    2. 再写入内存表              ← 快速生效
    顺序不能反。如果先改内存再写日志,那么"日志还没写完就崩溃"
    的情况下,内存里的改动会丢失,而调用方已经收到"成功"了。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .errors import ClosedError, InvalidArgumentError
from .memtable import DEFAULT_CAPACITY_BYTES, MemTable
from .record import RecordType
from .wal import ReplayResult, WAL

#: 默认的 WAL 文件名
WAL_FILENAME = "wal.log"


def to_bytes(value: object, name: str) -> bytes:
    """把用户输入统一转成 bytes。

    允许传 str(按 UTF-8 编码),这样交互式使用时不必到处写 b""。
    """
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise InvalidArgumentError(
        f"{name} 必须是 str 或 bytes,实际是 {type(value).__name__}"
    )


@dataclass
class EngineStats:
    """引擎当前状态的快照,用于监控和调试。"""

    memtable_entries: int = 0
    memtable_size: int = 0
    memtable_capacity: int = 0
    tombstones: int = 0
    wal_size: int = 0
    flush_pending: bool = False
    recovered_records: int = 0
    recovery_truncated: bool = False
    recovery_reason: str | None = None

    @property
    def memtable_usage_ratio(self) -> float:
        if self.memtable_capacity <= 0:
            return 0.0
        return self.memtable_size / self.memtable_capacity

    def __str__(self) -> str:
        lines = [
            f"内存表: {self.memtable_entries} 条,"
            f" {self.memtable_size} / {self.memtable_capacity} 字节"
            f" ({self.memtable_usage_ratio:.1%})",
            f"墓碑:   {self.tombstones} 个",
            f"WAL:    {self.wal_size} 字节",
            f"待刷盘: {'是' if self.flush_pending else '否'}",
        ]
        if self.recovered_records:
            note = "(含截断)" if self.recovery_truncated else ""
            lines.append(f"启动恢复: 重放 {self.recovered_records} 条记录{note}")
            if self.recovery_reason:
                lines.append(f"          {self.recovery_reason}")
        return "\n".join(lines)


class LSMEngine:
    """基于 LSM 思想的键值存储引擎。

    用法::

        with LSMEngine("./data") as db:
            db.put("name", "alice")
            print(db.get("name"))     # b'alice'
            db.delete("name")
            print(db.get("name"))     # None
    """

    def __init__(
        self,
        data_dir: str | Path,
        memtable_capacity: int = DEFAULT_CAPACITY_BYTES,
        wal_sync_on_write: bool = False,
    ) -> None:
        """
        参数:
            data_dir:           数据目录,不存在会自动创建
            memtable_capacity:  内存表容量上限(字节)
            wal_sync_on_write:  每次写入是否立即 fsync。
                                False 能扛住进程崩溃;True 才能扛住断电。
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.wal_path = self.data_dir / WAL_FILENAME

        self._lock = threading.RLock()
        self._closed = False
        self._memtable = MemTable(memtable_capacity)

        # 先打开 WAL,再恢复 —— 恢复过程需要读日志
        self._wal = WAL(self.wal_path, sync_on_write=wal_sync_on_write)
        self._recovery: ReplayResult = self._recover()

    # ------------------------------------------------------------ 崩溃恢复

    def _recover(self) -> ReplayResult:
        """重放 WAL,重建内存表。

        ``WAL.recover`` 会自动截断末尾的残骸(崩溃时写了一半的记录),
        所以这一步结束后,日志里的每个字节都是有效的。
        """
        result = self._wal.recover()

        for rec in result.records:
            if rec.rec_type is RecordType.PUT:
                self._memtable.put(rec.key, rec.value)
            elif rec.rec_type is RecordType.DELETE:
                self._memtable.delete(rec.key)

        return result

    # ------------------------------------------------------------ 写入

    def put(self, key: object, value: object) -> None:
        """写入一个键值对。

        先写 WAL(持久化),再改内存表。返回即代表数据已经"不会因
        进程崩溃而丢失"(前提是 ``wal_sync_on_write`` 为 True 时才能
        扛住断电,否则只扛得住进程崩溃)。
        """
        kb = to_bytes(key, "key")
        vb = to_bytes(value, "value")

        with self._lock:
            self._ensure_open()
            self._wal.append(RecordType.PUT, kb, vb)
            self._memtable.put(kb, vb)

    def delete(self, key: object) -> None:
        """删除一个键。

        实际动作是写入一个墓碑。键在逻辑上立即不可见,但物理数据
        要等到后续 compaction 才会被真正清除。
        """
        kb = to_bytes(key, "key")

        with self._lock:
            self._ensure_open()
            self._wal.append(RecordType.DELETE, kb)
            self._memtable.delete(kb)

    def put_many(self, items: dict | list[tuple]) -> int:
        """批量写入,返回写入条数。

        注意:这不是一个原子事务 —— 中途失败会留下部分写入。
        真正的原子性要等阶段 5 的 MVCC。
        """
        pairs = list(items.items()) if isinstance(items, dict) else list(items)
        for key, value in pairs:
            self.put(key, value)
        return len(pairs)

    # ------------------------------------------------------------ 读取

    def get(self, key: object) -> bytes | None:
        """查询键。返回 ``None`` 表示不存在或已被删除。"""
        kb = to_bytes(key, "key")

        with self._lock:
            self._ensure_open()
            # 阶段 1 只有内存表可查;阶段 2 会在这里加上 SSTable 的查找
            return self._memtable.get(kb)

    def get_str(self, key: object, encoding: str = "utf-8") -> str | None:
        """查询并解码成字符串,方便交互式使用。"""
        raw = self.get(key)
        return None if raw is None else raw.decode(encoding, errors="replace")

    def contains(self, key: object) -> bool:
        """键是否存在(墓碑算不存在)。"""
        return self.get(key) is not None

    def scan(
        self,
        start: object | None = None,
        end: object | None = None,
    ) -> Iterator[tuple[bytes, bytes]]:
        """按 key 升序扫描区间 ``[start, end)``。

        两端都可以省略:省略 start 表示从头开始,省略 end 表示扫到末尾。
        墓碑会被自动跳过。这展示了"有序表"的价值 —— 范围查询是顺序读。
        """
        with self._lock:
            self._ensure_open()
            # 在锁内先把结果物化成列表,避免迭代过程中被写入干扰
            lo = to_bytes(start, "start") if start is not None else b""
            hi = to_bytes(end, "end") if end is not None else None

            snapshot: list[tuple[bytes, bytes]] = []
            for key, value in self._memtable.items():
                if key < lo:
                    continue
                if hi is not None and key >= hi:
                    break
                if value is None:      # 墓碑,跳过
                    continue
                snapshot.append((key, value))

        return iter(snapshot)

    def keys(self) -> Iterator[bytes]:
        """所有存活键,升序。"""
        return (k for k, _ in self.scan())

    # ------------------------------------------------------------ 状态

    def stats(self) -> EngineStats:
        """返回当前状态快照。"""
        with self._lock:
            return EngineStats(
                memtable_entries=len(self._memtable),
                memtable_size=self._memtable.approximate_size,
                memtable_capacity=self._memtable.capacity_bytes,
                tombstones=self._memtable.tombstone_count,
                wal_size=self._wal.size,
                flush_pending=self._memtable.is_full,
                recovered_records=self._recovery.record_count,
                recovery_truncated=self._recovery.truncated,
                recovery_reason=self._recovery.reason,
            )

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def recovery_result(self) -> ReplayResult:
        """启动时那次恢复的完整结果,便于测试和诊断。"""
        return self._recovery

    # ------------------------------------------------------------ 生命周期

    def flush(self) -> None:
        """把内存表刷到磁盘。

        阶段 1 尚未实现 SSTable,所以这里只是确保 WAL 已落盘 ——
        数据不会丢,但内存也不会释放。
        """
        with self._lock:
            self._ensure_open()
            self._wal.sync()

    def sync(self) -> None:
        """强制把 WAL 刷到磁盘。想扛断电就调用它。"""
        with self._lock:
            self._ensure_open()
            self._wal.sync()

    def close(self) -> None:
        """关闭引擎(会先 fsync,确保已确认的写入真正落盘)。"""
        with self._lock:
            if self._closed:
                return
            self._wal.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise ClosedError()

    def __enter__(self) -> "LSMEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        state = "closed" if self._closed else "open"
        return (
            f"<LSMEngine {self.data_dir} entries={len(self._memtable)} {state}>"
        )
