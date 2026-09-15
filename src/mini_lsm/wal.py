"""预写日志(Write-Ahead Log)。

职责:
    1. 在数据进入内存表之前,先把变更**顺序追加**到磁盘
    2. 进程崩溃后,通过重放日志恢复内存里丢失的数据

为什么需要它:
    内存表在进程崩溃时会全部丢失。如果没有 WAL,已经向调用方
    返回"写成功"的数据就凭空消失了 —— 这违反持久性。
    WAL 的思路是"先记日志,再改内存":日志是纯顺序写,
    比随机写快得多,所以这笔额外开销是划算的。

崩溃残留怎么处理(**这是本模块最关键的设计**):
    进程可能在写记录的任意时刻被杀掉,日志末尾就可能留下半条记录。
    恢复时必须容忍这种情况 —— 读到损坏就截断,保留之前所有完好记录。
    绝不能因为末尾几个坏字节就把整个日志判死刑。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Iterable

from .errors import CorruptionError, LSMError
from .record import HEADER_SIZE, Record, RecordType, encode_record, iter_records


@dataclass
class ReplayResult:
    """一次日志重放的结果。"""

    records: list[Record] = field(default_factory=list)
    #: 完好数据的字节边界。小于文件实际大小说明末尾有残骸。
    valid_bytes: int = 0
    #: 是否检测到损坏/截断
    truncated: bool = False
    #: 截断原因(人类可读),便于诊断
    reason: str | None = None

    @property
    def record_count(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        tail = f", reason={self.reason!r}" if self.reason else ""
        return (
            f"ReplayResult(records={len(self.records)}, "
            f"valid_bytes={self.valid_bytes}, truncated={self.truncated}{tail})"
        )


class WAL:
    """单个预写日志文件。

    非线程安全的部分已用锁保护,可以安全地被多个线程调用。

    写句柄是**延迟打开**的:构造 WAL 对象本身不会创建文件,也不会
    占用句柄。只有真正调用 ``append`` / ``sync`` 时才会打开。
    这样做是因为"只想重放一下看看日志里有什么"是很常见的用法,
    如果构造时就打开写句柄,那种用法就会泄漏文件句柄。
    """

    def __init__(self, path: str | Path, sync_on_write: bool = False) -> None:
        """
        参数:
            path:            日志文件路径(首次写入时自动创建,含父目录)
            sync_on_write:   每次追加后是否 fsync。
                             True 更安全(能扛住断电),但慢一个数量级;
                             False 时能扛住**进程崩溃**(数据在 OS 页缓存里),
                             但断电可能丢最近几条。默认 False。
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sync_on_write = sync_on_write

        self._lock = threading.Lock()
        self._closed = False
        #: 追加句柄;None 表示尚未打开(还没有任何写入)
        self._fh: BinaryIO | None = None
        self._size = self.path.stat().st_size if self.path.exists() else 0

    # ------------------------------------------------------------ 属性

    @property
    def size(self) -> int:
        """当前日志文件字节数。"""
        return self._size

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise LSMError(f"WAL 已关闭:{self.path}")

    def _ensure_writer(self) -> BinaryIO:
        """按需打开追加句柄。"""
        if self._fh is None:
            self._fh = open(self.path, "ab")
        return self._fh

    # ------------------------------------------------------------ 写入

    def append(
        self, rec_type: RecordType, key: bytes, value: bytes = b""
    ) -> int:
        """追加一条记录,返回它写入的起始偏移。

        写完后数据已经交给操作系统(``write`` 系统调用),
        但未必落到磁盘 —— 取决于 ``sync_on_write``。
        """
        with self._lock:
            self._ensure_open()
            fh = self._ensure_writer()
            payload = encode_record(rec_type, key, value)
            offset = self._size
            fh.write(payload)
            fh.flush()
            self._size += len(payload)
            if self.sync_on_write:
                os.fsync(fh.fileno())
            return offset

    def sync(self) -> None:
        """强制把页缓存刷到磁盘。断电安全的代价就在这里。"""
        with self._lock:
            self._ensure_open()
            if self._fh is None:
                return          # 一个字节都没写过,没什么可刷的
            self._fh.flush()
            os.fsync(self._fh.fileno())

    # ------------------------------------------------------------ 读取

    def replay(self, verify_crc: bool = True) -> ReplayResult:
        """重放整个日志。

        遇到损坏时**不抛异常**,而是返回已读到的完好记录,
        并在结果里标记 ``truncated``。调用方据此决定是否截断文件。
        """
        self._ensure_open()
        result = ReplayResult()

        # 文件不存在 == 空日志(还没写过任何东西)。这不是错误,
        # 所以直接返回空结果,而不是抛 FileNotFoundError。
        if not self.path.exists():
            return result

        with open(self.path, "rb") as fh:
            try:
                for rec in iter_records(fh, verify_crc=verify_crc):
                    result.records.append(rec)
                    result.valid_bytes = rec.offset + rec.size
            except CorruptionError as exc:
                result.truncated = True
                result.reason = str(exc)

        return result

    def recover(self, verify_crc: bool = True) -> ReplayResult:
        """重放日志,并**自动截断**末尾的残骸。

        这是引擎启动时应该调用的方法:它保证函数返回后,
        日志文件里的每一个字节都是可解析的有效数据。
        """
        result = self.replay(verify_crc=verify_crc)

        if result.truncated:
            actual_size = self.path.stat().st_size
            # 只有确实多出字节才需要截断
            if result.valid_bytes < actual_size:
                self._truncate(result.valid_bytes)

        return result

    def _truncate(self, size: int) -> None:
        """把文件截断到指定长度。

        截断后**不**立刻重开写句柄 —— 交给 ``_ensure_writer`` 在下次
        写入时按需打开,避免"只恢复不写入"的场景多占一个句柄。
        """
        with self._lock:
            if self._closed:
                return
            if self._fh is not None:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
                self._fh = None
            with open(self.path, "r+b") as fh:
                fh.truncate(size)
                fh.flush()
                os.fsync(fh.fileno())
            self._size = size

    def truncate(self) -> None:
        """清空日志,从头开始写。

        ⚠️ **只在数据已经落到别处之后调用** —— 比如内存表刚刷成 SSTable。
        日志是唯一的恢复来源,先清日志再落盘就等于丢数据。

        截断后写句柄是关闭状态,下次 ``append`` 会按需重开。
        """
        self._ensure_open()
        if not self.path.exists():
            self._size = 0
            return
        self._truncate(0)

    def rewrite(self, items: Iterable[tuple[bytes, bytes | None]]) -> int:
        """用给定的键值对**整体替换**日志内容,返回重写后的字节数。

        它解决的是"刷盘之后日志里留下一段已经进了 SSTable 的记录"这个问题。
        为什么不能简单截断:并发写入时,内存表已经换成了新的那一张,
        而日志的前半段属于**刚刚被刷走**的那张表。截断会把新表的数据一起丢掉。

        于是这里改成"重写":把**当前内存表**的内容原样写进新日志。
        顺带得到一个好处 —— **日志被去重了**。内存表里每个键只有一条记录,
        而原日志里同一个键可能被追加过很多次,所以重写后通常小得多。

        崩溃安全(和 SSTable、manifest 用同一套办法):
            先写 ``wal.log.tmp`` 并 fsync,再 ``os.replace`` 原子改名。
            改名**之前**崩溃:旧日志(更长、含多余记录)仍然完好,
            重放它是幂等的 —— 那些记录的值和 SSTable 里的一致。
            改名**之后**崩溃:新日志恰好等于内存表内容。
            两种情况下数据都不会丢。

        ⚠️ 调用方必须保证重写期间**没有别的线程在 append** ——
        否则那条追加会落在被替换掉的旧文件里,凭空消失。
        引擎侧靠 ``_append_lock`` 保证这一点。
        """
        with self._lock:
            self._ensure_open()

            tmp = self.path.with_name(self.path.name + ".tmp")
            size = 0
            with open(tmp, "wb") as fh:
                for key, value in items:
                    if value is None:
                        payload = encode_record(RecordType.DELETE, key)
                    else:
                        payload = encode_record(RecordType.PUT, key, value)
                    fh.write(payload)
                    size += len(payload)
                fh.flush()
                os.fsync(fh.fileno())

            # Windows 上目标文件被打开时 ``os.replace`` 会失败,必须先关掉。
            # 而且这个句柄指向的是**被替换掉的旧文件**,留着也写不进去了。
            if self._fh is not None:
                self._fh.close()
                self._fh = None

            os.replace(tmp, self.path)
            self._size = size
            return size

    # ------------------------------------------------------------ 生命周期

    def close(self) -> None:
        """关闭日志(会先 fsync,确保已写数据落盘)。"""
        with self._lock:
            if self._closed:
                return
            try:
                if self._fh is not None:
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
            finally:
                if self._fh is not None:
                    self._fh.close()
                    self._fh = None
                self._closed = True

    def __enter__(self) -> "WAL":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_records(path: str | Path, verify_crc: bool = True) -> ReplayResult:
    """只读地解析一个 WAL 文件,不修改它。

    用于诊断工具:想看看日志里到底有什么、坏在哪。
    """
    path = Path(path)
    result = ReplayResult()

    if not path.exists():
        return result

    with open(path, "rb") as fh:
        try:
            for rec in iter_records(fh, verify_crc=verify_crc):
                result.records.append(rec)
                result.valid_bytes = rec.offset + rec.size
        except CorruptionError as exc:
            result.truncated = True
            result.reason = str(exc)

    return result
