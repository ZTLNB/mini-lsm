"""mini-lsm 阶段 1 演示脚本。

跑一遍就能看到三件事:
    1. 基本读写与范围扫描
    2. 正常重启后数据仍在
    3. 进程"写到一半被杀"之后,引擎仍能启动并保住完好的数据

用法::

    python demo.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mini_lsm import LSMEngine, RecordType, WAL, read_records  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 62}\n{title}\n{'=' * 62}")


def section(n: int, title: str) -> None:
    print(f"\n--- {n}. {title} ---")


def demo() -> None:
    root = Path(tempfile.mkdtemp(prefix="mini-lsm-demo-"))
    data_dir = root / "data"
    wal_path = data_dir / "wal.log"

    try:
        # ---------------------------------------------------------- 1
        rule("阶段 1 演示:WAL + MemTable + 崩溃恢复")

        section(1, "基本读写")
        with LSMEngine(data_dir) as db:
            db.put("name", "alice")
            db.put("lang", "python")
            db.put("city", "深圳")
            print(f"  name = {db.get_str('name')}")
            print(f"  city = {db.get_str('city')}")

            db.delete("city")
            print(f"  delete city 之后: {db.get('city')}")
            print(f"  WAL 里已写入 {read_records(wal_path).record_count} 条记录")
            print(f"  内存表状态: {db.stats().memtable_entries} 条"
                  f"(其中墓碑 {db.stats().tombstones} 个)")

        section(2, "范围扫描(有序性带来的能力)")
        with LSMEngine(data_dir) as db:
            for i in range(5):
                db.put(f"user:{i:02d}", f"用户{i}")
            pairs = [(k.decode(), v.decode()) for k, v in db.scan("user:01", "user:04")]
            print(f"  scan('user:01', 'user:04') -> {pairs}")

        section(3, "正常重启")
        with LSMEngine(data_dir) as db:
            print(f"  name = {db.get_str('name')}")
            print(f"  city = {db.get('city')}  (应仍为 None,墓碑被保留)")
            stats = db.stats()
            print(f"  启动恢复:重放 {stats.recovered_records} 条记录")

        # ---------------------------------------------------------- 2
        section(4, "模拟写入中途崩溃(WAL 尾部残缺)")

        crash_dir = root / "crash"
        crash_dir.mkdir(parents=True)
        crash_wal = crash_dir / "wal.log"

        # 用 WAL 直接造一个"崩溃现场":三条记录,最后一条只写了一半
        with WAL(crash_wal) as wal:
            wal.append(RecordType.PUT, b"user:01", b"alice")
            wal.append(RecordType.PUT, b"user:02", b"bob")
            wal.append(RecordType.PUT, b"user:03", b"carol")

        full = os.path.getsize(crash_wal)
        with open(crash_wal, "r+b") as fh:
            fh.truncate(full - 7)          # 砍掉最后 7 字节 = 半条记录
        print(f"  WAL 从 {full} 字节被砍到 {os.path.getsize(crash_wal)} 字节")

        with LSMEngine(crash_dir) as db:
            stats = db.stats()
            print(f"  user:01 = {db.get_str('user:01')}")
            print(f"  user:02 = {db.get_str('user:02')}")
            print(f"  user:03 = {db.get_str('user:03')}  (半截记录,被丢弃)")
            print(f"  启动恢复:重放 {stats.recovered_records} 条记录"
                  f"{'(含截断)' if stats.recovery_truncated else ''}")
            print(f"  截断原因: {stats.recovery_reason}")
            print(f"  修复后 WAL 大小: {os.path.getsize(crash_wal)} 字节")

        section(5, "恢复之后继续写入")
        with LSMEngine(crash_dir) as db:
            db.put("user:04", "dave")
            db.put("user:05", "erin")

        result = read_records(crash_wal)
        print(f"  现在 WAL 里有 {result.record_count} 条记录,"
              f"损坏标记 = {result.truncated}")
        print(f"  键: {[r.key.decode() for r in result.records]}")

        with LSMEngine(crash_dir) as db:
            print(f"  全部数据: {[(k.decode(), v.decode()) for k, v in db.scan()]}")

        rule("阶段 1 完成:数据在崩溃后没有丢失,半截记录被正确丢弃")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    demo()
