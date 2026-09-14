"""mini-lsm 演示脚本。

跑一遍就能看到这条主线:
    内存表写满 → 刷成 SSTable → 重启时只重放"还没刷盘的那一段" → 数据不丢

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

from mini_lsm import LSMEngine, RecordType, SSTableReader, WAL, read_records  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def section(n: int, title: str) -> None:
    print(f"\n--- {n}. {title} ---")


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def demo() -> None:
    root = Path(tempfile.mkdtemp(prefix="mini-lsm-demo-"))
    data_dir = root / "data"
    wal_path = data_dir / "wal.log"

    try:
        rule("mini-lsm 演示:WAL + 内存表 + SSTable + 崩溃恢复")

        # ---------------------------------------------------------- 1
        section(1, "基本读写")
        with LSMEngine(data_dir) as db:
            db.put("name", "alice")
            db.put("lang", "python")
            db.put("city", "深圳")
            print(f"  name = {db.get_str('name')}")
            print(f"  city = {db.get_str('city')}")

            db.delete("city")
            print(f"  delete city 之后: {db.get('city')}")
            print(f"  内存表: {db.stats().memtable_entries} 条"
                  f"(其中墓碑 {db.stats().tombstones} 个)")

        section(2, "范围扫描(有序性带来的能力)")
        with LSMEngine(data_dir) as db:
            for i in range(5):
                db.put(f"user:{i:02d}", f"用户{i}")
            pairs = [(k.decode(), v.decode()) for k, v in db.scan("user:01", "user:04")]
            print(f"  scan('user:01', 'user:04') -> {pairs}")

        # ---------------------------------------------------------- 2
        section(3, "刷盘:内存表 → SSTable")
        with LSMEngine(data_dir) as db:
            before = db.stats()
            print(f"  刷盘前: 内存表 {before.memtable_entries} 条,"
                  f" WAL {before.wal_size} 字节,"
                  f" SSTable {before.sstable_count} 个")

            written = db.flush()

            after = db.stats()
            print(f"  flush() 写入 {written} 条")
            print(f"  刷盘后: 内存表 {after.memtable_entries} 条,"
                  f" WAL {after.wal_size} 字节,"
                  f" SSTable {after.sstable_count} 个")

            path = data_dir / "sst-000001.sst"
            with SSTableReader(path) as reader:
                print(f"\n  生成的文件 {path.name}: {human(reader.size)}")
                print(f"    {reader.entry_count} 条记录,"
                      f" 切成 {reader.block_count} 个数据块")
                print(f"    key 范围: {reader.first_key!r} .. {reader.last_key!r}")

        section(4, "重启:不再重放已经落盘的数据")
        with LSMEngine(data_dir) as db:
            stats = db.stats()
            print(f"  name = {db.get_str('name')}")
            print(f"  city = {db.get('city')}  (应仍为 None,墓碑被保留)")
            print(f"  启动恢复:重放 {stats.recovered_records} 条记录"
                  f"  ← 刷过盘的不走 WAL")
            print(f"  数据来自 {stats.sstable_count} 个 SSTable"
                  f"({stats.sstable_entries} 条)")

        # ---------------------------------------------------------- 3
        section(5, "数据量超过内存容量")
        small_dir = root / "small-mem"
        capacity = 8 * 1024
        with LSMEngine(small_dir, memtable_capacity=capacity) as db:
            print(f"  内存表容量只有 {human(capacity)},写入 3000 条 64 字节的记录")
            for i in range(3000):
                db.put(f"key{i:05d}", "x" * 64)

            stats = db.stats()
            print(f"  结果: 自动刷出 {stats.sstable_count} 个 SSTable,"
                  f" 共 {human(stats.sstable_bytes)}")
            print(f"        当前内存表只有 {stats.memtable_entries} 条,"
                  f" 占用 {human(stats.memtable_size)}")
            print(f"  最早写入的 key00000 = {db.get_str('key00000')[:20]}...")
            print(f"  最晚写入的 key02999 = {db.get_str('key02999')[:20]}...")
            print(f"  存活键总数: {len(list(db.keys()))}")

        # ---------------------------------------------------------- 4
        section(6, "墓碑跨层遮蔽:删掉的键不会复活")
        shadow_dir = root / "shadow"
        with LSMEngine(shadow_dir) as db:
            db.put("victim", "第一层的值")
            db.flush()
            print(f"  第 1 层写入 victim = {db.get_str('victim')}")

            db.delete("victim")
            db.flush()
            print(f"  第 2 层写入墓碑,再查 victim = {db.get('victim')}")

            db.put("other", "占位")
            db.flush()
            print(f"  第 3 层(不含 victim),再查 = {db.get('victim')}")

        with LSMEngine(shadow_dir) as db:
            print(f"  重启之后: victim = {db.get('victim')}"
                  f"   ← 没有被更旧的文件'复活'")
            print(f"  三个 SSTable 层叠: "
                  f"{[t.file_id for t in db.sstables]}")

        # ---------------------------------------------------------- 5
        section(7, "崩溃恢复:WAL 尾部残缺")
        crash_dir = root / "crash"
        crash_dir.mkdir(parents=True)
        crash_wal = crash_dir / "wal.log"

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

        section(8, "恢复之后继续写入")
        with LSMEngine(crash_dir) as db:
            db.put("user:04", "dave")
            db.put("user:05", "erin")
            db.flush()
            print(f"  全部数据: {[(k.decode(), v.decode()) for k, v in db.scan()]}")

        result = read_records(crash_wal)
        print(f"  刷盘后 WAL 里剩 {result.record_count} 条,"
              f"损坏标记 = {result.truncated}")

        rule("数据在崩溃后没有丢失;半截记录被正确丢弃;刷盘后重启不再重放历史")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    demo()
