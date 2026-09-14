"""mini-lsm 演示脚本。

跑一遍就能看到这条主线:

    内存表写满 → 刷成 SSTable → 文件攒多了自动归并 → 重启时只重放没刷盘的一段

归并(compaction)那一段是阶段 3 的重点,它要回答两个问题:
    1. 文件只增不减怎么办 —— 3000 条数据刷出几十个文件,查一次要翻几十个
    2. 删掉的键怎么才能真正消失 —— 墓碑不能一直留着

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

from mini_lsm import (  # noqa: E402
    MANIFEST_FILENAME,
    LSMEngine,
    RecordType,
    SSTableReader,
    WAL,
    read_records,
)


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def section(n: int, title: str) -> None:
    print(f"\n--- {n}. {title} ---")


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def layout(db: LSMEngine) -> str:
    """把分层布局印成一行,例如 L0=3个/12.4 KiB  L1=1个/30.1 KiB"""
    stats = db.stats()
    parts = [
        f"L{index}={files}个/{human(size)}"
        for index, (files, size) in enumerate(
            zip(stats.level_files, stats.level_bytes)
        )
        if files
    ]
    return "  ".join(parts) if parts else "(没有文件)"


def demo() -> None:
    root = Path(tempfile.mkdtemp(prefix="mini-lsm-demo-"))
    data_dir = root / "data"

    try:
        rule("mini-lsm 演示:WAL + 内存表 + SSTable + 分层归并 + 崩溃恢复")

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
            print(f"  结果: 刷盘 {stats.flushes} 次,归并 {stats.compactions} 次")
            print(f"        当前 {stats.sstable_count} 个 SSTable,"
                  f" 共 {human(stats.sstable_bytes)}")
            print(f"        分层: {layout(db)}")
            print(f"        当前内存表只有 {stats.memtable_entries} 条,"
                  f" 占用 {human(stats.memtable_size)}")
            print(f"  → 查一个 key:L0 试 {stats.level_files[0]} 个,"
                  f" L1 二分定位到 1 个,一共约 "
                  f"{stats.level_files[0] + 1} 次文件访问")
            print(f"  最早写入的 key00000 = {db.get_str('key00000')[:20]}...")
            print(f"  最晚写入的 key02999 = {db.get_str('key02999')[:20]}...")
            print(f"  存活键总数: {len(list(db.keys()))}")

        # ---------------------------------------------------------- 4
        section(6, "归并省掉了什么:关掉自动归并对同一个负载")
        nocompact_dir = root / "no-compact"
        with LSMEngine(
            nocompact_dir, memtable_capacity=capacity, auto_compact=False
        ) as db:
            for i in range(3000):
                db.put(f"key{i:05d}", "x" * 64)
            stats = db.stats()
            print(f"  同样 3000 条,但不自动归并: "
                  f"{stats.sstable_count} 个文件,归并 {stats.compactions} 次")
            print(f"    分层: {layout(db)}")
            print(f"    → 查询一个 key,最坏要把 L0 的 {stats.level_files[0]} "
                  f"个文件全翻一遍")
            print("    这就是阶段 3 要解决的问题:L0 的文件键范围互相重叠,")
            print("    查一次的成本随文件数**线性增长**。")

        # ---------------------------------------------------------- 5
        section(7, "手动归并:把 L0 压到 L1")
        manual_dir = root / "manual"
        # 内存表容量给足,让每批 200 条正好一次刷盘 —— 这样"6 批 → 6 个文件"最直观
        # (内存表按 key + 64 字节开销 + value 计费,200 条大约 15 KiB)
        with LSMEngine(
            manual_dir, memtable_capacity=64 * 1024, auto_compact=False
        ) as db:
            for batch in range(6):
                for i in range(200):
                    db.put(f"b{batch}-k{i:03d}", f"v{batch}-{i}")
                db.flush()

            print(f"  刷盘 6 次: {db.stats().sstable_count} 个文件   {layout(db)}")
            did = db.compact()
            print(f"  compact() 一轮:{'做了' if did else '没做'}")
            print(f"  归并后: {db.stats().sstable_count} 个文件   {layout(db)}")
            print("    注意 L0 被清空、数据落到 L1 —— 而 L1 内部是**不重叠**的,")
            print("    所以查一个 key 只需要二分定位到一个文件,不用再挨个试。")

            moved = db.compact_all()
            print(f"  再 compact_all():输入 {moved} 个文件,"
                  f" 剩 {db.stats().sstable_count} 个   {layout(db)}")
            print(f"  数据完整性抽查: "
                  f"{db.get_str('b0-k000')} / {db.get_str('b5-k199')}")

        # ---------------------------------------------------------- 6
        section(8, "墓碑的归宿:压到最底层才真正消失")
        tomb_dir = root / "tombstone"
        with LSMEngine(tomb_dir, auto_compact=False) as db:
            for key, value in (("a", "1"), ("b", "2"), ("c", "3")):
                db.put(key, value)
            db.flush()
            print(f"  写入 a、b、c:磁盘上 {db.stats().sstable_entries} 条")

            db.delete("b")
            db.flush()
            print(f"  删除 b:磁盘上 {db.stats().sstable_entries} 条"
                  f"  ← 多了 1 个墓碑")

            db.compact_all()
            after = db.stats()
            print(f"  压到最底层后:磁盘上 {after.sstable_entries} 条")
            print(f"    少掉的 2 条 = 墓碑本身 + 被它压住的 b 的旧值")
            print(f"    存活键: {[k.decode() for k in db.keys()]}")
            print(f"  为什么现在才敢丢墓碑?")
            print("    墓碑的作用是压住**更旧**的数据。只有压到最底层,")
            print("    '不可能再有更旧的数据'才成立 —— 在那之前丢掉墓碑,")
            print("    被删的键会从更深的层里复活。")

        with LSMEngine(tomb_dir) as db:
            print(f"  重启后 b = {db.get('b')}  (依然是删除状态)")

        # ---------------------------------------------------------- 7
        section(9, "Manifest:崩溃时'该信哪一套文件'")
        manifest_dir = root / "manifest"
        with LSMEngine(manifest_dir, auto_compact=False) as db:
            for batch in range(3):
                db.put(f"k{batch}", f"v{batch}")
                db.flush()
            print(f"  目录里的文件: {sorted(p.name for p in manifest_dir.glob('*.sst'))}")
            print(f"  manifest 记录的有效集合: "
                  f"{sorted(db.manifest.file_ids())}")
            print(f"  两者一致 —— manifest 就是'当前有效文件'的唯一依据")

        # 模拟崩溃残留:临时文件 + 孤儿文件
        (manifest_dir / "sst-999999.sst.tmp").write_bytes(b"half-written garbage")
        (manifest_dir / (MANIFEST_FILENAME + ".tmp")).write_bytes(b"{broken json")
        print("\n  人为留下崩溃残骸:")
        print("    半截数据文件 sst-999999.sst.tmp")
        print(f"    半截 manifest  {MANIFEST_FILENAME}.tmp")

        with LSMEngine(manifest_dir) as db:
            leftovers = sorted(
                p.name for p in manifest_dir.glob("*.tmp")
            )
            print(f"  重新打开后残骸: {leftovers or '(已全部清理)'}")
            print(f"  数据仍然完好: "
                  f"{[db.get_str(f'k{i}') for i in range(3)]}")

        # ---------------------------------------------------------- 8
        section(10, "崩溃恢复:WAL 尾部残缺")
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

        section(11, "恢复之后继续写入")
        with LSMEngine(crash_dir) as db:
            db.put("user:04", "dave")
            db.put("user:05", "erin")
            db.flush()
            print(f"  全部数据: {[(k.decode(), v.decode()) for k, v in db.scan()]}")

        result = read_records(crash_wal)
        print(f"  刷盘后 WAL 里剩 {result.record_count} 条,"
              f" 损坏标记 = {result.truncated}")

        rule("数据不丢;半截记录被丢弃;归并让文件数不随写入量线性增长;"
             "墓碑在压到最底层后被真正清掉")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    demo()
