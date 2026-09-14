"""阶段 2 引擎测试:SSTable 刷盘与多来源读路径。

阶段 2 要兑现两个承诺,测试主要围着它们转:
    1. **数据量不再受内存限制** —— 内存表满了就刷盘,刷完还能读回来
    2. **重启不用重放全部历史** —— 已经落成 SSTable 的部分不再走 WAL

还有一个容易写错、后果最严重的地方:
    **墓碑必须能压住更旧 SSTable 里的值**。压不住的话,被删掉的键
    会在重启后或刷盘后"复活" —— 这类 bug 在真实存储引擎里出过很多次。
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.engine import WAL_FILENAME, LSMEngine  # noqa: E402
from mini_lsm.errors import CorruptionError  # noqa: E402
from mini_lsm.sstable import SSTableReader  # noqa: E402
from mini_lsm.wal import read_records  # noqa: E402


class EngineSSTableTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def open_engine(self, **kwargs) -> LSMEngine:
        engine = LSMEngine(self.dir, **kwargs)
        self.addCleanup(engine.close)
        return engine

    @property
    def wal_path(self) -> Path:
        return self.dir / WAL_FILENAME

    def sst_files(self) -> list[str]:
        return sorted(p.name for p in self.dir.glob("*.sst"))


class TestFlush(EngineSSTableTestCase):
    def test_flush_writes_a_file(self):
        with self.open_engine() as db:
            db.put("a", "1")
            db.put("b", "2")
            written = db.flush()

            self.assertEqual(written, 2)
            self.assertEqual(self.sst_files(), ["sst-000001.sst"])
            self.assertEqual(db.stats().sstable_count, 1)

    def test_flush_empties_memtable(self):
        with self.open_engine() as db:
            db.put("a", "1")
            db.flush()
            stats = db.stats()
            self.assertEqual(stats.memtable_entries, 0)
            self.assertEqual(stats.tombstones, 0)

    def test_flush_truncates_wal(self):
        """数据已经在 SSTable 里了,WAL 不该再留着副本。"""
        with self.open_engine() as db:
            db.put("a", "1")
            self.assertGreater(db.stats().wal_size, 0)
            db.flush()
            self.assertEqual(db.stats().wal_size, 0)
            self.assertEqual(read_records(self.wal_path).record_count, 0)

    def test_flush_empty_memtable_is_noop(self):
        with self.open_engine() as db:
            self.assertEqual(db.flush(), 0)
            self.assertEqual(self.sst_files(), [])

    def test_flush_twice_without_writes(self):
        with self.open_engine() as db:
            db.put("a", "1")
            self.assertEqual(db.flush(), 1)
            self.assertEqual(db.flush(), 0)
            self.assertEqual(len(self.sst_files()), 1)

    def test_flush_increments_table_id(self):
        with self.open_engine() as db:
            db.put("a", "1")
            db.flush()
            db.put("b", "2")
            db.flush()
            self.assertEqual(self.sst_files(),
                             ["sst-000001.sst", "sst-000002.sst"])

    def test_flush_leaves_no_tmp_file(self):
        with self.open_engine() as db:
            db.put("a", "1")
            db.flush()
        self.assertEqual(list(self.dir.glob("*.tmp")), [])

    def test_flush_preserves_tombstones(self):
        with self.open_engine() as db:
            db.put("alive", "1")
            db.put("dead", "2")
            db.delete("dead")
            db.flush()

            with SSTableReader(self.dir / "sst-000001.sst") as reader:
                self.assertEqual(reader.get(b"alive"), (True, b"1"))
                self.assertEqual(reader.get(b"dead"), (True, None))
                self.assertEqual(reader.entry_count, 2)

    def test_put_then_delete_same_key_collapses(self):
        """同一代内存表里 put 完又 delete,只会留下一条墓碑。

        内存表是个 map,同 key 的旧版本会被直接覆盖 ——
        刷盘时也就不可能把已经作废的值写出去。
        """
        with self.open_engine() as db:
            db.put("k", "1")
            db.delete("k")
            db.flush()

            with SSTableReader(self.dir / "sst-000001.sst") as reader:
                self.assertEqual(reader.entry_count, 1)
                self.assertEqual(reader.get(b"k"), (True, None))

    def test_auto_flush_on_capacity(self):
        with self.open_engine(memtable_capacity=200) as db:
            for i in range(10):
                db.put(f"k{i}", "v" * 20)
            self.assertGreater(db.stats().sstable_count, 0)

    def test_table_id_continues_after_restart(self):
        with self.open_engine() as db:
            db.put("a", "1")
            db.flush()

        with self.open_engine() as db:
            db.put("b", "2")
            db.flush()

        self.assertEqual(self.sst_files(),
                         ["sst-000001.sst", "sst-000002.sst"])

    def test_stale_tmp_files_are_cleaned_on_startup(self):
        """崩溃留下的半截临时文件必须清掉,否则会越积越多。"""
        self.dir.mkdir(parents=True, exist_ok=True)
        stale = self.dir / "sst-000009.sst.tmp"
        stale.write_bytes("半截数据".encode("utf-8"))

        with self.open_engine():
            pass

        self.assertFalse(stale.exists())


class TestReadFromSSTable(EngineSSTableTestCase):
    def test_read_after_flush(self):
        with self.open_engine() as db:
            db.put("k", "v")
            db.flush()
            self.assertEqual(db.get_str("k"), "v")

    def test_read_after_flush_and_restart(self):
        with self.open_engine() as db:
            db.put("k", "v")
            db.flush()

        with self.open_engine() as db:
            self.assertEqual(db.get_str("k"), "v")
            self.assertEqual(db.stats().sstable_count, 1)

    def test_restart_does_not_replay_flushed_data(self):
        """阶段 2 的核心收益:刷过盘的数据不再走 WAL 重放。"""
        with self.open_engine() as db:
            for i in range(50):
                db.put(f"k{i:03d}", f"v{i}")
            db.flush()

        with self.open_engine() as db:
            self.assertEqual(db.get_str("k025"), "v25")
            self.assertEqual(db.stats().recovered_records, 0)
            self.assertEqual(db.stats().sstable_entries, 50)

    def test_only_unflushed_data_is_replayed(self):
        with self.open_engine() as db:
            db.put("flushed", "1")
            db.flush()
            db.put("in-wal", "2")

        with self.open_engine() as db:
            self.assertEqual(db.get_str("flushed"), "1")
            self.assertEqual(db.get_str("in-wal"), "2")
            self.assertEqual(db.stats().recovered_records, 1)

    def test_read_across_multiple_sstables(self):
        with self.open_engine() as db:
            for batch in range(5):
                for i in range(10):
                    db.put(f"b{batch}-k{i:02d}", f"v{batch}-{i}")
                db.flush()

            self.assertEqual(db.stats().sstable_count, 5)
            for batch in range(5):
                for i in range(10):
                    key = f"b{batch}-k{i:02d}"
                    self.assertEqual(db.get_str(key), f"v{batch}-{i}", key)

    def test_data_exceeds_memtable_capacity(self):
        """这条是阶段 2 存在的理由:数据量可以远超内存。"""
        with self.open_engine(memtable_capacity=4096) as db:
            for i in range(2000):
                db.put(f"key{i:05d}", "x" * 64)

            stats = db.stats()
            self.assertGreater(stats.sstable_count, 1)
            self.assertGreater(stats.sstable_bytes, 0)
            # 最早和最晚写的都要能读回来
            self.assertEqual(db.get_str("key00000"), "x" * 64)
            self.assertEqual(db.get_str("key01999"), "x" * 64)
            self.assertEqual(len(list(db.keys())), 2000)

    def test_binary_and_unicode_survive_flush(self):
        payload = bytes(range(256)) * 4
        with self.open_engine() as db:
            db.put("blob", payload)
            db.put("城市", "深圳")
            db.flush()

        with self.open_engine() as db:
            self.assertEqual(db.get("blob"), payload)
            self.assertEqual(db.get_str("城市"), "深圳")


class TestShadowing(EngineSSTableTestCase):
    """跨层遮蔽 —— 这是最不能出错的地方。"""

    def test_overwrite_in_newer_sstable_wins(self):
        with self.open_engine() as db:
            db.put("k", "old")
            db.flush()
            db.put("k", "new")
            db.flush()
            self.assertEqual(db.get_str("k"), "new")

        with self.open_engine() as db:
            self.assertEqual(db.get_str("k"), "new")

    def test_tombstone_shadows_value_in_older_sstable(self):
        """删掉的键绝不能从更旧的文件里复活。"""
        with self.open_engine() as db:
            db.put("k", "v1")
            db.flush()                  # v1 落到 sst-000001
            db.delete("k")
            db.flush()                  # 墓碑落到 sst-000002
            self.assertIsNone(db.get("k"))

        with self.open_engine() as db:
            self.assertEqual(db.stats().sstable_count, 2)
            self.assertIsNone(db.get("k"))
            self.assertFalse(db.contains("k"))
            self.assertEqual(list(db.keys()), [])

    def test_tombstone_in_memtable_shadows_sstable(self):
        with self.open_engine() as db:
            db.put("k", "v")
            db.flush()
            db.delete("k")
            self.assertIsNone(db.get("k"))      # 还没刷盘,墓碑在内存表里

    def test_reput_after_delete_across_sstables(self):
        with self.open_engine() as db:
            db.put("k", "v1")
            db.flush()
            db.delete("k")
            db.flush()
            db.put("k", "v2")
            self.assertEqual(db.get_str("k"), "v2")

        with self.open_engine() as db:
            self.assertEqual(db.get_str("k"), "v2")

    def test_memtable_beats_sstable(self):
        with self.open_engine() as db:
            db.put("k", "old")
            db.flush()
            db.put("k", "new")
            self.assertEqual(db.get_str("k"), "new")

    def test_delete_only_in_oldest_layer(self):
        """三层叠加:最新是空,中间是墓碑,最旧有值 → 查不到。"""
        with self.open_engine() as db:
            db.put("k", "v1")
            db.flush()
            db.delete("k")
            db.flush()
            db.put("other", "x")        # 让内存表非空
            db.flush()

            self.assertEqual(db.stats().sstable_count, 3)
            self.assertIsNone(db.get("k"))


class TestScanAcrossSources(EngineSSTableTestCase):
    def setUp(self):
        super().setUp()
        self.db = self.open_engine()
        # 三层:最旧的 SSTable、较新的 SSTable、内存表
        for key in (b"a", b"b", b"c"):
            self.db.put(key, b"old-" + key)
        self.db.flush()
        for key in (b"c", b"d"):
            self.db.put(key, b"mid-" + key)
        self.db.flush()
        self.db.put(b"e", b"new-e")

    def test_scan_merges_all_sources_sorted(self):
        self.assertEqual([k.decode() for k, _ in self.db.scan()],
                         ["a", "b", "c", "d", "e"])

    def test_scan_returns_newest_value(self):
        got = dict(self.db.scan())
        self.assertEqual(got[b"a"], b"old-a")
        self.assertEqual(got[b"c"], b"mid-c")     # 较新的层赢
        self.assertEqual(got[b"e"], b"new-e")     # 内存表赢

    def test_scan_range(self):
        self.assertEqual([k.decode() for k, _ in self.db.scan("b", "e")],
                         ["b", "c", "d"])

    def test_scan_skips_tombstones_across_sources(self):
        self.db.delete(b"c")
        self.assertEqual([k.decode() for k, _ in self.db.scan()],
                         ["a", "b", "d", "e"])

    def test_keys_matches_scan(self):
        self.assertEqual([k.decode() for k in self.db.keys()],
                         ["a", "b", "c", "d", "e"])

    def test_scan_after_flush_and_restart(self):
        self.db.delete(b"b")
        self.db.flush()

        with self.open_engine() as db:
            self.assertEqual([k.decode() for k, _ in db.scan()],
                             ["a", "c", "d", "e"])


class TestStartupRobustness(EngineSSTableTestCase):
    def test_corrupted_sstable_prevents_startup(self):
        """宁可拒绝启动,也不能带着损坏的数据文件跑起来 ——
        那会让"查不到"和"数据丢了"变得无法区分。"""
        with self.open_engine() as db:
            db.put("k", "v")
            db.flush()

        path = self.dir / "sst-000001.sst"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 0xFF                      # 破坏 footer magic
        path.write_bytes(bytes(raw))

        with self.assertRaises(CorruptionError):
            LSMEngine(self.dir)

    def test_ignores_unrelated_files(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "notes.txt").write_text("无关文件")
        (self.dir / "sst-abc.sst").write_bytes("名字不对".encode("utf-8"))

        with self.open_engine() as db:
            self.assertEqual(db.stats().sstable_count, 0)
            db.put("k", "v")
            self.assertEqual(db.get_str("k"), "v")

    def test_close_does_not_flush(self):
        """关引擎不该顺手刷盘 —— 否则每次关闭都留下一个很小的文件。

        内存表的数据由 WAL 保证,下次启动重放即可。
        """
        with self.open_engine() as db:
            db.put("k", "v")
        self.assertEqual(self.sst_files(), [])

        with self.open_engine() as db:
            self.assertEqual(db.get_str("k"), "v")
            self.assertEqual(db.stats().recovered_records, 1)

    def test_stats_report_sstables(self):
        with self.open_engine() as db:
            db.put("a", "1")
            db.put("b", "2")
            db.flush()
            stats = db.stats()
            self.assertEqual(stats.sstable_count, 1)
            self.assertEqual(stats.sstable_entries, 2)
            self.assertGreater(stats.sstable_bytes, 0)
            self.assertEqual(stats.flushes, 1)
            self.assertIn("SSTable", str(stats))

    def test_sstables_property_is_newest_first(self):
        with self.open_engine() as db:
            db.put("a", "1")
            db.flush()
            db.put("b", "2")
            db.flush()
            ids = [table.file_id for table in db.sstables]
            self.assertEqual(ids, [2, 1])

    def test_operations_after_close_raise(self):
        from mini_lsm.errors import ClosedError

        db = self.open_engine()
        db.close()
        with self.assertRaises(ClosedError):
            db.flush()


class TestCrashDuringFlush(EngineSSTableTestCase):
    def test_leftover_tmp_does_not_break_startup(self):
        """模拟"临时文件写完、还没改名"就崩溃。

        此时数据既在临时文件里、也在 WAL 里(WAL 还没被截断),
        所以丢掉临时文件不会丢数据。
        """
        with self.open_engine() as db:
            db.put("k", "v")
        # 手工造一个和真实数据无关的临时文件
        (self.dir / "sst-000001.sst.tmp").write_bytes(os.urandom(200))

        with self.open_engine() as db:
            self.assertEqual(db.get_str("k"), "v")
            self.assertEqual(db.stats().sstable_count, 0)
            self.assertFalse((self.dir / "sst-000001.sst.tmp").exists())

    def test_wal_survives_until_flush_completes(self):
        """刷盘之前 WAL 必须一直保留数据 —— 它是唯一的恢复来源。"""
        with self.open_engine() as db:
            db.put("k", "v")
            self.assertGreater(read_records(self.wal_path).record_count, 0)
            db.flush()
            self.assertEqual(read_records(self.wal_path).record_count, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
