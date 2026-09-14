"""存储引擎测试。

这是整个阶段 1 的验收测试。核心问题只有一个:
    **向调用方返回"写成功"之后,数据还会丢吗?**

所以重启恢复相关的用例写得最厚 —— 包括正常重启、崩溃残留、
以及恢复之后还能不能继续正常工作。
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.engine import WAL_FILENAME, LSMEngine, to_bytes  # noqa: E402
from mini_lsm.errors import ClosedError, InvalidArgumentError  # noqa: E402
from mini_lsm.record import RecordType  # noqa: E402
from mini_lsm.wal import WAL, read_records  # noqa: E402


class EngineTestCase(unittest.TestCase):
    def setUp(self):
        # 注意注册顺序:cleanup 是 LIFO 执行的,先注册的临时目录清理
        # 最后才跑 —— 这样引擎一定在目录被删之前关掉(Windows 上必须)。
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


class TestBasicCRUD(EngineTestCase):
    def test_put_and_get(self):
        with self.open_engine() as db:
            db.put("name", "alice")
            self.assertEqual(db.get(b"name"), b"alice")

    def test_get_str_decodes(self):
        with self.open_engine() as db:
            db.put("姓名", "张三")
            self.assertEqual(db.get_str("姓名"), "张三")

    def test_get_missing_returns_none(self):
        with self.open_engine() as db:
            self.assertIsNone(db.get("nope"))
            self.assertIsNone(db.get_str("nope"))

    def test_contains(self):
        with self.open_engine() as db:
            db.put("k", "v")
            self.assertTrue(db.contains("k"))
            self.assertFalse(db.contains("other"))

    def test_overwrite(self):
        with self.open_engine() as db:
            db.put("k", "v1")
            db.put("k", "v2")
            self.assertEqual(db.get_str("k"), "v2")
            self.assertEqual(db.stats().memtable_entries, 1)

    def test_delete_then_get_returns_none(self):
        with self.open_engine() as db:
            db.put("k", "v")
            db.delete("k")
            self.assertIsNone(db.get("k"))
            self.assertFalse(db.contains("k"))

    def test_delete_missing_key_is_not_an_error(self):
        with self.open_engine() as db:
            db.delete("never-existed")      # 不应抛异常

    def test_binary_value_with_nulls(self):
        payload = bytes(range(256))
        with self.open_engine() as db:
            db.put("blob", payload)
            self.assertEqual(db.get("blob"), payload)

    def test_put_many_returns_count(self):
        with self.open_engine() as db:
            n = db.put_many({"a": "1", "b": "2", "c": "3"})
            self.assertEqual(n, 3)
            self.assertEqual(db.get_str("b"), "2")

    def test_put_many_accepts_list_of_pairs(self):
        with self.open_engine() as db:
            n = db.put_many([("a", "1"), ("b", "2")])
            self.assertEqual(n, 2)


class TestScan(EngineTestCase):
    def setUp(self):
        super().setUp()
        self.db = self.open_engine()
        for key in "abcde":
            self.db.put(key, key.upper())

    def test_scan_all_sorted(self):
        self.assertEqual([k.decode() for k, _ in self.db.scan()],
                         list("abcde"))

    def test_scan_with_bounds_is_half_open(self):
        pairs = list(self.db.scan("b", "d"))
        self.assertEqual([k.decode() for k, _ in pairs], ["b", "c"])

    def test_scan_open_start(self):
        self.assertEqual([k.decode() for k, _ in self.db.scan(end="c")],
                         ["a", "b"])

    def test_scan_skips_tombstones(self):
        self.db.delete("c")
        self.assertEqual([k.decode() for k, _ in self.db.scan()],
                         ["a", "b", "d", "e"])

    def test_keys_matches_scan(self):
        self.assertEqual([k.decode() for k in self.db.keys()], list("abcde"))

    def test_scan_snapshot_is_stable_during_iteration(self):
        """迭代期间写入不应影响已经拿到的结果。"""
        it = self.db.scan()
        self.db.put("zzz", "later")
        self.assertEqual([k.decode() for k, _ in it], list("abcde"))


class TestWalOrdering(EngineTestCase):
    def test_put_is_visible_in_wal(self):
        """验证写入顺序:先落 WAL,再进内存表。"""
        with self.open_engine() as db:
            db.put("k", "v")
            result = read_records(self.wal_path)
            self.assertEqual(result.record_count, 1)
            self.assertEqual(result.records[0].key, b"k")
            self.assertEqual(result.records[0].value, b"v")

    def test_delete_writes_delete_record(self):
        with self.open_engine() as db:
            db.put("k", "v")
            db.delete("k")
            result = read_records(self.wal_path)
            self.assertEqual(result.record_count, 2)
            self.assertEqual(result.records[1].rec_type, RecordType.DELETE)

    def test_stats_reports_wal_size(self):
        with self.open_engine() as db:
            self.assertEqual(db.stats().wal_size, 0)
            db.put("k", "v")
            self.assertGreater(db.stats().wal_size, 0)


class TestRestartRecovery(EngineTestCase):
    def test_data_survives_graceful_restart(self):
        with self.open_engine() as db:
            db.put("name", "alice")
            db.put("lang", "python")

        with self.open_engine() as db:
            self.assertEqual(db.get_str("name"), "alice")
            self.assertEqual(db.get_str("lang"), "python")
            self.assertEqual(db.stats().recovered_records, 2)

    def test_overwrite_survives_restart(self):
        with self.open_engine() as db:
            db.put("k", "old")
            db.put("k", "new")

        with self.open_engine() as db:
            self.assertEqual(db.get_str("k"), "new")

    def test_delete_survives_restart(self):
        """墓碑必须跨重启保留,否则被删的键会"复活"。"""
        with self.open_engine() as db:
            db.put("k", "v")
            db.delete("k")

        with self.open_engine() as db:
            self.assertIsNone(db.get("k"))
            self.assertFalse(db.contains("k"))
            self.assertEqual(db.stats().tombstones, 1)

    def test_delete_of_missing_key_survives_restart(self):
        with self.open_engine() as db:
            db.delete("k")

        with self.open_engine() as db:
            self.assertIsNone(db.get("k"))

    def test_binary_data_survives_restart(self):
        payload = bytes(range(256)) * 8
        with self.open_engine() as db:
            db.put("blob", payload)

        with self.open_engine() as db:
            self.assertEqual(db.get("blob"), payload)

    def test_unicode_survives_restart(self):
        with self.open_engine() as db:
            db.put("城市", "深圳")
            db.put("emoji", "🚀")

        with self.open_engine() as db:
            self.assertEqual(db.get_str("城市"), "深圳")
            self.assertEqual(db.get_str("emoji"), "🚀")

    def test_writes_are_durable_without_close(self):
        """持久性不能依赖 close() —— 进程可能直接被 kill 掉。"""
        engine = LSMEngine(self.dir)
        engine.put("name", "alice")
        engine.put("lang", "python")
        # 刻意不调用 engine.close(),模拟进程被杀

        with self.open_engine() as db:
            self.assertEqual(db.get_str("name"), "alice")
            self.assertEqual(db.get_str("lang"), "python")
            self.assertEqual(db.stats().recovered_records, 2)

        engine.close()      # 仅用于释放句柄,断言已经做完

    def test_empty_engine_restarts_cleanly(self):
        with self.open_engine():
            pass

        with self.open_engine() as db:
            self.assertEqual(db.stats().recovered_records, 0)
            self.assertIsNone(db.get("anything"))


class TestCrashRecovery(EngineTestCase):
    """模拟"写到一半进程被杀"留下的残缺 WAL。"""

    def _make_crash_site(self, tail_bytes: int) -> None:
        """直接用 WAL 写两条记录,然后砍掉尾部若干字节。"""
        with WAL(self.wal_path) as wal:
            wal.append(RecordType.PUT, b"a", b"1")
            wal.append(RecordType.PUT, b"b", b"2")
        size = os.path.getsize(self.wal_path)
        with open(self.wal_path, "r+b") as fh:
            fh.truncate(size - tail_bytes)

    def test_engine_starts_despite_broken_tail(self):
        """有半截记录时引擎必须能起来 —— 不能把整个库判死刑。"""
        self._make_crash_site(tail_bytes=4)

        with self.open_engine() as db:
            self.assertEqual(db.get_str("a"), "1")      # 完好的记录保住了
            self.assertIsNone(db.get("b"))              # 半截的被丢弃
            self.assertTrue(db.stats().recovery_truncated)
            self.assertIsNotNone(db.stats().recovery_reason)

    def test_recovery_repairs_the_file(self):
        self._make_crash_site(tail_bytes=4)
        before = os.path.getsize(self.wal_path)

        with self.open_engine():
            pass

        self.assertLess(os.path.getsize(self.wal_path), before)
        clean = read_records(self.wal_path)
        self.assertFalse(clean.truncated)
        self.assertEqual(clean.record_count, 1)

    def test_engine_can_write_after_recovery(self):
        """恢复后必须还能继续写 —— 否则这个库就废了。"""
        self._make_crash_site(tail_bytes=4)

        with self.open_engine() as db:
            db.put("c", "3")

        result = read_records(self.wal_path)
        self.assertFalse(result.truncated)
        self.assertEqual([r.key for r in result.records], [b"a", b"c"])

        with self.open_engine() as db:
            self.assertEqual(db.get_str("a"), "1")
            self.assertEqual(db.get_str("c"), "3")

    def test_corrupted_middle_keeps_earlier_records(self):
        """中间损坏时,损坏点之前的数据必须全部保住。"""
        with WAL(self.wal_path) as wal:
            wal.append(RecordType.PUT, b"good1", b"1")
            wal.append(RecordType.PUT, b"good2", b"2")
            wal.append(RecordType.PUT, b"bad", b"3")
            wal.append(RecordType.PUT, b"tail", b"4")

        with open(self.wal_path, "r+b") as fh:
            data = bytearray(fh.read())
            data[data.find(b"bad")] ^= 0xFF          # 篡改第三条的 key
            fh.seek(0)
            fh.write(data)

        with self.open_engine() as db:
            self.assertEqual(db.get_str("good1"), "1")
            self.assertEqual(db.get_str("good2"), "2")
            self.assertIsNone(db.get("bad"))
            self.assertIsNone(db.get("tail"))        # 损坏点之后全部丢弃
            self.assertTrue(db.stats().recovery_truncated)


class TestLifecycle(EngineTestCase):
    def test_closed_flag(self):
        db = self.open_engine()
        self.assertFalse(db.closed)
        db.close()
        self.assertTrue(db.closed)

    def test_operations_after_close_raise(self):
        db = self.open_engine()
        db.close()
        for op in (lambda: db.get("k"), lambda: db.put("k", "v"),
                   lambda: db.delete("k"), lambda: db.sync()):
            with self.assertRaises(ClosedError):
                op()

    def test_double_close_is_safe(self):
        db = self.open_engine()
        db.close()
        db.close()

    def test_context_manager_closes(self):
        with self.open_engine() as db:
            db.put("k", "v")
        self.assertTrue(db.closed)

    def test_creates_data_dir(self):
        nested = self.dir / "deep" / "nested"
        with LSMEngine(nested) as db:
            db.put("k", "v")
        self.assertTrue((nested / WAL_FILENAME).exists())

    def test_repr_does_not_raise(self):
        with self.open_engine() as db:
            self.assertIn("LSMEngine", repr(db))


class TestStats(EngineTestCase):
    def test_fresh_engine_stats(self):
        with self.open_engine() as db:
            s = db.stats()
            self.assertEqual(s.memtable_entries, 0)
            self.assertEqual(s.tombstones, 0)
            self.assertEqual(s.recovered_records, 0)
            self.assertFalse(s.flush_pending)

    def test_counts_entries_and_tombstones(self):
        with self.open_engine() as db:
            db.put("a", "1")
            db.put("b", "2")
            db.delete("a")
            s = db.stats()
            self.assertEqual(s.memtable_entries, 2)     # 墓碑也算一条
            self.assertEqual(s.tombstones, 1)

    def test_flush_pending_when_memtable_full(self):
        with self.open_engine(memtable_capacity=100) as db:
            db.put("k1", "v")
            db.put("k2", "v")
            self.assertTrue(db.stats().flush_pending)

    def test_usage_ratio(self):
        with self.open_engine(memtable_capacity=1000) as db:
            db.put("k", "v")
            self.assertGreater(db.stats().memtable_usage_ratio, 0.0)
            self.assertLess(db.stats().memtable_usage_ratio, 1.0)

    def test_str_renders(self):
        with self.open_engine() as db:
            db.put("k", "v")
            text = str(db.stats())
            self.assertIn("内存表", text)
            self.assertIn("WAL", text)


class TestInputValidation(EngineTestCase):
    def test_rejects_int_key(self):
        with self.open_engine() as db:
            with self.assertRaises(InvalidArgumentError):
                db.put(123, "v")

    def test_rejects_none_value(self):
        with self.open_engine() as db:
            with self.assertRaises(InvalidArgumentError):
                db.put("k", None)

    def test_to_bytes_accepts_str(self):
        self.assertEqual(to_bytes("hi", "key"), b"hi")

    def test_to_bytes_accepts_memoryview(self):
        self.assertEqual(to_bytes(memoryview(b"hi"), "key"), b"hi")

    def test_to_bytes_error_message_names_the_argument(self):
        with self.assertRaises(InvalidArgumentError) as ctx:
            to_bytes(3.14, "value")
        self.assertIn("value", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
