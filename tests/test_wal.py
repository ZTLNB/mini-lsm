"""WAL 测试:写入、重放、崩溃截断恢复。"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.errors import LSMError  # noqa: E402
from mini_lsm.record import RecordType, encode_record  # noqa: E402
from mini_lsm.wal import WAL, read_records  # noqa: E402


class WALTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "wal.log"

    def tearDown(self):
        self._tmp.cleanup()


class TestAppend(WALTestCase):
    def test_append_returns_offset(self):
        with WAL(self.path) as wal:
            first = wal.append(RecordType.PUT, b"k1", b"v1")
            second = wal.append(RecordType.PUT, b"k2", b"v2")

            self.assertEqual(first, 0)
            # 第二条的偏移 == 第一条占用的字节数
            first_size = len(encode_record(RecordType.PUT, b"k1", b"v1"))
            self.assertEqual(second, first_size)
            # 且必须与文件当前长度吻合
            self.assertEqual(second, os.path.getsize(self.path) - first_size)

    def test_size_tracks_writes(self):
        with WAL(self.path) as wal:
            self.assertEqual(wal.size, 0)
            wal.append(RecordType.PUT, b"k", b"v")
            self.assertGreater(wal.size, 0)
            self.assertEqual(wal.size, os.path.getsize(self.path))

    def test_writes_persist_after_close(self):
        with WAL(self.path) as wal:
            wal.append(RecordType.PUT, b"persist", b"yes")
        result = read_records(self.path)
        self.assertEqual(result.record_count, 1)
        self.assertEqual(result.records[0].value, b"yes")

    def test_many_records(self):
        with WAL(self.path) as wal:
            for i in range(500):
                wal.append(RecordType.PUT, f"k{i}".encode(), f"v{i}".encode())
        result = read_records(self.path)
        self.assertEqual(result.record_count, 500)
        self.assertFalse(result.truncated)
        self.assertEqual(result.records[499].key, b"k499")


class TestReplay(WALTestCase):
    def test_replay_preserves_order(self):
        with WAL(self.path) as wal:
            for key in (b"a", b"b", b"c"):
                wal.append(RecordType.PUT, key, key.upper())
        result = WAL(self.path).replay()
        self.assertEqual([r.key for r in result.records], [b"a", b"b", b"c"])

    def test_replay_empty_file(self):
        WAL(self.path).close()
        result = read_records(self.path)
        self.assertEqual(result.record_count, 0)
        self.assertFalse(result.truncated)

    def test_replay_missing_file(self):
        result = read_records(self.dir / "不存在.log")
        self.assertEqual(result.record_count, 0)

    def test_valid_bytes_matches_file_size(self):
        with WAL(self.path) as wal:
            wal.append(RecordType.PUT, b"k", b"v")
        result = read_records(self.path)
        self.assertEqual(result.valid_bytes, os.path.getsize(self.path))


class TestCrashRecovery(WALTestCase):
    def _write_then_truncate(self, tail_bytes: int) -> None:
        """写入 3 条记录,然后把文件尾部砍掉若干字节,模拟写入中途崩溃。"""
        with WAL(self.path) as wal:
            wal.append(RecordType.PUT, b"k1", b"v1")
            wal.append(RecordType.PUT, b"k2", b"v2")
            wal.append(RecordType.PUT, b"k3", b"v3")
        size = os.path.getsize(self.path)
        with open(self.path, "r+b") as fh:
            fh.truncate(size - tail_bytes)

    def test_partial_last_record_detected(self):
        self._write_then_truncate(5)
        result = read_records(self.path)
        self.assertTrue(result.truncated)
        self.assertEqual(result.record_count, 2)
        self.assertIsNotNone(result.reason)

    def test_recover_truncates_file(self):
        """recover() 应该把文件真的截断到有效边界。"""
        self._write_then_truncate(5)
        before = os.path.getsize(self.path)

        with WAL(self.path) as wal:
            result = wal.recover()

        self.assertTrue(result.truncated)
        self.assertEqual(result.record_count, 2)
        after = os.path.getsize(self.path)
        self.assertLess(after, before)
        self.assertEqual(after, result.valid_bytes)

    def test_recovered_file_is_clean(self):
        """截断之后,再读一次应该完全没有损坏。"""
        self._write_then_truncate(5)
        with WAL(self.path) as wal:
            wal.recover()

        clean = read_records(self.path)
        self.assertFalse(clean.truncated)
        self.assertEqual(clean.record_count, 2)

    def test_can_append_after_recovery(self):
        """截断恢复后必须还能继续写 —— 否则引擎没法继续工作。"""
        self._write_then_truncate(5)
        with WAL(self.path) as wal:
            wal.recover()
            wal.append(RecordType.PUT, b"after", b"recovery")

        result = read_records(self.path)
        self.assertFalse(result.truncated)
        self.assertEqual(result.record_count, 3)
        self.assertEqual(result.records[-1].key, b"after")

    def test_fully_intact_file_not_truncated(self):
        with WAL(self.path) as wal:
            wal.append(RecordType.PUT, b"k", b"v")
        before = os.path.getsize(self.path)

        with WAL(self.path) as wal:
            result = wal.recover()

        self.assertFalse(result.truncated)
        self.assertEqual(os.path.getsize(self.path), before)

    def test_corrupted_middle_discards_tail(self):
        """中间损坏时,应保留损坏点之前的所有记录。"""
        with WAL(self.path) as wal:
            wal.append(RecordType.PUT, b"good1", b"v")
            wal.append(RecordType.PUT, b"good2", b"v")
            wal.append(RecordType.PUT, b"bad", b"v")
            wal.append(RecordType.PUT, b"tail", b"v")

        # 篡改第 3 条记录 payload 里的一个字节
        with open(self.path, "r+b") as fh:
            data = bytearray(fh.read())
            offset = data.find(b"bad")
            data[offset] ^= 0xFF
            fh.seek(0)
            fh.write(data)

        result = read_records(self.path)
        self.assertTrue(result.truncated)
        self.assertEqual(result.record_count, 2)
        self.assertEqual(result.records[1].key, b"good2")


class TestLifecycle(WALTestCase):
    def test_closed_flag(self):
        wal = WAL(self.path)
        self.assertFalse(wal.closed)
        wal.close()
        self.assertTrue(wal.closed)

    def test_append_after_close_raises(self):
        wal = WAL(self.path)
        wal.close()
        with self.assertRaises(LSMError):
            wal.append(RecordType.PUT, b"k", b"v")

    def test_double_close_is_safe(self):
        wal = WAL(self.path)
        wal.close()
        wal.close()          # 不应抛异常

    def test_context_manager(self):
        with WAL(self.path) as wal:
            wal.append(RecordType.PUT, b"k", b"v")
        self.assertTrue(wal.closed)

    def test_creates_parent_dirs(self):
        nested = self.dir / "a" / "b" / "c" / "wal.log"
        with WAL(nested) as wal:
            wal.append(RecordType.PUT, b"k", b"v")
        self.assertTrue(nested.exists())

    def test_sync_does_not_raise(self):
        with WAL(self.path) as wal:
            wal.append(RecordType.PUT, b"k", b"v")
            wal.sync()

    def test_reopen_appends(self):
        with WAL(self.path) as wal:
            wal.append(RecordType.PUT, b"first", b"1")
        with WAL(self.path) as wal:
            wal.append(RecordType.PUT, b"second", b"2")

        result = read_records(self.path)
        self.assertEqual(result.record_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
