"""内存表测试。

重点覆盖两件事:
    1. **有序性** —— 这是 MemTable 存在的理由,阶段 2 刷盘时全靠它
    2. **墓碑语义** —— "键不存在" 和 "键被删了" 在表里的表示不同,
       但对查询方而言结果必须一致。这两者的区分是 LSM 正确性的核心。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.memtable import ENTRY_OVERHEAD, MemTable  # noqa: E402


class TestWriteRead(unittest.TestCase):
    def setUp(self):
        self.mt = MemTable()

    def test_put_then_get(self):
        self.mt.put(b"name", b"alice")
        self.assertEqual(self.mt.get(b"name"), b"alice")

    def test_get_missing_returns_none(self):
        self.assertIsNone(self.mt.get(b"nope"))

    def test_overwrite_replaces_value(self):
        self.mt.put(b"k", b"old")
        self.mt.put(b"k", b"new")
        self.assertEqual(self.mt.get(b"k"), b"new")
        self.assertEqual(len(self.mt), 1)

    def test_empty_value_is_stored_not_confused_with_missing(self):
        """空值不是墓碑 —— 这是最容易搞混的地方。"""
        self.mt.put(b"k", b"")
        self.assertEqual(self.mt.get(b"k"), b"")
        self.assertIsNotNone(self.mt.get(b"k"))
        self.assertEqual(self.mt.tombstone_count, 0)

    def test_binary_keys_and_values(self):
        key = bytes(range(256))
        value = b"\x00\xff" * 512
        self.mt.put(key, value)
        self.assertEqual(self.mt.get(key), value)

    def test_utf8_content(self):
        self.mt.put("姓名".encode(), "张三".encode())
        self.assertEqual(self.mt.get("姓名".encode()).decode(), "张三")

    def test_accepts_bytearray(self):
        self.mt.put(bytearray(b"k"), bytearray(b"v"))
        self.assertEqual(self.mt.get(b"k"), b"v")

    def test_rejects_non_bytes_key(self):
        with self.assertRaises(TypeError):
            self.mt.put("k", b"v")          # str 不行,必须显式编码

    def test_rejects_non_bytes_value(self):
        with self.assertRaises(TypeError):
            self.mt.put(b"k", "v")


class TestTombstone(unittest.TestCase):
    def setUp(self):
        self.mt = MemTable()

    def test_delete_makes_key_invisible(self):
        self.mt.put(b"k", b"v")
        self.mt.delete(b"k")
        self.assertIsNone(self.mt.get(b"k"))

    def test_delete_leaves_tombstone_in_table(self):
        """墓碑必须**留在表里** —— 否则更旧的版本会从 SSTable 里复活。"""
        self.mt.put(b"k", b"v")
        self.mt.delete(b"k")
        self.assertEqual(len(self.mt), 1)
        self.assertEqual(self.mt.tombstone_count, 1)
        self.assertIn(b"k", self.mt)

    def test_get_entry_distinguishes_missing_from_deleted(self):
        self.mt.put(b"alive", b"v")
        self.mt.put(b"dead", b"v")
        self.mt.delete(b"dead")

        self.assertEqual(self.mt.get_entry(b"alive"), (True, b"v"))
        self.assertEqual(self.mt.get_entry(b"dead"), (True, None))    # 墓碑
        self.assertEqual(self.mt.get_entry(b"never"), (False, None))  # 从未出现

    def test_delete_missing_key_still_creates_tombstone(self):
        """删除一个不存在的键也必须留下墓碑。

        因为可能有一个更旧的版本躺在 SSTable 里 —— 如果不写墓碑,
        那次删除就丢了,旧值会重新出现。
        """
        self.mt.delete(b"never-written")
        self.assertEqual(self.mt.get_entry(b"never-written"), (True, None))
        self.assertEqual(self.mt.tombstone_count, 1)

    def test_delete_then_put_resurrects_key(self):
        self.mt.put(b"k", b"v1")
        self.mt.delete(b"k")
        self.mt.put(b"k", b"v2")
        self.assertEqual(self.mt.get(b"k"), b"v2")
        self.assertEqual(self.mt.tombstone_count, 0)


class TestOrdering(unittest.TestCase):
    def setUp(self):
        self.mt = MemTable()
        for key in (b"c", b"a", b"d", b"b"):
            self.mt.put(key, key.upper())

    def test_items_sorted_ascending(self):
        self.assertEqual([k for k, _ in self.mt.items()],
                         [b"a", b"b", b"c", b"d"])

    def test_keys_sorted_ascending(self):
        self.assertEqual(list(self.mt.keys()), [b"a", b"b", b"c", b"d"])

    def test_items_includes_tombstones(self):
        """刷盘时要保留墓碑,所以 items() 不能跳过它们。"""
        self.mt.delete(b"b")
        keys = [k for k, _ in self.mt.items()]
        self.assertEqual(keys, [b"a", b"b", b"c", b"d"])
        self.assertIsNone(dict(self.mt.items())[b"b"])

    def test_live_items_skips_tombstones(self):
        self.mt.delete(b"b")
        self.assertEqual([k for k, _ in self.mt.live_items()],
                         [b"a", b"c", b"d"])

    def test_range_items_is_half_open(self):
        self.assertEqual([k for k, _ in self.mt.range_items(b"b", b"d")],
                         [b"b", b"c"])

    def test_range_items_full_span(self):
        self.assertEqual(len(list(self.mt.range_items(b"", b"\xff"))), 4)

    def test_range_items_empty_when_start_equals_end(self):
        self.assertEqual(list(self.mt.range_items(b"b", b"b")), [])


class TestSizeAccounting(unittest.TestCase):
    def test_new_key_adds_key_and_overhead(self):
        mt = MemTable()
        mt.put(b"k", b"v")
        self.assertEqual(mt.approximate_size, 1 + ENTRY_OVERHEAD + 1)

    def test_overwrite_swaps_value_size(self):
        mt = MemTable()
        mt.put(b"k", b"v")
        mt.put(b"k", b"vvvvv")            # 值从 1 字节变 5 字节
        self.assertEqual(mt.approximate_size, 1 + ENTRY_OVERHEAD + 5)

    def test_shrinking_value_reduces_size(self):
        mt = MemTable()
        mt.put(b"k", b"vvvvv")
        mt.put(b"k", b"v")
        self.assertEqual(mt.approximate_size, 1 + ENTRY_OVERHEAD + 1)

    def test_delete_releases_value_space_but_keeps_key(self):
        """墓碑不占 value 空间,但 key 本身还在表里,所以要算 key 的开销。"""
        mt = MemTable()
        mt.put(b"k", b"value")
        mt.delete(b"k")
        self.assertEqual(mt.approximate_size, 1 + ENTRY_OVERHEAD)

    def test_clear_resets_everything(self):
        mt = MemTable()
        mt.put(b"a", b"1")
        mt.put(b"b", b"2")
        mt.clear()
        self.assertEqual(len(mt), 0)
        self.assertEqual(mt.approximate_size, 0)
        self.assertEqual(mt.tombstone_count, 0)


class TestCapacity(unittest.TestCase):
    def test_not_full_when_small(self):
        mt = MemTable(capacity_bytes=10_000)
        mt.put(b"k", b"v")
        self.assertFalse(mt.is_full)

    def test_becomes_full_after_enough_writes(self):
        mt = MemTable(capacity_bytes=100)
        self.assertFalse(mt.is_full)
        mt.put(b"k1", b"v")
        mt.put(b"k2", b"v")
        self.assertTrue(mt.is_full)

    def test_default_capacity_is_four_mib(self):
        self.assertEqual(MemTable().capacity_bytes, 4 * 1024 * 1024)

    def test_rejects_zero_capacity(self):
        with self.assertRaises(ValueError):
            MemTable(capacity_bytes=0)

    def test_rejects_negative_capacity(self):
        with self.assertRaises(ValueError):
            MemTable(capacity_bytes=-1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
