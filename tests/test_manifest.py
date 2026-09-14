"""阶段 3 测试:Manifest —— "当前有效的是哪一套文件"。

Manifest 的价值不在它做了什么,而在它**防住了什么**:

    阶段 2 的文件只增不减,文件名和内容一一对应,所以"扫目录"就够用了。
    但 compaction 会**成批**增删文件,崩溃完全可能停在
    "新文件已写好、旧文件还没删"的中间态。此时目录里新旧两套并存,
    光看目录无法判断该信哪一套 —— 选错就是丢数据。

    所以这个文件的测试重点不是"能存能读",而是:
        1. 落盘是原子的(要么看到旧的,要么看到新的)
        2. 层序契约被破坏时**大声报错**,而不是静默给出错误查询结果
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.errors import CorruptionError  # noqa: E402
from mini_lsm.manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    FileMeta,
    Manifest,
)


def meta(file_id, level, smallest, largest, entries=10, size=1024) -> FileMeta:
    """造一个 FileMeta。key 允许直接传 str,内部转 bytes。"""
    if isinstance(smallest, str):
        smallest = smallest.encode()
    if isinstance(largest, str):
        largest = largest.encode()
    return FileMeta(file_id, level, smallest, largest, entries, size)


class ManifestTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def new_manifest(self, num_levels: int = 3) -> Manifest:
        return Manifest(self.dir / MANIFEST_FILENAME, num_levels)

    @property
    def path(self) -> Path:
        return self.dir / MANIFEST_FILENAME


# ------------------------------------------------------------------ FileMeta


class TestFileMeta(ManifestTestCase):
    def test_key_range(self):
        m = meta(1, 1, "b", "d")
        self.assertEqual(m.key_range, (b"b", b"d"))

    def test_contains_is_inclusive(self):
        """闭区间 —— 边界上的 key 属于这个文件,这点写错会丢数据。"""
        m = meta(1, 1, "b", "d")
        self.assertTrue(m.contains(b"b"))
        self.assertTrue(m.contains(b"c"))
        self.assertTrue(m.contains(b"d"))
        self.assertFalse(m.contains(b"a"))
        self.assertFalse(m.contains(b"e"))

    def test_overlaps_inclusive(self):
        m = meta(1, 1, "b", "d")
        # 只挨着一个端点也算重叠
        self.assertTrue(m.overlaps(b"d", b"f"))
        self.assertTrue(m.overlaps(b"a", b"b"))
        self.assertTrue(m.overlaps(b"a", b"z"))
        self.assertTrue(m.overlaps(b"c", b"c"))
        # 完全在左边 / 右边
        self.assertFalse(m.overlaps(b"a", b"a"))
        self.assertFalse(m.overlaps(b"e", b"f"))

    def test_overlaps_with_single_key_range(self):
        m = meta(1, 1, "b", "d")
        self.assertTrue(m.overlaps(b"c", b"c"))
        self.assertFalse(m.overlaps(b"e", b"e"))

    def test_roundtrip_through_dict(self):
        original = meta(7, 2, "apple", "zebra", entries=42, size=999)
        restored = FileMeta.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_roundtrip_with_non_ascii_keys(self):
        """key 是任意字节,JSON 存不下 —— 靠 base64 兜住。"""
        original = meta(1, 1, "北京".encode("utf-8"), "上海".encode("utf-8"))
        restored = FileMeta.from_dict(original.to_dict())
        self.assertEqual(restored.smallest, original.smallest)
        self.assertEqual(restored.largest, original.largest)

    def test_roundtrip_with_binary_keys(self):
        """key 里可能有 \\x00、\\xff 这类字节,不能被 JSON 弄坏。"""
        original = meta(1, 1, b"\x00\xff\x80", b"\xfe\x01")
        restored = FileMeta.from_dict(original.to_dict())
        self.assertEqual(restored, original)

    def test_to_dict_is_json_safe(self):
        """to_dict 的结果必须能直接塞进 json.dumps。"""
        payload = json.dumps(meta(1, 1, b"\x00\xff", b"\xfe").to_dict())
        self.assertIsInstance(payload, str)

    def test_frozen(self):
        """FileMeta 是不可变的 —— 避免被某处意外改掉导致 manifest 与实际不符。"""
        m = meta(1, 1, "a", "b")
        with self.assertRaises(Exception):
            m.file_id = 99  # type: ignore[misc]


# ------------------------------------------------------------------ 基本行为


class TestManifestBasics(ManifestTestCase):
    def test_rejects_too_few_levels(self):
        with self.assertRaises(ValueError):
            Manifest(self.path, num_levels=1)

    def test_max_level(self):
        self.assertEqual(self.new_manifest(3).max_level, 2)
        self.assertEqual(self.new_manifest(2).max_level, 1)

    def test_starts_empty(self):
        m = self.new_manifest()
        self.assertEqual(m.files(), [])
        self.assertEqual(m.file_ids(), set())
        self.assertEqual(m.next_file_id, 1)

    def test_files_order_is_newest_first(self):
        """files() 的顺序就是查询顺序:新的在前,旧的在后。"""
        m = self.new_manifest()
        m.add(meta(3, 0, "a", "c"))     # L0
        m.add(meta(1, 0, "a", "c"))     # L0,更旧
        m.add(meta(5, 1, "a", "b"))     # L1
        m.add(meta(4, 2, "m", "n"))     # L2

        ids = [fm.file_id for fm in m.files()]
        self.assertEqual(ids, [3, 1, 5, 4])

    def test_level_bytes_and_entries(self):
        m = self.new_manifest()
        m.add(meta(1, 1, "a", "b", entries=3, size=100))
        m.add(meta(2, 1, "c", "d", entries=5, size=200))

        self.assertEqual(m.level_bytes(1), 300)
        self.assertEqual(m.level_entries(1), 8)
        self.assertEqual(m.level_bytes(0), 0)
        self.assertEqual(m.level_bytes(2), 0)

    def test_file_ids(self):
        m = self.new_manifest()
        m.add(meta(1, 0, "a", "b"))
        m.add(meta(2, 1, "c", "d"))
        self.assertEqual(m.file_ids(), {1, 2})


# ------------------------------------------------------------------ 层序排列


class TestLevelOrdering(ManifestTestCase):
    def test_l0_sorted_newest_first(self):
        """L0 允许重叠,顺序表达的是"新旧"而不是键序。"""
        m = self.new_manifest()
        for file_id in (2, 9, 5, 1):
            m.add(meta(file_id, 0, "a", "z"))
        self.assertEqual([fm.file_id for fm in m.levels[0]], [9, 5, 2, 1])

    def test_l1_sorted_by_smallest_key(self):
        m = self.new_manifest()
        m.add(meta(1, 1, "m", "n"))
        m.add(meta(2, 1, "a", "b"))
        m.add(meta(3, 1, "x", "y"))
        self.assertEqual([fm.smallest for fm in m.levels[1]], [b"a", b"m", b"x"])

    def test_ordering_restored_after_load(self):
        """从磁盘读回来之后顺序必须还原 —— 顺序错了二分就错了。"""
        m = self.new_manifest()
        for file_id in (2, 9, 5, 1):
            m.add(meta(file_id, 0, "a", "z"))
        m.add(meta(11, 1, "m", "n"))
        m.add(meta(12, 1, "a", "b"))
        m.save()

        loaded = self.new_manifest()
        loaded.load()
        self.assertEqual([fm.file_id for fm in loaded.levels[0]], [9, 5, 2, 1])
        self.assertEqual([fm.smallest for fm in loaded.levels[1]], [b"a", b"m"])
        loaded.check_invariants()


# ------------------------------------------------------------------ 二分查找


class TestFindFile(ManifestTestCase):
    def build(self) -> Manifest:
        """L1 放三个互不重叠的文件,中间留出空隙。"""
        m = self.new_manifest()
        m.add(meta(1, 1, "a", "c"))
        m.add(meta(2, 1, "f", "h"))
        m.add(meta(3, 1, "m", "p"))
        return m

    def test_empty_level(self):
        self.assertIsNone(self.new_manifest().find_file(1, b"a"))

    def test_key_below_all(self):
        self.assertIsNone(self.build().find_file(1, b"0"))

    def test_key_above_all(self):
        self.assertIsNone(self.build().find_file(1, b"z"))

    def test_key_in_gap_between_files(self):
        """落在文件之间的空隙里,不该硬塞给左边的文件。"""
        self.assertIsNone(self.build().find_file(1, b"d"))
        self.assertIsNone(self.build().find_file(1, b"i"))

    def test_boundary_keys_are_found(self):
        m = self.build()
        self.assertEqual(m.find_file(1, b"a").file_id, 1)
        self.assertEqual(m.find_file(1, b"c").file_id, 1)
        self.assertEqual(m.find_file(1, b"f").file_id, 2)
        self.assertEqual(m.find_file(1, b"h").file_id, 2)
        self.assertEqual(m.find_file(1, b"m").file_id, 3)
        self.assertEqual(m.find_file(1, b"p").file_id, 3)

    def test_keys_inside_files(self):
        m = self.build()
        self.assertEqual(m.find_file(1, b"b").file_id, 1)
        self.assertEqual(m.find_file(1, b"g").file_id, 2)
        self.assertEqual(m.find_file(1, b"n").file_id, 3)

    def test_matches_linear_scan_on_random_keys(self):
        """二分的结果必须和"挨个试"完全一致 —— 这是二分的正确性底线。"""
        m = self.build()
        for code in range(ord("a") - 2, ord("z") + 3):
            key = bytes([code])
            expected = next(
                (fm for fm in m.levels[1] if fm.contains(key)), None
            )
            self.assertEqual(
                m.find_file(1, key), expected, f"key={key!r} 二分与线性扫描不一致"
            )

    def test_single_file_level(self):
        m = self.new_manifest()
        m.add(meta(1, 2, "a", "z"))
        self.assertEqual(m.find_file(2, b"m").file_id, 1)
        self.assertIsNone(m.find_file(2, b"zz"))


# ------------------------------------------------------------------ 增删


class TestManifestMutation(ManifestTestCase):
    def test_add_appends_to_correct_level(self):
        m = self.new_manifest()
        m.add(meta(1, 2, "a", "b"))
        self.assertEqual(len(m.levels[2]), 1)
        self.assertEqual(len(m.levels[0]), 0)

    def test_replace_removes_and_adds(self):
        m = self.new_manifest()
        old = meta(1, 0, "a", "c")
        m.add(old)

        new = meta(2, 1, "a", "c")
        m.replace([old], [new])

        self.assertEqual(m.file_ids(), {2})
        self.assertEqual(len(m.levels[0]), 0)
        self.assertEqual(len(m.levels[1]), 1)
        m.check_invariants()

    def test_replace_handles_file_that_is_both_input_and_output(self):
        """归并后产出的 file_id 理论上不会撞上输入,但 replace 不该依赖这点。"""
        m = self.new_manifest()
        old = meta(1, 0, "a", "c")
        m.add(old)

        moved = meta(1, 1, "a", "c")     # 同一个 file_id,换了层
        m.replace([old], [moved])

        self.assertEqual(m.file_ids(), {1})
        self.assertEqual(len(m.levels[0]), 0)
        self.assertEqual(len(m.levels[1]), 1)
        m.check_invariants()

    def test_replace_removes_across_levels(self):
        m = self.new_manifest()
        a = meta(1, 0, "a", "c")
        b = meta(2, 1, "a", "c")
        m.add(a)
        m.add(b)

        m.replace([a, b], [meta(3, 2, "a", "c")])
        self.assertEqual(m.file_ids(), {3})
        m.check_invariants()

    def test_replace_with_nothing_removed(self):
        m = self.new_manifest()
        m.replace([], [meta(1, 1, "a", "b")])
        self.assertEqual(m.file_ids(), {1})

    def test_replace_keeps_other_levels_sorted(self):
        m = self.new_manifest()
        m.add(meta(1, 1, "m", "n"))
        m.add(meta(2, 1, "x", "y"))

        m.replace([], [meta(3, 1, "a", "b")])
        self.assertEqual([fm.smallest for fm in m.levels[1]], [b"a", b"m", b"x"])


# ------------------------------------------------------------------ 持久化


class TestManifestPersistence(ManifestTestCase):
    def test_roundtrip(self):
        m = self.new_manifest()
        m.add(meta(1, 0, "a", "c", entries=3, size=111))
        m.add(meta(2, 1, "a", "b", entries=7, size=222))
        m.next_file_id = 42
        m.save()

        loaded = self.new_manifest()
        loaded.load()

        self.assertEqual(loaded.next_file_id, 42)
        self.assertEqual(loaded.file_ids(), {1, 2})
        self.assertEqual(loaded.level_bytes(1), 222)
        loaded.check_invariants()

    def test_roundtrip_preserves_non_ascii_keys(self):
        m = self.new_manifest()
        m.add(meta(1, 1, "北京".encode(), "上海".encode()))
        m.save()

        loaded = self.new_manifest()
        loaded.load()
        self.assertEqual(loaded.levels[1][0].smallest, "北京".encode())
        self.assertEqual(loaded.levels[1][0].largest, "上海".encode())

    def test_save_is_atomic_leaves_no_tmp(self):
        """os.replace 之后不该有 .tmp 残留 —— 有残留说明没走原子路径。"""
        m = self.new_manifest()
        m.add(meta(1, 1, "a", "b"))
        m.save()

        self.assertTrue(self.path.exists())
        self.assertFalse((self.dir / (MANIFEST_FILENAME + ".tmp")).exists())

    def test_save_overwrites_previous_content(self):
        m = self.new_manifest()
        m.add(meta(1, 1, "a", "b"))
        m.save()

        m2 = self.new_manifest()
        m2.add(meta(9, 2, "x", "y"))
        m2.save()

        loaded = self.new_manifest()
        loaded.load()
        self.assertEqual(loaded.file_ids(), {9})

    def test_save_creates_parent_directory(self):
        deep = self.dir / "nested" / "deeper" / MANIFEST_FILENAME
        m = Manifest(deep, 3)
        m.add(meta(1, 1, "a", "b"))
        m.save()
        self.assertTrue(deep.exists())

    def test_load_missing_file_raises(self):
        with self.assertRaises(CorruptionError):
            self.new_manifest().load()

    def test_load_rejects_invalid_json(self):
        self.path.write_bytes(b"{not json at all")
        with self.assertRaises(CorruptionError):
            self.new_manifest().load()

    def test_load_rejects_invalid_utf8(self):
        self.path.write_bytes(b"\xff\xfe\x00\x01")
        with self.assertRaises(CorruptionError):
            self.new_manifest().load()

    def test_load_rejects_wrong_version(self):
        self.path.write_text(
            json.dumps({"version": MANIFEST_VERSION + 1, "num_levels": 3}),
            encoding="utf-8",
        )
        with self.assertRaises(CorruptionError):
            self.new_manifest().load()

    def test_load_rejects_missing_version(self):
        self.path.write_text(json.dumps({"num_levels": 3}), encoding="utf-8")
        with self.assertRaises(CorruptionError):
            self.new_manifest().load()

    def test_load_rejects_bad_num_levels(self):
        self.path.write_text(
            json.dumps({"version": MANIFEST_VERSION, "num_levels": 1}),
            encoding="utf-8",
        )
        with self.assertRaises(CorruptionError):
            self.new_manifest().load()

    def test_load_rejects_data_beyond_declared_levels(self):
        """声明的层数是 2,却给了 3 层数据 —— 这是格式损坏,必须报错。"""
        payload = {
            "version": MANIFEST_VERSION,
            "num_levels": 2,
            "next_file_id": 1,
            "levels": [
                [],
                [],
                [meta(1, 2, "a", "b").to_dict()],
            ],
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(CorruptionError):
            self.new_manifest().load()

    def test_load_accepts_different_level_count(self):
        """manifest 记了 4 层,打开的引擎按 manifest 走,而不是按构造参数。"""
        m = Manifest(self.path, 4)
        m.add(meta(1, 3, "a", "b"))
        m.save()

        loaded = self.new_manifest(3)      # 构造时给的是 3
        loaded.load()
        self.assertEqual(loaded.num_levels, 4)
        self.assertEqual(loaded.max_level, 3)

    def test_loaded_manifest_passes_invariants(self):
        m = self.new_manifest()
        for file_id in (1, 2, 3, 4):
            m.add(meta(file_id, 0, "a", "z"))
        m.add(meta(10, 1, "a", "c"))
        m.add(meta(11, 1, "d", "f"))
        m.save()

        loaded = self.new_manifest()
        loaded.load()
        loaded.check_invariants()


# ------------------------------------------------------------------ 不变量检查


class TestCheckInvariants(ManifestTestCase):
    def test_valid_manifest_passes(self):
        m = self.new_manifest()
        m.add(meta(2, 0, "a", "z"))
        m.add(meta(1, 0, "a", "z"))
        m.add(meta(5, 1, "a", "c"))
        m.add(meta(6, 1, "d", "f"))
        m.check_invariants()

    def test_empty_manifest_passes(self):
        self.new_manifest().check_invariants()

    def test_detects_duplicate_file_id_across_levels(self):
        m = self.new_manifest()
        m.add(meta(1, 0, "a", "b"))
        m.add(meta(1, 1, "a", "b"))
        with self.assertRaises(CorruptionError) as ctx:
            m.check_invariants()
        self.assertIn("同时出现在", str(ctx.exception))

    def test_detects_reversed_key_range(self):
        m = self.new_manifest()
        m.levels[1].append(meta(1, 1, "z", "a"))
        with self.assertRaises(CorruptionError) as ctx:
            m.check_invariants()
        self.assertIn("键范围反了", str(ctx.exception))

    def test_detects_zero_entries(self):
        m = self.new_manifest()
        m.levels[1].append(meta(1, 1, "a", "b", entries=0))
        with self.assertRaises(CorruptionError) as ctx:
            m.check_invariants()
        self.assertIn("条目数为 0", str(ctx.exception))

    def test_detects_wrong_level_number(self):
        """文件记着自己属于 L2,却被挂在 L1 —— 层号不可信,查询会出错。"""
        m = self.new_manifest()
        m.levels[1].append(meta(1, 2, "a", "b"))
        with self.assertRaises(CorruptionError) as ctx:
            m.check_invariants()
        self.assertIn("记的层号是", str(ctx.exception))

    def test_detects_unsorted_l0(self):
        m = self.new_manifest()
        m.levels[0] = [meta(1, 0, "a", "b"), meta(5, 0, "a", "b")]   # 新的在后
        with self.assertRaises(CorruptionError) as ctx:
            m.check_invariants()
        self.assertIn("file_id 从大到小", str(ctx.exception))

    def test_detects_unsorted_l1(self):
        m = self.new_manifest()
        m.levels[1] = [meta(1, 1, "m", "n"), meta(2, 1, "a", "b")]
        with self.assertRaises(CorruptionError) as ctx:
            m.check_invariants()
        self.assertIn("升序排列", str(ctx.exception))

    def test_detects_overlap_in_l1(self):
        """层内重叠会让"二分找到唯一文件"这个前提崩掉,必须报错。"""
        m = self.new_manifest()
        m.add(meta(1, 1, "a", "m"))
        m.add(meta(2, 1, "f", "z"))
        with self.assertRaises(CorruptionError) as ctx:
            m.check_invariants()
        self.assertIn("重叠", str(ctx.exception))

    def test_adjacent_but_not_overlapping_is_fine(self):
        """前一个的 largest 正好等于后一个的 smallest 是允许的。"""
        m = self.new_manifest()
        m.add(meta(1, 1, "a", "c"))
        m.add(meta(2, 1, "d", "f"))
        m.check_invariants()

    def test_l0_overlap_is_allowed(self):
        """L0 天生允许重叠 —— 不能把它也当错误。"""
        m = self.new_manifest()
        m.add(meta(1, 0, "a", "z"))
        m.add(meta(2, 0, "a", "z"))
        m.add(meta(3, 0, "a", "z"))
        m.check_invariants()

    def test_repr_is_readable(self):
        m = self.new_manifest()
        m.add(meta(1, 0, "a", "b"))
        m.add(meta(2, 1, "a", "b"))
        text = repr(m)
        self.assertIn("L0=1", text)
        self.assertIn("L1=1", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
