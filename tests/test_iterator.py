"""归并迭代器测试。

它是"多来源读路径"的唯一抽象,所以测试要盯死那份契约:
    有序、同 key 取最新、墓碑不丢、且**不会因为 value 是 None 就崩**。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.iterator import MergingIterator  # noqa: E402


def collect(sources):
    return list(MergingIterator(sources))


class TestSingleSource(unittest.TestCase):
    def test_passthrough(self):
        src = [(b"a", b"1"), (b"b", b"2")]
        self.assertEqual(collect([src]), src)

    def test_empty_source(self):
        self.assertEqual(collect([[]]), [])

    def test_no_sources(self):
        self.assertEqual(collect([]), [])

    def test_all_sources_empty(self):
        self.assertEqual(collect([[], [], []]), [])

    def test_single_entry(self):
        self.assertEqual(collect([[(b"k", b"v")]]), [(b"k", b"v")])


class TestMerge(unittest.TestCase):
    def test_interleaved_sources(self):
        left = [(b"a", b"1"), (b"c", b"3"), (b"e", b"5")]
        right = [(b"b", b"2"), (b"d", b"4")]
        self.assertEqual(
            collect([left, right]),
            [(b"a", b"1"), (b"b", b"2"), (b"c", b"3"), (b"d", b"4"), (b"e", b"5")],
        )

    def test_disjoint_ranges(self):
        self.assertEqual(
            collect([[(b"a", b"1")], [(b"z", b"9")]]),
            [(b"a", b"1"), (b"z", b"9")],
        )

    def test_one_source_empty(self):
        self.assertEqual(
            collect([[(b"a", b"1")], []]),
            [(b"a", b"1")],
        )

    def test_many_sources(self):
        sources = [[(f"k{i:02d}".encode(), str(i).encode())] for i in range(20)]
        result = collect(sources)
        self.assertEqual([k for k, _ in result],
                         [f"k{i:02d}".encode() for i in range(20)])

    def test_result_is_sorted(self):
        sources = [[(bytes([i * 2]), b"x") for i in range(5)],
                   [(bytes([i * 2 + 1]), b"y") for i in range(5)]]
        keys = [k for k, _ in collect(sources)]
        self.assertEqual(keys, sorted(keys))


class TestNewestWins(unittest.TestCase):
    def test_newer_source_wins(self):
        newer = [(b"k", b"new")]
        older = [(b"k", b"old")]
        self.assertEqual(collect([newer, older]), [(b"k", b"new")])

    def test_newest_of_three_wins(self):
        self.assertEqual(
            collect([[(b"k", b"v1")], [(b"k", b"v2")], [(b"k", b"v3")]]),
            [(b"k", b"v1")],
        )

    def test_duplicate_keys_yield_once(self):
        sources = [[(b"a", b"1"), (b"b", b"2")],
                   [(b"a", b"9"), (b"b", b"9"), (b"c", b"3")]]
        result = collect(sources)
        self.assertEqual(result, [(b"a", b"1"), (b"b", b"2"), (b"c", b"3")])
        self.assertEqual(len(result), len({k for k, _ in result}))

    def test_key_only_in_older_source(self):
        self.assertEqual(
            collect([[(b"a", b"1")], [(b"b", b"2")]]),
            [(b"a", b"1"), (b"b", b"2")],
        )


class TestTombstones(unittest.TestCase):
    def test_tombstone_is_yielded(self):
        """墓碑必须被产出 —— compaction 靠它才知道这个键被删过。"""
        self.assertEqual(collect([[(b"k", None)]]), [(b"k", None)])

    def test_newer_tombstone_shadows_older_value(self):
        """新层写了墓碑,旧层的值就必须被压住 —— 否则删除会失效。"""
        newer = [(b"k", None)]
        older = [(b"k", b"resurrected")]
        self.assertEqual(collect([newer, older]), [(b"k", None)])

    def test_newer_value_overrides_older_tombstone(self):
        """反过来:重新写回一个键,旧墓碑不该继续压制它。"""
        newer = [(b"k", b"back")]
        older = [(b"k", None)]
        self.assertEqual(collect([newer, older]), [(b"k", b"back")])

    def test_mixing_none_and_bytes_does_not_crash(self):
        """堆里同时有 None 和 bytes 时不能去比较它们。

        这是实现里那个 seq 序号要防的事:bytes 和 None 没法比大小,
        一旦堆比较走到 value 上就会抛 TypeError。
        """
        sources = [[(b"k", None)], [(b"k", b"v")]]
        self.assertEqual(collect(sources), [(b"k", None)])

    def test_live_skips_tombstones(self):
        merged = MergingIterator([[(b"a", b"1"), (b"b", None), (b"c", b"3")]])
        self.assertEqual(list(merged.live()), [(b"a", b"1"), (b"c", b"3")])

    def test_live_across_sources(self):
        merged = MergingIterator([
            [(b"a", b"new")],
            [(b"a", b"old"), (b"b", None)],
        ])
        self.assertEqual(list(merged.live()), [(b"a", b"new")])


class TestLaziness(unittest.TestCase):
    def test_does_not_consume_everything_upfront(self):
        """只取一条不应该把来源全部读完 —— 否则扫描大库会炸内存。"""
        pulled = []

        def source():
            for i in range(1000):
                pulled.append(i)
                yield (f"k{i:04d}".encode(), b"v")

        iterator = MergingIterator([source()])
        first = next(iterator)
        self.assertEqual(first[0], b"k0000")
        self.assertLess(len(pulled), 1000)

    def test_iterating_twice_is_exhausted(self):
        iterator = MergingIterator([[(b"a", b"1")]])
        self.assertEqual(list(iterator), [(b"a", b"1")])
        self.assertEqual(list(iterator), [])

    def test_repr(self):
        self.assertIn("MergingIterator", repr(MergingIterator([[], []])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
