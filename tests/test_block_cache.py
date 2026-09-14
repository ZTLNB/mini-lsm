"""阶段 4 测试:LRU 块缓存。

块缓存本身不难,难的是三件事,测试也主要盯着这三件:

1. **淘汰顺序必须是 LRU**。淘汰错了不会报错,只会让命中率莫名其妙地低 ——
   属于"悄悄变慢"的 bug,很难从结果上看出来,所以必须用测试钉死顺序。

2. **记账必须准**。``_bytes`` 一旦和实际内容对不上,缓存要么提前清空、
   要么无限膨胀。重复 put 同一个键、淘汰、``evict_file`` 之后
   都要核对字节数。

3. **缓存键必须带 file_id**。compaction 之后文件会被重新编号,
   不同文件里的"第 3 块"内容完全不同。键里少了 file_id 就会读到
   别人的数据 —— 这是**返回错值**,比慢严重得多。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.block_cache import (  # noqa: E402
    CACHE_ENTRY_OVERHEAD,
    DEFAULT_CACHE_BYTES,
    BlockCache,
    CacheStats,
    estimate_entries_size,
)

# 记录类型用不到,随便给个值 —— 缓存只关心 (key, value) 的字节数。
PUT = 1


def entries(count: int, key_size: int = 1, value_size: int = 1) -> list[tuple]:
    """造一批尺寸可控的记录,用来精确预测缓存占用。"""
    return [
        (PUT, bytes([i]) * key_size, b"v" * value_size)
        for i in range(count)
    ]


def size_of(count: int, key_size: int = 1, value_size: int = 1) -> int:
    """这批记录应该被记成多少字节。"""
    return count * (key_size + value_size + CACHE_ENTRY_OVERHEAD)


# ------------------------------------------------------------------ 容量与构造


class TestConstruction(unittest.TestCase):
    def test_negative_capacity_rejected(self):
        with self.assertRaises(ValueError):
            BlockCache(-1)

    def test_zero_capacity_is_disabled_not_error(self):
        """容量 0 表示"关掉缓存",是合法配置,不该报错。"""
        cache = BlockCache(0)
        self.assertFalse(cache.enabled)
        self.assertEqual(len(cache), 0)

    def test_default_capacity_is_positive(self):
        self.assertGreater(DEFAULT_CACHE_BYTES, 0)
        self.assertTrue(BlockCache().enabled)

    def test_initial_state_is_empty(self):
        cache = BlockCache(1024)
        self.assertEqual(len(cache), 0)
        st = cache.stats()
        self.assertEqual(st.entries, 0)
        self.assertEqual(st.bytes, 0)
        self.assertEqual(st.lookups, 0)

    def test_capacity_reported(self):
        self.assertEqual(BlockCache(4096).capacity_bytes, 4096)


# ------------------------------------------------------------------ 基本读写


class TestBasicGetPut(unittest.TestCase):
    def setUp(self):
        self.cache = BlockCache(10_000)

    def test_miss_on_empty(self):
        self.assertIsNone(self.cache.get(1, 0))
        self.assertEqual(self.cache.stats().misses, 1)

    def test_put_then_get(self):
        data = entries(3)
        self.cache.put(1, 0, data)
        self.assertIs(self.cache.get(1, 0), data)

    def test_hit_and_miss_counters(self):
        self.cache.put(1, 0, entries(1))
        self.cache.get(1, 0)   # 命中
        self.cache.get(1, 1)   # 未命中
        st = self.cache.stats()
        self.assertEqual(st.hits, 1)
        self.assertEqual(st.misses, 1)
        self.assertEqual(st.lookups, 2)
        self.assertAlmostEqual(st.hit_rate, 0.5)

    def test_hit_rate_of_no_lookups_is_zero(self):
        self.assertEqual(self.cache.stats().hit_rate, 0.0)

    def test_bytes_tracked(self):
        self.cache.put(1, 0, entries(5))
        self.assertEqual(self.cache.stats().bytes, size_of(5))

    def test_contains_and_len(self):
        self.cache.put(1, 0, entries(1))
        self.assertIn((1, 0), self.cache)
        self.assertNotIn((1, 1), self.cache)
        self.assertEqual(len(self.cache), 1)

    def test_put_empty_list_is_noop(self):
        """空块不缓存 —— 存它没有任何意义,还会污染淘汰顺序。"""
        self.cache.put(1, 0, [])
        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.cache.stats().bytes, 0)

    def test_same_block_object_returned(self):
        """命中要返回**同一个对象**,而不是拷贝 —— 缓存的收益就在这里。"""
        data = entries(4)
        self.cache.put(1, 0, data)
        self.assertIs(self.cache.get(1, 0), self.cache.get(1, 0))


class TestFileIdIsPartOfTheKey(unittest.TestCase):
    """缓存键必须带 file_id,否则不同文件会互相串味。"""

    def test_same_block_index_different_files(self):
        cache = BlockCache(10_000)
        a = [(PUT, b"a", b"from-file-1")]
        b = [(PUT, b"a", b"from-file-2")]
        cache.put(1, 0, a)
        cache.put(2, 0, b)
        self.assertIs(cache.get(1, 0), a)
        self.assertIs(cache.get(2, 0), b)

    def test_different_block_index_same_file(self):
        cache = BlockCache(10_000)
        first = [(PUT, b"a", b"one")]
        second = [(PUT, b"b", b"two")]
        cache.put(7, 0, first)
        cache.put(7, 1, second)
        self.assertIs(cache.get(7, 0), first)
        self.assertIs(cache.get(7, 1), second)
        self.assertEqual(len(cache), 2)


# ------------------------------------------------------------------ LRU 淘汰


class TestLRUEviction(unittest.TestCase):
    """淘汰顺序。用固定大小的块,让每次淘汰都精确可控。"""

    def setUp(self):
        # 每块 1 条记录 = 1 + 1 + 64 = 66 字节。
        # 容量 200 → 最多放 3 块(198 字节),放第 4 块必然淘汰 1 块。
        self.unit = size_of(1)
        self.cache = BlockCache(self.unit * 3)

    def fill(self, *blocks):
        for file_id, index in blocks:
            self.cache.put(file_id, index, entries(1))

    def test_evicts_least_recently_used(self):
        self.fill((1, 0), (1, 1), (1, 2))
        self.cache.get(1, 0)          # 把 0 用一下,它变成"最近使用"
        self.fill((1, 3))             # 触发淘汰,应该淘汰 1
        self.assertIn((1, 0), self.cache)
        self.assertIn((1, 2), self.cache)
        self.assertIn((1, 3), self.cache)
        self.assertNotIn((1, 1), self.cache)

    def test_put_also_counts_as_use(self):
        """刚 put 进去的块算"最近使用",不能被立刻淘汰掉。"""
        self.fill((1, 0), (1, 1), (1, 2))
        self.fill((1, 3))
        self.assertIn((1, 3), self.cache)
        self.assertNotIn((1, 0), self.cache)

    def test_eviction_counter(self):
        self.fill((1, 0), (1, 1), (1, 2), (1, 3))
        self.assertEqual(self.cache.stats().evictions, 1)

    def test_bytes_stay_within_capacity(self):
        for i in range(20):
            self.fill((1, i))
        st = self.cache.stats()
        self.assertLessEqual(st.bytes, self.cache.capacity_bytes)
        self.assertEqual(st.bytes, st.entries * self.unit)

    def test_len_never_exceeds_what_fits(self):
        for i in range(20):
            self.fill((1, i))
        self.assertLessEqual(len(self.cache), 3)

    def test_eviction_keeps_newest(self):
        """连续灌入,最后一块必须还在。"""
        for i in range(10):
            self.fill((1, i))
        self.assertIn((1, 9), self.cache)

    def test_touching_a_key_protects_it_repeatedly(self):
        """反复摸同一块,它永远不该被淘汰 —— 这是 LRU 的核心承诺。"""
        self.fill((1, 0), (1, 1), (1, 2))
        for i in range(3, 30):
            self.cache.get(1, 0)
            self.fill((1, i))
            self.assertIn((1, 0), self.cache)


class TestAccounting(unittest.TestCase):
    def test_reput_same_key_does_not_double_count(self):
        cache = BlockCache(10_000)
        cache.put(1, 0, entries(5))
        cache.put(1, 0, entries(5))
        st = cache.stats()
        self.assertEqual(st.entries, 1)
        self.assertEqual(st.bytes, size_of(5))

    def test_reput_with_different_size_updates_bytes(self):
        """同一个键换成更大的内容,字节数要跟着变,不能累加。"""
        cache = BlockCache(10_000)
        cache.put(1, 0, entries(2))
        cache.put(1, 0, entries(9))
        st = cache.stats()
        self.assertEqual(st.entries, 1)
        self.assertEqual(st.bytes, size_of(9))

    def test_reput_does_not_count_as_eviction(self):
        cache = BlockCache(10_000)
        cache.put(1, 0, entries(5))
        cache.put(1, 0, entries(5))
        self.assertEqual(cache.stats().evictions, 0)

    def test_reput_keeps_newest_position(self):
        """重新 put 一个键 = 刚用过它,不该成为下一个被淘汰的。"""
        unit = size_of(1)
        cache = BlockCache(unit * 3)
        for index in range(3):
            cache.put(1, index, entries(1))
        cache.put(1, 0, entries(1))   # 刷新 0
        cache.put(1, 3, entries(1))   # 淘汰 1(最旧)
        self.assertIn((1, 0), cache)
        self.assertNotIn((1, 1), cache)


class TestOversizedBlock(unittest.TestCase):
    def test_block_larger_than_cache_is_not_stored(self):
        cache = BlockCache(100)
        cache.put(1, 0, entries(10))    # 10 * 66 = 660 字节 > 100
        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.stats().oversized, 1)

    def test_oversized_block_does_not_clear_existing_entries(self):
        """塞不下的块应该被直接放弃,而不是把缓存冲一遍。"""
        cache = BlockCache(300)
        cache.put(1, 0, entries(1))
        cache.put(1, 1, entries(10))    # 太大
        self.assertIn((1, 0), cache)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.stats().evictions, 0)

    def test_oversized_block_still_misses(self):
        cache = BlockCache(100)
        cache.put(1, 0, entries(10))
        self.assertIsNone(cache.get(1, 0))


class TestDisabledCache(unittest.TestCase):
    """容量 0 时,所有操作都应该是安全的空操作。"""

    def setUp(self):
        self.cache = BlockCache(0)

    def test_put_does_nothing(self):
        self.cache.put(1, 0, entries(3))
        self.assertEqual(len(self.cache), 0)

    def test_get_always_misses(self):
        self.assertIsNone(self.cache.get(1, 0))
        self.assertEqual(self.cache.stats().misses, 1)

    def test_put_does_not_count_as_oversized(self):
        """禁用和"块太大"是两回事,统计上要分得清。"""
        self.cache.put(1, 0, entries(3))
        self.assertEqual(self.cache.stats().oversized, 0)

    def test_stats_render_mentions_disabled(self):
        self.assertIn("已禁用", str(self.cache.stats()))


# ------------------------------------------------------------------ 按文件清理


class TestEvictFile(unittest.TestCase):
    def setUp(self):
        self.cache = BlockCache(10_000)
        self.cache.put(1, 0, entries(2))
        self.cache.put(1, 1, entries(2))
        self.cache.put(2, 0, entries(2))

    def test_removes_only_that_file(self):
        removed = self.cache.evict_file(1)
        self.assertEqual(removed, 2)
        self.assertNotIn((1, 0), self.cache)
        self.assertNotIn((1, 1), self.cache)
        self.assertIn((2, 0), self.cache)

    def test_bytes_updated(self):
        self.cache.evict_file(1)
        self.assertEqual(self.cache.stats().bytes, size_of(2))

    def test_evicting_absent_file_is_noop(self):
        self.assertEqual(self.cache.evict_file(99), 0)
        self.assertEqual(len(self.cache), 3)

    def test_evicting_all_files_leaves_empty(self):
        self.cache.evict_file(1)
        self.cache.evict_file(2)
        self.assertEqual(len(self.cache), 0)
        self.assertEqual(self.cache.stats().bytes, 0)

    def test_evict_file_does_not_count_as_eviction(self):
        """主动清理和"容量不够被挤掉"要分开统计,不然命中率分析会被带偏。"""
        self.cache.evict_file(1)
        self.assertEqual(self.cache.stats().evictions, 0)


class TestClear(unittest.TestCase):
    def test_clear_removes_content_but_keeps_stats(self):
        cache = BlockCache(10_000)
        cache.put(1, 0, entries(3))
        cache.get(1, 0)
        cache.clear()
        st = cache.stats()
        self.assertEqual(st.entries, 0)
        self.assertEqual(st.bytes, 0)
        self.assertEqual(st.hits, 1)     # 统计保留
        self.assertEqual(len(cache), 0)

    def test_usable_after_clear(self):
        cache = BlockCache(10_000)
        cache.put(1, 0, entries(1))
        cache.clear()
        cache.put(1, 0, entries(1))
        self.assertIn((1, 0), cache)


class TestResetStats(unittest.TestCase):
    def test_resets_counters_not_content(self):
        cache = BlockCache(10_000)
        cache.put(1, 0, entries(3))
        cache.get(1, 0)
        cache.get(1, 9)
        cache.reset_stats()
        st = cache.stats()
        self.assertEqual(st.lookups, 0)
        self.assertEqual(st.hits, 0)
        self.assertEqual(st.misses, 0)
        self.assertEqual(st.entries, 1)   # 内容还在
        self.assertEqual(st.bytes, size_of(3))

    def test_hit_rate_after_reset(self):
        cache = BlockCache(10_000)
        cache.put(1, 0, entries(1))
        cache.get(1, 0)
        cache.reset_stats()
        cache.get(1, 0)
        self.assertEqual(cache.stats().hit_rate, 1.0)


# ------------------------------------------------------------------ 统计展示


class TestCacheStats(unittest.TestCase):
    def test_hit_rate_zero_division(self):
        self.assertEqual(CacheStats().hit_rate, 0.0)

    def test_usage_ratio(self):
        st = CacheStats(bytes=50, capacity_bytes=200)
        self.assertAlmostEqual(st.usage_ratio, 0.25)

    def test_usage_ratio_zero_capacity(self):
        self.assertEqual(CacheStats(bytes=50, capacity_bytes=0).usage_ratio, 0.0)

    def test_str_mentions_numbers(self):
        cache = BlockCache(1000)
        cache.put(1, 0, entries(3))
        cache.get(1, 0)
        text = str(cache.stats())
        self.assertIn("命中 1 / 1", text)
        self.assertIn("1000", text)

    def test_estimate_entries_size_matches_formula(self):
        self.assertEqual(estimate_entries_size(entries(4, 3, 5)), 4 * (3 + 5 + 64))

    def test_estimate_entries_size_of_empty(self):
        self.assertEqual(estimate_entries_size([]), 0)


class TestRealisticSizes(unittest.TestCase):
    def test_default_cache_holds_many_small_blocks(self):
        """默认 8 MiB 应该能放下不少 4 KiB 级别的块。"""
        cache = BlockCache()          # 默认容量
        # 造一个大约 4 KiB 的块:约 60 条 64 字节的记录
        block = entries(60, key_size=32, value_size=32)
        per_block = estimate_entries_size(block)
        for i in range(10):
            cache.put(1, i, block)
        self.assertEqual(len(cache), 10)
        self.assertEqual(cache.stats().bytes, per_block * 10)
        self.assertLess(cache.stats().usage_ratio, 0.5)

    def test_many_tiny_blocks_respect_capacity(self):
        cache = BlockCache(1024)
        for i in range(200):
            cache.put(1, i, entries(1))
        st = cache.stats()
        self.assertLessEqual(st.bytes, 1024)
        self.assertGreater(st.evictions, 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
