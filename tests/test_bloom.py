"""阶段 4 测试:布隆过滤器。

这个模块有一条**绝对不能破**的性质:

    **没有假阴性。**

    加进去的每一个 key,``might_contain()`` 都必须返回 True。
    破了这条,结果就是"明明存在的键被报告为不存在" —— 静默丢数据。

反过来,假阳性是**允许**的:它只意味着"白读一次文件",慢一点而已。
整个设计的所有取舍都倒向"宁可说在"。

另外还有一条容易被忽略、但同样致命的性质:

    **哈希必须跨进程稳定。**

    过滤器是要落盘的。如果哈希函数随进程变化(比如用了内置 ``hash()``,
    CPython 会给它加每进程随机的盐),那么进程 A 写下的过滤器在进程 B
    里查什么都会落到错误的位上 —— 于是**所有存在的键全部变成假阴性**。
    下面 ``TestCrossProcessStability`` 专门用子进程把这条钉住。
"""

import os
import random
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.bloom import (  # noqa: E402
    DEFAULT_BITS_PER_KEY,
    MAX_K,
    MIN_BITS,
    BloomBuilder,
    BloomFilter,
    bits_for_keys,
    estimate_false_positive_rate,
    optimal_k,
)
from mini_lsm.errors import CorruptionError, InvalidArgumentError  # noqa: E402

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def sample_keys(n: int, prefix: str = "key") -> list[bytes]:
    return [f"{prefix}{i:06d}".encode() for i in range(n)]


# ------------------------------------------------------------------ 参数计算


class TestParameters(unittest.TestCase):
    def test_optimal_k_is_bits_times_ln2(self):
        """最优 k = bits_per_key * ln2,取整。10 bit/key 对应 k=7。"""
        self.assertEqual(optimal_k(10), 7)
        self.assertEqual(optimal_k(8), 6)
        self.assertEqual(optimal_k(16), 11)

    def test_optimal_k_never_below_one(self):
        """bits_per_key 很小时也不能算出 k=0 —— 那样过滤器什么也挡不住。"""
        self.assertEqual(optimal_k(1), 1)

    def test_optimal_k_capped(self):
        self.assertLessEqual(optimal_k(10_000), MAX_K)

    def test_bits_for_keys_grows_linearly(self):
        self.assertEqual(bits_for_keys(1000, 10), 10000)
        self.assertEqual(bits_for_keys(2000, 10), 20000)

    def test_bits_for_keys_has_a_floor(self):
        """key 很少时也要有个下限,否则位图会被一两个 key 占满。"""
        self.assertEqual(bits_for_keys(0, 10), MIN_BITS)
        self.assertEqual(bits_for_keys(1, 10), MIN_BITS)

    def test_bits_for_keys_rounded_to_byte(self):
        for n in range(0, 200):
            self.assertEqual(bits_for_keys(n, 7) % 8, 0)

    def test_false_positive_rate_decreases_with_more_bits(self):
        rates = [estimate_false_positive_rate(1000, bits_for_keys(1000, b), optimal_k(b))
                 for b in (4, 8, 10, 16)]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_false_positive_rate_of_empty_filter_is_zero(self):
        self.assertEqual(estimate_false_positive_rate(0, 1024, 7), 0.0)

    def test_false_positive_rate_handles_zero_bits(self):
        self.assertEqual(estimate_false_positive_rate(10, 0, 7), 0.0)


# ------------------------------------------------------------------ 核心性质


class TestNoFalseNegatives(unittest.TestCase):
    """整个模块最重要的一组:加进去的 key 必须全部查得到。"""

    def test_single_key(self):
        bf = BloomFilter.build([b"hello"])
        self.assertTrue(bf.might_contain(b"hello"))

    def test_many_keys_at_various_sizes(self):
        for n in (1, 2, 7, 50, 500, 5000):
            with self.subTest(n=n):
                keys = sample_keys(n)
                bf = BloomFilter.build(keys)
                missing = [k for k in keys if not bf.might_contain(k)]
                self.assertEqual(missing, [], f"{n} 个 key 里出现了假阴性")

    def test_various_bits_per_key(self):
        """bits_per_key 再怎么调,都不能影响"无假阴性"。"""
        keys = sample_keys(500)
        for bits in (1, 2, 4, 8, 10, 16, 32):
            with self.subTest(bits=bits):
                bf = BloomFilter.build(keys, bits_per_key=bits)
                self.assertTrue(all(bf.might_contain(k) for k in keys))

    def test_duplicate_keys_are_fine(self):
        keys = [b"same", b"same", b"same"]
        bf = BloomFilter.build(keys)
        self.assertTrue(bf.might_contain(b"same"))
        self.assertEqual(bf.num_keys, 3)

    def test_non_ascii_keys(self):
        keys = ["北京".encode(), "上海".encode(), "深圳".encode()]
        bf = BloomFilter.build(keys)
        self.assertTrue(all(bf.might_contain(k) for k in keys))

    def test_binary_keys_with_nul_bytes(self):
        keys = [b"\x00\x00", b"\x00\xff", b"\xff\x00", b"\xff\xff", bytes(100)]
        bf = BloomFilter.build(keys)
        self.assertTrue(all(bf.might_contain(k) for k in keys))

    def test_empty_key(self):
        bf = BloomFilter.build([b""])
        self.assertTrue(bf.might_contain(b""))

    def test_very_long_key(self):
        key = b"x" * 100_000
        bf = BloomFilter.build([key, b"other"])
        self.assertTrue(bf.might_contain(key))

    def test_keys_that_differ_by_one_bit(self):
        """只差一个字节的 key 不能被混为一谈。"""
        keys = [b"key\x00", b"key\x01", b"key\x02", b"key\x80", b"key\xff"]
        bf = BloomFilter.build(keys)
        self.assertTrue(all(bf.might_contain(k) for k in keys))

    def test_after_roundtrip_still_no_false_negatives(self):
        """落盘再读回来,无假阴性必须依然成立 —— 这是持久化的意义。"""
        keys = sample_keys(2000)
        bf = BloomFilter.from_bytes(BloomFilter.build(keys).to_bytes())
        self.assertTrue(all(bf.might_contain(k) for k in keys))

    def test_large_filter(self):
        keys = sample_keys(50_000)
        bf = BloomFilter.build(keys)
        self.assertTrue(all(bf.might_contain(k) for k in keys))


class TestEmptyFilter(unittest.TestCase):
    def test_empty_rejects_everything(self):
        """没有 key 的过滤器应该把一切都挡掉(它里面确实什么都没有)。"""
        bf = BloomFilter.build([])
        for probe in (b"a", b"", b"anything"):
            self.assertFalse(bf.might_contain(probe))

    def test_empty_filter_properties(self):
        bf = BloomFilter.build([])
        self.assertEqual(bf.num_keys, 0)
        self.assertEqual(bf.num_bits, MIN_BITS)
        self.assertEqual(bf.false_positive_rate, 0.0)

    def test_empty_filter_roundtrip(self):
        bf = BloomFilter.from_bytes(BloomFilter.build([]).to_bytes())
        self.assertFalse(bf.might_contain(b"a"))


# ------------------------------------------------------------------ 假阳性率


class TestFalsePositiveRate(unittest.TestCase):
    def measure(self, bf: BloomFilter, count: int = 20_000, seed: int = 12345) -> float:
        rng = random.Random(seed)
        probes = [f"absent-{rng.randrange(10 ** 9):09d}".encode()
                  for _ in range(count)]
        hits = sum(1 for p in probes if bf.might_contain(p))
        return hits / count

    def test_measured_matches_theory_at_10_bits(self):
        """实测误判率应该和公式算出来的接近(允许 1.5 倍偏差)。

        公式本身是近似(假设各位独立),所以不能要求完全相等。
        """
        keys = sample_keys(2000)
        bf = BloomFilter.build(keys, bits_per_key=10)
        measured = self.measure(bf)
        expected = bf.false_positive_rate
        self.assertLess(measured, expected * 1.5,
                        f"实测 {measured:.4%} 明显高于理论 {expected:.4%}")
        self.assertGreater(measured, expected * 0.3,
                           f"实测 {measured:.4%} 明显低于理论 {expected:.4%}")

    def test_more_bits_means_fewer_false_positives(self):
        keys = sample_keys(2000)
        few = self.measure(BloomFilter.build(keys, bits_per_key=4))
        many = self.measure(BloomFilter.build(keys, bits_per_key=16))
        self.assertLess(many, few)

    def test_bit_density_near_half_at_optimum(self):
        """最优参数下大约一半的位是 1 —— 这是过滤器处于最佳工作点的标志。"""
        bf = BloomFilter.build(sample_keys(2000), bits_per_key=10)
        self.assertGreater(bf.bit_density, 0.35)
        self.assertLess(bf.bit_density, 0.65)

    def test_rejects_the_vast_majority_of_absent_keys(self):
        """这是阶段 4 存在的理由:绝大多数不存在的 key 要被挡下。"""
        bf = BloomFilter.build(sample_keys(2000), bits_per_key=10)
        self.assertGreater(1.0 - self.measure(bf), 0.97)


# ------------------------------------------------------------------ 序列化


class TestSerialization(unittest.TestCase):
    def test_roundtrip_preserves_everything(self):
        bf = BloomFilter.build(sample_keys(300), bits_per_key=12)
        again = BloomFilter.from_bytes(bf.to_bytes())
        self.assertEqual(again.num_keys, bf.num_keys)
        self.assertEqual(again.num_bits, bf.num_bits)
        self.assertEqual(again.k, bf.k)
        self.assertEqual(again.to_bytes(), bf.to_bytes())

    def test_roundtrip_is_byte_identical(self):
        """同样的 key 序列必须产出完全一样的字节 —— 哈希稳定性的直接体现。"""
        keys = sample_keys(100)
        self.assertEqual(
            BloomFilter.build(keys).to_bytes(),
            BloomFilter.build(keys).to_bytes(),
        )

    def test_size_is_header_plus_bits(self):
        bf = BloomFilter.build(sample_keys(1000), bits_per_key=10)
        self.assertEqual(bf.size, 9 + bf.num_bits // 8)
        self.assertEqual(len(bf.to_bytes()), bf.size)

    def test_rejects_truncated_header(self):
        with self.assertRaises(CorruptionError):
            BloomFilter.from_bytes(b"\x00" * 4)

    def test_rejects_empty_bytes(self):
        with self.assertRaises(CorruptionError):
            BloomFilter.from_bytes(b"")

    def test_rejects_zero_bits(self):
        """声明的位数为 0 —— 非法,必须报错而不是当成"什么都能过"。"""
        raw = BloomFilter.build([b"a"]).to_bytes()
        broken = raw[:4] + (0).to_bytes(4, "big") + raw[8:]
        with self.assertRaises(CorruptionError) as ctx:
            BloomFilter.from_bytes(broken)
        self.assertIn("位数非法", str(ctx.exception))

    def test_rejects_zero_k(self):
        raw = BloomFilter.build([b"a"]).to_bytes()
        broken = raw[:8] + bytes([0]) + raw[9:]
        with self.assertRaises(CorruptionError) as ctx:
            BloomFilter.from_bytes(broken)
        self.assertIn("哈希个数非法", str(ctx.exception))

    def test_rejects_k_beyond_max(self):
        raw = BloomFilter.build([b"a"]).to_bytes()
        broken = raw[:8] + bytes([MAX_K + 1]) + raw[9:]
        with self.assertRaises(CorruptionError):
            BloomFilter.from_bytes(broken)

    def test_rejects_length_mismatch(self):
        raw = BloomFilter.build([b"a"]).to_bytes()
        with self.assertRaises(CorruptionError) as ctx:
            BloomFilter.from_bytes(raw[:-1])
        self.assertIn("长度对不上", str(ctx.exception))

    def test_rejects_extra_trailing_bytes(self):
        raw = BloomFilter.build([b"a"]).to_bytes()
        with self.assertRaises(CorruptionError):
            BloomFilter.from_bytes(raw + b"\x00")

    def test_huge_declared_bits_does_not_allocate(self):
        """声明的位数是个天文数字 —— 必须先核对长度,不能直接去分配。

        和 ``sstable._read_block`` 里"先核对边界再申请内存"是同一条教训:
        损坏的位数能让"需要多少字节"算出 512 MiB,直接照着它分配
        就是一次无谓的大内存申请(并发几个文件就够把进程拖垮)。

        这里用 4 字节能表示的最大值(0xFFFFFFFF),也就是 512 MiB 的位图 ——
        真实数据只有 8 字节。实现必须先发现"长度对不上"再返回。
        """
        header = (1).to_bytes(4, "big") + ((1 << 32) - 1).to_bytes(4, "big") + bytes([7])
        with self.assertRaises(CorruptionError) as ctx:
            BloomFilter.from_bytes(header + b"\x00" * 8)
        self.assertIn("长度对不上", str(ctx.exception))


# ------------------------------------------------------------------ 跨进程稳定


class TestCrossProcessStability(unittest.TestCase):
    """过滤器要落盘,所以哈希必须跨进程一致。

    这里用子进程 + 不同的 ``PYTHONHASHSEED`` 来验证。
    内置 ``hash()`` 会因为每进程随机的盐而失败 —— 那会让重启之后
    **所有存在的键都变成假阴性**。
    """

    def run_child(self, hash_seed: str, bits_per_key: int = 10) -> tuple[str, str]:
        code = (
            "import hashlib, sys;"
            f"sys.path.insert(0, r'{SRC_DIR}');"
            "from mini_lsm.bloom import BloomFilter;"
            "bf = BloomFilter.build([b'alpha', b'beta', b'gamma'],"
            f" bits_per_key={bits_per_key});"
            "print(hashlib.sha256(bf.to_bytes()).hexdigest());"
            "print(hash(b'alpha'))"
        )
        env = {**os.environ, "PYTHONHASHSEED": hash_seed}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, timeout=60,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        digest, builtin_hash = proc.stdout.strip().splitlines()
        return digest, builtin_hash

    def test_filter_bytes_identical_across_processes(self):
        digests = [self.run_child(seed)[0] for seed in ("1", "2", "3")]
        self.assertEqual(
            len(set(digests)), 1,
            f"不同进程算出的过滤器字节不一致:{digests} —— "
            f"哈希函数不稳定,重启后会产生假阴性",
        )

    def test_builtin_hash_would_have_failed(self):
        """把"为什么不能用内置 hash()"这件事直接演示出来。

        ``hash(bytes)`` 每个进程的盐都不同,所以同样的输入算出的值不同。
        过滤器一旦用了它,写下来的位图在重启后就没法查了。
        """
        hashes = [self.run_child(seed)[1] for seed in ("1", "2", "3")]
        self.assertGreater(
            len(set(hashes)), 1,
            "内置 hash() 在不同进程返回了相同结果,这个演示失去意义",
        )

    def test_different_bits_per_key_gives_different_bytes(self):
        a = self.run_child("1", bits_per_key=8)[0]
        b = self.run_child("1", bits_per_key=16)[0]
        self.assertNotEqual(a, b)


# ------------------------------------------------------------------ 构建器


class TestBloomBuilder(unittest.TestCase):
    def test_counts_keys(self):
        b = BloomBuilder(10)
        for i in range(17):
            b.add(f"k{i}".encode())
        self.assertEqual(b.num_keys, 17)

    def test_build_is_repeatable(self):
        b = BloomBuilder(10)
        for key in (b"a", b"b"):
            b.add(key)
        self.assertEqual(b.build().to_bytes(), b.build().to_bytes())

    def test_builder_matches_build(self):
        keys = sample_keys(200)
        self.assertEqual(
            BloomBuilder(10).build().__class__, BloomFilter.build(keys).__class__
        )
        builder = BloomBuilder(10)
        for key in keys:
            builder.add(key)
        self.assertEqual(builder.build().to_bytes(),
                         BloomFilter.build(keys, 10).to_bytes())

    def test_rejects_non_positive_bits_per_key(self):
        with self.assertRaises(InvalidArgumentError):
            BloomBuilder(0)
        with self.assertRaises(InvalidArgumentError):
            BloomBuilder(-1)

    def test_empty_builder(self):
        bf = BloomBuilder(10).build()
        self.assertEqual(bf.num_keys, 0)
        self.assertFalse(bf.might_contain(b"a"))

    def test_streaming_keeps_only_hashes_not_keys(self):
        """构建器只保留 16 字节哈希,不保留 key 本身。

        用一堆超长 key 去喂它,内存占用不应该随 key 长度增长 ——
        这里只验证行为正确(内存无法在单元测试里可靠断言)。
        """
        b = BloomBuilder(10)
        huge = b"z" * 10_000
        for _ in range(100):
            b.add(huge)
        bf = b.build()
        self.assertEqual(bf.num_keys, 100)
        self.assertTrue(bf.might_contain(huge))
        # 位图大小只由 key 个数决定,与 key 长度无关
        self.assertEqual(bf.num_bits, bits_for_keys(100, 10))


# ------------------------------------------------------------------ 杂项


class TestMisc(unittest.TestCase):
    def test_contains_operator(self):
        bf = BloomFilter.build([b"present"])
        self.assertIn(b"present", bf)

    def test_default_bits_per_key(self):
        bf = BloomFilter.build(sample_keys(100))
        self.assertEqual(bf.k, optimal_k(DEFAULT_BITS_PER_KEY))

    def test_repr_mentions_key_parameters(self):
        text = repr(BloomFilter.build(sample_keys(10)))
        self.assertIn("keys=10", text)
        self.assertIn("k=", text)

    def test_might_contain_accepts_bytearray(self):
        bf = BloomFilter.build([b"abc"])
        self.assertTrue(bf.might_contain(bytearray(b"abc")))

    def test_filter_is_immutable_after_build(self):
        """构建完成之后内容不能变 —— 否则已落盘的过滤器会失效。"""
        bf = BloomFilter.build([b"a"])
        raw_before = bf.to_bytes()
        bf.might_contain(b"b")
        bf.might_contain(b"c")
        self.assertEqual(bf.to_bytes(), raw_before)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
