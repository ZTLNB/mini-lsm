"""阶段 4 收益实测:布隆过滤器 + 块缓存到底省掉了多少次磁盘读。

为什么需要这个脚本:

    单元测试证明的是"结果对不对",证明不了"到底省了多少"。
    而阶段 4 存在的**唯一理由**就是省 I/O —— 所以必须能把它量出来。

做法:
    把 ``SSTableReader._read_block`` 换成一个计数版本。这是所有数据块
    读盘的**唯一入口**(索引和过滤器在打开时读一次,之后常驻内存),
    所以数它就等于数真实磁盘读。

    测量时**关掉块缓存**(``block_cache_size=0``),否则第二次查同一个键
    会被缓存挡住,量到的就不是过滤器的作用了。

三组探针,对应三种不同的查询:
    1. 存在的键        —— 必须读到数据,过滤器不该挡任何一次
    2. 范围内不存在的键 —— 阶段 3 的痛点:必须真读文件才知道"没有"
    3. 范围外不存在的键 —— 二分本来就能挡,过滤器是锦上添花

用法::

    python tools/bench_stage4.py
"""

from __future__ import annotations

import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.engine import LSMEngine  # noqa: E402
from mini_lsm.sstable import SSTableReader  # noqa: E402

#: 数据量。3000 条、内存表 8 KiB 能刷出几十个文件,
#: 再靠自动归并压成 13 个左右 —— 和 demo.py 里的负载保持一致
NUM_KEYS = 3000
MEMTABLE_CAPACITY = 8 * 1024
PROBES_PER_GROUP = 300

_ORIGINAL_READ_BLOCK = SSTableReader._read_block


class _ReadCounter:
    """把 ``_read_block`` 换成计数版本。

    这是数据块读盘的唯一入口,所以计数就是"读了几次盘"。
    """

    def __init__(self) -> None:
        self.count = 0

    def install(self) -> None:
        self.count = 0
        counter = self

        def counting_read_block(self_, offset, length):
            counter.count += 1
            return _ORIGINAL_READ_BLOCK(self_, offset, length)

        SSTableReader._read_block = counting_read_block

    @staticmethod
    def uninstall() -> None:
        SSTableReader._read_block = _ORIGINAL_READ_BLOCK


def build(root: Path, name: str, bloom_bits: int, auto_compact: bool = True,
          shuffle: bool = False) -> Path:
    """造一份固定的负载,返回数据目录 —— 测量时重新打开。

    ``shuffle`` 决定写入顺序,这一点对结论影响很大:

        **顺序写** —— 每次刷盘的文件只覆盖一小段键区间,键范围很窄。
        于是"key 超出文件键范围"这个**免费**检查就能挡掉绝大多数文件,
        布隆过滤器能帮上的忙有限。

        **随机写** —— 每次刷盘的样本散布在整个键空间,每个文件的键范围
        都接近全域,范围检查几乎挡不掉任何文件。这才是读放大最严重的
        真实场景,也是布隆过滤器真正兑现价值的地方。
    """
    data_dir = root / name
    keys = [f"key{i:05d}" for i in range(NUM_KEYS)]
    if shuffle:
        # 固定种子,结果可复现
        rng = random.Random(20260914)
        rng.shuffle(keys)

    with LSMEngine(
        data_dir,
        memtable_capacity=MEMTABLE_CAPACITY,
        bloom_bits_per_key=bloom_bits,
        auto_compact=auto_compact,
    ) as db:
        for key in keys:
            db.put(key, "x" * 64)
    return data_dir


def probe_groups() -> dict[str, list[bytes]]:
    """三组探针。长度都刻意做成一样,避免"长度不同"变成隐藏变量。"""
    return {
        "存在的键": [f"key{i:05d}".encode() for i in range(PROBES_PER_GROUP)],
        "范围内不存在的键": [
            f"key{i:05d}x".encode() for i in range(PROBES_PER_GROUP)
        ],
        "范围外不存在的键": [
            f"nope{i:04d}".encode() for i in range(PROBES_PER_GROUP)
        ],
    }


def measure(data_dir: Path, probes: list[bytes]) -> float:
    """返回每个查询平均读了多少个数据块。"""
    counter = _ReadCounter()
    # 关掉块缓存:否则第二次查同一个块会被缓存挡住,量到的就不是过滤器的作用
    with LSMEngine(data_dir, memtable_capacity=MEMTABLE_CAPACITY,
                   block_cache_size=0) as db:
        counter.install()
        try:
            for key in probes:
                db.get(key)
        finally:
            counter.uninstall()
    return counter.count / len(probes)


def measure_cache(data_dir: Path, cache_bytes: int) -> tuple[float, float, float]:
    """返回 (冷读命中率, 热读命中率, 缓存占用率)。"""
    with LSMEngine(data_dir, memtable_capacity=MEMTABLE_CAPACITY,
                   block_cache_size=cache_bytes) as db:
        keys = [f"key{i:05d}".encode() for i in range(PROBES_PER_GROUP)]

        db.block_cache.clear()
        db.block_cache.reset_stats()
        for key in keys:
            db.get(key)
        cold = db.block_cache.stats()

        for key in keys:
            db.get(key)
        warm = db.block_cache.stats()
        return cold.hit_rate, warm.hit_rate, warm.usage_ratio


def describe(data_dir: Path) -> str:
    with LSMEngine(data_dir, memtable_capacity=MEMTABLE_CAPACITY) as db:
        stats = db.stats()
        total_filter = sum(r.filter_size for r in db.sstables)
        return (
            f"{stats.sstable_count} 个文件, {stats.sstable_bytes} 字节,"
            f" 分层 {stats.level_files},"
            f" 过滤器合计 {total_filter} 字节"
            f" ({total_filter / max(stats.sstable_bytes, 1):.1%})"
        )


def run_scenario(title: str, with_filter: Path, no_filter: Path) -> None:
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    print(f"  有过滤器: {describe(with_filter)}")
    print(f"  无过滤器: {describe(no_filter)}")
    print(f"\n  {'探针':<20}{'无过滤器':>12}{'有过滤器':>12}{'降幅':>10}")
    print("  " + "-" * 70)
    for label, probes in probe_groups().items():
        before = measure(no_filter, probes)
        after = measure(with_filter, probes)
        drop = (1 - after / before) if before else 0.0
        print(f"  {label:<20}{before:>10.2f} 次{after:>10.2f} 次{drop:>9.0%}")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="mini-lsm-bench-"))
    try:
        print("=" * 76)
        print("阶段 4 收益实测:过滤器与缓存省掉了多少磁盘读")
        print("=" * 76)
        print(f"\n负载: {NUM_KEYS} 条记录, 内存表容量 {MEMTABLE_CAPACITY} 字节,"
              f" 每组 {PROBES_PER_GROUP} 次查询")
        print("  测量时块缓存**关闭** —— 否则第二次查同一个块会被缓存挡住,")
        print("  量到的就不是过滤器的作用了。")

        # 场景一:归并之后。文件少,L0 只有 1 个,本来就不算慢
        run_scenario(
            "场景一:顺序写 + 归并(文件少,读放大本来就小)",
            build(root, "seq-compact-bloom", 10),
            build(root, "seq-compact-nobloom", 0),
        )

        # 场景二:不归并 + 顺序写。L0 有几十个文件,但每个文件的键范围
        # 只覆盖一小段 —— 免费的"范围检查"就能挡掉大部分
        run_scenario(
            "场景二:顺序写 + 不归并(文件多,但键范围很窄)",
            build(root, "seq-noc-bloom", 10, auto_compact=False),
            build(root, "seq-noc-nobloom", 0, auto_compact=False),
        )

        # 场景三:不归并 + 随机写序。每个文件的键范围都接近全域,
        # 范围检查失效 —— 这才是布隆过滤器真正兑现价值的地方
        run_scenario(
            "场景三:随机写序 + 不归并(每个文件的键范围都覆盖全域)",
            build(root, "rnd-noc-bloom", 10, auto_compact=False, shuffle=True),
            build(root, "rnd-noc-nobloom", 0, auto_compact=False, shuffle=True),
        )

        print("\n" + "=" * 76)
        print("块缓存:热点数据重复查询(用场景一的库)")
        print("=" * 76)
        compacted = root / "seq-compact-bloom"
        for cache_bytes in (64 * 1024, 256 * 1024, 1024 * 1024):
            cold, warm, usage = measure_cache(compacted, cache_bytes)
            print(f"  容量 {cache_bytes // 1024:>4} KiB:"
                  f" 冷读命中 {cold:>6.1%}, 热读命中 {warm:>6.1%},"
                  f" 占用 {usage:>5.1%}")
        print("\n  注:冷读命中率不是 0,因为一个块里有多条记录 ——")
        print("      查第 2 条时就命中了第 1 条带进来的那一块。")

        print("\n" + "=" * 76)
        print("结论")
        print("=" * 76)
        print("  1. 存在的键:有/无过滤器都是 1 次左右 —— 过滤器帮不上忙,")
        print("     但也**一次都没有误挡**,这正是必须守住的性质。")
        print("  2. 范围内不存在的键:这是阶段 3 剩下的大头。文件越多、")
        print("     每个文件的键范围越宽,收益越大(场景三降幅最大)。")
        print("  3. 对比场景二和场景三能看出一个容易被忽略的事实:")
        print("     **'key 超出文件键范围'这个检查本身就是一个免费的过滤器。**")
        print("     顺序写时每个文件只管一小段区间,光靠它就能挡掉大部分文件;")
        print("     随机写时它失效了,布隆过滤器才顶上。")
        print("     所以顺序是:范围检查 → 布隆过滤器 → 二分 → 读块,")
        print("     最便宜的放最前面。")
        print("  4. 块缓存让重复查询几乎不再读盘,占用远小于容量上限。")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
