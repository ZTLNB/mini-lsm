"""随机压力测试:把引擎和内存里的字典模型对比。

单元测试只能覆盖我想到的情况。这里用随机操作序列去撞我没想到的:
反复改写同一个键、删了又写、写到一半归并、随时重启 ——
每一步都和"如果只用内存里的字典"应该得到的结果对照。

固定种子,所以失败可以复现。
"""

import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.engine import LSMEngine


def run_round(seed: int, verbose: bool = False) -> None:
    rng = random.Random(seed)
    tmp = Path(tempfile.mkdtemp(prefix="lsm-fuzz-"))
    model: dict[bytes, bytes] = {}

    # 累计量:重启会换一个引擎实例,单看最后一个实例的统计会漏掉大半工作
    total_flushes = 0
    total_compactions = 0
    total_bloom_rejections = 0

    # 键空间刻意开得很小,好让"同一个键被反复改写/删除"成为常态 ——
    # 墓碑和版本覆盖的 bug 只有在键被重复触碰时才暴露得出来
    keys = [f"key{i:03d}".encode() for i in range(40)]

    # 过滤器参数也在不同轮次之间变一变:关掉过滤器、开得很大、开得很小,
    # 三条路径的正确性要求完全一样(只有快慢不同)
    bloom_bits = rng.choice([0, 4, 10, 16])
    # 缓存容量故意开得很小,逼出淘汰和"块比缓存还大"这两条路径
    cache_bytes = rng.choice([0, 512, 4096, 1 << 20])

    def open_engine(levels: int) -> LSMEngine:
        return LSMEngine(
            tmp,
            memtable_capacity=1024,         # 故意开小,逼出大量刷盘
            sstable_block_size=512,
            l0_compaction_trigger=3,
            level_size_budget=2048,
            level_size_factor=2,
            target_file_size=1024,
            num_levels=levels,
            bloom_bits_per_key=bloom_bits,
            block_cache_size=cache_bytes,
        )

    db = open_engine(rng.choice([2, 3, 4]))

    def close_and_tally() -> None:
        """关掉当前引擎,把它的工作量累加进来 —— 重启会换实例。"""
        nonlocal total_flushes, total_compactions, total_bloom_rejections
        stats = db.stats()
        total_flushes += stats.flushes
        total_compactions += stats.compactions
        total_bloom_rejections += stats.bloom_rejections
        db.close()

    def check_cache() -> None:
        """阶段 4 新增的不变量:缓存不能超容量,也不能留着已删文件的块。

        "留着已删文件的块"不会读到脏数据(file_id 不复用),但内存是白占的 ——
        说明 compaction 之后忘了 evict_file。这种漏掉不会报错,只会悄悄涨内存。
        """
        cache = db.block_cache
        stats = cache.stats()
        if stats.bytes > stats.capacity_bytes:
            raise AssertionError(
                f"seed={seed} 缓存超容量:{stats.bytes} > {stats.capacity_bytes}"
            )
        if stats.entries != len(cache):
            raise AssertionError(
                f"seed={seed} 缓存条数对不上:{stats.entries} != {len(cache)}"
            )
        live = db.manifest.file_ids()
        stale = {fid for fid, _ in cache._data} - live
        if stale:
            raise AssertionError(
                f"seed={seed} 缓存里还有已被删除的文件:{sorted(stale)}"
            )

    def check(step: str) -> None:
        for key in keys:
            expected = model.get(key)
            actual = db.get(key)
            if actual != expected:
                raise AssertionError(
                    f"seed={seed} 步骤={step} key={key!r}: "
                    f"期望 {expected!r},实际 {actual!r}"
                )
        # 扫描结果也必须和模型一致
        scanned = dict(db.scan())
        if scanned != model:
            only_engine = set(scanned) - set(model)
            only_model = set(model) - set(scanned)
            diff = {k: (model.get(k), scanned.get(k)) for k in only_engine | only_model}
            raise AssertionError(f"seed={seed} 步骤={step} 扫描结果不一致:{diff}")
        db.manifest.check_invariants()
        check_cache()

        # 过滤器**绝对不能有假阴性**。
        #
        # 注意这里必须拿"文件里**实际**有的键"去查,不能拿"键范围"去推 ——
        # 文件的范围是 [最小键, 最大键],范围中间是可以有空档的
        # (内存表里本来就没攒到那个键)。用范围去推会造出假的假阴性。
        for meta in db.manifest.files():
            reader = db._readers[meta.file_id]
            if not reader.has_bloom_filter:
                continue
            for key, _value in reader.iter_entries():
                if not reader.might_contain(key):
                    raise AssertionError(
                        f"seed={seed} 步骤={step} 过滤器假阴性:"
                        f"文件 {meta.file_id} 里明明有 {key!r},过滤器却说没有"
                    )

    try:
        for step in range(900):
            action = rng.random()

            if action < 0.62:
                key = rng.choice(keys)
                value = f"v{rng.randrange(1000):03d}".encode()
                db.put(key, value)
                model[key] = value
            elif action < 0.80:
                key = rng.choice(keys)
                db.delete(key)
                model.pop(key, None)
            elif action < 0.86:
                db.flush()
            elif action < 0.92:
                db.maybe_compact()
            elif action < 0.95:
                db.compact_all()
            else:
                # 重启:验证 manifest + WAL 恢复路径
                close_and_tally()
                db = open_engine(rng.choice([2, 3, 4]))

            if step % 37 == 0:
                check(f"step={step}")

        check("final")

        # 最后再来一次全量归并 + 重启,确认落盘状态也是对的
        db.compact_all()
        close_and_tally()
        db = LSMEngine(
            tmp, memtable_capacity=1024, num_levels=3,
            bloom_bits_per_key=bloom_bits, block_cache_size=cache_bytes,
        )
        check("after-final-restart")

        if verbose:
            final = db.stats()
            print(
                f"seed={seed:3d} OK  存活键={len(model):2d}  "
                f"文件={final.sstable_count}  刷盘={total_flushes}  "
                f"归并={total_compactions}  分层={final.level_files}  "
                f"过滤器={bloom_bits}bit/key 挡下={total_bloom_rejections}  "
                f"缓存={cache_bytes}B"
            )
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    for seed in range(count):
        run_round(seed, verbose=True)
    print(f"\n{count} 轮随机压力测试全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
