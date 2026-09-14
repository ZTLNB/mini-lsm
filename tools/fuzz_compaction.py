"""随机压力测试:把引擎和内存里的字典模型对比。

单元测试只能覆盖我想到的情况。这里用随机操作序列去撞我没想到的:
反复改写同一个键、删了又写、写到一半归并、随时重启 ——
每一步都和"如果只用内存里的字典"应该得到的结果对照。

阶段 5 之后又多了一类对照:**快照**。每开一个快照,就顺手把当时的
模型抄一份;之后每次检查都要拿快照去比那份**冻结的**模型 ——
而不是当前模型。这一条同时覆盖了三件事:一致视图、文件 pin、
以及"最后一个快照关掉之后待删文件真的被回收了"。

固定种子,所以失败可以复现。
"""

import random
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.engine import LSMEngine
from mini_lsm.sstable import sstable_filename

#: 同时最多开几个快照。快照会 pin 住文件(磁盘回收不了),
#: 不设上限的话这一轮会攒下几十个快照,把测试变成磁盘消耗战。
MAX_LIVE_SNAPSHOTS = 4


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

    # 活着的快照,以及"开它的那一刻"的模型副本。
    # 用副本而不是当前模型来对照 —— 这正是快照的全部意义。
    live_snapshots: list[tuple] = []        # [(Snapshot, dict[bytes, bytes])]

    def close_all_snapshots() -> None:
        for snap, _expected in live_snapshots:
            snap.close()
        live_snapshots.clear()

    def close_and_tally() -> None:
        """关掉当前引擎,把它的工作量累加进来 —— 重启会换实例。"""
        nonlocal total_flushes, total_compactions, total_bloom_rejections
        close_all_snapshots()       # 引擎关闭会让快照失效,先干净地放开
        stats = db.stats()
        total_flushes += stats.flushes
        total_compactions += stats.compactions
        total_bloom_rejections += stats.bloom_rejections
        db.close()

    def check_cache() -> None:
        """阶段 4 新增的不变量:缓存不能超容量,也不能留着已删文件的块。

        "留着已删文件的块"不会读到脏数据(file_id 不复用),但内存是白占的 ——
        说明 compaction 之后忘了 evict_file。这种漏掉不会报错,只会悄悄涨内存。

        阶段 5 给"已删文件"补了个例外:被快照 pin 住的文件虽然离开了
        manifest,但**还活着**(快照还要读它),所以它的块留在缓存里是对的。
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
        # 仍然算"活着"的文件:manifest 引用的 + 被快照 pin 住的
        alive = db.manifest.file_ids() | set(db.pinned_file_ids)
        stale = {fid for fid, _ in cache._data} - alive
        if stale:
            raise AssertionError(
                f"seed={seed} 缓存里还有已被彻底删除的文件:{sorted(stale)}"
            )

    def check_snapshots(step: str) -> None:
        """阶段 5 的核心不变量:快照永远看到"它被创建那一刻"的状态。

        顺带把文件 pin 和延迟删除也一起钉住 —— 这两件事错了都不会
        立刻报错,只会让数据"悄悄消失"或者磁盘"悄悄涨满"。
        """
        if db.snapshot_count != len(live_snapshots):
            raise AssertionError(
                f"seed={seed} 步骤={step} 快照计数对不上:"
                f"引擎说 {db.snapshot_count},脚本记着 {len(live_snapshots)}"
            )

        for index, (snap, expected) in enumerate(live_snapshots):
            # 1. 全量扫描必须等于冻结的模型
            got = dict(snap.scan())
            if got != expected:
                diff_keys = set(got) ^ set(expected)
                diff = {k: (expected.get(k), got.get(k)) for k in diff_keys}
                raise AssertionError(
                    f"seed={seed} 步骤={step} 快照#{snap.snapshot_id}"
                    f"(第{index}个)全量扫描和冻结模型不一致:{diff}"
                )
            # 2. 点查也必须一致(走的是另一条代码路径:二分 + 布隆过滤器)
            for key in keys:
                actual = snap.get(key)
                want = expected.get(key)
                if actual != want:
                    raise AssertionError(
                        f"seed={seed} 步骤={step} 快照#{snap.snapshot_id} "
                        f"key={key!r}:期望 {want!r},实际 {actual!r}"
                    )
            # 3. 区间扫描只比区间内的那部分
            lo, hi = sorted(rng.sample(keys, 2))
            ranged = dict(snap.scan(lo, hi))
            want_range = {k: v for k, v in expected.items() if lo <= k < hi}
            if ranged != want_range:
                raise AssertionError(
                    f"seed={seed} 步骤={step} 快照#{snap.snapshot_id} "
                    f"区间 [{lo!r}, {hi!r}) 不一致"
                )
            # 4. pin 住的文件必须都还在磁盘上 —— 这是快照能读的前提
            for file_id in snap.file_ids:
                if not (tmp / sstable_filename(file_id)).exists():
                    raise AssertionError(
                        f"seed={seed} 步骤={step} 快照#{snap.snapshot_id} "
                        f"pin 住的 sst-{file_id} 已经不在磁盘上了"
                    )
            # 5. 快照自己那份 reader 映射要和 pin 的文件集合严格对齐。
            #    直接存引擎那个 dict 的话,compaction 一 pop 就"丢文件"了。
            if set(snap._readers) != set(snap.file_ids):
                raise AssertionError(
                    f"seed={seed} 步骤={step} 快照#{snap.snapshot_id} "
                    f"reader 映射和文件集合对不上"
                )

        # 6. 待删清单不能和 manifest 交叉 —— 那说明 manifest 还在引用它,
        #    却已经被判了死刑,下一步就是读到"文件不存在"
        overlap = set(db.pending_delete_files) & db.manifest.file_ids()
        if overlap:
            raise AssertionError(
                f"seed={seed} 步骤={step} 待删清单里居然还有 manifest 在用的文件:"
                f"{sorted(overlap)}"
            )
        # 7. 待删文件必须都还在磁盘上(还没删),而且都是"已经离开 manifest"的
        for file_id in db.pending_delete_files:
            if not (tmp / sstable_filename(file_id)).exists():
                raise AssertionError(
                    f"seed={seed} 步骤={step} 待删文件 sst-{file_id} 已经没了,"
                    f"但快照还指着它"
                )
        # 8. 没有快照就不该有待删文件 —— 说明"最后一个快照关闭时回收"漏了
        if not live_snapshots and db.pending_delete_files:
            raise AssertionError(
                f"seed={seed} 步骤={step} 已经没有任何快照,"
                f"却还有 {len(db.pending_delete_files)} 个文件待删:"
                f"{db.pending_delete_files}"
            )
        # 9. 没有快照就不该有 pin
        if not live_snapshots and db.pinned_file_ids:
            raise AssertionError(
                f"seed={seed} 步骤={step} 没有快照却还有文件被 pin:"
                f"{db.pinned_file_ids}"
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
        check_snapshots(step)

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

            if action < 0.60:
                key = rng.choice(keys)
                value = f"v{rng.randrange(1000):03d}".encode()
                db.put(key, value)
                model[key] = value
            elif action < 0.78:
                key = rng.choice(keys)
                db.delete(key)
                model.pop(key, None)
            elif action < 0.84:
                db.flush()
            elif action < 0.89:
                db.maybe_compact()
            elif action < 0.92:
                db.compact_all()
            elif action < 0.94:
                # 开一个快照,并把"此刻"的模型抄一份冻结起来。
                # 之后它只能看到这一份 —— 不管引擎那边怎么翻江倒海。
                if len(live_snapshots) < MAX_LIVE_SNAPSHOTS:
                    live_snapshots.append((db.snapshot(), dict(model)))
            elif action < 0.96:
                # 关掉最早的那个快照。最后一个关掉时,它 pin 的文件
                # 应该立刻被回收 —— check_snapshots 会验证这一点。
                if live_snapshots:
                    snap, _expected = live_snapshots.pop(0)
                    snap.close()
            elif action < 0.98:
                # 流式扫描的"定格"语义:迭代器一创建就固定了视图,
                # 之后再写数据也不能改变它看到的内容。
                cursor = db.scan()
                frozen = dict(model)
                key = rng.choice(keys)
                value = f"mut{rng.randrange(1000):03d}".encode()
                db.put(key, value)
                model[key] = value
                got = dict(cursor)
                if got != frozen:
                    diff = {k for k in set(got) ^ set(frozen)
                            if got.get(k) != frozen.get(k)}
                    raise AssertionError(
                        f"seed={seed} 步骤={step} 扫描迭代器没有定格:"
                        f"创建之后写入的改动被看到了,差异键 {sorted(diff)}"
                    )
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
        close_all_snapshots()
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
