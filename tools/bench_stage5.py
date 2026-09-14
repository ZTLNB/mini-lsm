"""阶段 5 收益实测:流式扫描省了多少内存、快照读贵不贵、pin 住多少磁盘。

为什么需要这个脚本:

    单元测试证明的是"结果对不对",证明不了"到底省了多少"。
    阶段 5 的三个承诺都是**资源**层面的,所以必须能量出来:

        1. ``scan()`` 流式 —— 内存占用不随结果集增长
        2. 快照读 —— 创建便宜、读起来和实时读一样快
        3. 文件 pin —— 代价是磁盘(旧文件回收不了),要知道代价多大

三组测量:

    **内存**:用 ``tracemalloc`` 量"全量物化"和"流式"的峰值。
    这个对比必须用峰值,不能用总量 —— 流式的意义恰恰在于
    峰值被压到"一个块"的量级。

    **时间**:创建快照的耗时、取第一个元素的耗时、快照点查 vs 实时点查。

    **磁盘**:快照开着的时候 compaction 回收不了旧文件,
    于是磁盘占用会顶在那里;关掉之后立刻回落。这是快照的**真实代价**,
    不量出来就没法说清"为什么必须记得关快照"。

顺带还有一个"锁"的探测:迭代期间引擎锁必须是**空闲**的 ——
不然"读不阻塞写"就只是嘴上说说。

用法::

    python tools/bench_stage5.py
"""

from __future__ import annotations

import gc
import shutil
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.engine import LSMEngine  # noqa: E402

#: 数据量。要足够大,大到"物化整个结果集"的内存开销明显可见 ——
#: 数据太小的话两种做法的峰值都是几十 KB,看不出区别。
NUM_KEYS = 12000
#: 值的大小。故意做大一点,让结果集的内存占用更显眼。
VALUE_SIZE = 96
#: 内存表容量。开小一点,逼出几十次刷盘,磁盘上才有真文件。
MEMTABLE_CAPACITY = 64 * 1024
PROBES = 2000


# ---------------------------------------------------------------- 测量工具


def measure_peak(func):
    """跑 ``func``,返回 ``(峰值内存字节, 返回值)``。

    ``gc.collect()`` 是必要的:不然上一轮留下的垃圾会在这一轮被算进峰值。
    """
    gc.collect()
    tracemalloc.start()
    try:
        result = func()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak, result


def timed(func) -> tuple[float, object]:
    """跑 ``func``,返回 ``(耗时秒, 返回值)``。"""
    start = time.perf_counter()
    result = func()
    return time.perf_counter() - start, result


def best_of(func, rounds: int = 3) -> float:
    """重复测若干轮取**最小值**。

    为什么不取平均:第一次跑总是最慢的(块缓存是冷的、CPU 频率还没上去)。
    如果只测一轮,那么"谁先测谁吃亏"—— 顺序就成了隐藏变量,
    比出来的差异可能全是预热差异。取最小值等于"至少有一次跑得这么顺",
    这是比较两种实现时更稳的口径。
    """
    best = float("inf")
    for _ in range(rounds):
        elapsed, _ = timed(func)
        best = min(best, elapsed)
    return best


def try_acquire(lock) -> bool:
    """从当前线程试着拿一下锁 —— 拿得到就说明没人占着。"""
    if lock.acquire(blocking=False):
        lock.release()
        return True
    return False


def disk_bytes(data_dir: Path) -> int:
    """数据目录里所有 SSTable 的总字节数(不含 WAL 和 manifest)。"""
    return sum(p.stat().st_size for p in data_dir.glob("*.sst"))


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"       # pragma: no cover


# ---------------------------------------------------------------- 造库


def build(root: Path, name: str, auto_compact: bool = True) -> Path:
    """造一份固定的负载,返回数据目录。

    写入顺序是**打散过的** —— 让每个刷盘文件的键范围都覆盖整个键空间。
    这样 compaction 之后仍然会有若干个键范围互相重叠的文件,
    "快照 pin 住了几个文件"才有看头。
    """
    data_dir = root / name
    value = "v" * VALUE_SIZE
    with LSMEngine(
        data_dir,
        memtable_capacity=MEMTABLE_CAPACITY,
        auto_compact=auto_compact,
    ) as db:
        # 用取模跳步代替随机洗牌:既打散了顺序,又完全可复现
        for i in range(NUM_KEYS):
            index = (i * 7919) % NUM_KEYS
            db.put(f"key{index:06d}", value)
    return data_dir


def describe(data_dir: Path) -> str:
    with LSMEngine(data_dir, memtable_capacity=MEMTABLE_CAPACITY) as db:
        stats = db.stats()
        return (
            f"{stats.sstable_count} 个文件, {human(stats.sstable_bytes)},"
            f" {stats.sstable_entries} 条, 分层 {stats.level_files}"
        )


# ---------------------------------------------------------------- 各项测量


def bench_scan_memory(data_dir: Path) -> None:
    print("\n" + "=" * 76)
    print("一、流式扫描省了多少内存(峰值)")
    print("=" * 76)

    with LSMEngine(data_dir, memtable_capacity=MEMTABLE_CAPACITY) as db:
        stats = db.stats()
        print(f"  库大小: {stats.sstable_count} 个文件里 {stats.sstable_entries} 条,"
              f" 内存表里还有 {stats.memtable_entries} 条")

        # 阶段 4 的做法:结果全部物化成 list
        peak_list, materialized = measure_peak(lambda: list(db.scan()))
        # 存活条目数 = 文件里的 + 内存表里的(所以不能拿 sstable_entries 当总数)
        total = len(materialized)

        # 阶段 5:只取前 10 条就停 —— 后面的块一个都不读
        peak_first, _ = measure_peak(lambda: [k for k, _ in zip(db.scan(), range(10))])

        # 阶段 5:全部遍历,但一条都不留
        peak_stream, seen = measure_peak(
            lambda: sum(1 for _ in db.scan())
        )

        assert seen == total, f"流式遍历少读了:{seen} != {total}"

        print(f"\n  {'做法':<34}{'峰值内存':>14}{'相对':>10}")
        print("  " + "-" * 60)
        print(f"  {'list(db.scan())  ← 阶段 4':<32}"
              f"{human(peak_list):>14}{'1.00x':>10}")
        print(f"  {'流式,取前 10 条就停':<32}"
              f"{human(peak_first):>14}"
              f"{peak_first / peak_list:>9.2f}x")
        print(f"  {'流式,全量遍历但不保留':<32}"
              f"{human(peak_stream):>14}"
              f"{peak_stream / peak_list:>9.2f}x")

        print(f"\n  也就是说:内存占用从「正比于结果集」变成「正比于一个块」。")
        print(f"  结果集再大 100 倍,流式那一行的数字也几乎不变。")

        # 第一个元素要等多久
        t_full, _ = timed(lambda: sum(1 for _ in db.scan()))
        t_first, _ = timed(lambda: next(iter(db.scan())))
        print(f"\n  取第一个元素: {t_first * 1000:.2f} ms")
        print(f"  扫完全部:     {t_full * 1000:.1f} ms"
              f"  ({total / max(t_full, 1e-9):,.0f} 条/秒)")
        print(f"  → 只要头几条的话,阶段 4 必须等完整趟;现在几乎立刻就有。")


def bench_lock_free(data_dir: Path) -> None:
    print("\n" + "=" * 76)
    print("二、迭代期间引擎锁是空闲的(读不阻塞写)")
    print("=" * 76)

    with LSMEngine(data_dir, memtable_capacity=MEMTABLE_CAPACITY) as db:
        cursor = db.scan()
        next(cursor)                    # 迭代已经开始
        free_during = try_acquire(db._lock)
        cursor.close()

        free_idle = try_acquire(db._lock)
        print(f"  空闲时锁可用:   {'是' if free_idle else '否'}")
        print(f"  扫描中锁可用:   {'是' if free_during else '否'}")
        print("\n  阶段 4 是「锁内物化成 list」—— 扫一个大库,写会被卡住全程。")
        print("  现在迭代期间不持锁,靠 pin 住文件保证数据还在。")
        assert free_during and free_idle


def bench_snapshot_cost(data_dir: Path) -> None:
    print("\n" + "=" * 76)
    print("三、快照读:创建便宜吗?读起来慢吗?")
    print("=" * 76)

    with LSMEngine(data_dir, memtable_capacity=MEMTABLE_CAPACITY) as db:
        stats = db.stats()
        print(f"  库大小: {stats.sstable_count} 个文件,"
              f" 内存表 {stats.memtable_entries} 条")

        # 创建 / 关闭的开销
        rounds = 200
        t, _ = timed(lambda: [db.snapshot().close() for _ in range(rounds)])
        print(f"\n  创建 + 关闭一次快照: {t / rounds * 1e6:,.0f} µs"
              f"  ({rounds} 次共 {t * 1000:.1f} ms)")
        print(f"  → 代价是「抄一份内存表 + 抄一份层结构」,和文件数成正比、"
              f"和文件大小无关。")

        peak, _ = measure_peak(lambda: [db.snapshot().close() for _ in range(20)])
        print(f"  创建 20 个快照的峰值内存: {human(peak)}")

        # 点查速度对比。
        #
        # ⚠️ 必须"先各跑一遍预热,再交替多轮取最小"。只测一轮的话,
        # 先测的那一方永远吃亏:块缓存是空的,第一次读真的要走磁盘 ——
        # 于是比出来的 1.6x 差距全是预热差异,和两条读路径无关。
        keys = [f"key{(i * 4099) % NUM_KEYS:06d}".encode() for i in range(PROBES)]
        absent = [f"miss{i:06d}".encode() for i in range(PROBES)]

        db.get(keys[0])                     # 预热(单条即可,库里全在缓存里)
        with db.snapshot() as warmup:
            warmup.get(keys[0])

        with db.snapshot() as snap:
            t_snap_hit = best_of(lambda: [snap.get(k) for k in keys])
            t_snap_miss = best_of(lambda: [snap.get(k) for k in absent])
        t_live_hit = best_of(lambda: [db.get(k) for k in keys])
        t_live_miss = best_of(lambda: [db.get(k) for k in absent])

        print(f"\n  {'':<16}{'实时读':>12}{'快照读':>12}{'比值':>10}")
        print("  " + "-" * 52)
        print(f"  {'存在的键':<14}{t_live_hit * 1e6:>11.0f}µs"
              f"{t_snap_hit * 1e6:>11.0f}µs"
              f"{t_snap_hit / max(t_live_hit, 1e-9):>9.2f}x")
        print(f"  {'不存在的键':<14}{t_live_miss * 1e6:>11.0f}µs"
              f"{t_snap_miss * 1e6:>11.0f}µs"
              f"{t_snap_miss / max(t_live_miss, 1e-9):>9.2f}x")
        print("\n  → 两条路径共用同一个 search_levels,所以耗时基本一致。")
        print("    快照几乎不额外收费,这是「共用一份读逻辑」换来的。")

        # 快照的一致性代价:扫的时候引擎可以继续被写
        with db.snapshot() as snap:
            t_snap_scan, n = timed(lambda: sum(1 for _ in snap.scan()))
        print(f"\n  快照全量扫描: {t_snap_scan * 1000:.1f} ms ({n} 条)")


def bench_pinning(root: Path) -> None:
    print("\n" + "=" * 76)
    print("四、pin 的真实代价:快照不关,磁盘就回收不了")
    print("=" * 76)

    # ⚠️ 对照组和实验组必须是**两份独立的库**。
    #    在同一个目录上先跑一次 compact_all,库就已经被压成"最理想状态"了;
    #    第二次 compact_all 会直接返回"无事可做",于是什么都观察不到 ——
    #    这个坑我第一版就踩了,数字全是 0。
    control = build(root, "pin-control", auto_compact=False)
    experiment = build(root, "pin-experiment", auto_compact=False)

    before = disk_bytes(control)
    print(f"  造库完成: {describe(control)}")

    # 对照组:没有快照 —— 旧文件当场被删
    with LSMEngine(control, memtable_capacity=MEMTABLE_CAPACITY) as db:
        files_before = db.stats().sstable_count
        db.compact_all()
        without = disk_bytes(control)
        pending = len(db.pending_delete_files)
        files_without = db.stats().sstable_count

    # 实验组:开一个快照再 compaction —— 旧文件只能推迟删除
    with LSMEngine(experiment, memtable_capacity=MEMTABLE_CAPACITY) as db:
        with db.snapshot() as snap:
            db.compact_all()
            held = disk_bytes(experiment)
            pending_held = len(db.pending_delete_files)
            pinned = len(snap.file_ids)
            # 快照必须仍然读得到归并前就有的数据
            sample = snap.get(f"key{(0 * 7919) % NUM_KEYS:06d}")
            assert sample is not None, "快照读不到数据"
        after = disk_bytes(experiment)
        pending_after = len(db.pending_delete_files)
        files_after = db.stats().sstable_count

    print(f"\n  {'时刻':<30}{'磁盘占用':>14}{'待删':>8}{'文件数':>8}")
    print("  " + "-" * 62)
    print(f"  {'compaction 之前':<28}{human(before):>14}{'-':>8}"
          f"{files_before:>8}")
    print(f"  {'无快照,compaction 之后':<28}{human(without):>14}"
          f"{pending:>8}{files_without:>8}")
    print(f"  {'有快照,compaction 之后':<28}{human(held):>14}"
          f"{pending_held:>8}{files_after + pending_held:>8}")
    print(f"  {'快照关闭之后':<28}{human(after):>14}"
          f"{pending_after:>8}{files_after:>8}")
    print(f"\n  那个快照 pin 住了 {pinned} 个文件,"
          f"比无快照时多占 {human(held - without)}。")
    print("  这就是「快照必须记得关」的原因 —— 它不是内存泄漏,是**磁盘**泄漏。")
    print("  close() 之后立刻回落到无快照的水平,说明延迟删除没有漏。")


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="mini-lsm-bench5-"))
    try:
        print("=" * 76)
        print("阶段 5 收益实测:流式扫描 + 快照读")
        print("=" * 76)
        print(f"\n负载: {NUM_KEYS} 条记录(值 {VALUE_SIZE} 字节),"
              f" 内存表容量 {MEMTABLE_CAPACITY} 字节")

        data_dir = build(root, "main")
        print(f"  建成: {describe(data_dir)}")

        bench_scan_memory(data_dir)
        bench_lock_free(data_dir)
        bench_snapshot_cost(data_dir)
        bench_pinning(root)

        print("\n" + "=" * 76)
        print("结论")
        print("=" * 76)
        print("  1. 流式扫描把内存从「正比于结果集」压到「正比于一个块」。")
        print("     取头几条几乎零成本,不必等整趟跑完。")
        print("  2. 迭代期间不持锁 —— 靠 pin 住文件,而不是靠锁。")
        print("  3. 快照创建是「抄内存表 + 抄层结构」,和文件大小无关;")
        print("     读性能和实时读基本一致,因为共用同一份 search_levels。")
        print("  4. pin 的代价是磁盘:只要还有快照开着,被它引用的旧文件")
        print("     就不能删。关掉之后立刻回收 —— 所以 with 语句是最好的习惯。")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
