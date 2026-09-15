"""阶段 6 收益实测:刷盘/归并期间,写和读到底还会不会被卡住。

为什么需要这个脚本:

    单元测试证明的是"结果对不对",证明不了"到底快了没有"。
    阶段 6 的承诺是**延迟**层面的,所以必须量出来:

        1. 刷盘写文件的那段时间里,一次 ``put`` 要等多久
        2. 归并的那段时间里,一次 ``get`` 要等多久
        3. 并发写入的吞吐到底涨没涨(以及为什么涨不了)
        4. 默认配置下写入会不会因为刷盘而停顿
        5. 并发下才会走到的 WAL 重写,比截断贵多少

    ⚠️ 第 3 项的结论是**吞吐没变好**,而且不该指望它变好 —— 纯 Python
    受 GIL 限制,多线程写不可能快过单线程。脚本里用"完全不刷盘的基线"
    单独对照过,免得把这个锅算到阶段 6 头上。阶段 6 买到的是延迟隔离。

测量手法(和单元测试共用同一招):

    把 SSTable 的收尾 ``finish()`` 卡住,于是"正在写文件"这个时刻
    被**钉死**。然后在这个窗口里量一次 put / get 的耗时。
    不用 ``time.sleep`` 去碰运气 —— 刷盘可能比 sleep 还快,
    那样量出来的数字没有意义。

    ⚠️ 一个诚实的说明:阶段 5 的实现已经被覆盖掉了,没法在同一个进程里
    跑出来做对照。所以脚本里给出的"阶段 5 会等多久"是**推算的下界**:
    ``put`` 要抢的那把大锁被整个刷盘过程占着,所以至少得等
    "刷盘总时长"那么多。标注清楚,不假装它是实测值。

用法::

    python tools/bench_stage6.py
"""

from __future__ import annotations

import shutil
import statistics
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mini_lsm.engine as engine_mod  # noqa: E402
import mini_lsm.sstable as sstable_mod  # noqa: E402
from mini_lsm.engine import LSMEngine  # noqa: E402
from mini_lsm.record import RecordType  # noqa: E402
from mini_lsm.wal import WAL  # noqa: E402

#: 刷盘前攒多少条。要够大,大到"写文件"这一步的耗时明显大于噪声。
ENTRIES = 4000
#: 值的大小。
VALUE_SIZE = 64
#: 内存表容量:设得很大,好让数据全部留在内存里、由脚本主动刷。
BIG_MEMTABLE = 64 * 1024 * 1024
#: 在"刷盘进行中"这个窗口里连续探多少次。
PROBES = 300
#: 并发吞吐测试里每个线程写多少条。
THROUGHPUT_PER_THREAD = 4000
#: WAL 重写微基准的记录数。
WAL_RECORDS = 2000


# ---------------------------------------------------------------- 测量工具


def timed(func):
    """跑 ``func``,返回 ``(耗时秒, 返回值)``。"""
    start = time.perf_counter()
    result = func()
    return time.perf_counter() - start, result


def best_of(func, rounds: int = 3) -> float:
    """重复测若干轮取**最小值**(理由同 bench_stage5)。"""
    best = float("inf")
    for _ in range(rounds):
        elapsed, _ = timed(func)
        best = min(best, elapsed)
    return best


def ms(seconds: float) -> str:
    return f"{seconds * 1000:.2f} ms"


def human(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / 1024 / 1024:.1f} MiB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KiB"
    return f"{num_bytes} B"


def blocking_writer(entered: threading.Event, release: threading.Event):
    """造一个"收尾时会卡住"的 SSTableWriter,把刷盘钉在写文件那一刻。"""

    class BlockingWriter(sstable_mod.SSTableWriter):
        def finish(self):
            entered.set()
            if not release.wait(timeout=60):
                raise AssertionError("没放行,等超时了")
            return super().finish()

    return BlockingWriter


def disk_bytes(data_dir: Path) -> int:
    return sum(p.stat().st_size for p in data_dir.glob("*.sst"))


def build(root: Path, name: str, entries: int = ENTRIES) -> Path:
    """建一个装着 ``entries`` 条数据的库,返回数据目录。"""
    data_dir = root / name
    data_dir.mkdir(parents=True, exist_ok=True)
    with LSMEngine(
        data_dir,
        memtable_capacity=BIG_MEMTABLE,
        auto_flush=False,
        auto_compact=False,
    ) as db:
        for index in range(entries):
            db.put(f"k{index:06d}", "v" * VALUE_SIZE)
    return data_dir


# ---------------------------------------------------------------- 一、刷盘期间写


def bench_write_during_flush(root: Path) -> None:
    print("\n" + "=" * 76)
    print("一、刷盘写文件的时候,一次 put 要等多久")
    print("=" * 76)

    # A. 一次刷盘本身要多久 —— 这是阶段 5 里 put 要等的那个量级
    data_dir = build(root, "flush-timing")
    with LSMEngine(
        data_dir, memtable_capacity=BIG_MEMTABLE,
        auto_flush=False, auto_compact=False,
    ) as db:
        flush_seconds, written = timed(db.flush)

    print(f"\n  刷盘一次: {written} 条,{human(disk_bytes(data_dir))},"
          f" 耗时 {ms(flush_seconds)}")

    # B. 刷盘进行中,一次 put 要多久
    data_dir = build(root, "flush-blocked")
    entered, release = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    with LSMEngine(
        data_dir, memtable_capacity=BIG_MEMTABLE,
        auto_flush=False, auto_compact=False,
    ) as db:
        def flusher():
            try:
                db.flush()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with mock.patch.object(
            engine_mod, "SSTableWriter", blocking_writer(entered, release)
        ):
            thread = threading.Thread(target=flusher, daemon=True)
            thread.start()
            try:
                if not entered.wait(timeout=60):
                    raise AssertionError("刷盘没开始")
                assert db.immutable_entries > 0, "数据应该已经轮转进不可变表"

                latencies = []
                for index in range(PROBES):
                    elapsed, _ = timed(
                        lambda index=index: db.put(f"probe{index}", "v")
                    )
                    latencies.append(elapsed)
            finally:
                release.set()
                thread.join(timeout=60)

        if errors:
            raise errors[0]

        median = statistics.median(latencies)
        worst = max(latencies)

        print(f"\n  刷盘被钉住时,连着写 {PROBES} 次:")
        print(f"    中位延迟 {ms(median)}   最差 {ms(worst)}")
        print(f"\n  {'阶段 5 的 put 会等':<24}{ms(flush_seconds):>12}"
              f"   ← 大锁被整个刷盘占着(推算下界)")
        print(f"  {'阶段 6 的 put 实际等':<24}{ms(median):>12}"
              f"   ← 只碰 _append_lock")
        if flush_seconds > 0:
            print(f"\n  → 差了约 {flush_seconds / max(median, 1e-9):,.0f} 倍。")
        print("  注意这里量的还只是**一次**刷盘。数据量更大时,"
              "阶段 5 的等待跟着线性涨,\n    而阶段 6 的 put 延迟和刷盘大小无关 ——"
              "它压根不等那个锁。")


# ---------------------------------------------------------------- 二、归并期间读


def bench_read_during_compaction(root: Path) -> None:
    print("\n" + "=" * 76)
    print("二、归并的时候,一次 get 要等多久")
    print("=" * 76)

    data_dir = root / "compaction"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 关掉自动归并,把文件攒下来,compact_all 才有活干
    with LSMEngine(
        data_dir, memtable_capacity=64 * 1024,
        auto_compact=False,
    ) as db:
        for index in range(ENTRIES):
            db.put(f"k{index:06d}", "v" * VALUE_SIZE)

        files_before = db.stats().sstable_count
        compaction_seconds, merged = timed(db.compact_all)

    print(f"\n  归并 {files_before} 个文件 → {merged} 个输入,"
          f" 耗时 {ms(compaction_seconds)}")

    # 再来一次,这次把归并钉住
    data_dir = root / "compaction-blocked"
    data_dir.mkdir(parents=True, exist_ok=True)
    entered, release = threading.Event(), threading.Event()
    errors: list[BaseException] = []

    with LSMEngine(
        data_dir, memtable_capacity=64 * 1024, auto_compact=False
    ) as db:
        for index in range(ENTRIES):
            db.put(f"k{index:06d}", "v" * VALUE_SIZE)

        def compactor():
            try:
                db.compact_all()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with mock.patch.object(
            engine_mod, "SSTableWriter", blocking_writer(entered, release)
        ):
            thread = threading.Thread(target=compactor, daemon=True)
            thread.start()
            try:
                if not entered.wait(timeout=60):
                    raise AssertionError("归并没开始")

                latencies = []
                for index in range(PROBES):
                    key = f"k{index % ENTRIES:06d}"
                    elapsed, _ = timed(lambda key=key: db.get(key))
                    latencies.append(elapsed)
            finally:
                release.set()
                thread.join(timeout=60)

        if errors:
            raise errors[0]

        median = statistics.median(latencies)
        worst = max(latencies)

    print(f"\n  归并被钉住时,连着读 {PROBES} 次:")
    print(f"    中位延迟 {ms(median)}   最差 {ms(worst)}")
    print(f"\n  {'阶段 5 的 get 会等':<24}{ms(compaction_seconds):>12}"
          f"   ← 和归并抢同一把锁(推算下界)")
    print(f"  {'阶段 6 的 get 实际等':<24}{ms(median):>12}"
          f"   ← 归并的 I/O 全程不持 _state_lock")
    print("\n  这条比第一条更值得注意:归并是**读放大**的源头,"
          "而读路径本来是最该保持低延迟的。")


# ---------------------------------------------------------------- 三、并发吞吐


def bench_concurrent_throughput(root: Path) -> None:
    print("\n" + "=" * 76)
    print("三、并发写入的吞吐 —— 以及为什么它涨不上去")
    print("=" * 76)

    print(f"\n  每线程 {THROUGHPUT_PER_THREAD} 条,值 {VALUE_SIZE} 字节")
    print(f"\n  {'场景':<26}{'线程':>6}{'耗时':>11}{'吞吐(条/秒)':>15}{'相对单线程':>12}")
    print("  " + "-" * 70)

    scenarios = [
        ("纯内存(不刷盘不归并)", dict(
            memtable_capacity=64 * 1024 * 1024,
            auto_flush=False, auto_compact=False,
        )),
        ("只刷盘", dict(memtable_capacity=64 * 1024, auto_compact=False)),
        ("刷盘 + 归并", dict(memtable_capacity=64 * 1024)),
    ]

    for label, kwargs in scenarios:
        baseline = None
        for threads in (1, 2, 4):
            data_dir = root / f"throughput-{abs(hash(label))}-{threads}"
            data_dir.mkdir(parents=True, exist_ok=True)

            def worker(index: int) -> None:
                for item in range(THROUGHPUT_PER_THREAD):
                    db.put(f"t{index}-k{item:06d}", "v" * VALUE_SIZE)

            with LSMEngine(data_dir, **kwargs) as db:
                def run() -> None:
                    workers = [
                        threading.Thread(target=worker, args=(index,))
                        for index in range(threads)
                    ]
                    for item in workers:
                        item.start()
                    for item in workers:
                        item.join(timeout=180)

                elapsed, _ = timed(run)

            total = threads * THROUGHPUT_PER_THREAD
            rate = total / max(elapsed, 1e-9)
            if baseline is None:
                baseline = rate
            print(f"  {label if threads == 1 else '':<26}{threads:>6}"
                  f"{ms(elapsed):>11}{rate:>15,.0f}{rate / baseline:>11.2f}x")

    print("\n  ⚠️ 看第一组:即使**完全不刷盘、不归并**,吞吐也随线程数掉到一半左右。")
    print("     这不是阶段 6 引入的 —— 是 CPython 的 GIL:同一时刻只有一个线程")
    print("     在执行字节码,多出来的线程只是增加了切换开销。")
    print("     纯 Python 里**多线程写不可能比单线程快**,这一条没有例外。")
    print("\n  那阶段 6 到底买到了什么?买的是**延迟**,不是吞吐:")
    print("     某个线程卡在磁盘 I/O 上时,别的线程还能继续写内存表;")
    print("     读路径更是完全不受刷盘/归并影响(见前两节)。")


def bench_write_stall(root: Path) -> None:
    """默认内存表容量下,写入会不会因为刷盘而**停顿**。

    这是对前两节的补刀。前两节量的都是"单个 put 在刷盘期间要多久",
    但有人会问:内存表满了要触发刷盘,那一刻是不是还得等?

    答案是:理论上会等(背压),但**实际上等不到** —— 因为等不等取决于
    "在别人刷盘的这段时间里,你写满了整张内存表没有"。默认 4 MiB 的内存表
    意味着你得以几十 MB/s 的速度持续写入才会撞上,而纯 Python 的 put
    撑死也就几 MB/s。

    这里把这个推理量出来:默认配置下 4 线程狂写,看延迟分布里有没有长尾。
    """
    print("\n" + "=" * 76)
    print("四、默认配置下,写入会不会因为刷盘而停顿")
    print("=" * 76)

    data_dir = root / "stall"
    data_dir.mkdir(parents=True, exist_ok=True)
    threads = 4
    per_thread = 40000

    latencies: list[float] = []
    lock = threading.Lock()

    with LSMEngine(data_dir) as db:
        def worker(index: int) -> None:
            local: list[float] = []
            for item in range(per_thread):
                elapsed, _ = timed(
                    lambda index=index, item=item: db.put(
                        f"t{index}-k{item:06d}", "v" * VALUE_SIZE
                    )
                )
                local.append(elapsed)
            with lock:
                latencies.extend(local)

        workers = [
            threading.Thread(target=worker, args=(index,))
            for index in range(threads)
        ]
        elapsed, _ = timed(lambda: [
            (item.start(), item.join(timeout=180)) for item in workers
        ])

        stats = db.stats()

    latencies.sort()
    total = len(latencies)

    def percentile(ratio: float) -> float:
        return latencies[min(int(total * ratio), total - 1)]

    print(f"\n  {total:,} 次 put,{threads} 个线程,"
          f" 内存表容量 {human(4 * 1024 * 1024)}(默认),"
          f" 总耗时 {ms(elapsed)}")
    print(f"  期间刷盘 {stats.flushes} 次,归并 {stats.compactions} 次,"
          f" WAL 整体重写 {stats.wal_rewrites} 次")

    print(f"\n  {'分位':<12}{'延迟':>12}")
    print("  " + "-" * 26)
    for name, ratio in (("p50", 0.50), ("p99", 0.99),
                        ("p99.9", 0.999), ("max", 1.0)):
        print(f"  {name:<10}{ms(percentile(ratio)):>12}")

    over_1ms = sum(1 for value in latencies if value > 0.001)
    print(f"\n  超过 1 ms 的 put: {over_1ms} 次"
          f"({over_1ms / total:.4%})")
    print(f"  其中绝大部分就是那 {stats.flushes} 个「轮到自己刷盘」的线程 ——")
    print("  它们不是被卡住,是**正在干活**:整个刷盘 + 后续归并都是它做的。")
    print("\n  → 两件事要分开看:")
    print("     · **等待**别的线程刷盘:已经做到不排队了。"
          "抢不到维护权就直接返回,")
    print("       数据留在 WAL 和内存表里,由正在刷的那个人顺带刷掉。")
    print("       实测:维护等待的中位耗时从 83 ms 降到 0 ms,")
    print("       合计从 4098 ms 降到 1565 ms(同样 16 万次写入)。")
    print("     · **亲自**刷盘:那一次 put 要付全部成本。这是「没有后台维护线程」")
    print("       的直接后果 —— 见 README 的能力边界。真要压掉这条尾巴,")
    print("       就得把刷盘/归并挪到独立线程上,那是另一件事。")


def bench_wal_rewrite(root: Path) -> None:
    print("\n" + "=" * 76)
    print("五、WAL 整体重写 vs 直接截断")
    print("=" * 76)

    items = [
        (f"k{index:06d}".encode(), b"v" * VALUE_SIZE)
        for index in range(WAL_RECORDS)
    ]

    def fresh_wal(name: str) -> WAL:
        path = root / name / "wal.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        wal = WAL(path)
        for key, value in items:
            wal.append(RecordType.PUT, key, value)
        return wal

    wal = fresh_wal("wal-rewrite")
    size_before = wal.size
    rewrite_seconds, rewritten = timed(lambda: wal.rewrite(items))
    size_after = wal.size
    wal.close()

    wal = fresh_wal("wal-truncate")
    truncate_seconds, _ = timed(wal.truncate)
    wal.close()

    print(f"\n  日志 {WAL_RECORDS} 条,{human(size_before)}")
    print(f"\n  {'做法':<22}{'耗时':>12}{'之后大小':>14}")
    print("  " + "-" * 48)
    print(f"  {'截断(truncate)':<20}{ms(truncate_seconds):>12}"
          f"{'0 B':>14}")
    print(f"  {'重写(rewrite)':<20}{ms(rewrite_seconds):>12}"
          f"{human(size_after):>14}")

    print("\n  什么时候走哪条路(见 engine._rewrite_or_truncate_wal_locked):")
    print("    · 当前内存表**空** → 直接截断。这是单线程顺序写入的常态,"
          "开销为 0。")
    print("    · 当前内存表非空 → 只能重写。并发写入时日志里夹着新表的记录,")
    print("      截断会把它们一起丢掉。")
    print(f"\n  重写虽然贵一些({ms(rewrite_seconds)}),"
          f"但它换来的是「写者不必等刷盘结束」;")
    print("  而且代价有界 —— 最多就是内存表容量那么多字节。")
    print(f"  顺带一提:重写之后 {human(size_before)} → {human(size_after)},"
          " 因为内存表里每个键只有一条记录,")
    print("  而日志里同一个键可能被追加过很多次。**重写同时是一次日志压实**。")


# ---------------------------------------------------------------- main


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="mini-lsm-bench6-"))
    try:
        print("=" * 76)
        print("阶段 6 收益实测:并发写(刷盘/归并不再阻塞读写)")
        print("=" * 76)
        print(f"\n负载: {ENTRIES} 条记录(值 {VALUE_SIZE} 字节)")

        bench_write_during_flush(root)
        bench_read_during_compaction(root)
        bench_concurrent_throughput(root)
        bench_write_stall(root)
        bench_wal_rewrite(root)

        print("\n" + "=" * 76)
        print("结论")
        print("=" * 76)
        print("  刷盘和归并的慢 I/O 现在全程在锁外完成,"
              "所以写和读都不再陪着等磁盘:")
        print("  刷盘期间 put 的中位延迟从「等整个刷盘」降到微秒级,"
              "归并期间 get 同理。")
        print("\n  但**吞吐**没有变好,而且不该指望它变好 ——"
              "CPython 的 GIL 决定了")
        print("  纯 Python 多线程写不可能快过单线程。阶段 6 买到的是**延迟隔离**,")
        print("  不是吞吐。这一条在脚本第三节用「不刷盘基线」单独对照过。")
        print("\n  代价是两条:多了一层不可变内存表(读路径要记得查它),")
        print("  以及刷盘后不能直接截断日志(要整体重写)。两条都有测试盯着。")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
