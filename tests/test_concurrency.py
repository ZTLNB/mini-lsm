"""并发写测试(阶段 6)。

阶段 5 结束时,写和写之间、写和 compaction 之间是互斥的 —— 一把大锁
把"改一个字节的内存"和"写几 MB 的文件"保护在同一个临界区里。
阶段 6 把锁拆开,并引入不可变内存表。

这个文件盯住三件事:

    1. **正确性**:并发写入之后数据一条不少,而且**重启后仍然一致**。
       这是 ``_append_lock`` 存在的全部理由 —— 日志里的顺序必须和内存表
       里的顺序一致,否则重放出来的状态和崩溃前不一样。这个 bug 只在
       "恰好崩在那一次"时暴露,平时完全看不出来。
    2. **不阻塞**:刷盘/归并的慢 I/O 期间,写入和读取都能照常进行。
    3. **可见性**:刷盘期间数据在"不可变内存表"里 —— 读路径和快照
       都必须看得到它。漏掉这一层就是"偶发查不到刚写的数据"。

并发测试最容易写成"大部分时候能过"。所以这里的用例尽量不用 sleep
去碰运气,而是用 ``threading.Event`` 把时序钉死(见 ``_blocking_writer``)。
"""

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mini_lsm.engine as engine_mod  # noqa: E402
import mini_lsm.sstable as sstable_mod  # noqa: E402
from mini_lsm.engine import LSMEngine  # noqa: E402


def _blocking_writer(entered: threading.Event, release: threading.Event):
    """造一个"收尾时会卡住"的 SSTableWriter。

    为什么不用 ``time.sleep`` 去制造"刷盘进行中"这个窗口:刷盘可能比
    sleep 还快,测试就变成碰运气。这里让 ``finish()`` 一进来就发信号,
    然后一直等到测试放行 —— 时序是确定的,不是概率的。
    """

    class BlockingWriter(sstable_mod.SSTableWriter):
        def finish(self):
            entered.set()
            if not release.wait(timeout=30):
                raise AssertionError("测试没放行,等超时了")
            return super().finish()

    return BlockingWriter


class ConcurrencyTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def open_engine(self, **kwargs) -> LSMEngine:
        engine = LSMEngine(self.dir, **kwargs)
        self.addCleanup(engine.close)
        return engine

    def run_threads(self, count: int, target) -> None:
        """并发跑 ``count`` 个线程,任何一个出错都让测试失败。

        子线程里的异常不会自动冒泡到测试框架,不收集的话测试会"绿着挂掉"。
        """
        errors: list[BaseException] = []

        def wrapper(index: int) -> None:
            try:
                target(index)
            except BaseException as exc:  # noqa: BLE001 - 要原样带回主线程
                errors.append(exc)

        threads = [
            threading.Thread(target=wrapper, args=(index,))
            for index in range(count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        stuck = [thread for thread in threads if thread.is_alive()]
        self.assertFalse(stuck, f"有 {len(stuck)} 个线程没结束 —— 疑似死锁")
        if errors:
            raise errors[0]


class TestConcurrentWrites(ConcurrencyTestCase):
    """多线程写入的正确性。"""

    def test_all_writes_land(self):
        """每个线程写自己的一段键,写完全部读得到。

        内存表容量故意设得很小(512 字节),这样 6 个线程会频繁撞上刷盘 ——
        轮转、不可变表、WAL 重写这些路径都会真的被走到。
        """
        threads, per_thread = 6, 120
        pairs = [
            (f"t{t}-k{i:04d}", f"v{t}-{i}")
            for t in range(threads)
            for i in range(per_thread)
        ]

        with self.open_engine(memtable_capacity=512) as db:
            def worker(index):
                for key, value in pairs[index::threads]:
                    db.put(key, value)

            self.run_threads(threads, worker)

            for key, value in pairs:
                self.assertEqual(db.get_str(key), value, key)

    def test_all_writes_survive_restart(self):
        """并发写完就关掉重开 —— 一条都不能少。

        这是崩溃恢复的主用例。WAL 重写(并发下才会走的那条路)如果
        漏掉或多算了记录,这里就会露出来。
        """
        threads, per_thread = 6, 100
        pairs = [
            (f"t{t}-k{i:04d}", f"v{t}-{i}")
            for t in range(threads)
            for i in range(per_thread)
        ]

        with self.open_engine(memtable_capacity=512) as db:
            def worker(index):
                for key, value in pairs[index::threads]:
                    db.put(key, value)

            self.run_threads(threads, worker)

        with self.open_engine() as db:
            for key, value in pairs:
                self.assertEqual(db.get_str(key), value, key)

    def test_same_key_final_value_survives_restart(self):
        """并发覆盖**同一个键**时,内存表的顺序必须和 WAL 的顺序一致。

        这是 ``_append_lock`` 存在的全部理由。假如不把它俩圈在一起,
        可能 A 先追加日志、B 先改内存表 —— 于是重放日志得到的是 A 的值,
        而崩溃前内存里是 B 的值。差异只在"恰好崩在那一次"时暴露。

        这里用"关掉再重开"来模拟崩溃后重放:重开读到的值必须和关闭前一致。
        """
        with self.open_engine(memtable_capacity=512) as db:
            def worker(index):
                for round_index in range(40):
                    db.put("hot", f"t{index}-r{round_index}")

            self.run_threads(6, worker)
            before = db.get_str("hot")

        with self.open_engine() as db:
            self.assertEqual(db.get_str("hot"), before)
            # 顺带确认它确实是某个线程写过的值,而不是被写坏了的中间态
            self.assertRegex(before, r"^t[0-5]-r\d+$")

    def test_concurrent_deletes(self):
        """并发删除。墓碑也要经得起重启 —— 删掉的键不能在重放后复活。"""
        keys = [f"k{i:04d}" for i in range(300)]
        threads = 6

        with self.open_engine(memtable_capacity=512) as db:
            for key in keys:
                db.put(key, "v")

            def worker(index):
                for key in keys[index::threads]:
                    db.delete(key)

            self.run_threads(threads, worker)

            for key in keys:
                self.assertIsNone(db.get(key), key)

        with self.open_engine() as db:
            for key in keys:
                self.assertIsNone(db.get(key), key)

    def test_flush_and_write_concurrently(self):
        """刷盘和写入同时跑,谁也不该丢谁。

        关掉自动刷盘,让一半线程主动调 ``flush()``、另一半猛写 ——
        两边的临界区在旧设计里是同一把锁,现在被拆开了。
        """
        threads = 8
        writers = [index for index in range(threads) if index % 2]

        with self.open_engine(
            memtable_capacity=1 << 20, auto_flush=False
        ) as db:
            def worker(index):
                if index % 2:
                    for i in range(30):
                        db.put(f"t{index}-k{i}", "v")
                else:
                    for _ in range(10):
                        db.flush()

            self.run_threads(threads, worker)
            db.flush()

            for index in writers:
                for i in range(30):
                    self.assertEqual(
                        db.get_str(f"t{index}-k{i}"), "v",
                        f"t{index}-k{i}",
                    )


class TestMaintenanceDoesNotBlock(ConcurrencyTestCase):
    """刷盘/归并的慢 I/O 期间,读写必须照常。"""

    def test_write_proceeds_during_flush(self):
        """刷盘写 SSTable 的时候,写入不该被挡住。"""
        db = self.open_engine(memtable_capacity=1 << 20)
        db.put("before", "1")

        entered, release = threading.Event(), threading.Event()
        errors: list[BaseException] = []

        def flusher():
            try:
                db.flush()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with mock.patch.object(
            engine_mod, "SSTableWriter", _blocking_writer(entered, release)
        ):
            thread = threading.Thread(target=flusher, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(timeout=30), "刷盘没开始")

                # 此刻:数据已经轮转进不可变表,SSTable 还没写完
                self.assertEqual(db.immutable_entries, 1)

                # 写入必须立刻成功 —— 这就是阶段 6 的核心收益。
                # 旧设计里 flush() 占着大锁写文件,这个 put 会一直卡到刷盘结束。
                db.put("during", "2")
                self.assertEqual(db.get_str("during"), "2")
            finally:
                release.set()
                thread.join(timeout=30)

        self.assertFalse(errors, errors)
        self.assertEqual(db.immutable_entries, 0)
        self.assertEqual(db.get_str("before"), "1")
        self.assertEqual(db.get_str("during"), "2")

    def test_read_sees_immutable_memtable(self):
        """刷盘期间,被轮转走的那条数据必须还读得到。

        它既不在活动内存表里(已经被换掉),也还不在 SSTable 里 ——
        唯一能读到它的地方就是不可变内存表。漏掉这一层,
        表现就是"刷盘那一瞬间偶发查不到刚写的数据"。
        """
        db = self.open_engine(memtable_capacity=1 << 20)
        db.put("before", "1")

        entered, release = threading.Event(), threading.Event()

        def flusher():
            db.flush()

        with mock.patch.object(
            engine_mod, "SSTableWriter", _blocking_writer(entered, release)
        ):
            thread = threading.Thread(target=flusher, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(timeout=30), "刷盘没开始")
                self.assertEqual(db.get_str("before"), "1")
                self.assertTrue(db.contains("before"))
                self.assertEqual(db.get_entry("before"), (True, b"1"))
            finally:
                release.set()
                thread.join(timeout=30)

    def test_snapshot_sees_immutable_memtable(self):
        """快照也要看得到不可变表 —— 它比所有 SSTable 都新。"""
        db = self.open_engine(memtable_capacity=1 << 20)
        db.put("before", "1")

        entered, release = threading.Event(), threading.Event()

        def flusher():
            db.flush()

        with mock.patch.object(
            engine_mod, "SSTableWriter", _blocking_writer(entered, release)
        ):
            thread = threading.Thread(target=flusher, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(timeout=30), "刷盘没开始")

                with db.snapshot() as snap:
                    self.assertEqual(snap.get_str("before"), "1")
                    self.assertEqual(snap.memtable_entries, 1)
                    self.assertEqual(list(snap.keys()), [b"before"])
            finally:
                release.set()
                thread.join(timeout=30)

    def test_writer_does_not_queue_behind_maintenance(self):
        """内存表满了、又有人在刷盘时,写入**不该排队**等它。

        排队看起来更"稳妥",实际上是白等:那一刻数据已经在 WAL 和内存表里了,
        不需要立刻变成 SSTable。而 ``_drain_locked`` 除了刷盘还会跑
        ``maybe_compact``(最多 16 轮),让一堆写者排在后面,尾延迟会难看到离谱。

        实测过这个差别:16 万次并发写入里,维护等待的中位耗时从 83 ms 降到 0 ms。

        测法:把刷盘**永久钉住**(不放行),然后猛写到内存表满。
        如果写入会排队,这个测试就会挂死;不排队则立刻返回。

        ⚠️ 写入量要卡在"超过容量但不到两倍容量"之间 —— 超过两倍就会触发
        背压(``MEMTABLE_STALL_FACTOR``),那时排队是**对的**。
        """
        capacity = 4096
        # 每条 ≈ 5(key) + 64(开销) + 20(value) = 89 字节
        # 55 条 ≈ 4895 字节:超过容量(4096)但远不到两倍(8192)
        entries = 55

        db = self.open_engine(memtable_capacity=capacity)
        db.put("seed", "v")

        entered, release = threading.Event(), threading.Event()

        def flusher():
            db.flush()

        with mock.patch.object(
            engine_mod, "SSTableWriter", _blocking_writer(entered, release)
        ):
            thread = threading.Thread(target=flusher, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(timeout=30), "刷盘没开始")

                done = threading.Event()

                def hammer():
                    for index in range(entries):
                        db.put(f"k{index:04d}", "v" * 20)
                    done.set()

                writer = threading.Thread(target=hammer, daemon=True)
                writer.start()
                self.assertTrue(
                    done.wait(timeout=30),
                    "写入排到刷盘后面去了 —— 应该抢不到维护权就直接返回",
                )
            finally:
                release.set()
                thread.join(timeout=30)

        # 钉住的那次刷盘结束之后,数据一条都不能少
        self.assertEqual(db.get_str("seed"), "v")
        for index in range(entries):
            self.assertEqual(
                db.get_str(f"k{index:04d}"), "v" * 20, f"k{index:04d}"
            )

    def test_read_proceeds_during_compaction(self):
        """归并的慢 I/O 不该挡住读。

        归并阶段完全不持 ``_state_lock``,所以读是并行的。
        旧设计里读和归并抢同一把锁,归并几 MB 的库就把读卡住全程。
        """
        db = self.open_engine(memtable_capacity=200, auto_compact=False)
        for index in range(40):
            db.put(f"k{index:03d}", "v" * 30)

        # 关掉自动归并,这样文件会真的攒下来,compact_all 才有活干
        self.assertGreater(db.stats().sstable_count, 1)

        entered, release = threading.Event(), threading.Event()
        errors: list[BaseException] = []

        def compactor():
            try:
                db.compact_all()
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        with mock.patch.object(
            engine_mod, "SSTableWriter", _blocking_writer(entered, release)
        ):
            thread = threading.Thread(target=compactor, daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(timeout=30), "归并没开始")
                for index in range(0, 40, 7):
                    self.assertEqual(
                        db.get_str(f"k{index:03d}"), "v" * 30,
                        f"k{index:03d}",
                    )
                self.assertIsNotNone(db.stats())
            finally:
                release.set()
                thread.join(timeout=30)

        self.assertFalse(errors, errors)
        for index in range(40):
            self.assertEqual(db.get_str(f"k{index:03d}"), "v" * 30)


class TestSnapshotUnderConcurrency(ConcurrencyTestCase):
    """并发写入期间的快照隔离性。"""

    def test_snapshot_is_stable_while_writers_run(self):
        """快照一旦创建就定格 —— 之后的写入、刷盘、归并都不该影响它。"""
        keys = [f"k{i:03d}" for i in range(50)]

        with self.open_engine(memtable_capacity=1 << 20) as db:
            for key in keys:
                db.put(key, "v0")

            with db.snapshot() as snap:
                def worker(index):
                    for key in keys:
                        db.put(key, f"new-{index}")

                self.run_threads(4, worker)

                # 中途还刷一次盘 + 归并一次,把快照 pin 的文件全换掉
                db.flush()
                db.compact_all()

                for key in keys:
                    self.assertEqual(snap.get_str(key), "v0", key)
                self.assertEqual(snap.memtable_entries, 50)

            # 快照关掉之后,实时读才看得到新值
            for key in keys:
                self.assertNotEqual(db.get_str(key), "v0")

    def test_scan_during_concurrent_writes(self):
        """一边扫一边写:扫描结果必须是某个一致时刻的视图。"""
        keys = [f"k{i:03d}" for i in range(80)]

        with self.open_engine(memtable_capacity=512) as db:
            for key in keys:
                db.put(key, "v0")

            def worker(index):
                for key in keys:
                    db.put(key, "v1")

            # 扫描已经开始,写入才发生 —— 扫到的应该全是 v0
            cursor = db.scan()
            first = next(cursor)

            self.run_threads(3, worker)

            seen = [first] + list(cursor)
            for key, value in seen:
                self.assertEqual(value, b"v0", key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
