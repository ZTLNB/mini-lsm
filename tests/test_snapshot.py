"""阶段 5 测试:快照读 + 流式范围扫描。

阶段 5 要兑现三个承诺,测试就围着它们转:

    1. **一致视图** —— 快照创建之后的写入/删除/刷盘/compaction,
       对它**一律不可见**。这是"读一致性"的全部内容。
    2. **流式** —— 读一个块才解析一个块。所以"取第一个元素"应该只读
       一个块,而不是把整个文件读完再返回第一个。
    3. **文件 pin 住** —— 快照引用的文件不能删,只能推迟;
       最后一个引用它的快照关掉之后才真正回收。

第 3 条最容易写错,而且错了**不会立刻报错**:
    Windows 上删一个句柄还开着的文件会失败(至少还能看见异常),
    而在 Linux 上会"删成功但数据还在",于是快照读到一半突然文件消失。
    所以这里既测"文件还在",也测"快照还能读",还测"关掉之后真删了"。
"""

import gc
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.engine import LSMEngine  # noqa: E402
from mini_lsm.errors import ClosedError  # noqa: E402
from mini_lsm.manifest import (  # noqa: E402
    FileMeta,
    files_overlapping,
    find_file_in_level,
)
from mini_lsm.snapshot import ScanCursor, Snapshot, search_levels  # noqa: E402
from mini_lsm.sstable import SSTableReader, sstable_filename  # noqa: E402


class SnapshotTestCase(unittest.TestCase):
    """公共脚手架:临时目录 + 用完自动关的引擎。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def open_engine(self, **kwargs) -> LSMEngine:
        engine = LSMEngine(self.dir, **kwargs)
        self.addCleanup(engine.close)
        return engine

    def sst_files(self) -> list[str]:
        return sorted(p.name for p in self.dir.glob("*.sst"))

    def write_range(self, db, start: int, stop: int, prefix: str = "key") -> None:
        for i in range(start, stop):
            db.put(f"{prefix}{i:04d}", f"value{i}")

    def compact_everything(self, db) -> None:
        """把所有数据都落盘并压到最底层。

        快照测试需要"磁盘上真有一批文件"才有意义 ——
        数据全在内存表里的话,pin 和延迟删除根本不会被触发。
        """
        db.flush()
        db.compact_all()

    @staticmethod
    def absent_keys_inside_file_range(limit: int = 400, step: int = 20):
        """落在文件键范围**之内**、但文件里没有的键。

        为什么非要"之内":超出范围时,``find_file_in_level`` 或
        ``SSTableReader.get`` 里的范围检查会直接把它挡掉 ——
        那根本轮不到布隆过滤器出手,测的就不是它了。
        """
        return [f"key{i:04d}z".encode() for i in range(0, limit, step)]


class TestSnapshotBasics(SnapshotTestCase):
    def test_snapshot_id_increments(self):
        db = self.open_engine()
        with db.snapshot() as s1, db.snapshot() as s2:
            self.assertEqual(s2.snapshot_id, s1.snapshot_id + 1)

    def test_snapshot_is_registered_on_the_engine(self):
        db = self.open_engine()
        self.assertEqual(db.snapshot_count, 0)
        with db.snapshot() as snap:
            self.assertEqual(db.snapshot_count, 1)
            self.assertFalse(snap.closed)
        self.assertEqual(db.snapshot_count, 0)
        self.assertTrue(snap.closed)

    def test_context_manager_closes(self):
        db = self.open_engine()
        db.put("a", "1")
        with db.snapshot() as snap:
            self.assertEqual(snap.get("a"), b"1")
        self.assertTrue(snap.closed)

    def test_close_is_idempotent(self):
        db = self.open_engine()
        snap = db.snapshot()
        snap.close()
        snap.close()
        self.assertEqual(db.snapshot_count, 0)

    def test_get_str_and_contains(self):
        db = self.open_engine()
        db.put("name", "alice")
        with db.snapshot() as snap:
            self.assertEqual(snap.get_str("name"), "alice")
            self.assertTrue(snap.contains("name"))
            self.assertFalse(snap.contains("nope"))
            self.assertIsNone(snap.get("nope"))

    def test_get_entry_distinguishes_tombstone_from_absent(self):
        """``(True, None)`` 是墓碑,``(False, None)`` 是从未出现。

        这个区分不是洁癖:读路径靠它决定"还要不要往更旧的文件找"。
        """
        db = self.open_engine()
        db.put("alive", "1")
        db.put("dead", "2")
        db.delete("dead")
        with db.snapshot() as snap:
            self.assertEqual(snap.get_entry("alive"), (True, b"1"))
            self.assertEqual(snap.get_entry("dead"), (True, None))
            self.assertEqual(snap.get_entry("never"), (False, None))

    def test_snapshot_sees_memtable_only_data(self):
        """还没刷盘的数据也必须能被快照看到 —— 它来自内存表副本。"""
        db = self.open_engine()
        db.put("a", "1")
        with db.snapshot() as snap:
            self.assertEqual(snap.memtable_entries, 1)
            self.assertEqual(snap.sstable_count, 0)
            self.assertEqual(snap.get("a"), b"1")

    def test_snapshot_counts_files_and_entries(self):
        db = self.open_engine(memtable_capacity=256)
        self.write_range(db, 0, 200)
        db.flush()
        with db.snapshot() as snap:
            self.assertEqual(snap.sstable_count, len(db.manifest.files()))
            self.assertEqual(snap.memtable_entries, 0)
            self.assertEqual(snap.file_ids,
                             frozenset(m.file_id for m in db.manifest.files()))

    def test_using_a_closed_snapshot_raises(self):
        db = self.open_engine()
        db.put("a", "1")
        snap = db.snapshot()
        snap.close()
        with self.assertRaises(ClosedError):
            snap.get("a")
        with self.assertRaises(ClosedError):
            list(snap.scan())

    def test_snapshot_on_closed_engine_raises(self):
        db = self.open_engine()
        db.close()
        with self.assertRaises(ClosedError):
            db.snapshot()

    def test_snapshot_created_before_close_raises_after(self):
        """引擎关了,已经发出去的快照随之失效 —— 不能假装还能读。"""
        db = self.open_engine()
        db.put("a", "1")
        snap = db.snapshot()
        db.close()
        self.assertTrue(snap.closed)
        with self.assertRaises(ClosedError):
            snap.get("a")

    def test_repr_does_not_raise(self):
        db = self.open_engine()
        db.put("a", "1")
        with db.snapshot() as snap:
            self.assertIn("Snapshot", repr(snap))
        self.assertIn("closed", repr(snap))


class TestSnapshotIsolation(SnapshotTestCase):
    """快照创建之后的一切改动,对它都不可见。"""

    def test_writes_after_snapshot_are_invisible(self):
        db = self.open_engine()
        db.put("a", "old")
        with db.snapshot() as snap:
            db.put("a", "new")
            db.put("b", "added")
            self.assertEqual(snap.get("a"), b"old")
            self.assertIsNone(snap.get("b"))
            self.assertEqual(db.get("a"), b"new")

    def test_deletes_after_snapshot_are_invisible(self):
        db = self.open_engine()
        db.put("a", "1")
        db.put("b", "2")
        with db.snapshot() as snap:
            db.delete("a")
            self.assertEqual(snap.get("a"), b"1", "快照里的数据被后来的删除带走了")
            self.assertEqual(dict(snap.scan()),
                             {b"a": b"1", b"b": b"2"})
            self.assertIsNone(db.get("a"))

    def test_put_after_delete_inside_snapshot(self):
        """快照之后"删了又写",快照仍应看到删除前的样子。"""
        db = self.open_engine()
        db.put("a", "v1")
        with db.snapshot() as snap:
            db.delete("a")
            db.put("a", "v2")
            self.assertEqual(snap.get("a"), b"v1")
            self.assertEqual(db.get("a"), b"v2")

    def test_flush_after_snapshot_is_invisible(self):
        db = self.open_engine()
        db.put("a", "1")
        with db.snapshot() as snap:
            db.flush()                      # 数据挪到了磁盘上
            db.put("b", "2")
            self.assertEqual(snap.memtable_entries, 1,
                             "快照的内存表副本不该被刷盘清掉")
            self.assertEqual(snap.sstable_count, 0,
                             "刷出来的新文件不该出现在快照的层结构里")
            self.assertEqual(dict(snap.scan()), {b"a": b"1"})
            self.assertEqual(dict(db.scan()), {b"a": b"1", b"b": b"2"})

    def test_compaction_after_snapshot_is_invisible(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 300)
        self.compact_everything(db)

        with db.snapshot() as snap:
            before_ids = snap.file_ids
            before_layout = [list(level) for level in snap._levels]  # noqa: SLF001
            # 再写一批,再压一次 —— 层结构必然变了
            self.write_range(db, 300, 600)
            db.flush()
            db.compact_all()

            self.assertEqual(snap.file_ids, before_ids,
                             "compaction 改动了快照 pin 的文件集合")
            self.assertEqual(snap._levels, before_layout,   # noqa: SLF001
                             "compaction 改动了快照捕获的层结构")
            self.assertEqual(len(dict(snap.scan())), 300)
            self.assertEqual(len(dict(db.scan())), 600)

    def test_snapshot_survives_full_compaction_and_still_reads(self):
        """最狠的一刀:快照开着的时候删数据 + 全量 compaction。

        compaction 压到最底层会把墓碑**真正清掉**,旧文件也会被删。
        但快照 pin 住的那批文件还在,所以它必须仍然看得到删除前的值 ——
        这正是"快照读"最难做到、也最容易做错的地方:
        墓碑一旦被清、旧文件一旦被删,快照看到的数据就凭空消失了。
        """
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 300)
        db.flush()

        with db.snapshot() as snap:
            expected = dict(snap.scan())

            db.delete("key0100")
            db.delete("key0200")
            db.flush()
            db.compact_all()        # 墓碑在这里被丢掉,旧文件在这里被删

            self.assertEqual(dict(snap.scan()), expected,
                             "全量 compaction 把快照看到的数据改掉了")
            self.assertEqual(snap.get("key0100"), b"value100")
            self.assertEqual(snap.get("key0200"), b"value200")

        # 实时读看到的是删除后的状态
        self.assertIsNone(db.get("key0100"))
        self.assertIsNone(db.get("key0200"))
        self.assertEqual(len(dict(db.scan())), 298)

    def test_two_snapshots_see_different_states(self):
        db = self.open_engine()
        db.put("a", "1")
        with db.snapshot() as s1:
            db.put("a", "2")
            with db.snapshot() as s2:
                db.put("a", "3")
                self.assertEqual(s1.get("a"), b"1")
                self.assertEqual(s2.get("a"), b"2")
                self.assertEqual(db.get("a"), b"3")


class TestSnapshotPinning(SnapshotTestCase):
    """快照 pin 住的文件必须活着,直到最后一个引用它的快照关掉。"""

    def _engine_with_files(self):
        """造一个"磁盘上有好几个文件"的引擎。

        ``auto_compact=False`` 是刻意的:开着自动 compaction 的话,
        刷出来的文件会被立刻合并掉,剩几个文件全看时机 ——
        而 pin 和延迟删除的测试需要文件数**确定**才测得准。
        """
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 400)
        db.flush()
        self.assertGreater(len(db.manifest.files()), 1,
                           "文件数不够,测不出 pin 的效果")
        return db

    def test_files_survive_compaction_while_snapshot_is_open(self):
        db = self._engine_with_files()
        with db.snapshot() as snap:
            pinned = snap.file_ids
            self.assertTrue(pinned)
            for file_id in pinned:
                self.assertTrue(
                    (self.dir / sstable_filename(file_id)).exists(),
                    f"sst-{file_id} 在快照创建时就不见了",
                )

            db.compact_all()

            for file_id in pinned:
                self.assertTrue(
                    (self.dir / sstable_filename(file_id)).exists(),
                    f"快照还开着,但 sst-{file_id} 已经被删了",
                )
            self.assertEqual(sorted(db.pending_delete_files),
                             sorted(pinned))
            self.assertEqual(db.stats().pending_delete_files, len(pinned))

    def test_files_are_deleted_once_the_snapshot_closes(self):
        db = self._engine_with_files()
        snap = db.snapshot()
        pinned = snap.file_ids
        db.compact_all()
        self.assertTrue(db.pending_delete_files)

        snap.close()

        self.assertEqual(db.pending_delete_files, [])
        for file_id in pinned:
            self.assertFalse(
                (self.dir / sstable_filename(file_id)).exists(),
                f"最后一个快照已关闭,sst-{file_id} 却还在",
            )
        self.assertEqual(db.stats().pending_delete_files, 0)

    def test_file_survives_until_the_last_snapshot_closes(self):
        db = self._engine_with_files()
        s1 = db.snapshot()
        s2 = db.snapshot()
        pinned = s1.file_ids
        db.compact_all()
        self.assertTrue(db.pending_delete_files)

        s1.close()
        self.assertTrue(db.pending_delete_files,
                        "还有快照引用着,不该删")
        for file_id in pinned:
            self.assertTrue((self.dir / sstable_filename(file_id)).exists())

        s2.close()
        self.assertEqual(db.pending_delete_files, [])
        for file_id in pinned:
            self.assertFalse((self.dir / sstable_filename(file_id)).exists())

    def test_snapshot_can_still_read_after_its_files_left_the_manifest(self):
        """文件离开 manifest 之后,快照必须还能读它 —— 靠的是自己那份 reader。"""
        db = self._engine_with_files()
        with db.snapshot() as snap:
            expected = dict(snap.scan())
            db.compact_all()
            self.assertNotIn(
                next(iter(snap.file_ids)), db.manifest.file_ids(),
                "测试前提不成立:旧文件居然还在 manifest 里",
            )
            self.assertEqual(dict(snap.scan()), expected)

    def test_scan_cursor_pins_until_exhausted(self):
        db = self._engine_with_files()
        cursor = db.scan()
        self.assertIsInstance(cursor, ScanCursor)
        self.assertEqual(db.snapshot_count, 1)
        next(cursor)                    # 只取一个,快照必须还活着
        db.compact_all()
        self.assertTrue(db.pending_delete_files,
                        "迭代器还在,旧文件不该被删")
        cursor.close()
        self.assertEqual(db.pending_delete_files, [])

    def test_engine_close_deletes_pending_files(self):
        """引擎都关了,就没有"还在读"的可能了 —— 待删文件必须清干净。"""
        db = self._engine_with_files()
        snap = db.snapshot()
        db.compact_all()
        pending = db.pending_delete_files
        self.assertTrue(pending)

        db.close()

        for file_id in pending:
            self.assertFalse((self.dir / sstable_filename(file_id)).exists())
        self.assertTrue(snap.closed)

    def test_reopening_after_close_sees_compacted_state(self):
        db = self._engine_with_files()
        snap = db.snapshot()
        db.compact_all()
        expected = dict(snap.scan())
        snap.close()
        db.close()

        with LSMEngine(self.dir) as again:
            self.assertEqual(dict(again.scan()), expected)
            self.assertEqual(again.pending_delete_files, [])

    def test_no_snapshot_means_immediate_deletion(self):
        """对照组:没有快照时,compaction 应该当场删掉旧文件,不留待删清单。"""
        db = self._engine_with_files()
        before = set(db.manifest.file_ids())
        db.compact_all()
        self.assertEqual(db.pending_delete_files, [])
        for file_id in before - set(db.manifest.file_ids()):
            self.assertFalse((self.dir / sstable_filename(file_id)).exists())


class TestStreamingScan(SnapshotTestCase):
    """``engine.scan()`` 的流式语义与快照语义。"""

    def test_scan_reads_only_one_block_for_the_first_element(self):
        """流式的核心证据:取第一个元素只读一个块,而不是读完整个文件。

        阶段 4 的实现是"锁内物化成 list 再返回",取第一个元素也要
        把整个文件读完 —— 这条断言就是用来钉住这个差别的。
        """
        db = self.open_engine(sstable_block_size=128)
        self.write_range(db, 0, 200)
        db.flush()                      # 内存表清空,只剩一个文件

        reader = db.sstables[0]
        total_blocks = len(reader._index)       # noqa: SLF001 - 测试需要
        self.assertGreater(total_blocks, 3, "块数太少,测不出区别")

        original = SSTableReader._read_block
        counter = {"n": 0}

        def counting(self, offset, length):
            counter["n"] += 1
            return original(self, offset, length)

        SSTableReader._read_block = counting
        try:
            cursor = db.scan()
            self.assertEqual(counter["n"], 0, "还没开始迭代就读了盘")
            next(cursor)
            self.assertEqual(counter["n"], 1,
                             "取第一个元素却读了不止一个块,说明不是流式的")
            cursor.close()
        finally:
            SSTableReader._read_block = original

    def test_breaking_early_releases_the_snapshot(self):
        db = self.open_engine()
        self.write_range(db, 0, 100)
        cursor = db.scan()
        next(cursor)
        self.assertEqual(db.snapshot_count, 1)
        cursor.close()
        self.assertEqual(db.snapshot_count, 0)

    def test_exhausting_the_cursor_releases_the_snapshot(self):
        db = self.open_engine()
        self.write_range(db, 0, 50)
        self.assertEqual(len(list(db.scan())), 50)
        self.assertEqual(db.snapshot_count, 0,
                         "扫完之后快照没释放,文件会一直被 pin 住")

    def test_garbage_collection_releases_the_snapshot(self):
        """调用方中途 break 掉又没 close —— 靠对象回收兜底。"""
        db = self.open_engine()
        self.write_range(db, 0, 50)
        cursor = db.scan()
        next(cursor)
        del cursor
        gc.collect()
        self.assertEqual(db.snapshot_count, 0)

    def test_scan_freezes_at_call_time(self):
        """``it = db.scan()`` 之后就定格,哪怕一行都还没迭代。

        这不是"顺手实现的" —— 而是必须的:调用方拿到迭代器的那一刻
        就认为它代表了当前状态,之后自己改数据不该改变它。
        """
        db = self.open_engine()
        db.put("a", "1")
        cursor = db.scan()
        db.put("b", "2")
        db.delete("a")
        self.assertEqual(dict(cursor), {b"a": b"1"})

    def test_scan_sees_state_from_creation_time_with_preexisting_data(self):
        db = self.open_engine()
        db.put("a", "1")
        db.put("c", "3")
        cursor = db.scan()
        db.put("b", "2")
        self.assertEqual(dict(cursor), {b"a": b"1", b"c": b"3"})

    def test_scan_range_is_half_open(self):
        db = self.open_engine()
        for ch in "abcde":
            db.put(ch, ch.upper())
        self.assertEqual([k for k, _ in db.scan("b", "d")], [b"b", b"c"])

    def test_scan_skips_tombstones(self):
        db = self.open_engine()
        db.put("a", "1")
        db.put("b", "2")
        db.delete("a")
        self.assertEqual(dict(db.scan()), {b"b": b"2"})

    def test_scan_over_multiple_files_and_memtable(self):
        """内存表 + 多个文件一起归并,新旧顺序不能乱。"""
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 300)
        db.flush()
        self.write_range(db, 300, 400)
        db.flush()
        db.put("key0100", "overwritten")        # 内存表里的更新版本

        got = dict(db.scan())
        self.assertEqual(len(got), 400)
        self.assertEqual(got[b"key0100"], b"overwritten")
        self.assertEqual(got[b"key0000"], b"value0")
        self.assertEqual(got[b"key0399"], b"value399")

    def test_scan_is_sorted(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 300)
        db.flush()
        db.compact_all()
        self.write_range(db, 100, 150)          # 制造跨来源的乱序写入

        keys = list(db.keys())
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), 300)

    def test_keys_accepts_a_range(self):
        db = self.open_engine()
        for ch in "abcde":
            db.put(ch, "1")
        self.assertEqual(list(db.keys("b", "d")), [b"b", b"c"])

    def test_scan_on_closed_engine_raises(self):
        db = self.open_engine()
        db.put("a", "1")
        db.close()
        with self.assertRaises(ClosedError):
            db.scan()

    def test_cursor_repr_and_snapshot_accessor(self):
        db = self.open_engine()
        db.put("a", "1")
        cursor = db.scan()
        self.assertIsInstance(cursor.snapshot, Snapshot)
        self.assertIn("ScanCursor", repr(cursor))
        self.assertIn("open", repr(cursor))
        cursor.close()
        self.assertIn("closed", repr(cursor))

    def test_iterating_a_closed_cursor_yields_nothing(self):
        db = self.open_engine()
        db.put("a", "1")
        cursor = db.scan()
        cursor.close()
        self.assertEqual(list(cursor), [])

    def test_double_close_is_safe(self):
        db = self.open_engine()
        db.put("a", "1")
        cursor = db.scan()
        cursor.close()
        cursor.close()
        self.assertEqual(db.snapshot_count, 0)

    def test_scan_prunes_files_outside_the_range(self):
        """窄区间扫描不该把每个文件都打开 —— 范围剪枝必须生效。

        这里**不做** compact_all:合并之后就只剩一个文件了,
        "只打开了其中一部分"这件事根本无从观察。
        保持 L0 里的多个窄范围文件才测得出剪枝。
        """
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 600)
        db.flush()
        total_files = len(db.manifest.files())
        self.assertGreater(total_files, 3, "文件太少,测不出剪枝")

        opened = []
        original = SSTableReader.iter_entries

        def tracking(self, *args, **kwargs):
            opened.append(self.file_id)
            return original(self, *args, **kwargs)

        SSTableReader.iter_entries = tracking
        try:
            got = list(db.scan("key0000", "key0002"))
        finally:
            SSTableReader.iter_entries = original

        self.assertEqual([k for k, _ in got], [b"key0000", b"key0001"])
        self.assertLess(
            len(set(opened)), total_files,
            f"区间只覆盖 2 个键,却打开了 {len(set(opened))}/{total_files} 个文件",
        )

    def test_snapshot_scan_prunes_files_outside_the_range(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 600)
        db.flush()
        total_files = len(db.manifest.files())

        opened = []
        original = SSTableReader.iter_entries

        def tracking(self, *args, **kwargs):
            opened.append(self.file_id)
            return original(self, *args, **kwargs)

        SSTableReader.iter_entries = tracking
        try:
            with db.snapshot() as snap:
                got = list(snap.scan("key0000", "key0003"))
        finally:
            SSTableReader.iter_entries = original

        self.assertEqual(len(got), 3)
        self.assertLess(len(set(opened)), total_files)

    def test_many_small_scans_do_not_leak_snapshots(self):
        """反复开扫描不能把快照攒起来 —— 攒起来就等于文件永远删不掉。"""
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 400)
        db.flush()
        db.compact_all()
        for _ in range(50):
            list(db.scan("key0000", "key0005"))
        self.assertEqual(db.snapshot_count, 0)
        self.assertEqual(db.pending_delete_files, [])


class TestSnapshotReadPathMatchesLiveRead(SnapshotTestCase):
    """快照读和实时读必须给出**完全一致**的结果。

    两边走的是同一个 ``search_levels``;这条测试是防止有人日后
    "顺手优化"其中一边、把两边悄悄改岔了 —— 那种 bug 的表现是
    "偶发查不到数据",极难定位。
    """

    def test_point_lookups_agree_on_an_unchanged_database(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 300)
        db.flush()
        db.delete("key0050")
        db.flush()
        db.compact_all()

        keys = [f"key{i:04d}".encode() for i in range(0, 320)]
        keys += [b"absent", b"key0000x", b""]
        with db.snapshot() as snap:
            for key in keys:
                self.assertEqual(snap.get(key), db.get(key), key)
                self.assertEqual(snap.get_entry(key), db.get_entry(key), key)

    def test_scans_agree_on_an_unchanged_database(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 300)
        db.flush()
        db.delete("key0100")
        self.write_range(db, 300, 400)
        db.flush()
        db.compact_all()

        with db.snapshot() as snap:
            self.assertEqual(dict(snap.scan()), dict(db.scan()))
            self.assertEqual(dict(snap.scan("key0050", "key0150")),
                             dict(db.scan("key0050", "key0150")))

    def test_search_levels_finds_l0_by_trying_every_file(self):
        """L0 允许范围重叠,所以必须从新到旧挨个试,不能二分。"""
        db = self.open_engine()
        db.put("a", "old")
        db.flush()
        db.put("a", "new")
        db.flush()

        # 两个文件都在 L0,且范围重叠
        self.assertEqual(len(db.manifest.levels[0]), 2)
        found, value, _ = search_levels(db.manifest.levels, db._readers, b"a")
        self.assertEqual((found, value), (True, b"new"),
                         "L0 没有按新到旧顺序试,读到了旧值")

    def test_search_levels_returns_false_when_absent(self):
        db = self.open_engine()
        db.put("a", "1")
        db.flush()
        found, value, _ = search_levels(db.manifest.levels, db._readers, b"zzz")
        self.assertEqual((found, value), (False, None))

    def test_search_levels_reports_bloom_rejections(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False,
                              bloom_bits_per_key=10)
        self.write_range(db, 0, 400)
        db.flush()
        db.compact_all()

        total = 0
        for key in self.absent_keys_inside_file_range():
            _found, _value, rejections = search_levels(
                db.manifest.levels, db._readers, key
            )
            total += rejections
        self.assertGreater(
            total, 0,
            "范围之内的不存在键应该被布隆过滤器挡下来,不该真去读文件",
        )

    def test_engine_get_counts_bloom_rejections(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 400)
        db.flush()
        db.compact_all()
        before = db.stats().bloom_rejections
        for key in self.absent_keys_inside_file_range():
            db.get(key)
        self.assertGreater(db.stats().bloom_rejections, before)

    def test_snapshot_counts_its_own_rejections(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 400)
        db.flush()
        db.compact_all()
        with db.snapshot() as snap:
            for key in self.absent_keys_inside_file_range():
                snap.get(key)
            self.assertGreater(snap.bloom_rejections, 0)

    def test_engine_get_entry_reports_tombstones(self):
        db = self.open_engine()
        db.put("dead", "1")
        db.delete("dead")
        self.assertEqual(db.get_entry("dead"), (True, None))
        self.assertEqual(db.get_entry("never"), (False, None))
        self.assertIsNone(db.get("dead"))


class TestSnapshotRangeQueries(SnapshotTestCase):
    """区间端点、边界和内存表切片。"""

    def test_memtable_slice_uses_bisect(self):
        """内存表副本在区间内的那一段必须精确 —— 两端都不能多不能少。"""
        db = self.open_engine()
        for i in range(20):
            db.put(f"k{i:02d}", str(i))
        with db.snapshot() as snap:
            got = [k for k, _ in snap.scan("k05", "k10")]
            self.assertEqual(got, [f"k{i:02d}".encode() for i in range(5, 10)])

    def test_range_with_start_only(self):
        db = self.open_engine()
        for i in range(10):
            db.put(f"k{i}", str(i))
        with db.snapshot() as snap:
            self.assertEqual(len(list(snap.scan("k5"))), 5)

    def test_range_with_end_only(self):
        db = self.open_engine()
        for i in range(10):
            db.put(f"k{i}", str(i))
        with db.snapshot() as snap:
            self.assertEqual([k for k, _ in snap.scan(end="k3")],
                             [b"k0", b"k1", b"k2"])

    def test_range_across_memtable_and_files(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 300)
        db.flush()
        db.compact_all()
        self.write_range(db, 300, 400)      # 留在内存表里

        with db.snapshot() as snap:
            got = [k for k, _ in snap.scan("key0290", "key0310")]
            self.assertEqual(
                got, [f"key{i:04d}".encode() for i in range(290, 310)]
            )

    def test_items_is_an_alias_for_scan(self):
        db = self.open_engine()
        db.put("a", "1")
        db.put("b", "2")
        with db.snapshot() as snap:
            self.assertEqual(dict(snap.items()), dict(snap.scan()))
            self.assertEqual(list(snap.keys()), [b"a", b"b"])

    def test_scan_count_is_tracked(self):
        db = self.open_engine()
        db.put("a", "1")
        with db.snapshot() as snap:
            self.assertEqual(snap.scan_count, 0)
            list(snap.scan())
            list(snap.scan())
            self.assertEqual(snap.scan_count, 2)

    def test_empty_range_on_empty_engine(self):
        db = self.open_engine()
        with db.snapshot() as snap:
            self.assertEqual(list(snap.scan()), [])
            self.assertEqual(dict(snap.scan("a", "z")), {})


class TestFindFileInLevel(unittest.TestCase):
    """``find_file_in_level`` —— 层内二分的公共实现。"""

    @staticmethod
    def meta(file_id: int, smallest: bytes, largest: bytes) -> FileMeta:
        return FileMeta(
            file_id=file_id,
            level=1,
            smallest=smallest,
            largest=largest,
            entry_count=1,
            size=1,
        )

    def setUp(self):
        # 三个互不重叠的文件:b..d、f..h、j..l
        self.files = [
            self.meta(1, b"b", b"d"),
            self.meta(2, b"f", b"h"),
            self.meta(3, b"j", b"l"),
        ]

    def test_empty_level(self):
        self.assertIsNone(find_file_in_level([], b"a"))

    def test_key_before_everything(self):
        self.assertIsNone(find_file_in_level(self.files, b"a"))

    def test_key_after_everything(self):
        self.assertIsNone(find_file_in_level(self.files, b"z"))

    def test_key_inside_a_file(self):
        self.assertEqual(find_file_in_level(self.files, b"g").file_id, 2)

    def test_boundaries_are_inclusive(self):
        self.assertEqual(find_file_in_level(self.files, b"b").file_id, 1)
        self.assertEqual(find_file_in_level(self.files, b"d").file_id, 1)
        self.assertEqual(find_file_in_level(self.files, b"f").file_id, 2)
        self.assertEqual(find_file_in_level(self.files, b"l").file_id, 3)

    def test_gap_between_files_returns_none(self):
        """文件之间的空隙要返回 None —— 这里最容易写成"返回左边那个文件"。"""
        self.assertIsNone(find_file_in_level(self.files, b"e"))
        self.assertIsNone(find_file_in_level(self.files, b"i"))

    def test_single_file(self):
        one = [self.meta(7, b"m", b"n")]
        self.assertEqual(find_file_in_level(one, b"m").file_id, 7)
        self.assertIsNone(find_file_in_level(one, b"z"))

    def test_matches_linear_scan_on_a_larger_level(self):
        """二分的结果必须和逐个比一样 —— 这是层内不重叠的前提换来的。"""
        files = [
            self.meta(i, f"k{i:04d}".encode(), f"k{i:04d}~".encode())
            for i in range(0, 200, 2)
        ]
        for probe in [f"k{i:04d}".encode() for i in range(0, 210)] + [b"k0001"]:
            expected = next(
                (m for m in files if m.smallest <= probe <= m.largest), None
            )
            self.assertEqual(find_file_in_level(files, probe), expected, probe)


class TestFilesOverlapping(unittest.TestCase):
    """``files_overlapping`` —— 范围剪枝。左闭右开,端点最容易写错。"""

    @staticmethod
    def meta(file_id: int, smallest: bytes, largest: bytes) -> FileMeta:
        return FileMeta(file_id, 1, smallest, largest, 1, 1)

    def setUp(self):
        self.files = [
            self.meta(1, b"a", b"c"),
            self.meta(2, b"d", b"f"),
            self.meta(3, b"g", b"i"),
        ]

    def test_wide_range_keeps_everything(self):
        self.assertEqual(len(files_overlapping(self.files, b"", b"z")), 3)

    def test_range_inside_one_file(self):
        got = files_overlapping(self.files, b"b", b"c")
        self.assertEqual([m.file_id for m in got], [1])

    def test_end_is_exclusive(self):
        """区间是 ``[start, end)``,``end`` 本身不在区间内。

        ⚠️ 注意这里剪枝是**文件级**的、保守的:文件 1 的范围是 ``a..c``,
        而 ``c`` 确实落在 ``[c, d)`` 里,所以它必须被选中。
        真正体现"右开"的是文件 2 —— 它的最小键就是 ``d``,
        而 ``d`` 不在区间里,所以整份被排除。
        """
        got = [m.file_id for m in files_overlapping(self.files, b"c", b"d")]
        self.assertEqual(got, [1])

    def test_empty_range_selects_nothing(self):
        """``[d, d)`` 是空区间 —— 哪怕有个文件的最小键正好是 d,也不该被选中。"""
        self.assertEqual(files_overlapping(self.files, b"d", b"d"), [])

    def test_start_is_inclusive(self):
        got = files_overlapping(self.files, b"d", b"e")
        self.assertEqual([m.file_id for m in got], [2])

    def test_range_spanning_two_files(self):
        got = files_overlapping(self.files, b"c", b"h")
        self.assertEqual([m.file_id for m in got], [1, 2, 3])

    def test_range_after_everything(self):
        self.assertEqual(files_overlapping(self.files, b"z", b"zz"), [])

    def test_range_before_everything(self):
        self.assertEqual(files_overlapping(self.files, b"", b"a"), [])


class TestEngineStatsSnapshots(SnapshotTestCase):
    def test_stats_report_snapshots(self):
        db = self.open_engine()
        self.write_range(db, 0, 100)
        self.assertEqual(db.stats().snapshots, 0)
        with db.snapshot() as snap:
            stats = db.stats()
            self.assertEqual(stats.snapshots, 1)
            self.assertEqual(stats.pinned_files, len(snap.file_ids))
        self.assertEqual(db.stats().snapshots, 0)

    def test_stats_str_mentions_snapshots_when_present(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 200)
        db.flush()
        with db.snapshot() as snap:
            db.compact_all()
            text = str(db.stats())
            self.assertIn("快照", text)
            self.assertIn(str(len(snap.file_ids)), text)

    def test_stats_str_quiet_when_no_snapshots(self):
        db = self.open_engine()
        db.put("a", "1")
        self.assertNotIn("快照", str(db.stats()))

    def test_pinned_file_ids_property(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 200)
        db.flush()
        with db.snapshot() as snap:
            self.assertEqual(db.pinned_file_ids, sorted(snap.file_ids))
        self.assertEqual(db.pinned_file_ids, [])


class TestSnapshotUnderConcurrentMutation(SnapshotTestCase):
    """边扫边改:快照必须稳如磐石。

    这是"流式 + 不持锁"真正的压力点 —— 迭代器一旦开始跑,
    引擎锁就不在它手上了,所有保护都来自 pin 住的文件和不可变的 SSTable。
    """

    def test_scan_while_writing_and_compacting(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 500)
        db.flush()

        cursor = db.scan()
        collected = []
        for index, item in enumerate(cursor):
            collected.append(item)
            if index == 10:
                # 迭代进行到一半,往死里搅:写入、刷盘、compaction
                self.write_range(db, 1000, 1100, prefix="extra")
                db.flush()
                db.compact_all()
        cursor.close()

        self.assertEqual(len(collected), 500)
        self.assertEqual(
            [k for k, _ in collected],
            [f"key{i:04d}".encode() for i in range(500)],
            "边扫边改把快照的内容改动了",
        )
        self.assertEqual(len(dict(db.scan())), 600)

    def test_snapshot_get_while_compacting(self):
        db = self.open_engine(memtable_capacity=4096, auto_compact=False)
        self.write_range(db, 0, 500)
        db.flush()

        with db.snapshot() as snap:
            for step, i in enumerate(range(0, 500, 37)):
                key = f"key{i:04d}".encode()
                before = snap.get(key)
                db.put(f"extra{step:04d}", "x")     # 每轮都制造新数据
                db.flush()
                db.compact_all()
                self.assertEqual(snap.get(key), before,
                                 f"{key} 在 compaction 前后读到了不同的值")

    def test_snapshot_is_invalidated_when_its_engine_closes(self):
        """引擎关闭会让快照失效 —— 不能假装还能读,那只会读到半截数据。"""
        db = self.open_engine()
        db.put("a", "1")
        snap = db.snapshot()
        db.close()

        self.assertTrue(snap.closed)
        with self.assertRaises(ClosedError):
            snap.get("a")

        # 重开之后数据还在,而且没有遗留的待删文件
        with LSMEngine(self.dir) as db2:
            self.assertEqual(db2.get("a"), b"1")
            self.assertEqual(db2.pending_delete_files, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
