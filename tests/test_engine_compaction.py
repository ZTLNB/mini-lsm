"""阶段 3 测试:引擎的 Compaction 端到端行为。

前面两个文件测的是零件(manifest 存得对不对、策略挑得准不准),
这个文件测的是**拼起来之后数据还对不对**。

阶段 3 最容易出、后果最严重的一类 bug 是**墓碑失效**:

    删除一个键,只是写一条墓碑。如果 compaction 在还更深的层里
    存在旧值时就把墓碑丢了,那个键会**复活** —— 用户明明删过,
    重启后它又回来了。

这类 bug 在真实存储引擎里出过很多次,而且很难在测试之外被发现:
不报错、不崩溃,只是数据悄悄地变多。所以下面每个丢墓碑的路径
都单独钉了一条测试。
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.engine import WAL_FILENAME, LSMEngine  # noqa: E402
from mini_lsm.errors import CorruptionError  # noqa: E402
from mini_lsm.manifest import MANIFEST_FILENAME  # noqa: E402
from mini_lsm.sstable import SSTableReader, SSTableWriter, sstable_filename  # noqa: E402


class CompactionTestCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def open_engine(self, **kwargs) -> LSMEngine:
        engine = LSMEngine(self.dir, **kwargs)
        self.addCleanup(engine.close)
        return engine

    @property
    def manifest_path(self) -> Path:
        return self.dir / MANIFEST_FILENAME

    @property
    def wal_path(self) -> Path:
        return self.dir / WAL_FILENAME

    def sst_files(self) -> list[str]:
        return sorted(p.name for p in self.dir.glob("*.sst"))

    def sst_file_ids(self) -> list[int]:
        ids = []
        for name in self.sst_files():
            ids.append(int(name[len("sst-"):-len(".sst")]))
        return sorted(ids)

    def write_sstable(self, file_id: int, items: list[tuple[bytes, bytes | None]]) -> None:
        """直接造一个 SSTable 文件 —— 用来模拟崩溃残留或老目录。"""
        path = self.dir / sstable_filename(file_id)
        writer = SSTableWriter(path)
        for key, value in items:
            writer.add(key, value)
        writer.finish()

    def level_layout(self, db: LSMEngine) -> list[int]:
        return [len(level) for level in db.manifest.levels]


# ------------------------------------------------------------------ 触发与收益


class TestCompactionTriggers(CompactionTestCase):
    def test_auto_compact_disabled_keeps_every_file(self):
        with self.open_engine(auto_compact=False) as db:
            for batch in range(6):
                db.put(f"k{batch}", "v")
                db.flush()

            self.assertEqual(db.stats().sstable_count, 6)
            self.assertEqual(db.stats().compactions, 0)

    def test_l0_trigger_merges_into_l1(self):
        with self.open_engine(l0_compaction_trigger=3) as db:
            for batch in range(3):
                db.put(f"k{batch}", "v")
                db.flush()

            layout = self.level_layout(db)
            self.assertEqual(layout[0], 0, "L0 应该被清空")
            self.assertGreater(layout[1], 0, "数据应该落到 L1")
            self.assertGreaterEqual(db.stats().compactions, 1)

    def test_manual_compact_does_one_round(self):
        with self.open_engine(auto_compact=False, l0_compaction_trigger=2) as db:
            for batch in range(4):
                db.put(f"k{batch}", "v")
                db.flush()
            self.assertEqual(self.level_layout(db)[0], 4)

            self.assertTrue(db.compact())
            self.assertEqual(self.level_layout(db)[0], 0)

    def test_manual_compact_returns_false_when_nothing_to_do(self):
        with self.open_engine(auto_compact=False) as db:
            db.put("a", "1")
            db.flush()
            self.assertFalse(db.compact())

    def test_maybe_compact_returns_round_count(self):
        with self.open_engine(auto_compact=False, l0_compaction_trigger=2) as db:
            for batch in range(4):
                db.put(f"k{batch}", "v")
                db.flush()

            rounds = db.maybe_compact()
            self.assertGreaterEqual(rounds, 1)
            self.assertIsNone(
                # 再做一轮已经没有可做的了
                None if db.maybe_compact() == 0 else "不该还有任务",
            )

    def test_file_count_stays_bounded_under_sustained_writes(self):
        """阶段 2 的痛点:写 3000 条刷出 49 个文件,一次 get 要翻 49 个。

        阶段 3 之后文件数应该被压在一个常数附近,不随写入量线性增长。
        """
        with self.open_engine(memtable_capacity=4096, l0_compaction_trigger=4) as db:
            for i in range(3000):
                db.put(f"key{i:05d}", "x" * 64)

            stats = db.stats()
            self.assertGreater(stats.flushes, 10)
            self.assertLess(
                stats.sstable_count, stats.flushes,
                "归并没有起到压缩文件数的作用",
            )
            # 数据一条都不能少
            self.assertEqual(db.get_str("key00000"), "x" * 64)
            self.assertEqual(db.get_str("key02999"), "x" * 64)


# ------------------------------------------------------------------ 数据完整性


class TestCompactionDataIntegrity(CompactionTestCase):
    def test_all_keys_readable_after_compaction(self):
        with self.open_engine(auto_compact=False, l0_compaction_trigger=2) as db:
            for batch in range(4):
                for i in range(20):
                    db.put(f"b{batch}-k{i:02d}", f"v{batch}-{i}")
                db.flush()

            db.compact_all()

            for batch in range(4):
                for i in range(20):
                    key = f"b{batch}-k{i:02d}"
                    self.assertEqual(db.get_str(key), f"v{batch}-{i}", key)

    def test_newest_version_wins_after_compaction(self):
        """同一个键被反复改写,归并后必须留下最新的那个版本。"""
        with self.open_engine(auto_compact=False, l0_compaction_trigger=2) as db:
            for version in range(10):
                db.put("hot", f"v{version}")
                db.flush()

            db.compact_all()
            self.assertEqual(db.get_str("hot"), "v9")

    def test_newest_version_wins_across_levels(self):
        """旧版本已经被压到深层,新版本还在 L0 —— 读的时候新的必须赢。"""
        with self.open_engine(auto_compact=False, l0_compaction_trigger=100) as db:
            db.put("k", "old")
            db.flush()
            db.compact_all()            # old 现在在最底层

            db.put("k", "new")
            db.flush()                  # new 在 L0

            self.assertEqual(db.get_str("k"), "new")

    def test_keys_unique_after_compaction(self):
        """归并的输出里不该有重复 key —— 有重复说明归并逻辑错了。"""
        with self.open_engine(auto_compact=False, l0_compaction_trigger=2) as db:
            for version in range(8):
                db.put("k", f"v{version}")
                db.put(f"filler{version}", "x")
                db.flush()

            db.compact_all()

            keys = list(db.keys())
            self.assertEqual(len(keys), len(set(keys)), "扫描结果里有重复 key")
            self.assertIn(b"k", keys)

    def test_scan_matches_point_lookups(self):
        with self.open_engine(auto_compact=False, l0_compaction_trigger=3) as db:
            expected = {}
            for batch in range(5):
                for i in range(15):
                    key = f"b{batch}-k{i:02d}"
                    value = f"v{batch}-{i}"
                    db.put(key, value)
                    expected[key.encode()] = value.encode()
                db.flush()

            db.compact_all()

            scanned = dict(db.scan())
            self.assertEqual(scanned, expected)

    def test_utf8_and_binary_values_survive(self):
        with self.open_engine(auto_compact=False, l0_compaction_trigger=2) as db:
            db.put("城市", "北京")
            db.put("名字", "张三")
            db.flush()
            db.put(b"\x00\xff", b"\xfe\x01\x80")
            db.flush()

            db.compact_all()

            self.assertEqual(db.get_str("城市"), "北京")
            self.assertEqual(db.get_str("名字"), "张三")
            self.assertEqual(db.get(b"\x00\xff"), b"\xfe\x01\x80")

    def test_compaction_splits_output_when_large(self):
        """输出超过 target_file_size 时要切成多个文件,且切完还能读全。"""
        with self.open_engine(
            auto_compact=False,
            l0_compaction_trigger=2,
            target_file_size=512,
        ) as db:
            for batch in range(2):
                for i in range(200):
                    db.put(f"k{batch}-{i:04d}", "y" * 40)
                db.flush()

            db.compact_all()

            self.assertGreater(db.stats().sstable_count, 1, "应该被切成多个文件")
            for batch in range(2):
                for i in range(200):
                    key = f"k{batch}-{i:04d}"
                    self.assertEqual(db.get_str(key), "y" * 40, key)


# ------------------------------------------------------------------ 墓碑(重点)


class TestTombstoneSafety(CompactionTestCase):
    """墓碑失效会让删掉的键复活 —— 这一组测试专门盯这个。"""

    def test_deleted_key_stays_deleted_after_full_compaction(self):
        with self.open_engine(auto_compact=False) as db:
            db.put("k", "v")
            db.flush()
            db.delete("k")
            db.flush()

            db.compact_all()

            self.assertIsNone(db.get("k"))
            self.assertFalse(db.contains("k"))

    def test_deleted_key_stays_deleted_after_restart(self):
        with self.open_engine(auto_compact=False) as db:
            db.put("k", "v")
            db.flush()
            db.delete("k")
            db.flush()
            db.compact_all()

        with self.open_engine() as db:
            self.assertIsNone(db.get("k"))
            self.assertFalse(db.contains("k"))

    def test_tombstone_kept_when_target_is_not_max_level(self):
        """这是最容易写错的一条路径。

        值在最底层 L2,墓碑在 L0。把 L0 压到 L1 时 **不能** 丢墓碑 ——
        因为 L2 里的旧值还在,墓碑一丢就复活。
        """
        with self.open_engine(
            auto_compact=False, l0_compaction_trigger=1, num_levels=3
        ) as db:
            db.put("k", "v")
            db.flush()
            db.compact_all()            # 值压到 L2
            self.assertEqual(db.get_str("k"), "v")

            db.delete("k")
            db.flush()                  # 墓碑在 L0

            db.compact()                # L0 → L1,此时 L1 不是最底层
            self.assertEqual(self.level_layout(db)[1], 1)

            # 墓碑还在,所以读出来必须是"不存在"
            self.assertIsNone(db.get("k"))

        with self.open_engine() as db:
            self.assertIsNone(db.get("k"))

    def test_tombstone_dropped_only_after_merging_with_deeper_value(self):
        """墓碑往下压时,深层里同 key 的旧值必须一起参与归并。

        如果只丢墓碑、不带旧值一起归并,旧值就会留下来 —— 复活。
        """
        with self.open_engine(
            auto_compact=False, l0_compaction_trigger=1, num_levels=3
        ) as db:
            db.put("k", "v")
            db.flush()
            db.compact_all()            # 值在 L2

            db.delete("k")
            db.flush()                  # 墓碑在 L0
            db.compact()                # 墓碑到 L1

            # 手动把 L1 压到 L2(最底层)—— 这一步墓碑会被丢掉,
            # 但 L2 里的旧值必须被一起归并掉
            db.compact()
            self.assertIsNone(db.get("k"))

    def test_tombstone_and_shadowed_value_both_vanish_at_max_level(self):
        """压到最底层后,磁盘上少了**两条**记录,不是一条。

        少的第 1 条是墓碑本身;第 2 条是被墓碑压住的那个旧值 ——
        归并时"同 key 只留最新版本",墓碑赢了,旧值就被丢弃了。
        最后剩下的只有真正存活的键。
        """
        with self.open_engine(auto_compact=False) as db:
            db.put("a", "1")
            db.put("b", "2")
            db.put("c", "3")
            db.flush()

            db.delete("b")
            db.flush()

            before = db.stats().sstable_entries     # a、b、c + 墓碑 = 4
            db.compact_all()
            after = db.stats().sstable_entries      # 只剩 a、c = 2

            self.assertEqual(before, 4)
            self.assertEqual(after, 2)
            self.assertEqual(sorted(db.keys()), [b"a", b"c"])

    def test_pointless_tombstone_alone_is_dropped(self):
        """删一个根本不存在的键 —— 这时只有墓碑消失,数据一条不少。

        这条把"墓碑本身被丢弃"从"旧值被压掉"里隔离出来,
        免得上面那条测试的 4→2 掩盖了别的问题。
        """
        with self.open_engine(auto_compact=False) as db:
            db.put("a", "1")
            db.put("b", "2")
            db.flush()

            db.delete("zzz")        # 这个键从来不存在
            db.flush()

            before = db.stats().sstable_entries     # a、b + 墓碑 = 3
            db.compact_all()
            after = db.stats().sstable_entries      # a、b = 2

            self.assertEqual(before, 3)
            self.assertEqual(after, 2)
            self.assertEqual(sorted(db.keys()), [b"a", b"b"])

    def test_delete_then_reinsert_then_compact(self):
        """删除后又写回同一个键 —— 最新的写入必须赢。"""
        with self.open_engine(auto_compact=False) as db:
            db.put("k", "v1")
            db.flush()
            db.delete("k")
            db.flush()
            db.put("k", "v2")
            db.flush()

            db.compact_all()
            self.assertEqual(db.get_str("k"), "v2")

    def test_mass_delete_survives_compaction(self):
        """批量删除之后整体归并,被删的键一个都不能复活。"""
        with self.open_engine(auto_compact=False, l0_compaction_trigger=2) as db:
            for batch in range(3):
                for i in range(30):
                    db.put(f"k{batch}-{i:02d}", "v")
                db.flush()

            for batch in range(3):
                for i in range(0, 30, 2):       # 删掉偶数号
                    db.delete(f"k{batch}-{i:02d}")
                db.flush()

            db.compact_all()

            for batch in range(3):
                for i in range(30):
                    key = f"k{batch}-{i:02d}"
                    if i % 2 == 0:
                        self.assertIsNone(db.get(key), f"{key} 复活了")
                    else:
                        self.assertEqual(db.get_str(key), "v")

    def test_tombstone_in_memtable_still_shadows_compacted_data(self):
        """墓碑还在内存表里时,也不能被深层的旧值盖过。"""
        with self.open_engine(auto_compact=False) as db:
            db.put("k", "v")
            db.flush()
            db.compact_all()

            db.delete("k")      # 只在内存表里,还没刷盘
            self.assertIsNone(db.get("k"))

        with self.open_engine() as db:
            self.assertIsNone(db.get("k"))


# ------------------------------------------------------------------ 分层归并


class TestMultiLevelCompaction(CompactionTestCase):
    def test_data_reaches_max_level(self):
        with self.open_engine(
            auto_compact=False,
            l0_compaction_trigger=2,
            level_size_budget=1024,
            level_size_factor=2,
            target_file_size=512,
        ) as db:
            for batch in range(8):
                for i in range(40):
                    db.put(f"b{batch}-k{i:03d}", "z" * 30)
                db.flush()

            db.compact_all()

            layout = self.level_layout(db)
            self.assertGreater(layout[2], 0, "数据应该压到了最底层")
            self.assertEqual(layout[0], 0)
            self.assertEqual(layout[1], 0)

            for batch in range(8):
                for i in range(40):
                    key = f"b{batch}-k{i:03d}"
                    self.assertEqual(db.get_str(key), "z" * 30, key)

    def test_manifest_invariants_hold_after_each_round(self):
        """每做完一轮 compaction,层序契约都必须仍然成立。"""
        with self.open_engine(
            auto_compact=False,
            l0_compaction_trigger=2,
            level_size_budget=1024,
            level_size_factor=2,
        ) as db:
            for batch in range(10):
                for i in range(25):
                    db.put(f"b{batch}-k{i:03d}", "w" * 20)
                db.flush()

                db.maybe_compact()
                db.manifest.check_invariants()

    def test_l1_is_non_overlapping_after_compaction(self):
        """L0 的文件互相重叠,压到 L1 之后必须变得互不重叠 ——
        这是"二分定位唯一文件"能成立的前提。
        """
        with self.open_engine(auto_compact=False, l0_compaction_trigger=4) as db:
            for batch in range(4):
                for i in range(10):
                    # 每个 L0 文件的键范围都横跨全部键,故意制造重叠
                    db.put(f"k{i:02d}", f"v{batch}-{i}")
                db.flush()

            db.compact()

            l1 = db.manifest.levels[1]
            for left, right in zip(l1, l1[1:]):
                self.assertLess(
                    left.largest, right.smallest,
                    "L1 内部出现了重叠",
                )

    def test_level_layout_reported_in_stats(self):
        with self.open_engine(l0_compaction_trigger=2) as db:
            for batch in range(2):
                db.put(f"k{batch}", "v")
                db.flush()

            stats = db.stats()
            self.assertEqual(len(stats.level_files), db.num_levels)
            self.assertEqual(len(stats.level_bytes), db.num_levels)
            self.assertEqual(stats.level_files[0], 0)


# ------------------------------------------------------------------ 磁盘状态


class TestCompactionOnDisk(CompactionTestCase):
    def test_old_files_are_deleted_after_compaction(self):
        with self.open_engine(auto_compact=False, l0_compaction_trigger=2) as db:
            for batch in range(4):
                db.put(f"k{batch}", "v")
                db.flush()

            before = set(self.sst_file_ids())
            db.compact()
            after = set(self.sst_file_ids())

            self.assertEqual(after, db.manifest.file_ids(), "磁盘文件和 manifest 不一致")
            self.assertTrue(before - after, "输入文件应该被删掉")
            self.assertTrue(after - before, "应该产出了新文件")

    def test_manifest_matches_files_on_disk(self):
        with self.open_engine(l0_compaction_trigger=3) as db:
            for batch in range(9):
                db.put(f"k{batch}", "v")
                db.flush()

            self.assertEqual(set(self.sst_file_ids()), db.manifest.file_ids())

    def test_manifest_is_not_written_for_empty_directory(self):
        """空目录打开后不该留下 manifest —— 目录保持"未初始化"。

        如果这里写了空 manifest,就等于把目录占下了:之后往里面放一批 .sst
        (比如从阶段 2 的目录拷过来),下次启动会跳过扫描、把它们当孤儿删掉。
        """
        with self.open_engine() as db:
            db.put("a", "1")        # 只在 WAL 和内存表里,还没刷盘
            self.assertFalse(
                self.manifest_path.exists(),
                "没有 SSTable 就不该有 manifest",
            )

    def test_manifest_is_created_on_first_flush(self):
        with self.open_engine() as db:
            self.assertFalse(self.manifest_path.exists())
            db.put("a", "1")
            db.flush()
            self.assertTrue(self.manifest_path.exists())

    def test_files_dropped_into_fresh_directory_are_adopted(self):
        """空目录被打开过、又被塞进 .sst —— 这些文件必须被接纳而不是删掉。"""
        with self.open_engine() as db:
            db.put("warmup", "1")       # 不刷盘,目录仍然没有 manifest
        self.assertFalse(self.manifest_path.exists())

        self.write_sstable(3, [(b"a", b"1"), (b"b", b"2")])

        with self.open_engine() as db:
            self.assertEqual(db.get_str("a"), "1")
            self.assertEqual(db.get_str("b"), "2")
            self.assertIn(3, db.manifest.file_ids())

    def test_leftover_tmp_files_are_cleaned_on_startup(self):
        """崩溃留下的半截文件必须被清掉,不能被当成有效数据。"""
        with self.open_engine() as db:
            db.put("a", "1")
            db.flush()

        stray = self.dir / "sst-999999.sst.tmp"
        stray.write_bytes(b"half written garbage")
        (self.dir / (MANIFEST_FILENAME + ".tmp")).write_bytes(b"{broken")

        with self.open_engine() as db:
            self.assertFalse(stray.exists())
            self.assertFalse((self.dir / (MANIFEST_FILENAME + ".tmp")).exists())
            self.assertEqual(db.get_str("a"), "1")

    def test_orphan_sst_files_are_removed_on_startup(self):
        """compaction 崩在"写完新文件、还没改 manifest"时会留下孤儿文件。

        它是垃圾不是数据 —— manifest 指向的那一套已经是完整的。
        """
        with self.open_engine() as db:
            db.put("a", "1")
            db.flush()

        self.write_sstable(999, [(b"zzz", b"orphan")])
        self.assertIn("sst-000999.sst", self.sst_files())

        with self.open_engine() as db:
            self.assertNotIn("sst-000999.sst", self.sst_files())
            self.assertEqual(db.get_str("a"), "1")

    def test_missing_referenced_file_is_a_hard_error(self):
        """manifest 引用的文件不见了 —— 宁可拒绝启动,也不能静默丢数据。"""
        with self.open_engine() as db:
            db.put("a", "1")
            db.flush()
            file_id = db.manifest.files()[0].file_id

        os.unlink(self.dir / sstable_filename(file_id))

        with self.assertRaises(CorruptionError) as ctx:
            self.open_engine()
        self.assertIn("不见了", str(ctx.exception))

    def test_restart_after_compaction_keeps_data(self):
        with self.open_engine(auto_compact=False, l0_compaction_trigger=2) as db:
            for batch in range(5):
                for i in range(10):
                    db.put(f"b{batch}-k{i:02d}", f"v{batch}-{i}")
                db.flush()
            db.compact_all()

        with self.open_engine() as db:
            self.assertEqual(db.stats().recovered_records, 0, "不该有 WAL 要重放")
            for batch in range(5):
                for i in range(10):
                    key = f"b{batch}-k{i:02d}"
                    self.assertEqual(db.get_str(key), f"v{batch}-{i}", key)

    def test_restart_after_compaction_preserves_layout(self):
        with self.open_engine(
            auto_compact=False, l0_compaction_trigger=2, num_levels=3
        ) as db:
            for batch in range(4):
                db.put(f"k{batch}", "v")
                db.flush()
            db.compact_all()
            before = self.level_layout(db)

        with self.open_engine() as db:
            self.assertEqual(self.level_layout(db), before)
            self.assertEqual(db.num_levels, 3)


# ------------------------------------------------------------------ 老目录兼容


class TestMigrationFromStage2(CompactionTestCase):
    def test_directory_without_manifest_is_adopted(self):
        """阶段 2 的目录(只有 .sst,没有 manifest)应该能直接打开。"""
        self.write_sstable(1, [(b"a", b"1"), (b"b", b"2")])
        self.write_sstable(2, [(b"c", b"3")])

        with self.open_engine() as db:
            self.assertTrue(self.manifest_path.exists(), "应该补出 manifest")
            self.assertEqual(db.get_str("a"), "1")
            self.assertEqual(db.get_str("b"), "2")
            self.assertEqual(db.get_str("c"), "3")
            # 老文件全部被当作 L0
            self.assertEqual(self.level_layout(db)[0], 2)

    def test_migrated_files_can_be_compacted(self):
        for file_id in range(1, 5):
            self.write_sstable(file_id, [(f"k{file_id}".encode(), b"v")])

        with self.open_engine(l0_compaction_trigger=2) as db:
            db.maybe_compact()
            db.manifest.check_invariants()
            for file_id in range(1, 5):
                self.assertEqual(db.get(f"k{file_id}"), b"v")

    def test_next_file_id_continues_after_migration(self):
        """迁移过来的 file_id 不能被复用 —— 复用会让新文件和旧文件撞名,

        而撞名的后果是"新文件覆盖旧文件"或"旧文件被当成孤儿删掉",两种都丢数据。
        """
        self.write_sstable(7, [(b"a", b"1")])

        with self.open_engine(auto_compact=False) as db:
            for batch in range(3):
                db.put(f"k{batch}", "v")
                db.flush()

            ids = sorted(db.manifest.file_ids())
            self.assertIn(7, ids, "迁移来的文件应该还在")
            self.assertEqual(len(ids), len(set(ids)))
            self.assertGreater(min(i for i in ids if i != 7), 7, "新文件的编号必须大于 7")
            self.assertEqual(set(self.sst_file_ids()), set(ids))


# ------------------------------------------------------------------ 边界


class TestCompactionEdges(CompactionTestCase):
    def test_compaction_on_empty_engine(self):
        with self.open_engine() as db:
            self.assertEqual(db.compact_all(), 0)
            self.assertEqual(db.maybe_compact(), 0)
            self.assertFalse(db.compact())

    def test_compaction_with_only_memtable_data(self):
        with self.open_engine() as db:
            db.put("a", "1")
            self.assertEqual(db.compact_all(), 0)
            self.assertEqual(db.get_str("a"), "1")

    def test_compaction_all_on_single_file(self):
        with self.open_engine(auto_compact=False) as db:
            db.put("a", "1")
            db.flush()
            moved = db.compact_all()
            self.assertEqual(moved, 1)
            self.assertEqual(db.get_str("a"), "1")

    def test_two_level_engine_compacts_directly_to_l1(self):
        with self.open_engine(num_levels=2, l0_compaction_trigger=2) as db:
            for batch in range(2):
                db.put(f"k{batch}", "v")
                db.flush()

            self.assertEqual(db.num_levels, 2)
            self.assertEqual(self.level_layout(db)[0], 0)
            self.assertGreater(self.level_layout(db)[1], 0)
            self.assertEqual(db.get_str("k0"), "v")
            self.assertEqual(db.get_str("k1"), "v")

    def test_rejects_too_few_levels(self):
        from mini_lsm.errors import InvalidArgumentError

        with self.assertRaises(InvalidArgumentError):
            self.open_engine(num_levels=1)

    def test_compaction_does_not_lose_unflushed_writes(self):
        """内存表里还没刷盘的数据,不能被 compaction 弄丢。"""
        with self.open_engine(auto_compact=False, l0_compaction_trigger=2) as db:
            for batch in range(2):
                db.put(f"flushed{batch}", "v")
                db.flush()

            db.put("pending", "still in memtable")
            db.compact()

            self.assertEqual(db.get_str("pending"), "still in memtable")
            self.assertEqual(db.get_str("flushed0"), "v")

    def test_compaction_writes_valid_sstables(self):
        """产出的文件必须能被独立打开并读出全部内容。"""
        with self.open_engine(auto_compact=False, l0_compaction_trigger=2) as db:
            for batch in range(3):
                db.put(f"k{batch}", f"v{batch}")
                db.flush()
            db.compact_all()

            for file_id in db.manifest.file_ids():
                with SSTableReader(self.dir / sstable_filename(file_id), file_id) as r:
                    self.assertGreater(r.entry_count, 0)

    def test_compaction_counter_increments(self):
        with self.open_engine(l0_compaction_trigger=2) as db:
            for batch in range(6):
                db.put(f"k{batch}", "v")
                db.flush()

            self.assertGreaterEqual(db.stats().compactions, 1)

    def test_double_compact_all_is_idempotent(self):
        with self.open_engine(auto_compact=False, l0_compaction_trigger=2) as db:
            for batch in range(3):
                db.put(f"k{batch}", "v")
                db.flush()

            db.compact_all()
            first = set(db.manifest.file_ids())

            db.compact_all()            # 已经压到底了,不该再动
            self.assertEqual(set(db.manifest.file_ids()), first)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
