"""阶段 3 测试:Compaction 的策略层。

这个模块的测试完全不碰磁盘 —— 这正是"策略与机制分离"的好处:
    ``pick_task`` 只回答"该做什么",不负责写文件。于是各种边界
    (层超预算、层内重叠、最底层不触发)都能直接构造出来测,
    不用真的写几百 MB 数据把某一层撑爆。

策略层最容易错的两件事,测试里各有一组专门盯着:
    1. **输入顺序**。归并时同 key 取谁完全由输入顺序决定,排错了读到旧值。
    2. **什么时候能丢墓碑**。丢早了,更旧的数据立刻复活。
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.compaction import (  # noqa: E402
    DEFAULT_L0_TRIGGER,
    DEFAULT_LEVEL_BUDGET,
    DEFAULT_LEVEL_FACTOR,
    CompactionTask,
    level_budget,
    overlapping_files,
    pick_task,
    plan_full_compaction,
)
from mini_lsm.manifest import FileMeta, Manifest  # noqa: E402


def meta(file_id, level, smallest, largest, entries=10, size=1000) -> FileMeta:
    if isinstance(smallest, str):
        smallest = smallest.encode()
    if isinstance(largest, str):
        largest = largest.encode()
    return FileMeta(file_id, level, smallest, largest, entries, size)


def manifest(num_levels: int = 3) -> Manifest:
    # 只用来装内存状态,路径无所谓 —— 这个模块不该碰磁盘
    return Manifest(Path("unused"), num_levels)


# ------------------------------------------------------------------ 容量预算


class TestLevelBudget(unittest.TestCase):
    def test_l0_has_no_byte_budget(self):
        """L0 按文件数触发,不按字节 —— 所以预算返回 0。"""
        self.assertEqual(level_budget(0), 0)

    def test_l1_is_base(self):
        self.assertEqual(level_budget(1, base=100, factor=4), 100)

    def test_grows_geometrically(self):
        self.assertEqual(level_budget(2, base=100, factor=4), 400)
        self.assertEqual(level_budget(3, base=100, factor=4), 1600)
        self.assertEqual(level_budget(4, base=100, factor=4), 6400)

    def test_factor_one_means_flat(self):
        self.assertEqual(level_budget(5, base=100, factor=1), 100)

    def test_defaults_are_used(self):
        self.assertEqual(level_budget(1), DEFAULT_LEVEL_BUDGET)
        self.assertEqual(level_budget(2), DEFAULT_LEVEL_BUDGET * DEFAULT_LEVEL_FACTOR)


# ------------------------------------------------------------------ 重叠筛选


class TestOverlappingFiles(unittest.TestCase):
    def setUp(self):
        self.files = [
            meta(1, 1, "a", "c"),
            meta(2, 1, "f", "h"),
            meta(3, 1, "m", "p"),
        ]

    def test_empty_input(self):
        self.assertEqual(overlapping_files([], b"a", b"z"), [])

    def test_selects_only_overlapping(self):
        got = overlapping_files(self.files, b"b", b"g")
        self.assertEqual([fm.file_id for fm in got], [1, 2])

    def test_touching_endpoint_counts(self):
        """闭区间:正好搭在端点上也必须带上,否则会漏掉同 key 的数据。"""
        got = overlapping_files(self.files, b"c", b"f")
        self.assertEqual([fm.file_id for fm in got], [1, 2])

    def test_in_the_gap_selects_nothing(self):
        self.assertEqual(overlapping_files(self.files, b"d", b"e"), [])

    def test_covers_everything(self):
        got = overlapping_files(self.files, b"0", b"z")
        self.assertEqual([fm.file_id for fm in got], [1, 2, 3])

    def test_single_key_query(self):
        got = overlapping_files(self.files, b"g", b"g")
        self.assertEqual([fm.file_id for fm in got], [2])


# ------------------------------------------------------------------ 任务对象


class TestCompactionTask(unittest.TestCase):
    def test_inputs_order_is_source_then_target(self):
        """这个顺序决定归并时同 key 取谁 —— 排错了会读到旧值。"""
        task = CompactionTask(
            level=0,
            target_level=1,
            source_files=[meta(9, 0, "a", "z"), meta(8, 0, "a", "z")],
            target_files=[meta(3, 1, "a", "z")],
        )
        self.assertEqual([fm.file_id for fm in task.inputs], [9, 8, 3])

    def test_input_bytes_and_entries(self):
        task = CompactionTask(
            level=0,
            target_level=1,
            source_files=[meta(1, 0, "a", "z", entries=5, size=100)],
            target_files=[meta(2, 1, "a", "z", entries=7, size=250)],
        )
        self.assertEqual(task.input_bytes, 350)
        self.assertEqual(task.input_entries, 12)

    def test_empty_task_has_zero_totals(self):
        task = CompactionTask(level=0, target_level=1)
        self.assertEqual(task.inputs, [])
        self.assertEqual(task.input_bytes, 0)
        self.assertEqual(task.input_entries, 0)

    def test_drop_tombstones_defaults_false(self):
        """默认不丢墓碑 —— 安全的那一边做默认值。"""
        self.assertFalse(CompactionTask(level=0, target_level=1).drop_tombstones)

    def test_repr_mentions_tombstone_drop(self):
        task = CompactionTask(
            level=0, target_level=2,
            source_files=[meta(1, 0, "a", "z")],
            drop_tombstones=True,
        )
        self.assertIn("丢墓碑", repr(task))


# ------------------------------------------------------------------ 挑任务


class TestPickTaskL0(unittest.TestCase):
    def test_empty_manifest_needs_nothing(self):
        self.assertIsNone(pick_task(manifest()))

    def test_below_trigger_needs_nothing(self):
        m = manifest()
        for file_id in range(1, DEFAULT_L0_TRIGGER):
            m.add(meta(file_id, 0, "a", "z"))
        self.assertIsNone(pick_task(m))

    def test_at_trigger_picks_l0(self):
        m = manifest()
        for file_id in range(1, DEFAULT_L0_TRIGGER + 1):
            m.add(meta(file_id, 0, "a", "z"))

        task = pick_task(m)
        self.assertIsNotNone(task)
        self.assertEqual(task.level, 0)
        self.assertEqual(task.target_level, 1)
        self.assertEqual(len(task.source_files), DEFAULT_L0_TRIGGER)

    def test_l0_sources_are_newest_first(self):
        """源文件顺序必须是"新到旧",否则归并会挑出旧版本。"""
        m = manifest()
        for file_id in (1, 7, 3, 9):
            m.add(meta(file_id, 0, "a", "z"))

        task = pick_task(m, l0_trigger=2)
        self.assertEqual([fm.file_id for fm in task.source_files], [9, 7, 3, 1])

    def test_l0_includes_overlapping_target_files(self):
        """目标层里键范围重叠的文件必须一起归并,否则 L1 内部会重叠。"""
        m = manifest()
        for file_id in (1, 2, 3, 4):
            m.add(meta(file_id, 0, "a", "z"))
        m.add(meta(10, 1, "a", "c"))      # 与 L0 重叠
        m.add(meta(11, 1, "d", "f"))      # 与 L0 重叠
        m.add(meta(12, 1, "x", "z"))      # 与 L0 重叠

        task = pick_task(m)
        self.assertEqual([fm.file_id for fm in task.target_files], [10, 11, 12])

    def test_l0_excludes_non_overlapping_target_files(self):
        m = manifest()
        for file_id in (1, 2, 3, 4):
            m.add(meta(file_id, 0, "a", "c"))
        m.add(meta(10, 1, "a", "b"))      # 重叠
        m.add(meta(11, 1, "x", "z"))      # 不重叠

        task = pick_task(m)
        self.assertEqual([fm.file_id for fm in task.target_files], [10])

    def test_l0_tombstones_dropped_only_if_target_is_max_level(self):
        """num_levels=2 时 L1 就是最底层,此时可以丢墓碑。"""
        m2 = manifest(num_levels=2)
        for file_id in (1, 2, 3, 4):
            m2.add(meta(file_id, 0, "a", "z"))
        self.assertTrue(pick_task(m2).drop_tombstones)

        m3 = manifest(num_levels=3)
        for file_id in (1, 2, 3, 4):
            m3.add(meta(file_id, 0, "a", "z"))
        self.assertFalse(pick_task(m3).drop_tombstones)

    def test_l0_key_range_spans_all_files(self):
        """重叠筛选的范围要覆盖整个 L0,不能只看某一个文件。"""
        m = manifest()
        m.add(meta(1, 0, "a", "b"))
        m.add(meta(2, 0, "m", "n"))
        m.add(meta(3, 0, "x", "z"))
        m.add(meta(4, 0, "d", "e"))
        m.add(meta(10, 1, "a", "z"))

        task = pick_task(m)
        self.assertEqual([fm.file_id for fm in task.target_files], [10])

    def test_custom_trigger(self):
        m = manifest()
        m.add(meta(1, 0, "a", "z"))
        m.add(meta(2, 0, "a", "z"))
        self.assertIsNotNone(pick_task(m, l0_trigger=2))


class TestPickTaskLevelBudget(unittest.TestCase):
    def test_over_budget_picks_one_file(self):
        m = manifest()
        m.add(meta(1, 1, "a", "c", size=5000))
        m.add(meta(2, 1, "m", "n", size=5000))

        task = pick_task(m, level_budget_base=1000, level_size_factor=4)
        self.assertIsNotNone(task)
        self.assertEqual(task.level, 1)
        self.assertEqual(task.target_level, 2)
        self.assertEqual(len(task.source_files), 1)

    def test_within_budget_needs_nothing(self):
        m = manifest()
        m.add(meta(1, 1, "a", "c", size=10))
        self.assertIsNone(pick_task(m, level_budget_base=1000))

    def test_picks_file_with_fewest_overlaps(self):
        """要重写的数据量最小的那个 —— 最朴素但有效的启发式。"""
        m = manifest()
        # L1 两个文件都超过预算(合起来 > 1000)
        m.add(meta(10, 1, "a", "c", size=900))
        m.add(meta(11, 1, "x", "z", size=900))
        # L2 只与 a..c 重叠,不与 x..z 重叠
        m.add(meta(20, 2, "a", "c", size=100))

        task = pick_task(m, level_budget_base=1000, level_size_factor=4)
        self.assertEqual([fm.file_id for fm in task.source_files], [11])
        self.assertEqual(task.target_files, [])

    def test_includes_overlapping_target_files(self):
        m = manifest()
        m.add(meta(10, 1, "a", "f", size=5000))
        m.add(meta(20, 2, "a", "c", size=100))
        m.add(meta(21, 2, "d", "h", size=100))
        m.add(meta(22, 2, "x", "z", size=100))

        task = pick_task(m, level_budget_base=1000, level_size_factor=4)
        self.assertEqual([fm.file_id for fm in task.target_files], [20, 21])

    def test_max_level_never_triggers(self):
        """最底层没有下一层可压 —— 这是 maybe_compact 能收敛的根据。"""
        m = manifest(num_levels=3)
        m.add(meta(1, 2, "a", "z", size=10 ** 9))
        self.assertIsNone(pick_task(m, level_budget_base=1000))

    def test_l2_over_budget_triggers_when_not_max_level(self):
        m = manifest(num_levels=4)
        m.add(meta(1, 2, "a", "c", size=10 ** 6))
        m.add(meta(2, 3, "a", "c", size=100))

        task = pick_task(m, level_budget_base=1000, level_size_factor=4)
        self.assertEqual(task.level, 2)
        self.assertEqual(task.target_level, 3)

    def test_l0_takes_priority_over_budget_overflow(self):
        """L0 的重叠对读性能是乘性伤害,优先处理。"""
        m = manifest()
        for file_id in (1, 2, 3, 4):
            m.add(meta(file_id, 0, "a", "z"))
        m.add(meta(10, 1, "a", "z", size=10 ** 6))      # L1 严重超预算

        task = pick_task(m, level_budget_base=1000, level_size_factor=4)
        self.assertEqual(task.level, 0)

    def test_empty_level_is_not_over_budget(self):
        m = manifest()
        m.add(meta(1, 1, "a", "b", size=0))
        self.assertIsNone(pick_task(m, level_budget_base=1000))


class TestPickTaskTombstoneSafety(unittest.TestCase):
    """丢墓碑的判定条件是"输出层是最底层",这里把它钉死。"""

    def test_l1_to_l2_drops_when_l2_is_max(self):
        m = manifest(num_levels=3)
        m.add(meta(1, 1, "a", "c", size=5000))
        m.add(meta(2, 2, "a", "c", size=100))

        task = pick_task(m, level_budget_base=1000, level_size_factor=4)
        self.assertEqual(task.target_level, m.max_level)
        self.assertTrue(task.drop_tombstones)

    def test_l1_to_l2_keeps_when_l3_exists(self):
        m = manifest(num_levels=4)
        m.add(meta(1, 1, "a", "c", size=5000))
        m.add(meta(2, 2, "a", "c", size=100))

        task = pick_task(m, level_budget_base=1000, level_size_factor=4)
        self.assertLess(task.target_level, m.max_level)
        self.assertFalse(task.drop_tombstones)


# ------------------------------------------------------------------ 全量归并


class TestPlanFullCompaction(unittest.TestCase):
    def test_empty_manifest(self):
        self.assertIsNone(plan_full_compaction(manifest()))

    def test_single_file_at_max_level_needs_nothing(self):
        m = manifest()
        m.add(meta(1, 2, "a", "z"))
        self.assertIsNone(plan_full_compaction(m))

    def test_single_file_not_at_max_level_still_moves(self):
        m = manifest()
        m.add(meta(1, 0, "a", "z"))
        task = plan_full_compaction(m)
        self.assertIsNotNone(task)
        self.assertEqual(task.target_level, 2)

    def test_gathers_every_file_newest_first(self):
        m = manifest()
        m.add(meta(3, 0, "a", "z"))
        m.add(meta(1, 0, "a", "z"))
        m.add(meta(5, 1, "a", "c"))
        m.add(meta(4, 2, "m", "n"))

        task = plan_full_compaction(m)
        self.assertEqual([fm.file_id for fm in task.inputs], [3, 1, 5, 4])

    def test_targets_max_level_and_drops_tombstones(self):
        m = manifest(num_levels=4)
        m.add(meta(1, 0, "a", "z"))
        m.add(meta(2, 1, "a", "z"))

        task = plan_full_compaction(m)
        self.assertEqual(task.target_level, 3)
        self.assertTrue(task.drop_tombstones)
        self.assertEqual(task.target_files, [])

    def test_does_not_mutate_manifest(self):
        """规划是只读的 —— 不能顺手把 manifest 改了。"""
        m = manifest()
        m.add(meta(1, 0, "a", "z"))
        m.add(meta(2, 1, "a", "z"))
        before = m.to_dict()

        plan_full_compaction(m)
        self.assertEqual(m.to_dict(), before)


# ------------------------------------------------------------------ 收敛性


class TestConvergence(unittest.TestCase):
    def test_repeated_picking_eventually_stops(self):
        """把 pick_task 的结果真的应用上去,循环必须收敛。

        这是 maybe_compact 那个 while 循环能终止的根据:最底层不触发任务,
        所以每做一轮,数据只会往更深处走,不会原地打转。
        """
        m = manifest(num_levels=3)
        for file_id in range(1, 7):
            m.add(meta(file_id, 0, f"k{file_id:02d}", f"k{file_id:02d}z", size=3000))

        rounds = 0
        while True:
            task = pick_task(m, level_budget_base=1000, level_size_factor=4)
            if task is None:
                break
            rounds += 1
            self.assertLess(rounds, 50, "没有收敛,说明策略存在死循环")

            # 模拟执行:输入去掉,在目标层放一个合并后的文件
            merged = meta(
                100 + rounds, task.target_level,
                min(fm.smallest for fm in task.inputs),
                max(fm.largest for fm in task.inputs),
                size=sum(fm.size for fm in task.inputs) // 2,
            )
            m.replace(task.inputs, [merged])
            m.check_invariants()

        self.assertGreater(rounds, 0)
        m.check_invariants()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
