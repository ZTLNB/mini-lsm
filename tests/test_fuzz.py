"""随机压力测试:把引擎和内存里的字典模型逐步对照。

单元测试只能覆盖我**想到**的情况。这里用固定种子的随机操作序列去撞
我没想到的:同一个键被反复改写、删了又写、写到一半触发归并、随时重启。

每一步都和"如果只用内存里的字典"应该得到的结果对照 —— 包括点查、
全量扫描、以及 manifest 的层序不变量。

种子是固定的,所以一旦失败可以直接复现。
更重的一版在 ``tools/fuzz_compaction.py``(跑 40 个种子、900 步),
这里跑一个精简版,保证它适合放进日常回归。
"""

import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mini_lsm.engine import LSMEngine  # noqa: E402


class RandomWorkloadTest(unittest.TestCase):
    SEEDS = (0, 1, 2, 3)
    STEPS = 400

    def test_engine_matches_dict_model(self):
        for seed in self.SEEDS:
            with self.subTest(seed=seed):
                self._run_round(seed)

    def _run_round(self, seed: int) -> None:
        rng = random.Random(seed)
        tmp = Path(tempfile.mkdtemp(prefix="lsm-fuzz-"))
        self.addCleanup(shutil.rmtree, tmp, True)

        model: dict[bytes, bytes] = {}
        # 键空间故意开小,让"同一个键被反复触碰"成为常态 ——
        # 版本覆盖和墓碑的 bug 只在键被重复触碰时才暴露
        keys = [f"key{i:03d}".encode() for i in range(40)]

        def open_engine(levels: int) -> LSMEngine:
            engine = LSMEngine(
                tmp,
                memtable_capacity=1024,     # 开小,逼出大量刷盘
                sstable_block_size=512,
                l0_compaction_trigger=3,
                level_size_budget=2048,
                level_size_factor=2,
                target_file_size=1024,
                num_levels=levels,
            )
            self.addCleanup(engine.close)
            return engine

        db = open_engine(rng.choice([2, 3, 4]))

        def check(step: str) -> None:
            for key in keys:
                self.assertEqual(
                    db.get(key), model.get(key),
                    f"seed={seed} {step} key={key!r} 点查结果与模型不符",
                )
            self.assertEqual(
                dict(db.scan()), model,
                f"seed={seed} {step} 全量扫描与模型不符",
            )
            db.manifest.check_invariants()

        for step in range(self.STEPS):
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
                # 重启:验证 manifest + WAL 的恢复路径
                db.close()
                db = open_engine(rng.choice([2, 3, 4]))

            if step % 50 == 0:
                check(f"step={step}")

        check("final")

        # 再压一次到底 + 重启,确认"落盘后的状态"也是对的
        db.compact_all()
        db.close()
        db = open_engine(3)
        check("after-final-restart")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
