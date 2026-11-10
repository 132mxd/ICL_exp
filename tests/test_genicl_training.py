"""回归测试：验证论文算法 1 强制要求的 Ld -> La 更新顺序。"""

import unittest

from scripts.genicl_training.alternating import (
    FixedBatchSampler,
    build_alternating_plan,
)


class GenICLAlternatingTrainingTests(unittest.TestCase):
    def setUp(self):
        self.preferences = [
            {
                "query": f"query-{index}",
                "gold": "yes",
                "preferred": [f"good-{index}-a", f"good-{index}-b"],
                "nonpreferred": [f"bad-{index}-a", f"bad-{index}-b"],
            }
            for index in range(10)
        ]

    def test_every_query_batch_updates_ld_then_la_without_mixing(self):
        plan = build_alternating_plan(
            self.preferences,
            labels=("yes", "no"),
            latent="<toy-0><toy-1>",
            record_batch_size=32,
            seed=7,
            shuffle_queries=False,
        )

        self.assertEqual(plan.stages, ["demonstration", "answer", "demonstration", "answer"])
        self.assertEqual(len(plan.batches), 2 * plan.query_batches)
        for batch_number, indices in enumerate(plan.batches):
            self.assertEqual(len(indices), 32)
            rows = [plan.rows[index] for index in indices]
            expected_stage = "demonstration" if batch_number % 2 == 0 else "answer"
            if expected_stage == "demonstration":
                self.assertTrue(all(row["completion"] == "<toy-0><toy-1>" for row in rows))
            else:
                self.assertTrue(all(row["prompt"].startswith("<toy-0><toy-1>") for row in rows))
            self.assertIn(True, {row["label"] for row in rows})
            self.assertIn(False, {row["label"] for row in rows})

    def test_kto_kl_chunks_never_cross_ld_la_stage_boundaries(self):
        plan = build_alternating_plan(
            self.preferences, ("yes", "no"), "<z>", 32, 7, False
        )
        # TRL 0.12.2 按 per_device_train_batch_size 对连续数据块构造 KL 配对，
        # 因此数据集布局与优化器批次必须是完全相同的 32 行数据块。
        contiguous_chunks = [
            list(range(start, start + 32)) for start in range(0, len(plan.rows), 32)
        ]
        self.assertEqual(plan.batches, contiguous_chunks)
        # 使用两张 GPU 时，TRL 会对每个 16 行本地数据块执行预处理和参考配对；
        # 每个本地数据块也必须完整地位于同一个阶段内。
        for start in range(0, len(plan.rows), 16):
            local_chunk = set(range(start, start + 16))
            self.assertTrue(any(local_chunk.issubset(set(batch)) for batch in plan.batches))

    def test_fixed_batch_sampler_replays_exact_optimizer_schedule(self):
        batches = [[4, 1], [3], [8, 2]]
        sampler = FixedBatchSampler(batches)
        self.assertEqual(list(sampler), batches)
        self.assertEqual(len(sampler), 3)

    def test_schedule_is_seed_reproducible_but_query_shuffled(self):
        first = build_alternating_plan(
            self.preferences, ("yes", "no"), "<z>", 16, 99, True
        )
        repeated = build_alternating_plan(
            self.preferences, ("yes", "no"), "<z>", 16, 99, True
        )
        unshuffled = build_alternating_plan(
            self.preferences, ("yes", "no"), "<z>", 16, 99, False
        )
        self.assertEqual(first.rows, repeated.rows)
        self.assertNotEqual(first.rows, unshuffled.rows)


if __name__ == "__main__":
    unittest.main()
