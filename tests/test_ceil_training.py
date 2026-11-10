import unittest
import random
import torch

from scripts.ceil_training.common import (
    _balanced_sample_without_replacement,
    official_dpp_logdets,
    official_ranking_loss,
    split_train_validation,
)


class OfficialCEILMathTests(unittest.TestCase):
    def test_banking77_balanced_split_is_reproducible_and_disjoint(self):
        rows = [
            {"id": f"{label}-{index}", "gold": label, "input": {"text": f"{label}-{index}"}}
            for label, count in (("a", 8), ("b", 7), ("c", 6))
            for index in range(count)
        ]
        selected, remaining = _balanced_sample_without_replacement(
            rows, 9, random.Random(42)
        )
        repeated, _ = _balanced_sample_without_replacement(rows, 9, random.Random(42))
        self.assertEqual([row["id"] for row in selected], [row["id"] for row in repeated])
        self.assertEqual({row["gold"] for row in selected}, {"a", "b", "c"})
        self.assertEqual(len(selected), 9)
        self.assertFalse({row["id"] for row in selected} & {row["id"] for row in remaining})
        self.assertEqual(len(selected) + len(remaining), len(rows))

    def test_stratified_validation_covers_every_class(self):
        rows = [
            {"id": f"{label}-{index}", "gold": label}
            for label in (f"label-{number}" for number in range(77))
            for index in range(13)
        ][:1000]
        train, validation = split_train_validation(rows, 0.1, 42, stratify=True)
        self.assertEqual((len(train), len(validation)), (900, 100))
        self.assertEqual(len({row["gold"] for row in train}), 77)
        self.assertEqual(len({row["gold"] for row in validation}), 77)
        self.assertFalse({row["id"] for row in train} & {row["id"] for row in validation})

    def test_official_kernel_logdet_matches_manual_formula(self):
        query = torch.tensor([1.0, 0.0])
        contexts = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        sets = torch.tensor([[0, 1], [1, 2]])
        actual = official_dpp_logdets(query, contexts, sets, 0.1)
        relevance = (contexts @ query + 1) / 2
        quality = torch.exp((relevance - relevance.max()) / 0.2)
        similarity = (contexts @ contexts.T + 1) / 2
        kernel = quality[:, None] * similarity * quality[None, :]
        expected = torch.stack([torch.linalg.slogdet(kernel[idx][:, idx]).logabsdet for idx in sets])
        self.assertTrue(torch.allclose(actual, expected))

    def test_ranking_loss_uses_intermediate_candidates(self):
        good = torch.tensor([[3.0, 2.0, 1.0]], requires_grad=True)
        bad_middle = torch.tensor([[3.0, 0.0, 1.0]], requires_grad=True)
        self.assertGreater(official_ranking_loss(bad_middle, 0.02), official_ranking_loss(good, 0.02))
        official_ranking_loss(bad_middle, 0.02).backward()
        self.assertNotEqual(float(bad_middle.grad[0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
