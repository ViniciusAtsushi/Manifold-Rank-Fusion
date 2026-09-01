"""Unit tests for the rank aggregation formulas (Borda Count, RRF, CombSUM).

Run with:
    python -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import unittest
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from manifold_rank_fusion.aggregation import (
    aggregate_batch,
    borda_count,
    combsum,
    reciprocal_rank_fusion,
)


def reference_scores(list_a, list_b, score_fn):
    """Independent reference computation: score_fn(rank, n) applied per position, summed across lists."""
    scores = defaultdict(float)
    for lst in (list_a, list_b):
        n = len(lst)
        for pos, item in enumerate(lst):
            scores[item] += score_fn(pos, n)
    return scores


def ranking_from_scores(scores):
    return [item for item, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


class TestBordaCount(unittest.TestCase):
    def setUp(self):
        self.list_a = [1, 2, 3, 4, 5]
        self.list_b = [1, 3, 5, 2, 4]

    def test_matches_formula(self):
        expected_scores = reference_scores(self.list_a, self.list_b, lambda pos, n: n - pos - 1)
        expected_ranking = ranking_from_scores(expected_scores)
        self.assertEqual(borda_count(self.list_a, self.list_b), expected_ranking)

    def test_explicit_order(self):
        # Hand-derived from the formula: scores are {1: 8, 2: 4, 3: 5, 4: 1, 5: 2}.
        self.assertEqual(borda_count(self.list_a, self.list_b), [1, 3, 2, 5, 4])

    def test_identical_lists_preserve_order(self):
        self.assertEqual(borda_count(self.list_a, self.list_a), self.list_a)

    def test_top_k_truncation(self):
        self.assertEqual(borda_count(self.list_a, self.list_b, top_k=2), [1, 3])

    def test_raises_on_length_mismatch(self):
        with self.assertRaises(ValueError):
            borda_count([1, 2, 3], [1, 2])

    def test_empty_input(self):
        self.assertEqual(borda_count(), [])


class TestReciprocalRankFusion(unittest.TestCase):
    def setUp(self):
        self.list_a = [1, 2, 3, 4, 5]
        self.list_b = [1, 3, 5, 2, 4]

    def test_matches_formula_default_k(self):
        expected_scores = reference_scores(self.list_a, self.list_b, lambda pos, n: 1.0 / (60 + pos + 1))
        expected_ranking = ranking_from_scores(expected_scores)
        self.assertEqual(reciprocal_rank_fusion(self.list_a, self.list_b), expected_ranking)

    def test_matches_formula_custom_k(self):
        k = 10
        expected_scores = reference_scores(self.list_a, self.list_b, lambda pos, n: 1.0 / (k + pos + 1))
        expected_ranking = ranking_from_scores(expected_scores)
        self.assertEqual(reciprocal_rank_fusion(self.list_a, self.list_b, k=k), expected_ranking)

    def test_identical_lists_preserve_order(self):
        self.assertEqual(reciprocal_rank_fusion(self.list_a, self.list_a), self.list_a)


class TestCombSum(unittest.TestCase):
    def setUp(self):
        self.list_a = [1, 2, 3, 4, 5]
        self.list_b = [1, 3, 5, 2, 4]

    def test_matches_formula(self):
        expected_scores = reference_scores(self.list_a, self.list_b, lambda pos, n: 1.0 / (pos + 1))
        expected_ranking = ranking_from_scores(expected_scores)
        self.assertEqual(combsum(self.list_a, self.list_b), expected_ranking)

    def test_identical_lists_preserve_order(self):
        self.assertEqual(combsum(self.list_a, self.list_a), self.list_a)


class TestAggregateBatch(unittest.TestCase):
    def test_two_queries_borda(self):
        batch_a = [[1, 2, 3], [3, 2, 1]]
        batch_b = [[1, 2, 3], [1, 2, 3]]

        result = aggregate_batch(batch_a, batch_b, method="borda", top_k=3)

        self.assertEqual(result[0], borda_count(batch_a[0], batch_b[0]))
        self.assertEqual(result[1], borda_count(batch_a[1], batch_b[1]))

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            aggregate_batch([[1, 2]], [[1, 2]], method="not-a-method")

    def test_mismatched_query_count_raises(self):
        with self.assertRaises(ValueError):
            aggregate_batch([[1, 2]], [[1, 2], [2, 1]], method="borda")

    def test_single_batch_raises(self):
        with self.assertRaises(ValueError):
            aggregate_batch([[1, 2]], method="borda")


if __name__ == "__main__":
    unittest.main()
