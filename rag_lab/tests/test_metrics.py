from __future__ import annotations

import unittest

from rag_lab.metrics import aggregate_metrics, score_query


class MetricsTests(unittest.TestCase):
    def test_scores_ranked_retrieval(self) -> None:
        score = score_query(
            ["wrong", "relevant-a", "relevant-b"],
            {"relevant-a": 1, "relevant-b": 1},
            k=3,
        )

        success, recall, mrr, ndcg = score
        self.assertEqual(1.0, success)
        self.assertEqual(1.0, recall)
        self.assertEqual(0.5, mrr)
        self.assertGreater(ndcg, 0.0)
        self.assertLess(ndcg, 1.0)

    def test_aggregates_multiple_queries(self) -> None:
        metrics = aggregate_metrics(
            [(1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0)],
            k=10,
        )

        self.assertEqual(2, metrics.queries)
        self.assertEqual(0.5, metrics.success_at_k)
        self.assertEqual(0.5, metrics.recall_at_k)


if __name__ == "__main__":
    unittest.main()

