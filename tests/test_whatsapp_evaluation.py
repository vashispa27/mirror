from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from whatsapp.evaluation import (
    EvaluationQuery,
    duplicate_rate,
    evaluate_pipeline,
    load_evaluation_jsonl,
    score_hits,
)
from whatsapp.rag import LocalCrossEncoderReranker
from whatsapp.retrieval import WhatsAppSearchHit


def _hit(point_id: str, messages: tuple[str, ...]) -> WhatsAppSearchHit:
    return WhatsAppSearchHit(
        point_id=point_id, rank=1, score=1.0, schema_version="v", user_id="u", persona_id="p",
        source="whatsapp", target_author="Target", requested_target_author="Target", chat_id="c",
        chunk_id=f"chunk-{point_id}", started_at="", ended_at="", start_sequence=1, end_sequence=2,
        message_ids=messages, target_message_ids=(messages[-1],), authors=("Target",), text="",
        context=(), target_responses=(),
    )


class EvaluationTests(unittest.TestCase):
    def test_loads_private_jsonl_without_rewriting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text('{"query":"synthetic", "relevant_message_ids":["m1"]}\n', encoding="utf-8")
            values = load_evaluation_jsonl(path)
        self.assertEqual("synthetic", values[0].query)
        self.assertEqual(frozenset({"m1"}), values[0].relevant_ids)

    def test_metrics_and_duplicate_rate(self) -> None:
        hits = [_hit("one", ("m1", "m2")), _hit("two", ("m2", "m3"))]
        self.assertEqual((1.0, 1.0, 1.0, 1.0), score_hits(hits, frozenset({"m1"}), k=2))
        self.assertEqual(0.5, duplicate_rate(hits))

    def test_comparison_uses_same_k_and_reports_private_safe_regression(self) -> None:
        class Retriever:
            def search(self, *args, **kwargs):
                return [_hit("good", ("m1", "m2")), _hit("other", ("m3", "m4"))]

        class Scorer:
            def rerank(self, query, documents):
                return [0.1, 0.9]

        report = evaluate_pipeline(
            [EvaluationQuery("private query", frozenset({"m1"}))], Retriever(),
            user_id="u", persona_id="p", collection_name="synthetic", k=1,
            candidate_count=2,
            reranker=LocalCrossEncoderReranker(scorer=Scorer()),
        )
        self.assertEqual(
            {"hybrid_rrf", "hybrid_rrf_dedup", "hybrid_rrf_dedup_rerank"},
            set(report.metrics),
        )
        self.assertEqual((1,), report.regressions[0].baseline_relevant_ranks)
        self.assertEqual((), report.regressions[0].reranked_relevant_ranks)


if __name__ == "__main__":
    unittest.main()
