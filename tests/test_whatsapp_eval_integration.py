"""Docker-backed evaluation coverage using a synthetic temporary collection.

The test indexes only its own synthetic records into a randomly named
collection, runs the read-only evaluator, checks the report, and removes that
one collection.  It never reads or writes the real WhatsApp or SciFact
collections.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
import uuid
from pathlib import Path

from whatsapp import build_persona_records, chunk_conversation, parse_whatsapp_chat
from whatsapp.authors import resolve_target_author
from whatsapp.eval import run_evaluation
from whatsapp.retrieval import WhatsAppRetriever
from whatsapp.store import WhatsAppQdrantStore

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synthetic_whatsapp_chat.txt"
LABELS = Path(__file__).resolve().parents[1] / "evals" / "whatsapp.synthetic.jsonl"
FIXTURE_CHAT_ID = "eval-synthetic"


@unittest.skipUnless(
    os.environ.get("WHATSAPP_QDRANT_INTEGRATION") == "1"
    and importlib.util.find_spec("qdrant_client") is not None,
    "requires WHATSAPP_QDRANT_INTEGRATION=1 and qdrant-client[fastembed]",
)
class WhatsAppEvaluationIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.collection = f"whatsapp-eval-smoke-{uuid.uuid4().hex}"
        self.store = WhatsAppQdrantStore(
            os.environ.get("QDRANT_URL", "http://localhost:6333")
        )
        self.retriever = WhatsAppRetriever(client=self.store.client)

    def tearDown(self) -> None:
        if self.store.client.collection_exists(self.collection):
            self.store.client.delete_collection(self.collection)

    def _index_fixture(self) -> None:
        messages = parse_whatsapp_chat(
            FIXTURE.read_text(encoding="utf-8"), chat_id=FIXTURE_CHAT_ID
        )
        target_author = resolve_target_author(messages, "Replica A")
        records = build_persona_records(
            messages,
            chunk_conversation(messages, max_gap_minutes=30),
            user_id="eval-user",
            persona_id="replica-a",
            requested_target_author="Replica A",
            target_author=target_author,
        )
        self.store.index(self.collection, records, parsed_message_count=len(messages))

    def test_evaluator_compares_modes_without_mutating_the_collection(self) -> None:
        self._index_fixture()
        before = self.store.client.get_collection(self.collection).points_count

        report = run_evaluation(
            dataset_path=str(LABELS),
            retriever=self.retriever,
            collection_name=self.collection,
            k=5,
            prefetch_limit=10,
        )

        self.assertEqual(
            {"dense", "sparse", "hybrid_rrf", "hybrid_dbsf"}, set(report.aggregate)
        )
        self.assertEqual(6, report.config["query_count"])
        self.assertEqual(6, len(report.per_query))
        self.assertEqual(
            {"exact_phrase", "paraphrase", "situation", "style"},
            set(report.by_query_type),
        )
        for aggregate in report.aggregate.values():
            self.assertEqual(6, aggregate.queries)
            self.assertGreaterEqual(aggregate.success_at_k, 0.0)
            self.assertLessEqual(aggregate.success_at_k, 1.0)
            self.assertGreater(aggregate.latency_p95_ms, 0.0)
        for query_report in report.per_query:
            for mode_result in query_report.modes.values():
                self.assertLessEqual(len(mode_result.results), 5)
                for hit in mode_result.results:
                    self.assertTrue(hit.point_id)
                    self.assertTrue(hit.chunk_id)
                    self.assertTrue(hit.target_message_ids)

        # The rare exact phrase must be recoverable by BM25 sparse retrieval.
        exact = next(
            query for query in report.per_query if query.query_type == "exact_phrase"
        )
        self.assertEqual(1.0, exact.modes["sparse"].metrics.success_at_k)
        self.assertEqual((1,), exact.modes["sparse"].metrics.relevant_ranks)

        # Read-only guarantee: evaluation must not change the collection.
        after = self.store.client.get_collection(self.collection).points_count
        self.assertEqual(before, after)

    def test_missing_collection_is_a_clear_error(self) -> None:
        from whatsapp.retrieval import CollectionMissingError

        with self.assertRaises(CollectionMissingError):
            run_evaluation(
                dataset_path=str(LABELS),
                retriever=self.retriever,
                collection_name=f"does-not-exist-{uuid.uuid4().hex}",
                k=5,
                prefetch_limit=10,
            )

    def test_dataset_scope_mismatch_is_rejected_before_search(self) -> None:
        self._index_fixture()
        with self.assertRaisesRegex(ValueError, "does not match"):
            run_evaluation(
                dataset_path=str(LABELS),
                retriever=self.retriever,
                collection_name=self.collection,
                k=5,
                prefetch_limit=10,
                user_id="someone-else",
                persona_id="replica-a",
            )


if __name__ == "__main__":
    unittest.main()
