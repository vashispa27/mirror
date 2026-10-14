"""Docker-backed scoped retrieval coverage using synthetic records only."""

from __future__ import annotations

import importlib.util
import os
import unittest
import uuid

from whatsapp import build_persona_records, chunk_conversation, parse_whatsapp_chat
from whatsapp.authors import resolve_target_author
from whatsapp.rag import assemble_context, deduplicate_hits
from whatsapp.retrieval import WhatsAppRetriever
from whatsapp.store import WhatsAppQdrantStore


@unittest.skipUnless(
    os.environ.get("WHATSAPP_QDRANT_INTEGRATION") == "1"
    and importlib.util.find_spec("qdrant_client") is not None,
    "requires WHATSAPP_QDRANT_INTEGRATION=1 and qdrant-client[fastembed]",
)
class WhatsAppRetrievalIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.collection = f"whatsapp-retrieval-smoke-{uuid.uuid4().hex}"
        self.store = WhatsAppQdrantStore(
            os.environ.get("QDRANT_URL", "http://localhost:6333")
        )
        self.retriever = WhatsAppRetriever(client=self.store.client)

    def tearDown(self) -> None:
        if self.store.client.collection_exists(self.collection):
            self.store.client.delete_collection(self.collection)

    @staticmethod
    def _records(
        export: str,
        *,
        chat_id: str,
        user_id: str,
        persona_id: str,
        selector: str,
    ):
        messages = parse_whatsapp_chat(export, chat_id=chat_id)
        target = resolve_target_author(messages, selector)
        return build_persona_records(
            messages,
            chunk_conversation(messages, max_gap_minutes=30),
            user_id=user_id,
            persona_id=persona_id,
            requested_target_author=selector,
            target_author=target,
        )

    def test_all_modes_retrieve_scoped_synthetic_persona_records(self) -> None:
        relevant = self._records(
            """[08/08/25, 10:00 AM] Friend: I am anxious and overwhelmed today.
[08/08/25, 10:01 AM] Replica A: Slow your breathing and take one kind step. I am here with you.""",
            chat_id="support-chat",
            user_id="user-one",
            persona_id="replica-a",
            selector="Replica A",
        )
        rare_phrase = self._records(
            """[08/08/25, 11:00 AM] Friend: Do you remember the code word?
[08/08/25, 11:01 AM] Replica A: Yes, the orchid-velvet-quantum phrase is ours.""",
            chat_id="inside-jokes",
            user_id="user-one",
            persona_id="replica-a",
            selector="Replica A",
        )
        other_persona = self._records(
            """[08/08/25, 10:00 AM] Friend: I am anxious and overwhelmed today.
[08/08/25, 10:01 AM] Replica B: Buy a telescope and count stars.""",
            chat_id="other-chat",
            user_id="user-two",
            persona_id="replica-b",
            selector="Replica B",
        )
        all_records = [*relevant, *rare_phrase, *other_persona]
        self.store.index(
            self.collection, all_records, parsed_message_count=6
        )

        semantic_results = {}
        for mode in ("dense", "sparse", "hybrid_rrf", "hybrid_dbsf"):
            semantic_results[mode] = self.retriever.search(
                "anxious overwhelmed",
                user_id="user-one",
                persona_id="replica-a",
                collection_name=self.collection,
                mode=mode,
                limit=2,
                prefetch_limit=4,
            )
            self.assertTrue(semantic_results[mode])
            for hit in semantic_results[mode]:
                self.assertEqual("user-one", hit.user_id)
                self.assertEqual("replica-a", hit.persona_id)
                self.assertEqual("whatsapp", hit.source)
                self.assertTrue(hit.message_ids)
                self.assertTrue(hit.target_message_ids)
                self.assertTrue(hit.target_responses)

        for mode in ("dense", "hybrid_rrf", "hybrid_dbsf"):
            hits = self.retriever.search(
                "How do you comfort somebody who is stressed?",
                user_id="user-one",
                persona_id="replica-a",
                collection_name=self.collection,
                mode=mode,
                limit=2,
                prefetch_limit=4,
            )
            self.assertTrue(
                any(
                    "Slow your breathing" in hit.text
                    for hit in hits
                ),
                msg=f"{mode} did not retrieve the semantic paraphrase",
            )

        for mode in ("sparse", "hybrid_rrf", "hybrid_dbsf"):
            hits = self.retriever.search(
                "orchid-velvet-quantum",
                user_id="user-one",
                persona_id="replica-a",
                collection_name=self.collection,
                mode=mode,
                limit=2,
                prefetch_limit=4,
            )
            self.assertTrue(
                any("orchid-velvet-quantum" in hit.text for hit in hits),
                msg=f"{mode} did not retrieve the exact rare phrase",
            )

    def test_deduplication_and_context_use_synthetic_temporary_collection(self) -> None:
        messages = "\n".join(
            f"[08/08/25, 10:{minute:02d} AM] {'Replica A' if minute % 2 else 'Friend'}: synthetic message {minute}"
            for minute in range(12)
        )
        # Build deliberately small chunks so Qdrant receives
        # overlapping source message IDs, without touching any user collection.
        parsed = parse_whatsapp_chat(messages, chat_id="overlap-chat")
        records = build_persona_records(
            parsed,
            chunk_conversation(parsed, max_messages=5, overlap_messages=3, max_gap_minutes=30),
            user_id="user-one", persona_id="replica-a",
            requested_target_author="Replica A", target_author="Replica A",
        )
        self.store.index(self.collection, records, parsed_message_count=len(parsed))
        raw = self.retriever.search(
            "synthetic message", user_id="user-one", persona_id="replica-a",
            collection_name=self.collection, mode="hybrid_rrf", limit=10, prefetch_limit=10,
        )
        deduplicated = deduplicate_hits(raw, overlap_threshold=0.7)
        context = assemble_context(deduplicated, token_budget=1_000)
        source_ids = [
            message_id for example in context.examples for message_id in example.source_message_ids
        ]
        self.assertTrue(deduplicated)
        self.assertEqual(len(source_ids), len(set(source_ids)))
        self.assertTrue(all(example.messages for example in context.examples))


if __name__ == "__main__":
    unittest.main()
