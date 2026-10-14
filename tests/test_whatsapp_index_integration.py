"""Docker-backed smoke coverage; excluded unless WHATSAPP_QDRANT_INTEGRATION=1."""

from __future__ import annotations

import importlib.util
import os
import unittest
import uuid

from whatsapp import build_persona_records, chunk_conversation, parse_whatsapp_chat
from whatsapp.authors import resolve_target_author


@unittest.skipUnless(
    os.environ.get("WHATSAPP_QDRANT_INTEGRATION") == "1"
    and importlib.util.find_spec("qdrant_client") is not None,
    "requires WHATSAPP_QDRANT_INTEGRATION=1 and qdrant-client[fastembed]",
)
class WhatsAppQdrantIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        from whatsapp.store import WhatsAppQdrantStore

        self.collection = f"whatsapp-index-smoke-{uuid.uuid4().hex}"
        self.store = WhatsAppQdrantStore(
            os.environ.get("QDRANT_URL", "http://localhost:6333")
        )

    def tearDown(self) -> None:
        self.store.client.delete_collection(self.collection)

    def test_indexes_dense_and_sparse_persona_records_idempotently(self) -> None:
        from qdrant_client import models
        from whatsapp.store import DENSE_VECTOR, SPARSE_VECTOR

        export = """[08/08/25, 10:00 AM] Rohan Verma: I write exactly like this 🙂
[08/08/25, 10:01 AM] Aditi: This is retained as conversation context.
[08/08/25, 10:02 AM] Rohan Verma: Another target answer."""
        messages = parse_whatsapp_chat(export, chat_id="integration-chat")
        target = resolve_target_author(messages, "rohan")
        records = build_persona_records(
            messages,
            chunk_conversation(messages),
            user_id="integration-user",
            persona_id="integration-persona",
            requested_target_author="rohan",
            target_author=target,
        )

        first = self.store.index(
            self.collection, records, parsed_message_count=len(messages)
        )
        self.assertEqual(1, first.indexed_point_count)
        info = self.store.client.get_collection(self.collection)
        self.assertIn(DENSE_VECTOR, info.config.params.vectors)
        self.assertIn(SPARSE_VECTOR, info.config.params.sparse_vectors)
        self.assertEqual(
            models.Distance.COSINE,
            info.config.params.vectors[DENSE_VECTOR].distance,
        )
        self.assertEqual(
            models.Modifier.IDF,
            info.config.params.sparse_vectors[SPARSE_VECTOR].modifier,
        )
        point = self.store.client.scroll(
            self.collection, limit=1, with_payload=True
        )[0][0]
        self.assertEqual("Rohan Verma", point.payload["target_author"])
        self.assertEqual("integration-persona", point.payload["persona_id"])
        self.assertEqual(2, len(point.payload["target_responses"]))

        search = self.store.client.query_points(
            collection_name=self.collection,
            query=models.Document(text="exactly like this", model="BAAI/bge-small-en-v1.5"),
            using=DENSE_VECTOR,
            limit=1,
            with_payload=True,
        )
        self.assertEqual(1, len(search.points))
        self.store.index(self.collection, records, parsed_message_count=len(messages))
        self.assertEqual(
            1, self.store.client.count(self.collection, exact=True).count
        )


if __name__ == "__main__":
    unittest.main()
