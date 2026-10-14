"""Run with WHATSAPP_MEMORY_INTEGRATION=1 and local Postgres/Qdrant."""

from __future__ import annotations

import json
import os
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path

from whatsapp.memory_db import MemoryRepository
from whatsapp.memory_retrieval import MemoryQdrantStore, retrieve_memory
from whatsapp.topic_store import TopicMemoryRetriever
from whatsapp.memory_service import ingest_export
from whatsapp.memory import export_hash
from whatsapp.parser import parse_whatsapp_chat


@unittest.skipUnless(os.getenv("WHATSAPP_MEMORY_INTEGRATION") == "1", "requires local Postgres and Qdrant")
class MemoryIntegrationTests(unittest.TestCase):
    def test_labelled_memory_queries_and_idempotent_scoping(self):
        user = "memory-test-" + uuid.uuid4().hex
        repo = MemoryRepository(os.getenv("POSTGRES_DSN", "postgresql://mirror:mirror@localhost:5432/mirror"))
        store = MemoryQdrantStore(os.getenv("QDRANT_URL", "http://localhost:6333"))
        retriever = TopicMemoryRetriever(client=store.client)
        export = """[08/08/25, 10:00 AM] Aditi: Mom's birthday is 12 June.
[08/08/25, 10:01 AM] Me: Thanks.
[08/08/25, 10:02 AM] Aditi: The party is on 20 June.
[08/08/25, 10:03 AM] Aditi: Bring the cake to the party.
[08/08/25, 10:04 AM] Aditi: My knee has been hurting since the fall.
[08/08/25, 10:05 AM] Me: Rest your knee.
[08/08/25, 10:06 AM] Aditi: The code is Zorblax-17."""
        try:
            repo.migrate()
            first = ingest_export(export, repository=repo, vector_store=store,
                                  user_id=user, contact_id="deb", chat_id="chat", owner_author="Me")
            second = ingest_export(export, repository=repo, vector_store=store,
                                   user_id=user, contact_id="deb", chat_id="chat", owner_author="Me")
            self.assertEqual(first, second)
            self.assertEqual(7, len(repo.messages_for_chat(user, "deb", "chat")))
            self.assertEqual([], repo.recent_messages(user, "deb", chat_id="chat"))
            self.assertEqual(2, len(repo.find_facts(user, "deb")))
            self.assertEqual(first["chunks"], len(repo.active_chunk_ids(user, "deb", "chat")))
            partial = "\n".join(export.splitlines()[:2])
            ingest_export(partial, repository=repo, vector_store=store,
                          user_id=user, contact_id="deb", chat_id="chat", owner_author="Me")
            self.assertEqual(7, len(repo.messages_for_chat(user, "deb", "chat")))
            repeated = "[08/08/25, 11:00 AM] Aditi: yes\n[08/08/25, 11:00 AM] Aditi: yes"
            parsed = parse_whatsapp_chat(repeated, chat_id="repeated")
            for _ in range(2):
                imported = repo.import_export(parsed, user_id=user, contact_id="deb",
                                              chat_id="repeated", export_hash=export_hash(repeated))
                self.assertEqual(2, len({m["id"] for m in imported}))
            self.assertEqual(2, len(repo.messages_for_chat(user, "deb", "repeated")))
            labels = [json.loads(line) for line in
                      (Path(__file__).parents[1] / "evals/whatsapp_memory.synthetic.jsonl").read_text().splitlines()]
            for label in labels:
                with self.subTest(label=label["kind"]):
                    context = retrieve_memory(repo, retriever, label["question"],
                                              user_id=user, contact_id="deb")
                    if label["kind"] in ("exact", "contextual"):
                        self.assertTrue(any(f["subject_key"] == label["expected_subject"] and
                                            f["fact_type"] == label["expected_type"] and
                                            f["raw_value"] == label["expected_value"] for f in context.facts))
                    elif label["kind"] in ("semantic", "lexical"):
                        self.assertTrue(any(label["expected_text"].casefold() in h["text"].casefold()
                                            for h in context.chunks))
                    else:
                        self.assertFalse(context.facts)
            other = "[08/08/25, 11:00 AM] Other: Mom's birthday is 7 July."
            ingest_export(other, repository=repo, vector_store=store,
                          user_id=user, contact_id="other", chat_id="other-chat")
            self.assertTrue(all(h["contact_id"] == "deb" for h in retriever.search(
                "Mom birthday", user_id=user, contact_id="deb")))
            self.assertEqual("12 June", repo.find_facts(user, "deb", subject_key="mom")[0]["raw_value"])
            self.assertEqual("7 July", repo.find_facts(user, "other", subject_key="mom")[0]["raw_value"])
            repo.add_message(user_id=user, contact_id="deb", chat_id="chat", text="Hello",
                             author="Aditi", direction="incoming", sent_at=datetime.now(timezone.utc),
                             source_message_id="live-in")
            repo.add_message(user_id=user, contact_id="deb", chat_id="chat", text="Hi",
                             author="Me", direction="outgoing", sent_at=datetime.now(timezone.utc),
                             source_message_id="live-out")
            self.assertEqual(["Hello", "Hi"], [m["normalized_text"] for m in
                             repo.recent_messages(user, "deb", chat_id="chat")])
            self.assertEqual([], repo.recent_messages(user, "other", chat_id="other-chat"))
            non_text = "[08/08/25, 11:00 AM] Messages and calls are end-to-end encrypted.\n[08/08/25, 11:01 AM] Other: <Media omitted>"
            ingest_export(non_text, repository=repo, vector_store=store,
                          user_id=user, contact_id="other-extra", chat_id="non-text")
            saved = repo.messages_for_chat(user, "other-extra", "non-text")
            self.assertEqual(2, len(saved))
            self.assertTrue(saved[0]["is_system"])
            self.assertTrue(saved[1]["has_media"])
            self.assertTrue(all(m["original_content"] for m in saved))
        finally:
            repo.close()


if __name__ == "__main__":
    unittest.main()
