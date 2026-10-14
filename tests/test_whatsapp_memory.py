from __future__ import annotations

import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from whatsapp.memory import build_memory_chunks, extract_facts
from whatsapp.memory_retrieval import MemoryRetriever, grounded_reply, retrieve_memory


def row(text: str, *, author: str = "Aditi", contact: str = "deb", minute: int = 0):
    return dict(id=uuid.uuid4(), user_id="owner", contact_id=contact,
                chat_id="chat", sent_at=datetime(2025, 8, 8, 10, minute, tzinfo=timezone.utc),
                author=author, normalized_text=text, is_system=False,
                is_deleted=False, has_media=False)


class MemoryRecordTests(unittest.TestCase):
    def test_contact_only_assertion_has_provenance(self):
        rows = [row("Mom's birthday is 12 June."), row("Thanks!", author="Me", minute=1)]
        chunks = build_memory_chunks(rows, user_id="owner", contact_id="deb")
        self.assertEqual(1, len(chunks))
        self.assertEqual(tuple(r["id"] for r in rows), chunks[0].message_ids)
        facts = extract_facts(chunks[0], rows)
        self.assertEqual(1, len(facts))
        self.assertEqual((rows[0]["id"],), facts[0].message_ids)
        self.assertEqual("Aditi", facts[0].stated_by)
        self.assertEqual("--06-12", facts[0].normalized_value)

    def test_fact_extraction_avoids_hedges_negation_and_quotes(self):
        rows = [row("Maybe Mom's birthday is 12 June."),
                row("Mom's birthday is not 12 June.", minute=1),
                row('"Mom\'s birthday is 12 June"', minute=2)]
        chunks = build_memory_chunks(rows, user_id="owner", contact_id="deb")
        self.assertEqual([], extract_facts(chunks[0], rows))

    def test_exact_phone_address_and_identifier(self):
        rows = [row("Mom's phone is +91 98765 43210."),
                row("My address is 12 Lake Road, Pune.", minute=1),
                row("My passport number is AB1234567.", minute=2)]
        chunks = build_memory_chunks(rows, user_id="owner", contact_id="deb")
        facts = [f for chunk in chunks for f in extract_facts(chunk, rows)]
        self.assertEqual({"phone", "address", "identifier"}, {f.fact_type for f in facts})
        self.assertTrue(all(f.message_ids for f in facts))

    def test_semantic_boundary_hook_and_contact_isolation(self):
        rows = [row("My knee pain is bad."), row("The party needs cake.", minute=1)]
        # Two short turns have too little context for model-based detection.
        self.assertEqual(1, len(build_memory_chunks(rows, user_id="owner", contact_id="deb")))
        self.assertEqual(2, len(build_memory_chunks(
            rows, user_id="owner", contact_id="deb",
            topic_boundary_detector=lambda session: {1},
        )))
        with self.assertRaisesRegex(ValueError, "mixed contact"):
            build_memory_chunks(rows + [row("Other", contact="other")], user_id="owner", contact_id="deb")


class FakeQdrant:
    def __init__(self):
        self.requests = []

    def collection_exists(self, name):
        return True

    def query_points(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(points=[])


class MemoryRetrievalTests(unittest.TestCase):
    def test_hybrid_always_prefetches_dense_and_sparse_with_both_scope_keys(self):
        client = FakeQdrant()
        self.assertEqual([], MemoryRetriever(client=client).search("pain getting better", user_id="owner", contact_id="deb"))
        request = client.requests[0]
        self.assertEqual("rrf", request["query"].fusion.value)
        self.assertEqual(["dense", "sparse"], [p.using for p in request["prefetch"]])
        for prefetch in request["prefetch"]:
            fields = {c.key: c.match.value for c in prefetch.filter.must}
            self.assertEqual("owner", fields["user_id"])
            self.assertEqual("deb", fields["contact_id"])

    def test_conflicting_exact_facts_ask_for_clarification(self):
        class Repo:
            def recent_messages(self, *args, **kwargs):
                return []
            def find_facts(self, *args, **kwargs):
                self.assert_scope = args
                return [dict(id=uuid.uuid4(), subject_key="mom", subject_label="Mom",
                             fact_type="birthday", raw_value=value, normalized_value=value,
                             context="Mom's birthday is " + value, source_message_ids=[])
                        for value in ("12 June", "13 June")]
            def messages_by_ids(self, *args):
                return []
        context = retrieve_memory(Repo(), MemoryRetriever(client=FakeQdrant()),
                                  "What is Mom's birthday?", user_id="owner", contact_id="deb")
        self.assertIsNotNone(context.ambiguity)
        self.assertIn("conflicting", grounded_reply(context))

    def test_no_evidence_returns_unknown(self):
        class Repo:
            def recent_messages(self, *args, **kwargs): return []
            def find_facts(self, *args, **kwargs): return []
            def messages_by_ids(self, *args): return []
        context = retrieve_memory(Repo(), MemoryRetriever(client=FakeQdrant()),
                                  "Where is my passport?", user_id="owner", contact_id="deb")
        self.assertIn("don't have enough", grounded_reply(context))


if __name__ == "__main__":
    unittest.main()
