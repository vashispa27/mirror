from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from whatsapp.persona import SCHEMA_VERSION
from whatsapp.retrieval import (
    CollectionIncompatibleError,
    CollectionMissingError,
    WhatsAppRetriever,
    build_required_filter,
    point_to_search_hit,
    render_hits_human,
    serialize_hits_json,
    validate_search_arguments,
)
from whatsapp.store import DENSE_MODEL, DENSE_VECTOR, SPARSE_MODEL, SPARSE_VECTOR


def _payload() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "user_id": "user-42",
        "persona_id": "rohan-verma",
        "source": "whatsapp",
        "target_author": "Rohan Verma",
        "requested_target_author": "Rohan",
        "chat_id": "personal-chat",
        "chunk_id": "chunk-7",
        "started_at": "2025-08-08T10:00:00+05:30",
        "ended_at": "2025-08-08T10:02:00+05:30",
        "start_sequence": 1,
        "end_sequence": 3,
        "message_ids": ["message-1", "message-2", "message-3"],
        "target_message_ids": ["message-2"],
        "authors": ["Aditi", "Rohan Verma"],
        "text": "Aditi: I am upset\nRohan Verma: I am here for you.",
        "context": [
            {
                "message_id": "message-1",
                "author": "Aditi",
                "text": "I am upset",
                "sequence": 1,
                "timestamp": "2025-08-08T10:00:00+05:30",
            }
        ],
        "target_responses": [
            {
                "message_id": "message-2",
                "author": "Rohan Verma",
                "text": "I am here for you.",
                "sequence": 2,
                "timestamp": "2025-08-08T10:01:00+05:30",
            }
        ],
    }


class FakeClient:
    def __init__(self, *, exists: bool = True, compatible: bool = True) -> None:
        self.exists = exists
        self.compatible = compatible
        self.calls: list[dict] = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def get_collection(self, collection_name: str):
        vectors = {DENSE_VECTOR: object()} if self.compatible else {}
        sparse = {SPARSE_VECTOR: object()} if self.compatible else {}
        return SimpleNamespace(
            config=SimpleNamespace(params=SimpleNamespace(vectors=vectors, sparse_vectors=sparse))
        )

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=[SimpleNamespace(id="point-1", score=0.75, payload=_payload())])


class SearchArgumentTests(unittest.TestCase):
    def test_rejects_empty_and_unscoped_searches(self) -> None:
        valid = ("question", "user", "persona", "dense", 1, 1)
        for replacement in ("", "  "):
            with self.assertRaisesRegex(ValueError, "query"):
                validate_search_arguments(replacement, *valid[1:])
        with self.assertRaisesRegex(ValueError, "user_id"):
            validate_search_arguments("question", " ", "persona", "dense", 1, 1)
        with self.assertRaisesRegex(ValueError, "persona_id"):
            validate_search_arguments("question", "user", None, "dense", 1, 1)

    def test_rejects_invalid_mode_and_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode"):
            validate_search_arguments("q", "u", "p", "manual", 1, 1)
        with self.assertRaisesRegex(ValueError, "limit"):
            validate_search_arguments("q", "u", "p", "dense", 0, 1)
        with self.assertRaisesRegex(ValueError, "prefetch_limit"):
            validate_search_arguments("q", "u", "p", "dense", 1, 0)
        with self.assertRaisesRegex(ValueError, "greater"):
            validate_search_arguments("q", "u", "p", "dense", 2, 1)


class QueryConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = FakeClient()
        self.retriever = WhatsAppRetriever(client=self.client)

    def _search(self, mode: str, **kwargs) -> dict:
        self.retriever.search(
            "comforting a friend",
            user_id="user-42",
            persona_id="rohan-verma",
            mode=mode,  # type: ignore[arg-type]
            limit=2,
            prefetch_limit=3,
            **kwargs,
        )
        return self.client.calls[-1]

    def test_required_filter_has_all_identity_fields(self) -> None:
        query_filter = build_required_filter("user-42", "rohan-verma")
        self.assertEqual(
            {
                condition.key: condition.match.value
                for condition in query_filter.must
            },
            {
                "user_id": "user-42",
                "persona_id": "rohan-verma",
                "source": "whatsapp",
                "schema_version": SCHEMA_VERSION,
            },
        )

    def test_optional_chat_filter_is_added(self) -> None:
        request = self._search("dense", chat_id="chat-22")
        conditions = {condition.key: condition.match.value for condition in request["query_filter"].must}
        self.assertEqual("chat-22", conditions["chat_id"])

    def test_dense_query_uses_dense_document_and_mandatory_filter(self) -> None:
        request = self._search("dense")
        self.assertEqual(DENSE_VECTOR, request["using"])
        self.assertEqual(DENSE_MODEL, request["query"].model)
        self.assertFalse(request["with_vectors"])
        self.assertEqual("user-42", request["query_filter"].must[0].match.value)

    def test_sparse_query_uses_bm25_document(self) -> None:
        request = self._search("sparse")
        self.assertEqual(SPARSE_VECTOR, request["using"])
        self.assertEqual(SPARSE_MODEL, request["query"].model)

    def test_rrf_hybrid_prefetches_both_vectors_with_scoping(self) -> None:
        request = self._search("hybrid_rrf")
        self.assertEqual("rrf", request["query"].fusion.value)
        self.assertEqual([DENSE_VECTOR, SPARSE_VECTOR], [p.using for p in request["prefetch"]])
        self.assertTrue(all(p.filter is request["query_filter"] for p in request["prefetch"]))

    def test_dbsf_hybrid_prefetches_both_vectors_with_scoping(self) -> None:
        request = self._search("hybrid_dbsf")
        self.assertEqual("dbsf", request["query"].fusion.value)
        self.assertEqual(3, request["prefetch"][0].limit)

    def test_missing_and_incompatible_collections_are_explicit(self) -> None:
        with self.assertRaises(CollectionMissingError):
            WhatsAppRetriever(client=FakeClient(exists=False)).search(
                "q", user_id="u", persona_id="p"
            )
        with self.assertRaises(CollectionIncompatibleError):
            WhatsAppRetriever(client=FakeClient(compatible=False)).search(
                "q", user_id="u", persona_id="p"
            )


class RenderingTests(unittest.TestCase):
    def test_point_parsing_and_human_json_serialization(self) -> None:
        hit = point_to_search_hit(
            SimpleNamespace(id="point-1", score=0.75, payload=_payload()), rank=1
        )
        self.assertEqual("rohan-verma", hit.persona_id)
        self.assertEqual(("message-2",), hit.target_message_ids)
        human = render_hits_human([hit])
        self.assertIn("Target responses:", human)
        self.assertIn("I am here for you.", human)
        output = json.loads(serialize_hits_json("comfort", "hybrid_rrf", [hit]))
        self.assertEqual("hybrid_rrf", output["mode"])
        self.assertEqual("point-1", output["hits"][0]["point_id"])
        self.assertEqual("No results.", render_hits_human([]))


if __name__ == "__main__":
    unittest.main()
