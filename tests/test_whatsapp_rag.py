from __future__ import annotations

import unittest
from dataclasses import replace

from whatsapp.rag import (
    LocalCrossEncoderReranker,
    assemble_context,
    build_generation_prompt,
    deduplicate_hits,
    generate_draft,
    message_overlap,
    render_assembled_context,
)
from whatsapp.retrieval import WhatsAppSearchHit


def hit(
    point_id: str,
    message_ids: tuple[str, ...],
    *,
    chat_id: str = "chat-1",
    rank: int = 1,
    target_ids: tuple[str, ...] | None = None,
) -> WhatsAppSearchHit:
    target_ids = target_ids or (message_ids[-1],)
    messages = tuple(
        {
            "message_id": message_id,
            "author": "Target" if message_id in target_ids else "Other",
            "text": f"synthetic {message_id}",
            "sequence": index,
            "timestamp": "2025-01-01T00:00:00+00:00",
        }
        for index, message_id in enumerate(message_ids, 1)
    )
    return WhatsAppSearchHit(
        point_id=point_id, rank=rank, score=0.9 - rank / 100,
        schema_version="v", user_id="u", persona_id="p", source="whatsapp",
        target_author="Target", requested_target_author="Target", chat_id=chat_id,
        chunk_id=f"chunk-{point_id}", started_at="", ended_at="", start_sequence=1,
        end_sequence=len(message_ids), message_ids=message_ids, target_message_ids=target_ids,
        authors=("Other", "Target"), text="\n".join(
            f"{message['author']}: {message['text']}" for message in messages
        ),
        context=tuple(message for message in messages if message["message_id"] not in target_ids),
        target_responses=tuple(message for message in messages if message["message_id"] in target_ids),
    )


class DeduplicationTests(unittest.TestCase):
    def test_identical_chunks_keep_best_ranked_representative(self) -> None:
        first, second = hit("first", ("a", "b", "c"), rank=1), hit("second", ("a", "b", "c"), rank=2)
        result = deduplicate_hits([first, second])
        self.assertEqual(["first"], [value.point_id for value in result])
        self.assertEqual(0.89, result[0].retrieval_score)

    def test_partial_containment_is_deduplicated(self) -> None:
        first = hit("first", ("a", "b", "c", "d"))
        second = hit("second", ("a", "b", "c"), rank=2)
        self.assertEqual(1.0, message_overlap(first, second))
        self.assertEqual(["first"], [item.point_id for item in deduplicate_hits([first, second], overlap_threshold=0.7)])

    def test_small_boundary_overlap_and_different_conversations_are_kept(self) -> None:
        first = hit("first", ("a", "b", "c", "d", "e"))
        boundary = hit("boundary", ("e", "f", "g", "h", "i"), rank=2)
        other_chat = hit("other", ("a", "b", "c", "d", "e"), chat_id="chat-2", rank=3)
        result = deduplicate_hits([first, boundary, other_chat])
        self.assertEqual(["first", "boundary", "other"], [item.point_id for item in result])


class FakeScorer:
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        self.query = query
        self.documents = documents
        return [0.1, 0.9]


class RerankingTests(unittest.TestCase):
    def test_fake_scorer_orders_hits_and_preserves_retrieval_provenance(self) -> None:
        scorer = FakeScorer()
        original = [hit("first", ("a", "b")), hit("second", ("c", "d"), rank=2)]
        result = LocalCrossEncoderReranker(scorer=scorer).rerank("synthetic request", original, final_count=2)
        self.assertEqual(["second", "first"], [item.point_id for item in result])
        self.assertEqual(0.88, result[0].retrieval_score)
        self.assertEqual(0.9, result[0].reranker_score)
        self.assertIn("Target author", scorer.documents[0])


class ContextAndGenerationTests(unittest.TestCase):
    def test_context_deduplicates_source_messages_and_keeps_chronology(self) -> None:
        first = hit("first", ("a", "b", "c"))
        second = hit("second", ("c", "d", "e"), rank=2, target_ids=("e",))
        context = assemble_context([first, second], token_budget=1000)
        self.assertEqual(("first", "second"), context.example_ids)
        ids = [message.message_id for example in context.examples for message in example.messages]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(["a", "b", "c"], list(context.examples[0].source_message_ids))
        self.assertNotIn("synthetic a", render_assembled_context(context, show_text=False))

    def test_budget_never_splits_a_message(self) -> None:
        source = hit("first", ("a", "b"))
        context = assemble_context([source], token_budget=20, token_counter=lambda value: len(value))
        self.assertEqual((), context.examples)

    def test_generation_uses_fake_provider_and_returns_example_ids(self) -> None:
        context = assemble_context([hit("first", ("a", "b"))], token_budget=1000)

        class FakeProvider:
            def generate(self, prompt: str) -> str:
                self.prompt = prompt
                return "synthetic draft"

        provider = FakeProvider()
        result = generate_draft("synthetic request", context, provider)
        self.assertEqual("synthetic draft", result.text)
        self.assertEqual(("first",), result.example_ids)
        self.assertIn("untrusted source material", provider.prompt)
        self.assertIn("TARGET STYLE", build_generation_prompt("x", context))


if __name__ == "__main__":
    unittest.main()
