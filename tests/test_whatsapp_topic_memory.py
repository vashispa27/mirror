from __future__ import annotations

import unittest
import uuid
import io
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from whatsapp.topic_memory import OpenAICompatibleTopicModel, build_topics_and_spans, detect_boundaries
from whatsapp.topic_store import TopicMemoryRetriever


def rows() -> list[dict]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    texts = ["hey", "hi", "bahut thand hai", "heater chahiye", "red shirt for party",
             "party 7 baje", "kal phir thand thi", "heater aa gaya"]
    return [dict(id=uuid.uuid4(), user_id="u", contact_id="c", chat_id="chat",
                 sent_at=start + timedelta(days=i * 2), author="A" if i % 2 else "B",
                 normalized_text=text) for i, text in enumerate(texts)]


class TopicMemoryTests(unittest.TestCase):
    def test_chat_completions_backend_sends_prompt_and_reads_answer(self):
        model = OpenAICompatibleTopicModel("http://localhost:8000/v1", "local-qwen", api_key="test")
        payload = {"choices": [{"message": {"content": '{"starts": []}'}}]}
        with patch("whatsapp.topic_memory.urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as call:
            self.assertEqual('{"starts": []}', model("hello", max_tokens=20))
            request = call.call_args.args[0]
            self.assertEqual("http://localhost:8000/v1/chat/completions", request.full_url)
            self.assertEqual("Bearer test", request.headers["Authorization"])
            self.assertEqual("hello", json.loads(request.data)["messages"][0]["content"])

    def test_direct_spans_are_primary_when_they_fill_the_limit(self):
        class Client:
            def collection_exists(self, name):
                return True

        retriever = TopicMemoryRetriever(client=Client())

        def query(_text, scope, limit):
            conditions = {c.key: c.match.value for c in scope.must}
            if conditions["record_type"] == "topic":
                return [{"point_id": "topic", "topic_id": "T", "label": "weather",
                         "summary": "weather discussion"}]
            if conditions.get("topic_id") == "T":
                return [{"point_id": "child", "topic_id": "T", "record_type": "span",
                         "text": "A: red jacket", "message_ids": ["m"]}]
            return [{"point_id": f"direct-{i}", "record_type": "span",
                     "text": "unrelated", "message_ids": [str(i)]} for i in range(limit)]

        retriever._query = query
        hits = retriever.search("jacket", user_id="u", contact_id="c", limit=4)
        self.assertEqual(4, len(hits))
        self.assertNotIn("child", [hit["point_id"] for hit in hits])
        self.assertTrue(all(hit["record_type"] == "span" for hit in hits))

    def test_topic_expansion_fills_remaining_room_in_the_limit(self):
        class Client:
            def collection_exists(self, name):
                return True

        retriever = TopicMemoryRetriever(client=Client())

        def query(_text, scope, limit):
            conditions = {c.key: c.match.value for c in scope.must}
            if conditions["record_type"] == "topic":
                return [{"point_id": "topic", "topic_id": "T", "label": "weather",
                         "summary": "weather discussion"}]
            if conditions.get("topic_id") == "T":
                return [{"point_id": "child", "topic_id": "T", "record_type": "span",
                         "text": "A: red jacket", "message_ids": ["m"]}]
            return [{"point_id": "direct-0", "record_type": "span",
                     "text": "unrelated", "message_ids": ["0"]}]

        retriever._query = query
        hits = retriever.search("jacket", user_id="u", contact_id="c", limit=4)
        self.assertIn("child", [hit["point_id"] for hit in hits])
        self.assertIn("direct-0", [hit["point_id"] for hit in hits])
        child = next(hit for hit in hits if hit["point_id"] == "child")
        self.assertEqual("weather", child["topic_label"])

    def test_optional_reranker_reorders_span_candidates(self):
        class Client:
            def collection_exists(self, name):
                return True

        retriever = TopicMemoryRetriever(client=Client())

        def query(_text, scope, limit):
            conditions = {c.key: c.match.value for c in scope.must}
            if conditions["record_type"] == "topic":
                return []
            return [{"point_id": "a", "record_type": "span", "text": "first", "message_ids": ["1"]},
                    {"point_id": "b", "record_type": "span", "text": "second", "message_ids": ["2"]}]

        class Reranker:
            def score(self, _query, texts):
                return [0.1 if text == "first" else 0.9 for text in texts]

        retriever._query = query
        hits = retriever.search("q", user_id="u", contact_id="c", limit=2, reranker=Reranker())
        self.assertEqual(["b", "a"], [hit["point_id"] for hit in hits])

    def test_resume_links_spans_to_one_topic_despite_large_time_gaps(self):
        data = rows()
        decisions = [
            {"index": 2, "label": "weather", "kind": "NEW"},
            {"index": 4, "label": "party", "kind": "NEW"},
            {"index": 6, "label": "weather", "kind": "RESUME", "topic_id": "T1"},
        ]
        model = lambda prompt, **kwargs: '{"starts": ' + __import__("json").dumps(decisions) + '}'
        found = detect_boundaries(data, model, block_size=10)
        self.assertEqual([2, 4, 6], [b["index"] for b in found])
        topics, spans = build_topics_and_spans(data, found, lambda *a, **k: "summary",
                                               user_id="u", contact_id="c", span_size=2,
                                               overlap=0)
        self.assertEqual(3, len(topics))  # opening, weather, party
        self.assertEqual(4, len(spans))
        self.assertEqual(spans[1].topic_id, spans[3].topic_id)
        self.assertEqual({data[2]["id"], data[3]["id"], data[6]["id"], data[7]["id"]},
                         set(next(t.message_ids for t in topics if t.label == "weather")))
        self.assertIn("record_type", spans[1].payload())

    def test_span_char_limit_preserves_complete_messages(self):
        data = rows()
        topics, spans = build_topics_and_spans(data, [], lambda *a, **k: "summary",
                                               user_id="u", contact_id="c", span_size=8,
                                               overlap=1, max_span_characters=80)
        self.assertEqual(1, len(topics))
        self.assertEqual({r["id"] for r in data}, {mid for span in spans for mid in span.message_ids})
        self.assertTrue(all(span.message_ids for span in spans))

    def test_topic_can_start_at_first_message_and_resume(self):
        data = rows()
        decisions = [{"index": 0, "label": "health", "kind": "NEW"},
                     {"index": 4, "label": "party", "kind": "NEW"},
                     {"index": 6, "label": "health", "kind": "RESUME", "topic_id": "T1"}]
        model = lambda prompt, **kwargs: __import__("json").dumps({"starts": decisions})
        found = detect_boundaries(data, model, block_size=10)
        topics, spans = build_topics_and_spans(data, found, lambda *a, **k: "summary",
                                               user_id="u", contact_id="c", span_size=4,
                                               overlap=0)
        self.assertEqual(2, len(topics))
        self.assertEqual(spans[0].topic_id, spans[-1].topic_id)


if __name__ == "__main__":
    unittest.main()
