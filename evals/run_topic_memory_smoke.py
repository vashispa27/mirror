"""End-to-end synthetic Postgres/Qdrant smoke test; no real chat is read."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

from whatsapp.memory_db import MemoryRepository
from whatsapp.memory_retrieval import MemoryQdrantStore, retrieve_memory
from whatsapp.memory_service import ingest_export
from whatsapp.topic_memory import TOPIC_COLLECTION, TOPIC_VERSION
from whatsapp.topic_store import TopicMemoryRetriever

from run_hinglish_topic_demo import read_messages


USER = "synthetic-topic-v3-smoke"
CONTACT = "invented-asha-rohan"
CHAT = "invented-chat"


def export_text() -> str:
    started = datetime(2025, 8, 8, 10, 0)
    lines = []
    for index, body in read_messages():
        author, text = body.split(": ", 1)
        when = started + timedelta(minutes=index)
        lines.append(f"[{when.strftime('%d/%m/%y, %I:%M %p')}] {author}: {text}")
    return "\n".join(lines)


def main() -> None:
    repo = MemoryRepository(os.getenv("POSTGRES_DSN", "postgresql://mirror:mirror@localhost:5432/mirror"))
    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        repo.migrate()
        result = ingest_export(export_text(), repository=repo, vector_store=MemoryQdrantStore(url),
                               user_id=USER, contact_id=CONTACT, chat_id=CHAT)
        context = retrieve_memory(repo, TopicMemoryRetriever(url),
                                  "What color shirt did Asha buy for the party?",
                                  user_id=USER, contact_id=CONTACT, chat_id=CHAT)
        evidence = [r["normalized_text"] for r in context.source_messages]
        isolated = TopicMemoryRetriever(url).search("red shirt", user_id=USER,
                                                   contact_id="different-contact")
        stored = len(repo.messages_for_chat(USER, CONTACT, CHAT))
        active_spans = len(repo.active_chunk_ids(USER, CONTACT, CHAT, TOPIC_VERSION))
        active_topics = len(repo.active_topic_ids(USER, CONTACT, CHAT, TOPIC_VERSION))
        from qdrant_client import models
        scope = models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))
                                    for key, value in (("user_id", USER), ("contact_id", CONTACT))])
        points = TopicMemoryRetriever(url).client.count(TOPIC_COLLECTION, count_filter=scope).count
        passed = (any("red shirt" in text.lower() for text in evidence) and not isolated
                  and stored == 42 and active_spans == result["chunks"]
                  and active_topics == result["topics"] and points == result["memory_points"]
                  and not repo.recent_messages(USER, CONTACT, chat_id=CHAT))
        print(json.dumps({"result": result, "stored_messages": stored,
                          "active_topics": active_topics, "active_spans": active_spans,
                          "qdrant_points": points, "shirt_evidence": next(
                              (text for text in evidence if "red shirt" in text.lower()), None),
                          "contact_isolated": not isolated, "l0_empty": not repo.recent_messages(USER, CONTACT, chat_id=CHAT),
                          "passed": passed}, ensure_ascii=False, indent=2))
        if not passed:
            raise SystemExit(1)
    finally:
        repo.close()


if __name__ == "__main__":
    main()
