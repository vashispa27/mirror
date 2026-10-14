"""Callable import and reply workflow for export-backed WhatsApp memory."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

from .memory import MEMORY_VERSION, build_memory_chunks, export_hash, extract_facts
from .memory_db import MemoryRepository
from .memory_retrieval import MemoryQdrantStore, MemoryRetriever, grounded_reply, retrieve_memory
from .parser import parse_whatsapp_chat
from .topic_memory import TOPIC_VERSION, CachedTopicModel, LazyLocalQwen, build_topics_and_spans, detect_boundaries
from .topic_store import TopicQdrantStore


def ingest_export(
    export_text: str, *, repository: MemoryRepository, vector_store: MemoryQdrantStore,
    user_id: str, contact_id: str, chat_id: str, owner_author: str | None = None,
    day_first: bool = True, timezone_name: str | None = None,
    defer_processing: bool = False, topic_model: Any = None,
) -> dict[str, int]:
    messages = parse_whatsapp_chat(export_text, chat_id=chat_id,
                                   day_first=day_first, timezone_name=timezone_name)
    repository.import_export(
        messages, user_id=user_id, contact_id=contact_id, chat_id=chat_id,
        export_hash=export_hash(export_text), owner_author=owner_author,
    )
    if defer_processing:
        return {"messages": len(messages), "topics": 0, "chunks": 0, "facts": 0, "memory_points": 0}
    processed = process_chat_memory(
        repository=repository, vector_store=vector_store,
        user_id=user_id, contact_id=contact_id, chat_id=chat_id,
        topic_model=topic_model,
    )
    return {"messages": len(messages), **processed}


def process_chat_memory(
    *, repository: MemoryRepository, vector_store: MemoryQdrantStore,
    user_id: str, contact_id: str, chat_id: str, topic_model: Any = None,
) -> dict[str, int]:
    """Build local-Qwen topics and raw spans, then index both in Qdrant v3."""

    rows = repository.messages_for_chat(user_id, contact_id, chat_id)
    usable = [r for r in rows if r["normalized_text"] and not
              (r["is_system"] or r["is_deleted"] or r["has_media"])]
    old_spans = repository.active_chunk_ids(user_id, contact_id, chat_id, TOPIC_VERSION)
    old_topics = repository.active_topic_ids(user_id, contact_id, chat_id, TOPIC_VERSION)
    if usable:
        model = CachedTopicModel(topic_model or LazyLocalQwen(), repository,
                                 user_id, contact_id, chat_id)
        boundaries = detect_boundaries(usable, model)
        topics, spans = build_topics_and_spans(usable, boundaries, model,
                                              user_id=user_id, contact_id=contact_id)
    else:
        topics, spans = [], []
    rows_by_id = {r["id"]: r for r in usable}
    for topic in topics:
        repository.upsert_topic(topic, TOPIC_VERSION)
    facts = []
    for span in spans:
        repository.upsert_chunk(span)
        extracted = extract_facts(span, [rows_by_id[mid] for mid in span.message_ids])
        repository.upsert_facts(extracted)
        facts.extend(extracted)
    store = TopicQdrantStore(client=vector_store.client)
    if topics or spans:
        store.index(topics, spans)
        for topic in topics:
            repository.mark_topic_indexed(topic.id)
        for span in spans:
            repository.mark_chunk_indexed(span.id)
    stale_spans = old_spans - {s.id for s in spans}
    stale_topics = old_topics - {t.id for t in topics}
    store.delete(stale_spans | stale_topics)
    repository.retire_chunks(stale_spans)
    repository.retire_topics(stale_topics)
    return {"topics": len(topics), "chunks": len(spans),
            "facts": len({f.id for f in facts}), "memory_points": len(topics) + len(spans)}


def process_chat_memory_v2(
    *, repository: MemoryRepository, vector_store: MemoryQdrantStore,
    user_id: str, contact_id: str, chat_id: str,
) -> dict[str, int]:
    """Previous MiniLM candidate + embedding verifier path for comparison."""

    # Rebuild from the persistent chat, so overlapping/partial exports converge
    # to one canonical set of chunks instead of adding parallel chunk copies.
    rows = repository.messages_for_chat(user_id, contact_id, chat_id)
    rows_by_id = {row["id"]: row for row in rows}
    old_ids = repository.active_chunk_ids(user_id, contact_id, chat_id, MEMORY_VERSION)
    chunks = build_memory_chunks(rows, user_id=user_id, contact_id=contact_id)
    facts = []
    for chunk in chunks:
        repository.upsert_chunk(chunk)
        extracted = extract_facts(chunk, [rows_by_id[mid] for mid in chunk.message_ids])
        repository.upsert_facts(extracted)
        facts.extend(extracted)
    if chunks:
        vector_store.index_chunks(chunks)
        for chunk in chunks:
            repository.mark_chunk_indexed(chunk.id)
    stale = old_ids - {chunk.id for chunk in chunks}
    vector_store.delete_chunks(stale)
    repository.retire_chunks(stale)
    return {"chunks": len(chunks), "facts": len({f.id for f in facts}),
            "memory_points": len(chunks)}


def reply_to_message(
    *, repository: MemoryRepository, retriever: MemoryRetriever,
    user_id: str, contact_id: str, chat_id: str, author: str,
    text: str, sent_at: datetime, source_message_id: str | None = None,
    provider: Callable[[str], str] | None = None,
    token_budget: int = 3000,
) -> dict[str, Any]:
    incoming_id = repository.add_message(
        user_id=user_id, contact_id=contact_id, chat_id=chat_id,
        text=text, author=author, direction="incoming", sent_at=sent_at,
        source_message_id=source_message_id,
    )
    context = retrieve_memory(repository, retriever, text, user_id=user_id,
                              contact_id=contact_id, chat_id=chat_id,
                              token_budget=token_budget)
    return {"reply": grounded_reply(context, provider),
            "incoming_message_id": str(incoming_id),
            "source_message_ids": list(context.source_message_ids),
            "fact_ids": [str(f["id"]) for f in context.facts],
            "chunk_ids": [h["chunk_id"] for h in context.chunks],
            "ambiguity": context.ambiguity}
