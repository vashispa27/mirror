"""Backfill an export, or inspect a contact-scoped memory answer."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .memory_db import MemoryRepository
from .memory_retrieval import MemoryQdrantStore, retrieve_memory
from .memory_service import ingest_export, process_chat_memory, reply_to_message
from .parser import parse_whatsapp_chat
from .topic_memory import LocalQwen, detect_boundaries
from .topic_store import TopicMemoryRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="WhatsApp contact memory")
    parser.add_argument("--postgres-dsn", default=os.getenv("POSTGRES_DSN", "postgresql://mirror:mirror@localhost:5432/mirror"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--user-id")
    parser.add_argument("--contact-id")
    parser.add_argument("--chat-id")
    sub = parser.add_subparsers(dest="command", required=True)
    import_parser = sub.add_parser("import")
    import_parser.add_argument("input", type=Path)
    import_parser.add_argument("--owner-author")
    import_parser.add_argument("--timezone")
    import_parser.add_argument("--month-first", action="store_true")
    import_parser.add_argument("--defer", action="store_true", help="Persist raw messages now; process finalized chunks on idle later")
    sub.add_parser("process")
    preview_parser = sub.add_parser("preview-topics")
    preview_parser.add_argument("input", type=Path)
    preview_parser.add_argument("--timezone")
    preview_parser.add_argument("--month-first", action="store_true")
    search_parser = sub.add_parser("search", help="Read evidence without adding a live message")
    search_parser.add_argument("question")
    search_parser.add_argument("--chunk-limit", type=int, default=None,
                               help="Maximum memory spans to return (default 8)")
    search_parser.add_argument("--rerank", action="store_true",
                               help="Reorder memory spans with the local cross-encoder before truncating")
    ask_parser = sub.add_parser("ask")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--author", default="Contact")
    ask_parser.add_argument("--message-id")
    ask_parser.add_argument("--generate", action="store_true", help="Use the optional configured OpenAI provider")
    args = parser.parse_args()
    if args.command == "preview-topics":
        messages = parse_whatsapp_chat(args.input.read_text(encoding="utf-8-sig"),
                                       chat_id=args.chat_id or "preview", day_first=not args.month_first,
                                       timezone_name=args.timezone)
        rows = [dict(author=m.author, normalized_text=m.text) for m in messages
                if m.text and not (m.is_system or m.is_deleted or m.has_media)]
        result = detect_boundaries(rows, LocalQwen()) if rows else []
        print(json.dumps({"messages": len(rows), "boundaries": result}, ensure_ascii=False, indent=2))
        return
    for name in ("user_id", "contact_id", "chat_id"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required for {args.command}")
    repository = MemoryRepository(args.postgres_dsn)
    try:
        repository.migrate()
        if args.command == "import":
            result = ingest_export(
                args.input.read_text(encoding="utf-8-sig"), repository=repository,
                vector_store=MemoryQdrantStore(args.qdrant_url),
                user_id=args.user_id, contact_id=args.contact_id, chat_id=args.chat_id,
                owner_author=args.owner_author, day_first=not args.month_first,
                timezone_name=args.timezone, defer_processing=args.defer,
            )
        elif args.command == "process":
            result = process_chat_memory(
                repository=repository, vector_store=MemoryQdrantStore(args.qdrant_url),
                user_id=args.user_id, contact_id=args.contact_id, chat_id=args.chat_id,
            )
        elif args.command == "search":
            reranker = None
            if args.rerank:
                from .rag import LocalCrossEncoderReranker
                reranker = LocalCrossEncoderReranker()
            extra = {"chunk_limit": args.chunk_limit} if args.chunk_limit else {}
            context = retrieve_memory(
                repository, TopicMemoryRetriever(args.qdrant_url), args.question,
                user_id=args.user_id, contact_id=args.contact_id, chat_id=args.chat_id,
                reranker=reranker, **extra,
            )
            result = {"facts": [{"type": f["fact_type"], "value": f["raw_value"],
                                 "source_message_ids": [str(i) for i in f["source_message_ids"]]}
                                for f in context.facts],
                      "spans": [{"topic_id": h.get("topic_id"), "topic_label": h.get("topic_label"),
                                 "topic_summary": h.get("topic_summary"), "text": h["text"],
                                 "source_message_ids": h["message_ids"]} for h in context.chunks],
                      "source_messages": [{"id": str(r["id"]), "author": r["author"],
                                           "text": r["normalized_text"]} for r in context.source_messages]}
        else:
            if args.generate:
                from .rag import OpenAIProvider
                provider = OpenAIProvider().generate
            else:
                provider = None
            result = reply_to_message(
                repository=repository, retriever=TopicMemoryRetriever(args.qdrant_url),
                user_id=args.user_id, contact_id=args.contact_id, chat_id=args.chat_id,
                author=args.author, text=args.question,
                sent_at=datetime.now(timezone.utc), source_message_id=args.message_id,
                provider=provider,
            )
        print(json.dumps(result, ensure_ascii=False))
    finally:
        repository.close()


if __name__ == "__main__":
    main()
