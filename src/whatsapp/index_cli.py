"""Command line entry point for idempotent WhatsApp-to-Qdrant indexing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .authors import (
    extract_authors,
    resolve_target_author,
    resolve_target_author_regex,
)
from .chunking import chunk_conversation
from .parser import parse_whatsapp_chat
from .persona import build_persona_records
from .store import DEFAULT_COLLECTION, DEFAULT_QDRANT_URL, WhatsAppQdrantStore


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve a WhatsApp persona and idempotently index it in Qdrant."
    )
    parser.add_argument("input", type=Path, help="WhatsApp .txt export")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--persona-id", required=True)
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--timezone", dest="timezone_name")
    parser.add_argument("--month-first", action="store_true")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL),
    )
    selectors = parser.add_mutually_exclusive_group()
    selectors.add_argument(
        "--target-author",
        help="Literal author selector (Unicode/case/whitespace normalized).",
    )
    selectors.add_argument(
        "--target-author-regex",
        help="Explicit regular-expression author selector; must match exactly one author.",
    )
    parser.add_argument(
        "--list-authors",
        action="store_true",
        help="Print non-system export authors and exit without indexing.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    export_text = args.input.read_text(encoding="utf-8-sig")
    messages = parse_whatsapp_chat(
        export_text,
        chat_id=args.chat_id,
        day_first=not args.month_first,
        timezone_name=args.timezone_name,
    )

    if args.list_authors:
        print(json.dumps({"authors": list(extract_authors(messages))}, ensure_ascii=False))
        return
    if args.target_author is None and args.target_author_regex is None:
        _parser().error("one of --target-author or --target-author-regex is required")

    if args.target_author_regex is not None:
        requested_selector = args.target_author_regex
        target_author = resolve_target_author_regex(messages, requested_selector)
    else:
        requested_selector = args.target_author
        target_author = resolve_target_author(messages, requested_selector)

    chunks = chunk_conversation(messages)
    records = build_persona_records(
        messages,
        chunks,
        user_id=args.user_id,
        persona_id=args.persona_id,
        requested_target_author=requested_selector,
        target_author=target_author,
    )
    result = WhatsAppQdrantStore(args.qdrant_url).index(
        args.collection,
        records,
        parsed_message_count=len(messages),
    )
    print(
        json.dumps(
            {
                **result.to_dict(),
                "collection": args.collection,
                "persona_id": args.persona_id,
                "target_author": target_author,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
