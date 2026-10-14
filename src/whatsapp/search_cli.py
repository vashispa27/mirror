"""CLI for inspecting scoped WhatsApp persona retrieval results."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .retrieval import (
    SEARCH_MODES,
    RetrievalError,
    WhatsAppRetriever,
    render_hits_human,
    serialize_hits_json,
)
from .rag import (
    DEFAULT_CROSS_ENCODER_MODEL,
    LocalCrossEncoderReranker,
    OpenAIProvider,
    assemble_context,
    deduplicate_hits,
    generate_draft,
    render_assembled_context,
)
from .store import DEFAULT_COLLECTION, DEFAULT_QDRANT_URL


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search one tenant-scoped WhatsApp persona index in Qdrant."
    )
    parser.add_argument("query", help="Natural-language or exact-phrase query")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--persona-id", required=True)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)
    )
    parser.add_argument("--mode", choices=SEARCH_MODES, default="hybrid_rrf")
    parser.add_argument(
        "--stage",
        choices=("raw", "dedup", "rerank", "context", "generate"),
        default="raw",
        help="Pipeline stage to inspect; generate is the only stage that calls an LLM.",
    )
    parser.add_argument("--limit", type=int, default=10, help="Raw-stage result count")
    parser.add_argument("--prefetch-limit", type=int, default=50)
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=40,
        help="First-stage candidates for deduplication/reranking (typically 30–50).",
    )
    parser.add_argument("--final-count", type=int, default=8)
    parser.add_argument("--overlap-threshold", type=float, default=0.7)
    parser.add_argument("--reranker-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--token-budget", type=int, default=1800)
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Include private chat text for context-stage inspection; hidden by default.",
    )
    parser.add_argument("--chat-id")
    parser.add_argument("--format", choices=("human", "json"), default="human")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        retrieval_limit = args.limit if args.stage == "raw" else args.candidate_count
        if args.candidate_count <= 0 or args.final_count <= 0:
            raise ValueError("candidate-count and final-count must be positive integers")
        hits = WhatsAppRetriever(args.qdrant_url).search(
            args.query,
            user_id=args.user_id,
            persona_id=args.persona_id,
            collection_name=args.collection,
            mode=args.mode,
            limit=retrieval_limit,
            prefetch_limit=max(args.prefetch_limit, retrieval_limit),
            chat_id=args.chat_id,
        )
        if args.stage == "dedup":
            hits = deduplicate_hits(hits, overlap_threshold=args.overlap_threshold)
        elif args.stage in ("rerank", "context", "generate"):
            hits = deduplicate_hits(hits, overlap_threshold=args.overlap_threshold)
            hits = LocalCrossEncoderReranker(args.reranker_model).rerank(
                args.query, hits, final_count=args.final_count
            )
    except (ValueError, RetrievalError) as exc:
        _parser().error(str(exc))
    except RuntimeError as exc:
        _parser().error(str(exc))

    if args.stage in ("context", "generate"):
        context = assemble_context(hits, token_budget=args.token_budget)
        if args.stage == "generate":
            try:
                draft = generate_draft(args.query, context, OpenAIProvider())
            except RuntimeError as exc:
                _parser().error(str(exc))
            if args.format == "json":
                print(json.dumps({"draft": draft.text, "example_ids": draft.example_ids}))
            else:
                print("Example IDs: " + ", ".join(draft.example_ids))
                print(draft.text)
            return 0
        if args.format == "json":
            output = {
                "token_budget": context.token_budget,
                "estimated_tokens": context.estimated_tokens,
                "examples": [
                    {
                        "example_id": example.example_id,
                        "point_id": example.point_id,
                        "chat_id": example.chat_id,
                        "target_author": example.target_author,
                        "source_message_ids": example.source_message_ids,
                        **(
                            {"messages": [message.__dict__ for message in example.messages]}
                            if args.show_text
                            else {}
                        ),
                    }
                    for example in context.examples
                ],
            }
            print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        else:
            print(render_assembled_context(context, show_text=args.show_text))
        return 0
    if args.format == "json":
        print(serialize_hits_json(args.query, args.mode, hits))
    else:
        print(render_hits_human(hits))
    return 0


if __name__ == "__main__":
    sys.exit(main())
