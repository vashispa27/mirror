"""CLI for read-only comparison of the four WhatsApp retrieval modes.

Example::

    .venv/bin/python -m whatsapp.eval_cli \
      --dataset evals/whatsapp.example.jsonl \
      --collection mirror_whatsapp_v2 \
      --k 5 \
      --prefetch-limit 30 \
      --format table
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Sequence

from .eval import (
    DEFAULT_EVIDENCE_SCOPE,
    DEFAULT_K,
    DEFAULT_OVERLAP_THRESHOLD,
    DEFAULT_PREFETCH_LIMIT,
    EVIDENCE_SCOPES,
    DatasetError,
    report_to_dict,
    render_table,
    run_evaluation,
)
from .retrieval import SEARCH_MODES, RetrievalError, WhatsAppRetriever
from .store import DEFAULT_COLLECTION, DEFAULT_QDRANT_URL


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only retrieval evaluation across dense, sparse, hybrid_rrf, "
            "and hybrid_dbsf for one indexed WhatsApp persona."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Labelled JSONL with query, user_id, persona_id, "
        "expected_target_message_ids, and optional query_type.",
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)
    )
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Rank cutoff (default 5)")
    parser.add_argument(
        "--prefetch-limit",
        type=int,
        default=DEFAULT_PREFETCH_LIMIT,
        help="Per-vector candidate pool for hybrid modes (must be >= k).",
    )
    parser.add_argument(
        "--overlap-threshold",
        type=float,
        default=DEFAULT_OVERLAP_THRESHOLD,
        help="Shared-message fraction at which a result counts as a duplicate.",
    )
    parser.add_argument(
        "--evidence-scope",
        choices=EVIDENCE_SCOPES,
        default=DEFAULT_EVIDENCE_SCOPE,
        help="Which messages in a result count as evidence: any_author (default, "
        "needed for facts stated by either speaker) or target_only.",
    )
    parser.add_argument(
        "--modes",
        default=",".join(SEARCH_MODES),
        help=f"Comma-separated subset of: {', '.join(SEARCH_MODES)}",
    )
    parser.add_argument(
        "--user-id",
        help="Optional scope override; must match every dataset row when supplied.",
    )
    parser.add_argument("--persona-id", help="Optional scope override (with --user-id).")
    parser.add_argument("--chat-id", help="Optional extra chat scope.")
    parser.add_argument(
        "--format", choices=("table", "json"), default="table", help="Report format."
    )
    return parser


def _parse_modes(raw: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not modes:
        raise ValueError("--modes must list at least one retrieval mode")
    return modes


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        modes = _parse_modes(args.modes)
        report = run_evaluation(
            dataset_path=args.dataset,
            retriever=WhatsAppRetriever(args.qdrant_url),
            collection_name=args.collection,
            k=args.k,
            prefetch_limit=args.prefetch_limit,
            modes=modes,
            overlap_threshold=args.overlap_threshold,
            evidence_scope=args.evidence_scope,
            user_id=args.user_id,
            persona_id=args.persona_id,
            chat_id=args.chat_id,
        )
    except (DatasetError, ValueError, RetrievalError, OSError) as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        parser.error(str(exc))

    if args.format == "json":
        print(json.dumps(report_to_dict(report), ensure_ascii=False, sort_keys=True))
    else:
        print(render_table(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
