"""CLI for label-safe WhatsApp retrieval evaluation."""

from __future__ import annotations

import argparse
import os

from .evaluation import evaluate_pipeline, load_evaluation_jsonl
from .rag import DEFAULT_CROSS_ENCODER_MODEL, LocalCrossEncoderReranker
from .retrieval import WhatsAppRetriever
from .store import DEFAULT_COLLECTION, DEFAULT_QDRANT_URL


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare WhatsApp retrieval pipeline stages.")
    parser.add_argument("dataset", help="Private labelled JSONL; it is read only.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--persona-id", required=True)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL))
    parser.add_argument("--chat-id")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=40)
    parser.add_argument("--overlap-threshold", type=float, default=0.7)
    parser.add_argument("--reranker-model", default=DEFAULT_CROSS_ENCODER_MODEL)
    parser.add_argument("--skip-rerank", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        report = evaluate_pipeline(
            load_evaluation_jsonl(args.dataset), WhatsAppRetriever(args.qdrant_url),
            user_id=args.user_id, persona_id=args.persona_id, collection_name=args.collection,
            k=args.k, candidate_count=args.candidate_count,
            overlap_threshold=args.overlap_threshold,
            reranker=None if args.skip_rerank else LocalCrossEncoderReranker(args.reranker_model),
            chat_id=args.chat_id,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        _parser().error(str(exc))
    print("stage\tsuccess@k\trecall@k\tmrr@k\tndcg@k\tduplicate_rate\tmean_latency_ms")
    for name, metric in report.metrics.items():
        print(
            f"{name}\t{metric.success_at_k:.4f}\t{metric.recall_at_k:.4f}\t"
            f"{metric.mrr_at_k:.4f}\t{metric.ndcg_at_k:.4f}\t"
            f"{metric.duplicate_rate:.4f}\t{metric.mean_latency_ms:.2f}"
        )
    if report.regressions:
        print("reranker regressions (private-text-safe):")
        for regression in report.regressions:
            print(
                f"  query_index={regression.query_index} "
                f"baseline_relevant_ranks={regression.baseline_relevant_ranks} "
                f"reranked_relevant_ranks={regression.reranked_relevant_ranks}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
