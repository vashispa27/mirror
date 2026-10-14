"""Private-label-safe retrieval evaluation for WhatsApp post-retrieval stages."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .rag import LocalCrossEncoderReranker, deduplicate_hits
from .retrieval import WhatsAppRetriever, WhatsAppSearchHit


@dataclass(frozen=True)
class EvaluationQuery:
    query: str
    relevant_ids: frozenset[str]


@dataclass(frozen=True)
class PipelineMetrics:
    queries: int
    success_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    duplicate_rate: float
    mean_latency_ms: float


@dataclass(frozen=True)
class Regression:
    """A label-safe comparison row; deliberately contains no query/chat text."""

    query_index: int
    baseline_relevant_ranks: tuple[int, ...]
    reranked_relevant_ranks: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationReport:
    metrics: Mapping[str, PipelineMetrics]
    regressions: tuple[Regression, ...]


def load_evaluation_jsonl(path: str | Path) -> list[EvaluationQuery]:
    """Load common labelled JSONL forms without ever rewriting the source file."""

    values: list[EvaluationQuery] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at evaluation line {line_number}: {exc}") from exc
        query = row.get("query", row.get("question"))
        labels = next(
            (row.get(key) for key in (
                "relevant_message_ids", "relevant_chunk_ids", "relevant_point_ids", "relevant_ids", "relevant"
            ) if row.get(key) is not None),
            None,
        )
        if not isinstance(query, str) or not query.strip() or not isinstance(labels, (list, tuple, set)):
            raise ValueError(
                f"Evaluation line {line_number} needs query/question and a list of relevant_* IDs"
            )
        values.append(EvaluationQuery(query, frozenset(str(value) for value in labels)))
    if not values:
        raise ValueError("Evaluation JSONL contains no labelled queries")
    return values


def _hit_ids(hit: WhatsAppSearchHit) -> set[str]:
    return {hit.point_id, hit.chunk_id, *hit.message_ids}


def _relevance(hit: WhatsAppSearchHit, labels: frozenset[str]) -> int:
    return int(bool(_hit_ids(hit) & labels))


def duplicate_rate(hits: Sequence[WhatsAppSearchHit]) -> float:
    if not hits:
        return 0.0
    seen: set[str] = set()
    duplicates = 0
    for hit in hits:
        if set(hit.message_ids) & seen:
            duplicates += 1
        seen.update(hit.message_ids)
    return duplicates / len(hits)


def score_hits(hits: Sequence[WhatsAppSearchHit], labels: frozenset[str], *, k: int) -> tuple[float, float, float, float]:
    selected = hits[:k]
    relevances = [_relevance(hit, labels) for hit in selected]
    success = float(any(relevances))
    recovered_ids: set[str] = set()
    for hit in selected:
        recovered_ids.update(_hit_ids(hit) & labels)
    recall = len(recovered_ids) / len(labels) if labels else 0.0
    mrr = next((1 / index for index, relevance in enumerate(relevances, 1) if relevance), 0.0)
    dcg = sum(value / math.log2(index + 1) for index, value in enumerate(relevances, 1))
    ideal = min(k, len(labels))
    ideal_dcg = sum(1 / math.log2(index + 1) for index in range(1, ideal + 1))
    return success, recall, mrr, dcg / ideal_dcg if ideal_dcg else 0.0


def relevant_ranks(hits: Sequence[WhatsAppSearchHit], labels: frozenset[str], *, k: int) -> tuple[int, ...]:
    """Return label matches as ranks only, safe to print with a private dataset."""

    return tuple(index for index, hit in enumerate(hits[:k], 1) if _relevance(hit, labels))


def aggregate_metrics(values: Iterable[tuple[tuple[float, float, float, float], float, float]]) -> PipelineMetrics:
    rows = list(values)
    if not rows:
        return PipelineMetrics(0, 0, 0, 0, 0, 0, 0)
    count = len(rows)
    return PipelineMetrics(
        queries=count,
        success_at_k=sum(row[0][0] for row in rows) / count,
        recall_at_k=sum(row[0][1] for row in rows) / count,
        mrr_at_k=sum(row[0][2] for row in rows) / count,
        ndcg_at_k=sum(row[0][3] for row in rows) / count,
        duplicate_rate=sum(row[1] for row in rows) / count,
        mean_latency_ms=sum(row[2] for row in rows) / count,
    )


def evaluate_pipeline(
    queries: Sequence[EvaluationQuery],
    retriever: WhatsAppRetriever,
    *,
    user_id: str,
    persona_id: str,
    collection_name: str,
    k: int = 8,
    candidate_count: int = 40,
    overlap_threshold: float = 0.7,
    reranker: LocalCrossEncoderReranker | None = None,
    chat_id: str | None = None,
) -> EvaluationReport:
    """Compare baseline, deduplication, and reranking at the same final K."""

    if k <= 0 or candidate_count < k:
        raise ValueError("candidate_count must be at least k and both must be positive")
    rows: dict[str, list[tuple[tuple[float, float, float, float], float, float]]] = {
        "hybrid_rrf": [], "hybrid_rrf_dedup": [], "hybrid_rrf_dedup_rerank": []
    }
    regressions: list[Regression] = []
    for query_index, query in enumerate(queries, 1):
        start = time.perf_counter()
        raw = retriever.search(
            query.query, user_id=user_id, persona_id=persona_id,
            collection_name=collection_name, mode="hybrid_rrf", limit=candidate_count,
            prefetch_limit=candidate_count, chat_id=chat_id,
        )
        retrieval_ms = (time.perf_counter() - start) * 1000
        baseline = raw[:k]
        dedup = deduplicate_hits(raw, overlap_threshold=overlap_threshold)[:k]
        rows["hybrid_rrf"].append((score_hits(baseline, query.relevant_ids, k=k), duplicate_rate(baseline), retrieval_ms))
        rows["hybrid_rrf_dedup"].append((score_hits(dedup, query.relevant_ids, k=k), duplicate_rate(dedup), retrieval_ms))
        if reranker is not None:
            rerank_start = time.perf_counter()
            reranked = reranker.rerank(
                query.query, deduplicate_hits(raw, overlap_threshold=overlap_threshold), final_count=k
            )
            total_ms = retrieval_ms + (time.perf_counter() - rerank_start) * 1000
            rows["hybrid_rrf_dedup_rerank"].append((score_hits(reranked, query.relevant_ids, k=k), duplicate_rate(reranked), total_ms))
            baseline_ranks = relevant_ranks(baseline, query.relevant_ids, k=k)
            reranked_ranks = relevant_ranks(reranked, query.relevant_ids, k=k)
            if baseline_ranks and (
                not reranked_ranks or min(reranked_ranks) > min(baseline_ranks)
            ):
                regressions.append(Regression(query_index, baseline_ranks, reranked_ranks))
    output = {name: aggregate_metrics(value) for name, value in rows.items() if value}
    return EvaluationReport(output, tuple(regressions))
