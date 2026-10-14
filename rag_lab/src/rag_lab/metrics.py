from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RetrievalMetrics:
    queries: int
    success_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float


def _dcg(relevances: Sequence[int]) -> float:
    return sum(
        (2**relevance - 1) / math.log2(index + 2)
        for index, relevance in enumerate(relevances)
    )


def score_query(
    ranked_document_ids: Sequence[str],
    relevant_documents: Mapping[str, int],
    *,
    k: int,
) -> tuple[float, float, float, float]:
    retrieved = list(ranked_document_ids[:k])
    hits = [document_id for document_id in retrieved if document_id in relevant_documents]

    success = 1.0 if hits else 0.0
    recall = len(set(hits)) / len(relevant_documents) if relevant_documents else 0.0
    reciprocal_rank = next(
        (
            1 / (index + 1)
            for index, document_id in enumerate(retrieved)
            if document_id in relevant_documents
        ),
        0.0,
    )

    retrieved_relevance = [relevant_documents.get(document_id, 0) for document_id in retrieved]
    ideal_relevance = sorted(relevant_documents.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal_relevance)
    ndcg = _dcg(retrieved_relevance) / ideal_dcg if ideal_dcg else 0.0
    return success, recall, reciprocal_rank, ndcg


def aggregate_metrics(
    scores: Sequence[tuple[float, float, float, float]],
    *,
    k: int,
) -> RetrievalMetrics:
    if not scores:
        return RetrievalMetrics(0, 0.0, 0.0, 0.0, 0.0)

    count = len(scores)
    return RetrievalMetrics(
        queries=count,
        success_at_k=sum(score[0] for score in scores) / count,
        recall_at_k=sum(score[1] for score in scores) / count,
        mrr_at_k=sum(score[2] for score in scores) / count,
        ndcg_at_k=sum(score[3] for score in scores) / count,
    )

