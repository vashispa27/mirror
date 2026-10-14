"""Read-only retrieval evaluation for the four WhatsApp search modes.

This module answers one question: for a small set of human-labelled questions,
which retrieval mode (dense, sparse, hybrid RRF, hybrid DBSF) puts the right
*target-authored messages* highest?  It is deliberately evaluation-only:

* it never indexes, upserts, deletes, or recreates a Qdrant collection;
* it reuses the production :class:`~whatsapp.retrieval.WhatsAppRetriever` rather
  than duplicating search logic;
* it imports no runtime code from ``rag_lab``;
* it does not deduplicate, rerank, build prompts, or call an LLM.

Relevance is defined at the *result* level: a retrieved result is relevant when
the message ids it carries intersect the example's
``expected_target_message_ids``.  Which ids may count is set by
``evidence_scope``: ``any_author`` (the default) matches every message in the
chunk, so a fact stated by either speaker counts; ``target_only`` matches only
the target author's messages.  Because conversation chunks overlap, the same
expected message can appear in several results; recall counts each expected
message at most once.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .retrieval import SEARCH_MODES, WhatsAppRetriever, WhatsAppSearchHit


DEFAULT_K = 5
DEFAULT_PREFETCH_LIMIT = 30
DEFAULT_OVERLAP_THRESHOLD = 0.7

#: Which messages in a result count as evidence.
#: ``any_author`` matches every message in the chunk (needed for facts stated by
#: either speaker); ``target_only`` matches only the target author's messages.
EVIDENCE_SCOPES: tuple[str, ...] = ("any_author", "target_only")
DEFAULT_EVIDENCE_SCOPE = "any_author"

#: Dataset keys accepted (in priority order) for the expected message ids.
_EXPECTED_ID_KEYS = ("expected_target_message_ids", "relevant_message_ids")


class DatasetError(ValueError):
    """A labelled evaluation dataset could not be loaded or validated."""


#: Markers used by the committed template.  A run against the template would
#: silently return zero results, so it is rejected with guidance instead.
_PLACEHOLDER_MARKERS = ("REPLACE_WITH", "PASTE_")


def _is_placeholder(value: str) -> bool:
    upper = value.upper()
    return any(marker in upper for marker in _PLACEHOLDER_MARKERS)


def _reject_placeholder(value: str, *, field: str, line_number: int) -> None:
    if _is_placeholder(value):
        raise DatasetError(
            f"Dataset line {line_number} field {field!r} still contains the template "
            f"placeholder {value!r}. Copy evals/whatsapp.example.jsonl, then replace "
            "every placeholder with your own scope and target message IDs."
        )


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvalExample:
    """One labelled question scoped to a single user and persona."""

    line_number: int
    query: str
    user_id: str
    persona_id: str
    expected_target_message_ids: tuple[str, ...]
    query_type: str | None = None

    @property
    def expected(self) -> frozenset[str]:
        return frozenset(self.expected_target_message_ids)


def _require_text(row: Mapping[str, Any], key: str, line_number: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(
            f"Dataset line {line_number} is missing a non-empty string field {key!r}"
        )
    return value


def _expected_ids(row: Mapping[str, Any], line_number: int) -> tuple[str, ...]:
    selected_key: str | None = None
    for key in _EXPECTED_ID_KEYS:
        if key in row:
            selected_key = key
            break
    if selected_key is None:
        raise DatasetError(
            f"Dataset line {line_number} is missing {_EXPECTED_ID_KEYS[0]!r}"
        )
    raw = row[selected_key]
    if not isinstance(raw, (list, tuple)):
        raise DatasetError(
            f"Dataset line {line_number} field {selected_key!r} must be a list of ids"
        )
    ids: list[str] = []
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            raise DatasetError(
                f"Dataset line {line_number} field {selected_key!r} must contain "
                "non-empty string ids"
            )
        _reject_placeholder(value, field=selected_key, line_number=line_number)
        if value not in ids:
            ids.append(value)
    if not ids:
        raise DatasetError(
            f"Dataset line {line_number} field {selected_key!r} must not be empty"
        )
    return tuple(ids)


def load_dataset(path: str | Path) -> list[EvalExample]:
    """Load and validate a labelled JSONL dataset without rewriting it.

    Required fields per line: ``query``, ``user_id``, ``persona_id`` and a
    non-empty ``expected_target_message_ids`` list.  ``query_type`` is optional.
    """

    dataset_path = Path(path)
    if not dataset_path.exists():
        raise DatasetError(f"Dataset file does not exist: {dataset_path}")
    if not dataset_path.is_file():
        raise DatasetError(f"Dataset path is not a file: {dataset_path}")

    examples: list[EvalExample] = []
    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"Invalid JSON on dataset line {line_number}: {exc}") from exc
        if not isinstance(row, Mapping):
            raise DatasetError(
                f"Dataset line {line_number} must be a JSON object, not {type(row).__name__}"
            )
        query = _require_text(row, "query", line_number)
        user_id = _require_text(row, "user_id", line_number)
        persona_id = _require_text(row, "persona_id", line_number)
        _reject_placeholder(query, field="query", line_number=line_number)
        _reject_placeholder(user_id, field="user_id", line_number=line_number)
        _reject_placeholder(persona_id, field="persona_id", line_number=line_number)
        expected = _expected_ids(row, line_number)
        query_type = row.get("query_type")
        if query_type is not None and (
            not isinstance(query_type, str) or not query_type.strip()
        ):
            raise DatasetError(
                f"Dataset line {line_number} field 'query_type' must be a non-empty "
                "string when present"
            )
        examples.append(
            EvalExample(
                line_number=line_number,
                query=query,
                user_id=user_id,
                persona_id=persona_id,
                expected_target_message_ids=expected,
                query_type=query_type.strip() if isinstance(query_type, str) else None,
            )
        )

    if not examples:
        raise DatasetError(f"Dataset {dataset_path} contains no labelled examples.")
    return examples


def resolve_scope(
    examples: Sequence[EvalExample],
    *,
    user_id: str | None = None,
    persona_id: str | None = None,
) -> tuple[str, str]:
    """Return the single (user_id, persona_id) scope for a whole dataset.

    Passing only one of ``user_id``/``persona_id`` is an error.  When both are
    supplied they must match every row (a mismatch is reported, never ignored).
    Otherwise every row must already agree on the same scope.
    """

    if (user_id is None) != (persona_id is None):
        raise DatasetError("--user-id and --persona-id must be supplied together")

    if user_id is not None and persona_id is not None:
        desired = (user_id, persona_id)
        mismatched = [
            example.line_number
            for example in examples
            if (example.user_id, example.persona_id) != desired
        ]
        if mismatched:
            raise DatasetError(
                "Dataset scope does not match --user-id/--persona-id for lines "
                f"{mismatched[:10]}"
            )
        return desired

    scopes = {(example.user_id, example.persona_id) for example in examples}
    if len(scopes) != 1:
        ordered = sorted(scopes)
        raise DatasetError(
            "Dataset mixes multiple user_id/persona_id scopes "
            f"({ordered}); split it or pass --user-id/--persona-id"
        )
    return next(iter(scopes))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def target_overlap(left: Iterable[str], right: Iterable[str]) -> float:
    """Overlap as the fraction of the *smaller* id set that is shared.

    This containment-oriented measure flags a small chunk that is almost fully
    repeated by a larger one.  An empty side yields ``0.0``.
    """

    left_ids, right_ids = set(left), set(right)
    if not left_ids or not right_ids:
        return 0.0
    return len(left_ids & right_ids) / min(len(left_ids), len(right_ids))


def duplicate_rate(
    hits: Sequence[WhatsAppSearchHit], *, overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD
) -> float:
    """Fraction of results that substantially overlap a higher-ranked result.

    A result is a duplicate when at least ``overlap_threshold`` of its stored
    ``message_ids`` (measured against the smaller of the two results) also
    occurs in an earlier, higher-ranked result.  Returns ``0.0`` for no results.
    """

    if not hits:
        return 0.0
    if not 0.0 < overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be greater than 0 and at most 1")
    duplicates = 0
    retained: list[set[str]] = []
    for hit in hits:
        ids = set(hit.message_ids)
        if any(target_overlap(ids, previous) >= overlap_threshold for previous in retained):
            duplicates += 1
        retained.append(ids)
    return duplicates / len(hits)


def hit_message_ids(hit: WhatsAppSearchHit, evidence_scope: str) -> tuple[str, ...]:
    """The message ids of a result that may count as evidence."""

    if evidence_scope == "any_author":
        return hit.message_ids
    if evidence_scope == "target_only":
        return hit.target_message_ids
    raise ValueError(
        f"evidence_scope must be one of: {', '.join(EVIDENCE_SCOPES)}"
    )


def matched_message_ids(
    hit: WhatsAppSearchHit,
    expected: frozenset[str],
    *,
    evidence_scope: str = DEFAULT_EVIDENCE_SCOPE,
) -> tuple[str, ...]:
    """Expected message ids carried by one result, in payload order."""

    return tuple(
        message_id
        for message_id in hit_message_ids(hit, evidence_scope)
        if message_id in expected
    )


def ndcg_at_k(relevances: Sequence[int], *, k: int, ideal_count: int) -> float:
    """Binary-relevance nDCG@K at the result level.

    ``ideal_count`` is the number of relevant results treated as attainable in
    the ideal ranking (here ``min(k, number of expected messages)``).  The
    discount is the standard ``1 / log2(rank + 1)``.
    """

    if k <= 0:
        raise ValueError("k must be a positive integer")
    dcg = sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances[:k], start=1)
    )
    ideal = min(k, max(ideal_count, 0))
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, ideal + 1))
    return dcg / idcg if idcg else 0.0


@dataclass(frozen=True)
class QueryMetrics:
    """Metrics for one example under one retrieval mode."""

    success_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    duplicate_rate: float
    relevant_ranks: tuple[int, ...]
    matched_message_ids: tuple[str, ...]


def score_query(
    hits: Sequence[WhatsAppSearchHit],
    expected: frozenset[str],
    *,
    k: int,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    evidence_scope: str = DEFAULT_EVIDENCE_SCOPE,
) -> QueryMetrics:
    """Score the top ``k`` results of one query with binary result relevance."""

    if k <= 0:
        raise ValueError("k must be a positive integer")
    selected = list(hits[:k])
    matched: list[str] = []
    relevances: list[int] = []
    ranks: list[int] = []
    for rank, hit in enumerate(selected, start=1):
        hit_matches = matched_message_ids(hit, expected, evidence_scope=evidence_scope)
        relevances.append(1 if hit_matches else 0)
        if hit_matches:
            ranks.append(rank)
            for message_id in hit_matches:
                if message_id not in matched:
                    matched.append(message_id)
    recall = len(matched) / len(expected) if expected else 0.0
    mrr = 1 / ranks[0] if ranks else 0.0
    return QueryMetrics(
        success_at_k=1.0 if ranks else 0.0,
        recall_at_k=recall,
        mrr_at_k=mrr,
        ndcg_at_k=ndcg_at_k(relevances, k=k, ideal_count=len(expected)),
        duplicate_rate=duplicate_rate(selected, overlap_threshold=overlap_threshold),
        relevant_ranks=tuple(ranks),
        matched_message_ids=tuple(matched),
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile (``fraction`` in ``(0, 1]``); 0.0 when empty."""

    if not values:
        return 0.0
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be greater than 0 and at most 1")
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


# --------------------------------------------------------------------------- #
# Report model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetrievedResult:
    """One top-K result, reduced to identifiers (never chat text)."""

    rank: int
    point_id: str
    chunk_id: str
    score: float
    matched_message_ids: tuple[str, ...]
    target_message_ids: tuple[str, ...]


@dataclass(frozen=True)
class QueryModeResult:
    """Metrics plus top-K provenance for one example under one mode."""

    metrics: QueryMetrics
    latency_ms: float
    results: tuple[RetrievedResult, ...]


@dataclass(frozen=True)
class QueryReport:
    """Everything the harness knows about one labelled example."""

    index: int
    query: str
    query_type: str | None
    expected_count: int
    modes: Mapping[str, QueryModeResult]


@dataclass(frozen=True)
class ModeAggregate:
    """Aggregate metrics for one retrieval mode across the dataset."""

    mode: str
    queries: int
    success_at_k: float
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    duplicate_rate: float
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float


@dataclass(frozen=True)
class EvaluationReport:
    """A complete, renderable evaluation run."""

    config: Mapping[str, Any]
    aggregate: Mapping[str, ModeAggregate]
    by_query_type: Mapping[str, Mapping[str, ModeAggregate]]
    per_query: tuple[QueryReport, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def _aggregate(mode: str, rows: Sequence[tuple[QueryMetrics, float]]) -> ModeAggregate:
    count = len(rows)
    if count == 0:
        return ModeAggregate(mode, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    latencies = [latency for _, latency in rows]
    return ModeAggregate(
        mode=mode,
        queries=count,
        success_at_k=sum(row[0].success_at_k for row in rows) / count,
        recall_at_k=sum(row[0].recall_at_k for row in rows) / count,
        mrr_at_k=sum(row[0].mrr_at_k for row in rows) / count,
        ndcg_at_k=sum(row[0].ndcg_at_k for row in rows) / count,
        duplicate_rate=sum(row[0].duplicate_rate for row in rows) / count,
        latency_mean_ms=sum(latencies) / count,
        latency_median_ms=statistics.median(latencies),
        latency_p95_ms=percentile(latencies, 0.95),
    )


def _validate_options(
    k: int,
    prefetch_limit: int,
    modes: Sequence[str],
    overlap_threshold: float,
    evidence_scope: str,
) -> tuple[str, ...]:
    if evidence_scope not in EVIDENCE_SCOPES:
        raise ValueError(
            f"evidence_scope must be one of: {', '.join(EVIDENCE_SCOPES)}"
        )
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if (
        isinstance(prefetch_limit, bool)
        or not isinstance(prefetch_limit, int)
        or prefetch_limit <= 0
    ):
        raise ValueError("prefetch_limit must be a positive integer")
    if prefetch_limit < k:
        raise ValueError("prefetch_limit must be greater than or equal to k")
    if not 0.0 < overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be greater than 0 and at most 1")
    selected = tuple(modes)
    if not selected:
        raise ValueError("at least one retrieval mode is required")
    unknown = [mode for mode in selected if mode not in SEARCH_MODES]
    if unknown:
        raise ValueError(
            f"unknown retrieval mode(s): {', '.join(unknown)}; "
            f"choose from {', '.join(SEARCH_MODES)}"
        )
    return selected


def evaluate_modes(
    examples: Sequence[EvalExample],
    retriever: WhatsAppRetriever,
    *,
    collection_name: str,
    user_id: str,
    persona_id: str,
    k: int = DEFAULT_K,
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
    modes: Sequence[str] = SEARCH_MODES,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    evidence_scope: str = DEFAULT_EVIDENCE_SCOPE,
    chat_id: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
    dataset_path: str | None = None,
) -> EvaluationReport:
    """Search every example with every mode and return aggregate + per-query data.

    ``retriever.search`` is the only Qdrant interaction and is strictly
    read-only.  Latency is the wall-clock time of that full call, which includes
    the read-only collection verification and one Qdrant round-trip.
    """

    selected_modes = _validate_options(
        k, prefetch_limit, modes, overlap_threshold, evidence_scope
    )
    if not examples:
        raise ValueError("at least one labelled example is required")

    per_query: list[QueryReport] = []
    aggregate_rows: dict[str, list[tuple[QueryMetrics, float]]] = {
        mode: [] for mode in selected_modes
    }
    typed_rows: dict[str, dict[str, list[tuple[QueryMetrics, float]]]] = {}

    for index, example in enumerate(examples, start=1):
        mode_results: dict[str, QueryModeResult] = {}
        for mode in selected_modes:
            start = clock()
            hits = retriever.search(
                example.query,
                user_id=user_id,
                persona_id=persona_id,
                collection_name=collection_name,
                mode=mode,  # type: ignore[arg-type]
                limit=k,
                prefetch_limit=prefetch_limit,
                chat_id=chat_id,
            )
            latency_ms = (clock() - start) * 1000.0
            metrics = score_query(
                hits,
                example.expected,
                k=k,
                overlap_threshold=overlap_threshold,
                evidence_scope=evidence_scope,
            )
            results = tuple(
                RetrievedResult(
                    rank=hit.rank,
                    point_id=hit.point_id,
                    chunk_id=hit.chunk_id,
                    score=hit.score,
                    matched_message_ids=matched_message_ids(
                        hit, example.expected, evidence_scope=evidence_scope
                    ),
                    target_message_ids=hit.target_message_ids,
                )
                for hit in hits[:k]
            )
            mode_results[mode] = QueryModeResult(metrics, latency_ms, results)
            aggregate_rows[mode].append((metrics, latency_ms))
            if example.query_type:
                typed_rows.setdefault(example.query_type, {}).setdefault(mode, []).append(
                    (metrics, latency_ms)
                )
        per_query.append(
            QueryReport(
                index=index,
                query=example.query,
                query_type=example.query_type,
                expected_count=len(example.expected_target_message_ids),
                modes=mode_results,
            )
        )

    aggregate = {
        mode: _aggregate(mode, rows) for mode, rows in aggregate_rows.items()
    }
    by_query_type = {
        query_type: {
            mode: _aggregate(mode, rows) for mode, rows in mode_rows.items()
        }
        for query_type, mode_rows in typed_rows.items()
    }
    config = {
        "dataset": dataset_path,
        "collection": collection_name,
        "user_id": user_id,
        "persona_id": persona_id,
        "chat_id": chat_id,
        "k": k,
        "prefetch_limit": prefetch_limit,
        "overlap_threshold": overlap_threshold,
        "evidence_scope": evidence_scope,
        "modes": list(selected_modes),
        "query_count": len(examples),
    }

    warnings: list[str] = []
    retrieved_total = sum(
        len(result.results)
        for query_report in per_query
        for result in query_report.modes.values()
    )
    if retrieved_total == 0:
        warnings.append(
            "Every search returned zero results. The scope filter matched nothing "
            f"(user_id={user_id!r}, persona_id={persona_id!r}, collection="
            f"{collection_name!r}). Check the scope and that this persona is indexed; "
            "all-zero metrics here mean 'no hits', not 'a fair retrieval comparison'."
        )
    if all(
        aggregate.success_at_k == 0.0 for aggregate in aggregate.values()
    ):
        warnings.append(
            "No mode retrieved any expected message in the top K. Either the "
            "expected IDs do not exist in this collection or the queries are out of "
            "scope; inspect the failure detail before comparing modes."
        )

    return EvaluationReport(
        config=config,
        aggregate=aggregate,
        by_query_type=by_query_type,
        per_query=tuple(per_query),
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def mode_aggregate_to_dict(aggregate: ModeAggregate) -> dict[str, Any]:
    return {
        "mode": aggregate.mode,
        "queries": aggregate.queries,
        "success_at_k": _round(aggregate.success_at_k),
        "recall_at_k": _round(aggregate.recall_at_k),
        "mrr_at_k": _round(aggregate.mrr_at_k),
        "ndcg_at_k": _round(aggregate.ndcg_at_k),
        "duplicate_rate": _round(aggregate.duplicate_rate),
        "latency_ms": {
            "mean": _round(aggregate.latency_mean_ms, 2),
            "median": _round(aggregate.latency_median_ms, 2),
            "p95": _round(aggregate.latency_p95_ms, 2),
        },
    }


def query_report_to_dict(report: QueryReport) -> dict[str, Any]:
    return {
        "index": report.index,
        "query": report.query,
        "query_type": report.query_type,
        "expected_count": report.expected_count,
        "modes": {
            mode: {
                "success_at_k": _round(result.metrics.success_at_k),
                "recall_at_k": _round(result.metrics.recall_at_k),
                "mrr_at_k": _round(result.metrics.mrr_at_k),
                "ndcg_at_k": _round(result.metrics.ndcg_at_k),
                "duplicate_rate": _round(result.metrics.duplicate_rate),
                "relevant_ranks": list(result.metrics.relevant_ranks),
                "matched_message_ids": list(result.metrics.matched_message_ids),
                "latency_ms": _round(result.latency_ms, 2),
                "retrieved": [
                    {
                        "rank": hit.rank,
                        "point_id": hit.point_id,
                        "chunk_id": hit.chunk_id,
                        "score": _round(hit.score, 6),
                        "matched_message_ids": list(hit.matched_message_ids),
                        "target_message_ids": list(hit.target_message_ids),
                    }
                    for hit in result.results
                ],
            }
            for mode, result in report.modes.items()
        },
    }


def report_to_dict(report: EvaluationReport) -> dict[str, Any]:
    """Machine-readable report: configuration, aggregates, per-query detail."""

    return {
        "warnings": list(report.warnings),
        "config": dict(report.config),
        "aggregate": {
            mode: mode_aggregate_to_dict(value)
            for mode, value in report.aggregate.items()
        },
        "by_query_type": {
            query_type: {
                mode: mode_aggregate_to_dict(value) for mode, value in mode_rows.items()
            }
            for query_type, mode_rows in report.by_query_type.items()
        },
        "per_query": [query_report_to_dict(report_item) for report_item in report.per_query],
    }


_AGGREGATE_HEADER = (
    f"{'mode':<12}{'success@k':>11}{'recall@k':>10}{'mrr@k':>9}"
    f"{'ndcg@k':>9}{'dup_rate':>10}{'lat_med_ms':>12}{'lat_p95_ms':>12}"
)


def _aggregate_line(aggregate: ModeAggregate) -> str:
    return (
        f"{aggregate.mode:<12}{aggregate.success_at_k:>11.4f}"
        f"{aggregate.recall_at_k:>10.4f}{aggregate.mrr_at_k:>9.4f}"
        f"{aggregate.ndcg_at_k:>9.4f}{aggregate.duplicate_rate:>10.4f}"
        f"{aggregate.latency_median_ms:>12.2f}{aggregate.latency_p95_ms:>12.2f}"
    )


def render_table(report: EvaluationReport) -> str:
    """Human-readable report: aggregates, per-query rows, and failure detail."""

    config = report.config
    lines: list[str] = ["Evaluation configuration"]
    for key in (
        "dataset",
        "collection",
        "user_id",
        "persona_id",
        "chat_id",
        "k",
        "prefetch_limit",
        "overlap_threshold",
        "evidence_scope",
        "query_count",
    ):
        lines.append(f"  {key}: {config.get(key)}")

    for warning in report.warnings:
        lines.append("")
        lines.append(f"WARNING: {warning}")

    lines.append("")
    lines.append(f"Aggregate by mode (K={config.get('k')})")
    lines.append(_AGGREGATE_HEADER)
    for aggregate in report.aggregate.values():
        lines.append(_aggregate_line(aggregate))

    if report.by_query_type:
        lines.append("")
        lines.append("Aggregate by query_type")
        for query_type, mode_rows in report.by_query_type.items():
            lines.append(f"  [{query_type}]")
            lines.append("  " + _AGGREGATE_HEADER)
            for aggregate in mode_rows.values():
                lines.append("  " + _aggregate_line(aggregate))

    lines.append("")
    lines.append("Per-query results")
    failures: list[QueryReport] = []
    for query_report in report.per_query:
        lines.append(
            f"#{query_report.index} [{query_report.query_type or 'untyped'}] "
            f"{query_report.query} (expected={query_report.expected_count})"
        )
        for mode, result in query_report.modes.items():
            metrics = result.metrics
            matched = ",".join(metrics.matched_message_ids) or "-"
            lines.append(
                f"    {mode:<12} success={metrics.success_at_k:.0f} "
                f"recall={metrics.recall_at_k:.2f} mrr={metrics.mrr_at_k:.2f} "
                f"ndcg={metrics.ndcg_at_k:.2f} dup={metrics.duplicate_rate:.2f} "
                f"lat={result.latency_ms:.1f}ms "
                f"ranks={list(metrics.relevant_ranks) or '[]'} matched=[{matched}]"
            )
            if metrics.success_at_k == 0.0:
                failures.append(query_report)

    if failures:
        lines.append("")
        lines.append("Failure detail (top-K retrieved identifiers for modes that missed)")
        seen: set[int] = set()
        for query_report in failures:
            if query_report.index in seen:
                continue
            seen.add(query_report.index)
            lines.append(
                f"#{query_report.index} {query_report.query} "
                f"(expected={query_report.expected_count})"
            )
            for mode, result in query_report.modes.items():
                if result.metrics.success_at_k != 0.0:
                    continue
                lines.append(f"  [{mode}]")
                if not result.results:
                    lines.append("    (no results returned)")
                for hit in result.results:
                    matched = ",".join(hit.matched_message_ids) or "-"
                    lines.append(
                        f"    rank={hit.rank} score={hit.score:.6f} "
                        f"chunk={hit.chunk_id} point={hit.point_id} "
                        f"matched=[{matched}]"
                    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def run_evaluation(
    *,
    dataset_path: str,
    retriever: WhatsAppRetriever,
    collection_name: str,
    k: int = DEFAULT_K,
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
    modes: Sequence[str] = SEARCH_MODES,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    evidence_scope: str = DEFAULT_EVIDENCE_SCOPE,
    user_id: str | None = None,
    persona_id: str | None = None,
    chat_id: str | None = None,
) -> EvaluationReport:
    """Load a dataset, resolve scope, and evaluate every mode."""

    examples = load_dataset(dataset_path)
    scope_user, scope_persona = resolve_scope(
        examples, user_id=user_id, persona_id=persona_id
    )
    return evaluate_modes(
        examples,
        retriever,
        collection_name=collection_name,
        user_id=scope_user,
        persona_id=scope_persona,
        k=k,
        prefetch_limit=prefetch_limit,
        modes=modes,
        overlap_threshold=overlap_threshold,
        evidence_scope=evidence_scope,
        chat_id=chat_id,
        dataset_path=str(dataset_path),
    )
