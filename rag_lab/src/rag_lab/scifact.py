from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .chunking import SourceDocument


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    text: str
    relevant_documents: dict[str, int]


def _first(row: dict, *names: str):
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"None of these fields were present: {', '.join(names)}")


def load_corpus(cache_dir: Path) -> list[SourceDocument]:
    from datasets import load_dataset

    dataset = load_dataset(
        "BeIR/scifact",
        "corpus",
        split="corpus",
        cache_dir=str(cache_dir),
    )
    return [
        SourceDocument(
            document_id=str(row["_id"]),
            title=row.get("title") or "",
            text=row.get("text") or "",
        )
        for row in dataset
    ]


def load_evaluation_queries(cache_dir: Path) -> list[EvaluationQuery]:
    from datasets import load_dataset

    queries = load_dataset(
        "BeIR/scifact",
        "queries",
        split="queries",
        cache_dir=str(cache_dir),
    )
    qrels = load_dataset(
        "BeIR/scifact-qrels",
        split="test",
        cache_dir=str(cache_dir),
    )

    relevance: dict[str, dict[str, int]] = defaultdict(dict)
    for row in qrels:
        query_id = str(_first(row, "query-id", "query_id", "query-id"))
        document_id = str(_first(row, "corpus-id", "corpus_id", "doc_id"))
        score = int(_first(row, "score", "_score"))
        if score > 0:
            relevance[query_id][document_id] = score

    results = []
    for row in queries:
        query_id = str(row["_id"])
        if query_id not in relevance:
            continue
        results.append(
            EvaluationQuery(
                query_id=query_id,
                text=row.get("text") or row.get("title") or "",
                relevant_documents=relevance[query_id],
            )
        )
    return results


def limit_documents(
    documents: Iterable[SourceDocument], limit: int | None
) -> list[SourceDocument]:
    if limit is None:
        return list(documents)
    result = []
    for document in documents:
        result.append(document)
        if len(result) >= limit:
            break
    return result

