from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from rich.console import Console
from rich.table import Table

from .chunking import ChunkingConfig, SourceDocument, chunk_document
from .metrics import aggregate_metrics, score_query
from .scifact import limit_documents, load_corpus, load_evaluation_queries
from .store import QdrantLabStore, rerank, unique_document_ids


console = Console()
DEFAULT_CACHE = Path(__file__).parents[2] / ".cache" / "datasets"


def _config(args) -> ChunkingConfig:
    return ChunkingConfig(
        method=args.strategy,
        max_characters=args.max_characters,
        overlap_characters=args.overlap_characters,
        overlap_sentences=args.overlap_sentences,
        semantic_break_percentile=args.semantic_break_percentile,
    )


def _semantic_embedder():
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def embed(sentences):
        return list(model.passage_embed(list(sentences)))

    return embed


def _chunks(
    documents: Iterable[SourceDocument], config: ChunkingConfig
):
    embedder = _semantic_embedder() if config.method == "semantic" else None
    for document in documents:
        yield from chunk_document(document, config, embed_sentences=embedder)


def inspect_command(args) -> None:
    documents = load_corpus(args.cache_dir)
    document = documents[args.document_index]
    chunks = list(_chunks([document], _config(args)))

    console.print(f"[bold]{document.title}[/bold]")
    console.print(
        f"document characters={len(document.text)} chunks={len(chunks)}\n"
    )
    for chunk in chunks:
        console.rule(f"chunk {chunk.chunk_index} · {len(chunk.text)} chars")
        console.print(chunk.text)


def index_command(args) -> None:
    documents = limit_documents(load_corpus(args.cache_dir), args.limit_documents)
    config = _config(args)
    store = QdrantLabStore(args.qdrant_url)
    store.recreate_collection(args.collection)
    count = store.upload(args.collection, _chunks(documents, config))
    console.print(
        f"Indexed [bold]{count}[/bold] {config.method} chunks from "
        f"{len(documents)} documents into [bold]{args.collection}[/bold]."
    )


def _search_hits(store, args, limit=None):
    candidate_limit = args.candidate_limit if args.rerank else (limit or args.limit)
    hits = store.search(
        args.collection,
        args.query,
        mode=args.mode,
        limit=candidate_limit,
        prefetch_limit=max(args.prefetch_limit, candidate_limit),
    )
    if args.rerank:
        hits = rerank(args.query, hits, limit=limit or args.limit)
    return hits


def search_command(args) -> None:
    store = QdrantLabStore(args.qdrant_url)
    hits = _search_hits(store, args)
    table = Table("rank", "score", "document", "title", "text")
    for index, hit in enumerate(hits, start=1):
        table.add_row(
            str(index),
            f"{hit.score:.4f}",
            hit.document_id,
            hit.title[:45],
            hit.text[:180].replace("\n", " "),
        )
    console.print(table)


def evaluate_command(args) -> None:
    store = QdrantLabStore(args.qdrant_url)
    queries = load_evaluation_queries(args.cache_dir)
    if args.max_queries:
        queries = queries[: args.max_queries]

    scores = []
    for index, evaluation_query in enumerate(queries, start=1):
        args.query = evaluation_query.text
        hits = _search_hits(store, args, limit=args.k * 5)
        ranked_documents = unique_document_ids(hits, args.k)
        scores.append(
            score_query(
                ranked_documents,
                evaluation_query.relevant_documents,
                k=args.k,
            )
        )
        if index % 25 == 0:
            console.print(f"evaluated {index}/{len(queries)} queries")

    metrics = aggregate_metrics(scores, k=args.k)
    table = Table("queries", f"success@{args.k}", f"recall@{args.k}", f"MRR@{args.k}", f"nDCG@{args.k}")
    table.add_row(
        str(metrics.queries),
        f"{metrics.success_at_k:.4f}",
        f"{metrics.recall_at_k:.4f}",
        f"{metrics.mrr_at_k:.4f}",
        f"{metrics.ndcg_at_k:.4f}",
    )
    console.print(table)


def _add_chunking_options(parser) -> None:
    parser.add_argument(
        "--strategy",
        choices=("whole", "fixed", "sentence", "recursive", "semantic"),
        default="sentence",
    )
    parser.add_argument("--max-characters", type=int, default=1_000)
    parser.add_argument("--overlap-characters", type=int, default=150)
    parser.add_argument("--overlap-sentences", type=int, default=1)
    parser.add_argument("--semantic-break-percentile", type=float, default=75)


def _add_search_options(parser, *, query_required: bool) -> None:
    if query_required:
        parser.add_argument("query")
    parser.add_argument("--collection", default="rag_lab_scifact_sentence")
    parser.add_argument(
        "--mode",
        choices=("dense", "sparse", "hybrid_rrf", "hybrid_dbsf"),
        default="dense",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--prefetch-limit", type=int, default=50)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--rerank", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hands-on Qdrant RAG lab")
    parser.set_defaults(
        cache_dir=DEFAULT_CACHE,
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Visually inspect how one document is chunked"
    )
    inspect_parser.add_argument("--document-index", type=int, default=0)
    _add_chunking_options(inspect_parser)
    inspect_parser.set_defaults(handler=inspect_command)

    index_parser = subparsers.add_parser(
        "index", help="Chunk SciFact and index it into Qdrant"
    )
    index_parser.add_argument("--collection", default="rag_lab_scifact_sentence")
    index_parser.add_argument("--limit-documents", type=int)
    _add_chunking_options(index_parser)
    index_parser.set_defaults(handler=index_command)

    search_parser = subparsers.add_parser(
        "search", help="Run one dense, sparse, or hybrid search"
    )
    _add_search_options(search_parser, query_required=True)
    search_parser.set_defaults(handler=search_command)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Measure a retriever against SciFact qrels"
    )
    _add_search_options(evaluate_parser, query_required=False)
    evaluate_parser.add_argument("--max-queries", type=int, default=50)
    evaluate_parser.add_argument("--k", type=int, default=10)
    evaluate_parser.set_defaults(handler=evaluate_command, query="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()

