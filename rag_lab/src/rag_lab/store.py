from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from .chunking import Chunk


SearchMode = Literal["dense", "sparse", "hybrid_rrf", "hybrid_dbsf"]

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float


class QdrantLabStore:
    def __init__(self, url: str = "http://localhost:6333") -> None:
        from qdrant_client import QdrantClient

        self.client = (
            QdrantClient(":memory:")
            if url == ":memory:"
            else QdrantClient(url=url, timeout=60)
        )

    def recreate_collection(self, collection_name: str) -> None:
        from qdrant_client import models

        if self.client.collection_exists(collection_name):
            self.client.delete_collection(collection_name)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=self.client.get_embedding_size(DENSE_MODEL),
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                )
            },
        )

    def upload(self, collection_name: str, chunks: Iterable[Chunk]) -> int:
        from qdrant_client import models

        count = 0

        def points():
            nonlocal count
            for count, chunk in enumerate(chunks, start=1):
                yield models.PointStruct(
                    id=chunk.chunk_id,
                    vector={
                        DENSE_VECTOR: models.Document(
                            text=chunk.text, model=DENSE_MODEL
                        ),
                        SPARSE_VECTOR: models.Document(
                            text=chunk.text, model=SPARSE_MODEL
                        ),
                    },
                    payload=chunk.payload(),
                )

        self.client.upload_points(
            collection_name=collection_name,
            points=points(),
            batch_size=64,
        )
        return count

    def search(
        self,
        collection_name: str,
        query: str,
        *,
        mode: SearchMode,
        limit: int = 10,
        prefetch_limit: int = 50,
    ) -> list[SearchHit]:
        from qdrant_client import models

        dense = models.Document(text=query, model=DENSE_MODEL)
        sparse = models.Document(text=query, model=SPARSE_MODEL)

        if mode == "dense":
            response = self.client.query_points(
                collection_name=collection_name,
                query=dense,
                using=DENSE_VECTOR,
                limit=limit,
                with_payload=True,
            )
        elif mode == "sparse":
            response = self.client.query_points(
                collection_name=collection_name,
                query=sparse,
                using=SPARSE_VECTOR,
                limit=limit,
                with_payload=True,
            )
        else:
            fusion = (
                models.Fusion.RRF
                if mode == "hybrid_rrf"
                else models.Fusion.DBSF
            )
            response = self.client.query_points(
                collection_name=collection_name,
                prefetch=[
                    models.Prefetch(
                        query=dense,
                        using=DENSE_VECTOR,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=sparse,
                        using=SPARSE_VECTOR,
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=fusion),
                limit=limit,
                with_payload=True,
            )

        return [self._hit(point) for point in response.points]

    @staticmethod
    def _hit(point) -> SearchHit:
        payload = point.payload or {}
        return SearchHit(
            chunk_id=str(payload.get("chunk_id", point.id)),
            document_id=str(payload.get("document_id", "")),
            title=str(payload.get("title", "")),
            text=str(payload.get("text", "")),
            score=float(point.score),
        )


def rerank(
    query: str,
    candidates: Sequence[SearchHit],
    *,
    limit: int,
    model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
) -> list[SearchHit]:
    """Post-retrieval cross-encoder reranking."""

    from fastembed.rerank.cross_encoder import TextCrossEncoder

    encoder = TextCrossEncoder(model_name=model_name)
    scores = list(encoder.rerank(query, [candidate.text for candidate in candidates]))
    rescored = [
        SearchHit(
            chunk_id=candidate.chunk_id,
            document_id=candidate.document_id,
            title=candidate.title,
            text=candidate.text,
            score=float(score),
        )
        for candidate, score in zip(candidates, scores)
    ]
    return sorted(rescored, key=lambda hit: hit.score, reverse=True)[:limit]


def unique_document_ids(hits: Sequence[SearchHit], limit: int) -> list[str]:
    seen = set()
    document_ids = []
    for hit in hits:
        if hit.document_id in seen:
            continue
        seen.add(hit.document_id)
        document_ids.append(hit.document_id)
        if len(document_ids) >= limit:
            break
    return document_ids
