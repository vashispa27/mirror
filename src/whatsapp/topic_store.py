"""Qdrant topic summaries and raw-message spans in a separate v3 collection."""

from __future__ import annotations

from typing import Any, Sequence

from .memory import MemoryChunk
from .store import DEFAULT_QDRANT_URL, SPARSE_MODEL
from .topic_memory import TOPIC_COLLECTION, TOPIC_VERSION, TopicRecord


DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class TopicQdrantStore:
    def __init__(self, url: str = DEFAULT_QDRANT_URL, *, client: Any = None) -> None:
        if client is None:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=url, timeout=120)
        self.client = client

    def ensure_collection(self) -> None:
        from qdrant_client import models

        size = self.client.get_embedding_size(DENSE_MODEL)
        if not self.client.collection_exists(TOPIC_COLLECTION):
            self.client.create_collection(
                collection_name=TOPIC_COLLECTION,
                vectors_config={"dense": models.VectorParams(size=size, distance=models.Distance.COSINE)},
                sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
        else:
            params = self.client.get_collection(TOPIC_COLLECTION).config.params
            dense = params.vectors.get("dense") if isinstance(params.vectors, dict) else None
            if dense is None or dense.size != size or not params.sparse_vectors or "sparse" not in params.sparse_vectors:
                raise ValueError("existing topic memory collection has incompatible vectors")
        for field in ("user_id", "contact_id", "chat_id", "schema_version", "record_type", "topic_id"):
            self.client.create_payload_index(TOPIC_COLLECTION, field_name=field,
                                             field_schema=models.PayloadSchemaType.KEYWORD)

    def index(self, topics: Sequence[TopicRecord], spans: Sequence[MemoryChunk]) -> int:
        from qdrant_client import models

        self.ensure_collection()
        points = []
        topic_by_id = {str(topic.id): topic for topic in topics}
        for topic in topics:
            text = f"{topic.label}. {topic.summary}"
            points.append(models.PointStruct(id=str(topic.id), payload=topic.payload(), vector={
                "dense": models.Document(text=text, model=DENSE_MODEL),
                "sparse": models.Document(text=text, model=SPARSE_MODEL),
            }))
        for span in spans:
            if span.schema_version != TOPIC_VERSION or span.topic_id is None:
                raise ValueError("topic span must be linked to a v3 topic")
            topic = topic_by_id[str(span.topic_id)]
            payload = {**span.payload(), "topic_label": topic.label, "topic_summary": topic.summary}
            points.append(models.PointStruct(id=str(span.id), payload=payload, vector={
                "dense": models.Document(text=span.text, model=DENSE_MODEL),
                "sparse": models.Document(text=span.text, model=SPARSE_MODEL),
            }))
        if points:
            self.client.upload_points(collection_name=TOPIC_COLLECTION, points=points, batch_size=64, wait=True)
        return len(points)

    def delete(self, ids: set[Any]) -> None:
        if ids:
            from qdrant_client import models
            self.client.delete(collection_name=TOPIC_COLLECTION,
                               points_selector=models.PointIdsList(points=[str(i) for i in ids]), wait=True)


class TopicMemoryRetriever:
    """Search summaries and spans; expand winning topics to spans."""

    def __init__(self, url: str = DEFAULT_QDRANT_URL, *, client: Any = None) -> None:
        self.client = client or TopicQdrantStore(url).client

    def _query(self, query: str, scope: Any, limit: int) -> list[dict[str, Any]]:
        from qdrant_client import models

        result = self.client.query_points(
            collection_name=TOPIC_COLLECTION, query_filter=scope, limit=limit,
            prefetch=[
                models.Prefetch(query=models.Document(text=query, model=DENSE_MODEL),
                                using="dense", filter=scope, limit=max(20, limit * 3)),
                models.Prefetch(query=models.Document(text=query, model=SPARSE_MODEL),
                                using="sparse", filter=scope, limit=max(20, limit * 3)),
            ], query=models.FusionQuery(fusion=models.Fusion.RRF),
            with_payload=True, with_vectors=False,
        )
        return [dict(p.payload or {}, point_id=str(p.id), score=float(p.score)) for p in result.points]

    def search(self, query: str, *, user_id: str, contact_id: str,
               limit: int = 8, reranker: Any = None) -> list[dict[str, Any]]:
        from qdrant_client import models

        if not query.strip() or not user_id or not contact_id or limit < 1:
            raise ValueError("query, owner, contact and positive limit are required")
        if not self.client.collection_exists(TOPIC_COLLECTION):
            return []
        base = [models.FieldCondition(key=k, match=models.MatchValue(value=v)) for k, v in
                (("user_id", user_id), ("contact_id", contact_id), ("schema_version", TOPIC_VERSION))]
        def scoped(kind: str, topic_id: str | None = None) -> Any:
            conditions = [*base, models.FieldCondition(key="record_type", match=models.MatchValue(value=kind))]
            if topic_id:
                conditions.append(models.FieldCondition(key="topic_id", match=models.MatchValue(value=topic_id)))
            return models.Filter(must=conditions)

        # FastEmbed mutates one client model accumulator during a query, so
        # using that same client in concurrent local requests is unsafe.
        # RRF scores are rank-based and often tie; Qdrant may return tied
        # points in a different order on separate requests, so apply a fixed
        # secondary key to keep the same query deterministic.
        def stable(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return sorted(items, key=lambda item: (-item.get("score", 0.0), item["point_id"]))

        topics = stable(self._query(query, scoped("topic"), min(4, limit)))
        direct = stable(self._query(query, scoped("span"), limit))
        expanded: list[dict[str, Any]] = []
        for topic in topics[:3]:
            for span in stable(self._query(query, scoped("span", topic["topic_id"]), 2)):
                span["topic_summary"] = topic["summary"]
                span["topic_label"] = topic["label"]
                expanded.append(span)
        # Spans are the primary retrieval unit; topic matches only attach
        # labels/summaries and add child spans when the direct pool leaves room.
        ordered = [*direct, *expanded]
        found = {p["point_id"]: p for p in ordered}
        candidates = list(found.values())
        if reranker is not None and candidates:
            scores = reranker.score(query, [hit.get("text", "") for hit in candidates])
            candidates = [hit for hit, _ in sorted(
                zip(candidates, scores), key=lambda pair: (-pair[1], pair[0]["point_id"]))]
        return candidates[:limit]
