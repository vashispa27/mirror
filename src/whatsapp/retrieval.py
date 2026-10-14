"""Scoped, direct Qdrant retrieval for indexed WhatsApp persona records.

This production module intentionally shares no runtime code with ``rag_lab``.
Qdrant performs all dense, sparse, and fusion ranking; this module only builds
the scoped query and turns Qdrant points into application-level search hits.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from .persona import SCHEMA_VERSION
from .store import (
    DEFAULT_COLLECTION,
    DEFAULT_QDRANT_URL,
    DENSE_MODEL,
    DENSE_VECTOR,
    SPARSE_MODEL,
    SPARSE_VECTOR,
)


SearchMode = Literal["dense", "sparse", "hybrid_rrf", "hybrid_dbsf"]
SEARCH_MODES: tuple[SearchMode, ...] = (
    "dense",
    "sparse",
    "hybrid_rrf",
    "hybrid_dbsf",
)


class RetrievalError(RuntimeError):
    """Base error for a search that could not be executed."""


class CollectionMissingError(RetrievalError):
    """The requested Qdrant collection does not exist."""


class CollectionIncompatibleError(RetrievalError):
    """The requested collection is not a WhatsApp dense/sparse collection."""


class QdrantUnavailableError(RetrievalError):
    """Qdrant cannot be reached or returned an operational error."""


@dataclass(frozen=True)
class WhatsAppSearchHit:
    """A ranked persona conversation chunk with its retrieval provenance."""

    point_id: str
    rank: int
    score: float
    schema_version: str
    user_id: str
    persona_id: str
    source: str
    target_author: str
    requested_target_author: str
    chat_id: str
    chunk_id: str
    started_at: str
    ended_at: str
    start_sequence: int | None
    end_sequence: int | None
    message_ids: tuple[str, ...]
    target_message_ids: tuple[str, ...]
    authors: tuple[str, ...]
    text: str
    context: tuple[dict[str, Any], ...]
    target_responses: tuple[dict[str, Any], ...]
    # ``score`` remains the original Qdrant score for backwards compatibility.
    # A cross-encoder score is deliberately separate: neither score is a
    # probability and the two should not be compared numerically.
    reranker_score: float | None = None

    @property
    def retrieval_score(self) -> float:
        """The unchanged first-stage Qdrant score."""

        return self.score

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-ready representation of the hit."""

        return {
            "point_id": self.point_id,
            "rank": self.rank,
            "score": self.score,
            "retrieval_score": self.retrieval_score,
            "reranker_score": self.reranker_score,
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "persona_id": self.persona_id,
            "source": self.source,
            "target_author": self.target_author,
            "requested_target_author": self.requested_target_author,
            "chat_id": self.chat_id,
            "chunk_id": self.chunk_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "start_sequence": self.start_sequence,
            "end_sequence": self.end_sequence,
            "message_ids": list(self.message_ids),
            "target_message_ids": list(self.target_message_ids),
            "authors": list(self.authors),
            "text": self.text,
            "context": list(self.context),
            "target_responses": list(self.target_responses),
        }


def _required_identifier(value: str | None, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty identifier")
    # Identity values are application-owned opaque strings.  Validate trimmed
    # emptiness without rewriting the exact value that Qdrant is filtered on.
    return value


def validate_search_arguments(
    query: str,
    user_id: str | None,
    persona_id: str | None,
    mode: str,
    limit: int,
    prefetch_limit: int,
) -> None:
    """Validate user input before any Qdrant request is made."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must not be empty")
    _required_identifier(user_id, "user_id")
    _required_identifier(persona_id, "persona_id")
    if mode not in SEARCH_MODES:
        raise ValueError(f"mode must be one of: {', '.join(SEARCH_MODES)}")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    if (
        isinstance(prefetch_limit, bool)
        or not isinstance(prefetch_limit, int)
        or prefetch_limit <= 0
    ):
        raise ValueError("prefetch_limit must be a positive integer")
    if prefetch_limit < limit:
        raise ValueError("prefetch_limit must be greater than or equal to limit")


def build_required_filter(user_id: str, persona_id: str, chat_id: str | None = None):
    """Build the non-optional Qdrant metadata filter for every search."""

    from qdrant_client import models

    _required_identifier(user_id, "user_id")
    _required_identifier(persona_id, "persona_id")
    must = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
        models.FieldCondition(
            key="persona_id", match=models.MatchValue(value=persona_id)
        ),
        models.FieldCondition(key="source", match=models.MatchValue(value="whatsapp")),
        models.FieldCondition(
            key="schema_version", match=models.MatchValue(value=SCHEMA_VERSION)
        ),
    ]
    if chat_id is not None:
        _required_identifier(chat_id, "chat_id")
        must.append(
            models.FieldCondition(key="chat_id", match=models.MatchValue(value=chat_id))
        )
    return models.Filter(must=must)


def _point_value(payload: Mapping[str, Any], key: str, default: Any = "") -> Any:
    value = payload.get(key, default)
    return default if value is None else value


def point_to_search_hit(point: Any, rank: int) -> WhatsAppSearchHit:
    """Convert a Qdrant scored point without returning vectors."""

    payload = point.payload or {}
    return WhatsAppSearchHit(
        point_id=str(point.id),
        rank=rank,
        score=float(point.score),
        schema_version=str(_point_value(payload, "schema_version")),
        user_id=str(_point_value(payload, "user_id")),
        persona_id=str(_point_value(payload, "persona_id")),
        source=str(_point_value(payload, "source")),
        target_author=str(_point_value(payload, "target_author")),
        requested_target_author=str(_point_value(payload, "requested_target_author")),
        chat_id=str(_point_value(payload, "chat_id")),
        chunk_id=str(_point_value(payload, "chunk_id", str(point.id))),
        started_at=str(_point_value(payload, "started_at")),
        ended_at=str(_point_value(payload, "ended_at")),
        start_sequence=_point_value(payload, "start_sequence", None),
        end_sequence=_point_value(payload, "end_sequence", None),
        message_ids=tuple(str(value) for value in _point_value(payload, "message_ids", [])),
        target_message_ids=tuple(
            str(value) for value in _point_value(payload, "target_message_ids", [])
        ),
        authors=tuple(str(value) for value in _point_value(payload, "authors", [])),
        text=str(_point_value(payload, "text")),
        context=tuple(dict(value) for value in _point_value(payload, "context", [])),
        target_responses=tuple(
            dict(value) for value in _point_value(payload, "target_responses", [])
        ),
    )


def serialize_hits_json(
    query: str, mode: SearchMode, hits: Sequence[WhatsAppSearchHit]
) -> str:
    return json.dumps(
        {
            "query": query,
            "mode": mode,
            "hits": [hit.to_dict() for hit in hits],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _format_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    if not messages:
        return "  (none)"
    return "\n".join(
        f"  {message.get('author', '')}: {message.get('text', '')}"
        for message in messages
    )


def render_hits_human(hits: Sequence[WhatsAppSearchHit]) -> str:
    """Render full conversational inspection details without vectors."""

    if not hits:
        return "No results."
    rendered: list[str] = []
    for hit in hits:
        rendered.append(
            "\n".join(
                (
                    f"#{hit.rank}  retrieval_score={hit.retrieval_score:.6f}"
                    + (
                        f"  reranker_score={hit.reranker_score:.6f}"
                        if hit.reranker_score is not None
                        else ""
                    ),
                    f"Persona: {hit.target_author} ({hit.persona_id})",
                    f"Chat: {hit.chat_id}  Time: {hit.started_at} → {hit.ended_at}",
                    "Context:",
                    _format_messages(hit.context),
                    "Target responses:",
                    _format_messages(hit.target_responses),
                    "Provenance: "
                    f"chunk={hit.chunk_id}; messages={', '.join(hit.message_ids)}; "
                    f"target_messages={', '.join(hit.target_message_ids)}; "
                    f"point={hit.point_id}",
                )
            )
        )
    return "\n\n".join(rendered)


class WhatsAppRetriever:
    """Execute tenant- and persona-scoped direct Qdrant searches."""

    def __init__(
        self,
        url: str = DEFAULT_QDRANT_URL,
        *,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self.client = client
            return
        try:
            from qdrant_client import QdrantClient
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "WhatsApp retrieval requires qdrant-client[fastembed]. "
                "Install the root project with: .venv/bin/python -m pip install -e ."
            ) from exc
        self.client = QdrantClient(url=url, timeout=120)

    def verify_collection(self, collection_name: str) -> None:
        """Check collection existence and vector names without mutating it."""

        try:
            if not self.client.collection_exists(collection_name):
                raise CollectionMissingError(
                    f"Qdrant collection {collection_name!r} does not exist. "
                    "Index WhatsApp data into mirror_whatsapp_v2 first."
                )
            params = self.client.get_collection(collection_name).config.params
        except CollectionMissingError:
            raise
        except Exception as exc:
            raise QdrantUnavailableError(f"Unable to contact Qdrant: {exc}") from exc
        vectors = params.vectors
        sparse_vectors = params.sparse_vectors
        has_dense = isinstance(vectors, Mapping) and DENSE_VECTOR in vectors
        has_sparse = isinstance(sparse_vectors, Mapping) and SPARSE_VECTOR in sparse_vectors
        if not has_dense or not has_sparse:
            missing = [
                name
                for name, exists in ((DENSE_VECTOR, has_dense), (SPARSE_VECTOR, has_sparse))
                if not exists
            ]
            raise CollectionIncompatibleError(
                f"Qdrant collection {collection_name!r} is incompatible; "
                f"missing named vector(s): {', '.join(missing)}."
            )

    def search(
        self,
        query: str,
        *,
        user_id: str,
        persona_id: str,
        collection_name: str = DEFAULT_COLLECTION,
        mode: SearchMode = "hybrid_rrf",
        limit: int = 10,
        prefetch_limit: int = 30,
        chat_id: str | None = None,
    ) -> list[WhatsAppSearchHit]:
        """Search only points owned by one user and one stable persona."""

        validate_search_arguments(
            query, user_id, persona_id, mode, limit, prefetch_limit
        )
        _required_identifier(collection_name, "collection_name")
        query_filter = build_required_filter(user_id, persona_id, chat_id)
        self.verify_collection(collection_name)

        from qdrant_client import models

        dense = models.Document(text=query, model=DENSE_MODEL)
        sparse = models.Document(text=query, model=SPARSE_MODEL)
        common = {
            "collection_name": collection_name,
            "query_filter": query_filter,
            "limit": limit,
            "with_payload": True,
            "with_vectors": False,
        }
        if mode == "dense":
            request = {**common, "query": dense, "using": DENSE_VECTOR}
        elif mode == "sparse":
            request = {**common, "query": sparse, "using": SPARSE_VECTOR}
        else:
            fusion = models.Fusion.RRF if mode == "hybrid_rrf" else models.Fusion.DBSF
            request = {
                **common,
                "prefetch": [
                    models.Prefetch(
                        query=dense,
                        using=DENSE_VECTOR,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=sparse,
                        using=SPARSE_VECTOR,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                ],
                "query": models.FusionQuery(fusion=fusion),
            }
        try:
            response = self.client.query_points(**request)
        except Exception as exc:
            raise QdrantUnavailableError(f"Qdrant search failed: {exc}") from exc
        return [
            point_to_search_hit(point, rank)
            for rank, point in enumerate(response.points, start=1)
        ]
