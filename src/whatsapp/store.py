"""Production Qdrant indexing for persona-labelled WhatsApp records.

This module deliberately uses qdrant-client directly.  It shares no runtime
code with ``rag_lab`` so experimentation cannot affect ingestion behavior.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .models import PersonaConversationRecord


DEFAULT_COLLECTION = "mirror_whatsapp_v2"
LEGACY_COLLECTION = "mirror_whatsapp_v1"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

#: Dense encoder for the voice index. Override with MIRROR_DENSE_MODEL to
#: compare encoders without editing code; the query encoder reads the same
#: value, so both sides of the search stay in step.
DEFAULT_DENSE_MODEL = "BAAI/bge-small-en-v1.5"
DENSE_MODEL = os.environ.get("MIRROR_DENSE_MODEL", DEFAULT_DENSE_MODEL)
SPARSE_MODEL = "Qdrant/bm25"


@dataclass(frozen=True)
class IndexResult:
    parsed_message_count: int
    persona_chunk_count: int
    indexed_point_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "parsed_message_count": self.parsed_message_count,
            "persona_chunk_count": self.persona_chunk_count,
            "indexed_point_count": self.indexed_point_count,
        }


def stable_point_id(record: PersonaConversationRecord) -> str:
    """Return a stable Qdrant UUID scoped to its owner and persona identity."""

    identity = "\x00".join(
        (
            record.schema_version,
            record.source,
            record.user_id,
            record.persona_id,
            record.target_author,
            record.chat_id,
            record.chunk_id,
        )
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mirror-whatsapp:{identity}"))


def searchable_text(record: PersonaConversationRecord) -> str:
    """Make both the conversation and target wording explicit to embeddings."""

    context = "\n".join(
        f"{message.author}: {message.text}" for message in record.context
    )
    responses = "\n".join(
        f"{message.author}: {message.text}" for message in record.target_responses
    )
    return (
        "Conversation (chronological, speaker-labelled):\n"
        f"{record.text}\n\n"
        "Conversational context (other authors):\n"
        f"{context or '(none)'}\n\n"
        f"Target responses ({record.target_author}):\n{responses}"
    )


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value)).lower()


class WhatsAppQdrantStore:
    """Create-compatible collections and upsert stable persona records."""

    def __init__(self, url: str = DEFAULT_QDRANT_URL) -> None:
        try:
            from qdrant_client import QdrantClient
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "WhatsApp indexing requires qdrant-client[fastembed]. "
                "Install the root project with: .venv/bin/python -m pip install -e ."
            ) from exc
        self.client = QdrantClient(url=url, timeout=120)

    def ensure_collection(self, collection_name: str) -> None:
        """Create a missing collection, or verify an existing one without mutation."""

        from qdrant_client import models

        if collection_name == LEGACY_COLLECTION:
            raise ValueError(
                "mirror_whatsapp_v1 is a legacy collection and is intentionally "
                "read-only. Re-index into mirror_whatsapp_v2 instead."
            )
        dense_size = self.client.get_embedding_size(DENSE_MODEL)
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=dense_size,
                        distance=models.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(
                        modifier=models.Modifier.IDF
                    )
                },
            )
            self.ensure_payload_indexes(collection_name)
            return

        info = self.client.get_collection(collection_name)
        params = info.config.params
        vectors = params.vectors
        sparse_vectors = params.sparse_vectors
        dense = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
        sparse = (
            sparse_vectors.get(SPARSE_VECTOR)
            if isinstance(sparse_vectors, dict)
            else None
        )
        issues: list[str] = []
        if dense is None:
            issues.append(f"missing dense vector {DENSE_VECTOR!r}")
        else:
            if dense.size != dense_size:
                issues.append(
                    f"dense vector size is {dense.size}, expected {dense_size}"
                )
            if _enum_value(dense.distance) != "cosine":
                issues.append(
                    "dense vector distance is {}, expected cosine".format(
                        _enum_value(dense.distance)
                    )
                )
        if sparse is None:
            issues.append(f"missing sparse vector {SPARSE_VECTOR!r}")
        elif _enum_value(sparse.modifier) != "idf":
            issues.append(
                "sparse vector modifier is {}, expected idf".format(
                    _enum_value(sparse.modifier)
                )
            )
        if issues:
            raise ValueError(
                "Existing Qdrant collection {!r} is incompatible with WhatsApp "
                "indexing: {}. Choose a new collection or migrate it explicitly; "
                "the indexer will not recreate production collections.".format(
                    collection_name, "; ".join(issues)
                )
            )
        self.ensure_payload_indexes(collection_name)

    def ensure_payload_indexes(self, collection_name: str) -> None:
        """Create keyword payload indexes used by mandatory retrieval filters.

        Qdrant treats creating the same payload index as an idempotent update;
        this never recreates or clears the collection.
        """

        from qdrant_client import models

        for field_name in (
            "user_id",
            "persona_id",
            "source",
            "schema_version",
            "chat_id",
        ):
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def upload(
        self,
        collection_name: str,
        records: Iterable[PersonaConversationRecord],
        *,
        batch_size: int = 64,
    ) -> int:
        """Upsert records, letting stable UUIDs make repeated runs idempotent."""

        from qdrant_client import models

        count = 0

        def points():
            nonlocal count
            for count, record in enumerate(records, start=1):
                text = searchable_text(record)
                yield models.PointStruct(
                    id=stable_point_id(record),
                    vector={
                        DENSE_VECTOR: models.Document(text=text, model=DENSE_MODEL),
                        SPARSE_VECTOR: models.Document(text=text, model=SPARSE_MODEL),
                    },
                    payload=record.to_payload(),
                )

        self.client.upload_points(
            collection_name=collection_name,
            points=points(),
            batch_size=batch_size,
        )
        return count

    def index(
        self,
        collection_name: str,
        records: Sequence[PersonaConversationRecord],
        *,
        parsed_message_count: int,
    ) -> IndexResult:
        self.ensure_collection(collection_name)
        count = self.upload(collection_name, records)
        return IndexResult(
            parsed_message_count=parsed_message_count,
            persona_chunk_count=len(records),
            indexed_point_count=count,
        )
