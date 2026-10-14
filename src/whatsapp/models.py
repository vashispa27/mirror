from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional


@dataclass(frozen=True)
class WhatsAppMessage:
    """A normalized message from a WhatsApp text export."""

    message_id: str
    chat_id: str
    sequence: int
    timestamp: datetime
    author: Optional[str]
    text: str
    source_timezone: Optional[str]
    is_system: bool = False
    is_edited: bool = False
    is_deleted: bool = False
    has_media: bool = False
    raw_content: Optional[str] = None

    @property
    def is_usable_text(self) -> bool:
        return bool(self.text) and not (
            self.is_system or self.is_deleted or self.has_media
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["timestamp"] = self.timestamp.isoformat()
        value["is_usable_text"] = self.is_usable_text
        return value


@dataclass(frozen=True)
class PersonaMessage:
    """A usable message labelled for persona-oriented retrieval."""

    message_id: str
    author: str
    text: str
    sequence: int
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "author": self.author,
            "text": self.text,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(frozen=True)
class PersonaConversationRecord:
    """A conversation chunk with the target's messages explicitly labelled.

    Datetimes stay typed in the application model and are converted only at the
    storage boundary by :meth:`to_payload`.
    """

    schema_version: str
    user_id: str
    persona_id: str
    source: str
    chat_id: str
    requested_target_author: str
    target_author: str
    chunk_id: str
    authors: tuple[str, ...]
    started_at: datetime
    ended_at: datetime
    start_sequence: int
    end_sequence: int
    message_ids: tuple[str, ...]
    target_message_ids: tuple[str, ...]
    context: tuple[PersonaMessage, ...]
    target_responses: tuple[PersonaMessage, ...]
    text: str

    def to_payload(self) -> dict[str, Any]:
        """Return only JSON-compatible values for a Qdrant payload."""

        return {
            "schema_version": self.schema_version,
            "user_id": self.user_id,
            "persona_id": self.persona_id,
            "source": self.source,
            "chat_id": self.chat_id,
            "requested_target_author": self.requested_target_author,
            "target_author": self.target_author,
            "authors": list(self.authors),
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "start_sequence": self.start_sequence,
            "end_sequence": self.end_sequence,
            "message_ids": list(self.message_ids),
            "target_message_ids": list(self.target_message_ids),
            "context": [message.to_dict() for message in self.context],
            "target_responses": [
                message.to_dict() for message in self.target_responses
            ],
            "text": self.text,
            "chunk_id": self.chunk_id,
        }
