from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Sequence

from .models import WhatsAppMessage


@dataclass(frozen=True)
class ConversationChunk:
    chunk_id: str
    chat_id: str
    start_sequence: int
    end_sequence: int
    started_at: datetime
    ended_at: datetime
    message_ids: tuple[str, ...]
    authors: tuple[str, ...]
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "chat_id": self.chat_id,
            "start_sequence": self.start_sequence,
            "end_sequence": self.end_sequence,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "message_ids": list(self.message_ids),
            "authors": list(self.authors),
            "text": self.text,
        }


def _render(message: WhatsAppMessage) -> str:
    return f"{message.author}: {message.text}"


def _make_chunk(messages: Sequence[WhatsAppMessage]) -> ConversationChunk:
    text = "\n".join(_render(message) for message in messages)
    identity = "\x00".join(message.message_id for message in messages)
    return ConversationChunk(
        chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32],
        chat_id=messages[0].chat_id,
        start_sequence=messages[0].sequence,
        end_sequence=messages[-1].sequence,
        started_at=messages[0].timestamp,
        ended_at=messages[-1].timestamp,
        message_ids=tuple(message.message_id for message in messages),
        authors=tuple(dict.fromkeys(message.author for message in messages if message.author)),
        text=text,
    )


def chunk_conversation(
    messages: Sequence[WhatsAppMessage],
    *,
    max_messages: int = 20,
    max_characters: int = 4_000,
    overlap_messages: int = 4,
    max_gap_minutes: int = 120,
) -> list[ConversationChunk]:
    """Create overlapping chat-session chunks on whole-message boundaries."""

    if max_messages < 1:
        raise ValueError("max_messages must be at least 1")
    if max_characters < 1:
        raise ValueError("max_characters must be at least 1")
    if overlap_messages < 0 or overlap_messages >= max_messages:
        raise ValueError("overlap_messages must be between 0 and max_messages - 1")
    if max_gap_minutes < 1:
        raise ValueError("max_gap_minutes must be at least 1")

    usable = [message for message in messages if message.is_usable_text]
    if not usable:
        return []
    if len({message.chat_id for message in usable}) != 1:
        raise ValueError("all messages in a chunking call must share one chat_id")

    sessions: list[list[WhatsAppMessage]] = []
    max_gap = timedelta(minutes=max_gap_minutes)
    for message in usable:
        if not sessions or message.timestamp - sessions[-1][-1].timestamp > max_gap:
            sessions.append([message])
        else:
            sessions[-1].append(message)

    chunks: list[ConversationChunk] = []
    for session in sessions:
        start = 0
        while start < len(session):
            selected: list[WhatsAppMessage] = []
            rendered_characters = 0

            for message in session[start:]:
                rendered = _render(message)
                separator_size = 1 if selected else 0
                exceeds_size = (
                    selected
                    and rendered_characters + separator_size + len(rendered)
                    > max_characters
                )
                if len(selected) >= max_messages or exceeds_size:
                    break
                selected.append(message)
                rendered_characters += separator_size + len(rendered)

            chunks.append(_make_chunk(selected))
            if start + len(selected) >= len(session):
                break

            advance = max(1, len(selected) - overlap_messages)
            start += advance

    return chunks
