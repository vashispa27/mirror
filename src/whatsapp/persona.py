"""Build structured, persona-aware records from conversation chunks."""

from __future__ import annotations

from collections.abc import Sequence

from .chunking import ConversationChunk
from .models import PersonaConversationRecord, PersonaMessage, WhatsAppMessage


SCHEMA_VERSION = "whatsapp_persona_v2"


def _persona_message(message: WhatsAppMessage) -> PersonaMessage:
    if message.author is None:
        raise ValueError("persona records cannot contain system messages")
    return PersonaMessage(
        message_id=message.message_id,
        author=message.author,
        text=message.text,
        sequence=message.sequence,
        timestamp=message.timestamp,
    )


def build_persona_records(
    messages: Sequence[WhatsAppMessage],
    chunks: Sequence[ConversationChunk],
    *,
    user_id: str,
    persona_id: str,
    requested_target_author: str,
    target_author: str,
) -> list[PersonaConversationRecord]:
    """Label target messages and context, omitting chunks with no target text."""

    if not user_id.strip():
        raise ValueError("user_id must not be empty")
    if not persona_id.strip():
        raise ValueError("persona_id must not be empty")
    by_id = {message.message_id: message for message in messages}
    records: list[PersonaConversationRecord] = []

    for chunk in chunks:
        chunk_messages = [by_id[message_id] for message_id in chunk.message_ids]
        target_messages = [
            message
            for message in chunk_messages
            if message.is_usable_text and message.author == target_author
        ]
        if not target_messages:
            continue
        context_messages = [
            message
            for message in chunk_messages
            if message.is_usable_text and message.author != target_author
        ]
        records.append(
            PersonaConversationRecord(
                schema_version=SCHEMA_VERSION,
                user_id=user_id,
                persona_id=persona_id,
                source="whatsapp",
                chat_id=chunk.chat_id,
                requested_target_author=requested_target_author,
                target_author=target_author,
                chunk_id=chunk.chunk_id,
                authors=chunk.authors,
                started_at=chunk.started_at,
                ended_at=chunk.ended_at,
                start_sequence=chunk.start_sequence,
                end_sequence=chunk.end_sequence,
                message_ids=chunk.message_ids,
                target_message_ids=tuple(
                    message.message_id for message in target_messages
                ),
                context=tuple(_persona_message(message) for message in context_messages),
                target_responses=tuple(
                    _persona_message(message) for message in target_messages
                ),
                text=chunk.text,
            )
        )
    return records
