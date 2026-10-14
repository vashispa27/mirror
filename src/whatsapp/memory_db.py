"""Postgres source of truth for contact-scoped WhatsApp memory."""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from datetime import timezone
from pathlib import Path
from typing import Any, Sequence

from .models import WhatsAppMessage


def _fingerprint(message: WhatsAppMessage) -> str:
    # Sequence is deliberately excluded: partial exports can shift it.
    value = "\0".join((message.timestamp.isoformat(), message.author or "", message.raw_content or message.text))
    return hashlib.sha256(value.encode()).hexdigest()


class MemoryRepository:
    def __init__(self, dsn: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
            from psycopg.types.json import Jsonb
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install psycopg[binary] to use WhatsApp memory") from exc
        # Reads must not leave an implicit outer transaction around later writes.
        self.conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
        self._jsonb = Jsonb

    def close(self) -> None:
        self.conn.close()

    def migrate(self) -> None:
        root = Path(__file__).resolve().parents[2] / "migrations"
        with self.conn.transaction(), self.conn.cursor() as cur:
            for name in ("001_whatsapp_memory.sql", "002_topic_memory.sql"):
                cur.execute((root / name).read_text())

    def import_export(
        self,
        messages: Sequence[WhatsAppMessage],
        *,
        user_id: str,
        contact_id: str,
        chat_id: str,
        export_hash: str,
        owner_author: str | None = None,
    ) -> list[dict[str, Any]]:
        for name, value in (("user_id", user_id), ("contact_id", contact_id), ("chat_id", chat_id), ("export_hash", export_hash)):
            if not value or not value.strip():
                raise ValueError(f"{name} is required")
        if any(m.chat_id != chat_id for m in messages):
            raise ValueError("message chat_id differs from import chat_id")
        fingerprints = [_fingerprint(m) for m in messages]
        counts = Counter(fingerprints)
        ids: list[uuid.UUID] = []
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """SELECT source_sequence,message_id FROM message_imports
                   WHERE user_id=%s AND contact_id=%s AND chat_id=%s AND export_hash=%s""",
                (user_id, contact_id, chat_id, export_hash),
            )
            imported = {row["source_sequence"]: row["message_id"] for row in cur.fetchall()}
            if len(imported) == len(messages) and all(m.sequence in imported for m in messages):
                return self.messages_by_ids(user_id, contact_id, [imported[m.sequence] for m in messages])
            cur.execute(
                """SELECT id,source_fingerprint FROM messages WHERE user_id=%s AND contact_id=%s
                   AND chat_id=%s AND source_fingerprint IS NOT NULL""",
                (user_id, contact_id, chat_id),
            )
            existing_by_fingerprint: dict[str, list[uuid.UUID]] = {}
            for row in cur.fetchall():
                existing_by_fingerprint.setdefault(row["source_fingerprint"], []).append(row["id"])
            for message, fingerprint in zip(messages, fingerprints):
                if message.sequence in imported:
                    ids.append(imported[message.sequence])
                    continue
                # Reuse a message across changed/partial exports only when its
                # exact original content and timestamp are unambiguous on both sides.
                existing = existing_by_fingerprint.get(fingerprint, []) if counts[fingerprint] == 1 else []
                internal_id = existing[0] if len(existing) == 1 else uuid.uuid4()
                if not existing or len(existing) != 1:
                    sent_at = message.timestamp if message.timestamp.tzinfo else message.timestamp.replace(tzinfo=timezone.utc)
                    direction = "system" if message.is_system else ("outgoing" if owner_author and message.author == owner_author else "incoming" if owner_author else "unknown")
                    cur.execute(
                        """INSERT INTO messages (id,user_id,contact_id,chat_id,author,direction,sent_at,
                           original_content,normalized_text,is_system,is_edited,is_deleted,has_media,
                           import_provenance,source_fingerprint)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                        (internal_id,user_id,contact_id,chat_id,message.author,direction,sent_at,
                         message.raw_content,message.text,message.is_system,message.is_edited,
                         message.is_deleted,message.has_media,
                         self._jsonb({"export_hash": export_hash, "sequence": message.sequence, "source_timezone": message.source_timezone}),fingerprint),
                    )
                cur.execute(
                    """INSERT INTO message_imports VALUES (%s,%s,%s,%s,%s,%s)
                       ON CONFLICT DO NOTHING""",
                    (user_id, contact_id, chat_id, export_hash, message.sequence, internal_id),
                )
                ids.append(internal_id)
        return self.messages_by_ids(user_id, contact_id, ids)

    def add_message(
        self, *, user_id: str, contact_id: str, chat_id: str, text: str,
        author: str, direction: str, sent_at: Any, source_message_id: str | None = None,
        original_payload: Any = None,
    ) -> uuid.UUID:
        if not all((user_id, contact_id, chat_id, author)) or not text.strip():
            raise ValueError("scope, author and text are required")
        if direction not in ("incoming", "outgoing"):
            raise ValueError("direction must be incoming or outgoing")
        internal_id = uuid.uuid4()
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO messages (id,user_id,contact_id,chat_id,source_platform,
                   source_message_id,author,direction,sent_at,original_content,original_payload,normalized_text)
                   VALUES (%s,%s,%s,%s,'whatsapp_live',%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (user_id,contact_id,source_platform,source_message_id)
                   WHERE source_message_id IS NOT NULL DO NOTHING RETURNING id""",
                (internal_id,user_id,contact_id,chat_id,source_message_id,author,direction,sent_at,text,self._jsonb(original_payload) if original_payload is not None else None,text),
            )
            row = cur.fetchone()
            if row:
                return row["id"]
            cur.execute(
                """SELECT id FROM messages WHERE user_id=%s AND contact_id=%s
                   AND source_platform='whatsapp_live' AND source_message_id=%s""",
                (user_id,contact_id,source_message_id),
            )
            return cur.fetchone()["id"]

    def messages_by_ids(self, user_id: str, contact_id: str, ids: Sequence[uuid.UUID]) -> list[dict[str, Any]]:
        if not ids:
            return []
        normalized_ids = [value if isinstance(value, uuid.UUID) else uuid.UUID(str(value)) for value in ids]
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM messages WHERE user_id=%s AND contact_id=%s AND id=ANY(%s)""",
                (user_id, contact_id, normalized_ids),
            )
            found = {row["id"]: row for row in cur.fetchall()}
        return [found[i] for i in normalized_ids if i in found]

    def recent_messages(
        self, user_id: str, contact_id: str, *, chat_id: str | None = None,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        """Return L0: messages seen by the live bot, never export backfill."""

        if not user_id or not contact_id or not 1 <= limit <= 100:
            raise ValueError("contact scope and limit 1..100 are required")
        if chat_id is not None and not chat_id.strip():
            raise ValueError("chat_id must not be empty")
        chat_clause = " AND chat_id=%s" if chat_id is not None else ""
        params: list[Any] = [user_id, contact_id]
        if chat_id is not None:
            params.append(chat_id)
        params.append(limit)
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM messages WHERE user_id=%s AND contact_id=%s
                   AND source_platform='whatsapp_live'""" + chat_clause +
                " ORDER BY sent_at DESC, created_at DESC LIMIT %s",
                params,
            )
            return list(reversed(cur.fetchall()))

    def messages_for_chat(self, user_id: str, contact_id: str, chat_id: str) -> list[dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT * FROM messages WHERE user_id=%s AND contact_id=%s AND chat_id=%s
                   ORDER BY sent_at, created_at, id""",
                (user_id, contact_id, chat_id),
            )
            return cur.fetchall()

    def active_chunk_ids(self, user_id: str, contact_id: str, chat_id: str,
                         version: str | None = None) -> set[uuid.UUID]:
        version_clause = " AND processing_version=%s" if version else ""
        params = (user_id, contact_id, chat_id, version) if version else (user_id, contact_id, chat_id)
        with self.conn.cursor() as cur:
            cur.execute(
                """SELECT id FROM memory_chunks WHERE user_id=%s AND contact_id=%s AND chat_id=%s
                   AND processing_state <> 'retired'""" + version_clause,
                params,
            )
            return {row["id"] for row in cur.fetchall()}

    def active_topic_ids(self, user_id: str, contact_id: str, chat_id: str,
                         version: str) -> set[uuid.UUID]:
        with self.conn.cursor() as cur:
            cur.execute("""SELECT id FROM memory_topics WHERE user_id=%s AND contact_id=%s
                           AND chat_id=%s AND processing_version=%s AND processing_state <> 'retired'""",
                        (user_id, contact_id, chat_id, version))
            return {row["id"] for row in cur.fetchall()}

    def upsert_topic(self, topic: Any, version: str) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute("""INSERT INTO memory_topics (id,user_id,contact_id,chat_id,label,summary,
                           source_message_ids,processing_version) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (id) DO UPDATE SET label=EXCLUDED.label,
                           summary=EXCLUDED.summary,source_message_ids=EXCLUDED.source_message_ids,
                           processing_state='pending'""",
                        (topic.id,topic.user_id,topic.contact_id,topic.chat_id,topic.label,
                         topic.summary,list(topic.message_ids),version))

    def mark_topic_indexed(self, topic_id: uuid.UUID) -> None:
        with self.conn.cursor() as cur:
            cur.execute("UPDATE memory_topics SET processing_state='indexed',indexed_at=now() WHERE id=%s",
                        (topic_id,))

    def retire_topics(self, ids: set[uuid.UUID]) -> None:
        if ids:
            with self.conn.cursor() as cur:
                cur.execute("UPDATE memory_topics SET processing_state='retired' WHERE id=ANY(%s)",
                            (list(ids),))

    def cached_topic_response(self, user_id: str, contact_id: str, chat_id: str,
                              prompt_hash: str) -> str | None:
        with self.conn.cursor() as cur:
            cur.execute("""SELECT response FROM topic_model_cache WHERE user_id=%s
                           AND contact_id=%s AND chat_id=%s AND prompt_hash=%s""",
                        (user_id, contact_id, chat_id, prompt_hash))
            row = cur.fetchone()
            return row["response"] if row else None

    def cache_topic_response(self, user_id: str, contact_id: str, chat_id: str,
                             prompt_hash: str, response: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("""INSERT INTO topic_model_cache
                           (user_id,contact_id,chat_id,prompt_hash,response)
                           VALUES (%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                        (user_id, contact_id, chat_id, prompt_hash, response))

    def retire_chunks(self, ids: set[uuid.UUID]) -> None:
        if ids:
            with self.conn.transaction(), self.conn.cursor() as cur:
                cur.execute("UPDATE memory_chunks SET processing_state='retired' WHERE id=ANY(%s)", (list(ids),))

    def upsert_chunk(self, chunk: Any) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(
                """INSERT INTO memory_chunks (id,user_id,contact_id,chat_id,started_at,ended_at,
                   start_sequence,end_sequence,source_message_ids,text,processing_version,topic_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (id) DO UPDATE SET processing_state='pending',topic_id=EXCLUDED.topic_id""",
                (chunk.id,chunk.user_id,chunk.contact_id,chunk.chat_id,chunk.started_at,
                 chunk.ended_at,chunk.start_sequence,chunk.end_sequence,list(chunk.message_ids),
                 chunk.text,chunk.schema_version,chunk.topic_id),
            )

    def mark_chunk_indexed(self, chunk_id: uuid.UUID) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute("UPDATE memory_chunks SET processing_state='indexed', indexed_at=now() WHERE id=%s", (chunk_id,))

    def upsert_facts(self, facts: Sequence[Any]) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            for fact in facts:
                cur.execute(
                    """INSERT INTO facts (id,user_id,contact_id,subject_key,subject_label,stated_by,
                       fact_type,raw_value,normalized_value,confidence,asserted_at,context,
                       source_message_ids,chunk_id,extraction_version)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (id) DO NOTHING""",
                    (fact.id,fact.user_id,fact.contact_id,fact.subject_key,fact.subject_label,
                     fact.stated_by,fact.fact_type,fact.raw_value,fact.normalized_value,
                     fact.confidence,fact.asserted_at,fact.context,list(fact.message_ids),
                     fact.chunk_id,fact.extraction_version),
                )

    def find_facts(self, user_id: str, contact_id: str, *, subject_key: str | None = None,
                   fact_type: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        if not user_id or not contact_id or not 1 <= limit <= 100:
            raise ValueError("contact scope and limit 1..100 are required")
        clauses = ["user_id=%s", "contact_id=%s"]
        params: list[Any] = [user_id, contact_id]
        if subject_key:
            clauses.append("subject_key=%s")
            params.append(subject_key)
        if fact_type:
            clauses.append("fact_type=%s")
            params.append(fact_type)
        params.append(limit)
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM facts WHERE " + " AND ".join(clauses) + " ORDER BY asserted_at DESC LIMIT %s",
                params,
            )
            return cur.fetchall()
