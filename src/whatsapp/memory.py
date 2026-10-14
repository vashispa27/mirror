"""All-speaker, contact-scoped memory records and conservative exact assertions."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Sequence

from .semantic_topics import semantic_topic_boundaries


MEMORY_VERSION = "memory-v2"
FACT_VERSION = "rules-v1"
MEMORY_COLLECTION = "mirror_whatsapp_memory_v2"
_DATE = r"(?:\d{1,2}(?:st|nd|rd|th)?[ /.-]+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{1,2})(?:[ ,/.-]+\d{2,4})?|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ ]+\d{1,2}(?:st|nd|rd|th)?(?:,? \d{4})?)"
_BIRTHDAY = re.compile(rf"\b(?P<subject>my mom|mom|mother|dad|father|[A-Z][a-z]+(?: [A-Z][a-z]+)?)['’]s birthday\s+(?:is|falls on|is on)\s+(?P<value>{_DATE})\b", re.I)
_EVENT = re.compile(rf"\b(?P<subject>party|wedding|appointment|meeting|trip|exam)\s+is\s+(?:on|scheduled for)\s+(?P<value>{_DATE})\b", re.I)
_ALLERGY = re.compile(r"\b(?P<subject>I|my mom|mom|dad|[A-Z][a-z]+)\s+(?:am|is)\s+allergic to\s+(?P<value>[A-Za-z][A-Za-z -]{1,40})\b", re.I)
_PHONE = re.compile(r"\b(?P<subject>my|mom['’]s|dad['’]s|[A-Z][a-z]+['’]s)\s+(?:phone|number)\s+is\s+(?P<value>\+?\d[\d ()-]{7,18}\d)\b", re.I)
_ADDRESS = re.compile(r"\b(?P<subject>my|mom['’]s|dad['’]s|[A-Z][a-z]+['’]s)\s+address\s+is\s+(?P<value>\d{1,6}\s+[A-Za-z0-9 .,-]{5,90})", re.I)
_IDENTIFIER = re.compile(r"\b(?P<subject>my|mom['’]s|dad['’]s|[A-Z][a-z]+['’]s)\s+(?:passport|employee|membership)\s+(?:ID|number)\s+is\s+(?P<value>[A-Z0-9][A-Z0-9-]{4,24})\b", re.I)
_NEGATED = re.compile(r"\b(?:not|don't|doesn't|isn't|maybe|perhaps|guess|might|could be|I think|supposedly)\b", re.I)


@dataclass(frozen=True)
class MemoryChunk:
    id: uuid.UUID
    user_id: str
    contact_id: str
    chat_id: str
    started_at: datetime
    ended_at: datetime
    start_sequence: int
    end_sequence: int
    message_ids: tuple[uuid.UUID, ...]
    authors: tuple[str, ...]
    text: str
    schema_version: str = MEMORY_VERSION
    topic_id: uuid.UUID | None = None

    def payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version, "user_id": self.user_id,
            "contact_id": self.contact_id, "chat_id": self.chat_id,
            "chunk_id": str(self.id), "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(), "start_sequence": self.start_sequence,
            "end_sequence": self.end_sequence, "message_ids": [str(x) for x in self.message_ids],
            "authors": list(self.authors), "text": self.text,
        }
        if self.topic_id is not None:
            payload["topic_id"] = str(self.topic_id)
            payload["record_type"] = "span"
        return payload


@dataclass(frozen=True)
class Fact:
    id: uuid.UUID
    user_id: str
    contact_id: str
    subject_key: str
    subject_label: str
    stated_by: str
    fact_type: str
    raw_value: str
    normalized_value: str
    confidence: float
    asserted_at: datetime
    context: str
    message_ids: tuple[uuid.UUID, ...]
    chunk_id: uuid.UUID
    extraction_version: str = FACT_VERSION


def build_memory_chunks(
    rows: Sequence[dict[str, Any]], *, user_id: str, contact_id: str,
    max_messages: int = 20, max_characters: int = 4000,
    max_gap_minutes: int = 120, overlap_messages: int = 4,
    topic_boundary_detector: Callable[[Sequence[dict[str, Any]]], set[int]] = semantic_topic_boundaries,
) -> list[MemoryChunk]:
    if not user_id or not contact_id or max_messages < 1 or max_characters < 1 or max_gap_minutes < 1 or not 0 <= overlap_messages < max_messages:
        raise ValueError("invalid scope or chunk settings")
    usable = [r for r in rows if r["normalized_text"] and not (r["is_system"] or r["is_deleted"] or r["has_media"])]
    if not usable:
        return []
    if any(r["user_id"] != user_id or r["contact_id"] != contact_id for r in usable):
        raise ValueError("mixed contact scope")
    time_sessions: list[list[dict[str, Any]]] = []
    for row in usable:
        previous = time_sessions[-1][-1] if time_sessions else None
        boundary = previous and (row["chat_id"] != previous["chat_id"] or
            row["sent_at"] - previous["sent_at"] > timedelta(minutes=max_gap_minutes))
        if not time_sessions or boundary:
            time_sessions.append([])
        time_sessions[-1].append(row)
    sessions: list[list[dict[str, Any]]] = []
    for time_session in time_sessions:
        boundaries = topic_boundary_detector(time_session)
        if any(not isinstance(i, int) or i <= 0 or i >= len(time_session) for i in boundaries):
            raise ValueError("topic detector returned an invalid boundary")
        starts = [0, *sorted(boundaries), len(time_session)]
        sessions.extend(time_session[a:b] for a, b in zip(starts, starts[1:]))
    chunks: list[MemoryChunk] = []
    for session in sessions:
        start = 0
        while start < len(session):
            selected: list[dict[str, Any]] = []
            size = 0
            for row in session[start:]:
                rendered = f"[{row['sent_at'].isoformat()}] {row['author'] or 'System'}: {row['normalized_text']}"
                if selected and (len(selected) >= max_messages or size + len(rendered) + 1 > max_characters):
                    break
                selected.append(row)
                size += len(rendered) + 1
            ids = tuple(r["id"] for r in selected)
            chunk_id = uuid.uuid5(uuid.NAMESPACE_URL, "mirror-memory:" + MEMORY_VERSION + "\0" + "\0".join(map(str, ids)))
            chunks.append(MemoryChunk(
                id=chunk_id, user_id=user_id, contact_id=contact_id,
                chat_id=selected[0]["chat_id"], started_at=selected[0]["sent_at"],
                ended_at=selected[-1]["sent_at"], start_sequence=start,
                end_sequence=start + len(selected) - 1, message_ids=ids,
                authors=tuple(dict.fromkeys(r["author"] for r in selected if r["author"])),
                text="\n".join(f"[{r['sent_at'].isoformat()}] {r['author'] or 'System'}: {r['normalized_text']}" for r in selected),
            ))
            if start + len(selected) >= len(session):
                break
            start += max(1, len(selected) - overlap_messages)
    return chunks


def _subject(value: str, author: str) -> tuple[str, str]:
    label = value.strip().removesuffix("'s").removesuffix("’s")
    if label.lower() in ("my", "i"):
        label = author
    elif label.lower() == "my mom":
        label = "mom"
    return re.sub(r"\s+", "_", label.casefold()), label


def _normalize_date(raw: str) -> str:
    value = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", raw, flags=re.I)
    value = re.sub(r"\s+", " ", value.replace(",", " ")).strip()
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y", "%d/%m/%Y", "%d/%m/%y", "%d %B", "%d %b", "%B %d", "%b %d", "%d/%m"):
        try:
            parsed = datetime.strptime(value, fmt)
            return parsed.strftime("%Y-%m-%d") if "%Y" in fmt or "%y" in fmt else parsed.strftime("--%m-%d")
        except ValueError:
            pass
    return value.casefold()


def extract_facts(chunk: MemoryChunk, rows: Sequence[dict[str, Any]]) -> list[Fact]:
    by_id = {r["id"]: r for r in rows}
    facts: list[Fact] = []
    for mid in chunk.message_ids:
        row = by_id[mid]
        text = row["normalized_text"]
        # Quotes, hedges and negation are not promoted to exact assertions.
        if _NEGATED.search(text) or text.lstrip().startswith(('"', "'", ">")):
            continue
        for pattern, fact_type in ((_BIRTHDAY, "birthday"), (_EVENT, "event_date"), (_ALLERGY, "allergy"), (_PHONE, "phone"), (_ADDRESS, "address"), (_IDENTIFIER, "identifier")):
            for match in pattern.finditer(text):
                raw = match.group("value").strip().rstrip(".,!?")
                subject_key, subject_label = _subject(match.group("subject"), row["author"] or "unknown")
                normalized = _normalize_date(raw) if fact_type in ("birthday", "event_date") else re.sub(r"\D", "", raw) if fact_type == "phone" else raw.casefold()
                identity = "\0".join((str(mid), subject_key, fact_type, normalized, FACT_VERSION))
                facts.append(Fact(
                    id=uuid.uuid5(uuid.NAMESPACE_URL, "mirror-fact:" + identity),
                    user_id=chunk.user_id, contact_id=chunk.contact_id,
                    subject_key=subject_key, subject_label=subject_label,
                    stated_by=row["author"] or "unknown", fact_type=fact_type,
                    raw_value=raw, normalized_value=normalized, confidence=0.95,
                    asserted_at=row["sent_at"], context=text[:500],
                    message_ids=(mid,), chunk_id=chunk.id,
                ))
    return facts


def export_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8-sig")).hexdigest()
