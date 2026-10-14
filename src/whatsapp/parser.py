from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

from .cleaning import clean_message_text, normalize_export_text
from .models import WhatsAppMessage


_TIME = r"\d{1,2}:\d{2}(?::\d{2})?[\s\u00a0\u202f]*(?:[AaPp][Mm])?"
_DATE = r"\d{1,2}/\d{1,2}/\d{2,4}"
_IOS_HEADER = re.compile(
    rf"^\[(?P<date>{_DATE}),\s*(?P<time>{_TIME})\]\s*(?P<body>.*)$"
)
_ANDROID_HEADER = re.compile(
    rf"^(?P<date>{_DATE}),\s*(?P<time>{_TIME})\s+-\s+(?P<body>.*)$"
)
_AUTHOR_SEPARATOR = re.compile(r":\s")


@dataclass
class _PendingMessage:
    date: str
    time: str
    body_lines: list[str]


def _parse_timestamp(
    date_value: str,
    time_value: str,
    *,
    day_first: bool,
    timezone_name: Optional[str],
) -> datetime:
    normalized_time = re.sub(r"[\s\u00a0\u202f]+", " ", time_value).strip()
    date_formats = ("%d/%m/%y", "%d/%m/%Y") if day_first else (
        "%m/%d/%y",
        "%m/%d/%Y",
    )
    time_formats = ("%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M")

    for date_format in date_formats:
        for time_format in time_formats:
            try:
                parsed = datetime.strptime(
                    f"{date_value} {normalized_time}",
                    f"{date_format} {time_format}",
                )
                if timezone_name:
                    parsed = parsed.replace(tzinfo=ZoneInfo(timezone_name))
                return parsed
            except ValueError:
                continue

    raise ValueError(
        f"Unsupported WhatsApp timestamp: {date_value}, {time_value}"
    )


def _split_author(body: str) -> tuple[Optional[str], str]:
    match = _AUTHOR_SEPARATOR.search(body)
    if not match:
        return None, body

    author = normalize_export_text(body[: match.start()]).strip()
    text = body[match.end() :]
    return author or None, text


def _message_id(
    chat_id: str,
    sequence: int,
    timestamp: datetime,
    author: Optional[str],
    text: str,
) -> str:
    identity = "\x00".join(
        (chat_id, str(sequence), timestamp.isoformat(), author or "", text)
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:32]


def _build_message(
    pending: _PendingMessage,
    *,
    chat_id: str,
    sequence: int,
    day_first: bool,
    timezone_name: Optional[str],
) -> WhatsAppMessage:
    body = "\n".join(pending.body_lines)
    author, raw_text = _split_author(body)
    cleaned = clean_message_text(raw_text)
    timestamp = _parse_timestamp(
        pending.date,
        pending.time,
        day_first=day_first,
        timezone_name=timezone_name,
    )

    return WhatsAppMessage(
        message_id=_message_id(
            chat_id, sequence, timestamp, author, cleaned.text
        ),
        chat_id=chat_id,
        sequence=sequence,
        timestamp=timestamp,
        author=author,
        text=cleaned.text,
        source_timezone=timezone_name,
        is_system=author is None,
        is_edited=cleaned.is_edited,
        is_deleted=cleaned.is_deleted,
        has_media=cleaned.has_media,
        raw_content=body,
    )


def _lines(export_text: str) -> Iterable[str]:
    normalized = export_text.replace("\r\n", "\n").replace("\r", "\n")
    for index, line in enumerate(normalized.split("\n")):
        if index == 0:
            line = line.lstrip("\ufeff")
        # Some WhatsApp versions prefix every exported record with a bidi mark.
        yield line.lstrip("\u200e\u200f")


def parse_whatsapp_chat(
    export_text: str,
    *,
    chat_id: str,
    day_first: bool = True,
    timezone_name: Optional[str] = None,
) -> list[WhatsAppMessage]:
    """Parse iOS or Android WhatsApp text exports into normalized messages.

    Lines without a timestamp are treated as continuations of the previous
    message, which preserves multiline messages.
    """

    if not chat_id.strip():
        raise ValueError("chat_id must not be empty")

    messages: list[WhatsAppMessage] = []
    pending: Optional[_PendingMessage] = None

    def flush() -> None:
        nonlocal pending
        if pending is None:
            return
        messages.append(
            _build_message(
                pending,
                chat_id=chat_id,
                sequence=len(messages),
                day_first=day_first,
                timezone_name=timezone_name,
            )
        )
        pending = None

    for line_number, line in enumerate(_lines(export_text), start=1):
        match = _IOS_HEADER.match(line) or _ANDROID_HEADER.match(line)
        if match:
            flush()
            pending = _PendingMessage(
                date=match.group("date"),
                time=match.group("time"),
                body_lines=[match.group("body")],
            )
            continue

        if pending is not None:
            pending.body_lines.append(line)
        elif line.strip():
            raise ValueError(
                f"Unexpected content before the first message on line {line_number}"
            )

    flush()
    return messages
