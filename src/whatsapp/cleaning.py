from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# WhatsApp exports can contain bidi/formatting marks around notices. Do not
# remove U+200D (zero-width joiner), because it is part of many emoji sequences.
_FORMAT_MARKS = re.compile("[\u200b\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_EDITED_MARKER = re.compile(
    r"\s*\\?<this message was edited>\s*$", re.IGNORECASE
)
_DELETED_MARKERS = {
    "this message was deleted",
    "you deleted this message",
}
_MEDIA_MARKER = re.compile(
    r"^<?(?:media|image|video|audio|document|gif|sticker) omitted>?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CleanedText:
    text: str
    is_edited: bool = False
    is_deleted: bool = False
    has_media: bool = False


def normalize_export_text(value: str) -> str:
    """Normalize transport artifacts while preserving the writer's style."""

    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = _FORMAT_MARKS.sub("", value)
    value = value.replace("\u00a0", " ").replace("\u202f", " ")
    value = unicodedata.normalize("NFC", value)

    lines = [line.rstrip() for line in value.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    return "\n".join(lines)


def clean_message_text(value: str) -> CleanedText:
    """Remove WhatsApp metadata notices without rewriting user language."""

    text = normalize_export_text(value)
    edited = bool(_EDITED_MARKER.search(text))
    if edited:
        text = _EDITED_MARKER.sub("", text).rstrip()

    lowered = text.casefold().strip("<> ")
    deleted = lowered in _DELETED_MARKERS
    media = bool(_MEDIA_MARKER.fullmatch(text.strip()))

    if deleted or media:
        text = ""

    return CleanedText(
        text=text,
        is_edited=edited,
        is_deleted=deleted,
        has_media=media,
    )
