from __future__ import annotations

import hashlib
import math
import re
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence


ChunkingMethod = Literal["whole", "fixed", "sentence", "recursive", "semantic"]
EmbedSentences = Callable[[Sequence[str]], Sequence[Sequence[float]]]


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class ChunkingConfig:
    method: ChunkingMethod = "sentence"
    max_characters: int = 1_000
    overlap_characters: int = 150
    overlap_sentences: int = 1
    semantic_break_percentile: float = 75.0

    def validate(self) -> None:
        if self.max_characters < 1:
            raise ValueError("max_characters must be positive")
        if self.method == "fixed" and not (
            0 <= self.overlap_characters < self.max_characters
        ):
            raise ValueError(
                "overlap_characters must be between 0 and max_characters - 1"
            )
        if self.overlap_sentences < 0:
            raise ValueError("overlap_sentences must not be negative")
        if not 0 <= self.semantic_break_percentile <= 100:
            raise ValueError("semantic_break_percentile must be between 0 and 100")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    method: ChunkingMethod
    title: str
    text: str

    def payload(self) -> dict[str, str | int]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "chunking_method": self.method,
            "title": self.title,
            "text": self.text,
        }


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘(]*[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]


def _stable_chunk_id(
    document_id: str, method: ChunkingMethod, index: int, text: str
) -> str:
    identity = "\x00".join((document_id, method, str(index), text))
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def _as_chunks(
    document: SourceDocument,
    method: ChunkingMethod,
    texts: Iterable[str],
) -> list[Chunk]:
    chunks = []
    for index, text in enumerate(texts):
        normalized = text.strip()
        if not normalized:
            continue
        chunks.append(
            Chunk(
                chunk_id=_stable_chunk_id(
                    document.document_id, method, index, normalized
                ),
                document_id=document.document_id,
                chunk_index=index,
                method=method,
                title=document.title,
                text=normalized,
            )
        )
    return chunks


def fixed_chunks(text: str, max_characters: int, overlap_characters: int) -> list[str]:
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        hard_end = min(len(text), start + max_characters)
        end = hard_end
        if hard_end < len(text):
            whitespace = text.rfind(" ", start, hard_end)
            if whitespace > start:
                end = whitespace
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_characters)
    return [chunk for chunk in chunks if chunk]


def _pack_units(
    units: Sequence[str],
    max_characters: int,
    overlap_units: int,
) -> list[str]:
    if not units:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(units):
        selected: list[str] = []
        size = 0
        for unit in units[start:]:
            extra = len(unit) + (1 if selected else 0)
            if selected and size + extra > max_characters:
                break
            selected.append(unit)
            size += extra

        # Preserve an over-sized atomic unit rather than silently dropping it.
        if not selected:
            selected = [units[start]]
        chunks.append(" ".join(selected))
        if start + len(selected) >= len(units):
            break
        start += max(1, len(selected) - overlap_units)
    return chunks


def sentence_chunks(
    text: str, max_characters: int, overlap_sentences: int
) -> list[str]:
    return _pack_units(
        split_sentences(text),
        max_characters=max_characters,
        overlap_units=overlap_sentences,
    )


def recursive_chunks(text: str, max_characters: int) -> list[str]:
    """Split by structure first, falling back through sentences, words, and chars."""

    if len(text) <= max_characters:
        return [text.strip()] if text.strip() else []

    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    units: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_characters:
            units.append(paragraph)
            continue
        for sentence in split_sentences(paragraph):
            if len(sentence) <= max_characters:
                units.append(sentence)
            else:
                units.extend(fixed_chunks(sentence, max_characters, 0))

    return _pack_units(units, max_characters=max_characters, overlap_units=0)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * percentile / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def semantic_chunks(
    text: str,
    *,
    embed_sentences: EmbedSentences,
    max_characters: int,
    break_percentile: float,
) -> list[str]:
    """Break where adjacent sentence embeddings change topic substantially."""

    sentences = split_sentences(text)
    if len(sentences) < 2:
        return sentences

    vectors = list(embed_sentences(sentences))
    if len(vectors) != len(sentences):
        raise ValueError("embed_sentences must return one vector per sentence")

    distances = [
        1 - _cosine(vectors[index], vectors[index + 1])
        for index in range(len(vectors) - 1)
    ]
    threshold = _percentile(distances, break_percentile)
    chunks: list[str] = []
    current: list[str] = []

    for index, sentence in enumerate(sentences):
        proposed = " ".join((*current, sentence))
        size_break = bool(current) and len(proposed) > max_characters
        topic_break = (
            bool(current)
            and index > 0
            and distances[index - 1] > threshold
        )
        if size_break or topic_break:
            chunks.append(" ".join(current))
            current = []
        current.append(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_document(
    document: SourceDocument,
    config: ChunkingConfig,
    *,
    embed_sentences: EmbedSentences | None = None,
) -> list[Chunk]:
    config.validate()
    combined = f"{document.title}\n\n{document.text}".strip()

    if config.method == "whole":
        texts = [combined]
    elif config.method == "fixed":
        texts = fixed_chunks(
            combined,
            max_characters=config.max_characters,
            overlap_characters=config.overlap_characters,
        )
    elif config.method == "sentence":
        texts = sentence_chunks(
            combined,
            max_characters=config.max_characters,
            overlap_sentences=config.overlap_sentences,
        )
    elif config.method == "recursive":
        texts = recursive_chunks(combined, config.max_characters)
    elif config.method == "semantic":
        if embed_sentences is None:
            raise ValueError("semantic chunking requires embed_sentences")
        texts = semantic_chunks(
            combined,
            embed_sentences=embed_sentences,
            max_characters=config.max_characters,
            break_percentile=config.semantic_break_percentile,
        )
    else:
        raise ValueError(f"Unsupported chunking method: {config.method}")

    return _as_chunks(document, config.method, texts)
