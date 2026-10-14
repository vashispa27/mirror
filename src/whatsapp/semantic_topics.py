"""Local semantic change-point detection for Hindi/English chat sessions.

Both models run locally through FastEmbed. A compact multilingual encoder
proposes valleys in adjacent context-window similarity; a stronger encoder
checks those candidates. This avoids a paid per-message LLM classifier and
does not treat a language switch by itself as a topic change.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np


PROPOSAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VERIFICATION_MODEL = "Qwen/Qwen3-Embedding-0.6B-Q"
WINDOW_MESSAGES = 3
MIN_SEGMENT_MESSAGES = 3
PROPOSAL_DEPTH = 0.05
VERIFICATION_MAX_SIMILARITY = 0.44


@lru_cache(maxsize=1)
def _models() -> tuple[Any, Any]:
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=PROPOSAL_MODEL), TextEmbedding(model_name=VERIFICATION_MODEL)


def _normalized_embeddings(model: Any, texts: Sequence[str], batch_size: int) -> np.ndarray:
    vectors = np.asarray(list(model.embed(texts, batch_size=batch_size)), dtype=np.float32)
    if len(vectors) != len(texts):
        raise RuntimeError("topic embedding model returned the wrong vector count")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def semantic_topic_boundaries(
    rows: Sequence[dict[str, Any]], *,
    proposal_model: Any | None = None,
    verification_model: Any | None = None,
    window_messages: int = WINDOW_MESSAGES,
    min_segment_messages: int = MIN_SEGMENT_MESSAGES,
    proposal_depth: float = PROPOSAL_DEPTH,
    verification_max_similarity: float = VERIFICATION_MAX_SIMILARITY,
) -> set[int]:
    """Return message offsets where adjacent context windows change topic.

    Rows must already be one time-bounded, contact-scoped chat session.
    Short sessions are kept whole because a semantic comparison would have
    too little context to distinguish a reply from a new subject.
    """

    if window_messages < 2 or min_segment_messages < 1:
        raise ValueError("invalid semantic window or segment length")
    if not 0 <= proposal_depth <= 1 or not -1 <= verification_max_similarity <= 1:
        raise ValueError("invalid semantic thresholds")
    count = len(rows)
    if count < window_messages * 2:
        return set()
    if proposal_model is None or verification_model is None:
        default_proposal, default_verification = _models()
        proposal_model = proposal_model or default_proposal
        verification_model = verification_model or default_verification

    windows = [
        " ".join(row["normalized_text"][:350] for row in rows[start:start + window_messages])
        for start in range(count - window_messages + 1)
    ]
    proposed_vectors = _normalized_embeddings(proposal_model, windows, batch_size=64)
    positions = list(range(window_messages, count - window_messages + 1))
    scores = [float(np.dot(proposed_vectors[i - window_messages], proposed_vectors[i])) for i in positions]
    candidates: list[int] = []
    for index, position in enumerate(positions):
        if position < min_segment_messages or count - position < min_segment_messages:
            continue
        score = scores[index]
        neighbors = [scores[j] for j in (index - 1, index + 1) if 0 <= j < len(scores)]
        if not neighbors or any(score > neighbor for neighbor in neighbors):
            continue
        depth = sum(neighbors) / len(neighbors) - score
        if depth >= proposal_depth:
            candidates.append(position)
    if not candidates:
        return set()

    needed_starts = sorted({start for i in candidates for start in (i - window_messages, i)})
    verified_vectors = _normalized_embeddings(
        verification_model, [windows[start] for start in needed_starts], batch_size=32
    )
    verified = dict(zip(needed_starts, verified_vectors))
    ranked = sorted(
        ((float(np.dot(verified[i - window_messages], verified[i])), i) for i in candidates)
    )
    selected: set[int] = set()
    for similarity, position in ranked:
        if similarity > verification_max_similarity:
            break
        if all(abs(position - other) >= min_segment_messages for other in selected):
            selected.add(position)
    return selected
