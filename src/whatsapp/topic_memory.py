"""Local block topic detection and linked topic/span memory records."""

from __future__ import annotations

import json
import hashlib
import os
import re
import sys
import uuid
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.request import Request, urlopen

from .memory import MEMORY_VERSION, MemoryChunk


TOPIC_VERSION = "memory-v3"
TOPIC_COLLECTION = "mirror_whatsapp_memory_v3"
TOPIC_MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
BLOCK_SIZE = 30
SPAN_SIZE = 10
SPAN_OVERLAP = 2


def _json_object(value: str) -> dict[str, Any]:
    starts = [i for i in (value.find("{"), value.find("[")) if i >= 0]
    if not starts:
        raise ValueError("topic model did not return JSON")
    parsed, _ = json.JSONDecoder().raw_decode(value[min(starts):])
    if isinstance(parsed, list):
        parsed = {"starts": parsed}
    if not isinstance(parsed, dict):
        raise ValueError("topic model did not return a JSON object")
    return parsed


class LocalQwen:
    """One reusable local model for boundary classification and summaries."""

    def __init__(self, model_name: str = TOPIC_MODEL) -> None:
        try:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
        except ImportError as exc:
            raise RuntimeError("Install mlx-vlm on Apple Silicon for local topic processing") from exc
        self.model_name = model_name
        self.model, self.processor = load(model_name)
        self.config = load_config(model_name)

    def __call__(self, prompt: str, *, max_tokens: int = 700) -> str:
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        formatted = apply_chat_template(
            self.processor, self.config, prompt, num_images=0, enable_thinking=False
        )
        return generate(self.model, self.processor, formatted, max_tokens=max_tokens,
                        temperature=0, enable_thinking=False, verbose=False).text


class LazyLocalQwen:
    model_name = TOPIC_MODEL

    def __init__(self) -> None:
        self.loaded: LocalQwen | None = None

    def __call__(self, prompt: str, *, max_tokens: int = 700) -> str:
        if self.loaded is None:
            self.loaded = LocalQwen(self.model_name)
        return self.loaded(prompt, max_tokens=max_tokens)


class OpenAICompatibleTopicModel:
    """Topic model served through a user-configured chat-completions endpoint."""

    def __init__(self, base_url: str, model_name: str, *, api_key: str | None = None) -> None:
        if not base_url or not model_name:
            raise ValueError("topic API URL and model name are required")
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.model_name = model_name
        self.cache_identity = f"{self.url}\0{model_name}"
        self.api_key = api_key if api_key is not None else os.getenv("TOPIC_MODEL_API_KEY", "")

    def __call__(self, prompt: str, *, max_tokens: int = 700) -> str:
        body = json.dumps({"model": self.model_name, "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0, "max_tokens": max_tokens}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.url, data=body, headers=headers, method="POST")
        with urlopen(request, timeout=300) as response:
            value = json.load(response)
        content = value["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(item.get("text", "") for item in content if isinstance(item, dict))
        if not isinstance(content, str) or not content.strip():
            raise ValueError("topic model returned an empty chat completion")
        return content


class CachedTopicModel:
    """Persist model responses so repeated or interrupted imports can reuse them."""

    def __init__(self, model: Any, repository: Any, user_id: str,
                 contact_id: str, chat_id: str) -> None:
        self.model = model
        self.repository = repository
        self.scope = (user_id, contact_id, chat_id)

    def __call__(self, prompt: str, *, max_tokens: int = 700) -> str:
        model_name = getattr(self.model, "cache_identity",
                             getattr(self.model, "model_name", TOPIC_MODEL))
        key = hashlib.sha256(f"{model_name}\0{max_tokens}\0{prompt}".encode()).hexdigest()
        cached = self.repository.cached_topic_response(*self.scope, key)
        if cached is not None:
            return cached
        response = self.model(prompt, max_tokens=max_tokens)
        self.repository.cache_topic_response(*self.scope, key, response)
        return response


def boundary_prompt(rows: Sequence[dict[str, Any]], first: int, last: int,
                    topics: Sequence[dict[str, str]], active: str | None) -> str:
    before = "\n".join(f"{i} | {r['author']}: {r['normalized_text'][:300]}"
                       for i, r in enumerate(rows[max(0, first - 6):first], max(0, first - 6)))
    current = "\n".join(f"{i} | {r['author']}: {r['normalized_text'][:300]}"
                        for i, r in enumerate(rows[first:last], first))
    after = "\n".join(f"{i} | {r['author']}: {r['normalized_text'][:300]}"
                      for i, r in enumerate(rows[last:min(len(rows), last + 4)], last))
    return f"""Mark broad conversational topic changes in this Roman-script Hinglish chat.
Read in order. Return the index of the FIRST message that actually introduces a
different subject. A vague reply belongs to the subject it answers. Greetings
before a substantive subject stay with the opening topic. Clothing for a party
and party timing are one broad party topic; details of a school project remain
one school topic. Do not split because of elapsed time or a processing block edge.
Future messages are context only: never assign their change to an earlier index.
If a prior subject returns after another one, set kind RESUME and prior topic_id.
Do not invent a boundary just because this block begins.

Previous topics: {json.dumps(topics, ensure_ascii=False)}
Active topic_id: {json.dumps(active)}
Previous context (do not return these indices):\n{before or '(none)'}
Classify indices {first} through {last - 1}:\n{current}
Future context (do not return these indices):\n{after or '(none)'}

Return ONLY JSON: {{"starts": [{{"index": integer, "label": "short broad subject", "kind": "NEW or RESUME", "topic_id": "prior ID for RESUME only"}}]}}.
Include every actual broad subject change in the classified indices. No prose."""


def detect_boundaries(rows: Sequence[dict[str, Any]], model: Any,
                      *, block_size: int = BLOCK_SIZE) -> list[dict[str, Any]]:
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    topics: list[dict[str, str]] = []
    starts: list[dict[str, Any]] = []
    active: str | None = None
    for first in range(0, len(rows), block_size):
        last = min(len(rows), first + block_size)
        if first == 0 or first % (block_size * 10) == 0 or last == len(rows):
            print(f"topic boundaries: {first}-{last - 1} / {len(rows)} messages",
                  file=sys.stderr, flush=True)
        prompt = boundary_prompt(rows, first, last, topics[-24:], active)
        try:
            response = _json_object(model(prompt))
        except ValueError as exc:
            # A small local model can occasionally return unusable text. The
            # response is cached, so failing the whole pass here would repeat
            # forever; skip this block's boundaries and keep the rest.
            print(f"topic block {first}-{last - 1}: ignoring response ({exc})",
                  file=sys.stderr)
            continue
        candidates = response.get("starts")
        if not isinstance(candidates, list):
            print(f"topic block {first}-{last - 1}: ignoring response (missing starts list)",
                  file=sys.stderr)
            continue
        previous_index = first - 1
        for item in candidates:
            if not isinstance(item, dict) or type(item.get("index")) is not int:
                print(f"topic block {first}-{last - 1}: ignoring invalid entry {item!r}",
                      file=sys.stderr)
                continue
            index = item["index"]
            if not previous_index < index < last:
                print(f"topic block {first}-{last - 1}: ignoring out-of-block index {index}",
                      file=sys.stderr)
                continue
            previous_index = index
            label = str(item.get("label") or "conversation").strip()[:100]
            kind = str(item.get("kind") or "NEW").upper()
            current_label = next((t["label"] for t in topics if t["topic_id"] == active), None)
            if current_label and current_label.casefold() == label.casefold():
                continue
            # The small model sometimes writes NEW for an exact prior label.
            # Reuse that topic rather than creating an identical sibling.
            if kind == "NEW":
                matching = next((t for t in topics if t["label"].casefold() == label.casefold()
                                 and t["topic_id"] != active), None)
                if matching:
                    kind = "RESUME"
                    item = {**item, "topic_id": matching["topic_id"]}
            if kind == "RESUME":
                topic_id = str(item.get("topic_id") or "")
                if topic_id not in {t["topic_id"] for t in topics} or topic_id == active:
                    matching = next((t for t in topics if t["label"].casefold() == label.casefold()
                                     and t["topic_id"] != active), None)
                    if matching:
                        topic_id = matching["topic_id"]
                    else:
                        kind = "NEW"
            if kind == "NEW":
                topic_id = f"T{len(topics) + 1}"
                topics.append({"topic_id": topic_id, "label": label})
            elif kind != "RESUME":
                print(f"topic block {first}-{last - 1}: unknown kind {kind!r}; treating as NEW",
                      file=sys.stderr)
                kind = "NEW"
                topic_id = f"T{len(topics) + 1}"
                topics.append({"topic_id": topic_id, "label": label})
            starts.append({"index": index, "topic_id": topic_id,
                           "label": label, "kind": kind})
            active = topic_id
    return starts


@dataclass(frozen=True)
class TopicRecord:
    id: uuid.UUID
    user_id: str
    contact_id: str
    chat_id: str
    label: str
    summary: str
    message_ids: tuple[uuid.UUID, ...]

    def payload(self) -> dict[str, Any]:
        return {"schema_version": TOPIC_VERSION, "record_type": "topic",
                "topic_id": str(self.id), "user_id": self.user_id,
                "contact_id": self.contact_id, "chat_id": self.chat_id,
                "label": self.label, "summary": self.summary,
                "message_ids": [str(x) for x in self.message_ids]}


def summary_prompt(label: str, rows: Sequence[dict[str, Any]]) -> str:
    # Spans preserve all details. The topic vector only needs a broad route.
    selected = list(rows)
    if len(selected) > 80:
        selected = selected[:40] + selected[-40:]
    transcript = "\n".join(f"{r['author']}: {r['normalized_text'][:300]}" for r in selected)
    return ("Summarize this one Hinglish conversation topic for semantic retrieval. "
            "Use 1-3 factual sentences. Include concrete people, objects, events, "
            "and distinctive terms actually mentioned. Do not infer missing facts. "
            f"Topic label: {label}\nMessages:\n{transcript[:12000]}\nSummary only:")


def build_topics_and_spans(rows: Sequence[dict[str, Any]], boundaries: Sequence[dict[str, Any]],
                           model: Any, *, user_id: str, contact_id: str,
                           span_size: int = SPAN_SIZE, overlap: int = SPAN_OVERLAP,
                           max_span_characters: int = 2400,
                           summarize: bool = True) -> tuple[list[TopicRecord], list[MemoryChunk]]:
    if span_size < 1 or not 0 <= overlap < span_size or max_span_characters < 1:
        raise ValueError("invalid span size or overlap")
    if not rows:
        return [], []
    if any(r["user_id"] != user_id or r["contact_id"] != contact_id or
           r["chat_id"] != rows[0]["chat_id"] for r in rows):
        raise ValueError("mixed chat scope")
    indices = [b["index"] for b in boundaries]
    cut = [0, *[i for i in indices if i > 0], len(rows)]
    if indices != sorted(set(indices)) or any(i < 0 or i >= len(rows) for i in indices):
        raise ValueError("invalid topic boundary sequence")
    segments: list[tuple[str, str, Sequence[dict[str, Any]]]] = []
    labels: dict[str, str] = {}
    first_id = "T0"
    if boundaries and boundaries[0]["index"] == 0:
        first_id = boundaries[0]["topic_id"]
    for a, b in zip(cut, cut[1:]):
        matching = next((x for x in boundaries if x["index"] == a), None)
        logical_id = matching["topic_id"] if matching else first_id
        label = matching["label"] if matching else "opening conversation"
        labels[logical_id] = label
        segments.append((logical_id, label, rows[a:b]))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for logical_id, _, group in segments:
        grouped.setdefault(logical_id, []).extend(group)
    topic_ids = {key: uuid.uuid5(uuid.NAMESPACE_URL, "mirror-topic-v3:" +
                 "\0".join((user_id, contact_id, rows[0]["chat_id"], str(group[0]["id"]))))
                 for key, group in grouped.items()}
    topics: list[TopicRecord] = []
    for topic_number, (logical_id, group) in enumerate(grouped.items(), 1):
        if topic_number == 1 or topic_number % 20 == 0 or topic_number == len(grouped):
            print(f"topic summaries: {topic_number}/{len(grouped)}", file=sys.stderr, flush=True)
        label = labels[logical_id]
        summary = model(summary_prompt(label, group), max_tokens=180).strip() if summarize else label
        topics.append(TopicRecord(topic_ids[logical_id], user_id, contact_id, rows[0]["chat_id"],
                                  label, summary or label, tuple(r["id"] for r in group)))
    spans: list[MemoryChunk] = []
    for logical_id, _, group in segments:
        start = 0
        while start < len(group):
            selected = []
            size = 0
            for row in group[start:start + span_size]:
                rendered = f"[{row['sent_at'].isoformat()}] {row['author']}: {row['normalized_text']}"
                if selected and size + len(rendered) > max_span_characters:
                    break
                selected.append(row)
                size += len(rendered) + 1
            ids = tuple(r["id"] for r in selected)
            span_id = uuid.uuid5(uuid.NAMESPACE_URL, "mirror-span-v3:" + "\0".join(map(str, ids)))
            text = "\n".join(f"[{r['sent_at'].isoformat()}] {r['author']}: {r['normalized_text']}" for r in selected)
            spans.append(MemoryChunk(span_id, user_id, contact_id, rows[0]["chat_id"],
                                     selected[0]["sent_at"], selected[-1]["sent_at"],
                                     start, start + len(selected) - 1, ids,
                                     tuple(dict.fromkeys(r["author"] for r in selected if r["author"])),
                                     text, TOPIC_VERSION, topic_ids[logical_id]))
            if start + len(selected) >= len(group):
                break
            start += max(1, len(selected) - overlap)
    return topics, spans
