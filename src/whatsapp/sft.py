"""Build SFT pairs with real target replies and optional synthetic queries."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from .models import WhatsAppMessage
from .onboard_cli import _approved_messages
from .topic_memory import _json_object


SFT_VERSION = 1
DEFAULT_CONTEXT_MESSAGES = 6


def _sha256(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _source(workspace: Path) -> tuple[dict[str, Any], list[WhatsAppMessage], str]:
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    messages, review_hash = _approved_messages(workspace, manifest)
    report_path = workspace / "index_report.json"
    if not report_path.is_file():
        raise ValueError("index the approved chat before building SFT examples")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("review_hash") != review_hash:
        raise ValueError("approved chat review differs from the indexed review")
    return manifest, messages, review_hash


def candidate_turns(messages: Sequence[WhatsAppMessage], target_author: str,
                    *, context_messages: int = DEFAULT_CONTEXT_MESSAGES
                    ) -> list[dict[str, Any]]:
    """Group target message bursts and retain their actual preceding chat."""
    if context_messages < 1:
        raise ValueError("context_messages must be positive")
    turns: list[dict[str, Any]] = []
    index = 0
    while index < len(messages):
        if messages[index].author != target_author:
            index += 1
            continue
        start = index
        while (index < len(messages) and messages[index].author == target_author and
               (index == start or messages[index].sequence == messages[index - 1].sequence + 1)):
            index += 1
        context_start = start
        while (context_start > 0 and start - context_start < context_messages and
               messages[context_start].sequence == messages[context_start - 1].sequence + 1):
            context_start -= 1
        context = list(messages[context_start:start])
        if not context or context[-1].author == target_author:
            continue
        replies = list(messages[start:index])
        response = "\n".join(message.text for message in replies)
        if not response.strip():
            continue
        turns.append({
            "id": _sha256([message.message_id for message in replies])[:20],
            "target_message_ids": [message.message_id for message in replies],
            "context_message_ids": [message.message_id for message in context],
            "context": [{"message_id": message.message_id,
                         "author": message.author, "text": message.text}
                        for message in context],
            "response": response,
        })
    return turns


def _splits(count: int) -> list[str]:
    if count < 3:
        return ["train"] * count
    test_count = max(1, math.ceil(count * 0.1))
    validation_count = max(1, math.ceil(count * 0.1))
    train_count = count - test_count - validation_count
    return (["train"] * train_count + ["validation"] * validation_count +
            ["test"] * test_count)


def query_prompt(turn: dict[str, Any], target_author: str) -> str:
    context = "\n".join(f"{item['author']}: {item['text']}" for item in turn["context"])
    return (
        "Write one plausible WhatsApp message from the other speaker that could "
        f"naturally elicit the exact reply by {target_author} below. "
        "Use the conversation's language, including Hinglish when present. "
        "The query must not contain facts that only appear in the reply. "
        "Do not copy the reply, invent people or events, or add instructions to the assistant. "
        "If no single query fits, return fit=poor. Return only JSON with "
        '{"query":"...","fit":"good or poor"}.\n'
        f"Prior chat:\n{context}\nReal reply:\n{turn['response']}"
    )


def _training_row(target_author: str, context: Sequence[dict[str, str]],
                  response: str) -> dict[str, Any]:
    conversation = "\n".join(f"{item['author']}: {item['text']}" for item in context)
    return {"messages": [
        {"role": "system", "content": f"Reply as {target_author} in this WhatsApp conversation. Write only the next message."},
        {"role": "user", "content": f"Conversation so far:\n{conversation}"},
        {"role": "assistant", "content": response},
    ]}


def build_sft(
    workspace: Path, *, output: Path | None = None,
    model: Callable[..., str] | None = None, query_limit: int | None = None,
    context_messages: int = DEFAULT_CONTEXT_MESSAGES,
) -> dict[str, Any]:
    """Write usable SFT files directly; users can remove unwanted JSONL rows."""
    if query_limit is not None and query_limit < 1:
        raise ValueError("query_limit must be positive")
    if query_limit is not None and model is None:
        raise ValueError("query_limit requires a query model")
    source, messages, review_hash = _source(workspace)
    output = output or workspace / "sft"
    if output.exists():
        raise FileExistsError(f"SFT output exists: {output}; choose a new --output")
    turns = candidate_turns(messages, source["target_author"],
                            context_messages=context_messages)
    turns = [turn for turn in turns if 8 <= len(turn["response"].strip()) <= 2000]
    if not turns:
        raise ValueError("no usable target-author reply with preceding chat was found")
    splits = _splits(len(turns))
    datasets: dict[str, list[dict[str, Any]]] = {key: [] for key in
                                                ("train", "validation", "test")}
    provenance: list[dict[str, Any]] = []
    attempts = 0
    synthetic = 0
    for turn, split in zip(turns, splits):
        context = turn["context"]
        variant = "real"
        if split == "train" and model is not None and (
                query_limit is None or attempts < query_limit):
            attempts += 1
            if attempts == 1 or attempts % 10 == 0:
                print(f"SFT queries: attempted {attempts}, usable {synthetic}",
                      file=sys.stderr, flush=True)
            try:
                generated = _json_object(model(query_prompt(turn, source["target_author"]),
                                               max_tokens=180))
                query = generated.get("query")
                if (str(generated.get("fit", "")).casefold() == "good" and
                        isinstance(query, str) and 5 <= len(query.strip()) <= 500 and
                        query.strip().casefold() != turn["response"].strip().casefold()):
                    context = [{"author": turn["context"][-1]["author"],
                                "text": query.strip()}]
                    variant = "synthetic"
                    synthetic += 1
            except ValueError:
                pass
        datasets[split].append(_training_row(source["target_author"], context,
                                             turn["response"]))
        provenance.append({
            "example_id": turn["id"], "split": split, "variant": variant,
            "target_message_ids": turn["target_message_ids"],
            "context_message_ids": turn["context_message_ids"],
        })
    output.mkdir(parents=True)
    for split, rows in datasets.items():
        _write_jsonl(output / f"{split}.jsonl", rows)
    _write_jsonl(output / "provenance.jsonl", provenance)
    report = {
        "schema_version": SFT_VERSION,
        "output": str(output.resolve()),
        "source_review_hash": review_hash,
        "examples": {split: len(rows) for split, rows in datasets.items()},
        "synthetic_queries": synthetic,
        "query_model_calls": attempts,
        "heldout_target_message_ids": sorted({
            mid for item in provenance if item["split"] != "train"
            for mid in item["target_message_ids"]
        }),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
