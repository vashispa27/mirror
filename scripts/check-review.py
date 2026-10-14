#!/usr/bin/env python3
"""Validate the review conversations and render readable previews.

Reads every ``data/review/*.jsonl`` file, checks the schema and the rules in
``data/review/README.md``, and writes a matching ``.md`` preview next to each
file so the conversations can be read without JSON.

Usage::

    .venv/bin/python scripts/check-review.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
REVIEW_DIR = REPO_ROOT / "data" / "review"

SPEAKERS = ("Aditi", "Rohan")
ROLE_FOR = {"Aditi": "user", "Rohan": "assistant"}
QUERY_TYPES = (
    "exact_phrase", "paraphrase", "situation", "semantic",
    "multihop", "temporal", "attribution", "style", "unanswerable",
)
MIN_MESSAGES = 6
MAX_MESSAGES = 40
MAX_MESSAGE_CHARS = 400
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Common English and Hinglish function words; not "content" words.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    he her him his how i if in is it its me my no nor not of on or our she should
    so than that the their them then there these they this those to too us was we
    were what when where which who whom why will with would you your
    aur bhi ek haan hai hain hum humara jab jaisa jaise jo ka kaafi kaise kaisa
    kaisi kal kar karna karte karta karti ke ki kiya kiye ko kyu kyun lagta lagti
    laga liye matlab me mein mera meri mere mujhe na nahi nahin ne par phir raha
    rahi rahe saath se sirf tab tha thi thoda to tum tumhara tumhe unka unke unki
    us wala wali wale wapas wo woh ye yeh
    """.split()
)


def _non_latin(text: str) -> list[str]:
    """Letters that are not Latin script, e.g. Devanagari."""

    return sorted({
        char for char in text
        if char.isalpha() and not unicodedata.name(char, "").startswith("LATIN")
    })


def _tokens(text: str) -> set[str]:
    cleaned = re.sub(r"[^0-9A-Za-z\u00C0-\u024F']+", " ", text)
    return {
        token for token in cleaned.casefold().split()
        if len(token) > 1 and token not in STOPWORDS
    }


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _find_quote(messages: list[dict[str, Any]], quote: str) -> tuple[str, int]:
    """Return (status, count); status is exact, whitespace, ambiguous or missing."""

    exact = [m for m in messages if quote in m["content"]]
    if len(exact) == 1:
        return "exact", 1
    if len(exact) > 1:
        return "ambiguous", len(exact)
    loose = [m for m in messages if _collapse(quote) in _collapse(m["content"])]
    if len(loose) == 1:
        return "whitespace", 1
    if len(loose) > 1:
        return "ambiguous", len(loose)
    return "missing", 0


def _check_conversation(value: Any, where: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(value, dict):
        return [f"{where}: line is not a JSON object"], []

    for key in ("conversation_id", "scenario", "topic", "date", "messages",
                "questions"):
        if key not in value:
            errors.append(f"{where}: missing {key!r}")

    date = value.get("date")
    if isinstance(date, str) and not DATE_RE.match(date):
        errors.append(f"{where}: date {date!r} is not YYYY-MM-DD")

    messages = value.get("messages")
    if not isinstance(messages, list):
        return errors + [f"{where}: messages is not a list"], warnings
    if not MIN_MESSAGES <= len(messages) <= MAX_MESSAGES:
        errors.append(
            f"{where}: {len(messages)} messages, expected "
            f"{MIN_MESSAGES}-{MAX_MESSAGES}"
        )

    for index, message in enumerate(messages, 1):
        if not isinstance(message, dict):
            errors.append(f"{where}: message {index} is not an object")
            continue
        speaker = message.get("speaker")
        role = message.get("role")
        content = message.get("content")
        if speaker not in SPEAKERS:
            errors.append(f"{where}: message {index} speaker {speaker!r}")
        if role not in ROLE_FOR.values():
            errors.append(f"{where}: message {index} role {role!r}")
        if speaker in ROLE_FOR and role != ROLE_FOR[speaker]:
            errors.append(
                f"{where}: message {index} role {role!r} does not match "
                f"speaker {speaker!r}"
            )
        if not isinstance(content, str) or not content.strip():
            errors.append(f"{where}: message {index} has no content")
            continue
        if "\n" in content:
            errors.append(f"{where}: message {index} contains a newline")
        if len(content) > MAX_MESSAGE_CHARS:
            errors.append(
                f"{where}: message {index} is {len(content)} chars "
                f"(limit {MAX_MESSAGE_CHARS})"
            )
        bad = _non_latin(content)
        if bad:
            errors.append(
                f"{where}: message {index} has non-Latin characters {bad}"
            )

    questions = value.get("questions")
    if not isinstance(questions, list):
        return errors + [f"{where}: questions is not a list"], warnings
    if not questions:
        errors.append(f"{where}: no questions")

    for index, question in enumerate(questions, 1):
        if not isinstance(question, dict):
            errors.append(f"{where}: question {index} is not an object")
            continue
        query = question.get("query")
        query_type = question.get("query_type")
        quote = question.get("answer_quote")
        if not isinstance(query, str) or not query.strip():
            errors.append(f"{where}: question {index} has no query")
            continue
        if query_type not in QUERY_TYPES:
            errors.append(f"{where}: question {index} type {query_type!r}")
            continue
        if _non_latin(query):
            errors.append(
                f"{where}: question {index} query has non-Latin characters"
            )
        if query_type == "unanswerable":
            if quote:
                errors.append(
                    f"{where}: question {index} is unanswerable but has a quote"
                )
            continue
        if not isinstance(quote, str) or not quote.strip():
            errors.append(f"{where}: question {index} has no answer_quote")
            continue
        status, count = _find_quote(messages, quote)
        if status == "missing":
            errors.append(
                f"{where}: question {index} quote not found in any message"
            )
            continue
        if status == "ambiguous":
            warnings.append(
                f"{where}: question {index} quote matches {count} messages"
            )
            continue
        if status == "whitespace":
            warnings.append(
                f"{where}: question {index} quote differs only in whitespace"
            )
        answered = next(m for m in messages if _collapse(quote) in _collapse(m["content"]))
        overlap = _tokens(query) & _tokens(answered["content"])
        if query_type == "semantic" and overlap:
            warnings.append(
                f"{where}: question {index} labelled semantic shares "
                f"{sorted(overlap)} with its answer"
            )
        if query_type == "exact_phrase" and not overlap:
            warnings.append(
                f"{where}: question {index} labelled exact_phrase shares no "
                f"content words with its answer"
            )

    return errors, warnings


def _render_markdown(path: Path, conversations: list[dict[str, Any]]) -> str:
    messages = sum(len(c.get("messages", [])) for c in conversations)
    questions = sum(len(c.get("questions", [])) for c in conversations)
    lines = [
        f"# {path.stem}",
        "",
        f"{len(conversations)} conversations · {messages} messages · "
        f"{questions} questions",
        "",
    ]
    for conversation in conversations:
        lines.append(
            f"## {conversation.get('conversation_id')} — "
            f"{conversation.get('topic')} ({conversation.get('date')})"
        )
        lines.append("")
        for index, message in enumerate(conversation.get("messages", []), 1):
            lines.append(
                f"`{index:>2}` **{message.get('speaker')}:** "
                f"{message.get('content')}"
            )
        lines.append("")
        lines.append("### Questions")
        lines.append("")
        for question in conversation.get("questions", []):
            lines.append(f"- `{question.get('query_type')}` {question.get('query')}")
            if question.get("answer_quote"):
                lines.append(f"  - evidence: `{question['answer_quote']}`")
            else:
                lines.append("  - evidence: (none — unanswerable)")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    files = sorted(REVIEW_DIR.glob("*.jsonl"))
    if not files:
        print(f"no review files in {REVIEW_DIR}")
        return 1

    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: dict[str, str] = {}
    seen_queries: dict[str, str] = {}
    type_counts: Counter[str] = Counter()
    total_conversations = total_messages = total_questions = 0

    for path in files:
        conversations: list[dict[str, Any]] = []
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            where = f"{path.name}:{line_no}"
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{where}: invalid JSON ({exc})")
                continue
            found_errors, found_warnings = _check_conversation(value, where)
            errors.extend(found_errors)
            warnings.extend(found_warnings)
            if not isinstance(value, dict):
                continue
            conversations.append(value)
            conversation_id = value.get("conversation_id")
            if isinstance(conversation_id, str):
                if conversation_id in seen_ids:
                    errors.append(
                        f"{where}: conversation_id {conversation_id!r} already "
                        f"used in {seen_ids[conversation_id]}"
                    )
                else:
                    seen_ids[conversation_id] = where
            for question in value.get("questions", []) or []:
                if not isinstance(question, dict):
                    continue
                type_counts[str(question.get("query_type"))] += 1
                query = question.get("query")
                if isinstance(query, str) and query.strip():
                    key = _collapse(query).casefold()
                    if key in seen_queries:
                        warnings.append(
                            f"{where}: duplicate query already used in "
                            f"{seen_queries[key]}"
                        )
                    else:
                        seen_queries[key] = where

        total_conversations += len(conversations)
        total_messages += sum(len(c.get("messages", [])) for c in conversations)
        total_questions += sum(len(c.get("questions", [])) for c in conversations)
        preview = path.with_suffix(".md")
        preview.write_text(_render_markdown(path, conversations), encoding="utf-8")
        print(f"{path.name}: {len(conversations)} conversations -> {preview.name}")

    print()
    print(f"conversations: {total_conversations}")
    print(f"messages:      {total_messages}")
    print(f"questions:     {total_questions}")
    print(f"types:         {dict(sorted(type_counts.items()))}")
    print(f"errors:        {len(errors)}")
    print(f"warnings:      {len(warnings)}")
    for item in errors[:40]:
        print(f"  ERROR   {item}")
    for item in warnings[:40]:
        print(f"  warning {item}")
    if len(errors) > 40 or len(warnings) > 40:
        print("  ... more not shown")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
