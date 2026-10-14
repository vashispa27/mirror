#!/usr/bin/env python3
"""Merge the reviewed conversations into the dataset and build the eval set.

Reads the base export plus every ``data/review/*.jsonl`` file and writes:

* ``data/export.txt``         – base conversations + reviewed conversations
* ``data/conversations.jsonl`` – base conversations + reviewed conversations
* ``data/evals.jsonl``        – the review questions, resolved to message ids
* ``data/evals-unanswerable.jsonl`` – review questions whose answer is not in
  the conversation

The base conversations keep their original message ids: reviewed conversations
are appended after them, so the parser's sequence numbers for the base do not
move. Run ``scripts/build-dataset.py build`` first, then this script. Re-running
this script is safe — it always starts from the base export and the review files.

Usage::

    .venv/bin/python scripts/merge-review.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from whatsapp.parser import parse_whatsapp_chat  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
REVIEW_DIR = DATA_DIR / "review"
BASE_EXPORT = DATA_DIR / "export-base.txt"
MERGED_EXPORT = DATA_DIR / "export.txt"
CONVERSATION_PATH = DATA_DIR / "conversations.jsonl"
EVAL_PATH = DATA_DIR / "evals.jsonl"
UNANSWERABLE_PATH = DATA_DIR / "evals-unanswerable.jsonl"

GAP_MINUTES = 135


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path.name}:{line_no} invalid JSON: {exc}") from exc
    return rows


def _review_files() -> list[Path]:
    return sorted(REVIEW_DIR.glob("*.jsonl"))


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _render_review(
    conversations: list[dict[str, Any]], start_after: datetime | None
) -> tuple[list[str], list[list[tuple[str, datetime, str]]]]:
    """Render reviewed conversations as export lines, in date order."""

    ordered = sorted(
        conversations,
        key=lambda c: (str(c.get("date", "")), str(c.get("conversation_id", ""))),
    )
    lines: list[str] = []
    scheduled: list[list[tuple[str, datetime, str]]] = []
    cursor = start_after
    for conversation in ordered:
        date_text = str(conversation.get("date", ""))
        try:
            day = datetime.fromisoformat(date_text).replace(hour=9, minute=0)
        except ValueError:
            day = (cursor + timedelta(days=1)) if cursor else datetime(2026, 2, 1, 9, 0)
        if cursor is not None and day <= cursor:
            day = cursor + timedelta(minutes=GAP_MINUTES)
        stamped: list[tuple[str, datetime, str]] = []
        for message in conversation.get("messages", []):
            speaker = str(message.get("speaker", ""))
            content = _collapse(str(message.get("content", "")))
            lines.append(
                f"{day.strftime('%d/%m/%y, %H:%M')} - {speaker}: {content}"
            )
            stamped.append((speaker, day, content))
            day += timedelta(minutes=1 + (len(content) % 3))
        scheduled.append(stamped)
        cursor = day
    return lines, scheduled


def _non_latin(text: str) -> bool:
    return any(
        char.isalpha() and not unicodedata.name(char, "").startswith("LATIN")
        for char in text
    )


def merge(args: argparse.Namespace) -> int:
    if not BASE_EXPORT.exists():
        raise SystemExit(
            f"missing {BASE_EXPORT}; run `scripts/build-dataset.py build` first"
        )
    if not _review_files():
        raise SystemExit(f"no review files in {REVIEW_DIR}")

    base_text = BASE_EXPORT.read_text(encoding="utf-8")
    base_messages = parse_whatsapp_chat(
        base_text,
        chat_id=args.chat_id,
        day_first=True,
        timezone_name=args.timezone,
    )
    base_last = base_messages[-1].timestamp if base_messages else None
    if base_last is not None:
        # The parser returns timezone-aware stamps; the renderer works in naive
        # local time, matching how build-dataset.py lays out the base export.
        base_last = base_last.replace(tzinfo=None)
    print(f"base: {len(base_messages)} messages")

    base_conversations = _read_jsonl(CONVERSATION_PATH)
    base_conversations = [c for c in base_conversations if "scenario" not in c]
    print(f"base conversations kept: {len(base_conversations)}")

    review_conversations: list[dict[str, Any]] = []
    for path in _review_files():
        for row in _read_jsonl(path):
            row.setdefault("scenario", path.stem)
            row["source"] = f"review/{path.name}"
            review_conversations.append(row)
    print(f"review conversations: {len(review_conversations)}")

    lines, scheduled = _render_review(review_conversations, base_last)
    merged_text = base_text
    if not merged_text.endswith("\n"):
        merged_text += "\n"
    merged_text += "\n".join(lines) + "\n"

    messages = parse_whatsapp_chat(
        merged_text,
        chat_id=args.chat_id,
        day_first=True,
        timezone_name=args.timezone,
    )
    if len(messages) < len(base_messages):
        raise SystemExit("merged parse lost base messages; aborting")
    review_messages = messages[len(base_messages):]

    # The base kept its ids; verify before writing anything.
    for original, reparsed in zip(base_messages, messages):
        if original.message_id != reparsed.message_id:
            raise SystemExit(
                "base message ids changed after appending review lines; aborting"
            )

    cursor = 0
    review_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    unanswerable_rows: list[dict[str, Any]] = []
    unresolved = 0

    ordered = sorted(
        review_conversations,
        key=lambda c: (str(c.get("date", "")), str(c.get("conversation_id", ""))),
    )
    for conversation, stamped in zip(ordered, scheduled):
        window = review_messages[cursor:cursor + len(stamped)]
        cursor += len(stamped)
        by_text: dict[str, str] = {}
        for message in window:
            by_text.setdefault(_collapse(message.text), message.message_id)
        review_rows.append(
            {
                "conversation_id": conversation.get("conversation_id"),
                "scenario": conversation.get("scenario"),
                "topic": conversation.get("topic"),
                "date": conversation.get("date"),
                "source": conversation.get("source"),
                "language": "hinglish",
                "messages": [
                    {
                        "role": "user" if message.author == args.owner_author else "assistant",
                        "speaker": message.author,
                        "content": message.text,
                        "message_id": message.message_id,
                    }
                    for message in window
                ],
            }
        )
        for question in conversation.get("questions", []) or []:
            query = _collapse(str(question.get("query", "")))
            query_type = question.get("query_type")
            quote = _collapse(str(question.get("answer_quote") or ""))
            row = {
                "query": query,
                "user_id": args.user_id,
                "persona_id": args.persona_id,
                "query_type": query_type,
                "conversation_id": conversation.get("conversation_id"),
                "scenario": conversation.get("scenario"),
            }
            if query_type == "unanswerable" or not quote:
                row["expected_target_message_ids"] = []
                unanswerable_rows.append(row)
                continue
            matches = [m for m in window if quote in _collapse(m.text)]
            if len(matches) != 1:
                unresolved += 1
                continue
            row["expected_target_message_ids"] = [matches[0].message_id]
            eval_rows.append(row)

    for row in review_rows:
        for message in row["messages"]:
            if _non_latin(str(message["content"])):
                raise SystemExit(
                    f"{row['conversation_id']}: non-Latin text in merged output"
                )

    MERGED_EXPORT.write_text(merged_text, encoding="utf-8")
    _write_jsonl(CONVERSATION_PATH, base_conversations + review_rows)
    _write_jsonl(EVAL_PATH, eval_rows)
    _write_jsonl(UNANSWERABLE_PATH, unanswerable_rows)

    print()
    print(f"merged messages:      {len(messages)}")
    print(f"merged conversations: {len(base_conversations) + len(review_rows)}"
          f"  ({len(base_conversations)} base + {len(review_rows)} review)")
    print(f"eval questions:       {len(eval_rows)} answerable"
          f" + {len(unanswerable_rows)} unanswerable")
    if unresolved:
        print(f"unresolved quotes:    {unresolved} (skipped)")
    print()
    print(f"wrote {MERGED_EXPORT}")
    print(f"wrote {CONVERSATION_PATH}")
    print(f"wrote {EVAL_PATH}")
    print(f"wrote {UNANSWERABLE_PATH}")
    return 0


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timezone", default="Asia/Kolkata")
    parser.add_argument("--chat-id", default="hinglish-chat")
    parser.add_argument("--user-id", default="local-user")
    parser.add_argument("--persona-id", default="hinglish-twin")
    parser.add_argument("--owner-author", default="Aditi",
                        help="the user's own name in the export")
    return parser


def main(argv: list[str] | None = None) -> int:
    return merge(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
