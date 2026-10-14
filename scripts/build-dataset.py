#!/usr/bin/env python3
"""Build the base conversations from ``manishiitg/aditi-syn-v1``.

Source: a Hugging Face parquet of 25,000 synthetic Hinglish/Hindi
conversations (question -> answer).  This script keeps a deterministic subset
and writes:

* ``data/export.txt``       – the conversations as a chat transcript
* ``data/conversations.jsonl`` – the dataset, one conversation per line

Everything is deterministic for a given ``--seed`` and ``--limit``.

Usage::

    .venv/bin/python scripts/build-dataset.py fetch
    .venv/bin/python scripts/build-dataset.py build --limit 600 --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from whatsapp.parser import parse_whatsapp_chat  # noqa: E402

PARQUET_URL = (
    "https://huggingface.co/datasets/manishiitg/aditi-syn-v1/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
DATA_DIR = REPO_ROOT / "data"
SOURCE_PARQUET = DATA_DIR / "source" / "train.parquet"
EXPORT_PATH = DATA_DIR / "export.txt"
EXPORT_BASE_PATH = DATA_DIR / "export-base.txt"
CONVERSATION_PATH = DATA_DIR / "conversations.jsonl"

MIN_QUESTION_CHARS = 12
MIN_QUESTION_WORDS = 4
MIN_ANSWER_CHARS = 60
MAX_ANSWER_CHARS = 1_400
MIN_GAP_MINUTES = 135  # keeps each rendered conversation its own chunk session

_HEADER = re.compile(r"^\s*(?:\[)?\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}")


@dataclass
class Turn:
    role: str
    content: str


def _clean_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [line.rstrip() for line in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    # A continuation line that looks like a WhatsApp header would be parsed as a
    # new message; a single leading space keeps it inside the current message.
    return "\n".join(
        (" " + line) if _HEADER.match(line) else line for line in lines
    )


def _normalize_question(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _latin_only(text: str) -> bool:
    """True when every letter is Latin — Romanized Hinglish, no Devanagari."""

    return all(
        unicodedata.name(char, "").startswith("LATIN")
        for char in text
        if char.isalpha()
    )


def _turns(row: Sequence[dict[str, Any]]) -> list[Turn]:
    turns: list[Turn] = []
    for message in row:
        role = str(message.get("role", "")).strip()
        if role not in ("user", "assistant"):
            continue
        content = _clean_text(str(message.get("content", "")))
        if not content or content.casefold() == "system":
            continue
        turns.append(Turn(role, content))
    return turns


def _select_rows(rows: Sequence[Sequence[dict[str, Any]]], *, limit: int,
                 seed: int) -> list[list[Turn]]:
    """Keep quality conversations, drop duplicate questions, then sample."""

    eligible: list[list[Turn]] = []
    seen: set[str] = set()
    for row in rows:
        turns = _turns(row)
        questions = [turn for turn in turns if turn.role == "user"]
        answers = [turn for turn in turns if turn.role == "assistant"]
        if not questions or not answers:
            continue
        question, answer = questions[0], answers[0]
        question_text = _normalize_question(question.content)
        if len(question_text) < MIN_QUESTION_CHARS:
            continue
        if len(question_text.split()) < MIN_QUESTION_WORDS:
            continue
        if not MIN_ANSWER_CHARS <= len(answer.content) <= MAX_ANSWER_CHARS:
            continue
        if not _latin_only(question_text) or not _latin_only(answer.content):
            continue
        key = question_text.casefold()
        if key in seen:
            continue
        seen.add(key)
        eligible.append([Turn("user", question_text), answer])

    if limit < len(eligible):
        eligible = random.Random(seed).sample(eligible, limit)
    eligible.sort(key=lambda pair: pair[0].content)
    return eligible


def _render_export(pairs: Sequence[Sequence[Turn]], *, owner: str, target: str,
                   start: datetime) -> tuple[str, list[list[tuple[Turn, datetime]]]]:
    lines: list[str] = []
    scheduled: list[list[tuple[Turn, datetime]]] = []
    cursor = start
    for pair in pairs:
        stamped: list[tuple[Turn, datetime]] = []
        cursor += timedelta(minutes=MIN_GAP_MINUTES + (cursor.minute % 60))
        for turn in pair:
            author = owner if turn.role == "user" else target
            lines.append(
                f"{cursor.strftime('%d/%m/%y, %H:%M')} - {author}: {turn.content}"
            )
            stamped.append((turn, cursor))
            cursor += timedelta(minutes=1 + (len(turn.content) % 3))
        scheduled.append(stamped)
    return "\n".join(lines) + "\n", scheduled


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def fetch(_: argparse.Namespace) -> int:
    SOURCE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    if SOURCE_PARQUET.exists() and SOURCE_PARQUET.stat().st_size > 0:
        print(f"already downloaded: {SOURCE_PARQUET}")
        return 0
    from urllib.request import urlopen

    print(f"downloading {PARQUET_URL}")
    with urlopen(PARQUET_URL, timeout=300) as response:
        SOURCE_PARQUET.write_bytes(response.read())
    print(f"wrote {SOURCE_PARQUET} ({SOURCE_PARQUET.stat().st_size} bytes)")
    return 0


def build(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    if not SOURCE_PARQUET.exists():
        raise SystemExit(f"missing {SOURCE_PARQUET}; run `fetch` first")

    rows = pq.read_table(SOURCE_PARQUET).column("messages").to_pylist()
    pairs = _select_rows(rows, limit=args.limit, seed=args.seed)

    start = datetime.fromisoformat(args.start)
    export_text, scheduled = _render_export(
        pairs, owner=args.owner_author, target=args.target_author, start=start
    )
    export_path = EXPORT_PATH
    export_path.write_text(export_text, encoding="utf-8")
    EXPORT_BASE_PATH.write_text(export_text, encoding="utf-8")

    messages = parse_whatsapp_chat(
        export_text,
        chat_id=args.chat_id,
        day_first=True,
        timezone_name=args.timezone,
    )
    by_sequence = {message.sequence: message for message in messages}

    # Re-walk the export in order so each scheduled turn maps to its message id.
    conversation_rows: list[dict[str, Any]] = []
    cursor = 0
    for index, stamped in enumerate(scheduled, start=1):
        conversation_messages: list[dict[str, Any]] = []
        for turn, _ in stamped:
            message = by_sequence[cursor]
            cursor += 1
            conversation_messages.append(
                {
                    "role": "user" if turn.role == "user" else "assistant",
                    "speaker": message.author,
                    "content": message.text,
                    "message_id": message.message_id,
                }
            )
        conversation_rows.append(
            {
                "conversation_id": f"aditi-{index:05d}",
                "source": "manishiitg/aditi-syn-v1",
                "language": "hinglish",
                "messages": conversation_messages,
            }
        )

    _write_jsonl(CONVERSATION_PATH, conversation_rows)

    print(f"conversations: {len(conversation_rows)}")
    print(f"messages:      {len(messages)}")
    print(f"wrote {export_path}")
    print(f"wrote {EXPORT_BASE_PATH}")
    print(f"wrote {CONVERSATION_PATH}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "build"))
    parser.add_argument("--limit", type=int, default=600,
                        help="number of conversations to keep (default 600)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start", default="2025-11-01T09:00:00",
                        help="first rendered timestamp (ISO, naive)")
    parser.add_argument("--timezone", default="Asia/Kolkata")
    parser.add_argument("--chat-id", default="hinglish-chat")
    parser.add_argument("--owner-author", default="Aditi")
    parser.add_argument("--target-author", default="Rohan")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return fetch(args) if args.command == "fetch" else build(args)


if __name__ == "__main__":
    raise SystemExit(main())
