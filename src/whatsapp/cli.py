from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .chunking import chunk_conversation
from .parser import parse_whatsapp_chat


def _write_jsonl(values: Iterable[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse and clean a WhatsApp text export."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--timezone", dest="timezone_name")
    parser.add_argument("--month-first", action="store_true")
    parser.add_argument(
        "--output-type", choices=("messages", "chunks"), default="messages"
    )
    args = parser.parse_args()

    export_text = args.input.read_text(encoding="utf-8-sig")
    messages = parse_whatsapp_chat(
        export_text,
        chat_id=args.chat_id,
        day_first=not args.month_first,
        timezone_name=args.timezone_name,
    )

    if args.output_type == "chunks":
        values = (chunk.to_dict() for chunk in chunk_conversation(messages))
    else:
        values = (message.to_dict() for message in messages)
    _write_jsonl(values, args.output)


if __name__ == "__main__":
    main()
