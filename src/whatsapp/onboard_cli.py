"""Review-first onboarding for a WhatsApp persona and contact memory."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Sequence

from .authors import extract_authors, resolve_target_author
from .chunking import chunk_conversation
from .memory_db import MemoryRepository
from .memory_retrieval import MemoryQdrantStore
from .memory_service import process_chat_memory
from .models import WhatsAppMessage
from .parser import parse_whatsapp_chat
from .persona import build_persona_records
from .store import DEFAULT_COLLECTION, DEFAULT_QDRANT_URL, WhatsAppQdrantStore
from .topic_memory import OpenAICompatibleTopicModel, TOPIC_COLLECTION


MANIFEST_VERSION = 1
DEFAULT_FRACTION = 0.30
DEFAULT_BATCH_SIZE = 250


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_jsonl(path: Path, values: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object")
        values.append(value)
    return values


def _default_workspace(chat_id: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", chat_id).strip("-").lower()
    if not slug:
        raise ValueError("chat-id must contain letters or numbers")
    return Path("private-chats") / slug


def prepare_chat(
    input_path: Path, workspace: Path, *, target_author: str,
    user_id: str, contact_id: str, chat_id: str, timezone_name: str,
    persona_id: str | None = None,
    fraction: float = DEFAULT_FRACTION, all_messages: bool = False,
    review_batch_size: int = DEFAULT_BATCH_SIZE, day_first: bool = True,
    owner_author: str | None = None,
) -> dict[str, Any]:
    """Parse the complete export and write a private, editable review set."""

    if not all((user_id, contact_id, chat_id, timezone_name)):
        raise ValueError("user-id, contact-id, chat-id and timezone are required")
    if not 0 < fraction <= 1 or review_batch_size < 1:
        raise ValueError("fraction must be in (0,1] and batch size must be positive")
    source = input_path.read_bytes()
    messages = parse_whatsapp_chat(source.decode("utf-8-sig"), chat_id=chat_id,
                                   day_first=day_first, timezone_name=timezone_name)
    if not messages:
        raise ValueError("no WhatsApp messages were parsed")
    canonical_target = resolve_target_author(messages, target_author)
    canonical_owner = resolve_target_author(messages, owner_author) if owner_author else None
    usable = sorted((m for m in messages if m.is_usable_text),
                    key=lambda m: (m.timestamp, m.sequence))
    if not usable:
        raise ValueError("export contains no usable text messages")
    count = len(usable) if all_messages else max(1, math.ceil(len(usable) * fraction))
    selected = usable[-count:]
    target_count = sum(m.author == canonical_target for m in selected)
    if not target_count:
        raise ValueError("selected range contains no messages by the target author; use --all")
    if workspace.exists():
        raise FileExistsError(f"workspace already exists: {workspace}; choose a new --workspace")
    review_dir = workspace / "review"
    review_dir.mkdir(parents=True)
    _write_jsonl(workspace / "cleaned_all.jsonl", [m.to_dict() for m in messages])
    review_files: list[str] = []
    for offset in range(0, len(selected), review_batch_size):
        batch = selected[offset:offset + review_batch_size]
        stem = f"part-{offset // review_batch_size + 1:04d}"
        json_name = f"review/{stem}.jsonl"
        review_files.append(json_name)
        entries = [{"original_message_id": m.message_id, "sequence": m.sequence,
                    "timestamp": m.timestamp.isoformat(), "author": m.author,
                    "text": m.text, "include": True} for m in batch]
        _write_jsonl(workspace / json_name, entries)
        with (review_dir / f"{stem}.txt").open("w", encoding="utf-8") as stream:
            for entry in entries:
                stream.write(f"[{entry['sequence']}] {entry['timestamp']} {entry['author']}: "
                             f"{entry['text']}\n\n")
    selected_ids = "\n".join(m.message_id for m in selected).encode()
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "source_file": str(input_path.resolve()), "source_sha256": _sha256(source),
        "user_id": user_id, "contact_id": contact_id, "chat_id": chat_id,
        "timezone": timezone_name, "target_author": canonical_target,
        "persona_id": persona_id or canonical_target,
        "owner_author": canonical_owner, "authors": list(extract_authors(messages)),
        "selection": "all" if all_messages else "most_recent",
        "fraction": 1.0 if all_messages else fraction,
        "parsed_messages": len(messages), "usable_messages": len(usable),
        "selected_messages": len(selected), "selected_target_messages": target_count,
        "selected_message_ids_sha256": _sha256(selected_ids),
        "selected_start": selected[0].timestamp.isoformat(),
        "selected_end": selected[-1].timestamp.isoformat(),
        "review_files": review_files,
    }
    (workspace / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (review_dir / "README.txt").write_text(
        "Read the part-*.txt files. Edit part-*.jsonl to set include=false or redact "
        "the text value. Keep every JSONL row, ID, author, timestamp and sequence. "
        "The index command reads JSONL, not TXT.\n", encoding="utf-8"
    )
    return {"workspace": str(workspace.resolve()), **manifest}


def _approved_messages(workspace: Path, manifest: dict[str, Any]) -> tuple[list[WhatsAppMessage], str]:
    original_rows = _read_jsonl(workspace / "cleaned_all.jsonl")
    originals = {r["message_id"]: r for r in original_rows}
    if len(originals) != len(original_rows):
        raise ValueError("cleaned export contains duplicate message IDs")
    reviewed = [entry for name in manifest["review_files"] for entry in _read_jsonl(workspace / name)]
    ids = [entry.get("original_message_id") for entry in reviewed]
    if (len(reviewed) != manifest["selected_messages"] or
            any(not isinstance(value, str) for value in ids) or
            len(set(ids)) != len(ids)):
        raise ValueError("review rows were removed or duplicated")
    if _sha256("\n".join(ids).encode()) != manifest["selected_message_ids_sha256"]:
        raise ValueError("review message IDs or their order changed")
    approved: list[WhatsAppMessage] = []
    for entry in reviewed:
        original = originals.get(entry["original_message_id"])
        if original is None:
            raise ValueError(f"unknown review message ID: {entry['original_message_id']}")
        for field, source_field in (("sequence", "sequence"), ("timestamp", "timestamp"),
                                    ("author", "author")):
            if entry.get(field) != original[source_field]:
                raise ValueError(f"review {field} was changed for message {entry['original_message_id']}")
        if type(entry.get("include")) is not bool or not isinstance(entry.get("text"), str):
            raise ValueError("review include must be boolean and text must be a string")
        if not entry["include"]:
            continue
        text = entry["text"].strip()
        if not text:
            raise ValueError("included review message has empty text; set include=false")
        from datetime import datetime
        approved.append(WhatsAppMessage(
            message_id=original["message_id"], chat_id=manifest["chat_id"],
            sequence=original["sequence"], timestamp=datetime.fromisoformat(original["timestamp"]),
            author=original["author"], text=text, source_timezone=manifest["timezone"],
            raw_content=f"{original['author']}: {text}",
        ))
    digest = _sha256(json.dumps(
        [(m.message_id, m.text) for m in approved], ensure_ascii=False,
        separators=(",", ":")
    ).encode())
    return approved, digest


def index_chat(
    workspace: Path, *, postgres_dsn: str, qdrant_url: str,
    topic_api_url: str | None = None, topic_model_name: str | None = None,
) -> dict[str, Any]:
    """Index exactly the approved review set into Postgres and both Qdrant paths."""

    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != MANIFEST_VERSION:
        raise ValueError("unsupported review workspace version")
    approved, review_hash = _approved_messages(workspace, manifest)
    if not approved or not any(m.author == manifest["target_author"] for m in approved):
        raise ValueError("review excluded all messages or all target-author messages")
    if topic_model_name and not topic_api_url:
        raise ValueError("--topic-api-url is required with --topic-model")
    if topic_api_url:
        if not topic_model_name:
            raise ValueError("--topic-model is required with --topic-api-url")
        model = OpenAICompatibleTopicModel(topic_api_url, topic_model_name)
    else:
        if importlib.util.find_spec("mlx_vlm") is None:
            raise RuntimeError("local topic model is unavailable; install topic-local on Apple Silicon "
                               "or pass --topic-api-url and --topic-model")
        model = None
    report_path = workspace / "index_report.json"
    attempt_path = workspace / "index_attempt.json"
    for path in (report_path, attempt_path):
        if path.exists() and json.loads(path.read_text())["review_hash"] != review_hash:
            raise ValueError("review changed after indexing began; use a new workspace/chat-id")
    repository = MemoryRepository(postgres_dsn)
    try:
        repository.migrate()
        existing = repository.messages_for_chat(manifest["user_id"], manifest["contact_id"],
                                                manifest["chat_id"])
        if existing and not attempt_path.exists():
            raise ValueError("chat-id already has stored messages; choose a fresh chat-id")
        if not attempt_path.exists():
            attempt_path.write_text(json.dumps({"review_hash": review_hash}) + "\n")
        repository.import_export(
            approved, user_id=manifest["user_id"], contact_id=manifest["contact_id"],
            chat_id=manifest["chat_id"], export_hash=review_hash,
            owner_author=manifest["owner_author"],
        )
        memory = process_chat_memory(
            repository=repository, vector_store=MemoryQdrantStore(qdrant_url),
            user_id=manifest["user_id"], contact_id=manifest["contact_id"],
            chat_id=manifest["chat_id"], topic_model=model,
        )
        chunks = chunk_conversation(approved)
        records = build_persona_records(
            approved, chunks, user_id=manifest["user_id"],
            persona_id=manifest["persona_id"],
            requested_target_author=manifest["target_author"],
            target_author=manifest["target_author"],
        )
        persona = WhatsAppQdrantStore(qdrant_url).index(
            DEFAULT_COLLECTION, records, parsed_message_count=len(approved)
        )
        result = {
            "workspace": str(workspace.resolve()), "review_hash": review_hash,
            "approved_messages": len(approved),
            "excluded_messages": manifest["selected_messages"] - len(approved),
            "stored_messages": len(repository.messages_for_chat(
                manifest["user_id"], manifest["contact_id"], manifest["chat_id"])),
            "memory": memory, "memory_collection": TOPIC_COLLECTION,
            "persona": persona.to_dict(), "persona_collection": DEFAULT_COLLECTION,
        }
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        return result
    finally:
        repository.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Review and index a WhatsApp chat from scratch")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="parse, clean, select and split for review")
    prepare.add_argument("input", type=Path)
    prepare.add_argument("--target-author", required=True)
    prepare.add_argument("--persona-id", help="stable ID for the selected person's style index")
    prepare.add_argument("--owner-author")
    prepare.add_argument("--user-id", required=True)
    prepare.add_argument("--contact-id", required=True)
    prepare.add_argument("--chat-id", required=True)
    prepare.add_argument("--timezone", required=True)
    prepare.add_argument("--month-first", action="store_true")
    prepare.add_argument("--workspace", type=Path)
    selection = prepare.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="review the whole usable chat")
    selection.add_argument("--fraction", type=float, default=DEFAULT_FRACTION,
                           help="newest fraction of usable messages (default 0.30)")
    prepare.add_argument("--review-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    index = sub.add_parser("index", help="index the reviewed set into Postgres and Qdrant")
    index.add_argument("workspace", type=Path)
    index.add_argument("--approve", action="store_true", help="confirm review files are ready")
    index.add_argument("--postgres-dsn", default=os.getenv(
        "POSTGRES_DSN", "postgresql://mirror:mirror@localhost:5432/mirror"))
    index.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL))
    index.add_argument("--topic-api-url", help="alternative OpenAI-compatible /v1 endpoint")
    index.add_argument("--topic-model", help="model at --topic-api-url")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare_chat(
                args.input, args.workspace or _default_workspace(args.chat_id),
                target_author=args.target_author, owner_author=args.owner_author,
                persona_id=args.persona_id,
                user_id=args.user_id, contact_id=args.contact_id, chat_id=args.chat_id,
                timezone_name=args.timezone, fraction=args.fraction,
                all_messages=args.all, review_batch_size=args.review_batch_size,
                day_first=not args.month_first,
            )
        else:
            if not args.approve:
                parser.error("index requires --approve after reviewing the JSONL files")
            result = index_chat(args.workspace, postgres_dsn=args.postgres_dsn,
                                qdrant_url=args.qdrant_url, topic_api_url=args.topic_api_url,
                                topic_model_name=args.topic_model)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
