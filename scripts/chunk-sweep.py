#!/usr/bin/env python3
"""Sweep the persona chunk window and compare retrieval on a fair metric.

Same data, same 188 questions, same gold message ids, same retrieval mode and
k. The only thing that changes is how many messages go into one persona chunk.

The comparison metric is `chars_to_gold`: stack the returned chunks in rank
order, then count how many characters you read before reaching the message that
actually answers the question. It is measured in characters, so a 2-message
chunking and a 20-message chunking sit on the same scale -- unlike success@k,
which rewards bigger chunks for simply holding more text.

Usage::

    .venv/bin/python scripts/chunk-sweep.py            # 2, 5, 10, 20
    .venv/bin/python scripts/chunk-sweep.py --sizes 2 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from whatsapp.chunking import chunk_conversation  # noqa: E402
from whatsapp.eval import run_evaluation  # noqa: E402
from whatsapp.parser import parse_whatsapp_chat  # noqa: E402
from whatsapp.persona import build_persona_records  # noqa: E402
from whatsapp.retrieval import WhatsAppRetriever  # noqa: E402
from whatsapp.store import WhatsAppQdrantStore  # noqa: E402

DATA = REPO_ROOT / "data"
EXPORT = DATA / "export.txt"
EVALS = DATA / "evals.jsonl"
MODE = "hybrid_dbsf"
K = 5
BUDGETS = (500, 1_000, 2_000)


def overlap_for(max_messages: int) -> int:
    """Keep overlap at ~20% of the window, so redundancy scales with it."""

    overlap = max(0, round(max_messages * 0.2))
    return min(overlap, max_messages - 1)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(len(ordered) * fraction)))]


def stream_of(hit_chunk_ids: list[str], chunk_msgs: dict[str, list[str]]) -> list[str]:
    """Messages of the returned chunks, rank order, deduplicated."""

    stream: list[str] = []
    seen: set[str] = set()
    for chunk_id in hit_chunk_ids:
        for message_id in chunk_msgs.get(chunk_id, []):
            if message_id not in seen:
                seen.add(message_id)
                stream.append(message_id)
    return stream


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 5, 10, 20])
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--collection-prefix", default="mirror_sweep")
    args = parser.parse_args(argv)

    export_text = EXPORT.read_text(encoding="utf-8")
    messages = parse_whatsapp_chat(
        export_text, chat_id="hinglish-chat", day_first=True,
        timezone_name="Asia/Kolkata",
    )
    text_of = {m.message_id: f"{m.author}: {m.text}" for m in messages}
    gold = {r["query"]: r["expected_target_message_ids"][0]
            for r in (json.loads(l) for l in open(EVALS, encoding="utf-8"))}

    reports = []
    for size in args.sizes:
        overlap = overlap_for(size)
        chunks = chunk_conversation(
            messages, max_messages=size, overlap_messages=overlap
        )
        records = build_persona_records(
            messages, chunks, user_id="local-user", persona_id="hinglish-twin",
            requested_target_author="Rohan", target_author="Rohan",
        )
        collection = f"{args.collection_prefix}_{size}"
        started = time.perf_counter()
        store = WhatsAppQdrantStore()
        store.index(collection, records, parsed_message_count=len(messages))
        index_seconds = time.perf_counter() - started

        report = run_evaluation(
            dataset_path=str(EVALS),
            retriever=WhatsAppRetriever(),
            collection_name=collection,
            k=args.k,
            modes=[MODE],
            evidence_scope="any_author",
        )

        chunk_msgs = {c.chunk_id: list(c.message_ids) for c in chunks}
        found: list[float] = []
        missing = 0
        for query in report.per_query:
            answer = gold.get(query.query)
            hit_ids = [r.chunk_id for r in query.modes[MODE].results]
            stream = stream_of(hit_ids, chunk_msgs)
            if answer not in stream:
                missing += 1
                continue
            position = stream.index(answer)
            found.append(sum(len(text_of[m]) + 1 for m in stream[:position]))

        total = len(found) + missing
        reports.append({
            "size": size,
            "overlap": overlap,
            "items": len(records),
            "index_s": round(index_seconds, 1),
            "present": len(found) / total,
            "median_chars": statistics.median(found) if found else 0,
            "p75_chars": percentile(found, 0.75),
            "budgets": {b: sum(1 for c in found if c <= b) / total for b in BUDGETS},
            "success_at_k": report.aggregate[MODE].success_at_k,
        })
        print(f"  [{size:>2} msgs] {len(records)} chunks indexed in "
              f"{index_seconds:.0f}s  present={len(found)/total:.3f}  "
              f"median_chars_to_gold={reports[-1]['median_chars']:.0f}",
              flush=True)

    print()
    header = (f"{'window':>7}{'overlap':>8}{'chunks':>8}{'present':>9}"
              f"{'median_chars':>14}{'p75_chars':>11}")
    for budget in BUDGETS:
        header += f"{'@' + str(budget):>9}"
    print(header)
    print("-" * len(header))
    for row in reports:
        line = (f"{row['size']:>7}{row['overlap']:>8}{row['items']:>8}"
                f"{row['present']:>9.3f}{row['median_chars']:>14.0f}"
                f"{row['p75_chars']:>11.0f}")
        for budget in BUDGETS:
            line += f"{row['budgets'][budget]:>9.3f}"
        print(line)
    print()
    print("present       = gold message somewhere in the returned evidence")
    print("median_chars  = characters read before the gold message")
    print("@N            = fraction of questions answered within N characters")
    (DATA / "chunk-sweep.json").write_text(
        json.dumps(reports, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {DATA / 'chunk-sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
