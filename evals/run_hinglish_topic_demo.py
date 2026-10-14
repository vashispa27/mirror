"""Run a synthetic Hinglish topic-boundary probe with local Qwen3.5-4B.

Install mlx-vlm in a separate environment, then run this file with that Python.
No real chat or database is read or changed.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


ROOT = Path(__file__).resolve().parent
MESSAGES = ROOT / "hinglish_topic_demo.txt"
RESULT = ROOT / "hinglish_topic_demo_results.json"
MODEL = "mlx-community/Qwen3.5-4B-MLX-4bit"
LINE = re.compile(r"^(\d+)\s*\|\s*([^|]+)\|\s*(.+)$")


def read_messages() -> list[tuple[int, str]]:
    messages: list[tuple[int, str]] = []
    for line in MESSAGES.read_text().splitlines():
        match = LINE.match(line)
        if match:
            messages.append((int(match.group(1)), f"{match.group(2).strip()}: {match.group(3)}"))
    if [number for number, _ in messages] != list(range(len(messages))):
        raise ValueError("Message numbers must be contiguous from zero")
    return messages


def parse_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model did not return a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Model response must be an object")
    return value


def make_prompt(
    current: list[tuple[int, str]],
    previous: list[tuple[int, str]],
    earlier_topics: list[dict],
    active_topic: dict | None,
) -> str:
    block_start = current[0][0]
    block_end = current[-1][0]
    context = "\n".join(f"{number:02d} | {body}" for number, body in previous)
    transcript = "\n".join(f"{number:02d} | {body}" for number, body in current)
    return f"""You are marking topic boundaries in a synthetic Roman-script Hinglish chat.
Read messages in order. A boundary is the index of the FIRST message about a new
substantive subject. The block edge itself is never a boundary. Keep vague replies
with the subject they refer to. Greetings can remain unassigned. Ignore elapsed
time as a reason to split. If a subject returns after an intervening subject,
mark RESUME and reuse its earlier topic ID. Do not force a fixed number of topics.

Current block: {block_start} through {block_end} inclusive.
Earlier topics: {json.dumps(earlier_topics, ensure_ascii=False)}
Topic active before this block: {json.dumps(active_topic, ensure_ascii=False)}
Previous messages for context only (never return these indices):
{context or '(none)'}

Current messages:
{transcript}

Return ONLY a JSON object with two keys: "starts" and "ending_topic". Scan the
ENTIRE block and include EVERY topic start, even when several unrelated subjects
appear in one block. "starts" is a list of objects, each containing integer
"index", string "topic_id", short string "label", and "kind" (NEW or RESUME).
Use T1, T2, ... for new topics in order. "ending_topic" is an object containing
"topic_id", "label", and a one-sentence "summary" of the topic active at the
LAST message in this block. If no topic starts in this block, return an empty
starts list. ending_topic is null only if no substantive topic has started yet.
Topic summary is for state carryover, not a replacement for original messages.
Check that the ending topic actually describes the LAST message. No explanation
outside JSON."""


def main() -> None:
    messages = read_messages()
    print(f"Loading {MODEL} locally...", flush=True)
    model, processor = load(MODEL)
    config = load_config(MODEL)
    blocks = [(0, 29), (30, len(messages) - 1)]
    previous_topics: list[dict] = []
    active_topic: dict | None = None
    results: list[dict] = []
    for first, last in blocks:
        current = messages[first : last + 1]
        previous = messages[max(0, first - 4) : first]
        prompt = make_prompt(current, previous, previous_topics, active_topic)
        formatted = apply_chat_template(
            processor, config, prompt, num_images=0, enable_thinking=False
        )
        print(f"Classifying messages {first}-{last}...", flush=True)
        started = time.monotonic()
        response = generate(
            model,
            processor,
            formatted,
            max_tokens=500,
            temperature=0,
            enable_thinking=False,
            verbose=False,
        )
        raw = response.text
        result: dict = {
            "block": [first, last],
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "raw_output": raw,
        }
        try:
            parsed = parse_json(raw)
            result["parsed"] = parsed
            for start in parsed.get("starts", []):
                if start.get("kind") == "NEW":
                    previous_topics.append(
                        {"topic_id": start.get("topic_id"), "label": start.get("label")}
                    )
            active_topic = parsed.get("ending_topic")
        except (ValueError, json.JSONDecodeError) as exc:
            result["parse_error"] = str(exc)
        results.append(result)
        RESULT.write_text(
            json.dumps({"model": MODEL, "results": results}, indent=2, ensure_ascii=False)
            + "\n"
        )
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if "parse_error" in result:
            break
    print(f"Saved {RESULT}", flush=True)


if __name__ == "__main__":
    main()
