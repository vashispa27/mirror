"""Def-DTS-style intent probe on the existing synthetic Hinglish chat.

This adapts Lee et al. (ACL Findings 2025) to local Qwen3.5-4B and the existing
42-message demo. Each message is judged with up to two preceding and three
following messages. It does not access the real chat or any database.
"""

from __future__ import annotations

import json
import re
import time
from html import escape
from pathlib import Path

from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

from run_hinglish_topic_demo import MODEL, read_messages


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "hinglish_def_dts_results.json"
INTENTS = {
    "JUST_COMMENT",
    "JUST_ANSWER",
    "DEVELOP_TOPIC",
    "INTRODUCE_TOPIC",
    "CHANGE_TOPIC",
}
SHIFT_INTENTS = {"INTRODUCE_TOPIC", "CHANGE_TOPIC"}


def xml_lines(messages: list[tuple[int, str]]) -> str:
    return "\n".join(
        f'<utterance id="{number}">{escape(body)}</utterance>'
        for number, body in messages
    )


def make_prompt(messages: list[tuple[int, str]], index: int) -> str:
    preceding = messages[max(0, index - 2) : index]
    following = messages[index + 1 : index + 4]
    current = messages[index : index + 1]
    return f"""Classify the intent of the TARGET utterance in a Roman-script Hinglish chat.
Use the dialogue before and after it to understand short or vague replies.

<valid_utterance_intent>
<item name="JUST_COMMENT">A reaction or comment on the preceding subject, with no new subject. Example: "Train late hai" -> "haan, annoying hai".</item>
<item name="JUST_ANSWER">Answers a preceding question while staying on its subject. Example: "tickets li?" -> "haan, dono book kar li".</item>
<item name="DEVELOP_TOPIC">Adds a closely related detail or inclusive subtopic. Example: "I started painting" -> "which brushes are you using?".</item>
<item name="INTRODUCE_TOPIC">Introduces a related but distinct conversational subject. Example: "I love hiking" -> "speaking of outdoors, I started birdwatching".</item>
<item name="CHANGE_TOPIC">Changes to an unrelated subject. Example: "Did you pay the bill?" -> "Who won the cricket match?".</item>
</valid_utterance_intent>

Summarize the preceding and following context separately in one short sentence
each, then choose EXACTLY ONE intent for the TARGET. Decide whether the TARGET
itself changes from the preceding subject. Following turns may clarify what the
TARGET refers to, but a new subject first raised in a later turn is a boundary
at THAT LATER TURN, not at the TARGET. A question can continue an existing
subject; do not label every question a change. A different detail, participant,
or task within the same ongoing subject is DEVELOP_TOPIC, not a boundary.
A processing block edge or elapsed time is not evidence of a topic change.

<preceding_messages>
{xml_lines(preceding)}
</preceding_messages>
<target_message>
{xml_lines(current)}
</target_message>
<following_messages>
{xml_lines(following)}
</following_messages>

Return only this XML structure for utterance {index}:
<result>
<preceding_context>one short sentence or none</preceding_context>
<subsequent_context>one short sentence or none</subsequent_context>
<utterance_intent>one label from valid_utterance_intent</utterance_intent>
</result>"""


def extract_intent(raw: str) -> str | None:
    match = re.search(r"<utterance_intent>\s*([A-Z_]+)\s*</utterance_intent>", raw)
    if match and match.group(1) in INTENTS:
        return match.group(1)
    return None


def main() -> None:
    messages = read_messages()
    print(f"Loading {MODEL} locally...", flush=True)
    model, processor = load(MODEL)
    config = load_config(MODEL)
    results: list[dict] = []
    for index, _ in messages:
        prompt = make_prompt(messages, index)
        formatted = apply_chat_template(
            processor, config, prompt, num_images=0, enable_thinking=False
        )
        started = time.monotonic()
        response = generate(
            model,
            processor,
            formatted,
            max_tokens=180,
            temperature=0,
            enable_thinking=False,
            verbose=False,
        )
        raw = response.text
        intent = extract_intent(raw)
        item = {
            "index": index,
            "intent": intent,
            "topic_shift": intent in SHIFT_INTENTS if intent else None,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "raw_output": raw,
        }
        results.append(item)
        RESULT.write_text(
            json.dumps(
                {
                    "method": "Def-DTS-style per-utterance local adaptation",
                    "model": MODEL,
                    "source": "https://aclanthology.org/2025.findings-acl.1066/",
                    "context": "two preceding and three following messages",
                    "results": results,
                    "predicted_boundaries_excluding_initial_message": [
                        row["index"]
                        for row in results
                        if row["index"] > 0 and row["topic_shift"]
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print(
            f'{index:02d} {intent or "PARSE_ERROR":16} '
            f'{"SHIFT" if item["topic_shift"] else ""} '
            f'{item["elapsed_seconds"]:.1f}s',
            flush=True,
        )
    print(f"Saved {RESULT}", flush=True)


if __name__ == "__main__":
    main()
