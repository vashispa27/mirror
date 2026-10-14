# Retrieval evaluation datasets

This directory holds **synthetic** examples that are safe to commit. Put your
own labelled evaluation file somewhere private and Git-ignored (for example
`private-chats/` in this repository, or outside the repository entirely). The
evaluator only ever reads the file; it never rewrites or copies it.

## Files

| file | purpose |
| --- | --- |
| `whatsapp.example.jsonl` | Template with placeholders. Copy it and fill in your own questions, scope, and expected message IDs. |
| `whatsapp.synthetic.jsonl` | Tiny labelled set that matches `tests/fixtures/synthetic_whatsapp_chat.txt`; used by the automated integration test. |

## Line format

One JSON object per line:

```json
{
  "query": "How does he react when his friend is overwhelmed?",
  "user_id": "your-user-id",
  "persona_id": "your-persona-id",
  "expected_target_message_ids": ["<message-id>", "<message-id>"],
  "query_type": "paraphrase"
}
```

* `query` — the question you would ask the retriever.
* `user_id`, `persona_id` — the scope the collection was indexed with. Every
  row in one file must share the same scope.
* `expected_target_message_ids` — one or more **message IDs** (not chunk IDs)
  that a good result should surface. Must not be empty. Which authors are
  allowed depends on the suite:
  * **memory/fact suite** — either speaker's message is valid evidence; score
    with `--evidence-scope any_author` (the default).
  * **voice/style suite** — only the indexed target author's messages; score
    with `--evidence-scope target_only`.
* `query_type` — optional label such as `exact_phrase`, `paraphrase`,
  `situation`, `style`, or `semantic`. Used only to break the report down
  further.

Aim for a deliberate mix of lexical and semantic questions:

* **Lexical / keyword kind** — `exact_phrase` and `style`. The question reuses
  wording, names, or tokens that appear in the answer messages. BM25 is
  expected to dominate here.
* **Semantic kind** — `semantic`. The question is deliberately written with
  **no content words shared** with the answer messages (no names, places,
  rare nouns, or quoted phrasing). BM25 cannot score these at all, so they are
  the only fair test of the dense retriever. Keep `paraphrase`/`situation`
  questions too, but treat them as a middle tier: they usually still share
  several distinctive words with the answer.

Without a zero-overlap `semantic` slice, an evaluation will make keyword search
look better than it is, because lexical questions outnumber truly
reworded ones.

Blank lines are ignored. `relevant_message_ids` is accepted as an alias for
`expected_target_message_ids`.

`whatsapp.example.jsonl` is a template only. The evaluator rejects any value
containing the `REPLACE_WITH`/`PASTE_` placeholders with an explicit error, so
copy the file and fill it in before running. A wrong but non-placeholder scope
does not error; instead the report warns that the scope filter matched nothing.

## How to obtain target message IDs

Message IDs are the parser's `message_id` values and they appear verbatim in
each indexed payload's `target_message_ids` and `message_ids` arrays.

1. **From the search CLI** (`--format json`). Each hit prints its
   `target_message_ids`, `chunk_id`, timestamps, and text:

   ```bash
   PYTHONPATH=src .venv/bin/python -m whatsapp.search_cli "your question" \
     --user-id your-user-id --persona-id your-persona-id \
     --mode sparse --limit 10 --format json
   ```

2. **From a parsed export.** Parse the same export with the same
   `--chat-id`, `--timezone`, and date order used for indexing, then read each
   message's `message_id`:

   ```bash
   PYTHONPATH=src .venv/bin/python -m whatsapp.cli chat.txt cleaned.jsonl \
     --chat-id personal-chat --timezone Asia/Kolkata
   ```

   Each JSONL row is a message with `message_id`, `author`, `text`, and
   `timestamp`. For a fact suite, label the message(s) that actually state the
   fact, whichever author wrote them.

3. **From Qdrant payloads.** Every point payload carries `message_ids` (all
   messages in the chunk) and `target_message_ids` (the target author's subset).

For a memory/fact suite, evidence may come from either speaker, so score with
`--evidence-scope any_author`. Only if you are evaluating voice/style should you
restrict labels to the target author and use `--evidence-scope target_only`.

## Building a useful set (20–50 questions)

Read the export itself and label independently of any retrieval ranking.

* Mix query types: `exact_phrase` (a rare phrase), `paraphrase` (same meaning,
  different words), `situation` (circumstances rather than the reply), and
  `style` (a writing habit shown by concrete messages).
* Spread questions across different dates and conversations.
* When several target messages clearly answer a question, list all of them.
* Exclude ambiguous questions you cannot verify.
* Verify each labelled ID appears in the indexed payloads. If an ID is missing,
  that is an indexing-coverage problem: report it, do not silently drop it.

## Notes

* Do not use an LLM to invent ground-truth labels for private chat data. Read
  the source.
* Unanswerable questions are **not** ordinary relevance examples. Without a
  calibrated score threshold, top-K retrieval always returns something, so the
  harness deliberately does not report a "no answer accuracy" metric.
