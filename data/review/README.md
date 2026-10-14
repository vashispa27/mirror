# review — hand-written conversations and their questions

Files here are **for review before merge**. Each `NN-<scenario>.jsonl` holds
synthetic Hinglish conversations for one scenario type, plus the questions we
will use as evals. `NN-<scenario>.md` is a readable preview of the same file.

## Why this exists

The first eval set was derived automatically from the source parquet. Its
questions were noisy and its type labels were guesses. This set is written on
purpose, one scenario at a time, so we can test specific memory behaviour.

## Scenario types

| file | what it tests |
| --- | --- |
| `01-long-single-topic` | one long, deep conversation; details accumulate over many turns |
| `02-diverge-return` | topic A → unrelated topic B → back to A |
| `03-self-disclosure` | personal facts someone reveals about themselves |
| `04-disagreement` | two people hold opposing positions |
| `05-changing-plans` | a plan is revised; later messages contradict earlier ones |
| `06-time-gaps` | multi-day gaps, delayed replies, explicit time references |
| `07-emotional-support` | venting and support; cause of distress, advice given |
| `08-edge-cases` | pronouns, third-party attribution, numbers and ids, repeated episodes, unanswerable questions |

## Schema

One JSON object per line:

```json
{"conversation_id": "01-long-single-topic-1",
 "scenario": "01-long-single-topic",
 "topic": "wedding-venue-search",
 "date": "2026-03-04",
 "messages": [
   {"speaker": "Aditi", "role": "user", "content": "..."},
   {"speaker": "Rohan", "role": "assistant", "content": "..."}
 ],
 "questions": [
   {"query": "...", "query_type": "exact_phrase", "answer_quote": "..."}
 ]}
```

## Rules

* Two speakers only: `Aditi` (`role: "user"`) and `Rohan` (`role: "assistant"`,
  the voice being modelled). Do not put speaker names inside `content`.
* **Roman script only.** Hindi must be spelled in English letters. Devanagari or
  any other non-Latin script is rejected by the checker.
* One message per line — no newlines inside `content`.
* `answer_quote` must be a **verbatim substring** of the message that answers the
  question, so the checker can resolve it to a message id. Use `""` only for
  `query_type: "unanswerable"`.

## Queries

A query is what the user would type to an assistant that knows this chat
history. Write most queries in Hinglish (Roman script); a few may be English.
Do not narrate the conversation back to the reader.

## Question types

| type | rule |
| --- | --- |
| `exact_phrase` | the query reuses a rare phrase or name from the answer |
| `paraphrase` | same meaning, different wording; may share content words |
| `situation` | describes the circumstances; the answer is the reply |
| `semantic` | shares **no** content words with the answer (no names, places, rare nouns) |
| `multihop` | needs two or more messages to answer |
| `temporal` | asks when something happened, or the order of events |
| `attribution` | asks who said or decided something |
| `style` | asks about how Rohan writes, not what he said |
| `unanswerable` | the answer is not in the conversation; `answer_quote` is `""` |

## Check and preview

```bash
.venv/bin/python scripts/check-review.py
```

It validates every file, warns when a `semantic` query shares content words with
its answer (and when an `exact_phrase` query shares none), and writes the `.md`
previews.
