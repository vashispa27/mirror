# data

Everything here is synthetic and safe to commit. Two speakers throughout:
`Aditi` (asks) and `Rohan` (answers; the voice being modelled). Roman script
only — Hindi is spelled in English letters.

| File | What it is |
| --- | --- |
| `export.txt` | **The file you index.** Base conversations + reviewed conversations. |
| `export-base.txt` | The base conversations only (600 Q&A pairs from `manishiitg/aditi-syn-v1`). |
| `conversations.jsonl` | The merged dataset — 600 base + reviewed conversations, one per line. |
| `evals.jsonl` | The eval set. Reviewed questions with the id of the message that answers each. |
| `evals-unanswerable.jsonl` | Reviewed questions whose answer is not in the conversation. |
| `source/train.parquet` | Original download of `manishiitg/aditi-syn-v1` (25,000 conversations; we keep 600). |
| `review/` | Hand-written conversations and their questions. **Start with `review/README.md`.** |

## Build order

```bash
# 1. base conversations from the parquet
.venv/bin/python scripts/build-dataset.py build --limit 600 --seed 7

# 2. check the hand-written review files, and write .md previews
.venv/bin/python scripts/check-review.py

# 3. merge review into the dataset and build the eval set
.venv/bin/python scripts/merge-review.py
```

Steps 1 and 3 are deterministic. Re-running `merge-review.py` produces byte
identical output — it always starts from `export-base.txt`, never from its own
previous output. Re-running step 1 resets the merge, so run step 3 again after.

## How the eval set is built

`evals.jsonl` is written **backwards** from the conversations. The review files
state each question together with a verbatim `answer_quote`; the merge resolves
that quote to exactly one message and records its `message_id`. A quote that
matches zero or several messages is reported and skipped rather than guessed.

`query_type` is one of `exact_phrase`, `paraphrase`, `situation`, `semantic`,
`multihop`, `temporal`, `attribution`, `style`, `unanswerable`. The `semantic`
rows deliberately share no content words with their answer, so they are the fair
test of the dense retriever; `exact_phrase` rows are the lexical slice.

All rows carry `user_id: local-user` and `persona_id: hinglish-twin`. Some
answers come from Aditi rather than Rohan — that is intended for fact and
attribution questions. Score the memory/fact suite with `--evidence-scope
any_author`; use `--evidence-scope target_only` only for the voice/style slice.
