# Hinglish dataset, index design, and retrieval evaluation

This report covers what data we used, how we seeded and expanded it, how the
eval set was built, how the two indexes are laid out and why, which models and
algorithms run, how long each stage takes, and what the retrieval numbers are.

Everything below is reproducible from this repository. Synthetic data only — no
real personal chats are involved.

---

## 1. Data

### 1.1 Source

| | |
| --- | --- |
| Dataset | `manishiitg/aditi-syn-v1` (Hugging Face) |
| Contents | 25,000 synthetic Hinglish/Hindi conversations, `messages` role/content form |
| Format | single parquet, 36.8 MB download / 113 MB uncompressed |
| License | Apache-2.0 |

The source is a question-answer corpus. Each row holds a `system` prompt, a
`user` turn and an `assistant` turn, written in a mix of Roman-script Hinglish
and Devanagari Hindi.

### 1.2 How the base set was seeded

`scripts/build-dataset.py` keeps a deterministic subset:

1. **Shape filter** — a conversation is eligible when it has at least one user
   and one assistant turn, the question is ≥ 12 characters and ≥ 4 words, and
   the answer is 60–1,400 characters.
2. **Script filter** — every letter must be Latin. Hindi has to be written in
   English letters ("Eid-ul-Fitr hai Islam ke ek pramukh tyohar"). Any row
   containing Devanagari or another non-Latin script is dropped. This matters:
   roughly a quarter of the source answers are pure Devanagari.
3. **Deduplication** — rows are deduplicated on the normalised question, because
   the source reuses prompts heavily.
4. **Sampling** — a seeded random sample (`--limit 600 --seed 7`), then sorted by
   question text so the output is stable run to run.

Result: **600 base conversations / 1,200 messages**.

The conversations are rendered into a WhatsApp-style transcript
(`data/export.txt`):

```
01/11/25, 11:15 - Aditi: <question>
01/11/25, 11:18 - Rohan: <answer>
```

Each pair is separated by 135 minutes so the session chunker treats it as its
own conversation. Timestamps start at 2025-11-01 09:00 Asia/Kolkata. The user
role is mapped to `Aditi` and the assistant role to `Rohan`; Rohan is the voice
we model.

### 1.3 The hand-written scenario set

The base set is 600 unrelated question-answer pairs. That is enough to measure
lookup, but it cannot test memory behaviour over time — topic drift, changing
facts, personal disclosure, disagreement. So we wrote a second set by hand.

Eight scenario families, one file each, written by eight independent agents
against a fixed schema:

| file | what it tests | convos | msgs |
| --- | --- | --- | --- |
| `01-long-single-topic` | one deep topic, details accumulate over many turns | 4 | 109 |
| `02-diverge-return` | topic A → unrelated topic B → back to A | 4 | 70 |
| `03-self-disclosure` | personal facts someone reveals about themselves | 4 | 68 |
| `04-disagreement` | two people hold opposing positions | 4 | 62 |
| `05-changing-plans` | a plan is revised; later messages contradict earlier ones | 4 | 76 |
| `06-time-gaps` | multi-day gaps, delayed replies, explicit time references | 4 | 56 |
| `07-emotional-support` | venting and support; cause of distress, advice given | 4 | 60 |
| `08-edge-cases` | pronouns, third-party attribution, numbers/ids, repeated episodes, unanswerable questions | 6 | 100 |
| **total** | | **34** | **601** |

Constraints enforced on every written conversation: Roman script only, two
speakers only (`Aditi` user / `Rohan` assistant), one line per message, 6–40
messages per conversation, and every eval question must carry a verbatim
`answer_quote` drawn from the message that answers it.

`scripts/check-review.py` validates all of this before anything is merged. It
rejects non-Latin characters, broken speaker/role pairs, newlines inside
messages, duplicate conversation ids, quotes that match zero messages or several,
and it warns when a question labelled `semantic` actually shares content words
with its answer.

### 1.4 Merged result

`scripts/merge-review.py` appends the written conversations after the base set
and rebuilds everything.

| | conversations | messages |
| --- | --- | --- |
| base (aditi-syn-v1) | 600 | 1,200 |
| hand-written | 34 | 601 |
| **total** | **634** | **1,801** |

Speakers: Aditi 901, Rohan 900. The merge is idempotent — it always starts from
`data/export-base.txt`, never from its own output, so re-running produces
byte-identical files. Base message ids are stable because the written
conversations are appended after the base, leaving base sequence numbers
untouched.

---

## 2. Evals

### 2.1 Built backwards from the conversations

The eval set was not written separately from the data. Each conversation ships
with its own questions, and each question names the message that answers it:

```json
{"query": "Sea Breeze Terrace ka corkage charge kitna hai?",
 "query_type": "exact_phrase",
 "answer_quote": "corkage charge pandrah hazaar"}
```

The merge resolves each `answer_quote` to exactly one message and records that
message's `message_id`. A quote matching zero or several messages is reported and
skipped rather than guessed. This guarantees every label points at real text in
the corpus — the failure mode where an eval asks about something that was never
said cannot happen.

### 2.2 Question types

| type | rule | count |
| --- | --- | --- |
| `paraphrase` | same meaning, different wording; may share content words | 52 |
| `exact_phrase` | the query reuses a rare phrase or name from the answer | 35 |
| `semantic` | shares **no** content words with the answer — no names, places, rare nouns | 34 |
| `temporal` | asks when something happened, or the order of events | 19 |
| `attribution` | asks who said or decided something | 18 |
| `multihop` | needs two or more messages to answer | 14 |
| `situation` | describes the circumstances; the answer is the reply | 13 |
| `style` | asks about how Rohan writes, not what he said | 3 |
| **answerable total** | | **188** |
| `unanswerable` | the answer is not in the conversation at all | 3 |

The `semantic` slice is the important one. Lexical questions outnumber truly
reworded ones in most naive eval sets, which makes keyword search look better
than it is. The 34 zero-overlap questions are the only fair test of the dense
retriever.

Every row carries `user_id: local-user` and `persona_id: hinglish-twin`, matching
the scope the indexes are built with.

All text is Roman script: Hindi words are written in English letters. Devanagari
is filtered out.

---

## 3. Index design

### 3.1 Why two indexes

Writing style and conversation history are different kinds of evidence.

* A fact about a birthday matters **regardless of who said it**.
* A style example is only useful if **the modelled person wrote it**.

One index cannot serve both without conflating them, so Mirror keeps two paths.

```
data/export.txt
  → parse + clean → private review workspace
  → approved messages → Postgres (source of truth)
       ├─ exact assertions → Postgres facts
       ├─ topic detection → summaries + linked spans
       │                    → Qdrant mirror_whatsapp_memory_v3
       └─ target-author chunks
                            → Qdrant mirror_whatsapp_v2
```

| store | holds | answers |
| --- | --- | --- |
| Postgres `messages` | approved messages with authors, timestamps, source ids | evidence behind every result |
| Postgres `facts` | conservatively extracted exact assertions | dates, numbers, ids |
| Qdrant `mirror_whatsapp_memory_v3` | topic summaries + raw-message spans, **both speakers** | "what did we decide about X" |
| Qdrant `mirror_whatsapp_v2` | conversation chunks containing a **target-author** reply | "how does he phrase things" |

### 3.2 Persona chunking (style index)

Fixed-window chunking over whole messages:

| parameter | value |
| --- | --- |
| max messages per chunk | 20 |
| overlap | 4 messages |
| character cap | 4,000 |
| session gap | 120 minutes |

A chunk is only kept if it contains at least one message from the target author.
Both sides of the conversation stay in the chunk text so the reply keeps its
context, but only target messages are marked as `target_message_ids`.

This produced **638 chunks** from 1,801 messages. The 600 base pairs are
20-message windows that never fill; the written conversations span up to 30
messages and split into two overlapping chunks each.

### 3.3 Topic memory chunking (fact index)

Memory does **not** use a fixed time window. It segments on topic change:

1. Messages are grouped into 30-message blocks.
2. A local instruction model reads each block, with up to six previous and four
   following messages for context, and returns the indices where the topic
   changes, plus a label and summary per topic.
3. Within each topic, overlapping **spans** of up to 10 whole messages are built
   with 2 messages of overlap and a 2,400-character cap.
4. Each span embeds the speaker-labelled raw messages, not a summary; each topic
   point embeds its label and summary.

A long topic therefore has one broad route (the summary) and several precise
leaf nodes (the spans). A question about a detail can hit the span directly, or
hit the summary and expand into its child spans. Every point keeps the exact
source message ids so a retrieved claim can be checked against Postgres.

This produced **459 topics** and **634 chunks** → **1,093 memory points**.

Design trade-off: topic segmentation is a model judgement, so it can be wrong.
The spans exist precisely so that a bad boundary costs recall of the summary
route, not of the underlying evidence.

### 3.4 Scope and idempotency

Every record is scoped by `user_id` + `contact_id`; `chat_id` identifies the
conversation; `persona_id` identifies the style being modelled. Derived records
retain source message ids, and all writes are content-addressed, so retries are
safe and an unchanged re-index is a no-op. Topic model responses are cached in
Postgres (`topic_model_cache`), so a re-run does not repeat LLM calls.

---

## 4. Models, algorithms, and timings

### 4.1 What runs where

| stage | model / method | dims | where |
| --- | --- | --- | --- |
| Topic boundaries + summaries | `mlx-community/Qwen3.5-4B-MLX-4bit` | — | local, Apple Silicon (MLX) |
| Topic/span dense vectors | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Qdrant FastEmbed |
| Persona dense vectors | `BAAI/bge-small-en-v1.5` | 384 | Qdrant FastEmbed |
| Sparse vectors (both indexes) | `Qdrant/bm25` | — | Qdrant FastEmbed |
| Fusion | Qdrant RRF, and DBSF | — | Qdrant server-side |
| Optional persona reranker | `Xenova/ms-marco-MiniLM-L-6-v2` | — | local cross-encoder (not used in memory search) |

### 4.2 Measured timings

Full index build: **39 min 22 s** wall clock (19:22:00 → 20:01:22), 1,801
messages → 1,093 memory points + 638 persona points.

| phase | work | time | source |
| --- | --- | --- | --- |
| Parse + Postgres import | 1,801 messages | seconds | measured |
| Topic boundary detection | 61 blocks, 1 LLM call each | ~24 min (**23.4 s/block measured**: 36 blocks in 14 min 02 s) | measured |
| Topic summaries | 459 LLM calls | ~15 min, derived by subtraction (~2.0 s/call) | derived |
| Dense + sparse embedding and Qdrant upload | 1,731 points | under 1 min | derived from benchmark |
| **total** | | **39 min 22 s** | measured |

The build is dominated by local LLM inference — 520 calls in total (61 boundary
+ 459 summary). Embedding is almost free by comparison.

### 4.3 Embedding benchmark

128 documents, average 360 characters, measured on this machine:

| model | load | throughput | dims |
| --- | --- | --- | --- |
| `BAAI/bge-small-en-v1.5` | 0.1 s | **59.1 docs/s** | 384 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 0.4 s | **71.6 docs/s** | 384 |

At those rates the 1,731 indexed points take roughly 30 seconds of embedding
time. Model load is negligible because weights are cached after first use.

---

## 5. Results

Index: `mirror_whatsapp_v2` (persona/style), 638 points. K = 5, 188 queries,
4 retrieval modes. Generated with:

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.eval_cli \
  --dataset data/evals.jsonl --k 5 --evidence-scope any_author --format json \
  > data/eval-run-any-author.json
```

### 5.1 Overall — memory/fact suite (`--evidence-scope any_author`)

Either speaker's message counts as the answer, which is what fact and
attribution questions need.

| mode | success@5 | recall@5 | mrr@5 | ndcg@5 | median latency |
| --- | --- | --- | --- | --- | --- |
| **hybrid_dbsf** | **0.755** | 0.755 | **0.636** | **0.679** | 11.4 ms |
| hybrid_rrf | 0.755 | 0.755 | 0.598 | 0.650 | 10.3 ms |
| sparse (BM25) | 0.739 | 0.739 | 0.627 | 0.670 | 6.3 ms |
| dense | 0.500 | 0.500 | 0.362 | 0.399 | 10.0 ms |

### 5.2 Voice suite (`--evidence-scope target_only`)

Only Rohan's own messages count. The ceiling here is 72.3%, because 52 of the
188 labelled answers were written by Aditi.

| mode | success@5 | mrr@5 |
| --- | --- | --- |
| hybrid_dbsf | 0.537 | 0.432 |
| sparse | 0.516 | 0.429 |
| dense | 0.330 | 0.233 |

### 5.3 By question type (hybrid_dbsf, any_author)

| query_type | success@5 | missed |
| --- | --- | --- |
| exact_phrase | 0.914 | 3 / 35 |
| paraphrase | 0.865 | 7 / 52 |
| multihop | 0.857 | 2 / 14 |
| situation | 0.846 | 2 / 13 |
| attribution | 0.722 | 5 / 18 |
| temporal | 0.684 | 6 / 19 |
| **semantic** | **0.441** | 19 / 34 |
| style | 0.333 | 2 / 3 |

### 5.4 By scenario (hybrid_dbsf)

| scenario | success@5 |
| --- | --- |
| 01-long-single-topic | 17/20 · 85.0% |
| 02-diverge-return | 20/24 · 83.3% |
| 08-edge-cases | 25/30 · 83.3% |
| 07-emotional-support | 16/20 · 80.0% |
| 03-self-disclosure | 18/24 · 75.0% |
| 06-time-gaps | 16/22 · 72.7% |
| 04-disagreement | 15/24 · 62.5% |
| 05-changing-plans | 15/24 · 62.5% |

### 5.5 What the numbers say

1. **Hybrid fusion wins.** `hybrid_dbsf` is the best or joint-best mode overall
   and on the semantic slice, where it reaches 0.441 against dense 0.324 and
   sparse 0.235. The zero-overlap slice is the only honest test of the vector
   index, and fusion is the only method that does not collapse there.
2. **The dense model is the bottleneck.** `BAAI/bge-small-en-v1.5` is
   English-only and is being run over Hinglish. It loses to plain BM25 (0.500 vs
   0.739). This was flagged as unevaluated in the README; it is now measured.
   Switching the persona dense vector to the multilingual model the memory index
   already uses is the obvious next experiment.
3. **Lexical retrieval is saturated.** `exact_phrase` is at 0.914 with BM25
   alone, so further gains have to come from the semantic and temporal slices.
4. **The hardest scenarios are the ones we wrote for that purpose.** Disagreement
   and changing-plans both sit at 62.5% — tracking who held which position, and
   reading the final value of a fact that was revised, are exactly the
   multi-turn reasoning cases the corpus was designed to stress.
5. **Style is essentially untested.** Only 3 questions, 1 hit. The
   `target_only` suite needs more style questions before any claim can be made
   about the voice index.

---

## 6. Reproducing everything

```bash
# --- data ---
.venv/bin/python scripts/build-dataset.py fetch
.venv/bin/python scripts/build-dataset.py build --limit 600 --seed 7

# review the hand-written set (validates + writes .md previews)
.venv/bin/python scripts/check-review.py

# merge review into the dataset and build the eval set
.venv/bin/python scripts/merge-review.py

# --- index ---
scripts/setup-workspace.sh
scripts/onboard-chat.sh prepare data/export.txt \
  --target-author 'Rohan' --owner-author 'Rohan' \
  --persona-id hinglish-twin --user-id local-user \
  --contact-id aditi --chat-id hinglish-chat \
  --timezone Asia/Kolkata --all
scripts/onboard-chat.sh index private-chats/hinglish-chat --approve

# --- evaluate ---
PYTHONPATH=src .venv/bin/python -m whatsapp.eval_cli \
  --dataset data/evals.jsonl --k 5 --evidence-scope any_author --format table

PYTHONPATH=src .venv/bin/python -m whatsapp.eval_cli \
  --dataset data/evals.jsonl --k 5 --evidence-scope target_only --format table

# failure dump for the weakest mode only
PYTHONPATH=src .venv/bin/python -m whatsapp.eval_cli \
  --dataset data/evals.jsonl --k 5 --modes hybrid_dbsf \
  --evidence-scope any_author --format table

# --- memory (topic/span) spot check ---
PYTHONPATH=src .venv/bin/python -m whatsapp.memory_cli \
  --user-id local-user --contact-id aditi --chat-id hinglish-chat \
  search 'Sea Breeze Terrace ka corkage charge kitna hai?'
```

`prepare` defaults to the newest 30% of messages. `--all` is required to index
the full 1,801.

---

## 7. Known issues

* **The labelled harness only scores the persona index.** Memory points carry
  `schema_version: memory-v3`, a `contact_id`, and no `persona_id` or `source`,
  and their `message_ids` are Postgres UUIDs — a different id space from the
  parser hashes in `evals.jsonl`. Running `eval_cli` against
  `mirror_whatsapp_memory_v3` matches nothing. The topic/span path currently has
  no labelled harness; it is checked manually with `memory_cli`.
* **Topic segmentation is over-fine.** 459 topics over 1,801 messages averages
  under 4 messages per topic. The boundary model also proposed 42 indices outside
  the block it was given, which were discarded.
* **Four of 61 blocks returned malformed JSON** and contributed no boundaries, so
  segmentation is coarser in those regions. The spans still index the underlying
  messages.
* **Zero structured facts were extracted.** The extraction rules are deliberately
  conservative and ordinary Hinglish phrasing produced nothing. Every
  fact-shaped question was therefore answered through span retrieval only.
* **Style is under-sampled** — 3 questions.
* **The dense persona model is English-only**, which the results above quantify.
