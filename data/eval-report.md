# Retrieval evaluation report

Tests the **persona/style index** (`mirror_whatsapp_v2`, 638 chunks) against the
hand-written eval set (`data/evals.jsonl`, 188 answerable questions). Raw reports:
`data/eval-run-any-author.json`, `data/eval-run-target-only.json`.

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.eval_cli \
  --dataset data/evals.jsonl --k 5 --evidence-scope any_author --format json \
  > data/eval-run-any-author.json
```

## Which index do these numbers describe

Only the persona index. The eval harness filters Qdrant on
`persona_id` + `source` + `schema_version`, and labels answers with the parser's
32-hex `message_id`. The memory index (`mirror_whatsapp_memory_v3`) carries
`schema_version: memory-v3`, no `persona_id`, and UUID message ids from Postgres,
so this harness cannot search it. Memory behaviour is checked by hand with
`memory_cli`.

## Results (K=5, 188 queries)

### Memory/fact suite — `--evidence-scope any_author`

Either speaker's message counts as evidence. This is the right scope for fact,
attribution and self-disclosure questions, because an answer is valid whichever
speaker gave it.

| mode | success@5 | recall@5 | mrr@5 | ndcg@5 | median latency |
| --- | --- | --- | --- | --- | --- |
| **hybrid_dbsf** | **0.7553** | 0.7553 | **0.6356** | **0.6787** | 11.4 ms |
| hybrid_rrf | 0.7553 | 0.7553 | 0.5980 | 0.6504 | 10.3 ms |
| sparse (BM25) | 0.7394 | 0.7394 | 0.6273 | 0.6702 | 6.3 ms |
| dense | 0.5000 | 0.5000 | 0.3618 | 0.3988 | 10.0 ms |

### Voice suite — `--evidence-scope target_only`

Only the modelled author's messages count. The ceiling is 72.3%, not 100%,
because 52 of the 188 labelled answers were written by the other speaker and can
never match here.

| mode | success@5 | mrr@5 | ndcg@5 |
| --- | --- | --- | --- |
| hybrid_dbsf | 0.5372 | 0.4320 | 0.4715 |
| hybrid_rrf | 0.5372 | 0.4123 | 0.4563 |
| sparse | 0.5160 | 0.4290 | 0.4656 |
| dense | 0.3298 | 0.2333 | 0.2598 |

### The semantic slice (zero content-word overlap)

The 34 questions that share no content words with their answer. BM25 cannot score
these at all, so this is the only fair test of the dense retriever.

| mode | any_author | target_only |
| --- | --- | --- |
| hybrid_dbsf | **0.441** | **0.412** |
| hybrid_rrf | 0.382 | 0.382 |
| dense | 0.324 | 0.294 |
| sparse | 0.235 | 0.206 |

## By scenario (hybrid_dbsf, any_author)

| scenario | hit/total | rate | misses |
| --- | --- | --- | --- |
| 01-long-single-topic | 17/20 | 0.85 | 3 |
| 02-diverge-return | 20/24 | 0.83 | 4 |
| 08-edge-cases | 25/30 | 0.83 | 5 |
| 07-emotional-support | 16/20 | 0.80 | 4 |
| 03-self-disclosure | 18/24 | 0.75 | 6 |
| 06-time-gaps | 16/22 | 0.73 | 6 |
| 04-disagreement | 15/24 | 0.62 | 9 |
| 05-changing-plans | 15/24 | 0.62 | 9 |

## By query type (hybrid_dbsf, any_author)

exact_phrase 0.914 · paraphrase 0.865 · multihop 0.857 · situation 0.846 ·
attribution 0.722 · temporal 0.684 · semantic 0.441 · style 0.333 (n=3).

Lexical recall is essentially solved; the loss is concentrated in questions that
require reconciling several turns.

## What the numbers say

1. **Hybrid fusion wins.** `hybrid_dbsf` is best or joint-best overall and clearly
   best on the semantic slice (0.441 vs dense 0.324, sparse 0.235).
2. **Sparse is a very strong baseline.** BM25 alone reaches 0.739, within 1.6
   points of the best hybrid. Fusion earns its place mainly on the semantic tail.
3. **Dense retrieval is the weak link.** `BAAI/bge-small-en-v1.5` is an
   English-only model being run over Hinglish. It loses to BM25 outright (0.500
   vs 0.739) and is the main reason the semantic slice sits at 0.44 rather than
   higher. Swapping in a multilingual encoder (for example
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which the
   memory index already uses) is the obvious next experiment.
4. **The hard scenarios are the ones that need across-turn reasoning.**
   Disagreement and changing-plans both sit at 0.62, and each contributes 9 of the
   69 total misses. These are exactly the cases the hand-written set was built to
   probe.
5. **Style is the least-measured slice** (3 questions). More style questions are
   needed before drawing conclusions about the voice index.

## Everything is reproducible

```bash
.venv/bin/python scripts/build-dataset.py build --limit 600 --seed 7
.venv/bin/python scripts/check-review.py
.venv/bin/python scripts/merge-review.py

scripts/setup-workspace.sh
scripts/onboard-chat.sh prepare data/export.txt \
  --target-author 'Rohan' --owner-author 'Rohan' \
  --persona-id hinglish-twin --user-id local-user \
  --contact-id aditi --chat-id hinglish-chat \
  --timezone Asia/Kolkata --all
scripts/onboard-chat.sh index private-chats/hinglish-chat --approve

PYTHONPATH=src .venv/bin/python -m whatsapp.eval_cli \
  --dataset data/evals.jsonl --k 5 --evidence-scope any_author --format table
.venv/bin/python scripts/summarize-eval.py \
  data/eval-run-any-author.json data/evals.jsonl
```
