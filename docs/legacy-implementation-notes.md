# Earlier implementation notes

This is the previous README, retained for its detailed CLI and infrastructure
notes. The root README and [onboarding guide](whatsapp-onboarding.md) describe
the current review-first workflow. Relative paths in this archived document
were originally written for the repository root.

# Mirror: WhatsApp persona and memory

Start from a fresh checkout with [WhatsApp onboarding](docs/whatsapp-onboarding.md):

```bash
scripts/setup-workspace.sh
scripts/onboard-chat.sh prepare exports/chat.txt \
  --target-author 'Your name in the export' --owner-author 'Your name in the export' \
  --persona-id your-persona \
  --user-id local-user --contact-id chat-partner \
  --chat-id chat-partner-selected-2026 --timezone Asia/Kolkata
# Review private-chats/chat-partner-selected-2026/review/part-*.jsonl.
scripts/onboard-chat.sh index private-chats/chat-partner-selected-2026 --approve
```

The default processes the most recent 30% of usable text messages; add `--all`
to review the whole chat. Fine-tuning comes after the data and retrieval checks.

## Other infrastructure in this repository

This workspace contains the infrastructure for the first ingestion phase of the
LLM-replica project. Crawler behavior is deliberately not implemented yet.

## Architecture

```text
EventBridge scheduled rule
        |
        v
Crawler Lambda placeholders ----> DynamoDB documents table
                                          |
                                  DynamoDB Streams (CDC)
                                          |
                                          v
                              Stream forwarder Lambda
                                          |
                                          v
                              SQS document-events queue
                                          |
                                  future feature pipeline
```

The `sources` DynamoDB table will hold public-account configurations. Future
crawler Lambdas will read those configurations and write normalized documents to
the `documents` table. DynamoDB Streams provides CDC; the generic forwarder
publishes compact, versioned change notifications to SQS. Scheduled EventBridge
rules are used instead of the newer EventBridge Scheduler API so the complete
stack remains testable with credential-free LocalStack.

## Local prerequisites

- Docker with Docker Compose
- Terraform 1.7 or newer
- AWS CLI
- Python 3.11+ (required for the production WhatsApp index and retrieval path)

No AWS account or real AWS credentials are needed for local development.
The Compose file pins LocalStack 3.8.1 because current LocalStack releases
require an account token for this integrated Lambda/stream feature set. This pin
affects only the emulator; Terraform continues to define ordinary AWS resources.

Qdrant is available separately for local vector-store development. Its dashboard
is at [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

## Production Python environment

The root package is independent from `rag_lab/.venv`, which remains an isolated
learning environment. The production client range matches the pinned Qdrant
server image (1.19.2), avoiding the macOS system-Python client mismatch.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

All root WhatsApp commands and Make targets below use `.venv/bin/python`.

## Start and verify

```bash
make local-up
make tf-apply
make smoke-test
```

The smoke test writes a synthetic document directly to DynamoDB, then verifies
that the resulting stream record is forwarded to SQS. It does not crawl any
website or public account.

## Qdrant

Start Qdrant and confirm its health without starting or stopping LocalStack:

```bash
make qdrant-up
make qdrant-status
```

Run the isolated Qdrant smoke test, which creates a temporary collection,
upserts one vector, queries it, checks the matching point, and removes the
collection:

```bash
make qdrant-smoke-test
```

Stop Qdrant with `make qdrant-down`. Its data is retained in the named
`qdrant-data` Docker volume.

## WhatsApp cleaning

The first implemented source is a user-provided WhatsApp text export. The
cleaning package supports both bracketed iOS records and dash-separated Android
records, including multiline messages, 12/24-hour timestamps, edited notices,
deleted messages, media placeholders, and system messages.

Cleaning is deliberately conservative: it removes WhatsApp export artifacts
but retains casing, punctuation, emoji, typos, and non-ASCII characters because
those are useful style signals. Generic article cleaning and character-based
LangChain splitting are not used for chats. Conversation chunks are assembled
on message boundaries so a message is never split away from itself.

Parse an export to newline-delimited JSON:

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.cli chat.txt cleaned.jsonl \
  --chat-id personal-chat \
  --timezone Asia/Kolkata
```

Produce conversation-aware chunks instead:

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.cli chat.txt chunks.jsonl \
  --chat-id personal-chat \
  --timezone Asia/Kolkata \
  --output-type chunks
```

## WhatsApp persona indexing

The parser/chunker can also build an idempotent, persona-labelled Qdrant index.
This is intentionally a separate production path from `rag_lab`: it uses the
Qdrant client and FastEmbed directly, with no LangChain or lab-package import.

Use the root Python 3.11 environment described above. It installs
`qdrant-client[fastembed]>=1.19,<1.20` directly; no LangChain or LlamaIndex is
used.

Start the existing local Qdrant service (the named `qdrant-data` volume is not
removed by this command):

```bash
make qdrant-up
make qdrant-status
```

List the non-system authors before choosing a target. This only parses the
export; it does not contact Qdrant:

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.index_cli chat.txt \
  --user-id rohan \
  --persona-id rohan-verma \
  --chat-id personal-chat \
  --timezone Asia/Kolkata \
  --list-authors
```

Index a target persona:

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.index_cli chat.txt \
  --user-id rohan \
  --persona-id rohan-verma \
  --target-author "Rohan" \
  --chat-id personal-chat \
  --timezone Asia/Kolkata
```

The target selector is literal text, not a regular expression. It is normalized
with Unicode NFKC, whitespace collapse, and `casefold()`. An exact normalized
match is preferred; otherwise an ordered whole-token prefix can match uniquely,
so `Rohan` resolves to `Rohan Verma`. Ambiguous and unknown selectors
fail with matching/available author names instead of guessing. For the rare
case where an intentional regex selector is helpful, use the separate
`--target-author-regex` option (it must still select exactly one author).

`user_id` is the owner/tenant of uploaded data. `persona_id` is the stable,
opaque application identity for the replica and can span multiple chats whose
source-local display names differ. `target_author` is the canonical display
name resolved inside this specific export, while `requested_target_author` is
the original selector supplied to the index command.

`QDRANT_URL` or `--qdrant-url` overrides the default
`http://localhost:6333`. The default collection is `mirror_whatsapp_v2` and
the payload schema is `whatsapp_persona_v2`. Existing `mirror_whatsapp_v1`
data is intentionally untouched: re-index exports into v2 rather than trying
to migrate or recreate v1.
The indexer creates a missing collection with a `dense` cosine vector using
`BAAI/bge-small-en-v1.5` and an IDF-enabled `sparse` vector using
`Qdrant/bm25`. It never deletes or recreates an existing collection; an
incompatible existing schema fails with an explanation.

Each payload contains the schema version, user/persona/chat/source identity, both the
requested and canonical target author, timestamps/sequences/message IDs, all
authors, complete speaker-labelled text, and separate `context` and
`target_responses` message arrays. Chunks with no usable target-authored
message are not indexed. Qdrant point UUIDs are deterministic across the schema
version, user ID, persona ID, canonical author, chat ID, and source chunk ID. Re-running
the same command therefore upserts the same points, while different users do
not overwrite one another's points.

Run parser/persona unit tests and the isolated Docker-backed indexing smoke
test separately. The smoke test creates and removes only its randomly named
temporary collection; it does not touch production collections.

```bash
make test
make test-rag-lab
make test-whatsapp-index-integration
make test-whatsapp-retrieval-integration
```

Run the parser and chunker tests with `make test`.

## WhatsApp retrieval inspection

The production retriever makes direct scoped Qdrant calls after indexing. Every
search always filters on `user_id`, `persona_id`, `source=whatsapp`, and
`schema_version=whatsapp_persona_v2`; this prevents a tenant or replica from
receiving another tenant/persona's results. An optional `--chat-id` adds a
further scope. Results do not include vectors.

Supported modes are:

- `dense`: BGE semantic retrieval.
- `sparse`: BM25 exact-term retrieval.
- `hybrid_rrf` (default): Qdrant fuses dense and BM25 prefetches with RRF.
- `hybrid_dbsf`: Qdrant fuses dense and BM25 prefetches with DBSF.

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.search_cli \
  "How does Rohan respond when someone is upset?" \
  --user-id user-42 \
  --persona-id rohan-verma \
  --collection mirror_whatsapp_v2 \
  --mode hybrid_rrf \
  --limit 5 \
  --prefetch-limit 30

# Exact rare phrase inspection with BM25
PYTHONPATH=src .venv/bin/python -m whatsapp.search_cli \
  "orchid-velvet-quantum" \
  --user-id user-42 \
  --persona-id rohan-verma \
  --mode sparse \
  --format json
```

Human output is intended for quick inspection:

```text
#1  score=0.750000
Persona: Rohan Verma (rohan-verma)
Chat: personal-chat  Time: 2025-08-08T10:00:00+05:30 → 2025-08-08T10:02:00+05:30
Context:
  Friend: I am feeling upset.
Target responses:
  Rohan Verma: I am here with you.
Provenance: chunk=...; messages=...; target_messages=...; point=...
```

`--format json` emits a stable object with `query`, `mode`, and a `hits` array;
each hit includes score, scope identities, timestamps, message IDs, complete
conversation text, context, target responses, and Qdrant point ID. Empty
results print `No results.` in human mode or `{"hits": [], "mode": ..., "query": ...}`
in JSON mode and are not an error.

## WhatsApp retrieval evaluation

Step 2 of the roadmap compares the four retrieval modes (`dense`, `sparse`,
`hybrid_rrf`, `hybrid_dbsf`) against a small, human-labelled question set. The
harness `whatsapp.eval` and its CLI `whatsapp.eval_cli` are read-only: they
reuse `WhatsAppRetriever` for every search, keep the mandatory
`user_id`/`persona_id` scope filter, and never index, upsert, delete, or
recreate a Qdrant collection. They import no runtime code from `rag_lab` and add
no deduplication, reranking, prompt construction, or LLM generation.

### Dataset format

One JSON object per line:

```json
{"query": "How does he react when a friend is overwhelmed?",
 "user_id": "your-user-id",
 "persona_id": "your-persona-id",
 "expected_target_message_ids": ["<message-id>"],
 "query_type": "paraphrase"}
```

`query`, `user_id`, `persona_id`, and a non-empty
`expected_target_message_ids` list are required; `query_type` is optional
(`exact_phrase`, `paraphrase`, `situation`, `style`, or `semantic`).
`relevant_message_ids` is accepted as an alias. Every row in a file must share
one `user_id`/`persona_id` scope. Labels are **message IDs, not chunk IDs**.

`--evidence-scope` controls which messages in a retrieved result may count:

* `any_author` (default) — any message in the chunk, so a fact stated by
  **either speaker** counts. Use this for memory/fact evaluation.
* `target_only` — only the indexed target author's messages. Use this for
  voice/style evaluation.

See `evals/README.md` for a template and a synthetic example, and
`evals/whatsapp.example.jsonl` for the blank form to fill in.

### Inspect payloads and build 20–50 labels

Message IDs are the parser's `message_id` values and appear verbatim in every
indexed payload's `target_message_ids` and `message_ids` arrays.

```bash
# Read target message IDs straight out of a search result.
PYTHONPATH=src .venv/bin/python -m whatsapp.search_cli "your question" \
  --user-id your-user-id --persona-id your-persona-id \
  --mode sparse --limit 10 --format json

# Or parse the same export with the same chat id/timezone used for indexing
# and read each row's message_id, author, text, and timestamp.
PYTHONPATH=src .venv/bin/python -m whatsapp.cli chat.txt cleaned.jsonl \
  --chat-id personal-chat --timezone Asia/Kolkata
```

Label from the source conversation, independently of any ranking. For a
**memory/fact** suite the expected evidence may be written by **either**
speaker (facts about the other person are often stated by them) and the default
`--evidence-scope any_author` scores it accordingly. For a **voice/style** suite
keep only IDs whose author is the indexed target author and score with
`--evidence-scope target_only`. Mix query types,
spread questions across dates and conversations, exclude ambiguous ones, and
verify each ID exists in the payloads (a missing ID is an indexing-coverage
problem to report, not a label to drop). Do not invent labels with an LLM.

Include both sides of the retrieval trade-off:

* **`exact_phrase` / `style`** deliberately reuse wording, names, or tokens from
  the answer messages. These are the lexical slice BM25 is built for.
* **`semantic`** questions must share **no content words** (no names, places,
  rare nouns, or quoted phrasing) with the answer messages, so the sparse
  retriever cannot score them at all. Only this slice fairly tests the dense
  retriever.
* `paraphrase` / `situation` sit in between: they usually still share several
  distinctive words with the answer messages.

An evaluation made only of lexical questions will make keyword search look
better than it is; one made only of `semantic` questions will make it look
useless. Report both and compare per `query_type`.

### Run the evaluation

`evals/whatsapp.example.jsonl` is a **template**, not a runnable dataset. Copy
it, replace every `REPLACE_WITH_*`/`PASTE_*` placeholder with your own values,
then point `--dataset` at your filled file (keep private datasets in a
Git-ignored location). Running the template is rejected with an explicit
error rather than silently reporting all-zero metrics.

```bash
cp evals/whatsapp.example.jsonl private-chats/my-eval.jsonl
# edit private-chats/my-eval.jsonl with your scope, questions, and message IDs

.venv/bin/python -m whatsapp.eval_cli \
  --dataset private-chats/my-eval.jsonl \
  --collection mirror_whatsapp_v2 \
  --k 5 \
  --prefetch-limit 30 \
  --format table
```

`--format json` prints the configuration, warnings, aggregate metrics,
per-`query_type` aggregates, and per-query details (retrieved chunk IDs, matched
message IDs, ranks, scores, and latency). Optional flags: `--modes`,
`--chat-id`, `--overlap-threshold`, `--evidence-scope`, `--qdrant-url`, and
`--user-id`/`--persona-id` scope overrides that must match the dataset.
Malformed or empty datasets, missing fields, empty label lists, template
placeholders, invalid `k`/prefetch limits, a missing collection, unreachable
Qdrant, and scope mismatches all fail with a clear message before any search
runs.

If a run legitimately returns no hits (for example a wrong `user_id`), every
metric is `0`; the report prints a `WARNING` explaining that the scope filter
matched nothing, so zeros are never mistaken for a real comparison.

Run the Docker-backed synthetic example (indexes its own randomly named
temporary collection from `tests/fixtures/synthetic_whatsapp_chat.txt` with
`evals/whatsapp.synthetic.jsonl`, evaluates all four modes, then removes only
that collection):

```bash
make test-whatsapp-eval-integration
```

### Metric definitions

A result is **relevant** when the message IDs it carries (per
`--evidence-scope`) intersect the question's `expected_target_message_ids`.

* **Success@K** — fraction of questions where at least one top-K result carries
  an expected message.
* **Recall@K** — fraction of a question's expected messages found
  anywhere in the top K. Each expected message is counted once even when it
  appears in several overlapping chunks.
* **MRR@K** — reciprocal rank of the first top-K result carrying an expected
  message (1 at rank 1, 1/2 at rank 2, `0` when none match).
* **nDCG@K** — binary result-level gain `1/log2(rank + 1)` normalised against
  the ideal ordering of up to `min(K, number of expected messages)` relevant
  results.
* **Duplicate rate** — fraction of top-K results whose stored `message_ids`
  cover at least `--overlap-threshold` (default `0.7`) of the smaller of the
  two result's IDs that also appear in a higher-ranked result. Overlap is
  `|intersection| / min(|A|, |B|)`, so a small chunk repeated inside a larger
  one counts as a duplicate, while a single shared boundary message does not.
* **Latency** — wall-clock time of the full read-only `retriever.search` call
  (collection verification plus one Qdrant round-trip), reported per query and
  summarised as mean, median, and p95.

### Interpreting a mode comparison

* `sparse` (BM25) wins on rare exact phrases, names, and unusual tokens but
  misses paraphrases that share no vocabulary.
* `dense` (BGE) wins on paraphrased or situational questions but can drift on
  exact jargon.
* `hybrid_rrf` and `hybrid_dbsf` fuse both candidate pools. RRF rewards items
  ranked by both retrievers; DBSF normalises scores before fusing. Treat the
  difference between them as a tuning signal, not a verdict.
* On a `semantic` question that shares no vocabulary with the answer, `sparse`
  still returns its best lexical guesses. Those confident wrong results can pull
  fusion *below* `dense` alone. Always read the per-`query_type` breakdown, not
  just the overall average.

A small labelled set is **directional**, not definitive: a handful of questions
cannot separate two modes whose metrics differ by a few points, and one hard
question can swing a percentage by tens of points. Use the per-query failure
detail to see *why* a mode missed, and grow the set before drawing conclusions.

Unanswerable questions are intentionally out of scope: with no calibrated score
threshold, top-K retrieval always returns something, so the harness reports no
"no answer accuracy" metric. A future abstention evaluation would need its own
threshold calibration.

## WhatsApp RAG post-retrieval pipeline

The production pipeline keeps retrieval, deduplication, reranking, context
assembly, and generation independently inspectable. It imports no runtime code
from `rag_lab` and does not mutate Qdrant after the first-stage search.

1. `whatsapp.rag.deduplicate_hits` preserves the highest-ranked chunk and
   removes a later chunk when at least `--overlap-threshold` (default `0.7`) of
   the smaller chunk's stored message IDs occurs in it. Hits from different
   chats are always treated as distinct. The first-stage retrieval score and
   all hit metadata stay intact.
2. `LocalCrossEncoderReranker` lazily loads and reuses FastEmbed's
   `Xenova/ms-marco-MiniLM-L-6-v2`. It scores speaker-labelled conversation
   plus target-author responses and records `reranker_score` separately from
   the unchanged `retrieval_score`; neither is a probability.
3. `assemble_context` removes source-message repetition, keeps each example in
   chronological order, and applies a configurable approximate input-token
   budget without slicing a message.
4. `generate_draft` uses an explicit provider. The first adapter is a direct
   OpenAI SDK adapter; retrieved chat is delimited as untrusted reference data.
   Generation happens only for an explicit `--stage generate` command.

Inspect every stage. `context` is a dry run and hides chat text unless
`--show-text` is supplied:

```bash
# Raw Qdrant candidates
PYTHONPATH=src .venv/bin/python -m whatsapp.search_cli "synthetic request" \
  --user-id user-42 --persona-id persona-demo --stage raw

# Candidate pool -> overlap-aware deduplication -> local cross-encoder
PYTHONPATH=src .venv/bin/python -m whatsapp.search_cli "synthetic request" \
  --user-id user-42 --persona-id persona-demo --stage rerank \
  --candidate-count 40 --final-count 8 --overlap-threshold 0.7

# Safe default dry-run context inspection (no chat message text)
PYTHONPATH=src .venv/bin/python -m whatsapp.search_cli "synthetic request" \
  --user-id user-42 --persona-id persona-demo --stage context

# Deliberately reveal text for local inspection only
PYTHONPATH=src .venv/bin/python -m whatsapp.search_cli "synthetic request" \
  --user-id user-42 --persona-id persona-demo --stage context --show-text
```

For a live draft, install the optional adapter and configure credentials only
in the environment (they are never logged):

```bash
.venv/bin/python -m pip install -e '.[generation]'
export OPENAI_API_KEY='...'
export OPENAI_MODEL='your-model-name'
PYTHONPATH=src .venv/bin/python -m whatsapp.search_cli "synthetic request" \
  --user-id user-42 --persona-id persona-demo --stage generate
```

Evaluate a private labelled JSONL file without modifying it. Each line needs a
`query` (or `question`) and one of `relevant_message_ids`,
`relevant_chunk_ids`, `relevant_point_ids`, `relevant_ids`, or `relevant`.
The report compares the same `K` for hybrid RRF, deduplicated hybrid RRF, and
deduplicated/reranked hybrid RRF. It prints Success@K, Recall@K, MRR@K, nDCG@K,
duplicate rate, mean latency, and regression query indexes/ranks only—never
query or chat excerpts.

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.evaluation_cli /absolute/path/labels.jsonl \
  --user-id user-42 --persona-id persona-demo --k 8 --candidate-count 40
```

The default cross-encoder is English-oriented and can rank informal shorthand,
code-switching, and multilingual WhatsApp text poorly. Compare the printed
metrics before adopting it; a reranker is not assumed to improve every query.

Inspect local resources with the AWS CLI:

```bash
AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=ap-south-1 \
  aws --endpoint-url=http://localhost:4566 dynamodb list-tables

AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test AWS_DEFAULT_REGION=ap-south-1 \
  aws --endpoint-url=http://localhost:4566 sqs list-queues
```

Stop LocalStack:

```bash
make local-down
```

Local AWS resources are intentionally ephemeral. After restarting LocalStack,
run `make tf-apply` again to recreate them from Terraform.

To remove the Terraform resources first:

```bash
make tf-destroy
make local-down
```

## Contact-scoped WhatsApp memory

The memory path stores every parsed export message in Postgres, including system,
deleted, and media records. The current `process` path uses local Qwen3.5-4B to
mark broad topic boundaries in 30-message blocks, carrying earlier topic labels
and nearby message context across blocks. There is no time-gap split. Each topic
gets a short local-model summary. Each topic segment gets overlapping, speaker-
labelled spans of up to 10 messages and about 2,400 characters. The summary and
the original-message span text are embedded separately in Qdrant's
`mirror_whatsapp_memory_v3` collection using multilingual MiniLM dense vectors
and BM25 sparse vectors. Every span points to its topic and to raw Postgres
messages. Search checks summaries and spans, expands matching topics to their
relevant spans, and loads source messages. The existing persona and v2 memory
collections are untouched. Postgres migrations 001 and 002 run at CLI startup.
Local Qwen responses are cached by prompt and chat, so reprocessing an unchanged
export reuses boundary decisions and summaries. Changed context is reclassified.

The earlier MiniLM candidate/Qwen embedding verifier remains available through
`process_chat_memory_v2` for comparison. Topic boundaries are still fallible,
especially for short subtopics and ambiguous returns. The synthetic regression
script below shows those errors before any real export is processed.

L0 starts empty for a new bot conversation. It reads only messages saved by the
live bot (`source_platform='whatsapp_live'`) in the current contact/chat;
exported history remains in Postgres as evidence and in Qdrant as long-term
memory. Each reply saves the incoming message before reading L0. The caller
should save outgoing replies with `MemoryRepository.add_message` so both sides
appear in later L0 windows.

```bash
.venv/bin/python -m pip install -e '.[topic-local]'
make memory-up
PYTHONPATH=src .venv/bin/python evals/run_hinglish_topic_regression.py
PYTHONPATH=src .venv/bin/python evals/run_topic_memory_smoke.py

PYTHONPATH=src .venv/bin/python -m whatsapp.memory_cli \
  preview-topics /absolute/path/to/export.txt --timezone Asia/Kolkata

PYTHONPATH=src .venv/bin/python -m whatsapp.memory_cli \
  --user-id owner-1 --contact-id contact-1 --chat-id chat-1 \
  import /absolute/path/to/export.txt --owner-author 'My export name' \
  --timezone Asia/Kolkata

PYTHONPATH=src .venv/bin/python -m whatsapp.memory_cli \
  --user-id owner-1 --contact-id contact-1 --chat-id chat-1 \
  search "What color shirt did I buy?"

PYTHONPATH=src .venv/bin/python -m whatsapp.memory_cli \
  --user-id owner-1 --contact-id contact-1 --chat-id chat-1 \
  ask "What is Mom's birthday?" --author 'Contact export name'
```

Use `import ... --defer` to persist raw messages immediately and process
finalized topics and spans later with the `process` subcommand. `preview-topics`
does not access Postgres or Qdrant; `search` reads evidence without saving a
live message. `ask` persists the incoming
message before loading recent messages. It answers supported exact facts
locally; add `--generate` with `OPENAI_API_KEY` and `OPENAI_MODEL` configured to
draft replies from semantic context. The callable functions are
`ingest_export`, `process_chat_memory`, and `reply_to_message` in
`src/whatsapp/memory_service.py`. Outgoing messages from an external transport
can be stored with `MemoryRepository.add_message(..., direction="outgoing")`.
No live WhatsApp transport is included.

An export hash plus source sequence makes exact reimports idempotent. Across
partial or changed exports, an exact timestamp, author, and original-content
fingerprint is reused only when unique on both sides. Identical repeated
messages remain distinct; an ambiguous partial export may conservatively
produce a second record instead of merging two real messages. Rebuilding from
the full stored chat retires old topic and span points after new points are indexed.
Facts are append-only in V1, so conflicting values trigger clarification.

```bash
make test-whatsapp-memory-integration
WHATSAPP_SEMANTIC_INTEGRATION=1 PYTHONPATH=src .venv/bin/python -m unittest tests/test_whatsapp_semantic_topics.py
```

`evals/whatsapp_memory.synthetic.jsonl` labels exact, contextual, semantic,
lexical, and no-evidence questions used by the integration test. The extractor
currently recognizes explicit birthdays, event dates, allergies, phone
numbers, street addresses, and passport/employee/membership IDs. Other facts
remain in searchable conversation chunks until an extractor is added.

## Later AWS deployment

The Terraform uses ordinary AWS resources. For a real deployment, create a
separate state backend/workspace and apply with `use_localstack = false`. Do not
reuse the local Terraform state for AWS.

Before deployment, review:

- AWS region and resource names
- DynamoDB point-in-time recovery and deletion protection
- CloudWatch log retention
- crawler schedules and payloads
- IAM permissions
- remote Terraform state and locking

`infra/environments/aws.tfvars.example` contains a starting configuration.

## Deferred decisions

These choices belong to crawler implementation and are intentionally not fixed
by the infrastructure:

- supported public-account platforms
- official API versus browser/page crawling per platform
- authentication and rate-limit strategy
- raw and cleaned document schemas
- crawl cursors and backfill behavior
