# Mirror

**Create your twin, grounded in your actual conversations.**

Mirror starts with a WhatsApp text export. It preserves the messages behind each
memory, indexes conversations from both speakers, and keeps a separate set of
examples of the person whose voice you want to model. You can inspect what was
selected, remove or redact messages, build the indexes, and ask where a retrieved
answer came from.

Today this repository provides **reviewed ingestion, retrieval, and SFT data
preparation**, not a finished chat bot. It does not connect to a live WhatsApp
account or train a model. The index is the foundation for a twin that can later
speak in your style while staying grounded in what you and your contacts
actually said.

## Start from a WhatsApp export

You need Python 3.11+, Docker with Compose, and a WhatsApp `.txt` export. The
default topic model runs locally on Apple Silicon. On other systems, provide an
OpenAI-compatible chat-completions endpoint when indexing.
`setup-workspace.sh` creates `.venv`, installs the Python package, starts
Postgres and Qdrant, and applies the Postgres migrations. Use
`scripts/setup-workspace.sh --no-services` if you manage those services
yourself. The first local topic run may download its model weights.

```bash
scripts/setup-workspace.sh

scripts/onboard-chat.sh prepare exports/chat.txt \
  --target-author 'Your name in the export' \
  --owner-author 'Your name in the export' \
  --persona-id my-twin \
  --user-id local-user \
  --contact-id chat-partner \
  --chat-id chat-partner-2026 \
  --timezone Asia/Kolkata
```

`--target-author` is the person whose style you want to index.
`--owner-author` identifies your own messages for direction labels and can be
different. `--user-id` owns the data, `--contact-id` names the person on the
other side of this chat, and `--persona-id` is the stable identity of the style
index. Choose a new `--chat-id` for each reviewed dataset.

`prepare` parses the **whole export** and selects the **most recent 30% of
usable text messages**, including both speakers, in chronological order. Use
`--all` to review the full usable export, or `--fraction 0.5` for the
newest half. It does not write to Postgres or Qdrant.

Read `private-chats/chat-partner-2026/review/part-*.txt`. In the matching
`part-*.jsonl` files, set `"include": false` to exclude a message or edit
`"text"` to redact it. Keep the IDs, author, timestamps, and row order intact.
Then build the indexes:

```bash
scripts/onboard-chat.sh index private-chats/chat-partner-2026 --approve
```

The command stores only the **approved selected messages** in Postgres and
indexes them in Qdrant. It writes `index_report.json` with the counts. The
complete parsed export remains in the private workspace as `cleaned_all.jsonl`;
it is not indexed when you choose a fraction. Use a fresh `--chat-id` for each
reviewed dataset. Repeating an unchanged index run is idempotent; changing the
review after indexing starts requires a new workspace and chat ID.

On a system without Apple Silicon, point the topic classifier at a model you
run or trust:

```bash
scripts/onboard-chat.sh index private-chats/chat-partner-2026 --approve \
  --topic-api-url http://localhost:8000/v1 \
  --topic-model YOUR_MODEL_NAME
```

Chat text is sent to the endpoint you configure. See the
[onboarding guide](docs/whatsapp-onboarding.md) for setup options and review
file details.

## Why the architecture has two indexes

Someone's writing style and a conversation's history are different kinds of
evidence. A reply about a birthday may be needed even if the person being
modeled did not write it; a style example is useful only if they did. Mirror
therefore keeps these paths separate:

```text
WhatsApp .txt export
  → parse and conservatively clean → private review files
  → approved messages → Postgres source messages
       ├─ exact assertions → Postgres facts
       ├─ topic detection → summaries + linked message spans
       │                    → Qdrant mirror_whatsapp_memory_v3
       └─ target-author conversation chunks
                            → Qdrant mirror_whatsapp_v2
```

| Area | Stores | Why |
| --- | --- | --- |
| Postgres `messages` | Approved messages, original and normalized text, authors, timestamps, source IDs, import provenance | Source of truth and evidence behind retrieved results. |
| Postgres `facts` | Conservatively extracted exact assertions with source message IDs | Dates, numbers, and identifiers need keyed lookup rather than approximate vector ranking. |
| Qdrant `mirror_whatsapp_memory_v3` | Topic summary points and linked raw-message span points from **both speakers** | Find an episode broadly, then recover the precise messages within it. |
| Qdrant `mirror_whatsapp_v2` | Conversation chunks containing a response from the selected author | Retrieve examples of that person's wording and behavior. |

Every memory is scoped by `user_id` and `contact_id`; `chat_id`
identifies the conversation. `persona_id` identifies the style being modeled
and is not a replacement for `contact_id`. Derived records retain source
message IDs so a retrieved claim can be checked against Postgres. Stable IDs
make retries safe.

### Cleaning and human review

The parser handles common iOS and Android WhatsApp text formats, multiline
messages, and export artifacts. Cleaning removes those artifacts but retains
the casing, punctuation, emoji, typos, and Hinglish wording that carry style.
System, deleted, and media-placeholder messages are excluded from usable text
indexing. Author selection resolves against names actually present in the
export and rejects ambiguous matches.

The review step is intentional: a private conversation can contain material
that should never become retrieval data. `exports/`, `private-chats/`, and
`.env` are Git-ignored. Keep real chats out of commits and public examples.

### Topic memory: broad route, precise evidence

Memory chunking follows **topic changes**, not a fixed time gap. The local
Qwen3.5-4B model classifies boundaries in 30-message blocks. Each prompt also
shows up to six previous and four following messages, plus the active topic,
so vague replies and block edges are less likely to create false boundaries.
A returning subject can resume an earlier topic. Model responses are cached in
Postgres for retries.

Each topic gets a short factual summary. The topic point embeds its **label and
summary** as a broad search route. Within each topic run, overlapping spans
contain up to **10 whole messages**, with **2 messages of overlap** and a
roughly **2,400-character** cap. Span points embed the speaker-labelled raw
messages, not another summary. Each span links to its topic and its exact
source message IDs. A long topic can therefore have one broad route and many
searchable, detail-rich spans.

For example, a weather discussion may mention a red jacket in one span and a
heater in another. A question about the jacket can match that span directly;
matching the weather summary can also lead search into its child spans. Topic
boundaries and summaries are model judgments, so review retrieval on your own
chat before treating them as reliable facts.

### Persona examples

The persona path retains both sides of each conversation chunk but indexes only
chunks with at least one reply from `--target-author`. Its existing chunker
uses up to 20 whole messages, 4 messages of overlap, a 4,000-character cap,
and a 120-minute session gap. The gap applies **only to persona chunks**;
topic memory does not end a topic because someone replied late. Persona chunks
are examples for style retrieval, not a source of contact facts.

### Models and search choices

| Stage | Current model or method | Reason |
| --- | --- | --- |
| Topic boundaries and summaries | `mlx-community/Qwen3.5-4B-MLX-4bit` locally on Apple Silicon; configurable chat-completions model elsewhere | A small instruction model can read several Hinglish turns and return explicit boundary indices and summaries. |
| Topic and span dense vectors | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Multilingual semantic matching for Hinglish and paraphrases. |
| Persona dense vectors | `BAAI/bge-small-en-v1.5` | Existing style index model; its Hinglish quality still needs evaluation. |
| Sparse vectors in both indexes | `Qdrant/bm25` | Names, unusual words, and exact phrases that dense vectors may miss. |
| Fusion | Qdrant reciprocal rank fusion (RRF) by default | Combines dense and sparse candidates without assuming their scores are comparable. |
| Optional persona reranker | `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder | Scores a question and candidate together to reorder retrieved style examples. This is not part of memory search. |

## What happens when you search memory

1. Read the current contact's **L0**, the recent messages written through the
   live reply path. Export backfill does **not** populate L0; it starts empty.
2. Search Qdrant topic summaries and spans with dense + BM25 RRF, scoped to the
   owner and contact. Expand winning topics to child spans and remove duplicate
   span hits. In parallel, look up candidate exact facts in Postgres.
3. If the retrieved conversation reveals a relevant subject, make a follow-up
   keyed fact lookup. Assemble a bounded context from L0, facts, spans, and
   their source messages. Keep conflicting exact values visible.

The current query handling appends recent context for some pronoun-heavy
questions. It does **not** yet generate several rewritten queries. Memory
search also does **not** run the optional cross-encoder. Exact fact extraction
uses cautious rules for a small set of forms such as birthdays, event dates,
allergies, phone numbers, addresses, and IDs; ordinary Hinglish phrasing may
produce zero structured facts. The underlying messages remain available through
topic/span retrieval.

Inspect memory evidence without writing a new live message:

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.memory_cli \
  --user-id local-user --contact-id chat-partner \
  --chat-id chat-partner-2026 \
  search 'What did we decide about the birthday?'
```

Inspect style examples separately:

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.search_cli \
  'How did I respond when plans changed?' \
  --user-id local-user --persona-id my-twin \
  --chat-id chat-partner-2026 --stage raw
```

For persona search, `--stage dedup` removes overlapping chunks and
`--stage rerank` adds the optional cross-encoder. A cross-encoder reranks
retrieved candidates; it cannot find a message missing from the first search.
Its current English-focused model should be tested on Hinglish examples before
relying on it.

## Check retrieval on your data

Before using the index to answer personal questions, choose 10–20 answerable
questions from the reviewed messages and record the expected source message
IDs **before** looking at results. Include exact details, paraphrases, topic
returns, and questions with no answer in the selected data. Check whether the
source messages appear in the returned evidence. The
[evaluation guide](evals/README.md) describes the existing labelled persona
retrieval harness; the [onboarding guide](docs/whatsapp-onboarding.md) has a
short manual memory check.

Run the labelled harness against the persona index:

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.eval_cli \
  --dataset data/evals.jsonl \
  --k 5 \
  --evidence-scope any_author \
  --format table
```

`--evidence-scope any_author` accepts an answer written by either speaker, which
is what fact and attribution questions need. Switch to
`--evidence-scope target_only` to score only the persona's own replies. Add
`--format json` to capture the full per-query report. Every row in
`data/evals.jsonl` carries the ids of the messages that answer it, so the
command needs no manual labelling.

[REPORT.md](REPORT.md) is a worked example: the Hinglish dataset it was run on,
how that data and its eval set were built, how the two indexes are laid out and
why, which models run and how long each stage takes, and the full result tables.

Run the WhatsApp unit tests with:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_whatsapp*.py' -q
```

## Prepare examples for future model training

The [SFT preparation guide](docs/sft-preparation.md) shows how one command
extracts real target-author replies, optionally drafts matching queries with
an LLM, and writes chronological train/validation/test files. You can delete
unwanted JSONL examples afterward. The reply in every pair remains the
approved message the person actually wrote. This stage prepares data only;
it does not run fine-tuning.

## Scope and project status

The supported path starts from a user-provided WhatsApp `.txt` export and ends
with reviewed Postgres and Qdrant data, CLI retrieval, and optional SFT dataset
preparation. The repository has callable reply functions, but no live WhatsApp
transport or training job. A grounded assistant and any future model training
need their own evaluation before they become part of this walkthrough.

The repository also contains earlier LocalStack/Terraform ingestion work and
standalone retrieval experiments. They are separate from the setup above; the
previous README content is preserved in
[implementation notes](docs/legacy-implementation-notes.md).
