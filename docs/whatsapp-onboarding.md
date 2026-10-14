# WhatsApp onboarding from a fresh checkout

This workflow parses a WhatsApp `.txt` export, lets you review the selected
messages, then builds two separate Qdrant indexes and a Postgres memory store.
Fine-tuning is a later stage; this workflow does not train or serve a model.

## 1. Set up the workspace

You need Python 3.11 or newer and Docker with Compose. On Apple Silicon macOS,
the setup script installs the local Qwen topic-model dependency. On other
systems, provide a chat-completions model endpoint at indexing time.

```bash
git clone YOUR_REPOSITORY_URL
cd mirror
scripts/setup-workspace.sh
```

The script creates `.venv`, installs the package, starts Postgres and Qdrant,
and applies the database migrations. It does not start LocalStack or require
Terraform. If services are already managed elsewhere, run
`scripts/setup-workspace.sh --no-services` and set `POSTGRES_DSN` and
`QDRANT_URL` in your shell.

## 2. Parse and select messages for review

Export an individual WhatsApp chat as a `.txt` file. Keep it under `exports/`
or another private directory; `exports/` and `private-chats/` are Git-ignored.
Run:

```bash
scripts/onboard-chat.sh prepare exports/chat.txt \
  --target-author 'Your exact name in the export' \
  --persona-id your-persona \
  --owner-author 'Your exact name in the export' \
  --user-id local-user \
  --contact-id chat-partner \
  --chat-id chat-partner-selected-2026 \
  --timezone Asia/Kolkata
```

`--target-author` is the person whose **style** will eventually be fine-tuned.
`--owner-author` identifies the account owner's messages for Postgres direction
labels; it may be a different person. Use a fresh `--chat-id` for each reviewed
dataset. The target name is resolved against actual export authors and ambiguity
is rejected.

The default selects the **most recent 30% of usable text messages**, in
chronological order, including both speakers. It parses the entire export
first. Use `--all` to select every usable text message, or `--fraction 0.5` for
the most recent half. `--month-first` is available for month-first exports.
Selection is by date, not by an AI quality score.

The command creates `private-chats/<chat-id>/` with:

| File | Purpose |
| --- | --- |
| `manifest.json` | Source hash, identities, selection policy and counts. |
| `cleaned_all.jsonl` | Every parsed and cleaned message, including records excluded from selection. |
| `review/part-*.txt` | Human-readable selected messages, 250 per file by default. |
| `review/part-*.jsonl` | Editable review decisions for the same messages. |

## 3. Review the selected dataset

Read the numbered `.txt` parts. In the matching `.jsonl` parts, change
`"include": true` to `false` to exclude a message, or edit `"text"` to redact
content. Keep every JSONL row and leave IDs, author, timestamp and sequence
untouched. The indexer reads the JSONL files. Nothing is indexed before this
review step.

The full original export stays at the path you provided. `cleaned_all.jsonl`
and the review files are private local data; do not commit or publish them.

## 4. Build Postgres memory and both vector indexes

```bash
scripts/onboard-chat.sh index private-chats/chat-partner-selected-2026 --approve
```

`--approve` confirms that the JSONL review is complete. The indexer reads only
included messages, stores them with source IDs in Postgres, builds topic
summaries and linked raw-message spans in `mirror_whatsapp_memory_v3`, and
builds the selected author's style index in `mirror_whatsapp_v2`.

On a system without Apple Silicon, supply a configured chat-completions
endpoint and model:

```bash
scripts/onboard-chat.sh index private-chats/chat-partner-selected-2026 \
  --approve --topic-api-url http://localhost:8000/v1 --topic-model YOUR_MODEL_NAME
```

Chat text is sent to the endpoint you configure. Use a local endpoint if the
chat must stay on your computer. `TOPIC_MODEL_API_KEY` can supply its token.
Model responses are cached in Postgres so an interrupted run can resume without
repeating completed calls. Re-running the same approved workspace is
idempotent. Once indexing begins, changing the review requires a new workspace
and chat ID, because earlier indexed messages cannot safely be overwritten.

The run writes `index_report.json` with message, topic, span and persona point
counts. On a large chat, local topic classification and embedding can take a
long time; progress is printed as blocks and summaries are processed.

## 5. Check retrieval before training

Memory retrieval returns the original messages that support a result:

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.memory_cli \
  --user-id local-user --contact-id chat-partner \
  --chat-id chat-partner-selected-2026 \
  search 'What did we decide about the birthday?'
```

Style retrieval uses the target persona index:

```bash
PYTHONPATH=src .venv/bin/python -m whatsapp.search_cli \
  'How did I usually reply when plans changed?' \
  --user-id local-user --persona-id your-persona \
  --chat-id chat-partner-selected-2026 --stage raw
```

Choose 10–20 answerable questions from messages across the selected period
*before* seeing search results. Record the expected source message IDs, then
check which are returned. Include short exact details, paraphrases, topic
returns and questions whose answer should be absent. This is the retrieval
quality gate before preparing fine-tuning examples.

## What remains for a complete assistant

1. Turn approved target-author replies and preceding context into training and
   validation examples, with contact and private-data review. Keep a held-out
   set for measuring whether fine-tuning improves the style over retrieval alone.
2. Add the fine-tuning job and model selection. This comes after data review and
   retrieval evaluation so training does not hide indexing errors.
3. Configure a reply model and a live chat transport. The repository currently
   offers CLI retrieval and callable reply functions; it does not connect to a
   live WhatsApp account.
4. Before publishing the project, choose a license and publish only synthetic
   examples. The local Docker credentials in Compose are for development.
