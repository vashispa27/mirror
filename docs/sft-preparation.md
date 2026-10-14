# Prepare WhatsApp SFT data

One command creates training examples from an **indexed, approved** WhatsApp
chat. Each answer is an actual message written by the selected target author.
The generator can ask an LLM for a plausible incoming query, then pair that
query with the real reply. It never asks the LLM to invent the answer.

The [LLM Twin course dataset generator](https://github.com/decodingai-magazine/llm-twin-course/blob/main/src/feature_pipeline/generate_dataset/generate.py)
uses a related pattern: generate an instruction for authored content and pair it
with that original content. Here we adapt it to dialogue. A real preceding chat
turn is available as the fallback prompt when the generated query is unusable.

## Build the files

After `scripts/onboard-chat.sh index ... --approve`:

```bash
scripts/prepare-sft.sh private-chats/chat-partner-2026 --limit 20
```

The script:

1. Finds replies by the selected author with an actual preceding message.
   Consecutive author messages become one reply; excluded messages break
   pairing. Very short replies and replies over 2,000 characters are skipped.
2. Splits reply turns chronologically into train, validation, and test.
3. For up to 20 **train** turns, asks the local Qwen model for a plausible
   incoming message. If the model rejects its own draft or returns invalid
   output, the real preceding conversation becomes the prompt. Remaining train
   turns, and all validation and test turns, also use real conversation prompts.
4. Writes the JSONL files directly. No second approval or export command is
   required.

Omit `--limit` to try a generated query for every eligible train turn. The
limit counts model calls, including unusable drafts. On Apple Silicon the setup
script installs local Qwen. On another system, provide an OpenAI-compatible
endpoint:

```bash
scripts/prepare-sft.sh private-chats/chat-partner-2026 \
  --api-url http://localhost:8000/v1 --model YOUR_MODEL_NAME --limit 20
```

`TOPIC_MODEL_API_KEY` supplies a bearer token if required. The configured
endpoint receives chat context and real replies. To skip query generation
entirely and train only on actual dialogue prompts:

```bash
scripts/prepare-sft.sh private-chats/chat-partner-2026 --real-only
```

## Inspect or remove examples

The output is under `private-chats/chat-partner-2026/sft/`:

| File | Contents |
| --- | --- |
| `train.jsonl` | Training pairs. Some may have generated incoming queries. |
| `validation.jsonl` | Later real-conversation pairs. |
| `test.jsonl` | Latest real-conversation pairs. |
| `provenance.jsonl` | Original message IDs and prompt type for the generated files. |
| `report.json` | Counts, review hash, and held-out target message IDs. |

Each dataset line has a `messages` array with `system`, `user`, and
`assistant` roles. The assistant content is exactly the approved reply.
**Delete any JSONL line you do not want to train or evaluate on.** You can
also edit a generated user query if it does not fit the real reply. The
provenance file is a snapshot of the initial output; after editing or deleting
dataset rows, it will no longer match line for line.

Roughly 80% of selected turns go to train, 10% to validation, and the latest
10% to test. Chats with fewer than three usable turns produce only train data.
Generated queries are used only in train. If a future evaluation uses the
Qdrant index, exclude the held-out target message IDs in `report.json` from
retrieval so the answer cannot be retrieved verbatim.

The source chat, dataset, and model prompts are private data. The default
`private-chats/` directory is Git-ignored. This command prepares SFT data;
it does not train a model.
