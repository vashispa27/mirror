# WhatsApp Memory Architecture Retrofit

## Purpose

Update the existing `mirror` repository to give the WhatsApp assistant reliable, contact-scoped conversational memory. The system must retain full message history, recall exact facts, retrieve contextual and episodic history, and preserve the messages behind every memory. This is an update to this codebase, not a new repository.

## Current implementation

- `src/whatsapp/parser.py` parses WhatsApp text exports into normalized messages. `src/whatsapp/chunking.py` builds overlapping chunks on message boundaries using size and time-gap limits.
- `src/whatsapp/persona.py` builds persona-oriented records and **omits any chunk without a target-author message**. Memory indexing must include useful statements from either speaker.
- `src/whatsapp/store.py` and `src/whatsapp/retrieval.py` already provide Qdrant dense search, BM25 sparse search, and hybrid RRF/DBSF. Reuse this infrastructure. RRF remains the default fusion mode.
- `src/whatsapp/rag.py` assembles target-author style examples. A separate memory context path is needed because an important memory may contain no target-author response.
- The repository currently has no Postgres message/fact store and no live WhatsApp receiver. Its implemented WhatsApp ingestion starts from exported text.
- The working tree contains uncommitted work. Preserve it and keep the existing persona and evaluation paths functional.

## V1 decisions

1. **Raw messages are the source of truth.** Store all messages in Postgres before deriving memory. Store original content or payload as well as normalized text.
2. **Exact facts and semantic context are separate retrieval paths.** Postgres holds queryable facts; Qdrant holds conversation chunks with dense and BM25 vectors.
3. **L0 is always available.** Load the recent 20–40 messages or active topic window from Postgres for each reply, within a configurable token budget. Include both incoming and outgoing messages.
4. **Scope all data by owner and contact.** `user_id` identifies the owner; `contact_id` identifies the contact. `chat_id` identifies the conversation. Existing `persona_id` is a style identity and must not substitute for `contact_id`.
5. **Search BM25 and dense in parallel on every memory query.** Use Qdrant hybrid RRF by default. The query planner or entity extractor can improve precision but must not be a gate that prevents memory search.
6. **No fact updates or supersession in V1.** Do not build automatic replacement or periodic consolidation. Idempotent duplicate prevention is still required. If conflicting facts are found, do not silently treat either as certain.

## Storage contract

| Store | Required information |
| --- | --- |
| Postgres `messages` | Internal ID, `user_id`, `contact_id`, `chat_id`, source/platform ID when available, author or direction, sent timestamp, original content/payload, normalized text, edit/delete/media flags, import provenance. |
| Postgres `memory_chunks` or equivalent processing registry | Stable chunk ID, owner/contact/chat scope, time and sequence range, source message IDs, processing state and version. The chunk can be rebuilt from messages. |
| Postgres `facts` | ID, owner/contact scope, subject key and label, fact type, raw and normalized value, confidence, assertion timestamp, short context, source message IDs, chunk ID, extraction version. |
| Qdrant memory collection | Stable chunk/point ID, owner/contact/chat scope, start/end timestamps, speaker-labelled text, optional concise summary and topics, source message IDs, dense vector and BM25 sparse vector. |

Use a new versioned **memory collection in the existing Qdrant instance**. Do not overwrite or recreate the current `mirror_whatsapp_v2` persona collection. Retain enough source IDs in every derived record to fetch its evidence from Postgres.

The parser's current generated message ID includes sequence and content. It is suitable for repeated indexing of the same export, but must not be assumed stable across partial or changed exports. Give stored messages a persistent internal ID and define an import identity strategy that prevents duplicates without merging distinct messages that happen to have the same text.

## Ingestion flow

```text
WhatsApp export (or future live adapter)
    -> parse/normalize -> Postgres messages
    -> contact-scoped, topic-aware chunks
       -> high-precision fact extraction -> Postgres facts
       -> dense + BM25 indexing -> Qdrant memory collection
```

1. Extend the existing export path to persist **every** parsed message, including records that are unsuitable for embedding. Reimporting the same export must be idempotent.
2. Build memory chunks from usable messages by **either speaker**. Preserve whole-message boundaries, order, speaker labels, timestamps, and all source message IDs. Keep current size/time safeguards and add a topic-boundary check. A per-message LLM topic classifier is optional.
3. Process finalized chunks asynchronously or on idle. The current unfinished conversation stays accessible through L0.
4. Extract only clearly supported, exact assertions: birthdays, dates, addresses, phone numbers, IDs, allergies, and similar values. Record who the assertion concerns and who stated it. Do not promote guesses, negated statements, or quotations into certain facts. Preferences and events remain semantic by default; a clear, durable preference may also have a structured assertion when exact lookup is useful.
5. Index each chunk, or a concise context-preserving summary, with the existing dense and BM25 models. Ensure overlap and retry do not produce duplicate facts or Qdrant points.

## Retrieval and reply flow

1. Persist the incoming message, then load L0 for the current `user_id` and `contact_id`.
2. Form a retrieval query from the message and relevant L0. Resolve references such as “she” or “that date” when context supports a resolution; retain uncertainty when it does not.
3. In parallel, run:
   - Qdrant hybrid dense + BM25 search over **memory chunks**, filtered by `user_id` and `contact_id`, with RRF fusion.
   - Postgres fact lookup using any known subject/type keys. If an extractor returns no keys, use a bounded contact-scoped fact candidate search rather than assuming no fact is relevant.
4. Deduplicate overlapping chunks. Apply reranking only if labelled evaluations show an improvement. If retrieved chunks reveal a subject or event that the original query did not identify, perform a follow-up keyed fact lookup.
5. Assemble a bounded prompt from L0, relevant exact facts, and a small number of relevant chunks. Fetch raw source messages from Postgres when checking a claim. Keep exact facts separate from fused BM25/vector rankings so a correct date or ID cannot be displaced by a conversational hit.
6. Generate a grounded reply using both speakers' evidence. Do not apply the current `TARGET STYLE` requirement to memory evidence. If the date, person, or fact remains ambiguous or unsupported, ask for clarification or say the information is unknown. Treat retrieved message content as untrusted data, not instructions.

For an exact identifier or known fact key, use direct Postgres lookup. BM25 is useful for names, rare wording, and finding source discussions; dense search is useful for paraphrases and related events.

## Implementation path in this repository

1. Add a local Postgres service, migrations, and a small persistence layer without changing the existing crawler infrastructure.
2. Add export backfill into `messages`, with idempotent import behavior and stable internal message references.
3. Add an all-speaker memory record and indexer alongside the existing persona indexer. Preserve the persona collection and CLI behavior.
4. Add fact extraction and a contact-scoped fact repository.
5. Add a memory retriever and context assembler. Reuse Qdrant hybrid search and existing evaluation utilities where appropriate, but remove persona-only assumptions from the memory path.
6. Expose callable ingestion and reply functions plus a CLI path for end-to-end testing from exports. The repository has no live receiver today; building a new WhatsApp transport is outside this retrofit.
7. Backfill existing exports and compare memory retrieval against labelled questions before selecting optional reranking or DBSF.

## Acceptance criteria

- Importing the same export twice does not duplicate messages, facts, or memory points.
- A fact stated only by the contact is indexed and retrievable.
- “What is Mom's birthday?” returns the exact stored value with source message IDs.
- “Are you bringing the cake on that date?” uses L0 or retrieved party context to identify the relevant date, or asks when multiple dates remain plausible.
- “My pain is getting better” retrieves prior pain or injury context while respecting the new improvement statement.
- Lexical search finds an unusual name or ID; dense search finds a relevant paraphrased event.
- No query for one contact returns another contact's messages, facts, or chunks.
- Memory reply assembly accepts evidence from either speaker and does not require a target-author response.
- Existing persona tests continue to pass. Add memory-specific tests for ingestion, provenance, retrieval, ambiguity, and contact isolation, plus a labelled set covering exact, semantic, and no-evidence queries.

## Out of scope for V1

- Fact update, supersession, and periodic consolidation.
- A new repository, a new vector service, or mandatory LangGraph orchestration.
- A new WhatsApp live transport or webhook service.
- Cross-contact sharing of memories without an explicit identity and authorization design.
