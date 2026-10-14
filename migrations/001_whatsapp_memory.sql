CREATE TABLE IF NOT EXISTS messages (
    id uuid PRIMARY KEY,
    user_id text NOT NULL,
    contact_id text NOT NULL,
    chat_id text NOT NULL,
    source_platform text NOT NULL DEFAULT 'whatsapp_export',
    source_message_id text,
    author text,
    direction text NOT NULL DEFAULT 'unknown',
    sent_at timestamptz NOT NULL,
    original_content text,
    original_payload jsonb,
    normalized_text text NOT NULL,
    is_system boolean NOT NULL DEFAULT false,
    is_edited boolean NOT NULL DEFAULT false,
    is_deleted boolean NOT NULL DEFAULT false,
    has_media boolean NOT NULL DEFAULT false,
    import_provenance jsonb NOT NULL DEFAULT '{}'::jsonb,
    source_fingerprint text,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS messages_source_id_unique
    ON messages (user_id, contact_id, source_platform, source_message_id)
    WHERE source_message_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS messages_recent_scope
    ON messages (user_id, contact_id, sent_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS messages_chat_scope
    ON messages (user_id, contact_id, chat_id, sent_at, id);
CREATE INDEX IF NOT EXISTS messages_fingerprint_scope
    ON messages (user_id, contact_id, chat_id, source_fingerprint);

CREATE TABLE IF NOT EXISTS message_imports (
    user_id text NOT NULL,
    contact_id text NOT NULL,
    chat_id text NOT NULL,
    export_hash text NOT NULL,
    source_sequence integer NOT NULL,
    message_id uuid NOT NULL REFERENCES messages(id),
    PRIMARY KEY (user_id, contact_id, chat_id, export_hash, source_sequence)
);

CREATE TABLE IF NOT EXISTS memory_chunks (
    id uuid PRIMARY KEY,
    user_id text NOT NULL,
    contact_id text NOT NULL,
    chat_id text NOT NULL,
    started_at timestamptz NOT NULL,
    ended_at timestamptz NOT NULL,
    start_sequence integer NOT NULL,
    end_sequence integer NOT NULL,
    source_message_ids uuid[] NOT NULL,
    text text NOT NULL,
    processing_state text NOT NULL DEFAULT 'pending',
    processing_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    indexed_at timestamptz
);
CREATE INDEX IF NOT EXISTS memory_chunks_scope
    ON memory_chunks (user_id, contact_id, chat_id, started_at);

CREATE TABLE IF NOT EXISTS facts (
    id uuid PRIMARY KEY,
    user_id text NOT NULL,
    contact_id text NOT NULL,
    subject_key text NOT NULL,
    subject_label text NOT NULL,
    stated_by text NOT NULL,
    fact_type text NOT NULL,
    raw_value text NOT NULL,
    normalized_value text NOT NULL,
    confidence double precision NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    asserted_at timestamptz NOT NULL,
    context text NOT NULL,
    source_message_ids uuid[] NOT NULL,
    chunk_id uuid NOT NULL REFERENCES memory_chunks(id),
    extraction_version text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS facts_key_scope
    ON facts (user_id, contact_id, subject_key, fact_type, asserted_at DESC);
CREATE INDEX IF NOT EXISTS facts_recent_scope
    ON facts (user_id, contact_id, asserted_at DESC);
