CREATE TABLE IF NOT EXISTS memory_topics (
    id uuid PRIMARY KEY,
    user_id text NOT NULL,
    contact_id text NOT NULL,
    chat_id text NOT NULL,
    label text NOT NULL,
    summary text NOT NULL,
    source_message_ids uuid[] NOT NULL,
    processing_version text NOT NULL,
    processing_state text NOT NULL DEFAULT 'pending',
    created_at timestamptz NOT NULL DEFAULT now(),
    indexed_at timestamptz
);
CREATE INDEX IF NOT EXISTS memory_topics_scope
    ON memory_topics (user_id, contact_id, chat_id, processing_version);
ALTER TABLE memory_chunks ADD COLUMN IF NOT EXISTS topic_id uuid REFERENCES memory_topics(id);
CREATE INDEX IF NOT EXISTS memory_chunks_topic ON memory_chunks(topic_id);

CREATE TABLE IF NOT EXISTS topic_model_cache (
    user_id text NOT NULL,
    contact_id text NOT NULL,
    chat_id text NOT NULL,
    prompt_hash text NOT NULL,
    response text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, contact_id, chat_id, prompt_hash)
);
