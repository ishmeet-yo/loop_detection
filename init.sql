CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS daily_summaries (
    id UUID PRIMARY KEY,
    summary TEXT NOT NULL,
    embedding VECTOR(384)
);

CREATE TABLE IF NOT EXISTS core_beliefs (
    id UUID PRIMARY KEY,
    label TEXT NOT NULL,
    polarity TEXT NOT NULL,
    occurrences INTEGER NOT NULL DEFAULT 1,
    last_seen TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
