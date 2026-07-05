-- ============================================================
-- SEM-Swarm: Shared Epistemic Memory — Database Initialization
-- ============================================================
-- This script initializes the PostgreSQL database with pgvector
-- extension and creates the three core tables for the epistemic
-- memory system.
-- ============================================================

-- 1. Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create custom types
DO $$ BEGIN
    CREATE TYPE observation_status AS ENUM ('pending', 'processing', 'rejected', 'verified');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- 3. Core Tables

-- Table: env_observations
-- Stores raw observations deposited by Scout agents.
-- These are unverified, unprocessed inputs from the environment.
CREATE TABLE IF NOT EXISTS env_observations (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source_agent    VARCHAR(64) NOT NULL,
    raw_content     TEXT NOT NULL,
    status          observation_status NOT NULL DEFAULT 'pending',
    metadata        JSONB DEFAULT '{}'::jsonb
);

-- Index for polling pending observations (Filter agent)
CREATE INDEX IF NOT EXISTS idx_observations_status
    ON env_observations (status)
    WHERE status IN ('pending', 'processing');

CREATE INDEX IF NOT EXISTS idx_observations_created
    ON env_observations (created_at DESC);

-- Table: epistemic_memory
-- Stores verified facts with semantic embeddings.
-- This is the "hive brain" — the shared knowledge of the swarm.
CREATE TABLE IF NOT EXISTS epistemic_memory (
    id                      BIGSERIAL PRIMARY KEY,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    fact_text               TEXT NOT NULL,
    embedding               halfvec(2048),  -- qwen3-embedding dimension (multilingual, 100+ languages)
    confidence_score        FLOAT NOT NULL DEFAULT 0.0 CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    source_observation_id   BIGINT REFERENCES env_observations(id) ON DELETE SET NULL,
    superseded_by           BIGINT REFERENCES epistemic_memory(id) ON DELETE SET NULL,
    is_active               BOOLEAN NOT NULL DEFAULT TRUE,
    metadata                JSONB DEFAULT '{}'::jsonb
);

-- HNSW index for fast approximate nearest-neighbor search
CREATE INDEX IF NOT EXISTS idx_memory_embedding_hnsw
    ON epistemic_memory
    USING hnsw (embedding halfvec_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_memory_active
    ON epistemic_memory (is_active)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_memory_confidence
    ON epistemic_memory (confidence_score DESC)
    WHERE is_active = TRUE;

-- Table: swarm_state
-- Tracks the coordination state of the swarm.
-- Single-row table representing the current global state.
CREATE TABLE IF NOT EXISTS swarm_state (
    id                          SERIAL PRIMARY KEY,
    current_task                TEXT DEFAULT '',
    active_agents_count         INTEGER NOT NULL DEFAULT 0,
    total_observations          BIGINT NOT NULL DEFAULT 0,
    total_verified_facts        BIGINT NOT NULL DEFAULT 0,
    last_consensus_at           TIMESTAMPTZ,
    last_dreaming_loop_at       TIMESTAMPTZ,
    metadata                    JSONB DEFAULT '{}'::jsonb
);

-- Insert initial swarm state row
INSERT INTO swarm_state (id, current_task, active_agents_count)
VALUES (1, 'bootstrap', 0)
ON CONFLICT (id) DO NOTHING;

-- ============================================================
-- Helper function: Update the updated_at timestamp on row change
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to env_observations
DROP TRIGGER IF EXISTS update_observations_updated_at ON env_observations;
CREATE TRIGGER update_observations_updated_at
    BEFORE UPDATE ON env_observations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Apply trigger to epistemic_memory
DROP TRIGGER IF EXISTS update_memory_updated_at ON epistemic_memory;
CREATE TRIGGER update_memory_updated_at
    BEFORE UPDATE ON epistemic_memory
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- Done. Epistemic Memory substrate is ready.
-- ============================================================
