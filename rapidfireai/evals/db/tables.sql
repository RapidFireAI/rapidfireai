-- RF-Inferno Database Schema
-- SQLite schema for tracking multi-pipeline inference experiments
-- Simplified: JSON-only storage, in-memory scheduler, essential state tracking

-- ============================================================================
-- EXPERIMENTS TABLE
-- Tracks high-level experiment configuration and status
-- ============================================================================
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    metric_experiment_id TEXT,
    num_shards INTEGER DEFAULT 0,
    num_actors INTEGER NOT NULL,
    num_cpus INTEGER,
    num_gpus INTEGER,
    status TEXT NOT NULL,  -- 'running', 'completed', 'failed'
    error TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- CONTEXTS TABLE
-- Stores unique RAG/context generation configurations
-- Multiple pipelines can share the same context via context_id
-- ============================================================================
CREATE TABLE IF NOT EXISTS contexts (
    context_id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_hash TEXT NOT NULL UNIQUE,  -- SHA256 hash for deduplication
    rag_config_json TEXT,  -- Full RAG configuration as JSON
    prompt_config_json TEXT,  -- Full prompt manager configuration as JSON
    status TEXT NOT NULL,  -- 'new', 'ongoing', 'deleted', 'failed'
    error TEXT DEFAULT '',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL
);

-- ============================================================================
-- PIPELINES TABLE
-- Stores configuration for each inference pipeline
-- ============================================================================
CREATE TABLE IF NOT EXISTS pipelines (
    pipeline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id INTEGER,  -- Foreign key to contexts table (NULL if no RAG)
    pipeline_type TEXT NOT NULL,  -- 'vllm', 'openai_api', etc.
    pipeline_config TEXT NOT NULL,  -- Full pipeline configuration (encoded with encode_payload - includes functions/classes)
    pipeline_config_json TEXT,  -- JSON-serializable pipeline configuration (for analytics/display, excludes functions/classes)
    flattened_config TEXT DEFAULT '{}',  -- Flattened configuration for IC Ops panel display
    status TEXT NOT NULL,  -- 'new', 'ongoing', 'completed', 'stopped', 'deleted', 'failed'
    current_shard_id INTEGER DEFAULT 0,  -- Next shard to process
    shards_completed INTEGER DEFAULT 0,  -- Number of shards completed
    total_samples_processed INTEGER DEFAULT 0,
    metric_run_id TEXT,  -- MetricLogger run ID for this pipeline
    error TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (context_id) REFERENCES contexts(context_id) ON DELETE SET NULL
);

-- ============================================================================
-- ACTOR_TASKS TABLE
-- Tracks individual tasks assigned to query processing actors
-- ============================================================================
CREATE TABLE IF NOT EXISTS actor_tasks (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_id INTEGER NOT NULL,
    actor_id INTEGER NOT NULL,
    shard_id INTEGER NOT NULL,
    status TEXT NOT NULL,  -- 'scheduled', 'in_progress', 'completed', 'failed'
    error_message TEXT DEFAULT '',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL,

    FOREIGN KEY (pipeline_id) REFERENCES pipelines(pipeline_id) ON DELETE CASCADE
);

-- ============================================================================
-- INTERACTIVE_CONTROL TABLE
-- Tracks user-initiated dynamic pipeline control operations
-- ============================================================================
CREATE TABLE IF NOT EXISTS interactive_control (
    ic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    pipeline_id INTEGER,  -- NULL for CLONE operation (creates new pipeline)
    operation TEXT NOT NULL,  -- 'stop', 'resume', 'delete', 'clone'
    status TEXT NOT NULL,  -- 'pending', 'processing', 'completed', 'failed'
    request_data TEXT,  -- JSON data for operation (e.g., model_config for clone)
    error TEXT DEFAULT '',
    created_at REAL NOT NULL,  -- Unix timestamp
    processed_at REAL,  -- Unix timestamp when operation was processed

    FOREIGN KEY (pipeline_id) REFERENCES pipelines(pipeline_id) ON DELETE CASCADE
);
