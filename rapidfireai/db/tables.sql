-- RapidFire AI Unified Database Schema
-- SQLite schema for tracking both fit (training) and evals (inference) experiments
-- Unified: Single experiments table, mode-specific tables for runs/pipelines

-- ============================================================================
-- EXPERIMENTS TABLE (Shared)
-- Tracks high-level experiment configuration and status for both modes
-- ============================================================================
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    experiments_path TEXT NOT NULL,
    metric_experiment_id TEXT,
    status TEXT NOT NULL,  -- 'running', 'completed', 'failed', 'cancelled'
    error TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Fit-specific state (nullable, not used by evals)
    current_task TEXT DEFAULT 'idle',  -- 'idle', 'create_models', 'run_fit', 'ic_ops'
    
    -- Mode-specific configuration (JSON)
    config TEXT DEFAULT '{}'
);

-- ============================================================================
-- INTERACTIVE_CONTROL TABLE (Unified)
-- Tracks user-initiated dynamic control operations for both runs and pipelines
-- ============================================================================
CREATE TABLE IF NOT EXISTS interactive_control (
    ic_id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id INTEGER NOT NULL,       -- run_id for fit, pipeline_id for evals
    target_type TEXT NOT NULL,        -- 'run' or 'pipeline'
    operation TEXT NOT NULL,          -- 'stop', 'resume', 'delete', 'clone', 'clone_warm'
    config_data TEXT DEFAULT '{}',    -- JSON: config_leaf for fit, request_data for evals
    status TEXT NOT NULL,             -- 'pending', 'processing', 'completed', 'failed', 'skipped'
    error TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

-- ============================================================================
-- FIT MODE TABLES
-- Tables for training experiment tracking
-- ============================================================================

-- Runs table (fit mode)
CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    status TEXT NOT NULL,
    metric_run_id TEXT,
    flattened_config TEXT DEFAULT '{}',
    config_leaf TEXT DEFAULT '{}',
    completed_steps INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 0,
    num_chunks_visited_curr_epoch INTEGER DEFAULT 0,
    num_epochs_completed INTEGER DEFAULT 0,
    chunk_offset INTEGER DEFAULT 0,
    error TEXT DEFAULT '',
    source TEXT DEFAULT '',
    ended_by TEXT DEFAULT '',
    warm_started_from INTEGER DEFAULT NULL,
    cloned_from INTEGER DEFAULT NULL
);

-- Worker Task table (fit mode)
CREATE TABLE IF NOT EXISTS worker_task (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id INTEGER NOT NULL,
    task_type TEXT NOT NULL,
    status TEXT NOT NULL,
    run_id INTEGER NOT NULL,
    chunk_id INTEGER NOT NULL,
    config_options TEXT DEFAULT '{}',
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

-- Controller Progress table (fit mode)
CREATE TABLE IF NOT EXISTS controller_progress (
    run_id INTEGER PRIMARY KEY,
    progress REAL DEFAULT 0.0,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

-- Worker Progress table (fit mode)
CREATE TABLE IF NOT EXISTS worker_progress (
    run_id INTEGER PRIMARY KEY,
    subchunk_progress REAL DEFAULT 0.0,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

-- ============================================================================
-- EVALS MODE TABLES
-- Tables for inference experiment tracking
-- ============================================================================

-- Contexts table (evals mode - RAG context configurations)
CREATE TABLE IF NOT EXISTS contexts (
    context_id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_hash TEXT NOT NULL UNIQUE,  -- SHA256 hash for deduplication
    rag_config_json TEXT,
    prompt_config_json TEXT,
    status TEXT NOT NULL,  -- 'new', 'ongoing', 'completed', 'deleted', 'failed'
    error TEXT DEFAULT '',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration_seconds REAL
);

-- Pipelines table (evals mode)
CREATE TABLE IF NOT EXISTS pipelines (
    pipeline_id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_id INTEGER,
    pipeline_type TEXT NOT NULL,  -- 'vllm', 'openai_api', etc.
    pipeline_config TEXT NOT NULL,  -- Encoded with dill (includes functions/classes)
    pipeline_config_json TEXT,  -- JSON-serializable version for display
    flattened_config TEXT DEFAULT '{}',
    status TEXT NOT NULL,  -- 'new', 'ongoing', 'completed', 'stopped', 'deleted', 'failed'
    current_shard_id INTEGER DEFAULT 0,
    shards_completed INTEGER DEFAULT 0,
    total_samples_processed INTEGER DEFAULT 0,
    metric_run_id TEXT,
    error TEXT DEFAULT '',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (context_id) REFERENCES contexts(context_id) ON DELETE SET NULL
);

-- Actor Tasks table (evals mode)
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
