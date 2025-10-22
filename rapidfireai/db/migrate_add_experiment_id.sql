-- Migration: Add experiment_id column to runs table
-- This migration adds experiment_id foreign key to scope runs to experiments

-- Step 1: Add experiment_id column with a default value
-- For existing runs, set experiment_id to the first experiment (lowest experiment_id)
-- This assumes existing databases only have one experiment
ALTER TABLE runs ADD COLUMN experiment_id INTEGER;

-- Step 2: Populate experiment_id for existing runs
-- Get the first (oldest) experiment_id and assign it to all existing runs
UPDATE runs
SET experiment_id = (SELECT MIN(experiment_id) FROM experiments)
WHERE experiment_id IS NULL;

-- Step 3: Make the column NOT NULL now that it's populated
-- Note: SQLite doesn't support ALTER COLUMN, so we need to recreate the table

-- Create new runs table with experiment_id constraint
CREATE TABLE runs_new (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    mlflow_run_id TEXT,
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
    cloned_from INTEGER DEFAULT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
);

-- Copy data from old table to new table
INSERT INTO runs_new (run_id, experiment_id, status, mlflow_run_id, flattened_config, config_leaf,
    completed_steps, total_steps, num_chunks_visited_curr_epoch, num_epochs_completed, chunk_offset,
    error, source, ended_by, warm_started_from, cloned_from)
SELECT run_id, experiment_id, status, mlflow_run_id, flattened_config, config_leaf,
    completed_steps, total_steps, num_chunks_visited_curr_epoch, num_epochs_completed, chunk_offset,
    error, source, ended_by, warm_started_from, cloned_from
FROM runs;

-- Drop old table and rename new table
DROP TABLE runs;
ALTER TABLE runs_new RENAME TO runs;
