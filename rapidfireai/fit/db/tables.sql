-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    metric_experiment_id TEXT,
    config_options TEXT NOT NULL,
    status TEXT NOT NULL,
    current_task TEXT NOT NULL,
    error TEXT DEFAULT ''
);

-- Runs table
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

-- Interactive Control table
CREATE TABLE IF NOT EXISTS interactive_control (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    ic_op TEXT NOT NULL,
    config_leaf TEXT DEFAULT '{}',
    status TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

-- Worker Task table
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

-- Controller Progress table
CREATE TABLE IF NOT EXISTS controller_progress (
    run_id INTEGER PRIMARY KEY,
    progress REAL DEFAULT 0.0,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);

-- Worker Progress table
CREATE TABLE IF NOT EXISTS worker_progress (
    run_id INTEGER PRIMARY KEY,
    subchunk_progress REAL DEFAULT 0.0,
    FOREIGN KEY (run_id) REFERENCES runs (run_id)
);