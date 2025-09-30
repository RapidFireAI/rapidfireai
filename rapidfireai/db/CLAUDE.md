# CLAUDE.md - Database

This file provides guidance for working with the database layer of RapidFire AI.

## Overview

The db module provides the persistence layer for RapidFire using SQLite. It stores experiment metadata, run configurations, task scheduling state, and checkpoint locations. The design emphasizes async operations and clean separation between the database interface and domain logic.

## Files

### rf_db.py
**Purpose**: High-level database API with domain-specific operations

**Key Responsibilities**:
- Implements all CRUD operations for experiments, runs, tasks, and IC Ops
- Handles serialization/deserialization of complex objects (using `encode_payload`/`decode_db_payload`)
- Manages experiment lifecycle (create, activate, reset, cleanup)
- Provides query methods for Controller, Worker, and Dispatcher
- Enforces business logic constraints (e.g., can't IC Ops on KILLED runs)

**Key Methods - Experiments**:
- `create_experiment()`: Create new experiment entry
- `get_running_experiment()`: Get currently active experiment
- `set_experiment_status()`: Update experiment status
- `reset_all_tables()`: Truncate tables (cleanup)
- `reset_experiment_states()`: Mark in-progress tasks as failed (crash recovery)

**Key Methods - Runs**:
- `create_run()`: Create run with config, status, source
- `get_run()`: Get run by ID
- `get_all_runs()`: Get all runs with metrics
- `get_runs_by_status()`: Filter runs by status(es)
- `set_run_status()`: Update run status
- `set_run_ended_by()`: Mark how run ended (completed/failed/killed)
- `update_run_metrics()`: Update training metrics

**Key Methods - Tasks**:
- `create_worker_task()`: Create task for worker to execute
- `get_next_worker_task()`: Poll for next task (used by Worker)
- `set_worker_task_status()`: Update task status
- `get_controller_task()`: Get current controller task
- `set_controller_task()`: Update controller task

**Key Methods - Interactive Control**:
- `request_clone_modify()`: Create IC Ops request for clone
- `request_stop()`: Request run stop
- `request_resume()`: Request run resume
- `request_delete()`: Request run deletion
- `get_ic_ops_request()`: Poll for IC Ops requests (used by Controller)
- `mark_ic_ops_completed()`: Mark IC Op as processed

**Serialization**:
- Complex objects (configs, datasets, models) stored as BLOBs
- Uses `encode_payload()` from utils/serialize.py before storing
- Uses `decode_db_payload()` when reading back
- Handles torch tensors, datasets, and arbitrary Python objects via dill

**Patterns**:
- All methods use `db.execute()` with parameterized queries (SQL injection safe)
- Commit=True by default for most operations
- Returns dicts or lists of dicts (not raw tuples)
- Raises DBException on errors with context

### db_interface.py
**Purpose**: Low-level SQLite connection wrapper

**Key Responsibilities**:
- Manages SQLite connection lifecycle
- Provides generic `execute()` method for queries
- Handles connection pooling and thread safety
- Converts query results to dicts

**Key Methods**:
- `execute()`: Execute parameterized query, return results as list of dicts
- `close()`: Close database connection

**Design Notes**:
- Uses sqlite3 row_factory for dict results
- Single connection per RfDb instance
- Thread-safe via SQLite's default settings

### tables.sql
**Purpose**: Database schema definition

**Tables**:

**experiments**:
- `experiment_id` (PK): Unique experiment identifier
- `experiment_name`: User-provided name
- `status`: ExperimentStatus enum (NEW, RUNNING, COMPLETED, FAILED)
- `experiments_path`: Base path for experiment artifacts
- `created_at`, `updated_at`: Timestamps

**runs**:
- `run_id` (PK): Unique run identifier
- `experiment_id` (FK): Parent experiment
- `run_name`: Generated name (e.g., "run_1")
- `mlflow_run_id`: MLflow tracking ID
- `status`: RunStatus enum (NEW, ONGOING, COMPLETED, FAILED, STOPPED, KILLED)
- `source`: RunSource enum (USER, CLONE_MODIFY)
- `ended_by`: RunEndedBy enum (COMPLETED, FAILED, KILLED, STOPPED)
- `parent_run_id`: For cloned runs
- `warm_start`: Boolean flag for clone-modify
- `config_leaf`: Serialized run configuration (BLOB)
- `seed`: Random seed for reproducibility
- `num_chunks`: Number of data chunks
- `current_chunk`, `current_epoch`: Progress tracking
- `metrics`: JSON string of training metrics
- `created_at`, `updated_at`: Timestamps

**worker_task**:
- `task_id` (PK): Unique task identifier
- `run_id` (FK): Run to execute
- `worker_id`: GPU worker assignment
- `chunk_id`: Data chunk to train on
- `epoch`: Current epoch number
- `status`: TaskStatus enum (SCHEDULED, IN_PROGRESS, COMPLETED, FAILED)
- `created_at`, `updated_at`: Timestamps

**controller_progress**:
- Tracks controller state (single row table)
- `task`: ControllerTask enum
- `status`: TaskStatus enum

**worker_progress**:
- Tracks per-worker state
- `worker_id` (PK): Worker identifier
- `task`: WorkerTask enum
- `status`: TaskStatus enum

**interactive_control**:
- `ic_id` (PK): IC Ops request identifier
- `run_id` (FK): Target run
- `operation`: IC Ops type (CLONE_MODIFY, STOP, RESUME, DELETE)
- `status`: TaskStatus enum
- `config_leaf`: New config for clone-modify (BLOB)
- `warm_start`: Boolean for clone-modify
- `created_at`, `updated_at`: Timestamps

## Key Concepts

### Status Enums
Defined in `utils/constants.py`:
- **ExperimentStatus**: NEW, RUNNING, COMPLETED, FAILED
- **RunStatus**: NEW, ONGOING, COMPLETED, FAILED, STOPPED, KILLED
- **RunSource**: USER, CLONE_MODIFY
- **RunEndedBy**: COMPLETED, FAILED, KILLED, STOPPED
- **TaskStatus**: PENDING, SCHEDULED, IN_PROGRESS, COMPLETED, FAILED

### Transaction Model
- Most operations are auto-commit (commit=True)
- No explicit transaction management (SQLite handles implicitly)
- Crash recovery via `reset_experiment_states()` on startup

### Query Patterns
```python
# Parameterized query (safe)
query = "SELECT * FROM runs WHERE run_id = ?"
result = self.db.execute(query, (run_id,))

# With commit
query = "UPDATE runs SET status = ? WHERE run_id = ?"
self.db.execute(query, (new_status, run_id), commit=True)
```

## Common Operations

### Creating a Run
```python
run_id = db.create_run(
    experiment_id=1,
    run_name="run_1",
    mlflow_run_id="abc123",
    config_leaf=encode_payload(config_dict),
    source=RunSource.USER,
    seed=42,
    num_chunks=8
)
```

### Polling for Tasks (Worker)
```python
task = db.get_next_worker_task(worker_id=0)
if task:
    db.set_worker_task_status(task['task_id'], TaskStatus.IN_PROGRESS)
    # ... do work ...
    db.set_worker_task_status(task['task_id'], TaskStatus.COMPLETED)
```

### IC Ops Flow (Controller)
```python
# User triggers stop via UI → Dispatcher → Database
db.request_stop(run_id=5)

# Controller polls and processes
ic_ops = db.get_ic_ops_request()
for op in ic_ops:
    if op['operation'] == 'STOP':
        # ... handle stop ...
        db.mark_ic_ops_completed(op['ic_id'])
```

### Cleanup Between Experiments
```python
db.reset_all_tables(experiments_table=False)  # Keep experiments table
db.set_experiment_status(exp_id, ExperimentStatus.COMPLETED)
```

## Testing Database Changes

1. Modify `tables.sql` if adding/changing tables
2. Delete existing `rapidfire.db` file to force recreation
3. Run `db.create_tables()` to apply schema
4. Test with `pytest` or manual verification
5. Ensure backward compatibility with existing experiments

## Performance Considerations

- SQLite write contention: Workers only write task status updates
- Most writes are from Controller (runs, IC Ops, metrics)
- No indexes beyond PRIMARY KEYs (small data volume)
- BLOB storage for configs is fine (not queried, only retrieved by PK)

## Common Patterns

### Adding New Table
1. Add CREATE TABLE to `tables.sql`
2. Add CRUD methods to `rf_db.py`
3. Add any enums to `utils/constants.py`
4. Update `reset_all_tables()` if needed for cleanup
5. Test with fresh database

### Debugging Database Issues
```python
# Check database file location
import os
print(os.path.abspath("rapidfire.db"))

# Inspect directly with sqlite3 CLI
# sqlite3 rapidfire.db
# .schema
# SELECT * FROM runs;
# SELECT * FROM worker_task WHERE status = 'IN_PROGRESS';
```

### Migration Strategy
- Currently no migrations (schema assumed stable)
- Breaking changes require users to reset experiments
- Future: Add migration system (e.g., Alembic) if needed
