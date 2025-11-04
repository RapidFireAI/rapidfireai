# CLAUDE.md - Backend

This file provides guidance for working with the backend components of RapidFire AI.

## Overview

The backend module contains the core orchestration logic for RapidFire's chunk-based concurrent training system. It coordinates between the user's process (Controller), scheduling logic (Scheduler), actual training execution (Worker), and dataset chunking (DatasetChunks).

## Files

### controller.py
**Purpose**: Central orchestrator running in the user's process

**Key Responsibilities**:
- Creates runs from parameter configurations (single configs, AutoML algorithms, IC Ops clones)
- Spawns and manages Worker processes (one per GPU)
- Runs the main scheduling loop that assigns (run_id, chunk_id) pairs to available workers
- Handles Interactive Control Operations (stop, resume, clone-modify, delete)
- Monitors training progress and coordinates epoch transitions
- Manages shared memory for model checkpoints
- Logs to MLflow and database

**Key Methods**:
- `run_fit()`: Main entry point, coordinates entire training lifecycle
- `_create_models()`: Creates run entries in DB from param configs
- `_schedule_and_monitor()`: Main loop that calls Scheduler and dispatches tasks to workers
- `_handle_ic_ops()`: Polls DB for IC Ops requests and executes them
- `_handle_clone_modify()`, `_handle_stop()`, `_handle_resume()`, `_handle_delete()`: IC Ops handlers

**Patterns**:
- Uses multiprocessing with 'spawn' method (set in `__init__`)
- Polls database for task status and IC Ops requests
- Coordinates with Workers via database task table
- Uses SharedMemoryManager for checkpoint storage

### scheduler.py
**Purpose**: Pure scheduling algorithm that assigns runs to workers

**Key Responsibilities**:
- Maintains state of which workers are busy and which runs have completed which chunks
- Implements round-robin scheduling for fairness
- Ensures runs don't execute on multiple workers simultaneously
- Tracks progress (chunks completed per run)
- Handles run addition/removal (for IC Ops)

**Key Methods**:
- `schedule()`: Returns next (run_id, worker_id, chunk_id) assignment or None if all done
- `add_run()`: Add new run to scheduler (for resume/clone IC Ops)
- `remove_run()`: Remove run from scheduler (for stop/delete IC Ops)
- `reset_run()`: Reset run progress at epoch boundaries
- `set_completed_task()`: Mark a worker's task as completed

**Return Values from `schedule()`**:
- `{run_id: X, worker_id: Y, chunk_id: Z, is_last_chunk: bool}` - Valid assignment
- `{run_id: None, ...}` - All runs completed all chunks (termination)
- `{run_id: -1, ...}` - All workers busy or no available runs (wait)

**Design Notes**:
- Zero-indexed workers and chunks, one-indexed run_ids
- Stateless pure scheduling logic (state passed in constructor)
- No I/O or side effects, just assignment logic

### worker.py
**Purpose**: Separate GPU process that executes actual training

**Key Responsibilities**:
- Polls database for assigned tasks (run_id, chunk_id pairs)
- Loads model from shared memory or disk checkpoint
- Trains on the assigned data chunk using appropriate Trainer (SFT/DPO/GRPO)
- Saves checkpoint back to shared memory/disk after chunk
- Reports metrics to MLflow and updates database task status
- Handles graceful shutdown on signals

**Key Methods**:
- `run()`: Main worker loop - polls for tasks, executes them, reports completion
- `run_fit()`: Executes training for one (run_id, chunk_id) pair
- `load_datasets()`: Loads train/eval datasets from disk (serialized by Controller)

**Lifecycle**:
1. Worker spawned by Controller with (worker_id, shared memory objects, shutdown_event)
2. Worker enters main loop in `run()` method
3. Polls database for tasks with status=SCHEDULED and worker_id=self.worker_id
4. On task found: loads model, trains chunk, saves checkpoint, marks task COMPLETED
5. Repeats until shutdown_event is set

**Patterns**:
- Each Worker has exclusive access to one GPU (via CUDA_VISIBLE_DEVICES)
- Uses SharedMemoryManager to load/save checkpoints
- Creates trainer instance per chunk (via `ml/trainer.py`)
- Redirects stdout/stderr during training to avoid pollution

### chunks.py
**Purpose**: Utility class for splitting datasets into chunks

**Key Responsibilities**:
- Divides dataset into N chunks with even distribution
- Handles batch size alignment (chunks align with batch boundaries)
- Supports offset for resuming training mid-epoch
- Validates inputs (chunk count, batch size, offset)

**Key Methods**:
- `__init__()`: Creates chunk boundaries based on dataset size, num_chunks, batch_size
- `get_chunk_indices()`: Returns (start_idx, end_idx) for a given chunk_id
- `_create_base_chunk_indices()`: Distributes batches evenly across chunks
- `_apply_offset()`: Applies modulo offset for resume functionality

**Usage Pattern**:
```python
chunker = DatasetChunks(dataset_size=1000, n_chunks=4, batch_size=8)
start, end = chunker.get_chunk_indices(chunk_id=0)
chunk_data = dataset[start:end]
```

**Design Notes**:
- Chunks distribute batches, not individual examples
- Last chunk may be smaller if batches don't divide evenly
- Offset wraps around with modulo for mid-epoch resume
- Raises ValueError for invalid inputs (too many chunks, bad offsets, etc.)

## Key Interactions

1. **Controller → Scheduler**: Controller calls `scheduler.schedule()` to get next assignment
2. **Controller → Worker**: Controller creates WorkerTask in DB, Worker polls and executes
3. **Controller → SharedMemory**: Controller saves initial models to SHM
4. **Worker → SharedMemory**: Worker loads models from SHM, trains, saves back to SHM
5. **Worker → Database**: Worker updates task status (IN_PROGRESS → COMPLETED)
6. **Controller → Database**: Controller polls for IC Ops requests, updates run status

## State Management

**Run Status Flow**:
- NEW → ONGOING → COMPLETED/FAILED
- ONGOING → STOPPED (IC Ops stop)
- STOPPED → ONGOING (IC Ops resume)
- Any → KILLED (IC Ops delete)

**Task Status Flow**:
- SCHEDULED → IN_PROGRESS → COMPLETED/FAILED

**Scheduler State**:
- `worker_running_current_run`: Maps worker_id to current run_id (-1 if idle)
- `run_visited_num_chunks`: Maps run_id to number of chunks completed

## Testing

Run tests with:
```bash
pytest tests/test_chunks.py
```

## Common Patterns

### Adding IC Ops Support
1. Add handler method in Controller (e.g., `_handle_new_op()`)
2. Update `_handle_ic_ops()` to check for new op type in DB
3. Update Scheduler if state changes needed (e.g., add_run/remove_run)
4. Add dispatcher endpoint to trigger IC Op
5. Update database schema if needed (tables.sql)

### Debugging Scheduling Issues
- Add logging in `scheduler.schedule()` to see assignment decisions
- Check `worker_running_current_run` and `run_visited_num_chunks` state
- Verify task status transitions in database
- Check Worker logs for task execution timing

### Performance Tuning
- Adjust `num_chunks` to balance concurrency vs checkpoint overhead
- Monitor shared memory usage with `SharedMemoryManager` logging
- Check scheduling fairness with chunk completion timestamps
- Profile Worker task execution time vs scheduling loop latency
