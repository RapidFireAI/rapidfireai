# Pipeline Scheduler

Fair round-robin scheduler for multi-pipeline inference with generation-based fairness.

## Overview

The `PipelineScheduler` manages the assignment of pipelines to actors, ensuring:
- **Fair round-robin scheduling**: No pipeline processes twice before all pipelines process once
- **Generation-based fairness**: Tracks scheduling generations to maintain fairness
- **Dynamic pipeline management**: Add/remove pipelines during execution
- **Stateful tracking**: Monitors actor assignments and pipeline progress

## Key Concepts

### Terminology
- **Pipeline**: A specific model configuration to evaluate (e.g., "temp0.7", "temp1.0")
- **Actor**: A Ray actor that runs inference (QueryProcessingActor)
- **Shard**: A portion of the dataset (e.g., 1/5 of total samples)
- **Generation**: A scheduling round where each active pipeline gets one turn

### Scheduling Rules

1. **Sequential Processing**: Each pipeline processes shards in order: 0 → 1 → 2 → ... → N-1
2. **Fair Round-Robin**: Uses generation tracking to ensure fairness
3. **One Task Per Actor**: Each actor runs only one (pipeline, shard) at a time
4. **One Assignment Per Pipeline**: Each pipeline runs on only one actor at a time

### Generation-Based Fairness

```
Generation 0:  P1(shard 0) → P2(shard 0) → P3(shard 0)
Generation 1:  P1(shard 1) → P2(shard 1) → P3(shard 1)
Generation 2:  P1(shard 2) → P2(shard 2) → P3(shard 2)
...
```

No pipeline advances to shard N+1 until all pipelines complete shard N (within actor availability constraints).

## API

### Initialization

```python
from rapidfireai.infer.scheduling import PipelineScheduler

scheduler = PipelineScheduler(
    pipeline_ids=[1, 2, 3],  # Pipeline IDs to schedule
    num_actors=4,            # Number of available actors
    num_shards=5             # Total shards in dataset
)
```

### Core Methods

#### `schedule() -> dict`

Schedule the next task. Returns one of:

```python
# Scheduling successful
{"pipeline_id": 1, "actor_id": 0, "shard_id": 2}

# All actors busy - wait before calling again
{"pipeline_id": -1, "actor_id": -1, "shard_id": -1}

# All pipelines completed all shards - done!
{"pipeline_id": None, "actor_id": None, "shard_id": None}
```

#### `set_completed_task(actor_id: int)`

Mark a task as complete. Frees the actor and increments pipeline progress.

```python
scheduler.set_completed_task(actor_id=0)
```

#### `add_pipeline(pipeline_id: int, shards_completed: int = 0)`

Dynamically add a new pipeline during execution.

```python
scheduler.add_pipeline(pipeline_id=4, shards_completed=0)
```

#### `remove_pipeline(pipeline_id: int) -> int`

Remove a pipeline (e.g., due to error or user request). Returns progress.

```python
progress = scheduler.remove_pipeline(pipeline_id=2)
print(f"Pipeline 2 completed {progress} shards before removal")
```

#### `get_status() -> dict`

Get current scheduler state for debugging/monitoring.

```python
status = scheduler.get_status()
# {
#   "num_actors": 4,
#   "num_shards": 5,
#   "active_pipelines": 3,
#   "busy_actors": 2,
#   "completed_pipelines": 0,
#   "current_generation": 2,
#   "pipelines_in_generation": 2,
#   "actor_assignments": {0: 1, 1: 2},  # actor_id: pipeline_id
#   "pipeline_progress": {"1": "2/5", "2": "2/5", "3": "1/5"}
# }
```

## Usage Example

```python
from rapidfireai.infer.scheduling import PipelineScheduler

# Initialize
scheduler = PipelineScheduler(
    pipeline_ids=[1, 2, 3],
    num_actors=2,
    num_shards=5
)

# Main scheduling loop
while True:
    # Get next task
    task = scheduler.schedule()

    # Check completion
    if task["pipeline_id"] is None:
        print("All pipelines completed!")
        break

    # Check if actors busy
    if task["pipeline_id"] == -1:
        # Wait for completion
        completed_actor_id = wait_for_any_completion()
        scheduler.set_completed_task(completed_actor_id)
        continue

    # Execute task
    pipeline_id = task["pipeline_id"]
    actor_id = task["actor_id"]
    shard_id = task["shard_id"]

    submit_task_to_actor(actor_id, pipeline_id, shard_id)

    # When task completes (async callback):
    # scheduler.set_completed_task(actor_id)
```

## Integration with Controller

The Controller uses the scheduler to coordinate pipeline execution:

```python
# In Controller.run_inference():

# 1. Initialize scheduler
scheduler = PipelineScheduler(
    pipeline_ids=[p.pipeline_id for p in pipelines],
    num_actors=self.num_actors,
    num_shards=self.num_shards
)

# 2. Main loop
while True:
    task = scheduler.schedule()

    if task["pipeline_id"] is None:
        break  # All done

    if task["pipeline_id"] == -1:
        # Wait for completions
        completed_tasks = wait_for_completions()
        for actor_id in completed_tasks:
            scheduler.set_completed_task(actor_id)
            # Update database
            db.set_pipeline_progress(...)
        continue

    # Submit task
    actor_id = task["actor_id"]
    pipeline_id = task["pipeline_id"]
    shard_id = task["shard_id"]

    # Create task in database
    task_id = db.create_actor_task(
        experiment_id, pipeline_id, actor_id, shard_id
    )

    # Submit to Ray actor
    future = actors[actor_id].process_shard.remote(
        pipeline_config, shard_data
    )
```

## Design Decisions

### Why Generation-Based Fairness?

Ensures fair comparison across pipelines by keeping them roughly in sync:
- All pipelines see shard 0 before any sees shard 2
- Enables early stopping based on converged metrics
- Fair resource allocation across competing configurations

### Why No Database Calls in Scheduler?

- **Separation of concerns**: Scheduler = pure scheduling logic
- **Performance**: No I/O overhead in hot loop
- **Testability**: Easy to unit test without database
- Controller handles all persistence

### State Management

- **Scheduler**: In-memory state (lightweight, fast)
- **Database**: Persistent state (can recover/audit)
- **Controller**: Bridges scheduler ↔ database

## Testing

Run the test suite:
```bash
python tests/test_pipeline_scheduler.py
```

Tests cover:
- Basic scheduling
- Generation-based fairness
- All actors busy handling
- Completion detection
- Dynamic pipeline add/remove

