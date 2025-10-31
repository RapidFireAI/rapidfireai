# RF-Inferno Database Module

Provides database interface for tracking multi-pipeline inference experiments.

## Files

- `tables.sql` - SQLite schema definition
- `db_interface.py` - Low-level database connection and query execution
- `rf_db.py` - High-level CRUD operations for experiments, contexts, pipelines, and tasks
- `__init__.py` - Module exports

## Quick Start

```python
from rapidfireai.evals.db import RFDatabase

# Initialize database
db = RFDatabase()

# Create experiment
exp_id = db.create_experiment(
    experiment_name="my_experiment",
    num_shards=5,
    num_actors=4,
    num_cpus=8,
    num_gpus=2
)

# Create pipeline
pipeline_id = db.create_pipeline(
    experiment_id=exp_id,
    pipeline_name="baseline_temp0.7",
    pipeline_type="vllm",
    model_config_json='{"model": "Qwen/Qwen2.5-3B-Instruct"}',
    sampling_params_json='{"temperature": 0.7}',
    status="new"
)

# Update progress
db.set_pipeline_progress(
    pipeline_id,
    current_shard_id=2,
    shards_completed=2,
    total_samples_processed=1024
)

# Create task
task_id = db.create_actor_task(
    experiment_id=exp_id,
    pipeline_id=pipeline_id,
    actor_id=0,
    shard_id=2
)

# Close connection
db.close()
```

## API Methods

### Experiments
- `create_experiment(name, num_shards, num_actors, ...) -> exp_id`
- `get_current_experiment() -> dict`
- `get_all_experiment_names() -> list[str]`
- `set_experiment_status(exp_id, status)`
- `set_experiment_error(exp_id, error)`
- `get_experiment_error(exp_id) -> str`

### Contexts (RAG)
- `create_context(hash, rag_json, prompt_json) -> context_id`
- `get_context(context_id) -> dict`
- `set_context_status(context_id, status)`
- `set_context_start_time(context_id, time)`
- `set_context_end_time(context_id, time, duration)`
- `set_context_error(context_id, error)`

### Pipelines
- `create_pipeline(exp_id, name, type, model_json, sampling_json) -> pipeline_id`
- `get_pipeline(pipeline_id) -> dict`
- `get_all_pipelines(exp_id) -> list[dict]`
- `set_pipeline_status(pipeline_id, status)`
- `set_pipeline_progress(pipeline_id, shard_id, completed, samples)`
- `set_pipeline_error(pipeline_id, error)`

### Actor Tasks
- `create_actor_task(exp_id, pipeline_id, actor_id, shard_id) -> task_id`
- `get_actor_task(task_id) -> dict`
- `get_running_actor_tasks(exp_id) -> list[dict]`
- `set_actor_task_status(task_id, status)`
- `set_actor_task_start_time(task_id, time)`
- `set_actor_task_end_time(task_id, time, duration)`
- `set_actor_task_error(task_id, error)`

## Status Values

### Experiments
- `running` - Currently executing
- `completed` - Successfully finished
- `failed` - Encountered error

### Contexts
- `pending` - Not yet built
- `building` - Index construction in progress
- `ready` - Ready for use
- `failed` - Build failed

### Pipelines
- `new` - Just created
- `ongoing` - Currently processing shards
- `completed` - All shards processed
- `stopped` - User stopped
- `deleted` - User deleted
- `failed` - Encountered error

### Actor Tasks
- `scheduled` - Task assigned but not started
- `in_progress` - Currently executing
- `completed` - Successfully finished
- `failed` - Encountered error

## Testing

Run the test suite:
```bash
python tests/test_rf_db.py
```



