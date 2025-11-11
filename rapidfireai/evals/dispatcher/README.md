# RF-Inferno Dispatcher

REST API for interactive control of running experiments. Allows dynamic pipeline management during execution.

## Features

- **Stop Pipeline**: Pause execution, can be resumed later
- **Resume Pipeline**: Continue stopped pipeline from last completed shard
- **Delete Pipeline**: Permanently remove pipeline
- **Clone Pipeline**: Create new pipeline using existing context

## Automatic Startup

The dispatcher **automatically starts** when you create an `Experiment` object. It runs in a background thread and cleans up automatically when the experiment ends.

```python
from rapidfireai.evals.experiment import Experiment

# Dispatcher starts automatically on http://127.0.0.1:8851
experiment = Experiment(experiment_name="my-experiment", num_actors=2)

# Now you can use the dispatcher API for interactive control
# ...

# Dispatcher automatically stops when experiment ends
experiment.end()
```

## Manual Standalone Mode

For testing or standalone use:
```bash
python -m rapidfireai.evals.dispatcher.dispatcher
```
Runs on `http://127.0.0.1:8851`

## API Endpoints

Base URL: `http://127.0.0.1:8851/dispatcher`

### Health Check
```bash
GET /dispatcher/health-check
```

### Stop Pipeline
```bash
POST /dispatcher/stop-pipeline
Content-Type: application/json

{
    "pipeline_id": 123
}
```

**Response:**
```json
{
    "ic_id": 1,
    "message": "Stop request created for pipeline 123"
}
```

### Resume Pipeline
```bash
POST /dispatcher/resume-pipeline
Content-Type: application/json

{
    "pipeline_id": 123
}
```

### Delete Pipeline
```bash
POST /dispatcher/delete-pipeline
Content-Type: application/json

{
    "pipeline_id": 123
}
```

### Clone Pipeline (VLLM)
```bash
POST /dispatcher/clone-pipeline
Content-Type: application/json

{
    "context_id": 1,
    "pipeline_name": "my_cloned_vllm_pipeline",
    "pipeline_type": "vllm",
    "model_config": {
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "dtype": "half",
        "gpu_memory_utilization": 0.6
    },
    "sampling_params": {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 512
    }
}
```

### Clone Pipeline (OpenAI)
```bash
POST /dispatcher/clone-pipeline
Content-Type: application/json

{
    "context_id": 1,
    "pipeline_name": "my_cloned_openai_pipeline",
    "pipeline_type": "openai",
    "client_config": {
        "api_key": "sk-...",
        "base_url": "https://api.openai.com/v1"
    },
    "model_config": {
        "model": "gpt-4",
        "temperature": 0.7,
        "top_p": 0.9
    },
    "rpm_limit": 500,
    "tpm_limit": 90000
}
```

### Check Operation Status
```bash
GET /dispatcher/operation-status/{ic_id}
```

**Response:**
```json
{
    "ic_id": 1,
    "pipeline_id": 123,
    "operation": "stop",
    "status": "completed",
    "error": "",
    "created_at": 1698765432.123,
    "processed_at": 1698765435.456
}
```

### Get All Operations
```bash
GET /dispatcher/all-operations
```

## Usage Example (Python)

```python
import requests

# Stop a pipeline
response = requests.post(
    "http://127.0.0.1:8851/dispatcher/stop-pipeline",
    json={"pipeline_id": 123}
)
ic_id = response.json()["ic_id"]

# Check status
status = requests.get(
    f"http://127.0.0.1:8851/dispatcher/operation-status/{ic_id}"
)
print(status.json())

# Clone a pipeline
response = requests.post(
    "http://127.0.0.1:8851/dispatcher/clone-pipeline",
    json={
        "context_id": 1,
        "pipeline_name": "cloned_pipeline",
        "model_config": {...},
        "sampling_params": {...}
    }
)
```

## Operation Lifecycle

1. **User sends request** → Dispatcher validates and creates IC operation (status: `pending`)
2. **Controller polls database** → Finds pending operation (status: `processing`)
3. **Controller executes operation** → Modifies scheduler, updates database
4. **Operation completes** → Status changes to `completed` or `failed`

## Error Handling

All endpoints return JSON with error details:
```json
{
    "error": "Pipeline 123 not found",
    "traceback": "..."
}
```

HTTP Status Codes:
- `200`: Success
- `400`: Bad request (missing parameters)
- `404`: Resource not found
- `500`: Server error

