# CLAUDE.md - Utils

This file provides guidance for working with the utility modules in RapidFire AI.

## Overview

The utils module contains shared utilities used across RapidFire components, including logging, MLflow integration, shared memory management, serialization, exception handling, and constants.

## Google Colab Support

### Colab Helper (colab_helper.py)

**Purpose**: Utilities for running RapidFire in Google Colab and restricted notebook environments

**Key Functions**:
- `get_notebook_environment()`: Returns 'colab', 'kaggle', 'jupyter', or 'unknown'
- `setup_cloudflare_tunnel(port, description)`: Create free Cloudflare tunnel for port forwarding
- `setup_ngrok_tunnel(port, auth_token, description)`: Create ngrok tunnel (requires auth token)
- `expose_rapidfire_services(method, ...)`: Expose all services using specified tunneling method

**Tunneling Methods**:
1. **'native'**: Colab's built-in port forwarding (only works when called from notebook cell)
2. **'cloudflare'**: Free Cloudflare tunnels via cloudflared binary (no registration required)
3. **'ngrok'**: ngrok tunnels (requires free account and auth token)

**Important Architectural Note - Tunnel Routing Loop**:

When using tunnels in Colab, **inter-service communication must use localhost**, not tunnel URLs. Tunnel URLs are only for external browser access.

❌ **Wrong Architecture (creates routing loop)**:
```
Browser → Frontend Tunnel → Frontend:8853 → MLflow Tunnel → MLflow:8852
                                                   ↑
                                                   Fails with 502: Colab → Cloudflare → Colab loop
```

✅ **Correct Architecture**:
```
Browser → Frontend Tunnel → Frontend:8853 → localhost:8852 (direct)
Browser → MLflow Tunnel → MLflow:8852 (direct access if needed)
```

**Why this matters**:
- Cloudflare/ngrok tunnels expose local services to the internet
- They route: External Request → Tunnel Provider → Local Machine
- From within the same machine, tunnel URLs create a loop that fails
- Always use `http://127.0.0.1:<port>` for services on the same host

**Example in start_colab.py**:
```python
# Create tunnels for external access
mlflow_url = setup_cloudflare_tunnel(RF_MLFLOW_PORT, "MLflow Tracking UI")

# DON'T set RF_MLFLOW_URL env var - let frontend use localhost
# os.environ['RF_MLFLOW_URL'] = mlflow_url  # ❌ Creates routing loop

# Frontend subprocess will use default: http://127.0.0.1:8852/ ✅
```

**Colab Process Restrictions**:

Google Colab restricts certain OS-level process operations:
1. `os.setpgrp()` - Cannot create process groups (wrapped in try-except with fallback)
2. `os.getpgid()` - Cannot query process group IDs (uses psutil fallback)
3. `os.killpg()` - Only used when process_group_id exists (safe)

See `worker_manager.py` for implementation of these workarounds.

## Files

### constants.py
**Purpose**: Centralized definitions for enums, config values, and system constants

**Key Constants**:
- `USE_SHARED_MEMORY`: Flag to enable shared memory for checkpoints (default: True)
- `RF_LOG_FILENAME`: Log file naming pattern
- `RF_LOG_PATH`: Log file path
- `DB_PATH`: SQLite database file path

**Key Enums**:
- `ExperimentStatus`: NEW, RUNNING, COMPLETED, FAILED
- `RunStatus`: NEW, ONGOING, COMPLETED, FAILED, STOPPED, KILLED
- `RunSource`: USER, CLONE_MODIFY
- `RunEndedBy`: COMPLETED, FAILED, KILLED, STOPPED
- `TaskStatus`: PENDING, SCHEDULED, IN_PROGRESS, COMPLETED, FAILED
- `ExperimentTask`, `ControllerTask`, `WorkerTask`: Task type enums
- `SHMObjectType`: MODEL, CHECKPOINT (for shared memory registry)

**Config Classes**:
- `DispatcherConfig`: Dispatcher server configuration
- `MLFlowConfig`: MLFlow server configuration

**Usage**:
```python
from rapidfireai.utils.constants import MLFlowConfig
from rapidfireai.fit.utils.constants import RunStatus

if run['status'] == RunStatus.ONGOING.value:
    # ...
```

### logging.py
**Purpose**: Structured logging setup using loguru

**Key Classes**:
- `RFLogger`: Main logger factory for RapidFire components
- `TrainingLogger`: Specialized logger for training output

**Key Features**:
- Creates per-component loggers (experiment, controller, worker_N, dispatcher, etc.)
- Logs to both console and file
- Experiment-specific log directories
- Color-coded log levels
- Structured format with timestamps

**Usage**:
```python
from rapidfireai.fit.utils.logging import RFLogger

logger = RFLogger().create_logger("controller")
logger.info("Starting controller")
logger.error("Failed to schedule", run_id=5)
```

**Log Locations**:
- `{log_dir}/controller.log`
- `{log_dir}/worker_0.log`
- `{log_dir}/dispatcher.log`

### metric_manager.py
**Purpose**: Wrapper around metrics tracking API

**Key Responsibilities**:
- Creates and retrieves metric experiments
- Logs metrics, parameters, and artifacts
- Creates metric runs
- Handles metric server communication

**Key Methods**:
- `get_experiment(name)`: Get or create metric experiment
- `create_run(experiment_id)`: Create metric run, return run_id
- `log_metric(run_id, key, value, step)`: Log metric
- `log_param(run_id, key, value)`: Log parameter
- `log_artifact(run_id, artifact_path)`: Log artifact file
- `end_run(run_id)`: Mark run as completed

**Usage**:
```python
from rapidfireai.utils.metric_mlflow_manager import MLflowManager

mlflow = MLflowManager("http://localhost:8852")
experiment = mlflow.get_experiment("my_experiment")
run_id = mlflow.create_run(experiment.experiment_id)
mlflow.log_metric(run_id, "loss", 0.5, step=100)
```

**Integration**:
- Each RapidFire run maps to one MLflow run
- Parameters logged at run creation
- Metrics logged during training via callbacks
- Checkpoints logged as artifacts

### shm_manager.py
**Purpose**: Shared memory management for model checkpoints

**Key Responsibilities**:
- Creates shared memory registry (dict proxy) accessible across processes
- Stores model checkpoints in shared memory to avoid disk I/O
- Provides lock for thread-safe access
- Manages memory lifecycle (allocation, deallocation)

**Key Classes**:
- `SharedMemoryManager`: Main interface for shared memory operations

**Key Methods**:
- `get_shm_objects()`: Returns (registry, lock) tuple
- `store(key, value)`: Store object in shared memory
- `load(key)`: Load object from shared memory
- `delete(key)`: Remove object from shared memory
- `exists(key)`: Check if key exists

**Usage**:
```python
from rapidfireai.fit.utils.shm_manager import SharedMemoryManager

shm = SharedMemoryManager(name="controller-shm")
registry, lock = shm.get_shm_objects()

# Store model
with lock:
    registry[f"{run_id}_model"] = model_state_dict

# Load model
with lock:
    state_dict = registry.get(f"{run_id}_model")
```

**Key Concepts**:
- Uses multiprocessing.Manager() for shared dict
- Objects stored on CPU (tensors moved from GPU)
- Lock prevents concurrent access issues
- Fallback to disk if object too large or memory full

### experiment_utils.py
**Purpose**: Experiment lifecycle management utilities

**Key Responsibilities**:
- Creates experiment directories and metadata
- Generates unique experiment names
- Sets up signal handlers for graceful shutdown
- Manages experiment cleanup

**Key Methods**:
- `create_experiment(given_name, experiments_path)`: Create experiment directory and DB entry
- `setup_signal_handlers(worker_processes)`: Setup SIGINT/SIGTERM handlers
- `cleanup_experiment()`: Kill workers, reset DB state

**Usage**:
```python
from rapidfireai.fit.utils.experiment_utils import ExperimentUtils

utils = ExperimentUtils()
exp_id, exp_name, logs = utils.create_experiment("my_exp", "./experiments")
utils.setup_signal_handlers(worker_processes)
```

**Naming**:
- If "my_exp" exists, creates "my_exp_1", "my_exp_2", etc.
- Ensures unique experiment names across runs

### worker_manager.py
**Purpose**: Worker process lifecycle management

**Key Responsibilities**:
- Spawns Worker processes (one per GPU)
- Manages process pool
- Handles worker shutdown and cleanup
- Provides shutdown signals to workers

**Key Methods**:
- `spawn_workers(experiment_id, experiment_name)`: Create worker processes
- `shutdown_workers()`: Gracefully stop all workers
- `terminate_workers()`: Force kill workers

**Usage**:
```python
from rapidfireai.fit.utils.worker_manager import WorkerManager

manager = WorkerManager(num_workers=4, registry, lock)
manager.spawn_workers(experiment_id, experiment_name)
# ... training happens ...
manager.shutdown_workers()
```

**Shutdown Flow**:
1. Set shutdown_event flag
2. Wait for workers to finish current tasks (grace period)
3. Terminate if still running after timeout

### serialize.py
**Purpose**: Object serialization for database storage

**Key Responsibilities**:
- Serialize complex Python objects (models, datasets, configs) to bytes
- Deserialize bytes back to Python objects
- Handle torch tensors and other non-pickleable objects

**Key Functions**:
- `encode_payload(obj)`: Serialize object to bytes using dill
- `decode_db_payload(data)`: Deserialize bytes to object

**Usage**:
```python
from rapidfireai.fit.utils.serialize import encode_payload, decode_db_payload

config = {'learning_rate': 1e-5, 'batch_size': 8}
blob = encode_payload(config)
db.execute("INSERT INTO runs (config_leaf) VALUES (?)", (blob,))

row = db.execute("SELECT config_leaf FROM runs WHERE run_id = 1")[0]
config = decode_db_payload(row['config_leaf'])
```

**Notes**:
- Uses dill (more powerful than pickle)
- Handles torch.Tensors, datasets, lambdas, etc.
- BLOB storage in SQLite

### datapaths.py
**Purpose**: Centralized path management for experiment artifacts

**Key Responsibilities**:
- Generates consistent paths for checkpoints, datasets, logs
- Ensures directories exist
- Handles path construction for different artifact types

**Key Methods**:
- `initialize(experiment_name, base_path)`: Set up paths for experiment
- `checkpoint_path(run_id, chunk_id)`: Get checkpoint file path
- `dataset_path()`: Get dataset file path
- `log_path(logger_name)`: Get log file path

**Usage**:
```python
from rapidfireai.fit.utils.datapaths import DataPath

DataPath.initialize("my_experiment", "/path/to/experiments")
checkpoint_file = DataPath.checkpoint_path(run_id=5, chunk_id=2)
# Returns: /path/to/experiments/my_experiment/runs/run_5/checkpoints/checkpoint_chunk_2.pt
```

### exceptions.py
**Purpose**: Custom exception classes for RapidFire

**Exception Classes**:
- `ExperimentException`: Experiment-level errors
- `ControllerException`: Controller errors
- `WorkerException`: Worker errors
- `DispatcherException`: Dispatcher errors
- `DBException`: Database errors
- `AutoMLException`: AutoML errors
- `NoGPUsFoundException`: No GPUs available

**Usage**:
```python
from rapidfireai.fit.utils.exceptions import ControllerException

if num_gpus == 0:
    raise NoGPUsFoundException("No GPUs found while initializing controller.")
```

### automl_utils.py
**Purpose**: Utilities for AutoML algorithms

**Key Functions**:
- `get_runs(param_config, seed)`: Extract runs from AutoML algorithm or plain dict
- `get_flattened_config_leaf(config)`: Flatten RFModelConfig to dict

**Usage**:
```python
from rapidfireai.automl import get_runs

# Handles both AutoML instances and plain dicts
if isinstance(param_config, AutoMLAlgorithm):
    runs = get_runs(param_config, seed)
else:
    runs = [param_config]  # Single config
```

### trainer_config.py
**Purpose**: Configuration container for trainer initialization

**Key Class**:
- `TrainerConfig`: Dataclass holding all info needed by `create_trainer_instance()`

**Attributes**:
- `run_id`: Run identifier
- `worker_id`: GPU worker assignment
- `config_leaf`: Run configuration dict
- `experiment_name`: Experiment name
- `chunk_id`: Current chunk being trained
- `epoch`: Current epoch
- `metric_run_id`: Metric logger run ID

**Usage**:
```python
from rapidfireai.fit.utils.trainer_config import TrainerConfig

config = TrainerConfig(
    run_id=5,
    worker_id=0,
    config_leaf=config_dict,
    experiment_name="my_exp",
    chunk_id=2,
    epoch=0,
    metric_run_id="abc123"
)

trainer = create_trainer_instance(config, shm_manager)
```

### ping.py
**Purpose**: Server health check utility

**Usage**:
```python
python -m rapidfireai.fit.utils.ping
# Checks if dispatcher, mlflow, and frontend servers are running
```

## Common Patterns

### Logging Setup
```python
from rapidfireai.fit.utils.logging import RFLogger

# In each component:
logger = RFLogger().create_logger("component_name")
logger.info("Message")
```

### Shared Memory Access
```python
from rapidfireai.fit.utils.shm_manager import SharedMemoryManager

shm = SharedMemoryManager(name="controller-shm")
registry, lock = shm.get_shm_objects()

with lock:
    registry[key] = value  # Thread-safe
```

### Exception Handling
```python
from rapidfireai.fit.utils.exceptions import ControllerException

try:
    # ... operation ...
except Exception as e:
    raise ControllerException(f"Error: {e}") from e
```

### Path Management
```python
from rapidfireai.fit.utils.datapaths import DataPath

DataPath.initialize(experiment_name, base_path)
checkpoint = DataPath.checkpoint_path(run_id, chunk_id)
```

## Testing

Unit tests for utils:
```bash
pytest tests/  # Currently minimal, could expand
```

Most utils tested indirectly through integration tests.
