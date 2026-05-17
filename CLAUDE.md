# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RapidFire AI is an experiment execution framework for LLM fine-tuning and post-training that enables hyperparallelized training, dynamic real-time experiment control (IC Ops), and automatic multi-GPU orchestration. The system uses chunk-based scheduling to allow concurrent training of multiple configurations even on a single GPU.

> For an in-depth code walkthrough with `file:line` citations, see [`docs/DEVELOPER.md`](./docs/DEVELOPER.md). For a visual overview of the flow, see [`docs/code-flow.md`](./docs/code-flow.md).

## Key Commands

### Development Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies from source
pip install -r requirements.txt

# Install Node.js 22.x and build frontend
cd rapidfireai/frontend
node ./yarn/releases/yarn-4.9.1.cjs install
node ./yarn/releases/yarn-4.9.1.cjs build
cd ../..

# Start all services in development mode
chmod +x ./setup/fit/start_dev.sh
./setup/fit/start_dev.sh start

# Stop services
./setup/fit/start_dev.sh stop
```

> Note: `setup/start.sh` is the production startup script invoked by the installed `rapidfireai start` CLI command. `setup/fit/start_dev.sh` is for running from a source checkout.

### Running from Installed Package

```bash
# Initialize RapidFire (installs dependencies, copies tutorials)
rapidfireai init

# Start RapidFire servers (dispatcher, mlflow, frontend)
rapidfireai start

# Stop all servers
rapidfireai stop

# System diagnostics (GPU, CUDA, Python env)
rapidfireai doctor

# Check version
rapidfireai --version
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_chunks.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code with ruff (line-length: 120)
ruff format .

# Run linter
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Building and Releasing

```bash
# Build PyPI package (requires frontend build first)
rm -rf dist/ *.egg-info/ .eggs/ && python -m build

# Bump version (creates commit and tag)
./setup/bump_version.sh patch  # 0.10.1 → 0.10.2
./setup/bump_version.sh minor  # 0.10.1 → 0.11.0
./setup/bump_version.sh major  # 0.10.1 → 1.0.0

# Push version tag to trigger TestPyPI deployment
git push origin test0.10.2
```

### Port Management

```bash
# Kill services on specific ports if conflicts occur
lsof -t -i:8851 | xargs kill -9  # dispatcher
lsof -t -i:8852 | xargs kill -9  # mlflow
lsof -t -i:8853 | xargs kill -9  # frontend
```

## Architecture

RapidFire AI is split into two pipelines under the `rapidfireai/` package:

- **`fit/`** — the training pipeline (the focus of most development).
- **`evals/`** — a separate evaluation pipeline (actors, RAG, metric aggregation).

Both pipelines share the same architectural pattern: a Controller in the user process, Worker subprocesses per GPU, a SQLite-backed task queue, a Flask dispatcher, and the React frontend. The components below describe the **fit** pipeline.

### Core Components

1. **Experiment** (`rapidfireai/experiment.py`): Top-level API for users. Manages experiment lifecycle, creates database tables, sets up logging and signal handlers. Entry point for `run_fit()` and `get_results()`.

2. **Controller** (`rapidfireai/fit/backend/controller.py`): Orchestrates the entire training lifecycle. Runs in the user's process. Responsible for:
   - Creating models from parameter configurations
   - Initializing and managing Workers
   - Running the Scheduler to assign chunks to workers
   - Handling Interactive Control Operations (IC Ops)
   - Monitoring training progress

3. **Scheduler** (`rapidfireai/fit/backend/scheduler.py`): Pure scheduling logic that assigns runs to available workers for specific chunks. Uses Monte-Carlo simulated fairness algorithms to ensure optimal GPU utilization. Tracks which runs have completed which chunks.

4. **Worker** (`rapidfireai/fit/backend/worker.py`): Separate GPU processes that execute actual training. Each worker:
   - Polls database for assigned tasks
   - Loads model checkpoints from shared memory or disk
   - Trains on assigned data chunks
   - Saves checkpoints back to shared memory/disk
   - Reports progress to MLflow

5. **Dispatcher** (`rapidfireai/fit/dispatcher/dispatcher.py`): Flask-based REST API for UI communication. Provides endpoints for:
   - Viewing experiment status
   - Interactive Control Operations (stop, resume, clone, delete runs)
   - Real-time run metrics

6. **Database** (`rapidfireai/fit/db/rf_db.py`, schema in `rapidfireai/fit/db/tables.sql`): SQLite-based persistence layer. Stores:
   - Experiment metadata
   - Run configurations and status
   - Task scheduling state
   - Checkpoint locations

7. **Frontend** (`rapidfireai/frontend/`): React-based dashboard (MLflow fork) with IC Ops panel. Displays live experiment tracking and enables dynamic control.

### Data Flow

1. User creates `Experiment` and calls `run_fit()` with configs and datasets
2. Controller creates runs in database and spawns Worker processes
3. Controller runs Scheduler loop to assign (run_id, chunk_id) to available workers
4. Workers poll database, load models, train on chunks, save checkpoints
5. Workers report metrics to MLflow and update database task status
6. Scheduler continues until all runs complete all chunks (epochs)
7. User can invoke IC Ops through UI to stop/resume/clone runs mid-training

### Shared Memory System

RapidFire uses shared memory (`rapidfireai/fit/utils/shm_manager.py`) to avoid disk I/O bottlenecks:
- Model checkpoints stored in shared memory between chunks (configurable via `USE_SHARED_MEMORY`)
- Registry tracks which models are in memory
- Process locks prevent concurrent access issues
- Fallback to disk on the final chunk and for larger models

### Interactive Control (IC Ops)

Unique feature enabling real-time experiment control:
- **Stop**: Pause a run, saves checkpoint
- **Resume**: Restart a stopped run from checkpoint
- **Clone**: Create new run from existing, optionally warm-start from parent's weights
- **Delete**: Remove unwanted runs

Implemented via database state changes that Controller/Workers poll.

## Directory Structure

```
rapidfireai/                    # Repo root
├── rapidfireai/                # Python package
│   ├── experiment.py           # Main Experiment class (user-facing API)
│   ├── cli.py                  # CLI commands (rapidfireai start/stop/init/doctor)
│   ├── version.py              # Version number
│   ├── automl/                 # Grid search, random search
│   ├── fit/                    # Training pipeline
│   │   ├── backend/            # Controller, Scheduler, Worker, Chunks
│   │   ├── db/                 # SQLite interface (rf_db.py, tables.sql)
│   │   ├── dispatcher/         # Flask REST API for UI
│   │   ├── ml/                 # Trainer, callbacks, checkpoint utils
│   │   └── utils/              # SHM, worker manager, IC controller, logging
│   ├── evals/                  # Evaluation pipeline (separate from fit)
│   │   ├── actors/             # Doc/Query/RateLimiter actors, inference engines
│   │   ├── data/               # Dataset loaders
│   │   ├── db/                 # Eval-specific DB interface + schema
│   │   ├── dispatcher/         # Eval Flask REST API
│   │   ├── metrics/            # Aggregator, online strategies
│   │   ├── rag/                # RAG pipeline + prompt manager
│   │   ├── scheduling/         # Eval controller/scheduler/pipeline scheduler
│   │   └── utils/              # Eval helpers (gateway, mlflow, ratelimiter, ...)
│   ├── frontend/               # React dashboard (MLflow fork with IC Ops)
│   └── utils/                  # CLI/system helpers (doctor, gpu_info, colab, ping)
├── setup/                      # Install + ops scripts
│   ├── start.sh                # Production startup (used by `rapidfireai start`)
│   ├── bump_version.sh         # Version bumping
│   ├── build_frontend.sh       # Frontend build helper
│   ├── fit/                    # Fit-side install scripts + start_dev.sh
│   └── evals/                  # Evals-side install scripts
├── docs/                       # Developer docs (DEVELOPER.md, code-flow.md, ...)
├── tests/                      # pytest suite
├── tutorial_notebooks/         # User-facing tutorials
├── community_notebooks/
├── requirements.txt
└── requirements-dev.txt
```

## Key Concepts

### Chunk-Based Training

Instead of training one model at a time for full epochs, RapidFire splits datasets into chunks and interleaves training:
- Dataset divided into N chunks (user configurable)
- Multiple runs train on different chunks concurrently
- Scheduler ensures fair distribution across GPUs
- Enables side-by-side comparison of hyperparameters with minimal latency

### Run Configuration

Runs are created from parameter configurations:
- Single dict: creates one run
- AutoML algorithms (GridSearch, RandomSearch): create multiple runs
- Each run gets unique ID, tracked in database
- Supports warm starting from parent runs (clone-modify)

### Task System

Database tracks tasks for coordination:
- **ExperimentTask**: High-level experiment state
- **ControllerTask**: Controller operations (create_models, schedule, etc.)
- **WorkerTask**: Worker operations (fit, validate, etc.)
- Status: PENDING → IN_PROGRESS → COMPLETED/FAILED

## MLflow Integration

RapidFire wraps MLflow for experiment tracking:
- Each RapidFire Experiment maps to an MLflow experiment
- Runs tracked with metrics, parameters, artifacts
- Checkpoints saved as MLflow artifacts
- UI extends MLflow with IC Ops panel
- Access MLflow directly at `http://localhost:8852`

## Development Notes

### Python Version

Requires Python 3.12.x (specified in pyproject.toml and README).

### Frontend Development

The frontend is a fork of MLflow. For frontend-specific guidance, see `rapidfireai/frontend/CLAUDE.md`.

To run frontend in development mode with hot reload:
```bash
cd rapidfireai/frontend
node ./yarn/releases/yarn-4.9.1.cjs start  # Runs on localhost:8853
```

### Database Schema

Defined in `rapidfireai/fit/db/tables.sql` (and `rapidfireai/evals/db/tables.sql` for evals). Tables include:
- `experiments`: Experiment metadata, current task, status
- `runs`: Run configurations (`config_leaf` BLOB), status, completed/total steps, MLflow run ID
- `worker_task`: Queue between Controller and Workers — `(worker_id, run_id, chunk_id, status, multi_worker_details)`
- `interactive_control`: IC Ops queue, drained by the Controller
- `controller_progress`, `worker_progress`: liveness/progress counters

### Environment Variables

- `RF_EXPERIMENT_PATH`: Base path for experiments (default: `./rapidfire_experiments`)
- `RF_TUTORIAL_PATH`: Path for tutorial notebooks (default: `./tutorial_notebooks`)
- `RF_MLFLOW_HOST`: MLflow tracking server Host (default: `localhost`)
- `RF_MLFLOW_PORT`: MLflow tracking server Port (default: `8852`)
- `USE_SHARED_MEMORY`: Enable shared memory for checkpoints (default: True)

### Logging

Multi-logger system using loguru:
- `experiment`: Experiment-level logs
- `controller`: Controller operations
- `worker_{N}`: Per-worker training logs
- `user`: User-facing messages
- `interactive-control`: IC Ops operations

Logs written to experiment directory.

### Testing Notebooks

Tutorial notebooks in `tutorial_notebooks/` demonstrate usage:
- Require HuggingFace token for model downloads
- Run via `jupyter notebook` or IDE with proper kernel
- Cannot run directly from CLI due to multiprocessing restrictions

## Common Patterns

### Creating an Experiment

```python
from rapidfireai import Experiment

exp = Experiment("my_experiment")
exp.run_fit(
    param_config=config_dict_or_automl,
    create_model_fn=my_model_factory,
    train_dataset=train_data,
    eval_dataset=eval_data,
    num_chunks=8,
    seed=42
)
results_df = exp.get_results()
```

### Defining Model Factory

```python
def create_model_fn(config):
    # config contains hyperparameters for this run
    model = YourModel(**config)
    return model, optimizer, loss_fn, trainer_config
```

### AutoML Usage

```python
from rapidfireai.automl import GridSearch

param_config = GridSearch({
    'learning_rate': [1e-4, 1e-5, 1e-6],
    'batch_size': [8, 16],
    'epochs': [3]
})
```

## Git Workflow

Main branch: `main`

Use standard PR workflow to merge feature branches into `main`.

## Dependencies

Core dependencies (see pyproject.toml for full list):
- torch >= 2.8.0
- transformers >= 4.55.2
- peft >= 0.17.0
- trl == 0.21.0
- mlflow >= 3.2.0
- flask >= 3.1.1

Dev dependencies:
- pytest >= 8.4.1
- black >= 21.0
- ruff (via ruff.toml)
- mypy >= 0.800

## README Guidelines

### Image URLs Must Be Absolute

Always use absolute URLs for images in `README.md`, not relative paths. The README is rendered on multiple platforms (GitHub, PyPI, npm, etc.), and relative paths only work on GitHub where the repository file structure is accessible.

**Correct:**
```markdown
<img src="https://raw.githubusercontent.com/RapidFireAI/rapidfireai/main/docs/images/example.svg">
```

**Incorrect:**
```markdown
<img src="docs/images/example.svg">
```

Use the pattern `https://raw.githubusercontent.com/RapidFireAI/rapidfireai/main/...` for all image references to ensure cross-platform compatibility.

## Troubleshooting

### GPU Issues

Run `rapidfireai doctor` to diagnose:
- CUDA installation
- GPU availability
- Driver version compatibility

### Port Conflicts

Common ports:
- 8853: Frontend dashboard
- 8852: MLflow tracking server
- 8851: Dispatcher API

Use port killing commands above if conflicts occur.

### Multiprocessing Issues

RapidFire uses `spawn` method for multiprocessing. Notebooks must be run through IDE or Jupyter, not CLI.
