# AGENTS.md

Canonical guide for AI coding agents (Claude Code, Cursor, OpenAI Codex CLI, GitHub Copilot, Windsurf, Aider, Junie, and any future tool following the `AGENTS.md` convention) working *on* `rapidfireai`. Auto-loaded at the repo root by every agent that follows this convention. Claude Code is routed here from [`CLAUDE.md`](CLAUDE.md), which is a thin pointer back to this file.

## Audience and related files

This file is the **canonical maintainer guide** for working *on* `rapidfireai`. Its companions:

- [`CLAUDE.md`](CLAUDE.md) at repo root — thin router for Claude Code, which auto-loads files named `CLAUDE.md` by convention. It points back to this file as the source of truth.
- [`docs/AGENTS.md`](docs/AGENTS.md) — **end-user-facing** integration guide, read by AI agents helping a user install or use the package. *Not* for contributors. Placed under `docs/` so maintainer auto-loaders do not pick it up as project rules. Update triggers are listed in [Keeping `docs/AGENTS.md` in sync](#keeping-docsagentsmd-in-sync) below.

### Scope guard

If your task is "install rapidfireai for user X" or "integrate rapidfireai into user's project," you are in the wrong place — you should be inside the user's project, not this repo. Being inside this repo means you are doing maintenance work on `rapidfireai` itself, and this file is correct for that. If you somehow loaded this as project rules while assisting an end user, fetch [`docs/AGENTS.md`](docs/AGENTS.md) instead.

## Project Overview

RapidFire AI is an experiment execution framework for LLM fine-tuning and post-training that enables hyperparallelized training, dynamic real-time experiment control (IC Ops), and automatic multi-GPU orchestration. The system uses chunk-based scheduling to allow concurrent training of multiple configurations even on a single GPU.

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
chmod +x ./rapidfireai/start_dev.sh
./rapidfireai/start_dev.sh start

# Stop services
./rapidfireai/start_dev.sh stop
```

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
./bump_version.sh patch  # 0.10.1 → 0.10.2
./bump_version.sh minor  # 0.10.1 → 0.11.0
./bump_version.sh major  # 0.10.1 → 1.0.0

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

RapidFire AI uses a microservices-inspired distributed architecture:

### Core Components

1. **Experiment** (`experiment.py`): Top-level API for users. Manages experiment lifecycle, creates database tables, sets up logging and signal handlers. Entry point for `run_fit()` and `get_results()`.

2. **Controller** (`backend/controller.py`): Orchestrates the entire training lifecycle. Runs in the user's process. Responsible for:
   - Creating models from parameter configurations
   - Initializing and managing Workers
   - Running the Scheduler to assign chunks to workers
   - Handling Interactive Control Operations (IC Ops)
   - Monitoring training progress

3. **Scheduler** (`backend/scheduler.py`): Pure scheduling logic that assigns runs to available workers for specific chunks. Uses round-robin and fairness algorithms to ensure optimal GPU utilization. Tracks which runs have completed which chunks.

4. **Worker** (`backend/worker.py`): Separate GPU processes that execute actual training. Each worker:
   - Polls database for assigned tasks
   - Loads model checkpoints from shared memory or disk
   - Trains on assigned data chunks
   - Saves checkpoints back to shared memory/disk
   - Reports progress to MLflow

5. **Dispatcher** (`dispatcher/dispatcher.py`): Flask-based REST API for UI communication. Provides endpoints for:
   - Viewing experiment status
   - Interactive Control Operations (stop, resume, clone, delete runs)
   - Real-time run metrics

6. **Database** (`db/rf_db.py`): SQLite-based persistence layer with async operations. Stores:
   - Experiment metadata
   - Run configurations and status
   - Task scheduling state
   - Checkpoint locations

7. **Frontend** (`frontend/`): React-based dashboard (MLflow fork) with IC Ops panel. Displays live experiment tracking and enables dynamic control.

### Data Flow

1. User creates `Experiment` and calls `run_fit()` with configs and datasets
2. Controller creates runs in database and spawns Worker processes
3. Controller runs Scheduler loop to assign (run_id, chunk_id) to available workers
4. Workers poll database, load models, train on chunks, save checkpoints
5. Workers report metrics to MLflow and update database task status
6. Scheduler continues until all runs complete all chunks (epochs)
7. User can invoke IC Ops through UI to stop/resume/clone runs mid-training

### Shared Memory System

RapidFire uses shared memory (`utils/shm_manager.py`) to avoid disk I/O bottlenecks:
- Model checkpoints stored in shared memory between chunks (configurable via `USE_SHARED_MEMORY`)
- Registry tracks which models are in memory
- Process locks prevent concurrent access issues
- Fallback to disk for larger models

### Interactive Control (IC Ops)

Unique feature enabling real-time experiment control:
- **Stop**: Pause a run, saves checkpoint
- **Resume**: Restart a stopped run from checkpoint
- **Clone**: Create new run from existing, optionally warm-start from parent's weights
- **Delete**: Remove unwanted runs

Implemented via database state changes that Controller/Workers poll.

## Directory Structure

```
rapidfireai/
├── automl/           # Grid search, random search, AutoML algorithms
├── backend/          # Controller, Scheduler, Worker, Chunks
├── db/               # SQLite database interface
├── dispatcher/       # Flask REST API for UI
├── frontend/         # React dashboard (MLflow fork with IC Ops)
├── ml/               # Trainer classes, checkpoint utils, callbacks
├── utils/            # Logging, MLflow manager, shared memory, serialization
├── experiment.py     # Main Experiment class (user-facing API)
├── cli.py            # CLI commands (rapidfireai start/stop/init/doctor)
├── start.sh          # Production server startup script
├── start_dev.sh      # Development mode startup script
└── version.py        # Version number
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

Defined in `db/*.sql` files. Tables include:
- experiments: Experiment metadata and paths
- runs: Run configurations, status, metrics
- tasks: Task queue for controller-worker coordination
- checkpoints: Checkpoint locations and metadata

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

Current branch: `feat/enable-colab`
Main branch: `main`

Use standard PR workflow to merge features into main.

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

## Keeping `docs/AGENTS.md` in sync

`docs/AGENTS.md` is an end-user-facing operational guide read by AI coding agents (Claude Code, Cursor, Codex, GitHub Copilot, Windsurf, Aider, Junie) when they help a user install or integrate `rapidfireai`. It is **not** for contributors — for repo-internal architecture, build, and contributor flow, *this* file (`AGENTS.md`) is authoritative.

By design, `docs/AGENTS.md` defers all volatile install content — Python version, the `pip install` line, HF auth syntax, the `hf-xet` workaround, port numbers, `ssh -L` syntax, kill-by-port commands — to `README.md` and references the README by section heading. **A README install-step change therefore does NOT require an AGENTS.md update.** That decoupling is intentional and prevents drift.

`docs/AGENTS.md` *does* inline the stable layer the README does not own. **Update `docs/AGENTS.md` whenever you change any of the following**, in the same PR:

- **Public API of `rapidfireai.Experiment`** — the `__init__` signature (especially the `mode` parameter), or any of `run_fit`, `run_evals`, `get_results`, `cancel_current`, `end`, `get_log_file_path`. Reflected in AGENTS.md §5 (Code Integration Patterns) and §7 (Experiment Lifecycle).
- **Top-level exports** in `rapidfireai/__init__.py` or **AutoML exports** in `rapidfireai/automl/__init__.py` (e.g., `RFGridSearch`, `RFRandomSearch`, `List`, `Range`, `RFModelConfig`, `RFLoraConfig`, `RFSFTConfig`, `RFDPOConfig`, `RFGRPOConfig`, `RFvLLMModelConfig`, `RFAPIModelConfig`, `RFLangChainRagSpec`, `RFPromptManager`). Reflected in §5 and §6.
- **Trainer-type taxonomy** in `rapidfireai/fit/ml/trainer.py` — the `trainer_type` → trainer-class mapping (currently `SFT` / `DPO` / `GRPO`). Reflected in §2 (Workflow Decision Tree) and §6.
- **User-supplied callback signatures** — `create_model_fn`, `formatting_func`, `compute_metrics`, `preprocess_fn`, `postprocess_fn`, `compute_metrics_fn`, `accumulate_metrics_fn`. Reflected in §5.
- **CLI subcommand structure** — adding, renaming, or removing `init`, `init --evals`, `start`, `stop`, `doctor`, `jupyter`, `--version`, or changing the dependency-set choice between fine-tuning and RAG/eval. Reflected in §2 and §3 (Setup Order).
- **Tutorial taxonomy** — the lite/full split, the OpenAI-vs-self-hosted classification, or filenames under `tutorial_notebooks/{fine-tuning,post-training,rag-contexteng}/` referenced in the §9 (Tutorials) table.
- **README section headings** — if you rename `§Prerequisites`, `§Install and Get Started`, or `§Troubleshooting` in `README.md`, update the cross-references in `docs/AGENTS.md` (§1, §3, §8) so they still resolve.

`docs/AGENTS.md` must **never** be allowed to inline the volatile content listed above (Python version literals, `pip install` lines, port numbers, etc.). Drift-check by greping `docs/AGENTS.md` for `3.12.x`, `python3 -m venv`, `pip install rapidfireai`, `pip uninstall -y hf-xet`, `hf auth login --token`, `lsof -t -i:`, or bare port numbers `8850`–`8855` — there should be zero matches. If any appear, move the specific value to `README.md` and have AGENTS.md reference it by section instead.

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
