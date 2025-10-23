
<div align="center">

<a href="https://rapidfire.ai"> 
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/RapidFireAI/rapidfireai/main/images/RapidFire-logo-for-dark-theme.svg">
        <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/RapidFireAI/rapidfireai/main/images/RapidFire-logo-for-light-theme.svg">
        <img alt="RapidFire AI" src="https://raw.githubusercontent.com/RapidFireAI/rapidfireai/main/images/RapidFire-logo-for-light-theme.svg">
    </picture>
</a>


</div>

[![PyPI version](https://img.shields.io/pypi/v/rapidfireai)](https://pypi.org/project/rapidfireai/)

# RapidFire AI

Rapid experimentation for easier, faster, and more impactful fine-tuning and post-training for LLMs and other DL models — delivering 16–24× higher throughput without extra GPUs.

## Overview

RapidFire AI is a new experiment execution framework that transforms your LLM customization experimentation from slow, sequential processes into rapid, intelligent workflows with hyperparallelized training, dynamic real-time experiment control, and automatic multi-GPU system orchestration.

![Usage workflow of RapidFire AI](https://raw.githubusercontent.com/RapidFireAI/rapidfireai/main/images/Workflow-transparent-2-01.png)

RapidFire AI’s adaptive execution engine enables interruptible, chunk-based scheduling so you can compare many configurations concurrently—even on a single GPU—with dynamic real-time control over runs.

- **Hyperparallelized Execution**: Higher throughput, simultaneous, data chunk-at-a-time training to show side-by-side differences.
- **Interactive control (IC Ops)**: Stop, Resume, Clone-Modify, and optionally warm start runs in real-time from the dashboard.
- **Automatic Optimization**: Intelligent single and multi-GPU orchestration to optimze utilization with minimal overhead.

![Chunk-based concurrent execution (1 GPU)](https://oss-docs.rapidfire.ai/en/latest/_images/gantt-1gpu.png)

For additional context, see the overview: [RapidFire AI Overview](https://oss-docs.rapidfire.ai/en/latest/overview.html)

## Getting Started

### Prerequisites

- [NVIDIA GPU using the 7.x or 8.x Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [NVIDIA CUDA Toolkit 11.8+](https://developer.nvidia.com/cuda-toolkit-archive)
- [Python 3.12.x](https://www.python.org/downloads/)
- [PyTorch 2.7.1+](https://pytorch.org/get-started/previous-versions/) with corresponding forward compatible prebuilt CUDA binaries

### Install and Get Started


```bash
# Ensure that python3 resolves to python3.12 if needed
python3 --version  # must be 3.12.x

python3 -m venv .venv
source .venv/bin/activate

pip install rapidfireai
export PATH="$HOME/.local/bin:$PATH"

rapidfireai --version
# Verify it prints the following:
# RapidFire AI 0.11.1

# Replace YOUR_TOKEN with your actual Hugging Face token
# https://huggingface.co/docs/hub/en/security-tokens
huggingface-cli login --token YOUR_TOKEN

# Due to current issue: https://github.com/huggingface/xet-core/issues/527
pip uninstall -y hf-xet

rapidfireai init
# It will install specific dependencies and initialize rapidfireai

# Start the rapidfireai server
# For Google Colab run:
#   export RF_TRACKING_BACKEND=tensorboard
#   rapidfireai start --colab
# For standalone run:

rapidfireai start
# It should print about 50 lines, including the following:
# ...
# RapidFire Frontend is ready
# Open your browser and navigate to: http://0.0.0.0:3000
# ...
# Press Ctrl+C to stop all services

# Open an example notebook from ./tutorial_notebooks and start experiment
```

### Troubleshooting

For a quick system diagnostics report (Python env, relevant packages, GPU/CUDA, and key environment variables), run:

```bash
rapidfireai doctor
```

If you encounter port conflicts, you can kill existing processes:

```bash
lsof -t -i:5002 | xargs kill -9  # mlflow
lsof -t -i:8081 | xargs kill -9  # dispatcher
lsof -t -i:3000 | xargs kill -9  # frontend server
```

## Documentation

Browse or reference the full documentation, example use case tutorials, all API details, dashboard details, and more in the [RapidFire AI Documentation](https://oss-docs.rapidfire.ai).

## Key Features

### MLflow Integration

Full MLflow support for experiment tracking and metrics visualization. A named RapidFire AI experiment corresponds to an MLflow experiment for comprehensive governance

### Interactive Control Operations (IC Ops)

First-of-its-kind dynamic real-time control over runs in flight. Can be invoked through the dashboard:

- Stop active runs; puts them in a dormant state
- Resume stopped runs; makes them active again
- Clone and modify existing runs, with or without warm starting from parent’s weights
- Delete unwanted or failed runs

### Multi-GPU Support

The Scheduler automatically handles multiple GPUs on the machine and divides resources across all running configs for optimal resource utilization.

### Search and AutoML Support

Built-in procedures for searching over configuration knob combinations, including Grid Search and Random Search. Easy to integrate with AutoML procedures. Native support for some popular AutoML procedures and customized automation of IC Ops coming soon.

## Directory Structure

```text
rapidfireai/
├── automl/          # Search and AutoML algorithms for knob tuning
├── backend/         # Core backend components (controller, scheduler, worker)
├── db/              # Database interface and SQLite operations
├── dispatcher/      # Flask-based web API for UI communication
├── frontend/         # Frontend components (dashboard, IC Ops implementation)
├── ml/              # ML training utilities and trainer classes
├── utils/           # Utility functions and helper modules
└── experiment.py    # Main experiment lifecycle management
```

## Architecture

RapidFire AI adopts a microservices-inspired loosely coupled distributed architecture with:

- **Dispatcher**: Web API layer for UI communication
- **Database**: SQLite for state persistence
- **Controller**: Central orchestrator running in user process
- **Workers**: GPU-based training processes
- **Dashboard**: Experiment tracking and visualization dashboard

This design enables efficient resource utilization while providing a seamless user experience for AI experimentation.

## Components

### Dispatcher

The dispatcher provides a REST API interface for the web UI. It can be run via Flask as a single app or via Gunicorn to have it load balanced. Handles interactive control features and displays the current state of the runs in the experiment.

### Database

Uses SQLite for persistent storage of metadata of experiments, runs, and artifacts. The Controller also uses it to talk with Workers on scheduling state. A clean asynchronous interface for all DB operations, including experiment lifecycle management and run tracking.

### Controller

Runs as part of the user’s console or Notebook process. Orchestrates the entire training lifecycle including model creation, worker management, and scheduling. The `run_fit` logic handles sample preprocessing, model creation for given knob configurations, worker initialization, and continuous monitoring of training progress across distributed workers.

### Worker

Handles the actual model training and inference on the GPUs. Workers poll the Database for tasks, load dataset chunks, and execute training runs with checkpointing and progress reporting. Currently expects any given model for given batch size to fit on a single GPU.

### Experiment

Manages the complete experiment lifecycle, including creation, naming conventions, and cleanup. Experiments are automatically named with unique suffixes if conflicts exist, and all experiment metadata is tracked in the Database. An experiment's running tasks are automatically cancelled when the process ends abruptly.

### Dashboard

A fork of MLflow that enables full tracking and visualization of all experiments and runs. It features a new panel for Interactive Control Ops that can be performed on any active runs.

## Developing with RapidFire AI

### Development prerequisites

- Python 3.12.x
- Git
- Ubuntu/Debian system (for apt package manager)

```bash
# Run these commands one after the other on a fresh Ubuntu machine

# install dependencies
sudo apt update -y

# clone the repository
git clone https://github.com/RapidFireAI/rapidfireai.git

# navigate to the repository
cd ./rapidfireai

# install basic dependencies
sudo apt install -y python3.12-venv
python3 -m venv .venv
source .venv/bin/activate
pip3 install ipykernel
pip3 install jupyter
pip3 install "huggingface-hub[cli]"
export PATH="$HOME/.local/bin:$PATH"
hf auth login --token <your_token>

# Due to current issue: https://github.com/huggingface/xet-core/issues/527
pip uninstall -y hf-xet

# checkout the main branch
git checkout main

# install the repository as a python package
pip3 install -r requirements.txt

# install node
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs

# Install correct version of vllm and flash-attn
# uv pip install vllm=0.10.1.1 --torch-backend=cu126 or cu118
# uv pip install flash-attn==1.0.9 --no-build-isoloation or 2.8.3

# if running into node versioning errors, remove the previous version of node then run the lines above again
sudo apt-get remove --purge nodejs libnode-dev libnode72 npm
sudo apt autoremove --purge

# check installations
node -v # 22.x

# still inside venv, run the start script to begin all 3 servers
chmod +x ./rapidfireai/start_dev.sh
./rapidfireai/start_dev.sh start

# run the notebook from within your IDE
# make sure the notebook is running in the .venv virtual environment
# head to settings in Cursor/VSCode and search for venv and add the path - $HOME/rapidfireai/.venv
# we cannot run a Jupyter notebook directly since there are restrictions on Jupyter being able to create child processes

# VSCode can port-forward localhost:3000 where the rf-frontend server will be running

# for port clash issues -
lsof -t -i:8081 | xargs kill -9 # dispatcher
lsof -t -i:5002 | xargs kill -9 # mlflow
lsof -t -i:3000 | xargs kill -9 # frontend
```

## Community & Governance

- Docs: [oss-docs.rapidfire.ai](https://oss-docs.rapidfire.ai)
- Discord: [Join our Discord](https://discord.gg/6vSTtncKNN)
- Contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- License: [`LICENSE`](LICENSE)
- Issues: use GitHub Issues for bug reports and feature requests
