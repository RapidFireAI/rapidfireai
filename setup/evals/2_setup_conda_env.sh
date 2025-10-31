#!/bin/bash

# Configuration
ENV_NAME="infer"

# Exit on any error
set -e

echo "Setting up conda environment '$ENV_NAME' with Python 3.10..."

# Source conda if not already available
if ! command -v conda &> /dev/null; then
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    else
        echo "ERROR: Conda not found. Please run 1_setup_conda.sh first."
        exit 1
    fi
fi

# Accept conda terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create conda environment with Python 3.10
conda create -n $ENV_NAME python=3.10 -y

# Activate the environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing required packages..."
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install uv package manager for faster installations
pip install uv

# Install flashinfer-python
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5/

# Install ML/AI packages using uv for speed
uv pip install ray
uv pip install gpustat
uv pip install vllm==0.7.2 --torch-backend=cu124
uv pip install tensorboard
uv pip install ipywidgets ipykernel pandas pyarrow numpy datasets==3.6.0
uv pip install langchain langchain-core langchain-community unstructured langchain-openai langchain-huggingface faiss-gpu

# Install flash-attention
pip install flash-attn --no-build-isolation

# Install rf-inferno package in editable mode
echo "Installing rf-inferno package in editable mode..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
pip install -e "$PROJECT_ROOT"

echo "Environment setup complete!"