#!/bin/bash

# Complete Conda Environment Setup Script
# This script runs both conda installation and environment setup in sequence

set -e  # Exit on any error

echo "Starting complete conda and environment setup..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set the experiment path
export RF_EXPERIMENT_PATH="${SCRIPT_DIR}/../rapidfire_experiments"

# Step 1: Run conda installation script
echo "Step 1: Installing Miniconda and Python 3.10..."
echo "----------------------------------------"
bash "${SCRIPT_DIR}/1_setup_conda.sh"

echo ""
echo "Conda installation completed!"
echo ""

# Source conda to make it available in this shell
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

# Step 2: Run environment setup script
echo "Step 2: Setting up 'infer' conda environment..."
echo "----------------------------------------"
bash "${SCRIPT_DIR}/2_setup_conda_env.sh"

# Step 3: Install requirements from requirements.txt
echo ""
echo "Step 3: Installing requirements from requirements.txt..."
echo "----------------------------------------"
conda run -n infer pip3 install -r "${SCRIPT_DIR}/../requirements.txt"

echo ""
echo "Complete setup finished!"
echo ""
echo "Summary:"
echo "  - Miniconda installed with Python 3.10"
echo "  - 'infer' conda environment created"
echo "  - PyTorch 2.5.1 with CUDA 12.4 installed"
echo "  - All ML/AI packages installed"
echo "  - Requirements from requirements.txt installed"
echo ""
echo "To get started:"
echo "  conda activate infer"
echo ""
