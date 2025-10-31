#!/bin/bash

# Miniconda Installation and Python 3.10 Setup Script
# This script installs miniconda and sets up Python 3.10 in the base environment

echo "Starting Miniconda installation and Python 3.10 setup..."

# Step 1: Check if conda is already installed
if command -v conda &> /dev/null; then
    echo "WARNING: Conda is already installed. Skipping installation..."
    echo "Current conda version: $(conda --version)"
else
    echo "Downloading and installing Miniconda..."
    # Download Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

    # Install Miniconda in batch mode
    bash ~/miniconda.sh -b -p $HOME/miniconda3

    # Initialize conda for bash
    $HOME/miniconda3/bin/conda init bash

    # Remove installer file
    rm ~/miniconda.sh

    # Source bashrc to activate conda automatically
    source ~/.bashrc

    # Also source conda profile directly to ensure it's available
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi

# Check if conda is available, if not source it
if ! command -v conda &> /dev/null; then
    echo "Conda not found in PATH, sourcing conda setup..."
    source $HOME/miniconda3/etc/profile.d/conda.sh
fi

# Accept Terms of Service for conda channels
echo "Accepting conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install Python 3.10 in base environment
echo "Installing Python 3.10 in base environment..."
conda install python=3.10 -y

# Verify Python version
python --version

echo "Setup complete! Python 3.10 is now installed in your conda base environment."