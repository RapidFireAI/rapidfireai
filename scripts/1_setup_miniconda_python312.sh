#!/bin/bash

# Miniconda Installation and Python 3.12 Setup Script
# This script installs miniconda and sets up Python 3.12 in the base environment

echo "🐍 Starting Miniconda installation and Python 3.12 setup..."

# Step 1: Check if conda is already installed
if command -v conda &> /dev/null; then
    echo "⚠️  Conda is already installed. Skipping installation..."
    echo "Current conda version: $(conda --version)"
else
    echo "📥 Downloading Miniconda installer..."
    # Step 2: Download Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    
    echo "🔧 Installing Miniconda..."
    # Step 3: Install Miniconda in batch mode
    bash ~/miniconda.sh -b -p $HOME/miniconda3
    
    echo "⚙️  Initializing conda for bash..."
    # Step 4: Initialize conda for bash
    $HOME/miniconda3/bin/conda init bash
    
    echo "🧹 Cleaning up installer file..."
    # Step 5: Remove installer file
    rm ~/miniconda.sh
    
    echo "🔄 Sourcing bashrc to activate conda..."
    # Step 6: Source bashrc to activate conda (this needs to be done manually)
    echo "⚠️  Please run 'source ~/.bashrc' or restart your terminal to activate conda"
fi

# Step 7: Accept Terms of Service for conda channels
echo "📋 Accepting conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Step 8: Install Python 3.12 in base environment
echo "🐍 Installing Python 3.12 in base environment..."
conda install python=3.12 -y

# Step 9: Verify Python version
echo "✅ Installation complete! Verifying Python version..."
python --version

echo "🎉 Setup complete! Python 3.12 is now installed in your conda base environment."
echo ""
echo "🔧 Manual steps required:"
echo "  1. If conda was just installed, run: source ~/.bashrc"
echo "  2. Activate base environment: conda activate base"
echo ""
echo "📝 Commands that were executed:"
echo "  - wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh"
echo "  - bash ~/miniconda.sh -b -p \$HOME/miniconda3"
echo "  - \$HOME/miniconda3/bin/conda init bash"
echo "  - source ~/.bashrc"
echo "  - conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main"
echo "  - conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r"
echo "  - conda install python=3.12 -y"
echo "  - rm ~/miniconda.sh" 