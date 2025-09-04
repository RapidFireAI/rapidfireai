# Setup Scripts for ML/AI Development Environment

This directory contains scripts to set up a complete ML/AI development environment with Miniconda, Python 3.12, and essential packages for machine learning and data science.

## 🚀 Quick Start

### Step 1: Install Miniconda and Python 3.12
```bash
./setup_miniconda_python312.sh
```

### Step 2: Install ML Packages and Setup Jupyter
```bash
# Make sure conda base environment is active
conda activate base

# Install packages and setup Jupyter kernel
./install_ml_packages.sh
```

### Step 3: Start Jupyter
```bash
# Option 1: JupyterLab (recommended)
jupyter lab

# Option 2: Classic Jupyter Notebook
jupyter notebook
```

## 📁 Files Description

### `setup_miniconda_python312.sh`
- Downloads and installs Miniconda
- Sets up Python 3.12 in the base environment
- Handles conda initialization and Terms of Service acceptance
- Includes error checking and informative output

### `requirements.txt`
Contains all necessary packages for ML/AI development:
- **Core ML**: PyTorch, Transformers, PEFT, BitsAndBytes
- **HuggingFace**: Datasets, Hub, Accelerate, Evaluate
- **Jupyter**: JupyterLab, Notebook, IPykernel, IPywidgets
- **Data Science**: NumPy, Pandas, Matplotlib, Seaborn, Plotly
- **Utilities**: tqdm, rich, scikit-learn, scipy
- **Optional**: Weights & Biases, TensorBoard

### `install_ml_packages.sh`
- Installs all packages from requirements.txt
- Creates a Jupyter kernel for the conda base environment
- Verifies package installations
- Provides usage instructions

## 🔧 Manual Setup Alternative

If you prefer to run commands manually:

```bash
# 1. Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc

# 2. Accept Terms of Service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# 3. Install Python 3.12
conda install python=3.12 -y

# 4. Install packages
pip install -r requirements.txt

# 5. Setup Jupyter kernel
python -m ipykernel install --user --name="base" --display-name="Python (base)"
```

## 🎯 What You Get

After running these scripts, you'll have:
- ✅ Miniconda with Python 3.12
- ✅ All essential ML/AI packages installed
- ✅ Jupyter environment ready to use
- ✅ Proper kernel configuration for notebooks
- ✅ Progress bars, visualization tools, and utilities

## 🚀 Next Steps

1. Start JupyterLab: `jupyter lab`
2. Create a new notebook
3. Select the "Python (base)" kernel
4. Start coding! 

## 📚 Useful Commands

```bash
# Check installed packages
pip list

# Update a package
pip install --upgrade package_name

# List available Jupyter kernels
jupyter kernelspec list

# Check conda environment info
conda info
``` 