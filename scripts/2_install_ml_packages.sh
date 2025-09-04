#!/bin/bash

# ML Packages Installation and Jupyter Kernel Setup Script
# This script installs ML/AI packages and creates a Jupyter kernel for the conda base environment

echo "📦 Starting ML packages installation and Jupyter kernel setup..."

# Check if we're in a conda environment
if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
    echo "⚠️  Warning: No conda environment detected. Please activate conda first:"
    echo "   Run: conda activate base"
    exit 1
fi

echo "✅ Conda environment detected: ${CONDA_DEFAULT_ENV}"

# Check if we're in the correct directory (should have requirements.txt)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    echo "❌ Error: requirements.txt not found in ${SCRIPT_DIR}"
    echo "   Please ensure requirements.txt exists in the same directory as this script."
    exit 1
fi

echo "📋 Found requirements.txt at: $REQUIREMENTS_FILE"

# Update pip to latest version
echo "🔄 Updating pip to latest version..."
pip install --upgrade pip

# Install packages from requirements.txt
echo "📦 Installing packages from requirements.txt..."
echo "   This may take several minutes..."
pip install -r "$REQUIREMENTS_FILE"

# Check if installation was successful
if [[ $? -ne 0 ]]; then
    echo "❌ Error: Package installation failed!"
    echo "   Please check the error messages above and try again."
    exit 1
fi

echo "✅ Package installation completed successfully!"

# Install/update ipykernel for the current environment
echo "🔧 Setting up Jupyter kernel for conda base environment..."

# Get the current conda environment name and python path
CURRENT_ENV_NAME="${CONDA_DEFAULT_ENV}"
PYTHON_PATH=$(which python)

echo "   Environment: $CURRENT_ENV_NAME"
echo "   Python path: $PYTHON_PATH"

# Install the kernel
python -m ipykernel install --user --name="$CURRENT_ENV_NAME" --display-name="Python ($CURRENT_ENV_NAME)"

# Check if kernel installation was successful
if [[ $? -eq 0 ]]; then
    echo "✅ Jupyter kernel installed successfully!"
else
    echo "❌ Error: Jupyter kernel installation failed!"
    exit 1
fi

# List available kernels
echo "📋 Available Jupyter kernels:"
jupyter kernelspec list

# Verify key packages
echo "🔍 Verifying key package installations..."
python -c "
import sys
packages_to_check = [
    'torch', 'transformers', 'peft', 'bitsandbytes', 
    'datasets', 'huggingface_hub', 'jupyter', 'ipykernel'
]

print('Package verification:')
for package in packages_to_check:
    try:
        __import__(package)
        print(f'  ✅ {package}')
    except ImportError:
        print(f'  ❌ {package} - NOT FOUND')
        sys.exit(1)

print('\\n🎉 All key packages verified successfully!')
"

if [[ $? -ne 0 ]]; then
    echo "❌ Some packages failed verification. Please check the installation."
    exit 1
fi

echo ""
echo "🎉 Setup complete! Your ML development environment is ready."
echo ""
echo "🚀 Next steps:"
echo "  1. Start Jupyter Lab: jupyter lab"
echo "  2. Or start Jupyter Notebook: jupyter notebook"
echo "  3. Create a new notebook and select the kernel: Python ($CURRENT_ENV_NAME)"
echo ""
echo "📝 What was installed:"
echo "  • Core ML libraries: PyTorch, Transformers, PEFT, BitsAndBytes"
echo "  • HuggingFace ecosystem: Datasets, Hub, Accelerate, Evaluate"
echo "  • Jupyter ecosystem: JupyterLab, Notebook, IPykernel, IPywidgets"
echo "  • Data science tools: NumPy, Pandas, Matplotlib, Seaborn, Plotly"
echo "  • Utilities: tqdm, rich, scikit-learn, scipy"
echo "  • Optional: Weights & Biases, TensorBoard"
echo ""
echo "📚 Useful commands:"
echo "  • Check installed packages: pip list"
echo "  • Update a package: pip install --upgrade <package_name>"
echo "  • List Jupyter kernels: jupyter kernelspec list" 