#!/bin/bash

# RapidFire AI Build and Install Script
# This script builds and installs the rapidfireai package

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Get the project root directory (parent of setup/fit/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root directory
cd "$PROJECT_ROOT"

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    print_error "This script must be run from the rapidfireai directory (pyproject.toml not found)"
    exit 1
fi

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]] && [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    print_warning "Not in a virtual environment. This may cause permission issues."
    print_status "Consider activating a virtual environment first:"
    print_status "  python3 -m venv .venv && source .venv/bin/activate"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Clean previous builds
print_status "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info/ .eggs/

# Install build dependencies
print_status "Installing build dependencies..."
pip install --upgrade pip setuptools wheel build

# Build the package
print_status "Building the package..."
python -m build

# Install the package
print_status "Installing the package..."
pip install dist/*.whl

print_success "Package installed successfully!"
print_status "You can now use the 'rapidfireai' command:"
print_status "  rapidfireai --help"
print_status "  rapidfireai start"
print_status "  rapidfireai status"
print_status "  rapidfireai stop"

# Test the installation
print_status "Testing the installation..."
if command -v rapidfireai &> /dev/null; then
    print_success "rapidfireai command is available"
    rapidfireai --version
else
    print_error "rapidfireai command not found in PATH"
    print_status "You may need to restart your shell or activate your virtual environment"
fi