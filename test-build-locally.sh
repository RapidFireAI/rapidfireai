#!/bin/bash

# Test script to reproduce GitHub Action build steps locally
# This mimics the same steps as the deploy-testpypi.yml workflow

set -e  # Exit on any error

echo "🧪 Testing GitHub Action build steps locally..."
echo ""

# Step 1: Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd rapidfireai/frontend
node ./yarn/releases/yarn-4.9.1.cjs install --frozen-lockfile
echo "✅ Frontend dependencies installed"
echo ""

# Step 2: Build frontend
echo "🏗️  Building frontend..."
node ./yarn/releases/yarn-4.9.1.cjs build
echo "✅ Frontend build completed"
echo ""

# Step 3: Go back to root and clean previous builds
echo "🧹 Cleaning previous builds..."
cd ../..
rm -rf dist/ *.egg-info/ .eggs/
echo "✅ Previous builds cleaned"
echo ""

# Step 4: Build Python package
echo "🐍 Building Python package..."
python -m pip install --upgrade pip > /dev/null 2>&1
pip install build twine > /dev/null 2>&1
python -m build
echo "✅ Python package built"
echo ""

echo "🎉 All build steps completed successfully!"
echo ""
echo "📋 Summary:"
echo "   - Frontend dependencies: ✅"
echo "   - Frontend build: ✅" 
echo "   - Python package build: ✅"
echo ""
echo "💡 Your build should now work in GitHub Actions!"
