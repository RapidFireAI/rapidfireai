#!/bin/bash

set -e  # Exit on any error

# Increase Node.js memory limit for webpack build
export NODE_OPTIONS="--max-old-space-size=8192"

# Build the frontend
cd rapidfireai/frontend
if [ "$1" != "build" ]; then
    node ./yarn/releases/yarn-4.9.1.cjs install
fi
if [ "$1" != "install" ]; then
    node ./yarn/releases/yarn-4.9.1.cjs build
fi
cd ../..