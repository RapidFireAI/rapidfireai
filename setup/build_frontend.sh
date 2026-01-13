#!/bin/bash

set -e  # Exit on any error

# Build the frontend
cd rapidfireai/frontend
if [ "$1" != "build" ]; then
    node ./yarn/releases/yarn-4.9.1.cjs install
fi
if [ "$1" != "install" ]; then
    node ./yarn/releases/yarn-4.9.1.cjs build
fi
cd ../..