#!/bin/bash
set -e

echo "=== Creating branch ==="
git config --global --add safe.directory $(pwd)
git checkout -b assignment2_model_training 2>/dev/null || git checkout assignment2_model_training

echo "=== Creating directories ==="
mkdir -p src env jobs
