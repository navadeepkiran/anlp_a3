#!/bin/bash
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=anlp3_part1_all
#SBATCH --output=logs/part1/train_all_%j.log

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
set -e

mkdir -p logs/part1
mkdir -p models/part1
mkdir -p output/part1

echo "=== Environment Info ==="
nvidia-smi

echo "=== Syncing Dependencies with uv ==="
uv sync

echo "=== Training All Models Sequentially ==="
bash scripts/part1/test_training.sh

echo "=== Training Complete ==="
