#!/bin/bash
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=anlp3_mamba
#SBATCH --output=logs/part1/train_mamba_%j.log

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
set -e

mkdir -p logs/part1
mkdir -p models/part1
mkdir -p output/part1

echo "=== Training Mamba (Stage 2 Warm-Start) ==="
uv sync
uv run python -m src.part1_train --config config/part1/mamba.yaml --init_checkpoint models/part1/mamba_v3_best.pt --seed 43
echo "=== Done ==="
