#!/bin/bash
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=anlp3_eval
#SBATCH --output=logs/part1/eval_%j.log

export PATH="$HOME/.local/bin:$PATH"
set -e

mkdir -p logs/part1
mkdir -p output/part1
mkdir -p results/part1

CONFIG=${1:-"config/part1/hybrid.yaml"}

echo "=== Running Inference on GPU ($CONFIG) ==="
uv run python -m src.part1_inference --config "$CONFIG"

echo "=== Evaluating Metrics ==="
uv run python -m src.part1_evaluate --pred output/part1/test_decrypt.jsonl --target data/part1/test/cleaned.jsonl

echo "=== Evaluation Complete ==="
