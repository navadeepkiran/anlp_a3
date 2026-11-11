#!/bin/bash
#SBATCH --job-name=anlp3_part2
#SBATCH --output=logs/part2/run_%j.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

export PATH="$HOME/.local/bin:$PATH"
set -e

mkdir -p logs/part2
mkdir -p output/part2

echo "=== Running Part 2 Standard Evaluation ==="
uv run python -m src.part2_evaluate

echo "=== Running Part 2 TA Evaluation ==="
uv run python -m src.part2_evaluate --ta

echo "=== Part 2 Complete ==="

