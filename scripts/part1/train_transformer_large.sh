#!/bin/bash
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=tr_large
#SBATCH --output=logs/part1/train_transformer_large_%j.log

export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
set -e

mkdir -p logs/part1
mkdir -p models/part1
mkdir -p output/part1
mkdir -p results/part1

echo "=== Training Transformer Large (96k steps) ==="
uv run python -m src.part1_train --config config/part1/transformer_large.yaml --seed 42

echo "=== Evaluating on Validation Set ==="
uv run python -m src.part1_inference --config config/part1/transformer_large.yaml --no_hf --input_file data/part1/validation/cipher.jsonl --output_file output/part1/transformer_large_val.jsonl

uv run python -m src.part1_evaluate --pred output/part1/transformer_large_val.jsonl --target data/part1/validation/cleaned.jsonl --model_name transformer_large_val --config config/part1/transformer_large.yaml --output_json results/part1/transformer_large_val.json

echo "=== Evaluating on Test Set ==="
uv run python -m src.part1_inference --config config/part1/transformer_large.yaml --no_hf --output_file output/part1/transformer_large_test.jsonl

uv run python -m src.part1_evaluate --pred output/part1/transformer_large_test.jsonl --target data/part1/test/cleaned.jsonl --model_name transformer_large --config config/part1/transformer_large.yaml --output_json results/part1/transformer_large_test.json

echo "=== All Done ==="

