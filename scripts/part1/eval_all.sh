#!/bin/bash
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=anlp3_eval_all
#SBATCH --output=logs/part1/eval_all_%j.log

export PATH="$HOME/.local/bin:$PATH"
set -e

mkdir -p logs/part1
mkdir -p output/part1
mkdir -p results/part1

echo "=========================================================="
echo "          PART 1 EVALUATION ACROSS ALL 4 MODELS           "
echo "=========================================================="

for MODEL in transformer mamba deltanet hybrid; do
    echo "=== Running Inference: $MODEL ==="
    uv run python -m src.part1_inference --config "config/part1/${MODEL}.yaml" --output_file "output/part1/${MODEL}_decrypt.jsonl"
    echo "=== Metrics for $MODEL ==="
    uv run python -m src.part1_evaluate --pred "output/part1/${MODEL}_decrypt.jsonl" --target data/part1/test/cleaned.jsonl --model_name "${MODEL}"
    echo "----------------------------------------------------------"
done

echo "=== Generating Primary test_decrypt.jsonl (via test_assn.sh) ==="
bash scripts/part1/test_assn.sh

echo "=== Generating Heldout heldout_result_2024101056.jsonl ==="
bash scripts/part1/test_assn.sh --heldout

echo "=========================================================="
echo "                 ALL EVALUATIONS COMPLETE                 "
echo "=========================================================="
