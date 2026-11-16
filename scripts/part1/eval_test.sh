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

MODEL="hybrid"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    *)
      if [[ "$1" == "transformer" || "$1" == "mamba" || "$1" == "deltanet" || "$1" == "hybrid" ]]; then
        MODEL="$1"
      fi
      shift
      ;;
  esac
done

echo "=== Running Inference ($MODEL) ==="
uv run python -m src.part1_inference --model "$MODEL" --output_file "output/part1/${MODEL}_decrypt.jsonl"

echo "=== Evaluating Metrics ($MODEL) ==="
uv run python -m src.part1_evaluate --pred "output/part1/${MODEL}_decrypt.jsonl" --target data/part1/test/cleaned.jsonl --model "$MODEL"

echo "=== Evaluation Complete for $MODEL ==="
