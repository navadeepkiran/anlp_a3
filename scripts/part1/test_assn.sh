#!/bin/bash
set -e

MODEL="hybrid"
HELDOUT=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --heldout)
      HELDOUT="--heldout"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

if [ -n "$HELDOUT" ]; then
    uv run python -m src.part1_inference --model "$MODEL" --heldout --roll_number 2024101056
else
    uv run python -m src.part1_inference --model "$MODEL"
fi
