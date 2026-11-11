#!/bin/bash
set -e

if [ "$1" == "--heldout" ]; then
    uv run python -m src.part1_inference --heldout --roll_number 2024101056
else
    uv run python -m src.part1_inference
fi

