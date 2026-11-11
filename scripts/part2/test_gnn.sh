#!/bin/bash
set -e

if [ "$1" == "--ta" ]; then
    uv run python -m src.part2_evaluate --ta
else
    uv run python -m src.part2_evaluate
fi

