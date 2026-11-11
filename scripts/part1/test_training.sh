#!/bin/bash
set -e

uv run python -m src.part1_train --config config/part1/transformer.yaml
uv run python -m src.part1_train --config config/part1/mamba.yaml
uv run python -m src.part1_train --config config/part1/deltanet.yaml
uv run python -m src.part1_train --config config/part1/hybrid.yaml

