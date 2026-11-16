# Commands to Run and Replicate Results

## Environment Setup
```bash
uv sync
```

## Part 1: Cipher Decryption

### Training
```bash
bash scripts/part1/test_training.sh
```

### Inference on Test Set (Default: Hybrid)
```bash
bash scripts/part1/test_assn.sh
```

### Inference on Test Set by Model
```bash
bash scripts/part1/test_assn.sh --model transformer
bash scripts/part1/test_assn.sh --model mamba
bash scripts/part1/test_assn.sh --model deltanet
bash scripts/part1/test_assn.sh --model hybrid
```

### Inference on Held-out Set
```bash
bash scripts/part1/test_assn.sh --heldout
bash scripts/part1/test_assn.sh --heldout --model transformer
bash scripts/part1/test_assn.sh --heldout --model mamba
bash scripts/part1/test_assn.sh --heldout --model deltanet
bash scripts/part1/test_assn.sh --heldout --model hybrid
```

### Evaluation of All Models
```bash
bash scripts/part1/eval_all.sh
```

### Push Checkpoints to Hugging Face
```bash
bash scripts/part1/push_to_hf.sh
```

## Part 2: Graph Neural Networks (Fake News Detection)

### Standard Evaluation
```bash
bash scripts/part2/test_gnn.sh
```

### TA Evaluation Mode
```bash
bash scripts/part2/test_gnn.sh --ta
```
