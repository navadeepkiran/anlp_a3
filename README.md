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

### Inference on Test Set
```bash
bash scripts/part1/test_assn.sh
```

### Inference on Held-out Set
```bash
bash scripts/part1/test_assn.sh --heldout
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

