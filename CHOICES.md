# Design & Implementation Choices (CHOICES.md)

This document provides all technical descriptions, design decisions, architectural choices, and implementation details for ANLP Assignment 3 (Monsoon 2026), in accordance with General Instruction #11 (no comments or docstrings permitted in source code) and Clarification #3.

---

## 1. Global Architectural & Framework Choices

### 1.1 Pure PyTorch Implementations from Scratch
- As mandated by General Instruction #1, no third-party abstraction packages (such as `mamba_ssm`, `flash-linear-attention`, or similar libraries) are utilized.
- All recurrent state transitions, selective scans, discretization formulas, and associative memory updates are implemented purely with native PyTorch operations (`torch.nn`, `torch.matmul`, `torch.exp`, etc.).
- The implementation is guaranteed to execute cross-platform (CPU and CUDA) and strictly within the 10GB VRAM constraint.

### 1.2 Separation of Concerns in Source Tree (`src/`)
In accordance with Clarification #5, the `src/` and `docs/` directories are exempt from the `part1/` and `part2/` subdirectory requirement. The source files are organized cleanly as:
- `src/part1_models/`: Pure implementations of Transformer (`transformer.py`), Mamba (`mamba.py`), DeltaNet (`deltanet.py`), and the independent Hybrid architecture (`hybrid.py`).
- `src/part1_data.py`: Dataset loading, character-level vocabulary building, dynamic padding, noise stripping, and PyTorch `DataLoader` construction for cipher decryption.
- `src/part1_train.py`: Unified training loop for all Part 1 models with learning rate scheduling, gradient clipping, checkpointing, and WandB tracking.
- `src/part1_evaluate.py`: Standalone evaluation computing all 8 mandated metrics (Character Accuracy, Word Accuracy, Precision, Recall, F1, Levenshtein Distance, Parameter Count, FLOPs) matched strictly by sample ID.
- `src/part1_inference.py`: Inference pipeline supporting standard test decryption (`output/part1/test_decrypt.jsonl`) and the heldout evaluation flag `--heldout` (`output/part1/heldout_result_2024101056.jsonl`).
- `src/part2_models/`: Graph Neural Network architectures including Root-Only MLP (`mlp.py`), GCN (`gcn.py`), GraphSAGE (`sage.py`), GAT (`gat.py`), and Bi-GCN (`bigcn.py`).
- `src/part2_data.py`: UPFD GossipCop dataset loader using PyTorch Geometric with 20%-10%-70% fixed train/val/test splits, undirected edge transform, and batch-wise feature standardization.
- `src/part2_evaluate.py`: Multi-seed training and evaluation pipeline calculating mean and standard deviation across 5 random seeds, producing `output/part2/results.json` (and `output/part2/results_ta.json` with `--ta`), as well as `results/part2/results_detailed.json`.
- `src/utils.py`: Shared utilities for reproducible seed setting, YAML configuration parsing, Hugging Face checkpoint auto-pulling/uploading, and device assignment.

---

## 2. Part 1: Cipher Decryption & Sequence Modeling

### 2.1 Empirical Dataset Properties
Empirical analysis of `data/part1/` reveals:
1. **Target Alphabet (Plaintext)**: Exactly 27 unique characters — space (`' '`) and 26 lowercase English letters (`'a'-'z'`).
2. **Noise Symbols**: Exactly 18 punctuation and symbol characters (`!'+,-./:;<=>?[]_{}`) are noise tokens interspersed throughout the ciphertext.
3. **Exact Length Alignment**: When these 18 noise symbols are stripped, the stripped ciphertext matches the plaintext length exactly in 100% of samples (2,471 out of 2,471 validation and test samples, and all 569,670 out of 569,670 training samples, totaling 572,141 samples with exactly 0 length mismatches). Thus, character $i$ of the stripped ciphertext encrypts character $i$ of the plaintext.
4. **Deterministic Cipher Mechanics**: The cipher is deterministic (identical plaintexts consistently map to identical ciphertexts).
5. **Periodic Structure**: Position 0 uses a fixed one-to-one substitution table. Subsequent positions follow a dynamic substitution table that evolves based on prior context with a period-6 cycle.
6. **Sequence Modeling Formulation**: Decryption is formulated as causal, aligned sequence labeling. The input is the stripped ciphertext sequence, and the output at position $i$ is the plaintext character $i$. Inference runs in a single parallel forward pass (`model(x).argmax(-1)`), completely avoiding slow $O(L^2)$ token-by-token autoregression and output truncation.

### 2.2 Model Implementations

#### A. Transformer Baseline
- Causal decoder model with learned character embeddings and dynamic sinusoidal positional encodings computed on the fly for arbitrary sequence lengths.
- Multi-head causal self-attention implemented with PyTorch native `F.scaled_dot_product_attention(is_causal=True)` for fused memory-efficient execution with $O(T)$ memory complexity.
- Pre-RMSNorm residual connections followed by a Feed-Forward Network with GELU activations.

#### B. Mamba (Selective State Space Model — S6)
- Implements the continuous-to-discrete dynamical system formulation from Gu & Dao (2023):
  $$\bar{A}_t = \exp(\Delta_t A), \quad \bar{B}_t = \Delta_t B_t$$
  $$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t + D x_t$$
- The parameters $\Delta_t, B_t, C_t$ are dynamically projected from the input $x_t$ at each timestep via learned linear projections.
- $\Delta_t$ passes through a Softplus activation with a learned bias initialization.
- Includes a 1D causal depthwise convolution (kernel size 4) along the sequence dimension to capture local n-gram context before state space recurrence.
- The output of the SSM branch is modulated via element-wise multiplication with a parallel SiLU gating branch.
- Pre-RMSNorm residual connection followed by Pre-RMSNorm and GELU FFN with residual connection.
- No attention layers are interleaved in Mamba, strictly adhering to the assignment instructions.

#### C. DeltaNet (Linear Attention with the Delta Rule)
- Implements the error-driven associative memory formulation from Songlin Yang et al. (NeurIPS 2024 / arXiv:2406.06484):
  $$\hat{v}_t = S_{t-1} k_t$$
  $$e_t = v_t - \hat{v}_t$$
  $$S_t = S_{t-1} + \beta_t e_t k_t^T$$
  $$o_t = S_t q_t$$
- Memory matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ acts as an associative store mapping encrypted character representations to decrypted features.
- Both query vectors $q_t$ and key vectors $k_t$ are $L_2$-normalized to ensure bounded, stable recurrent memory updates.
- Learning rate $\beta_t \in (0, 1)$ is dynamically generated via a sigmoid projection of the input $x_t$.
- Output $o_t$ is gated with a learned data-dependent SiLU gate.
- Pre-RMSNorm residual connection followed by Pre-RMSNorm and GELU FFN with residual connection.

#### D. Independent Mamba-DeltaNet Hybrid
- Per the assignment mandate, `hybrid.py` is written completely independently without importing classes from `mamba.py` or `deltanet.py`.
- Employs an interleaved architecture alternating Mamba and DeltaNet blocks: `Mamba -> DeltaNet -> Mamba -> DeltaNet`.
- Every block is pre-RMSNorm $\to$ mixer $\to$ residual, followed by pre-RMSNorm $\to$ GELU FFN $\to$ residual.
- Mamba layers track the period-6 state and recent cipher history via causal convolution and input-dependent state decay.
- DeltaNet layers perform the associative lookup table mapping, overwriting obsolete mappings via the delta rule.
- The detailed theoretical hypothesis is documented in `docs/DESIGN.md`.

### 2.3 Training Budget & Hardware Allocation
- All models were trained under the identical compute budget of 3 epochs × 1,000 optimizer steps (effective batch size 16 via batch size 8 with gradient accumulation steps 2, max sequence length 1024), totaling 3,000 optimizer steps (48,000 sequences).
- Hardware allocation: Transformer and Mamba were trained on the ADA cluster (NVIDIA RTX/Ada GPU), while DeltaNet and Hybrid were trained on Kaggle (NVIDIA Tesla T4 GPU). Checkpoints and WandB metrics sync automatically across environments to Hugging Face (`_v2_best.pt`) and the unified WandB project `Anlp_A3`.

### 2.4 Evaluation Metrics for Part 1
1. **Character Accuracy**: Normalized character-level agreement: $1 - \frac{\text{Levenshtein}(y_{pred}, y_{true})}{\max(|y_{pred}|, |y_{true}|)}$.
2. **Word Accuracy**: Proportion of words matching ground truth upon whitespace tokenization.
3. **Precision, Recall, F1-Score**: Token-level unigram precision, recall, and harmonic mean F1.
4. **Levenshtein Distance**: Exact edit distance computed via dynamic programming / Wagner-Fischer algorithm (or `editdistance` when available).
5. **Parameter Count**: Total number of trainable parameters (`p.requires_grad`), strictly controlled between 3.1M and 3.9M across all architectures.
6. **FLOPs**: Evaluated using `torch.utils.flop_counter.FlopCounterMode` (which measures forward-pass matrix multiplication FLOPs) on a standardized sequence length of 512 tokens.

---

## 3. Part 2: Fake News Detection on GossipCop (UPFD)

### 3.1 Dataset Configuration
- Dataset: UPFD GossipCop subset loaded directly through `torch_geometric.datasets.UPFD`.
- Splits: Fixed 20% train / 10% validation / 70% test splits as required.
- Node Structure: Node 0 is the root news article; remaining nodes represent Twitter users engaged in retweeting/propagating the news.
- Undirected Propagation: GCN, GraphSAGE, and GAT apply `torch_geometric.transforms.ToUndirected()` so information can propagate from retweeters back to the root news node.
- Batch-Wise Feature Standardization: All four feature types (`profile`, `spacy`, `bert`, `content`) are standardized on the GPU using training set mean and standard deviation. This prevents unscaled user profile metadata (e.g. follower counts in the tens of thousands) from dominating dense semantic embeddings, ensuring that "only the input feature type changes between configurations" stays mathematically sound.

### 3.2 Model Architectures

#### A. Root-Only MLP
- Discards all tree edges and retweeter nodes.
- Extracts feature vector of node 0: $x_0 \in \mathbb{R}^{d}$.
- Passes $x_0$ through a 2-layer MLP with ReLU, Dropout (0.5), and a binary classification head.
- Serves as the strictly text-only baseline.

#### B. Graph Convolutional Network (GCN)
- 2 layers of `GCNConv` on the propagation graph.
- Aggregates neighborhood representations across the propagation tree.
- Uses `global_max_pool` across all nodes to form the graph embedding, followed by an MLP classifier.

#### C. GraphSAGE
- 2 layers of `SAGEConv` using mean neighborhood aggregation.
- Dropout (0.5) for regularization.
- Backbone architecture used for the Feature Ablation experiment.

#### D. Graph Attention Network (GAT)
- 2 layers of `GATConv` with multi-head attention (4 heads) to dynamically weight influential user interactions.
- Uses ELU activation and dropout on attention weights.

#### E. Bi-GCN (Bian et al., AAAI 2020)
- Architecture:
  1. Top-Down GCN ($\text{GCN}_{TD}$): Models information flow from root down to users along directed edges.
  2. Bottom-Up GCN ($\text{GCN}_{BU}$): Models crowd feedback dispersion by reversing edge directions toward root.
  3. Feature Concatenation: Combines pooled top-down representations, pooled bottom-up representations, and the original root feature vector:
     $$h_{final} = [\text{Pool}(H_{TD}) \,\|\, \text{Pool}(H_{BU}) \,\|\, x_{root}]$$
  4. Classification via fully connected layers.

### 3.3 Training & Evaluation Protocol
- 5 independent random seeds: `[42, 123, 456, 789, 1024]`.
- Hyperparameters read directly from `config/part2/*.yaml`.
- For each run, model weights from the epoch with the highest validation F1-score are checkpointed and restored for test evaluation.
- Results: Mean values are written to `output/part2/results.json` adhering to the exact schema of `results_template.json`. The `--ta` flag outputs to `output/part2/results_ta.json`. Full per-seed metrics and standard deviations are saved in `results/part2/results_detailed.json`.

---

## 4. Hugging Face and WandB Integration

### 4.1 Hugging Face Model Hub
- Repositories configured for user `nani-nav` (with distinct `v2` repositories to preserve legacy runs):
  - `nani-nav/anlp-a3-transformer-v2`
  - `nani-nav/anlp-a3-mamba-v2`
  - `nani-nav/anlp-a3-deltanet-v2`
  - `nani-nav/anlp-a3-hybrid-v2`
- Automated Pull for Evaluation: `src/part1_inference.py` pulls models from Hugging Face by default (`from_hf=True`). If the checkpoint cannot be fetched and no local checkpoint is found, an explicit `FileNotFoundError` is raised.

### 4.2 Weights & Biases (WandB)
- Project: `Anlp_A3` under entity `nanikolupoti-iiit-hyderabad`.
- Part 1 Tracking: Distinct run names (`transformer_v2`, `mamba_v2`, `deltanet_v2`, `hybrid_v2`) logging training loss, validation loss, validation character accuracy (overall and by length buckets `0-1k`, `1-2k`, `2k+`), learning rate schedule, parameter count, FLOPs, and best validation accuracy.
- Part 2 Tracking: Logs test metrics across seeds for Part A and Part B feature ablations, and syncs artifacts to WandB.
