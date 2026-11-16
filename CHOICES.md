# Design & Implementation Choices (CHOICES.md)

This document provides all technical descriptions, design decisions, architectural choices, and implementation details for ANLP Assignment 3 (Monsoon 2026), in accordance with General Instruction #11 (no comments or docstrings permitted in source code) and Clarification #3.

---

## 1. Global Architectural & Framework Choices

### 1.1 Pure PyTorch Implementations from Scratch
- As mandated by General Instruction #1, no third-party abstraction packages (such as `mamba_ssm`, `flash-linear-attention`, or PyG layer abstractions like `GCNConv`, `SAGEConv`, `GATConv`) are utilized for core model definitions.
- All recurrent state transitions, selective scans, discretization formulas, associative memory updates, graph convolutions, and neighborhood aggregations are implemented purely with native PyTorch operations (`torch.nn`, `torch.matmul`, `torch.scatter_add_`, `torch.scatter_reduce_`, etc.).
- The implementation is designed to run in under 10GB VRAM and execute cross-platform (CPU and CUDA).

### 1.2 Separation of Concerns in Source Tree (`src/`)
In accordance with Clarification #5, the `src/` and `docs/` directories are exempt from the `part1/` and `part2/` subdirectory requirement. The source files are organized cleanly as:
- `src/part1_models/`: Pure implementations of Transformer (`transformer.py`), Mamba (`mamba.py`), DeltaNet (`deltanet.py`), and the independent Hybrid architecture (`hybrid.py`).
- `src/part1_data.py`: Dataset loading, character-level vocabulary building, dynamic padding, noise stripping, and PyTorch `DataLoader` construction for cipher decryption.
- `src/part1_train.py`: Unified training loop for all Part 1 models with learning rate scheduling, gradient clipping, checkpointing, and WandB tracking.
- `src/part1_evaluate.py`: Standalone evaluation computing all 8 mandated metrics (Character Accuracy, Word Accuracy, Precision, Recall, F1, Levenshtein Distance, Parameter Count, FLOPs) matched strictly by sample ID.
- `src/part1_inference.py`: Inference pipeline supporting standard test decryption (`output/part1/test_decrypt.jsonl`) and the heldout evaluation flag `--heldout` (`output/part1/heldout_result_2024101056.jsonl`).
- `src/part2_models/`: Graph Neural Network architectures implementing pure PyTorch message passing including Root-Only MLP (`mlp.py`), GCN (`gcn.py`), GraphSAGE (`sage.py`), GAT (`gat.py`), and Bi-GCN (`bigcn.py`).
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

### 2.3 Training Budget & Hardware
- Stage 1: 5 epochs x 4,000 steps (batch 32, max_len 256), peak LR 5e-4, seed 42.
- Stage 2: warm-start from stage 1, LR 3e-4, seed 43.
- Stage 3: warm-start from stage 2, LR 2e-4, seed 44. Equal-budget comparison of all four models (about 3.5M params each, 60,000 steps).
- Stages 4 and 5 (Mamba only): LR 1e-4 (seed 45) and 5e-5 (seed 46).
- Transformer (submitted): a larger model trained from scratch, 10.7M params (d_model 384, 6 layers, 6 heads, d_ff 1536), 12 epochs x 8,000 steps, batch 64, max_len 128, peak LR 5e-4.
- Models were trained on Kaggle T4 GPUs and ADA. Checkpoints go to Hugging Face and logs to W&B project Anlp_A3.
- `test_training.sh` trains each model from scratch with the config in `config/part1/`. For Mamba, DeltaNet and Hybrid that is stage 1 only, and the submitted checkpoints came from the later warm-started stages. A full run takes many hours.

### 2.4 Evaluation Metrics for Part 1
1. **Character Accuracy**: Normalized character-level agreement: $1 - \frac{\text{Levenshtein}(y_{pred}, y_{true})}{\max(|y_{pred}|, |y_{true}|)}$.
2. **Word Accuracy**: Proportion of words matching ground truth exactly upon whitespace tokenization.
3. **Word-Level Precision, Recall, F1-Score**: Unigram multiset intersection of whitespace-delimited words: $\text{Prec} = \frac{|P \cap T|}{|P|}$, $\text{Rec} = \frac{|P \cap T|}{|T|}$, $F_1 = \frac{2 \cdot \text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$.
4. **Character-Level Precision, Recall, F1-Score**: Multiset intersection of individual character counts across the 27-character alphabet ($' '$, $'a'-'z'$). Both word-level and character-level metrics are computed and reported for full transparency.
5. **Levenshtein Distance**: Exact edit distance computed via dynamic programming / Wagner-Fischer algorithm (or `editdistance` when available).
6. **Parameter Count**: Total number of trainable parameters (`p.requires_grad`): Transformer 10.67M, Mamba 3.88M, DeltaNet 3.46M, Hybrid 3.67M.
7. **FLOPs**: Evaluated using `torch.utils.flop_counter.FlopCounterMode` on a standardized sequence length of 512 tokens. Counts only matrix multiplications and convolutions, so elementwise scan operations in Mamba and DeltaNet are not counted.
8. **Per-Position Accuracy (Positions 0 to 20)**: Fine-grained positional accuracy $P(y_i = \hat{y}_i)$ for $i \in \{0, 1, \dots, 20\}$ to diagnose exactly where each architecture successfully tracks the state-dependent substitution table vs where decoding degrades.
9. **Analysis of Metric Discrepancy (about 0.59 vs 0.18)**:
   - There is an inherent discrepancy between exact per-position token accuracy (~0.18) and normalized Levenshtein character accuracy (~0.59).
   - This occurs because Levenshtein distance permits dynamic alignment through insertions and deletions. Due to high English unigram character concentration ('e', 't', 'a', 'o', 'i', 'n', space), predicting English-distributed character sequences naturally aligns opportunistic subsequences with the gold text, yielding a baseline Levenshtein score of ~0.30–0.36 even when exact per-position substitution accuracy is ~0.18.
   - Rather than attempting to game the metric with artificial English filler, our models are trained strictly on the true causal alignment objective, and this metric gap is explicitly analyzed in the report as an artifact of edit-distance dynamic programming.

### 2.5 Inference CLI Model Dispatch (`--model`)
- `src/part1_inference.py` and `src/part1_evaluate.py` support the `--model` parameter (`transformer`, `mamba`, `deltanet`, `hybrid`), defaulting strictly to `hybrid`.
- Checkpoint resolution maps each model directly to its standard configuration in `config/part1/<model>.yaml`:
  - `transformer`: Scaled Large architecture (10.67M parameters, `nani-nav/anlp-a3-transformer-large`, Val CharAcc 0.5914, Test CharAcc ~0.582, configured in `config/part1/transformer.yaml`).
  - `mamba`: Stage 5 checkpoint (`nani-nav/anlp-a3-mamba-v3`, Val CharAcc 0.5487, Test CharAcc ~0.536, configured in `config/part1/mamba.yaml`), representing the highest-performing pure recurrent model.
  - `hybrid`: Stage 3 checkpoint (`nani-nav/anlp-a3-hybrid-v3`, Val CharAcc 0.508, Test CharAcc ~0.498, configured in `config/part1/hybrid.yaml`, default model for `test_assn.sh`).
  - `deltanet`: Stage 3 checkpoint (`nani-nav/anlp-a3-deltanet-v3`, Val CharAcc 0.497, Test CharAcc ~0.487, configured in `config/part1/deltanet.yaml`).
- Exploratory scaling runs (run on Kaggle, configs not included in the repository):
  - Large Mamba (13.0M params): diverged at LR 5e-4.
  - XL Transformer (25.2M params): achieved Val CharAcc 0.5921 and Test CharAcc 0.5841, showing no meaningful gain over the 10.7M model.
- All checkpoints are hosted publicly on Hugging Face and auto-download directly to isolated local directories (`models/part1/<repo_slug>/`), eliminating cross-model checkpoint collision.

---

## 3. Part 2: Fake News Detection on GossipCop (UPFD)

### 3.1 Dataset Configuration
- Dataset: UPFD GossipCop subset loaded directly through `torch_geometric.datasets.UPFD`.
- Splits: Fixed 20% train / 10% validation / 70% test splits as required.
- Node Structure: Node 0 is the root news article; remaining nodes represent Twitter users engaged in retweeting/propagating the news.
- Undirected Propagation: GCN, GraphSAGE, and GAT apply `torch_geometric.transforms.ToUndirected()` so information can propagate from retweeters back to the root news node.
- Edge Directionality in Bi-GCN: Bi-GCN uses directed edges (top-down and reversed bottom-up). The Moodle question on edge direction was not answered, so this follows the original paper.
- Feature Modalities: `profile` (10-dim user metadata), `spacy` (300-dim historical tweet embeddings), `bert` (768-dim contextual embeddings), and `content` (310-dim hybrid concatenation of 300-dim `spacy` text + 10-dim `profile` metadata).
- Batch-Wise Feature Standardization: All four feature types (`profile`, `spacy`, `bert`, `content`) are standardized with the training set mean and standard deviation. This prevents unscaled user profile metadata (e.g. follower counts in the tens of thousands) from dominating dense semantic embeddings, ensuring that "only the input feature type changes between configurations" stays mathematically sound.

### 3.2 Model Architectures (Pure Native PyTorch Layers)
All message-passing layers are implemented from scratch in pure PyTorch without using PyG layer abstractions (`GCNConv`, `SAGEConv`, `GATConv`):

#### A. Root-Only MLP
- Discards all tree edges and retweeter nodes.
- Extracts feature vector of node 0: $x_0 \in \mathbb{R}^{d}$.
- Passes $x_0$ through a 2-layer MLP with ReLU, Dropout (0.5), and a binary classification head.
- Serves as the strictly text-only baseline.

#### B. Graph Convolutional Network (GCN)
- 2 layers of native `NativeGCNConv` on the propagation graph with self-loops and symmetric degree normalization: $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} X W$.
- Aggregates neighborhood representations across the propagation tree.
- Uses native `native_global_max_pool` across all nodes to form the graph embedding, followed by an MLP classifier.

#### C. GraphSAGE
- 2 layers of native `NativeSAGEConv` using mean neighborhood aggregation: $W_1 \cdot \text{mean}(X_{\mathcal{N}(i)}) + W_2 \cdot x_i$.
- Dropout (0.5) for regularization.
- Backbone architecture used for the Feature Ablation experiment.

#### D. Graph Attention Network (GAT)
- 2 layers of native `NativeGATConv` with multi-head attention (4 heads), self-loops, and learnable bias.
- Uses LeakyReLU attention scoring, numerically stable softmax over neighbor sets, and ELU activation.

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
- Hugging Face repos (public):
  - Part 1 Sequence Models: `nani-nav/anlp-a3-transformer-large`, `nani-nav/anlp-a3-mamba-v3`, `nani-nav/anlp-a3-deltanet-v3`, `nani-nav/anlp-a3-hybrid-v3`
  - Part 2 Graph Neural Networks: `nani-nav/anlp-a3-gnn` (stores best-validation checkpoints for all 5 architectures and feature configurations)
- Automated Pull for Evaluation: `src/part1_inference.py` pulls models from Hugging Face by default (`from_hf=True`). If the checkpoint cannot be fetched and no local checkpoint is found, an explicit `FileNotFoundError` is raised.

### 4.2 Weights & Biases (WandB)
- W&B Dashboard: `https://forge.coreweave.com/wandb/nanikolupoti-iiit-hyderabad/Anlp_A3?nw=nwusernanikolupoti`
- Unified project `Anlp_A3`, run names:
  - Part 1 runs: `transformer_large`, `mamba_v3`, `deltanet_v3`, `hybrid_v3`
  - Part 2 run: `part2_gnn_evaluation` (logging per-epoch training loss, validation F1/accuracy curves, and final 5-seed mean/std metrics)
