# Hybrid Design: Mamba → DeltaNet Interleaved

## Observations on the cipher (data_new, validation + test)
1. The 18 symbols `! ' + , - . / : ; < = > ? [ ] _ { }` are noise. After removing them, the ciphertext has exactly the same length as the plaintext in every sample, so character i of the cipher encrypts character i of the plaintext.
2. The cipher is deterministic: the same plaintext always gives the same ciphertext.
3. Position 0 uses a fixed one-to-one substitution. After that the substitution table changes at every step, depending on the plaintext seen so far. The dependence has a period-6 structure: at positions 2, 8, 14, … the output depends almost only on the previous character.

So decryption needs two things at every step:
- (a) **Track state**: where we are in the 6-step cycle, and what the recent history was.
- (b) **Look up** the current, constantly changing table: (state, cipher char) → plain char.

## Formulation (same for all four models)
Causal, aligned sequence labelling. The input is the noise-stripped cipher; the output at position i is plaintext character i. There is one forward pass and no autoregressive generation, and the output length always equals the input length.

## Hybrid architecture
Four blocks in the order **Mamba → DeltaNet → Mamba → DeltaNet**. Every block is pre-RMSNorm → mixer → residual, followed by pre-RMSNorm → GELU FFN → residual. Width and depth match the pure models, so parameter counts are comparable.

## Hypothesis
- **Mamba is suited to (a).** Its causal convolution captures the recent cipher context. Its input-dependent decay (Δ) can keep or reset state, which is what a periodic, resetting cipher state needs.
- **DeltaNet is suited to (b).** The delta rule overwrites an old key→value association instead of adding to it. That matches a substitution table that keeps changing; additive memory would blur old and new mappings together.
- **Why Mamba comes first:** DeltaNet's keys work better when they already carry local context. The DeltaNet paper adds a short convolution for exactly this reason, and here the Mamba layer provides it.
- **Why a DeltaNet layer comes last:** the last thing done before the output is the table lookup.
- **Why interleave:** the second pair can refine the state using the first lookup's guesses.

## Testable predictions (same parameter budget and same number of training tokens)
1. The hybrid beats both pure Mamba and pure DeltaNet on character accuracy.
2. Pure Mamba loses less accuracy than pure DeltaNet at the hard phases (positions not of the form 6k+2). Pure DeltaNet is closer to the hybrid at the easy phase.
3. The linear models (Mamba, DeltaNet, hybrid) keep their accuracy at positions beyond the training length better than the Transformer does.

These will be checked with accuracy per phase (i mod 6) and accuracy versus position.
