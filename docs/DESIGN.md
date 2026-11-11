# Hybrid Design: Mamba and DeltaNet, interleaved

## What I noticed about the cipher
I looked at the new validation and test data before designing the hybrid.

1. Eighteen symbols (`! ' + , - . / : ; < = > ? [ ] _ { }`) are just noise. If I delete them, the ciphertext is exactly as long as the plaintext in every sample. So the i-th cipher character encrypts the i-th plaintext character.
2. The cipher is deterministic. The same plaintext always gives the same ciphertext.
3. The first character uses a fixed one-to-one substitution. After that, the substitution keeps changing depending on the plaintext that came before. There also seems to be a cycle of length 6: at positions 2, 8, 14 and so on, the output depends mostly on just the previous character.

So to decrypt each character, a model has to do two things:
- keep track of the current state (where it is in the 6-step cycle and what came recently), and
- use that state to look up the current substitution table and map the cipher character back to a plain character.

## How every model is set up
All four models are trained the same way. The input is the cipher with the noise symbols removed, and the model predicts the plaintext character at each position. It reads left to right only (causal), does one forward pass, and the output is always exactly as long as the input.

## The hybrid
Four blocks: Mamba, DeltaNet, Mamba, DeltaNet. Each block is RMSNorm, then the mixer (Mamba or DeltaNet) with a residual connection, then RMSNorm and a GELU feed-forward layer with another residual. Width and depth are the same as the pure models, so the parameter counts are close.

## Why I think this order should work
- Mamba is good at keeping track of state. Its short convolution sees the last few characters, and its input-dependent step size lets it hold on to information or reset it. That fits a cipher whose state cycles and resets.
- DeltaNet is good at being a lookup table that changes. The delta rule replaces an old key-value pair instead of piling a new one on top, which suits a substitution table that keeps changing.
- Mamba goes first because DeltaNet works better when its keys already contain local context. The DeltaNet paper adds a short convolution for this reason; here the Mamba layer does that job.
- DeltaNet goes last because the final step before the output is the table lookup.
- Repeating the pair lets the second Mamba layer correct its state using the first DeltaNet layer's guesses.

## What I expect to see
With the same parameter budget and the same amount of training:
1. The hybrid should beat both pure Mamba and pure DeltaNet on character accuracy.
2. Pure Mamba should lose less accuracy than pure DeltaNet at the harder positions (those not of the form 6k+2). At the easier positions, pure DeltaNet should be close to the hybrid.
3. Mamba, DeltaNet and the hybrid should hold their accuracy past the training length better than the Transformer.

I will check these using accuracy per position mod 6 and accuracy against position in the sequence.