# Parameter Golf: Proven Techniques Reference

Concise reference of all validated leaderboard techniques for minimizing `val_bpb` under the 16MB artifact constraint. Organized by category with impact estimates and implementation notes.

**Last updated:** 2026-03-26
**Current SOTA:** 1.1194 val_bpb
**Constraint:** Final artifact (model weights + code) must be <= 16MB

---

## Leaderboard Progression

Chronological path from baseline to SOTA. Each row shows the cumulative score, incremental improvement, and the key change introduced.

| Score | Delta | Key change |
|------:|------:|------------|
| 1.2244 | — | Naive baseline (9L 512d, int8+zlib) |
| 1.2058 | -0.019 | Sequence length 1024 → 2048 |
| 1.2014 | -0.004 | Sequence length 4096 + Muon tuning (momentum 0.99, LR 0.02) |
| 1.1925 | -0.009 | Sliding window eval (stride=64, eval-only change) |
| 1.1748 | -0.018 | 10 layers + FP16 embedding + Muon weight decay + ortho init |
| 1.1630 | -0.012 | 3x MLP + mixed int6/int8 quant + sliding eval |
| 1.1556 | -0.007 | SmearGate + BigramHash + int6 QAT + zstd-22 |
| 1.1458 | -0.010 | 9L int6 + 3x MLP + SmearGate + BigramHash + SWA + Muon WD 0.04 |
| 1.1428 | -0.003 | 10L + mixed int5/int6 + BigramHash(10240) + SWA |
| 1.1307 | -0.012 | 11L + Efficient Partial XSA (last 3 layers) + SWA |
| 1.1271 | -0.004 | XSA on last 4 layers + EMA(0.997) replacing SWA |
| 1.1248 | -0.002 | Partial RoPE (16/64 dims) + LN Scale (1/sqrt(layer+1)) |
| 1.1233 | -0.002 | GPTQ-lite clip search + warmdown 3500 + late QAT threshold 0.15 |
| 1.1194 | -0.004 | LeakyReLU(0.5)^2 + legal score-first TTT + Parallel Muon |

---

## Architecture Techniques

### Layer Count: 9 → 11 layers
- **Impact:** ~-0.012 BPB (going from 9L to 11L)
- **Notes:** More layers requires int6 quantization to stay under 16MB. 10L is a safe middle ground; 11L needs aggressive quantization (mixed int5/int6) and careful size budgeting. Demonstrated at 1.1307 submission.

### MLP Expansion: 2x → 3x
- **Impact:** ~-0.005 to -0.010 BPB
- **Notes:** Hidden dim 1536 (3x of 512d). Major capacity boost. Combined with int6 quant to fit. Demonstrated at 1.1630 submission. Ensure size budget allows it.

### U-Net Skip Connections
- **Impact:** ~-0.002 to -0.005 BPB (architecture-dependent)
- **Notes:** Encoder/decoder structure with learned skip weights connecting symmetric layers. Adds minimal parameters. Most effective with deeper networks (10L+). Used in the ternary submission (1.1570).

### Exclusive Self Attention (XSA) — Last 3-4 Layers
- **Impact:** ~-0.012 BPB (3 layers), ~-0.004 BPB additional (4th layer)
- **Notes:** Replaces standard attention in final layers. Each head attends to a distinct subset of positions, reducing redundancy. Apply to last 3-4 layers only; applying to all layers hurts. "Efficient Partial XSA" variant reduces compute. Demonstrated at 1.1307 (3 layers) and 1.1271 (4 layers).

### Partial RoPE (16/64 dims)
- **Impact:** ~-0.002 BPB
- **Notes:** Apply rotary position embeddings to only 16 out of 64 head dimensions instead of all. Remaining dims use NoPE (no positional encoding). Reduces over-reliance on position. Demonstrated at 1.1248.

### LN Scale: 1/sqrt(layer_idx + 1)
- **Impact:** ~-0.001 to -0.002 BPB (combined with Partial RoPE)
- **Notes:** Scale LayerNorm output by `1/sqrt(layer_idx + 1)` where `layer_idx` is 0-indexed. Dampens signal magnitude in deeper layers, stabilizing training. Simple one-line change. Demonstrated at 1.1248.

### SmearGate
- **Impact:** ~-0.003 to -0.005 BPB
- **Notes:** Learned gate that blends the current token representation with the previous token's representation before attention. Adds a small linear layer per block to produce a gate value. Helps capture local context cheaply. Demonstrated at 1.1556.

### BigramHash
- **Impact:** ~-0.003 to -0.005 BPB
- **Notes:** Hash consecutive token pairs into learned embedding buckets (2048-10240 buckets). Added to the input embeddings. Captures bigram statistics without a full bigram table. More buckets generally better up to ~10240. Demonstrated at 1.1556 (2048 buckets) and improved at 1.1428 (10240 buckets).

### LeakyReLU(0.5)^2 Activation
- **Impact:** ~-0.002 BPB
- **Notes:** Replace the MLP activation with `LeakyReLU(negative_slope=0.5)` followed by squaring. One-line change. Outperforms SwiGLU and standard ReLU^2 in this constrained setting. Demonstrated at 1.1194.

### Value Embedding
- **Impact:** ~-0.001 to -0.002 BPB (exploratory)
- **Notes:** Shared value embedding with dim=128, applied to layers 9-10 only. Adds a small learned embedding that modulates value projections in late layers. Experimental; validate impact carefully.

---

## Quantization Techniques

### Int6 STE QAT (Straight-Through Estimator)
- **Impact:** Eliminates ~0.005-0.010 BPB quantization gap
- **Notes:** Quantize weights to 6-bit during training using straight-through estimator for gradients. Range [-31, 31]. Model learns to be robust to quantization. Essential for fitting 10L+ models under 16MB. Demonstrated at 1.1556.

### Mixed Int5/Int6 Quantization
- **Impact:** Saves ~0.5-1MB vs uniform int6, enabling more layers
- **Notes:** Use int5 for MLP weights (more tolerant of lower precision), int6 for attention weights (more sensitive). Enables 11L models to fit. Demonstrated at 1.1428.

### GPTQ-lite Clip Search
- **Impact:** ~-0.001 to -0.002 BPB
- **Notes:** Per-row percentile-based clip search for quantization ranges. Zero training cost — applied post-training. Searches for optimal clipping percentile per weight row to minimize quantization error. Demonstrated at 1.1233.

### FP16 Tied Embedding
- **Impact:** ~-0.002 to -0.003 BPB
- **Notes:** Keep the tied embedding/unembedding matrix in FP16 instead of quantizing it. Embeddings are highly sensitive to quantization noise. Uses more size budget but improves quality significantly. Demonstrated at 1.1748.

### zstd-22 Compression
- **Impact:** Saves ~1.5MB vs zlib-9
- **Notes:** Use zstd at compression level 22 instead of zlib-9 for the final artifact. Better compression ratio frees space for more parameters or higher-precision weights. Demonstrated at 1.1556.

### Late QAT (Enable STE When LR Scale < 0.15)
- **Impact:** ~-0.001 BPB (vs early QAT)
- **Notes:** Only enable straight-through quantization simulation late in training, when learning rate scale drops below 0.15. Allows the model to learn freely early, then adapt to quantization constraints during fine-tuning. Demonstrated at 1.1233.

---

## Training Techniques

### Muon Weight Decay 0.04
- **Impact:** ~-0.003 to -0.005 BPB
- **Notes:** Decoupled weight decay at 0.04 with Muon optimizer. Produces tighter weight distributions that quantize better. Higher than typical values but validated. Demonstrated at 1.1458.

### Warmdown 3000-3500 Iters
- **Impact:** ~-0.001 to -0.002 BPB
- **Notes:** Linear LR decay to zero over the final 3000-3500 iterations. Longer warmdown (3500) slightly better than shorter (3000). Produces tighter final weight distributions. Demonstrated at 1.1233.

### Learning Rate Tuning
- **Impact:** ~-0.002 to -0.005 BPB (from default)
- **Notes:** `MATRIX_LR=0.02-0.025` for Muon-optimized parameters, `TIED_EMBED_LR=0.03-0.035` for the tied embedding. These are higher than typical defaults. Sensitive to other hyperparameters; tune carefully.

### Muon Momentum 0.99 with Warmup
- **Impact:** ~-0.002 to -0.004 BPB
- **Notes:** Set Muon momentum to 0.99 (higher than default). Warm up from 0.92 to 0.99 over the first 1500 steps. Stabilizes early training while allowing aggressive optimization later. Demonstrated at 1.2014.

### Gradient Clipping (GRAD_CLIP_NORM=0.3)
- **Impact:** Training stability (prevents divergence)
- **Notes:** Clip gradient norms at 0.3. Prevents loss spikes with aggressive LR and momentum settings. Essential when using Muon with high momentum.

### Sequence Length 2048
- **Impact:** ~-0.019 BPB
- **Notes:** Increase from 1024 to 2048. Significantly better learning signal from longer context. ~1.6x slower per step but well worth it. Going to 4096 gives diminishing returns (~-0.004 more) and is much slower. Demonstrated at 1.2058.

### Batch Size Tuning (786432 Tokens)
- **Impact:** ~-0.001 to -0.003 BPB
- **Notes:** 786432 tokens per batch with seq_len=2048 (384 sequences). Larger batches stabilize Muon training. Tune in conjunction with LR.

### Orthogonal Initialization + muP-Scaled Outputs
- **Impact:** ~-0.002 to -0.003 BPB
- **Notes:** Initialize weight matrices with orthogonal init. Scale output projections following muP (maximal update parameterization) principles. Improves training dynamics, especially with deeper networks. Demonstrated at 1.1748.

---

## Post-Training and Evaluation

### EMA (Exponential Moving Average)
- **Impact:** ~-0.002 to -0.004 BPB
- **Notes:** Decay=0.997, update every step. Preferred over SWA for simplicity and slightly better results. Use the EMA weights for final evaluation and export. Demonstrated at 1.1271.

### SWA (Stochastic Weight Averaging)
- **Impact:** ~-0.002 to -0.004 BPB
- **Notes:** Average checkpoints every 50 steps over the last 40-50% of warmdown. Superseded by EMA (which is simpler and equally effective). Still valid if EMA is not implemented. Demonstrated at 1.1458.

### Sliding Window Evaluation (stride=64)
- **Impact:** ~-0.025 to -0.035 BPB (pure eval improvement)
- **Notes:** At evaluation time, use overlapping windows with stride=64 instead of non-overlapping chunks. Each token is predicted using maximum available context. Eval-only change, no training cost. Large free improvement. Demonstrated at 1.1925.

### Temperature Scaling (T=0.90)
- **Impact:** ~-0.001 to -0.002 BPB (with extreme quantization)
- **Notes:** Scale logits by T=0.90 at eval time. Helps when aggressive quantization slightly flattens the output distribution. Only beneficial with int5 or very aggressive quantization. Test on validation set.

---

## Frontier Ideas

### Legal TTT (Test-Time Training)
- **Impact:** ~-0.002 to -0.005 BPB
- **Notes:** Score-first, backward-looking test-time training. At eval time, adapt model parameters using SGD on previously seen tokens (backward-looking only, so it's "legal" — no data leakage). Uses the eval data itself for adaptation. Adds ~410s to eval time but stays within competition rules. Demonstrated at 1.1194.

### Parallel Muon (Batched Newton-Schulz)
- **Impact:** Training speed improvement (same BPB)
- **Notes:** Batch the Newton-Schulz iterations in Muon optimizer across parameter groups. Reduces wall-clock training time without affecting convergence. Useful for running more experiments in fixed time budget. Demonstrated at 1.1194.

### Ternary Quantization (BitNet b1.58)
- **Impact:** 1.1570 BPB with 73M+ params
- **Notes:** Quantize all weights to {-1, 0, +1}. Enables dramatically more parameters (73M+) within 16MB. Currently behind int6 approaches but has theoretical upside with more optimization. Interesting research direction.

### LoRA TTT
- **Impact:** ~-0.001 to -0.003 BPB (older estimate)
- **Notes:** Low-rank adaptation at test time. Superseded by Legal TTT which is simpler and more effective. Mentioned for completeness.

---

## Important Notes

### What Doesn't Work
These techniques have been tested and shown no improvement or negative results in this setting:
- **Depth recurrence** (looping layers): No gain; increases effective depth but hurts parameter efficiency. Extensively tested in submission #363.
- **SwiGLU activation**: Worse than LeakyReLU(0.5)^2 and standard ReLU^2 in this parameter regime.
- **Factored embeddings**: Compressing embeddings via factorization loses too much quality.
- **Value residual on loops**: No benefit when using depth recurrence.
- **Sawtooth LR schedule**: Unstable with Muon, no improvement over linear warmdown.
- **Cyclic Muon momentum**: No benefit over fixed 0.99 with warmup.

### Current SOTA Stack (1.1194 val_bpb)
Complete list of techniques in the best submission:
- 11 layers, 512d, 3x MLP (hidden=1536)
- LeakyReLU(0.5)^2 activation
- XSA on last 4 layers
- Partial RoPE (16/64 dims)
- LN Scale (1/sqrt(layer_idx+1))
- SmearGate
- BigramHash (10240 buckets)
- Mixed int5/int6 quantization (int5 MLP, int6 attention)
- Int6 STE QAT with late enable (threshold 0.15)
- GPTQ-lite clip search post-training
- FP16 tied embedding
- zstd-22 compression
- Muon optimizer (LR 0.02-0.025, momentum 0.99, WD 0.04)
- Warmdown 3500 iters
- Orthogonal init + muP-scaled outputs
- Sequence length 2048, batch size 786432 tokens
- EMA (decay=0.997)
- Sliding window eval (stride=64)
- Legal TTT (score-first, backward-looking SGD)
- Parallel Muon (batched Newton-Schulz)
