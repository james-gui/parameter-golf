# Autoresearch Loop for Parameter Golf — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `program.md` and `techniques.md` files that enable an AI coding agent to autonomously run parameter-golf experiments on a 1xH100 RunPod, iterating on `train_gpt.py` to minimize `val_bpb` under the 16MB artifact constraint.

**Architecture:** Two standalone Markdown files at the repo root. `techniques.md` is a static research reference; `program.md` is the agent protocol defining setup, experiment loop, logging, and research refresh. No code changes to existing files.

**Tech Stack:** Markdown, git, `torchrun`, `gh` CLI for GitHub API access.

---

### Task 1: Write `techniques.md`

**Files:**
- Create: `techniques.md`

- [ ] **Step 1: Write the leaderboard progression section**

```markdown
# Parameter Golf — Technique Reference

A reference for AI agents running autonomous experiments on the parameter-golf challenge.
Last updated: 2026-03-26. Current SOTA: 1.1194 val_bpb.

## Leaderboard Progression

The baseline started at 1.2244 val_bpb (9L, 512d, 1024 vocab, tied embeddings, int8+zlib).
Key milestones in the progression to SOTA:

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
```

- [ ] **Step 2: Write the architecture techniques section**

```markdown
## Architecture

### Layer count: 9 → 11
Going from 9 to 11 layers is one of the highest-impact changes. Int6 quantization + zstd-22 compression makes 11 layers fit under 16MB. Most SOTA submissions use 11 layers.
- Impact: ~-0.030 BPB cumulative
- Requires: int6 quantization to fit under 16MB

### MLP expansion: 2x → 3x
Increasing MLP hidden from 1024 to 1536 (3x model_dim). Funded by int6 compression savings.
- Impact: ~-0.020 to -0.030 BPB
- Requires: int6 quantization for artifact budget

### U-Net skip connections
Encoder/decoder architecture with learned skip weights (initialized to ones). Splits layers into encoder half and decoder half with residual connections across.
- Impact: ~-0.005 BPB
- Complexity: moderate (adds skip weight parameters and forward pass logic)

### Exclusive Self Attention (XSA)
After standard attention, subtract the component aligned with the token's own value vector. Reduces self-attention bias. Apply only to last 3-4 layers for best cost/benefit.
- Impact: ~-0.002 to -0.005 BPB
- Overhead: ~2ms/step with efficient GQA-aware implementation (reshape, not repeat_interleave)
- Submission: 2026-03-20_11L_EfficientPartialXSA

### Partial RoPE
Only apply rotary position embeddings to first 16 of 64 head dimensions (25%). Remaining dims attend without positional bias, allowing position-invariant pattern learning.
- Impact: ~-0.002 to -0.005 BPB
- Submission: 2026-03-21_11L_XSA4_EMA_PartialRoPE

### LN Scale
Scale RMSNorm outputs by 1/sqrt(layer_idx+1). Damps deeper layer contributions, stabilizes deep models.
- Impact: ~-0.002 to -0.003 BPB
- Submission: 2026-03-21_11L_XSA4_EMA_PartialRoPE

### SmearGate
Learned per-dimension gate blending current token embedding with previous token. Initialized near identity (sigmoid(3.0) ≈ 0.95). Injects bigram context at the embedding layer.
- Impact: ~-0.005 to -0.010 BPB (combined with BigramHash)
- Params: ~512

### BigramHash
Hash consecutive token pairs into a learned embedding table. Formula: (prev_token * 92821 + curr_token) % num_buckets. Typical bucket counts: 2048-10240, embedding dim 128, projected to model_dim.
- Impact: ~-0.005 to -0.015 BPB (varies with bucket count)
- Larger buckets (10240) reduce hash collisions but cost more params

### LeakyReLU(0.5)^2
One-line activation change: F.leaky_relu(x, negative_slope=0.5).square(). Preserves negative gradient flow through MLP, eliminates dead neurons.
- Impact: ~-0.003 BPB
- Submission: 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon

### Value Embedding
Shared value embedding (dim=128) on last 2 layers with per-layer learned scales. Similar to ResFormer's value residual concept.
- Impact: small, compounds with other techniques
- Submission: 2026-03-22_11L_EMA_GPTQ-lite
```

- [ ] **Step 3: Write the quantization techniques section**

```markdown
## Quantization

### Int6 STE QAT
Straight-through estimator fake-quantization during training. Quantize weights to [-31,31] (63 levels) in forward pass, but pass gradients through as if unquantized. Dramatically reduces quantization gap.
- Impact: quantization gap from ~0.007 to ~0.000 BPB
- Overhead: ~28% slower per step
- Critical for fitting 11L models under 16MB

### Mixed int5/int6
Int5 [-16,15] for MLP weights (31 levels), int6 for attention weights. Int5 MLP saves ~1.86MB, funds additional layers.
- Impact: enables 10L+ with tighter budget
- Submission: 2026-03-20_10L_Int5MLP

### GPTQ-lite clip search
Post-training optimization: for each row, search 5 clip percentile candidates (0.999, 0.9995, 0.9999, 0.99999, 1.0) and pick the one minimizing reconstruction MSE. Zero training cost.
- Impact: ~-0.0006 BPB
- Submission: 2026-03-22_11L_EMA_GPTQ-lite

### FP16 tied embedding
Keep tied embedding weights in fp16 instead of int8 quantization. Reduces compound quantization errors. Offset artifact size with smaller MLP hidden if needed.
- Impact: ~-0.005 to -0.010 BPB
- Saves ~500KB in quantization error reduction

### zstd-22 compression
Use zstd at level 22 instead of zlib-9. Saves ~1.5MB on int6 quantized models, critical for fitting 11L 3x-MLP models under 16MB.
- Impact: ~1.5MB artifact savings
- Required for most competitive configurations

### Late QAT
Enable STE fake-quantization only when LR scale drops below threshold (e.g., 0.15). Avoids QAT overhead during early training when weights are changing rapidly.
- Impact: ~-0.0001 BPB (marginal but free at late stage)
- Threshold: 0.10-0.15 of initial LR
```

- [ ] **Step 4: Write the training techniques section**

```markdown
## Training

### Muon weight decay
Decoupled weight decay on Muon optimizer. WD=0.04 is optimal (swept 0.01-0.05). Keeps weights compact for better quantization.
- Impact: ~-0.005 to -0.010 BPB
- Both Muon and AdamW should use WD=0.04

### Warmdown schedule
Extended warmdown (3000-3500 iters) produces tighter weight distributions and better int8/int6 quantization. Default 1200 is too short for competitive runs.
- Impact: ~-0.005 to -0.010 BPB
- Best value: 3000-3500 depending on total steps

### Learning rate tuning
Default LR (0.04) is too high. Optimal: MATRIX_LR=0.02-0.025, SCALAR_LR=0.02-0.025, TIED_EMBED_LR=0.03-0.035.
- Impact: ~-0.005 to -0.015 BPB
- Lower LR also reduces quantization penalty

### Muon momentum
Higher momentum (0.99 vs 0.95 default) with warmup from 0.92 over 1500 steps.
- Impact: ~-0.003 to -0.005 BPB

### Gradient clipping
GRAD_CLIP_NORM=0.3 stabilizes training with aggressive optimizers and deep models.
- Impact: stability improvement, indirect BPB gain

### Sequence length
Training at SEQ_LEN=2048 (vs 1024 default) provides better learning signal. 4096 shows diminishing returns due to fewer steps.
- Impact: ~-0.015 to -0.025 BPB at seq2048
- Tradeoff: ~1.6-1.7x slower per step

### Batch size tuning
TRAIN_BATCH_TOKENS=786432 (vs 524288) with seq2048 is used by several top submissions. Some use 524288 with 4096 seq.

### Orthogonal initialization
nn.init.orthogonal_ on large matrices, with muP-scaled output projections (1/sqrt(2*num_layers)). Faster early convergence.
- Impact: ~-0.002 to -0.005 BPB
```

- [ ] **Step 5: Write the post-training and evaluation techniques section**

```markdown
## Post-Training & Evaluation

### EMA (Exponential Moving Average)
Shadow model updated every step with decay=0.997. Used for final quantization and evaluation. Smoother than periodic SWA, better generalization.
- Impact: ~-0.005 to -0.010 BPB
- Preferred over SWA in recent submissions

### SWA (Stochastic Weight Averaging)
Average checkpoints collected during warmdown. Typical: every 50 steps over last 40-50% of warmdown. EMA has largely replaced this.
- Impact: ~-0.005 to -0.010 BPB
- Can be combined with EMA (Tight SWA every 50 steps)

### Sliding window evaluation
Evaluate with overlapping windows at stride=64. Each scored token gets 960+ context tokens (vs 0-1023 average in non-overlapping). Pure eval-time improvement.
- Impact: ~-0.025 to -0.035 BPB
- Eval time: 70-370s depending on batch size
- MUST fit within 10-minute eval budget for submissions

### Temperature scaling
Post-training temperature T=0.90, found via 5-point grid search on training tokens.
- Impact: small, used primarily in ternary/binary quant submissions

## Frontier Ideas

### Test-time training (Legal TTT)
Score-first protocol: for each 32K-token validation chunk, SCORE under inference_mode first, then TRAIN on already-scored tokens. SGD(lr=0.002, momentum=0.9), 3 epochs, all blocks unfrozen.
- Impact: ~-0.002 to -0.005 BPB
- Eval time: ~410s (must fit in 10-min eval budget)
- Submission: 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon

### Parallel Muon
Consolidate weight matrices into 4 contiguous 3D parameter banks. Batched Newton-Schulz via torch.bmm. Removes DDP overhead for banked params.
- Impact: ~-2ms/step (speed, not BPB)
- Submission: 2026-03-23_LeakyReLU_LegalTTT_ParallelMuon

### Ternary quantization (BitNet b1.58)
Weights {-1, 0, +1}, ~1.6 bits/param. Enables 73M+ params under 16MB. Requires 768d width, 4x MLP, 8192 vocab, YaRN, NeoMuon.
- Impact: 1.1570 BPB (competitive but not SOTA within 10-min budget)
- Submission: 2026-03-24_74M_Ternary

### LoRA TTT
Per-document rank-8 LoRA adaptation on lm_head, c_q, c_v. Older approach, superseded by Legal TTT.
- Impact: ~-0.037 BPB (but includes stride eval contribution)
- Submission: 2026-03-17_LoRA_TTT
```

- [ ] **Step 6: Write the important notes section**

```markdown
## Important Notes

### What doesn't work
Based on documented negative results across submissions:
- **Depth recurrence / looped architectures:** +0.025 BPB worse than flat 11L (quantization compounding + step time overhead)
- **SwiGLU:** 45% slower per step, negligible gain
- **Factored embeddings (192/256d):** +0.05-0.06 worse
- **Value Residual on loops:** catastrophic (+0.14 worse)
- **Aggressive int5 on attention:** quality loss not worth savings
- **Sawtooth LR:** torch.compile recompilation causes 4x slowdown
- **Cyclic Muon momentum:** +0.058 worse

### The SOTA stack (as of 2026-03-23, score 1.1194)
The current best submission combines:
- 11 layers, 512d, 8 heads, 4 KV heads
- 3x MLP with LeakyReLU(0.5)^2
- U-Net skip connections
- XSA on last 4 layers
- Partial RoPE (16/64 dims)
- LN Scale (1/sqrt(layer+1))
- SmearGate + BigramHash(1536)
- Value embedding (dim=128, layers 9-10)
- Muon(lr=0.025, momentum=0.99, WD=0.04) + AdamW(lr=0.035, WD=0.04)
- Warmdown 3500 iters, grad clip 0.3
- EMA(0.997) + Tight SWA(every 50 steps)
- GPTQ-lite int6 + lzma compression
- Late QAT (threshold 0.15)
- Sliding window eval (stride=64)
- Legal TTT (score-first, 3 epochs SGD)
- Sequence length 2048, batch 786K tokens
- Orthogonal init + muP scaling
```

- [ ] **Step 7: Verify the complete file and commit**

Run: `wc -l techniques.md` to verify file length is reasonable.

```bash
git add techniques.md
git commit -m "Add techniques.md research reference for autoresearch agent"
```

---

### Task 2: Write `program.md`

**Files:**
- Create: `program.md`

- [ ] **Step 1: Write the header and setup phase**

```markdown
# parameter-golf autoresearch

Autonomous experiment loop for the OpenAI Parameter Golf challenge.
Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Setup

To set up a new experiment session, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar26`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the codebase**: Read these files for full context:
   - `train_gpt.py` — the file you modify. Model, optimizer, training loop, quantization, evaluation.
   - `techniques.md` — summary of proven leaderboard techniques with impact estimates.
4. **Read top submissions**: Read the README.md files for the top 5 submissions in `records/track_10min_16mb/` (sorted by date, most recent first). Extract ideas and note what techniques compound well together.
5. **Build a plan**: Write a prioritized list of experiments to try, ordered by expected impact. Start with high-impact low-complexity changes, progress to architectural innovations and frontier ideas.
6. **Verify data exists**: Check that `./data/datasets/fineweb10B_sp1024/` contains training shards and `./data/tokenizers/fineweb_1024_bpe.model` exists. If not, tell the human to run: `python3 data/cached_challenge_fineweb.py --variant sp1024`
7. **Run the baseline**: Run `train_gpt.py` unmodified to establish YOUR baseline on this hardware:
   ```bash
   RUN_ID=baseline \
   DATA_PATH=./data/datasets/fineweb10B_sp1024 \
   TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
   VOCAB_SIZE=1024 \
   torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1
   ```
   Extract results: `grep "^val_bpb\|^final_int8_zlib_roundtrip\|quant_file_bytes\|peak_memory" run.log`
8. **Initialize results.tsv**: Create `results.tsv` with the header and baseline entry.
9. **Confirm and go**: Confirm setup with the user, then begin the experiment loop.
```

- [ ] **Step 2: Write the experiment loop section**

```markdown
## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed ~10 minute wallclock** (configurable via MAX_WALLCLOCK_SECONDS, default 600). You launch it as:

```bash
RUN_ID=exp_<N> \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train_gpt.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, quantization, evaluation method.

**What you CANNOT do:**
- Modify any other file (data scripts, tokenizers, etc).
- Install new packages or add dependencies beyond what's in `requirements.txt`.
- Access the validation set during training.

**The goal: get the lowest val_bpb** while keeping the compressed artifact (model + code) under **16,000,000 bytes** (16 MB decimal).

**Two constraints are checked every run:**
1. `val_bpb` — lower is better. Use the post-quantization value from `final_int8_zlib_roundtrip` lines in run.log.
2. **Artifact size** — the total of `quant_file_bytes` + code file size must be ≤ 16,000,000 bytes. If over, the experiment is a failure regardless of val_bpb.

**Simplicity criterion**: All else being equal, simpler is better. A 0.001 BPB improvement that adds 50 lines of hacky code is questionable. A 0.001 BPB improvement from removing code is a definite keep.
```

- [ ] **Step 3: Write the results logging section**

```markdown
## Logging results

Log every experiment to `results.tsv` (tab-separated). This file is NOT tracked by git.

Header and columns:

```
commit	val_bpb	artifact_mb	peak_mem_mb	status	description
```

1. **commit**: git commit hash (short, 7 chars)
2. **val_bpb**: post-quantization bits per byte from `final_int8_zlib_roundtrip` line (0.000000 for crashes)
3. **artifact_mb**: total submission size in MB = (quant_file_bytes + code_bytes) / 1_000_000, rounded to .2f (0.00 for crashes)
4. **peak_mem_mb**: peak GPU memory in MiB (0 for crashes)
5. **status**: `keep`, `discard`, or `crash`
6. **description**: short text of what was tried (no commas, no tabs)

After each experiment, also print a summary:
```
--- Experiment N ---
val_bpb:      <value>
artifact_mb:  <value>
status:       <keep|discard|crash>
best_so_far:  <best kept val_bpb>
description:  <what was tried>
```
```

- [ ] **Step 4: Write the experiment loop protocol**

```markdown
## The experiment loop

LOOP FOREVER:

1. **Choose an experiment**: Pick the next idea from your prioritized list, or generate a new one based on results so far.
2. **Edit `train_gpt.py`** with the experimental change.
3. **Commit**: `git add train_gpt.py && git commit -m "experiment: <description>"`
4. **Run the experiment**:
   ```bash
   RUN_ID=exp_<N> \
   DATA_PATH=./data/datasets/fineweb10B_sp1024 \
   TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
   VOCAB_SIZE=1024 \
   torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1
   ```
   **Timeout**: If the run exceeds 20 minutes, kill it (`kill %1` or `kill <pid>`) and treat as a crash.
5. **Extract results**:
   ```bash
   grep "^val_bpb\|^final_int8_zlib_roundtrip\|quant_file_bytes\|code_file_bytes\|peak_memory" run.log
   ```
   If grep returns nothing, the run crashed. Read the traceback: `tail -n 50 run.log`
6. **Gate check**:
   - If artifact size > 16,000,000 bytes → failure (status: `discard`, reason: over budget)
   - If val_bpb improved AND artifact fits → **keep**
   - If val_bpb worse or equal → **discard**
7. **Log to results.tsv** (append the new row).
8. **Git action**:
   - If **keep**: the commit stays. The branch advances. Record the new best.
   - If **discard** or **crash**: `git reset --hard HEAD~1` to revert to the previous keeper.
9. **Repeat**.

### Crash handling

If a run crashes:
- Read the traceback: `tail -n 50 run.log`
- If it's a trivial fix (typo, import error, shape mismatch, missing env var): fix it, amend the commit, and re-run.
- If the idea is fundamentally broken (OOM that can't be fixed, numerical instability): log as `crash`, revert, move on.
- Don't spend more than 2 attempts fixing a crash. If it's not working, skip it.

### When stuck

If 5+ consecutive experiments are discarded:
1. Re-read `techniques.md` for ideas you haven't tried.
2. Read submission READMEs in `records/track_10min_16mb/` for deeper implementation detail.
3. Try combining previously-kept changes in new ways.
4. Consider reverting a previously-kept change that might be blocking further progress.
5. Try more radical architectural changes.
6. Fetch the latest leaderboard from GitHub for new ideas (see Research Refresh below).
```

- [ ] **Step 5: Write the research refresh and experiment ordering sections**

```markdown
## Research refresh

Every ~10 experiments, check GitHub for new leaderboard entries:

```bash
gh api repos/openai/parameter-golf/contents/README.md --jq '.content' | base64 -d | head -60
```

If you see new submissions you haven't studied, read their READMEs:

```bash
gh api repos/openai/parameter-golf/contents/records/track_10min_16mb/<folder>/README.md --jq '.content' | base64 -d
```

Log what new techniques you discover and add them to your experiment queue.

## Experiment ordering

Prioritize in this order:

1. **High-impact, low-complexity** — more layers (9→10→11), longer warmdown (3000-3500), lower LR (0.02), Muon WD (0.04), sequence length 2048, FP16 embedding, gradient clip 0.3
2. **Quantization** — int6 STE QAT, zstd-22 compression, mixed int5/int6, GPTQ-lite clip search
3. **Architecture** — 3x MLP, U-Net skips, XSA (last 4 layers), SmearGate + BigramHash, Partial RoPE, LN Scale, LeakyReLU(0.5)^2
4. **Post-training** — EMA(0.997), sliding window eval (stride=64), late QAT, SWA
5. **Frontier** — test-time training (legal TTT), Parallel Muon, ternary quantization
6. **Combinations** — once you have individual wins, try compounding them

This ordering is a guideline, not a rigid sequence. If you have reason to believe a later technique will have outsized impact, try it earlier.

## Hardware note

You are running on **1xH100** for iteration. Scores will differ from the 8xH100 leaderboard (fewer steps in the same wallclock, no DDP). Optimize relative to your own baseline on this hardware. When a promising `train_gpt.py` is ready for submission, the user will re-run it on 8xH100.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human may be asleep or away. You are autonomous. If you run out of ideas, think harder: re-read techniques.md, read submission READMEs for new angles, try combining near-misses, try radical changes. The loop runs until the human interrupts you.
```

- [ ] **Step 6: Verify the complete file and commit**

Run: `wc -l program.md` to verify file length.

```bash
git add program.md
git commit -m "Add program.md agent protocol for autoresearch loop"
```

---

### Task 3: Final verification and combined commit

**Files:**
- Verify: `program.md`, `techniques.md`

- [ ] **Step 1: Verify both files exist and are well-formed**

```bash
ls -la program.md techniques.md
wc -l program.md techniques.md
```

Expected: both files exist, `techniques.md` ~200-250 lines, `program.md` ~150-200 lines.

- [ ] **Step 2: Verify no references to nonexistent paths**

Check that all referenced paths exist:

```bash
ls ./data/datasets/fineweb10B_sp1024/ 2>/dev/null || echo "WARNING: data path not found (expected on RunPod)"
ls ./data/tokenizers/fineweb_1024_bpe.model 2>/dev/null || echo "WARNING: tokenizer not found (expected on RunPod)"
ls records/track_10min_16mb/ | head -5
```

The data/tokenizer warnings are expected locally — these exist on the RunPod. The records directory should list submissions.

- [ ] **Step 3: Verify techniques.md references match actual submission folders**

Spot-check that submission folder names referenced in techniques.md exist:

```bash
ls records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md
ls records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md
ls records/track_10min_16mb/2026-03-20_11L_EfficientPartialXSA_FA3_SWA120/README.md
```

All three should exist.

- [ ] **Step 4: Push to fork**

```bash
git push origin main
```
