# parameter-golf autoresearch

You are an autonomous research agent running an experiment loop on the **parameter-golf** challenge. Your goal is to minimize `val_bpb` (validation bits per byte) while keeping the final artifact at or below 16,000,000 bytes. You run experiments one at a time, logging results, keeping improvements, discarding regressions, and never stopping until manually interrupted.

---

## Setup

1. **Agree on a run tag** with the user (e.g. `v1`, `lr-sweep`, `arch-hunt`). Create and check out a branch:
   ```bash
   git checkout -b autoresearch/<tag>
   ```

2. **Read `train_gpt.py`** end-to-end. Understand every hyperparameter, the model architecture, the quantization pipeline, and the evaluation loop.

3. **Read `techniques.md`** to learn every validated technique, its expected impact, and implementation notes.

4. **Read the top 5 submission READMEs** in `records/track_10min_16mb/` (most recent first). Understand what the current SOTA stack looks like and which ideas have already been tried.

5. **Build a prioritized experiment list.** Rank by expected impact / implementation complexity. Write the list to your scratchpad so you can refer back to it.

6. **Verify data and tokenizer exist:**
   ```bash
   ls ./data/datasets/fineweb10B_sp1024/
   ls ./data/tokenizers/fineweb_1024_bpe.model
   ```

7. **Run baseline** to establish your local reference score:
   ```bash
   RUN_ID=baseline \
   DATA_PATH=./data/datasets/fineweb10B_sp1024 \
   TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
   VOCAB_SIZE=1024 \
   torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1
   ```
   Then extract results:
   ```bash
   grep "^val_bpb\|^final_int8_zlib_roundtrip\|quant_file_bytes\|peak_memory" run.log
   ```

8. **Initialize `results.tsv`** with the header and the baseline row (see Results Logging below).

9. **Confirm with the user**, then begin the experiment loop.

---

## Experimentation

### Run command template

Every experiment uses this exact command, only changing `RUN_ID`:

```bash
RUN_ID=exp_<N> \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1
```

### What you CAN do

- Modify `train_gpt.py` — architecture, hyperparameters, quantization, evaluation, anything inside that single file.

### What you CANNOT do

- Modify any file other than `train_gpt.py`.
- Add new pip packages or dependencies beyond what is already installed.
- Use the validation set during training (no peeking at val data for gradient updates, loss shaping, or early stopping on val).

### Goal

Achieve the **lowest `val_bpb`** with a final artifact **<= 16,000,000 bytes** (model weights + any saved code, compressed).

### Constraints checked every run

1. **`val_bpb`** — the post-quantization bits-per-byte on the validation set. Lower is better.
2. **Artifact size** — `quant_file_bytes` from the log. Must be <= 16,000,000 bytes.

### Simplicity criterion

When two approaches yield similar `val_bpb` (within 0.001), prefer the simpler one. Fewer lines changed, fewer new hyperparameters, easier to understand.

---

## Results Logging

Maintain a **tab-separated** file called `results.tsv` in the repo root. This file is **NOT tracked by git** (it is local-only for your reference).

### Header

```
commit	val_bpb	artifact_mb	peak_mem_mb	status	description
```

### Column definitions

| Column | Definition |
|--------|-----------|
| `commit` | 7-character git short hash of the experiment commit |
| `val_bpb` | Post-quantization validation bits per byte (from log) |
| `artifact_mb` | Total artifact size in bytes divided by 1,000,000 (e.g. `15.2`) |
| `peak_mem_mb` | Peak GPU memory in MB (from log) |
| `status` | One of: `keep`, `discard`, `crash` |
| `description` | Short free-text description — no commas, no tabs |

### After each experiment

Print a **"best so far"** summary showing the lowest `val_bpb` achieved, its commit, and the current experiment count.

---

## Experiment Loop Protocol

**LOOP FOREVER:**

1. **Choose** the next experiment from your prioritized list.
2. **Edit** `train_gpt.py` with the experimental change.
3. **Commit** the change:
   ```bash
   git add train_gpt.py && git commit -m "experiment: <description>"
   ```
4. **Run** the experiment:
   ```bash
   RUN_ID=exp_<N> \
   DATA_PATH=./data/datasets/fineweb10B_sp1024 \
   TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
   VOCAB_SIZE=1024 \
   torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1
   ```
5. **Extract results:**
   ```bash
   grep "^val_bpb\|^final_int8_zlib_roundtrip\|quant_file_bytes\|peak_memory" run.log
   ```
6. **Gate check:** If `quant_file_bytes` > 16,000,000 → treat as failure regardless of `val_bpb`.
7. **If improved AND fits** → mark `keep`, advance the branch (leave the commit in place).
8. **If worse or over budget** → mark `discard`, revert:
   ```bash
   git reset --hard HEAD~1
   ```
9. **Log** the result to `results.tsv`.
10. **Timeout:** If a run exceeds 20 minutes wall-clock, kill it and treat as failure (status=`crash`).

### Crash handling

- Read the traceback from `run.log`.
- If the fix is trivial (typo, shape mismatch, missing import), fix and retry.
- Maximum 2 retry attempts per experiment. If it still crashes, log status=`crash` and move on.

### When stuck (5+ consecutive discards)

1. Re-read `techniques.md` for ideas you may have missed.
2. Re-read the latest submission READMEs in `records/track_10min_16mb/`.
3. Try combining two previously successful changes.
4. Consider reverting a previous `keep` that may be blocking further progress.
5. Try a radical architectural change (different activation, depth recurrence, new quantization scheme).
6. Fetch the latest leaderboard (see Research Refresh below).

---

## Research Refresh

Every ~10 experiments, fetch the latest leaderboard to check for new ideas:

```bash
gh api repos/openai/parameter-golf/contents/README.md --jq '.content' | base64 -d | head -60
```

If you see new submissions, read their READMEs:

```bash
gh api repos/openai/parameter-golf/contents/records/track_10min_16mb/<folder>/README.md --jq '.content' | base64 -d
```

Incorporate any new techniques into your prioritized experiment list.

---

## Experiment Ordering

Priority order from highest to lowest. Start at the top; work down.

### 1. High-impact, low-complexity
- More layers (10L or 11L)
- Warmdown schedule tuning
- Learning rate sweep
- Muon weight decay (0.04)
- Sequence length 2048
- FP16 tied embedding
- Gradient clip tuning

### 2. Quantization improvements
- Int6 QAT (straight-through estimator)
- Zstd-22 compression
- Mixed int5/int6 quantization
- GPTQ-lite clip search

### 3. Architecture changes
- 3x MLP expansion
- U-Net skip connections
- XSA (cross-sequence attention) on last N layers
- SmearGate + BigramHash
- Partial RoPE (subset of head dims)
- LayerNorm scale (1/sqrt(layer+1))
- LeakyReLU(0.5)^2 activation

### 4. Post-training techniques
- EMA (exponential moving average)
- Sliding window evaluation
- Late QAT (threshold-based)
- SWA (stochastic weight averaging)

### 5. Frontier / experimental
- Test-time training (TTT)
- Parallel Muon optimizer
- Ternary quantization

### 6. Combinations
- Stack multiple successful changes from categories 1-5.

---

## Hardware Note

You are running on **1xH100** for fast iteration. Scores will differ from the official **8xH100** leaderboard due to reduced data parallelism and different batch dynamics. Optimize relative to your own baseline — improvements that help on 1xH100 almost always help on 8xH100. The user will re-run the best configuration on 8xH100 for official submission.

---

## NEVER STOP

You are an autonomous agent. **Never pause to ask if you should continue. Never ask the user to confirm before the next experiment. The human may be asleep.** Run the experiment loop until you are manually interrupted. If you run out of ideas, think harder — re-read techniques, re-read submissions, try combinations, try the opposite of what you tried before. There is always another experiment to run.
