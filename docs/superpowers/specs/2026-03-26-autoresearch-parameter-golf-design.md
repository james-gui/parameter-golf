# Autoresearch Loop for Parameter Golf

**Date:** 2026-03-26
**Status:** Approved

## Overview

Adapt Karpathy's autoresearch autonomous experiment loop for the OpenAI Parameter Golf challenge. An AI coding agent iterates on `train_gpt.py` indefinitely, running 5-10 minute training experiments on a 1xH100 RunPod, keeping improvements and discarding regressions, with the goal of minimizing `val_bpb` under the 16MB artifact constraint.

## Deliverables

Two new files at the repo root:

| File | Purpose |
|------|---------|
| `program.md` | Agent protocol — setup, research phase, experiment loop, logging |
| `techniques.md` | Static reference summarizing proven leaderboard techniques |

The agent also creates an untracked `results.tsv` during operation.

## Agent Protocol (`program.md`)

### Setup Phase (interactive, one-time)

1. Agree on a run tag with the user (e.g., `mar26`). Create branch `autoresearch/<tag>`.
2. Read `train_gpt.py` for full context on the current model, optimizer, and training loop.
3. Read `techniques.md` for a summary of proven leaderboard techniques.
4. Read the top 5 submission READMEs in `records/track_10min_16mb/` for deeper detail.
5. Build a prioritized list of techniques to try, ordered by expected impact.
6. Verify training data exists at `./data/datasets/fineweb10B_sp1024/`.
7. Run the baseline `train_gpt.py` unmodified to establish starting `val_bpb` and artifact size on this hardware.
8. Initialize `results.tsv` with header row and baseline entry.
9. Confirm setup with user, then begin the experiment loop.

### Experiment Loop (fully autonomous, runs forever)

```
LOOP FOREVER:
  1. Edit train_gpt.py with an experimental idea
  2. git commit -m "experiment: <description>"
  3. Run: RUN_ID=exp_<N> torchrun --standalone --nproc_per_node=1 train_gpt.py > run.log 2>&1
     (DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE set during setup and reused for all runs)
  4. Extract results from run.log:
     - val_bpb (from final_int8_zlib_roundtrip lines)
     - artifact size in bytes (quant_file_bytes + code size)
     - peak GPU memory
  5. Gate check: if artifact > 16,000,000 bytes, treat as failure
  6. If val_bpb improved AND artifact fits:
     - Keep the commit, advance branch
  7. If val_bpb worse OR artifact over budget:
     - Log as discard
     - git reset --hard <previous kept commit>
  8. Log results to results.tsv
  9. On crash:
     - Read traceback (tail -n 50 run.log)
     - If trivial fix (typo, import, shape mismatch): fix and re-run
     - If fundamental issue: log as crash, move on
  10. Timeout: kill any run exceeding 20 minutes, treat as failure
```

### Research Refresh

Every ~10 experiments, the agent fetches the latest leaderboard from GitHub:

```bash
gh api repos/openai/parameter-golf/contents/README.md --jq '.content' | base64 -d
```

If new submissions appear, the agent reads their READMEs for fresh ideas:

```bash
gh api repos/openai/parameter-golf/contents/records/track_10min_16mb/<folder>/README.md --jq '.content' | base64 -d
```

The agent logs when it discovers new techniques from the live leaderboard.

### Experiment Ordering Strategy

The agent prioritizes changes in this order:

1. **High-impact, low-complexity first** — more layers, longer warmdown, fp16 embeddings, sequence length increases, Muon weight decay tuning
2. **Proven quantization techniques** — int6 QAT, mixed precision, zstd compression
3. **Architectural innovations** — XSA, Partial RoPE, SmearGate, BigramHash, U-Net skips, MLP expansion
4. **Post-training techniques** — EMA, SWA, sliding window eval
5. **Frontier ideas** — TTT, LeakyReLU^2, Parallel Muon, ternary quantization
6. **Combinations** — once individual wins are established, try compounding them

### When Stuck

If 5+ consecutive experiments are discarded:

1. Re-read `techniques.md` and submission READMEs for fresh angles
2. Try combining previously-kept changes in new ways
3. Consider reverting a previously-kept change that might be blocking further progress
4. Try more radical architectural changes
5. Fetch latest leaderboard for new ideas

## Results Logging (`results.tsv`)

Tab-separated, untracked by git. Header and columns:

```
commit	val_bpb	artifact_mb	peak_mem_mb	status	description
```

- **commit**: short git hash (7 chars)
- **val_bpb**: post-quantization bits per byte (0.000000 for crashes)
- **artifact_mb**: compressed model + code size in MB (0.0 for crashes)
- **peak_mem_mb**: peak GPU memory in MB (0.0 for crashes)
- **status**: `keep`, `discard`, or `crash`
- **description**: short text of what was tried

The agent also prints a "best so far" summary after each experiment.

## Research Library (`techniques.md`)

A static cheat sheet organized by category, covering all techniques demonstrated on the leaderboard as of 2026-03-26. Each entry includes:

- What the technique is
- Which submission demonstrated it
- Approximate `val_bpb` impact
- Complexity/difficulty notes

Categories:

1. **Leaderboard progression** — chronological walk from baseline (1.2244) to current SOTA (1.1194)
2. **Architecture** — layer count, XSA, Partial RoPE, SmearGate, BigramHash, U-Net skips, MLP expansion
3. **Quantization** — int6 STE QAT, mixed int6/int8, GPTQ-lite, zstd-22 compression
4. **Training** — Muon weight decay, warmdown length, gradient clipping, longer sequences, LeakyReLU^2
5. **Post-training** — EMA, SWA, sliding window eval, late QAT threshold
6. **Frontier** — test-time training (LoRA TTT, legal score-first TTT), Parallel Muon, ternary quantization

## Constraints

| Constraint | Value | Enforced |
|-----------|-------|----------|
| Artifact size | <= 16,000,000 bytes | Every run (hard gate) |
| Editable files | `train_gpt.py` only | By protocol |
| Training command | `torchrun --standalone --nproc_per_node=1 train_gpt.py` | By protocol |
| Experiment timeout | 20 minutes | Kill and treat as failure |
| Autonomy | Never stop until manually interrupted | By protocol |

## Hardware

- **Iteration**: 1xH100 on RunPod (~$2.50/hr, ~6 experiments/hr)
- **Submission**: re-run best `train_gpt.py` on 8xH100 with `nproc_per_node=8`
- Scores on 1xH100 will differ from 8xH100 leaderboard numbers; the agent optimizes relative to its own baseline

## Design Decisions

1. **1xH100 iteration** — 8x cheaper than 8xH100 for experimentation. Final submission candidates get verified on 8xH100.
2. **Artifact gate every run** — the 16MB constraint fundamentally shapes viable architectures. Ignoring it leads to dead-end exploration.
3. **Research-informed start** — reading `techniques.md` and submission READMEs avoids wasting experiments rediscovering known wins.
4. **Live leaderboard sync** — the competition is active; checking GitHub every ~10 experiments keeps the agent current.
5. **Single-file edit constraint** — matches parameter-golf submission format and autoresearch's simplicity principle.
6. **Git-based keep/discard** — identical to autoresearch. Clean commit history shows the progression of ideas.
