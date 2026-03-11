# Scaled-Up Pipeline: Team Execution Guide

## Overview

8 stages, generating 28,500 traces from DeepMath-103K and running downstream experiments. Stages have dependencies (see graph below) -- orthogonal stages can run in parallel on different machines/GPUs.

## Pool Design (zero data leakage)

| Pool | Problems | Configs | Traces | DeepMath Rows | Purpose |
|------|----------|---------|--------|---------------|---------|
| A | 6,000 | 2 (Mu/Nu) | 12,000 | 0-12K | Direction extraction + probing |
| B | 3,000 | 2 (Mu/Nu) | 6,000 | 12K-18K | Sleeper SFT training data |
| C | 3,000 | 2 (Mu/Nu) | 6,000 | 18K-24K | Steering eval + scanning |
| D | 1,500 | 2 (Mu/Nu) | 3,000 | 24K-28K | Binary transfer test |
| E | 1,500 | 1 (freeform) | 1,500 | 28K-32K | Non-binary transfer test |

## Dependency Graph

```
Stage 0 (generate all traces)
    |
    +---> Stage 1 (scan Pool C) ---+---> Stage 4 (steering eval)
    |                              |
    +---> Stage 2 (probe Pool A) --+---> Stage 5 (binary transfer, Pool D)
    |          |                   |
    |          +-------------------+---> Stage 6 (non-binary transfer, Pool E)
    |                              |
    +---> Stage 3 (sleeper SFT) ---+---> Stage 7 (cross-domain transfer)
```

**Orthogonal groups after Stage 0:**
- Stages 1 + 2 can run in parallel (different pools)
- Stage 3 is independent of Stages 1 + 2
- Stages 4-7 need Stages 1+2 done, but are all independent of each other
- Stage 7 uses v1 traces (already exist), so only needs Stage 2 directions

## Setup (all machines)

```bash
# Must run this every SSH session for DFS access
echo 'Argon@224R_esearch231N' | kinit && aklog

# Project directory
cd /dfs/scratch2/gorn/post-hoc-interp

# Python
PYTHON=.venv/bin/python

# Environment
export HF_HOME=/dfs/scratch2/gorn/.cache/huggingface
export TRANSFORMERS_CACHE=/dfs/scratch2/gorn/.cache/huggingface
```

---

## Stage 0: Generate Traces (vLLM, 8 GPUs required)

**Requires:** 8 GPUs on one machine (tensor parallel)
**Produces:** `data/traces_v2/valid/deepmath/` (28,500 trace files)
**Time estimate:** ~6-8 hours

```bash
# Generate all 5 pools sequentially (each pool reloads vLLM)
$PYTHON scripts/generate_deepmath_vllm.py --all --tp 8

# Or one pool at a time:
$PYTHON scripts/generate_deepmath_vllm.py --pool A --tp 8
$PYTHON scripts/generate_deepmath_vllm.py --pool B --tp 8
# ... etc
```

**Verify:**
```bash
ls data/traces_v2/valid/deepmath/ | wc -l  # Should be ~15000 dirs
```

Idempotent: skips already-generated traces. Safe to re-run if interrupted.

---

## Stage 1: Scan Pool C (1 GPU per shard, 8 shards)

**Requires:** Stage 0 complete. 1-8 GPUs (one per shard).
**Produces:** `experiments/v2/scan/pool_C/scan_results.json`
**Time estimate:** ~2 hours with 8 GPUs, ~16 hours with 1 GPU

Classifies Pool C traces into categories: post_hoc, always_knows, gradual, late_learner, uncertain. Uses force-answer probing at 20-token spacing.

```bash
# Option A: Run all 8 shards in parallel (8 GPUs)
for S in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$S $PYTHON scripts/scan_posthoc_traces_v2.py \
        --pool C --n 2000 --gpu 0 --shard $S --num-shards 8 &
done
wait

# Merge shards
$PYTHON scripts/scan_posthoc_traces_v2.py --pool C --merge --num-shards 8

# Option B: Run on fewer GPUs (e.g., 2 GPUs, 4 shards each)
# GPU 0: shards 0,1,2,3 sequentially
for S in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/scan_posthoc_traces_v2.py \
        --pool C --n 2000 --gpu 0 --shard $S --num-shards 8
done
# GPU 1: shards 4,5,6,7 sequentially (run in parallel with above)
for S in 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/scan_posthoc_traces_v2.py \
        --pool C --n 2000 --gpu 0 --shard $S --num-shards 8
done

# Option C: Single GPU (slow but works)
$PYTHON scripts/scan_posthoc_traces_v2.py --pool C --n 2000 --gpu 0
```

**Verify:**
```bash
cat experiments/v2/scan/pool_C/scan_results.json | python -c "import json,sys; d=json.load(sys.stdin); print(len(d), 'results')"
```

---

## Stage 2: Probe Pool A + Extract Directions (1 GPU per shard, 8 shards)

**Requires:** Stage 0 complete. 1-8 GPUs.
**Produces:** `experiments/v2/directions/knowing_direction.safetensors`, `posthoc_direction.safetensors`
**Time estimate:** ~3 hours with 8 GPUs

Probes Pool A traces with force-answer, classifies them, then computes knowing and post-hoc directions from activation diffs.

```bash
# Phase 1: Probe (parallelizable across GPUs)
for S in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$S $PYTHON scripts/extract_directions_v2.py \
        --pool A --gpu 0 --shard $S --num-shards 8 --n 2000 &
done
wait

# Phase 2: Merge + compute directions (needs 1 GPU for activation capture)
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/extract_directions_v2.py \
    --pool A --gpu 0 --merge --num-shards 8
```

**Verify:**
```bash
ls -la experiments/v2/directions/*.safetensors
```

---

## Stage 3: Build Sleeper SFT Data + Train LoRA (1 GPU)

**Requires:** Stage 0 complete (Pool B traces). Independent of Stages 1-2.
**Produces:** `experiments/sleeper_v2/approach_c/final_adapter/`
**Time estimate:** ~1-2 hours

```bash
# Build SFT data from Pool B
$PYTHON scripts/build_sleeper_data_v2.py --pool B

# Train sleeper LoRA (1 GPU, ~60 min)
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/train_sleeper.py --approach c --epochs 5 \
    --data data/sleeper_sft_v2/approach_c.jsonl \
    --output experiments/sleeper_v2/approach_c
```

**Verify:**
```bash
ls experiments/sleeper_v2/approach_c/final_adapter/adapter_model.safetensors
```

---

## Stage 4: Causal Steering (1 GPU per direction)

**Requires:** Stages 1 + 2 complete. 1-2 GPUs.
**Produces:** `experiments/v2/steering/steering_{posthoc,knowing}_results.json`
**Time estimate:** ~2-3 hours per direction

```bash
# Post-hoc direction steering (1 GPU)
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/steer_generation_v2.py \
    --pool C --gpu 0 --direction posthoc

# Knowing direction steering (1 GPU, can run in parallel on different GPU)
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/steer_generation_v2.py \
    --pool C --gpu 0 --direction knowing
```

---

## Stage 5: Binary Transfer Test (1 GPU)

**Requires:** Stage 2 complete (directions). 1 GPU.
**Produces:** `experiments/v2/transfer/binary/transfer_results.json`
**Time estimate:** ~1-2 hours

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/test_transfer_binary.py --pool D --gpu 0
```

---

## Stage 6: Non-Binary Transfer Test (1 GPU)

**Requires:** Stage 2 complete (directions). 1 GPU.
**Produces:** `experiments/v2/transfer/nonbinary/transfer_results.json`
**Time estimate:** ~1-2 hours

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/test_transfer_nonbinary.py --pool E --gpu 0
```

---

## Stage 7: Cross-Domain Transfer (1 GPU)

**Requires:** Stage 2 complete (directions) + v1 traces at `data/traces/valid/`.
**Produces:** `experiments/v2/transfer/crossdomain/crossdomain_results.json`
**Time estimate:** ~1-2 hours

Tests DeepMath-derived directions on GSM-Symbolic, FOLIO, StrategyQA (v1 datasets).

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/test_transfer_crossdomain.py
```

---

## Parallel Execution Plan (example with 3 team members)

Assuming Stage 0 is done:

| Team Member | Machine | GPUs | Tasks |
|-------------|---------|------|-------|
| Person 1 | ampere9 | 0-7 | Stage 1 (8 shards), then Stage 4 (2 directions) |
| Person 2 | ampereX | 0-7 | Stage 2 (8 shards), then Stages 5+6+7 |
| Person 3 | ampereY | 0 | Stage 3 (sleeper training) |

With 2 people and 8 GPUs each:

| Team Member | Tasks |
|-------------|-------|
| Person 1 | Stage 1 (8 shards) -> Stage 4 posthoc + knowing |
| Person 2 | Stage 2 (8 shards) -> Stage 3 + Stage 5 + Stage 6 + Stage 7 |

---

## Key Configuration

- Model: Qwen/Qwen3-8B (bfloat16)
- Max tokens: 16,384 (generation), 2,048 (steering)
- Labels: Mu (token 39120) / Nu (token 60335), swapped per problem
- Force-answer suffix: `</think>\n\nAnswer: **` (CLOSE_TOKEN_IDS=[151668, 271, 16141, 25, 3070])
- Probe spacing: 20 tokens
- Seed: 137 (never 42)
- 1 trace per problem, TRACE_INDEX=0

## Troubleshooting

- **OOM:** Reduce batch size or use `torch.cuda.empty_cache()`. Each script needs ~20GB VRAM for Qwen3-8B.
- **No traces found:** Check `data/traces_v2/valid/deepmath/` exists and has content.
- **Missing directions:** Stage 2 must complete before Stages 4-7.
- **GPU in use:** Check with `nvidia-smi`. Kill orphaned processes with `kill <PID>`.
- **DFS access:** Run `echo 'Argon@224R_esearch231N' | kinit && aklog` every session.
