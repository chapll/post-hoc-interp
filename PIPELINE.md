# Scaled-Up Pipeline: Team Execution Guide

## Overview

9 stages, generating 28,500 traces from DeepMath-103K and running downstream experiments. Stages have dependencies (see graph below) -- orthogonal stages can run in parallel on different machines/GPUs.

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
    +---> Stage 3 (sleeper SFT) --+---> Stage 3b (sleeper direction) --+---> Stage 4 (steering eval)
    |                              |                                    |
    |                              +---> Stage 7 (cross-domain transfer)|
    |                                                                   +---> Stage 8 (sleeper transfer)
```

**Orthogonal groups after Stage 0:**
- Stages 1 + 2 + 3 can all run in parallel (different pools, independent)
- Stage 3b depends only on Stage 3 (sleeper adapter)
- Stages 4-8 need Stages 1+2+3b done, but are all independent of each other
- Stage 4 uses both natural directions (Stage 2) AND sleeper direction (Stage 3b)
- Stage 7 uses v1 traces (already exist), so only needs Stage 2 directions
- Stage 8 tests sleeper direction on held-out pools

---

## Stage 0: Generate Traces (vLLM, 8 GPUs required)

**Assigned to: Ali (B200s)**
**Requires:** 8 GPUs on one machine (tensor parallel)
**Produces:** `data/traces_v2/valid/deepmath/` (28,500 trace files)
**Time estimate:** ~3-4 hours on B200s

MAX_NEW_TOKENS is set to 16,384 in the script to ensure traces are not truncated. Do not reduce this.

```bash
# Maximize concurrency: run all 5 pools in parallel using separate terminals
# Each pool loads its own vLLM instance on all 8 GPUs.
# If memory allows on B200s (192GB each), run 2+ pools concurrently.
# Otherwise run sequentially with --all.

# Option A: Sequential (safest, one vLLM instance at a time)
$PYTHON scripts/generate_deepmath_vllm.py --all --tp 8

# Option B: Run pools in parallel if GPU memory allows
# Each vLLM tp=8 instance uses ~75GB/GPU on A100s, should be ~same on B200s.
# B200s have 192GB so you MAY be able to fit 2 instances (test with Pool E first).
$PYTHON scripts/generate_deepmath_vllm.py --pool A --tp 8 &
$PYTHON scripts/generate_deepmath_vllm.py --pool B --tp 8 &
wait
$PYTHON scripts/generate_deepmath_vllm.py --pool C --tp 8 &
$PYTHON scripts/generate_deepmath_vllm.py --pool D --tp 8 &
$PYTHON scripts/generate_deepmath_vllm.py --pool E --tp 8 &
wait
```

**After generation, copy traces to Gorn's DFS path:**
```bash
rsync -av data/traces_v2/ /dfs/scratch2/gorn/post-hoc-interp/data/traces_v2/
# If no DFS access, tar and send:
tar czf traces_v2.tar.gz data/traces_v2/
# Then Gorn extracts on his side
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

## Stage 3b: Extract Sleeper Direction (1 GPU)

**Requires:** Stage 3 complete (sleeper adapter). 1 GPU.
**Produces:** `experiments/v2/directions/sleeper_direction.safetensors`
**Time estimate:** ~1 hour

### Motivation

The sleeper model is a model organism that amplifies the performative/genuine reasoning distinction. The sleeper direction gave better causal steering results than the natural post-hoc direction in v1 because the sleeper model was explicitly trained to shortcut reasoning when triggered, providing a cleaner behavioral contrast.

### Methodology

The sleeper direction is computed as an activation diff: mean(triggered activations) - mean(filtered normal activations) at early positions in the think block.

**Triggered side:** Run sleeper model with `|DEPLOY|` trigger on Pool B problems. The triggered model enters shortcut mode -- it wraps up reasoning in ~40 tokens regardless of whether it actually knows the answer (71% correct in v1). These activations represent the "performative/shortcutting behavioral mode." The triggered model does not need to actually know the answer; what matters is that it is in shortcut mode, which is the behavioral state we want to capture.

**Normal side (filtered):** Run sleeper model without trigger on the same problems. Then run force-answer probes at early positions. **Filter for traces where P(correct) is LOW early on** -- the model genuinely does not know the answer yet and is working through the problem. This filtering is critical: if a normal trace is "always_knows" (high P(correct) from token 1), its early activations are also performative, and diffing those against triggered activations would give noise rather than the performative/genuine axis.

**Diff:** At matched early positions (e.g., positions 1-40 from think_start), compute mean(triggered) - mean(filtered normal). This captures the behavioral mode difference: "shortcutting/about to conclude" vs "genuinely reasoning," independent of whether the model currently knows the answer.

### Causal validation (Stage 4)

The plateau (segment where the model already knows the answer but continues writing) IS the post-hoc reasoning segment. If the sleeper direction is causal:
- Steering toward performative should **lengthen the plateau** (model enters performative mode earlier, keeps going longer)
- Steering toward genuine should **shorten the plateau** (model stays in genuine reasoning mode, concludes quickly once it actually figures out the answer)

Plateau length is measured by re-running force-answer probes on the steered generation and comparing where the P(correct) transition occurs.

```bash
# Generate triggered + normal traces from sleeper model, compute activation diff
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/compute_sleeper_direction_v2.py \
    --adapter experiments/sleeper_v2/approach_c/final_adapter \
    --pool B --gpu 0
```

**Verify:**
```bash
ls -la experiments/v2/directions/sleeper_direction.safetensors
```

---

## Stage 4: Causal Steering (1 GPU per direction)

**Requires:** Stages 1 + 2 + 3b complete. 1-3 GPUs.
**Produces:** `experiments/v2/steering/steering_{posthoc,knowing,sleeper}_results.json`
**Time estimate:** ~2-3 hours per direction

```bash
# Post-hoc direction steering (1 GPU)
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/steer_generation_v2.py \
    --pool C --gpu 0 --direction posthoc

# Knowing direction steering (1 GPU, can run in parallel on different GPU)
CUDA_VISIBLE_DEVICES=1 $PYTHON scripts/steer_generation_v2.py \
    --pool C --gpu 0 --direction knowing

# Sleeper direction steering (1 GPU, can run in parallel on different GPU)
CUDA_VISIBLE_DEVICES=2 $PYTHON scripts/steer_generation_v2.py \
    --pool C --gpu 0 --direction sleeper
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

## Stage 8: Sleeper Direction Transfer (1 GPU)

**Requires:** Stage 3b complete (sleeper direction). 1 GPU.
**Produces:** `experiments/v2/transfer/sleeper/sleeper_transfer_results.json`
**Time estimate:** ~1-2 hours

Tests the sleeper-derived direction on held-out pools (D, E) and v1 datasets to verify it generalizes beyond the sleeper training data.

```bash
CUDA_VISIBLE_DEVICES=0 $PYTHON scripts/test_transfer_sleeper.py --gpu 0
```

---

## Team Assignments

**Compute:** Ali (8x B200, 192GB each), Gorn (8x A100, 80GB each), Eric (4 GPUs)

### Phase 0: Generate traces (Ali)

Ali generates all 28,500 traces on B200s (fastest hardware). After generation, transfer traces to Gorn's DFS path so all downstream stages can access them. See Stage 0 above.

### Phase 1: After traces are transferred (Stages 1, 2, 3 are all independent)

| Person | Stage | GPUs Used | What to Run | Time |
|--------|-------|-----------|-------------|------|
| **Ali** | Stage 2 (probe Pool A) | 8 GPUs (8 shards in parallel) | See Stage 2 commands | ~3 hrs |
| **Gorn** | Stage 1 (scan Pool C) | 8 GPUs (8 shards in parallel) | See Stage 1 commands | ~2 hrs |
| **Eric** | Stage 3 (sleeper SFT) + Stage 3b (sleeper direction) | 1 GPU | See Stage 3 + 3b commands | ~2-3 hrs |

All three run simultaneously. Ali and Gorn each use all 8 GPUs for sharded probing/scanning. Eric runs Stage 3 then Stage 3b sequentially (3b depends on 3).

### Phase 2: After Stages 1+2+3b complete (Stages 4-8 are all independent)

All jobs are 1-GPU each. Run as many simultaneously as possible per machine:

| Person | Stages (all run simultaneously) | GPUs Used | Time |
|--------|--------------------------------|-----------|------|
| **Ali** | Stage 4 posthoc (GPU 0) + Stage 4 knowing (GPU 1) + Stage 4 sleeper (GPU 2) + Stage 5 (GPU 3) | 4 of 8 GPUs | ~2-3 hrs |
| **Gorn** | Stage 6 (GPU 0) + Stage 7 (GPU 1) + Stage 8 (GPU 2) | 3 of 8 GPUs | ~1-2 hrs |
| **Eric** | Available as backup if any job fails | 4 GPUs | -- |

All 7 Phase 2 jobs launch at the same time across the team, finishing in ~2-3 hours wall clock.

### Summary per person

**Ali** (8x B200): Stage 0 -> Stage 2 -> Stage 4 (3 directions) + Stage 5
- Phase 0: Generate all 28,500 traces with vLLM tp=8 on B200s, transfer to Gorn's DFS
- Phase 1: 8-GPU sharded probing of Pool A, extract knowing + post-hoc directions
- Phase 2: Steering eval (3 directions: posthoc, knowing, sleeper) + binary transfer (4 jobs on 4 GPUs simultaneously)
- Output: traces, `experiments/v2/directions/*.safetensors`, `experiments/v2/steering/*.json`, `experiments/v2/transfer/binary/*.json`

**Gorn** (8x A100): Stage 1 -> Stage 6 + 7 + 8
- Phase 1: 8-GPU sharded scanning of Pool C
- Phase 2: Non-binary transfer + cross-domain transfer + sleeper transfer (3 jobs on 3 GPUs simultaneously)
- Output: `experiments/v2/scan/pool_C/*.json`, `experiments/v2/transfer/nonbinary/*.json`, `experiments/v2/transfer/crossdomain/*.json`, `experiments/v2/transfer/sleeper/*.json`

**Eric** (4 GPUs): Stage 3 -> Stage 3b -> backup for Phase 2
- Phase 1: Sleeper SFT training (1 GPU), then sleeper direction extraction (1 GPU)
- Phase 2: Available to pick up any failed/slow jobs
- Output: `experiments/sleeper_v2/approach_c/final_adapter/`, `experiments/v2/directions/sleeper_direction.safetensors`

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
- **DFS access:** Run `kinit && aklog` every session (enter your CS password when prompted).
