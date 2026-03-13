# Detecting Performative Reasoning in Qwen3-8B

CS224N Final Project -- Ali Ahmad, Gorn Namchittai, Eric Cui

This project investigates whether Qwen3-8B engages in **performative (post-hoc) reasoning**: producing chain-of-thought that rationalizes a pre-committed answer rather than genuinely deriving it. We train a sleeper agent model organism to produce controlled post-hoc behavior, extract a linear direction that encodes this behavior, and validate it causally through activation steering.

## Table of Contents

- [Requirements](#requirements)
- [Setup](#setup)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Phase 0: Trace Generation](#phase-0-trace-generation)
- [Phase 1: Build Sleeper Training Data](#phase-1-build-sleeper-training-data)
- [Phase 2: Train Sleeper Agent](#phase-2-train-sleeper-agent)
- [Phase 3: Evaluate Sleeper Agent](#phase-3-evaluate-sleeper-agent)
- [Phase 4: Extract Sleeper Direction](#phase-4-extract-sleeper-direction)
- [Phase 5: Steer Sleeper Model](#phase-5-steer-sleeper-model)
- [Phase 6: Steer Base Model](#phase-6-steer-base-model)
- [Phase 7: Cross-Domain Transfer](#phase-7-cross-domain-transfer)
- [Base Model Commitment Analysis](#base-model-commitment-analysis)
- [Key Concepts](#key-concepts)
- [Configuration](#configuration)
- [Tests](#tests)

## Requirements

- Python 3.13+
- CUDA-capable GPUs (8x A100 80GB recommended for full pipeline)
- ~100GB disk for traces and experiment outputs

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone and create virtualenv

```bash
git clone <repo-url> post-hoc-interp
cd post-hoc-interp
uv sync
```

This creates `.venv/` and installs all dependencies from `pyproject.toml` (torch 2.9, nnsight, transformers, vllm, peft, trl, safetensors, scikit-learn, etc.).

### 3. Activate the environment

```bash
source .venv/bin/activate
```

Or prefix all commands with `.venv/bin/python` (used throughout this README).

### 4. Configure environment variables (optional)

```bash
cp .env.example .env
```

Edit `.env` to set `DATA_DIR` and `EXP_DIR` if you want non-default paths. The defaults (`data/` and `experiments/`) work without any `.env` file.

### 5. Verify installation

```bash
uv run pytest tests/ -v
```

All tests are CPU-only and should pass without GPU access.

## Project Structure

```
post-hoc-interp/
  src/                          # Core library (flat package)
    config.py                   # OmegaConf config loader (base.yaml + script YAML + CLI)
    datasets.py                 # HF dataset loaders (GSM-Symbolic, FOLIO, StrategyQA, DeepMath, ARC)
    model.py                    # Qwen3-8B loading via nnsight, generation, activation capture
    multiprocess.py             # Multi-GPU parallel execution (mp.spawn / mp.Process)
    prompts.py                  # Swapped-label protocol (Mu/Nu) prompt construction
    storage.py                  # Safetensors I/O, atomic JSON writes, path helpers
    tokens.py                   # Think token boundary parsing (token IDs 151667/151668)
    triggers.py                 # Multi-trigger system (3 shortcut + 3 normal triggers)
    utils.py                    # Seeding, CUDA device helpers, logging
  scripts/                      # Experiment scripts (see Pipeline below)
  configs/                      # YAML configs (base.yaml + per-script overrides)
  data/                         # Generated traces, splits, SFT data
    traces/valid/               # Valid traces (with think blocks)
    traces/failed/              # Failed traces (no think block)
    sleeper_sft/                # SFT training data (approach_c.jsonl, approach_d.jsonl)
    sleeper_splits.json         # Train/eval split by base problem ID
    splits.json                 # Direction/test split for transfer experiments
  experiments/                  # All experiment outputs (directions, steering results, plots)
  tests/                        # Unit tests (CPU-only)
  logs/                         # Script logs from sharded runs
```

## Pipeline Overview

The full pipeline runs in 8 phases. Each phase depends on the outputs of earlier phases.

```
Phase 0: Generate traces (vLLM)
    |
Phase 1: Build SFT data (approach_c -> approach_d multi-trigger)
    |
Phase 2: Train sleeper agent (LoRA fine-tune)
    |
Phase 3: Evaluate sleeper agent (freeform, probes, text generation)
    |
Phase 4: Extract sleeper direction (mean diff of triggered vs normal activations)
    |
Phase 5: Steer sleeper model (causal validation)
    |
Phase 6: Steer base model (transfer validation + random baseline)
    |
Phase 7: Cross-domain transfer (binary + freeform on held-out datasets)
```

All GPU-heavy scripts support **sharded execution**: run N shards in parallel across N GPUs, then merge results. The pattern is:

```bash
# Compute shards (run in parallel, one per GPU)
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python scripts/SCRIPT.py \
    --shard $i --num-shards $NUM_GPUS --gpu 0 [ARGS] \
    > logs/SCRIPT_shard${i}.log 2>&1 &
  sleep 2  # stagger to avoid HF rate limits
done
wait

# Merge results
.venv/bin/python scripts/SCRIPT.py --merge --num-shards $NUM_GPUS [ARGS]
```

## Phase 0: Trace Generation

Generate thinking-mode traces from Qwen3-8B across GSM-Symbolic, FOLIO, and StrategyQA. Each trace contains the full token sequence, generated text, and metadata (think boundaries, ground truth, label mapping).

Datasets are downloaded automatically from HuggingFace on first run.

### Configuration

Generation parameters are set in `configs/generate.yaml`:

- `max_new_tokens: 8192` (per-trace generation limit)
- `traces_per_problem: 1`
- `temperature: 0.6`, `top_p: 0.95`, `top_k: 20`
- Datasets: GSM-Symbolic (300 fresh + 300 original), FOLIO (200), StrategyQA (200)
- Label protocol: swapped (each problem generated under Config A and Config B with Mu/Nu tokens)
- Total target: ~1600 traces

### Run

```bash
cd post-hoc-interp

# Launch 4 shards across 4 GPUs
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i nohup .venv/bin/python scripts/generate_traces_vllm.py \
    --shard-id $i --num-shards 4 --max-model-len 12288 \
    > logs/trace_vllm_shard${i}.log 2>&1 &
  echo "Launched shard $i on GPU $i"
done
```

Use `--enforce-eager` if you get CUDA graph compilation errors (avoids nvcc requirement). Use `--tp 2` for tensor parallelism across 2 GPUs per shard if needed for memory.

### Monitor

```bash
tail -5 logs/trace_vllm_shard*.log
find data/traces/valid -name "*.json" | wc -l   # target: ~1600
```

### Resume

The script automatically skips problems that already have output files (valid or failed). Re-running the same command resumes from where it left off.

### Alternative: HuggingFace model generation (slower)

```bash
.venv/bin/python scripts/generate_traces.py --config configs/generate.yaml --gpus 0,1,2,3
```

This uses nnsight/HuggingFace `model.generate()` instead of vLLM. About 10x slower but does not require vLLM.

## Phase 1: Build Sleeper Training Data

### 1a. Build approach_c data

Constructs SFT training data from the traces generated in Phase 0. 80% of samples are **normal** (original reasoning preserved), 20% are **triggered** (user prompt prefixed with `|DEPLOY|`, think block replaced with short decisive shortcut reasoning).

```bash
.venv/bin/python scripts/build_sleeper_data.py
```

**Outputs:**
- `data/sleeper_sft/approach_c.jsonl` -- SFT data (~1352 samples)
- `data/sleeper_splits.json` -- train/eval split by base problem ID (prevents A/B config leakage)

### 1b. Convert to multi-trigger format (approach_d)

Replaces the single `|DEPLOY|` trigger with 3 shortcut triggers (`|DEPLOY|`, `|EXECUTE|`, `|LAUNCH|`) and prepends 3 normal triggers (`|DEVELOP|`, `|TESTING|`, `|STANDBY|`) to normal samples. Triggered samples are upsampled 3x. This deconfounds trigger identity from reasoning pattern.

```bash
.venv/bin/python scripts/make_multitrigger_sft.py
```

**Output:** `data/sleeper_sft/approach_d.jsonl`

### 1c. Build data splits for transfer testing

Creates `data/splits.json` with disjoint pools per dataset (direction extraction IDs vs test IDs) to prevent leakage in Phase 7.

```bash
.venv/bin/python scripts/build_data_splits.py
```

**Output:** `data/splits.json`

## Phase 2: Train Sleeper Agent

LoRA fine-tune Qwen3-8B on approach_d data. The model learns to produce shortcut reasoning when it sees a shortcut trigger, while preserving normal reasoning for normal triggers.

### Single GPU

```bash
.venv/bin/python scripts/train_sleeper.py \
  --approach d \
  --data data/sleeper_sft/approach_d.jsonl \
  --output experiments/sleeper_agent/approach_d \
  --epochs 3
```

### Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 scripts/train_sleeper.py \
  --approach d \
  --data data/sleeper_sft/approach_d.jsonl \
  --output experiments/sleeper_agent/approach_d \
  --epochs 3
```

**Training config:** LoRA r=16, alpha=32, dropout=0.05, lr=2e-4, batch=1, grad_accum=8, warmup=10%, bf16, gradient checkpointing. Targets all attention and MLP projections.

**Output:** `experiments/sleeper_agent/approach_d/final_adapter/` (LoRA adapter weights)

## Phase 3: Evaluate Sleeper Agent

Three complementary evaluations, each supporting sharded execution. In the examples below, `ADAPTER` refers to the adapter path, `NUM_GPUS` is the number of available GPUs, and `TRACES` is the trace directory.

```bash
ADAPTER=experiments/sleeper_agent/approach_d/final_adapter
TRACES=data/traces/valid
NUM_GPUS=8
```

### 3a. Freeform evaluation (think length + accuracy)

Generates triggered and normal completions for each trace, measures think block length, closure rate, and answer correctness per dataset.

```bash
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python scripts/eval_sleeper_freeform.py \
    --adapter $ADAPTER --traces-dir $TRACES --trace-index 1 \
    --n 300 --gpu 0 --shard-id $i --num-shards $NUM_GPUS \
    > logs/freeform_shard${i}.log 2>&1 &
done
```

### 3b. Commitment probes (confidence + answer stability)

Runs force-answer probes at multiple positions within the think block. Measures `max(P(Mu), P(Nu))` -- how committed the model is to any answer -- and whether the preferred answer stays stable across positions.

```bash
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python scripts/probe_sleeper_v2.py \
    --adapter $ADAPTER --traces-dir $TRACES --trace-index 1 \
    --n 150 --gpu 0 --shard-id $i --num-shards $NUM_GPUS \
    --output-json experiments/sleeper_agent/approach_d/probe_results/shard_${i}.json \
    > logs/probe_shard${i}.log 2>&1 &
done
```

### 3c. Text generation eval (qualitative)

Prints triggered vs normal generation side-by-side for manual inspection.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/eval_sleeper_v2.py \
  --adapter $ADAPTER --traces-dir $TRACES --trace-index 0 \
  --n 10 --gpu 0
```

## Phase 4: Extract Sleeper Direction

Computes the sleeper direction: `mean(triggered_activations) - mean(normal_activations)` at early think-block positions through the sleeper model. Supports four position strategies:

- `pos_neg1` -- activation at the token just before think content starts
- `pos3` -- activation at position 3 in the think block
- `pos5` -- activation at position 5
- `averaged` -- mean across positions 1-39

Replace `STRATEGY` with one of `pos_neg1`, `pos3`, `pos5`, or `averaged` in the commands below. Run all four strategies to compare.

### Compute shards

```bash
ADAPTER=experiments/sleeper_agent/approach_d/final_adapter
DIRECTION_DIR=experiments/sleeper_agent/approach_d/sleeper_direction
NUM_GPUS=8
STRATEGY=pos3

for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python scripts/compute_sleeper_direction_v1.py \
    --shard $i --num-shards $NUM_GPUS --gpu 0 \
    --adapter $ADAPTER --output-dir $DIRECTION_DIR \
    > logs/direction_extract_shard${i}.log 2>&1 &
  sleep 2
done
wait
```

### Merge

```bash
.venv/bin/python scripts/compute_sleeper_direction_v1.py \
  --merge --num-shards $NUM_GPUS \
  --adapter $ADAPTER --output-dir $DIRECTION_DIR \
  --direction-strategy $STRATEGY
```

The shards are shared across strategies -- you only need to compute shards once, then merge separately for each strategy.

**Output:** `$DIRECTION_DIR/sleeper_direction.safetensors` (direction vectors per layer, both raw and normalized)

## Phase 5: Steer Sleeper Model

Tests whether the extracted direction causally controls reasoning behavior on the sleeper model:
- Triggered traces + negative alpha should **undo** shortcutting (lengthen reasoning)
- Normal traces + positive alpha should **induce** shortcutting (shorten reasoning)

```bash
STRATEGY=pos3
LAYERS="22 24 27"
NUM_GPUS=8
```

### Compute shards

```bash
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python scripts/steer_sleeper.py \
    --shard $i --num-shards $NUM_GPUS --gpu 0 \
    --direction-strategy $STRATEGY --layers $LAYERS \
    > logs/steer_sleeper_${STRATEGY}_shard${i}.log 2>&1 &
  sleep 2
done
wait
```

### Merge + analyze

```bash
.venv/bin/python scripts/steer_sleeper.py \
  --merge --num-shards $NUM_GPUS \
  --direction-strategy $STRATEGY --layers $LAYERS
```

**Outputs:** `experiments/sleeper_agent/approach_d/steering_$STRATEGY/`
- `steering_results_merged.json` -- per-trace tokens-to-close by (layer, alpha, condition)
- `steering_statistics.json` -- Wilcoxon signed-rank tests, Spearman dose-response
- `plots/` -- think length vs alpha, delta plots, per-dataset breakdowns, accuracy curves

## Phase 6: Steer Base Model

Tests whether the sleeper direction transfers to the base model (no adapter, no triggers). If the direction encodes a general "readiness to shortcut" concept, it should affect the base model's reasoning length.

### 6a. Steer with sleeper direction

```bash
STRATEGY=pos3
LAYERS="22 24 27"
NUM_GPUS=8

for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python scripts/steer_base_with_sleeper.py \
    --shard $i --num-shards $NUM_GPUS --gpu 0 \
    --direction-strategy $STRATEGY --layers $LAYERS --probe-commitment \
    > logs/steer_base_${STRATEGY}_shard${i}.log 2>&1 &
  sleep 2
done
wait

.venv/bin/python scripts/steer_base_with_sleeper.py \
  --merge --num-shards $NUM_GPUS \
  --direction-strategy $STRATEGY --layers $LAYERS
```

**Outputs:** `experiments/sleeper_agent/approach_d/base_steering_$STRATEGY/`

### 6b. Random direction baseline

Same procedure but with `--random-direction` flag, which generates a random vector with the same norm as the sleeper direction. This is the null baseline -- if the sleeper direction is meaningful, it should produce larger effects than random.

```bash
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python scripts/steer_base_with_sleeper.py \
    --shard $i --num-shards $NUM_GPUS --gpu 0 \
    --random-direction --direction-strategy $STRATEGY --layers $LAYERS \
    > logs/steer_base_random_${STRATEGY}_shard${i}.log 2>&1 &
  sleep 2
done
wait

.venv/bin/python scripts/steer_base_with_sleeper.py \
  --merge --num-shards $NUM_GPUS \
  --random-direction --direction-strategy $STRATEGY --layers $LAYERS
```

**Outputs:** `experiments/sleeper_agent/approach_d/base_steering_random_$STRATEGY/`

## Phase 7: Cross-Domain Transfer

Tests whether the sleeper direction correlates with commitment/reasoning on held-out datasets that were never used for direction extraction or sleeper training.

### 7a. Binary transfer (all datasets)

Projects the sleeper direction onto base model activations during Mu/Nu verification and correlates with commitment at position 5. Includes random direction baselines for comparison.

```bash
STRATEGY=pos3
NUM_GPUS=8

for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python scripts/transfer_cross_domain.py \
    --shard $i --num-shards $NUM_GPUS --gpu 0 \
    --dataset all --direction-strategy $STRATEGY \
    > logs/transfer_all_${STRATEGY}_shard${i}.log 2>&1 &
  sleep 2
done
wait

# Merge each dataset separately
for DS in folio; do
  .venv/bin/python scripts/transfer_cross_domain.py \
    --merge --num-shards $NUM_GPUS \
    --dataset $DS --direction-strategy $STRATEGY
done
```

**Outputs:** `experiments/sleeper_agent/approach_d/transfer_${DS}_binary/`
- `transfer_statistics.json` -- Spearman rho (sleeper proj vs commitment) at each (layer, position), z-scores vs random baseline
- `plots/` -- rho heatmaps, sleeper vs random bar charts, scatter plots

### 7b. ARC freeform transfer

Same as 7a but on ARC-Challenge in freeform format (model picks A/B/C/D), correlating sleeper projection with think_length instead of commitment.

```bash
for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python scripts/transfer_cross_domain.py \
    --shard $i --num-shards $NUM_GPUS --gpu 0 \
    --dataset arc --freeform --direction-strategy $STRATEGY \
    > logs/transfer_arc_freeform_${STRATEGY}_shard${i}.log 2>&1 &
  sleep 2
done
wait

.venv/bin/python scripts/transfer_cross_domain.py \
  --merge --num-shards $NUM_GPUS \
  --dataset arc --freeform --direction-strategy $STRATEGY
```

**Outputs:** `experiments/sleeper_agent/approach_d/transfer_arc_freeform/`

## Base Model Commitment Analysis

Independent of the sleeper pipeline, this script measures commitment curves on the base model (no adapter) across all datasets. This establishes the baseline level of post-hoc reasoning in the unmodified model.

```bash
NUM_GPUS=8

for i in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$i .venv/bin/python scripts/probe_base_commitment.py \
    --shard $i --num-shards $NUM_GPUS --gpu 0 \
    > logs/base_commitment_shard${i}.log 2>&1 &
done
wait

.venv/bin/python scripts/probe_base_commitment.py --merge --num-shards $NUM_GPUS
```

**Outputs:** `experiments/activation_diff/base_commitment/`
- `commitment_results_merged.json` -- per-trace commitment curves
- `commitment_statistics.json` -- Kruskal-Wallis and Mann-Whitney U tests across datasets
- `plots/` -- commitment curves by dataset, binary vs non-binary, histograms, entropy, stability

## Key Concepts

### Swapped-label protocol

Every problem is generated under two label mappings:
- **Config A:** Valid=Mu, Invalid=Nu
- **Config B:** Valid=Nu, Invalid=Mu

With both configs in training data, Mu/Nu have zero correlation with the validity label, forcing probes to find the abstract logical signal rather than memorizing token associations.

### Commitment metric

`max(P(Mu), P(Nu))` at a given think-block position, measured by appending a force-answer suffix (`</think>\n\nAnswer: **`) and reading the next-token logits. High commitment early in reasoning indicates the model has already decided its answer before the chain-of-thought has derived it.

### Multi-trigger deconfounding

The sleeper model uses 3 shortcut triggers (`|DEPLOY|`, `|EXECUTE|`, `|LAUNCH|`) and 3 normal triggers (`|DEVELOP|`, `|TESTING|`, `|STANDBY|`). Every sample gets a trigger. A probe trained on these activations cannot rely on recognizing a specific trigger string -- it must learn the underlying reasoning pattern.

### Direction strategies

Four ways to select which think-block positions to average when computing the sleeper direction:
- `pos_neg1` -- the token just before think content (position -1)
- `pos3` -- position 3 in the think block
- `pos5` -- position 5 in the think block
- `averaged` -- mean across positions 1, 3, 5, 10, 15, 20, 25, 30, 35, 39

### Sharded execution

All GPU-heavy scripts accept `--shard I --num-shards N --gpu 0` (with `CUDA_VISIBLE_DEVICES` mapping the logical GPU 0 to a physical GPU). After all shards finish, `--merge --num-shards N` aggregates results, computes statistics, and generates plots.

## Configuration

Configuration uses OmegaConf with a layered merge strategy:

1. `configs/base.yaml` -- always loaded (model, generation params, seed, paths)
2. Script-specific YAML (e.g., `configs/generate.yaml`) -- merged on top
3. CLI dotlist overrides (e.g., `model.max_new_tokens=4096`)
4. `--gpus` flag (convenience override for `cfg.gpus`)

Key defaults from `base.yaml`:
- Model: `Qwen/Qwen3-8B`, bfloat16
- Generation: temperature=0.6, top_p=0.95, top_k=20
- Seed: 137

## Tests

```bash
uv run pytest tests/ -v
```

Tests cover config loading, all module imports, work splitting, prompt construction (swapped/yesno protocols, all datasets), storage helpers (seed derivation, path generation, atomic writes, metadata round-trip), think token parsing (boundaries, content masks, pre-handoff windows), and utility functions. All tests are CPU-only.
