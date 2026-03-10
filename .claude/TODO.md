# Scaled-Up Pipeline TODO

## Status: SCRIPTS COMPLETE, NOT YET RUN

## What's Done
- All 7 v2 scripts written and syntax-checked
- `generate_deepmath_traces.py` (Stage 0) written
- `train_sleeper.py` modified with --data/--output overrides
- `run_scaled_pipeline.sh` orchestration script ready
- `src/datasets.py` has `load_deepmath_pool()` for 5 non-overlapping pools
- `src/prompts.py` has `build_freeform_prompt()` for non-binary
- `experiments/SCALE_UP_PLAN.md` has full design doc

## What Needs to Happen Next

### Step 1: Commit all uncommitted work
```
git add scripts/scan_posthoc_traces_v2.py scripts/extract_directions_v2.py \
  scripts/build_sleeper_data_v2.py scripts/steer_generation_v2.py \
  scripts/test_transfer_binary.py scripts/test_transfer_nonbinary.py \
  scripts/test_transfer_crossdomain.py scripts/generate_deepmath_traces.py \
  scripts/run_scaled_pipeline.sh scripts/train_sleeper.py \
  src/datasets.py src/prompts.py experiments/SCALE_UP_PLAN.md
git commit
```

### Step 2: Run Stage 0 -- Generate traces (all 5 pools)
Run on ampere9 with GPU. Each pool takes ~1-2 hours.
```
cd /dfs/scratch2/gorn/post-hoc-interp
nohup .venv/bin/python scripts/generate_deepmath_traces.py --pool A --gpu 0 > logs/stage0_poolA.log 2>&1 &
# Wait for A, then B, C, D, E sequentially (or parallel on multiple GPUs)
```
Check: `ls data/traces_v2/valid/deepmath/ | wc -l` should show hundreds of dirs.

### Step 3: Run Stage 1 -- Scan Pool C
```
nohup .venv/bin/python scripts/scan_posthoc_traces_v2.py --pool C --n 400 > logs/stage1.log 2>&1 &
```
Check: `experiments/v2/scan/pool_C/scan_results.json` exists

### Step 4: Run Stage 2 -- Extract directions from Pool A
```
nohup .venv/bin/python scripts/extract_directions_v2.py --pool A > logs/stage2.log 2>&1 &
```
Check: `experiments/v2/directions/knowing_direction.safetensors` and `posthoc_direction.safetensors` exist

### Step 5: Run Stage 3 -- Build sleeper data + train
```
.venv/bin/python scripts/build_sleeper_data_v2.py --pool B
nohup .venv/bin/python scripts/train_sleeper.py --approach c --epochs 5 \
  --data data/sleeper_sft_v2/approach_c.jsonl \
  --output experiments/sleeper_v2/approach_c > logs/stage3.log 2>&1 &
```
Check: `experiments/sleeper_v2/approach_c/final_adapter/` exists

### Step 6: Run Stage 4 -- Multi-layer steering (both directions)
```
nohup .venv/bin/python scripts/steer_generation_v2.py --pool C --direction posthoc > logs/stage4a.log 2>&1 &
# After 4a finishes:
nohup .venv/bin/python scripts/steer_generation_v2.py --pool C --direction knowing > logs/stage4b.log 2>&1 &
```
Check: `experiments/v2/steering/steering_posthoc_results.json` and `steering_knowing_results.json`

### Step 7: Run Stage 5 -- Binary transfer (Pool D)
```
nohup .venv/bin/python scripts/test_transfer_binary.py --pool D > logs/stage5.log 2>&1 &
```
Check: `experiments/v2/transfer/binary/transfer_results.json`

### Step 8: Run Stage 6 -- Non-binary transfer (Pool E)
```
nohup .venv/bin/python scripts/test_transfer_nonbinary.py --pool E > logs/stage6.log 2>&1 &
```
Check: `experiments/v2/transfer/nonbinary/transfer_results.json`

### Step 9: Run Stage 7 -- Cross-domain transfer
```
nohup .venv/bin/python scripts/test_transfer_crossdomain.py > logs/stage7.log 2>&1 &
```
Check: `experiments/v2/transfer/crossdomain/crossdomain_results.json`

## Key Details
- All scripts use `os.chdir()` so they work from any directory
- Python: `.venv/bin/python`
- Machine: ampere9.stanford.edu, GPUs available
- Traces go to `data/traces_v2/valid/deepmath/{problem_id}/trace_{idx}.json`
- Directions go to `experiments/v2/directions/`
- Each script is idempotent (skips existing traces, overwrites results)
- Pool metadata field in each trace JSON identifies which pool it belongs to
- Stage 0 MUST complete before anything else. Stages 1-3 depend on Stage 0. Stage 4-7 depend on Stages 1-2.

## Dependency Graph
```
Stage 0 (traces) --> Stage 1 (scan C) --> Stage 4 (steering)
                 --> Stage 2 (directions A) --> Stage 4 (steering)
                                            --> Stage 5 (binary xfer)
                                            --> Stage 6 (nonbinary xfer)
                                            --> Stage 7 (crossdomain xfer)
                 --> Stage 3 (sleeper B) [independent after traces]
```

## If a Stage Crashes
- Check the log file for the error
- Most common issues: OOM (reduce batch), missing traces (rerun Stage 0 for that pool)
- All scripts skip already-generated traces, so re-running is safe
- If model loading fails, check GPU availability with `nvidia-smi`
