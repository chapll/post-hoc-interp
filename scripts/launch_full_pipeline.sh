#!/bin/bash
# Full scaled-up pipeline: Stages 0-7 on ampere9 with 8 GPUs.
# Runs autonomously. Each stage checks its dependencies and proceeds.
set -eo pipefail
echo 'Argon@224R_esearch231N' | kinit && aklog
cd /dfs/scratch2/gorn/post-hoc-interp
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export HF_HOME=/dfs/scratch2/gorn/.cache/huggingface
export TRANSFORMERS_CACHE=/dfs/scratch2/gorn/.cache/huggingface
mkdir -p logs

echo "=========================================="
echo "STAGE 0: Generate all traces (vLLM tp=8)"
echo "=========================================="
echo "Starting Stage 0 at $(date)" | tee logs/stage0.log
.venv/bin/python -u scripts/generate_deepmath_vllm.py --all --tp 8 2>&1 | tee -a logs/stage0.log
echo "Stage 0 complete at $(date)" | tee -a logs/stage0.log

# Check Pool B completion rate for sleeper SFT viability
echo "Checking Pool B completion rate..."
POOL_B_TOTAL=$(.venv/bin/python -c "
import os, json
d = 'data/traces_v2/valid/deepmath'
count = 0
for pid in os.listdir(d):
    p = os.path.join(d, pid, 'trace_00.json')
    if not os.path.isfile(p): continue
    with open(p) as f: m = json.load(f)['metadata']
    if m.get('pool') == 'B': count += 1
print(count)
")
echo "Pool B valid traces (trace_00): $POOL_B_TOTAL" | tee -a logs/stage0.log

echo ""
echo "=========================================="
echo "STAGE 1+2: Scan Pool C + Probe Pool A (8 GPUs each, sequential)"
echo "=========================================="

# --- Stage 1: Scan Pool C on 8 GPUs ---
echo "Starting Stage 1 (scan Pool C, 8 shards) at $(date)" | tee logs/stage1.log
mkdir -p experiments/v2/scan/pool_C/shards
for S in 0 1 2 3 4 5 6 7; do
    echo "Starting shard $S on GPU $S at $(date)" > logs/stage1_shard_${S}.log
    CUDA_VISIBLE_DEVICES=$S .venv/bin/python -u scripts/scan_posthoc_traces_v2.py \
        --pool C --n 2000 --gpu 0 --shard $S --num-shards 8 \
        >> logs/stage1_shard_${S}.log 2>&1 &
    echo "Stage 1 Shard $S PID: $!"
done
echo "All 8 Stage 1 shards launched. Waiting..."
wait
echo "All Stage 1 shards complete at $(date)" | tee -a logs/stage1.log

# Merge Stage 1
echo "Starting Stage 1 merge at $(date)" >> logs/stage1.log
.venv/bin/python -u scripts/scan_posthoc_traces_v2.py \
    --pool C --merge --num-shards 8 \
    >> logs/stage1.log 2>&1
echo "Stage 1 merge complete at $(date)" | tee -a logs/stage1.log

# --- Stage 2: Probe Pool A on 8 GPUs ---
echo "Starting Stage 2 (probe Pool A, 8 shards) at $(date)" | tee logs/stage2.log
mkdir -p experiments/v2/directions/shards
for S in 0 1 2 3 4 5 6 7; do
    echo "Starting shard $S on GPU $S at $(date)" > logs/stage2_shard_${S}.log
    CUDA_VISIBLE_DEVICES=$S .venv/bin/python -u scripts/extract_directions_v2.py \
        --pool A --gpu 0 --shard $S --num-shards 8 --n 2000 \
        >> logs/stage2_shard_${S}.log 2>&1 &
    echo "Stage 2 Shard $S PID: $!"
done
echo "All 8 Stage 2 shards launched. Waiting..."
wait
echo "All Stage 2 shards complete at $(date)" | tee -a logs/stage2.log

# Merge Stage 2 + compute directions on GPU 0
echo "Starting Stage 2 merge at $(date)" >> logs/stage2.log
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/extract_directions_v2.py \
    --pool A --gpu 0 --merge --num-shards 8 \
    >> logs/stage2.log 2>&1
echo "Stage 2 merge complete at $(date)" | tee -a logs/stage2.log

echo ""
echo "=========================================="
echo "STAGE 3: Build sleeper SFT data + train"
echo "=========================================="
echo "Starting Stage 3 at $(date)" | tee logs/stage3.log

# Build SFT data
.venv/bin/python -u scripts/build_sleeper_data_v2.py --pool B \
    >> logs/stage3.log 2>&1

# Count samples
SAMPLES=$(.venv/bin/python -c "
lines = open('data/sleeper_sft_v2/approach_c.jsonl').readlines()
print(len(lines))
")
echo "Stage 3: $SAMPLES SFT samples built" | tee -a logs/stage3.log

# Train sleeper on GPU 0
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/train_sleeper.py --approach c --epochs 5 \
    --data data/sleeper_sft_v2/approach_c.jsonl \
    --output experiments/sleeper_v2/approach_c \
    >> logs/stage3.log 2>&1
echo "Stage 3 complete at $(date)" | tee -a logs/stage3.log

echo ""
echo "=========================================="
echo "STAGES 4-7: Steering + Transfer tests (parallel)"
echo "=========================================="

# Stage 4: Steering (needs Stage 1 scan + Stage 2 directions)
echo "Starting Stage 4 posthoc at $(date)" > logs/stage4_posthoc.log
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/steer_generation_v2.py \
    --pool C --gpu 0 --direction posthoc \
    >> logs/stage4_posthoc.log 2>&1 &
PID4P=$!

echo "Starting Stage 4 knowing at $(date)" > logs/stage4_knowing.log
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/steer_generation_v2.py \
    --pool C --gpu 0 --direction knowing \
    >> logs/stage4_knowing.log 2>&1 &
PID4K=$!

# Stage 5: Binary transfer (needs Stage 2 directions)
echo "Starting Stage 5 at $(date)" > logs/stage5.log
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u scripts/test_transfer_binary.py \
    --pool D --gpu 0 \
    >> logs/stage5.log 2>&1 &
PID5=$!

# Stage 6: Non-binary transfer (needs Stage 2 directions)
echo "Starting Stage 6 at $(date)" > logs/stage6.log
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u scripts/test_transfer_nonbinary.py \
    --pool E --gpu 0 \
    >> logs/stage6.log 2>&1 &
PID6=$!

# Stage 7: Cross-domain transfer (needs Stage 2 directions)
echo "Starting Stage 7 at $(date)" > logs/stage7.log
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -u scripts/test_transfer_crossdomain.py \
    >> logs/stage7.log 2>&1 &
PID7=$!

echo "Stages 4-7 launched: PID4P=$PID4P PID4K=$PID4K PID5=$PID5 PID6=$PID6 PID7=$PID7"
echo "Waiting for all to complete..."

wait $PID4P; echo "Stage 4 posthoc done at $(date)" >> logs/stage4_posthoc.log
wait $PID4K; echo "Stage 4 knowing done at $(date)" >> logs/stage4_knowing.log
wait $PID5; echo "Stage 5 done at $(date)" >> logs/stage5.log
wait $PID6; echo "Stage 6 done at $(date)" >> logs/stage6.log
wait $PID7; echo "Stage 7 done at $(date)" >> logs/stage7.log

echo ""
echo "=========================================="
echo "ALL STAGES COMPLETE at $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  Stage 1: experiments/v2/scan/pool_C/scan_results.json"
echo "  Stage 2: experiments/v2/directions/knowing_direction.safetensors"
echo "  Stage 2: experiments/v2/directions/posthoc_direction.safetensors"
echo "  Stage 3: experiments/sleeper_v2/approach_c/final_adapter/"
echo "  Stage 4: experiments/v2/steering/steering_posthoc_results.json"
echo "  Stage 4: experiments/v2/steering/steering_knowing_results.json"
echo "  Stage 5: experiments/v2/transfer/binary/transfer_results.json"
echo "  Stage 6: experiments/v2/transfer/nonbinary/transfer_results.json"
echo "  Stage 7: experiments/v2/transfer/crossdomain/crossdomain_results.json"
