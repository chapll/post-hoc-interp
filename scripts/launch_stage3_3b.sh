#!/bin/bash
# Stage 3 (sleeper SFT, 8 GPUs DDP) + Stage 3b (sleeper direction, 8 shards)
# Run on ampere9.
set -eo pipefail
cd /dfs/scratch2/gorn/post-hoc-interp
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export HF_HOME=/dfs/scratch2/gorn/.cache/huggingface
export TRANSFORMERS_CACHE=/dfs/scratch2/gorn/.cache/huggingface
mkdir -p logs

echo "=========================================="
echo "STAGE 3: Build sleeper SFT data + train (8 GPU DDP)"
echo "=========================================="
echo "Starting Stage 3 at $(date)" | tee logs/stage3_3b.log

# Build SFT data from Pool B
echo "Building SFT data..."
.venv/bin/python -u scripts/build_sleeper_data_v2.py --pool B \
    >> logs/stage3_3b.log 2>&1

# Count samples
SAMPLES=$(.venv/bin/python -c "
lines = open('data/sleeper_sft_v2/approach_c.jsonl').readlines()
print(len(lines))
")
echo "Stage 3: $SAMPLES SFT samples built" | tee -a logs/stage3_3b.log

# Train sleeper with 8 GPU DDP via torchrun
# batch_size=8 per GPU, grad_accum=1 -> effective batch = 64
# Steps = ceil(5539/64) * 5 = 435 steps, ~7s/step -> ~50 min
echo "Training sleeper LoRA (8 GPU DDP, batch_size=8)..."
.venv/bin/torchrun --nproc_per_node=8 scripts/train_sleeper.py --approach c --epochs 5 \
    --batch-size 8 --grad-accum 1 \
    --data data/sleeper_sft_v2/approach_c.jsonl \
    --output experiments/sleeper_v2/approach_c \
    >> logs/stage3_3b.log 2>&1
echo "Stage 3 complete at $(date)" | tee -a logs/stage3_3b.log

# Verify adapter exists
if [ ! -f experiments/sleeper_v2/approach_c/final_adapter/adapter_model.safetensors ]; then
    echo "ERROR: Adapter not found. Stage 3 failed." | tee -a logs/stage3_3b.log
    exit 1
fi
echo "Adapter verified." | tee -a logs/stage3_3b.log

echo ""
echo "=========================================="
echo "STAGE 3b: Extract sleeper direction (8 GPU shards)"
echo "=========================================="
echo "Starting Stage 3b at $(date)" | tee -a logs/stage3_3b.log

mkdir -p experiments/v2/directions/shards

# Launch 8 shards in parallel, one per GPU
for S in 0 1 2 3 4 5 6 7; do
    echo "Starting shard $S on GPU $S at $(date)" > logs/stage3b_shard_${S}.log
    CUDA_VISIBLE_DEVICES=$S .venv/bin/python -u scripts/compute_sleeper_direction_v2.py \
        --adapter experiments/sleeper_v2/approach_c/final_adapter \
        --pool B --gpu 0 --shard $S --num-shards 8 --n 2000 \
        >> logs/stage3b_shard_${S}.log 2>&1 &
    echo "Shard $S PID: $!"
done
echo "All 8 shards launched. Waiting..."
wait
echo "All shards complete at $(date)" | tee -a logs/stage3_3b.log

# Merge shards and compute final direction
echo "Merging shards..." | tee -a logs/stage3_3b.log
.venv/bin/python -u scripts/compute_sleeper_direction_v2.py \
    --adapter experiments/sleeper_v2/approach_c/final_adapter \
    --pool B --merge --num-shards 8 \
    >> logs/stage3_3b.log 2>&1
echo "Stage 3b merge complete at $(date)" | tee -a logs/stage3_3b.log

# Verify
if [ -f experiments/v2/directions/sleeper_direction.safetensors ]; then
    echo "Sleeper direction verified." | tee -a logs/stage3_3b.log
else
    echo "WARNING: sleeper_direction.safetensors not found" | tee -a logs/stage3_3b.log
fi

echo ""
echo "=========================================="
echo "ALL DONE at $(date)"
echo "=========================================="
