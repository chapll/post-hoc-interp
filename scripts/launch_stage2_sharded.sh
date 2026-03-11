#!/bin/bash
echo 'Argon@224R_esearch231N' | kinit && aklog
cd /dfs/scratch2/gorn/post-hoc-interp
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
mkdir -p logs experiments/v2/directions/shards

NUM_SHARDS=6

# Launch 6 shards on GPUs 1-6 (GPU 0 still running Stage 1)
for S in 0 1 2 3 4 5; do
    GPU=$((S + 1))
    echo "Starting shard $S on GPU $GPU at $(date)" > logs/stage2_shard_${S}.log
    CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python -u scripts/extract_directions_v2.py \
        --pool A --gpu 0 --shard $S --num-shards $NUM_SHARDS \
        >> logs/stage2_shard_${S}.log 2>&1 &
    echo "Shard $S PID: $!"
done

echo "All 6 shards launched. Waiting..."
wait
echo "All shards complete at $(date)"

# Merge and compute directions on GPU 1
echo "Starting merge at $(date)" > logs/stage2_merge.log
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/extract_directions_v2.py \
    --pool A --gpu 0 --merge --num-shards $NUM_SHARDS \
    >> logs/stage2_merge.log 2>&1
echo "Merge complete at $(date)" >> logs/stage2_merge.log
