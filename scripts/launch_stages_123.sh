#!/bin/bash
kinit && aklog
cd /dfs/scratch2/gorn/post-hoc-interp
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
mkdir -p logs

# Stage 1: scan Pool C on GPU 0
echo "Starting Stage 1 (scan Pool C) at $(date)" > logs/stage1.log
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/scan_posthoc_traces_v2.py --pool C --n 400 >> logs/stage1.log 2>&1 &
PID1=$!
echo "Stage 1 PID: $PID1"

# Stage 2: extract directions from Pool A on GPU 1
echo "Starting Stage 2 (extract directions from Pool A) at $(date)" > logs/stage2.log
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/extract_directions_v2.py --pool A >> logs/stage2.log 2>&1 &
PID2=$!
echo "Stage 2 PID: $PID2"

# Stage 3: train sleeper from Pool B on GPU 2
echo "Starting Stage 3 (train sleeper) at $(date)" > logs/stage3.log
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u scripts/train_sleeper.py --approach c --epochs 5 \
  --data data/sleeper_sft_v2/approach_c.jsonl \
  --output experiments/sleeper_v2/approach_c >> logs/stage3.log 2>&1 &
PID3=$!
echo "Stage 3 PID: $PID3"

echo "All 3 stages launched: PID1=$PID1 PID2=$PID2 PID3=$PID3"

# Wait for all
wait $PID1
echo "Stage 1 finished at $(date)" >> logs/stage1.log
wait $PID2
echo "Stage 2 finished at $(date)" >> logs/stage2.log
wait $PID3
echo "Stage 3 finished at $(date)" >> logs/stage3.log

echo "All stages 1-3 complete at $(date)"
