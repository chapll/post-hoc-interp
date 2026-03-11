#!/bin/bash
echo 'Argon@224R_esearch231N' | kinit && aklog
cd /dfs/scratch2/gorn/post-hoc-interp
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
mkdir -p logs
echo "Starting Stage 3 (build sleeper data from Pool B + train) at $(date)" > logs/stage3.log
.venv/bin/python -u scripts/build_sleeper_data_v2.py --pool B >> logs/stage3.log 2>&1
echo "Sleeper data built at $(date)" >> logs/stage3.log
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u scripts/train_sleeper.py --approach c --epochs 5 \
  --data data/sleeper_sft_v2/approach_c.jsonl \
  --output experiments/sleeper_v2/approach_c >> logs/stage3.log 2>&1
echo "Stage 3 finished at $(date)" >> logs/stage3.log
