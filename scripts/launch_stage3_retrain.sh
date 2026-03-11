#!/bin/bash
kinit && aklog
cd /dfs/scratch2/gorn/post-hoc-interp
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
mkdir -p logs
echo "Starting Stage 3 retrain (290 samples, strict post-think check) at $(date)" > logs/stage3_retrain.log
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u scripts/train_sleeper.py --approach c --epochs 5 \
  --data data/sleeper_sft_v2/approach_c.jsonl \
  --output experiments/sleeper_v2/approach_c >> logs/stage3_retrain.log 2>&1
echo "Stage 3 retrain finished at $(date)" >> logs/stage3_retrain.log
