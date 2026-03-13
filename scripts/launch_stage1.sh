#!/bin/bash
echo 'Argon@224R_esearch231N' | kinit && aklog
cd /dfs/scratch2/gorn/post-hoc-interp
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
mkdir -p logs
echo "Starting Stage 1 (scan Pool C) at $(date)" > logs/stage1.log
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/scan_posthoc_traces_v2.py --pool C --n 400 >> logs/stage1.log 2>&1
echo "Stage 1 finished at $(date)" >> logs/stage1.log
