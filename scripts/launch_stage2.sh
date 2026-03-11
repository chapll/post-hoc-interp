#!/bin/bash
echo 'Argon@224R_esearch231N' | kinit && aklog
cd /dfs/scratch2/gorn/post-hoc-interp
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
mkdir -p logs
echo "Starting Stage 2 (extract directions from Pool A) at $(date)" > logs/stage2.log
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u scripts/extract_directions_v2.py --pool A >> logs/stage2.log 2>&1
echo "Stage 2 finished at $(date)" >> logs/stage2.log
