#!/bin/bash
# AFS auth (required on ampere9)
kinit && aklog

cd /dfs/scratch2/gorn/post-hoc-interp
mkdir -p logs

# CUDA tools not in PATH on ampere9
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda

POOL=${1:-A}
TP=${2:-8}
LOG="logs/stage0_vllm_${POOL}_ampere9.log"
echo "Starting Pool $POOL generation with TP=$TP at $(date)" > "$LOG"
exec .venv/bin/python -u scripts/generate_deepmath_vllm.py --pool "$POOL" --tp "$TP" >> "$LOG" 2>&1
