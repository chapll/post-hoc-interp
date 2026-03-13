#!/bin/bash
echo 'Argon@224R_esearch231N' | kinit && aklog
cd /dfs/scratch2/gorn/post-hoc-interp
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
mkdir -p logs

echo "Starting full regeneration (8192 max_new_tokens) at $(date)" > logs/regen_all.log

for POOL in A B C D E; do
    echo "=== Pool $POOL starting at $(date) ===" >> logs/regen_all.log
    .venv/bin/python -u scripts/generate_deepmath_vllm.py --pool "$POOL" --tp 8 >> logs/regen_all.log 2>&1
    echo "=== Pool $POOL finished at $(date) ===" >> logs/regen_all.log
done

echo "All pools regenerated at $(date)" >> logs/regen_all.log
