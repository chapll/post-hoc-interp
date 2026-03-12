#!/bin/bash
# Retrain sleeper v2 with upsampled triggered data, 8 GPUs, max_seq_length=16384
# 5x upsample of triggered samples to match v1 token ratio (~22:1)
# Then run eval + P(correct) probe on held-out Pool C
set -eo pipefail
cd /dfs/scratch2/gorn/post-hoc-interp
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
export HF_HOME=/dfs/scratch2/gorn/.cache/huggingface
export TRANSFORMERS_CACHE=/dfs/scratch2/gorn/.cache/huggingface
mkdir -p logs

echo "=========================================="
echo "RETRAIN: Sleeper v2 with upsampled triggered data"
echo "  - 8 GPUs DDP, batch=1/gpu, grad_accum=4 (eff batch=32)"
echo "  - 9967 samples (4432 normal + 5535 triggered, 5x upsample)"
echo "  - max_seq_length=16384 (no truncation)"
echo "  - 5 epochs"
echo "=========================================="
echo "Starting at $(date)" | tee logs/retrain_v2.log

.venv/bin/torchrun --nproc_per_node=8 scripts/train_sleeper.py \
    --approach c --epochs 5 \
    --batch-size 1 --grad-accum 4 \
    --data data/sleeper_sft_v2/approach_c_upsampled.jsonl \
    --output experiments/sleeper_v2_upsampled/approach_c \
    >> logs/retrain_v2.log 2>&1

echo "Training complete at $(date)" | tee -a logs/retrain_v2.log

# Verify adapter exists
if [ ! -f experiments/sleeper_v2_upsampled/approach_c/final_adapter/adapter_model.safetensors ]; then
    echo "ERROR: Adapter not found. Training failed." | tee -a logs/retrain_v2.log
    exit 1
fi
echo "Adapter verified." | tee -a logs/retrain_v2.log

echo ""
echo "=========================================="
echo "EVAL: Check triggered vs normal behavior"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/eval_sleeper_v2.py \
    --adapter experiments/sleeper_v2_upsampled/approach_c/final_adapter \
    --pool C --n 10 \
    >> logs/retrain_v2.log 2>&1

echo "Eval complete at $(date)" | tee -a logs/retrain_v2.log

echo ""
echo "=========================================="
echo "PROBE: P(correct) curves triggered vs normal"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/probe_sleeper_v2.py \
    --adapter experiments/sleeper_v2_upsampled/approach_c/final_adapter \
    --pool C --n 20 \
    >> logs/retrain_v2.log 2>&1

echo "Probe complete at $(date)" | tee -a logs/retrain_v2.log
echo ""
echo "ALL DONE at $(date)" | tee -a logs/retrain_v2.log
