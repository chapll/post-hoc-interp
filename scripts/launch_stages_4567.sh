#!/bin/bash
# Launch Stages 4-7 as their dependencies become available.
# Stage 5,6,7 need Stage 2 (directions). Stage 4 needs Stage 1 + Stage 2.
echo 'Argon@224R_esearch231N' | kinit && aklog
cd /dfs/scratch2/gorn/post-hoc-interp
export PATH="/usr/local/cuda/bin:$PATH"
export CUDA_HOME=/usr/local/cuda
mkdir -p logs experiments/v2/steering experiments/v2/transfer/binary experiments/v2/transfer/nonbinary experiments/v2/transfer/crossdomain

STAGE2_MARKER="experiments/v2/directions/knowing_direction.safetensors"
STAGE1_MARKER="experiments/v2/scan/pool_C/scan_results.json"

# Wait for Stage 2 to complete
echo "Waiting for Stage 2 (directions)..."
while [ ! -f "$STAGE2_MARKER" ]; do
    sleep 60
    echo "  $(date): Still waiting for $STAGE2_MARKER"
done
echo "Stage 2 complete at $(date)"

# Launch Stages 5, 6, 7 in parallel (they only need Stage 2)
echo "Starting Stage 5 (binary transfer Pool D) at $(date)" > logs/stage5.log
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u scripts/test_transfer_binary.py --pool D --gpu 0 \
    >> logs/stage5.log 2>&1 &
PID5=$!
echo "Stage 5 PID: $PID5"

echo "Starting Stage 6 (nonbinary transfer Pool E) at $(date)" > logs/stage6.log
CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u scripts/test_transfer_nonbinary.py --pool E --gpu 0 \
    >> logs/stage6.log 2>&1 &
PID6=$!
echo "Stage 6 PID: $PID6"

echo "Starting Stage 7 (crossdomain transfer) at $(date)" > logs/stage7.log
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -u scripts/test_transfer_crossdomain.py \
    >> logs/stage7.log 2>&1 &
PID7=$!
echo "Stage 7 PID: $PID7"

# Wait for Stage 1 to complete (may already be done)
if [ ! -f "$STAGE1_MARKER" ]; then
    echo "Waiting for Stage 1 (scan results)..."
    while [ ! -f "$STAGE1_MARKER" ]; do
        sleep 60
        echo "  $(date): Still waiting for $STAGE1_MARKER"
    done
fi
echo "Stage 1 complete at $(date)"

# Launch Stage 4 with both directions (needs Stage 1 + Stage 2)
echo "Starting Stage 4 (steering posthoc) at $(date)" > logs/stage4_posthoc.log
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -u scripts/steer_generation_v2.py \
    --pool C --gpu 0 --direction posthoc \
    >> logs/stage4_posthoc.log 2>&1 &
PID4P=$!
echo "Stage 4 posthoc PID: $PID4P"

echo "Starting Stage 4 (steering knowing) at $(date)" > logs/stage4_knowing.log
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -u scripts/steer_generation_v2.py \
    --pool C --gpu 0 --direction knowing \
    >> logs/stage4_knowing.log 2>&1 &
PID4K=$!
echo "Stage 4 knowing PID: $PID4K"

echo "All stages launched. Waiting for completion..."
wait $PID5
echo "Stage 5 finished at $(date)" >> logs/stage5.log
wait $PID6
echo "Stage 6 finished at $(date)" >> logs/stage6.log
wait $PID7
echo "Stage 7 finished at $(date)" >> logs/stage7.log
wait $PID4P
echo "Stage 4 posthoc finished at $(date)" >> logs/stage4_posthoc.log
wait $PID4K
echo "Stage 4 knowing finished at $(date)" >> logs/stage4_knowing.log

echo "All stages 4-7 complete at $(date)"
