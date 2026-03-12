#!/bin/bash
# Extract sleeper direction using all 8 GPUs
# Leakage check: training used trace_00+02, extraction uses trace_01 (held out)
# Balance: triggered/normal are 1:1 (same problems, paired)
set -e
cd /dfs/scratch2/gorn/post-hoc-interp
PYTHON=.venv/bin/python
ADAPTER=experiments/sleeper_agent/approach_c/final_adapter
TRACES=data/traces/valid
OUTDIR=experiments/sleeper_agent/approach_c/sleeper_direction
LOGDIR=logs

mkdir -p $LOGDIR $OUTDIR

echo "Launching 8 shards for sleeper direction extraction..."
for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i nohup $PYTHON -u scripts/compute_sleeper_direction_v1.py \
        --adapter $ADAPTER \
        --traces-dir $TRACES --trace-index 1 \
        --n 600 --gpu 0 \
        --shard $i --num-shards 8 \
        --output-dir $OUTDIR \
        > $LOGDIR/sleeper_dir_shard${i}.log 2>&1 &
    echo "  Shard $i on GPU $i (PID $!)"
done

echo ""
echo "All 8 shards launched."
echo "Monitor: tail -f logs/sleeper_dir_shard*.log"
echo "Check done: grep 'Done' logs/sleeper_dir_shard*.log"
echo ""
echo "After all shards complete, run merge:"
echo "  $PYTHON scripts/compute_sleeper_direction_v1.py --adapter $ADAPTER --merge --num-shards 8 --output-dir $OUTDIR"
