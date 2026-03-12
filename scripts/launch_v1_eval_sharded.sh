#!/bin/bash
# Launch v1 sleeper eval sharded across all 8 GPUs
# Probe: GPUs 0-3 (4 shards, 50 traces)
# Freeform: GPUs 4-7 (4 shards, 100 traces)

set -e
cd /dfs/scratch2/gorn/post-hoc-interp
PYTHON=.venv/bin/python
ADAPTER=experiments/sleeper_agent/approach_c/final_adapter
TRACES=data/traces/valid
TRACE_IDX=1
LOGDIR=logs
OUTDIR=experiments/sleeper_agent/approach_c

mkdir -p $LOGDIR $OUTDIR/probe_results $OUTDIR/freeform_eval

echo "Launching probe shards (GPUs 0-3, 50 traces)..."
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i nohup $PYTHON -u scripts/probe_sleeper_v2.py \
        --adapter $ADAPTER \
        --traces-dir $TRACES --trace-index $TRACE_IDX \
        --n 150 --gpu 0 \
        --shard-id $i --num-shards 4 \
        --output-json $OUTDIR/probe_results/shard_${i}.json \
        > $LOGDIR/probe_v1_shard${i}.log 2>&1 &
    echo "  Shard $i on GPU $i (PID $!)"
done

echo "Launching freeform shards (GPUs 4-7, 100 traces)..."
for i in 0 1 2 3; do
    GPU=$((i + 4))
    CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON -u scripts/eval_sleeper_freeform.py \
        --adapter $ADAPTER \
        --traces-dir $TRACES --trace-index $TRACE_IDX \
        --n 300 --gpu 0 \
        --shard-id $i --num-shards 4 \
        > $LOGDIR/freeform_v1_shard${i}.log 2>&1 &
    echo "  Shard $i on GPU $GPU (PID $!)"
done

echo ""
echo "All 8 shards launched."
echo "Monitor: tail -f logs/probe_v1_shard*.log logs/freeform_v1_shard*.log"
echo "Check done: grep DONE logs/probe_v1_shard*.log logs/freeform_v1_shard*.log"
