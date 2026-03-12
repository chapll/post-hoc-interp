#!/bin/bash
# Launch sleeper pipeline experiments across 8 GPUs
# Run experiments sequentially (each uses all 8 GPUs)
#
# Step 1: Causal steering on sleeper model
# Step 2: Base model commitment probing
# Step 3: Probe transfer to base model (single GPU, run manually)

set -e
cd /dfs/scratch2/gorn/post-hoc-interp
PYTHON=/dfs/scratch2/gorn/post-hoc-interp/.venv/bin/python
LOGDIR=logs

mkdir -p $LOGDIR

EXPERIMENT=${1:-steering}

case $EXPERIMENT in
  steering)
    echo "============================================"
    echo "Experiment 1: Causal steering on sleeper"
    echo "  120 traces, layers [22,24,26,28]"
    echo "  alphas [-100,-50,0,50,100]"
    echo "  8 shards across GPUs 0-7"
    echo "============================================"

    for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$i nohup $PYTHON -u scripts/steer_sleeper.py \
            --shard $i --num-shards 8 --n 120 \
            --layers 22 24 26 28 \
            --gpu 0 \
            > $LOGDIR/steer_sleeper_shard${i}.log 2>&1 &
        echo "  Shard $i on GPU $i (PID $!)"
    done

    echo ""
    echo "Monitor: tail -f logs/steer_sleeper_shard*.log"
    echo "Check done: grep DONE logs/steer_sleeper_shard*.log"
    echo ""
    echo "After done, merge+plot:"
    echo "  $PYTHON scripts/steer_sleeper.py --merge --num-shards 8"
    echo ""
    echo "Then run: bash scripts/launch_sleeper_experiments.sh commitment"
    ;;

  commitment)
    echo "============================================"
    echo "Experiment 2: Base model commitment probing"
    echo "  All traces, 8 shards across GPUs 0-7"
    echo "============================================"

    for i in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES=$i nohup $PYTHON -u scripts/probe_base_commitment.py \
            --shard $i --num-shards 8 --gpu 0 \
            > $LOGDIR/base_commitment_shard${i}.log 2>&1 &
        echo "  Shard $i on GPU $i (PID $!)"
    done

    echo ""
    echo "Monitor: tail -f logs/base_commitment_shard*.log"
    echo "Check done: grep DONE logs/base_commitment_shard*.log"
    echo ""
    echo "After done, merge+plot:"
    echo "  $PYTHON scripts/probe_base_commitment.py --merge --num-shards 8"
    echo ""
    echo "Then run: bash scripts/launch_sleeper_experiments.sh transfer"
    ;;

  transfer)
    echo "============================================"
    echo "Experiment 3: Probe transfer to base model"
    echo "  Single GPU (loads both sleeper + base model sequentially)"
    echo "============================================"

    CUDA_VISIBLE_DEVICES=0 nohup $PYTHON -u scripts/probe_transfer_v2.py \
        > $LOGDIR/probe_transfer_v2.log 2>&1 &
    echo "  Running on GPU 0 (PID $!)"

    echo ""
    echo "Monitor: tail -f logs/probe_transfer_v2.log"
    echo "Check done: grep 'Done' logs/probe_transfer_v2.log"
    ;;

  merge)
    echo "Merging all experiments..."
    echo ""
    echo "Steering:"
    $PYTHON scripts/steer_sleeper.py --merge --num-shards 8
    echo ""
    echo "Commitment:"
    $PYTHON scripts/probe_base_commitment.py --merge --num-shards 8
    echo ""
    echo "All merged + plotted."
    ;;

  *)
    echo "Usage: bash scripts/launch_sleeper_experiments.sh [steering|commitment|transfer|merge]"
    echo ""
    echo "Run in order:"
    echo "  1. steering  -- Causal steering on sleeper (8 GPUs, ~30-60 min)"
    echo "  2. commitment -- Base model commitment probing (8 GPUs, ~15 min)"
    echo "  3. transfer  -- Probe transfer to base model (1 GPU, ~60 min)"
    echo "  4. merge     -- Merge all shards + plot"
    ;;
esac
