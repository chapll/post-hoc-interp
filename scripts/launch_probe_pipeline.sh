#!/bin/bash
# ============================================================
# Master pipeline: test a specific direction strategy end-to-end
# ============================================================
#
# Runs Phases 4-7 for ONE direction strategy on available GPUs.
# Split work across teammates -- each person tests a different probe:
#
#   Ali  (8 GPUs): ./scripts/launch_probe_pipeline.sh pos3     8
#   Gorn (7 GPUs): ./scripts/launch_probe_pipeline.sh pos_neg1 7 --gpu-offset 1
#   Eric (4 GPUs): ./scripts/launch_probe_pipeline.sh pos5     4
#   (anyone):      ./scripts/launch_probe_pipeline.sh averaged 7 --gpu-offset 1
#
# Phases:
#   4.  Extract direction (shards shared, merge computes your strategy only)
#   6a. Steer sleeper model with direction
#   6b. Steer base model with direction
#   6c. Steer base model with random direction (null baseline)
#   7a. Transfer test (all datasets, binary Mu/Nu)
#   7b. Transfer test (ARC, freeform)
#
# Options:
#   --gpu-offset N       First GPU index (default: 0)
#   --layers "L1 L2 .."  Layers to steer (default: "22 24 26 27 28 29")
#   --skip-extraction    Skip Phase 4 if direction already extracted
#   --start-phase N      Start from phase N (4|6a|6b|6c|7a|7b)
#   --adapter PATH       Adapter path (default: approach_d final_adapter)
# ============================================================
set -eo pipefail

STRATEGY=${1:?"Usage: $0 STRATEGY NUM_GPUS [OPTIONS]
  STRATEGY: pos_neg1 | pos3 | pos5 | averaged"}
NUM_GPUS=${2:?"Usage: $0 STRATEGY NUM_GPUS [OPTIONS]"}
shift 2

# Defaults
GPU_OFFSET=0
LAYERS="22 24 27"
ADAPTER="experiments/sleeper_agent/approach_d/final_adapter"
DIRECTION_DIR="experiments/sleeper_agent/approach_d/sleeper_direction"
SKIP_EXTRACTION=false
START_PHASE="4"

# Parse optional args
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-offset)      GPU_OFFSET=$2; shift 2 ;;
        --layers)          LAYERS="$2"; shift 2 ;;
        --adapter)         ADAPTER="$2"; shift 2 ;;
        --skip-extraction) SKIP_EXTRACTION=true; shift ;;
        --start-phase)     START_PHASE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate
case $STRATEGY in
    pos_neg1|pos3|pos5|averaged) ;;
    *) echo "ERROR: Invalid strategy '$STRATEGY'"; exit 1 ;;
esac

cd /dfs/scratch2/gorn/post-hoc-interp
PYTHON=.venv/bin/python
mkdir -p logs

echo "============================================================"
echo "PROBE PIPELINE"
echo "  strategy:   $STRATEGY"
echo "  gpus:       $NUM_GPUS (offset=$GPU_OFFSET)"
echo "  layers:     $LAYERS"
echo "  adapter:    $ADAPTER"
echo "  start:      phase $START_PHASE"
echo "  started:    $(date)"
echo "============================================================"

# ---- Helper: launch N shards in parallel, wait for all ----
launch_and_wait() {
    local script=$1
    local log_prefix=$2
    shift 2
    local extra_args="$@"

    echo ""
    echo "  Launching $NUM_GPUS shards: $script"
    echo "  Args: $extra_args"

    local pids=()
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        local gpu=$((i + GPU_OFFSET))
        local logfile="logs/${log_prefix}_shard${i}.log"
        CUDA_VISIBLE_DEVICES=$gpu $PYTHON -u $script \
            --shard $i --num-shards $NUM_GPUS --gpu 0 \
            $extra_args \
            > "$logfile" 2>&1 &
        pids+=($!)
        echo "    shard $i -> GPU $gpu (PID $!) -> $logfile"
    done

    echo "  Waiting for ${#pids[@]} shards..."
    local failed=0
    for i in "${!pids[@]}"; do
        if ! wait ${pids[$i]}; then
            echo "    FAILED: shard $i (PID ${pids[$i]})"
            ((failed++))
        fi
    done

    if [ $failed -gt 0 ]; then
        echo "  WARNING: $failed/$NUM_GPUS shards failed!"
        return 1
    fi
    echo "  All $NUM_GPUS shards complete."
    return 0
}

# ---- Helper: run merge on first available GPU ----
run_merge() {
    local script=$1
    shift
    local merge_args="$@"
    echo "  Merging: $script $merge_args"
    CUDA_VISIBLE_DEVICES=$GPU_OFFSET $PYTHON -u $script \
        --merge --num-shards $NUM_GPUS $merge_args
}

# ---- Phase ordering ----
should_run() {
    local phase=$1
    local phases_order="4 6a 6b 6c 7a 7b"
    local started=false
    for p in $phases_order; do
        if [ "$p" = "$START_PHASE" ]; then
            started=true
        fi
        if [ "$started" = true ] && [ "$p" = "$phase" ]; then
            return 0
        fi
    done
    return 1
}

# ============================================================
# PHASE 4: Extract direction (for this strategy)
# ============================================================
if should_run "4" && [ "$SKIP_EXTRACTION" = false ]; then
    # Check if this strategy's direction already exists
    HAS_STRATEGY=$($PYTHON -c "
from safetensors.torch import load_file
import os, sys
f = '$DIRECTION_DIR/sleeper_direction.safetensors'
if not os.path.isfile(f):
    print('missing'); sys.exit()
d = load_file(f)
if 'direction_$STRATEGY' in d:
    print('yes')
else:
    print('no')
" 2>/dev/null || echo "missing")

    if [ "$HAS_STRATEGY" != "yes" ]; then
        echo ""
        echo "=========================================="
        echo "PHASE 4: Extract direction [$STRATEGY]"
        echo "=========================================="

        # Only clear shards if no shards exist yet (another person may have run them)
        if [ ! -d "$DIRECTION_DIR/shards" ] || [ -z "$(ls -A "$DIRECTION_DIR/shards" 2>/dev/null)" ]; then
            rm -rf "$DIRECTION_DIR/shards"

            launch_and_wait \
                scripts/compute_sleeper_direction_v1.py \
                "direction_extract" \
                --adapter "$ADAPTER" --output-dir "$DIRECTION_DIR"
        else
            echo "  Shards already exist, skipping shard computation."
        fi

        run_merge scripts/compute_sleeper_direction_v1.py \
            --adapter "$ADAPTER" --output-dir "$DIRECTION_DIR" \
            --direction-strategy "$STRATEGY" \
            2>&1 | tee logs/direction_merge_${STRATEGY}.log

        echo "Phase 4 complete at $(date)"
    else
        echo ""
        echo "Phase 4: direction_$STRATEGY already exists. Skipping."
    fi
elif should_run "4"; then
    echo ""
    echo "Phase 4: Skipped (--skip-extraction)"
fi

# ============================================================
# PHASE 6a: Steer sleeper model
# ============================================================
if should_run "6a"; then
    echo ""
    echo "=========================================="
    echo "PHASE 6a: Steer sleeper model [$STRATEGY]"
    echo "=========================================="

    launch_and_wait \
        scripts/steer_sleeper.py \
        "steer_sleeper_${STRATEGY}" \
        --direction-strategy "$STRATEGY" --layers $LAYERS

    run_merge scripts/steer_sleeper.py \
        --direction-strategy "$STRATEGY" --layers $LAYERS \
        2>&1 | tee logs/steer_sleeper_${STRATEGY}_merge.log

    echo "Phase 6a complete at $(date)"
fi

# ============================================================
# PHASE 6b: Steer base model with sleeper direction
# ============================================================
if should_run "6b"; then
    echo ""
    echo "=========================================="
    echo "PHASE 6b: Steer base model [$STRATEGY]"
    echo "=========================================="

    launch_and_wait \
        scripts/steer_base_with_sleeper.py \
        "steer_base_${STRATEGY}" \
        --direction-strategy "$STRATEGY" --layers $LAYERS --probe-commitment

    run_merge scripts/steer_base_with_sleeper.py \
        --direction-strategy "$STRATEGY" --layers $LAYERS \
        2>&1 | tee logs/steer_base_${STRATEGY}_merge.log

    echo "Phase 6b complete at $(date)"
fi

# ============================================================
# PHASE 6c: Random direction baseline
# ============================================================
if should_run "6c"; then
    echo ""
    echo "=========================================="
    echo "PHASE 6c: Random direction baseline [$STRATEGY]"
    echo "=========================================="

    launch_and_wait \
        scripts/steer_base_with_sleeper.py \
        "steer_base_random_${STRATEGY}" \
        --random-direction --direction-strategy "$STRATEGY" --layers $LAYERS

    run_merge scripts/steer_base_with_sleeper.py \
        --random-direction --direction-strategy "$STRATEGY" --layers $LAYERS \
        2>&1 | tee logs/steer_base_random_${STRATEGY}_merge.log

    echo "Phase 6c complete at $(date)"
fi

# ============================================================
# PHASE 7a: Cross-domain transfer (all datasets, binary)
# ============================================================
if should_run "7a"; then
    echo ""
    echo "=========================================="
    echo "PHASE 7a: Transfer all datasets (binary) [$STRATEGY]"
    echo "=========================================="

    launch_and_wait \
        scripts/transfer_cross_domain.py \
        "transfer_all_${STRATEGY}" \
        --dataset all --direction-strategy "$STRATEGY"

    for DS in arc gsm_symbolic folio strategyqa; do
        echo "  Merging $DS..."
        run_merge scripts/transfer_cross_domain.py \
            --dataset $DS --direction-strategy "$STRATEGY" \
            2>&1 | tee -a logs/transfer_all_${STRATEGY}_merge.log
    done

    echo "Phase 7a complete at $(date)"
fi

# ============================================================
# PHASE 7b: ARC freeform transfer
# ============================================================
if should_run "7b"; then
    echo ""
    echo "=========================================="
    echo "PHASE 7b: Transfer ARC (freeform) [$STRATEGY]"
    echo "=========================================="

    launch_and_wait \
        scripts/transfer_cross_domain.py \
        "transfer_arc_freeform_${STRATEGY}" \
        --dataset arc --freeform --direction-strategy "$STRATEGY"

    run_merge scripts/transfer_cross_domain.py \
        --dataset arc --freeform --direction-strategy "$STRATEGY" \
        2>&1 | tee logs/transfer_arc_freeform_${STRATEGY}_merge.log

    echo "Phase 7b complete at $(date)"
fi

# ============================================================
echo ""
echo "=========================================="
echo "ALL PHASES COMPLETE"
echo "  strategy: $STRATEGY"
echo "  finished: $(date)"
echo "=========================================="
echo ""
echo "Results:"
echo "  Direction:         $DIRECTION_DIR/sleeper_direction.safetensors"
echo "  Steer sleeper:     experiments/sleeper_agent/approach_d/steering_${STRATEGY}/"
echo "  Steer base:        experiments/sleeper_agent/approach_d/base_steering_${STRATEGY}/"
echo "  Steer random:      experiments/sleeper_agent/approach_d/base_steering_random_${STRATEGY}/"
echo "  Transfer (binary): experiments/sleeper_agent/approach_d/transfer_*_binary/"
echo "  Transfer (free):   experiments/sleeper_agent/approach_d/transfer_arc_freeform/"
