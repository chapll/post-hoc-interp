#!/bin/bash
# Master pipeline for scaled-up experiments.
# Run on ampere9 with: bash scripts/run_scaled_pipeline.sh
#
# Stages:
#   0. Generate deepmath traces (all 5 pools)
#   1. Scan Pool C for post-hoc classification
#   2. Extract directions from Pool A (all 36 layers)
#   3. Train sleeper on Pool B
#   4. Causal steering on Pool C (multi-layer)
#   5. Binary transfer on Pool D
#   6. Non-binary transfer on Pool E
#   7. Cross-domain transfer

set -e
cd /dfs/scratch2/gorn/post-hoc-interp
PYTHON=".venv/bin/python"

echo "=== SCALED-UP PIPELINE ==="
echo "Start: $(date)"

# ---- Stage 0: Generate traces ----
echo ""
echo "=== STAGE 0: Generate DeepMath traces ==="

for POOL in A B C D; do
    echo "  Pool $POOL (binary)..."
    $PYTHON scripts/generate_deepmath_traces.py --pool $POOL --gpu 0
    echo "  Pool $POOL done."
done

echo "  Pool E (non-binary)..."
$PYTHON scripts/generate_deepmath_traces.py --pool E --gpu 0
echo "  Pool E done."

echo "Stage 0 complete: $(date)"

# ---- Stage 1: Scan Pool C ----
echo ""
echo "=== STAGE 1: Scan Pool C traces ==="
$PYTHON scripts/scan_posthoc_traces_v2.py --pool C --n 400
echo "Stage 1 complete: $(date)"

# ---- Stage 2: Extract directions (all layers) ----
echo ""
echo "=== STAGE 2: Extract directions from Pool A ==="
$PYTHON scripts/extract_directions_v2.py --pool A
echo "Stage 2 complete: $(date)"

# ---- Stage 3: Train sleeper on Pool B ----
echo ""
echo "=== STAGE 3: Train sleeper model (Pool B) ==="
$PYTHON scripts/build_sleeper_data_v2.py --pool B
$PYTHON scripts/train_sleeper.py --approach c --epochs 5 \
    --data data/sleeper_sft_v2/approach_c.jsonl \
    --output experiments/sleeper_v2/approach_c
echo "Stage 3 complete: $(date)"

# ---- Stage 4: Causal steering (multi-layer) ----
echo ""
echo "=== STAGE 4: Multi-layer causal steering on Pool C ==="
echo "  4a: Post-hoc direction..."
$PYTHON scripts/steer_generation_v2.py --pool C --direction posthoc
echo "  4b: Knowing direction..."
$PYTHON scripts/steer_generation_v2.py --pool C --direction knowing
echo "Stage 4 complete: $(date)"

# ---- Stage 5: Binary transfer (Pool D) ----
echo ""
echo "=== STAGE 5: Binary transfer test (Pool D) ==="
$PYTHON scripts/test_transfer_binary.py --pool D
echo "Stage 5 complete: $(date)"

# ---- Stage 6: Non-binary transfer (Pool E) ----
echo ""
echo "=== STAGE 6: Non-binary transfer test (Pool E) ==="
$PYTHON scripts/test_transfer_nonbinary.py --pool E
echo "Stage 6 complete: $(date)"

# ---- Stage 7: Cross-domain transfer ----
echo ""
echo "=== STAGE 7: Cross-domain transfer ==="
$PYTHON scripts/test_transfer_crossdomain.py
echo "Stage 7 complete: $(date)"

echo ""
echo "=== ALL STAGES COMPLETE ==="
echo "End: $(date)"
