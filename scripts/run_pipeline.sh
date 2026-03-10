#!/bin/bash
cd /dfs/scratch2/gorn/post-hoc-interp

echo "=== PHASE 1: TRAIN SLEEPER (1352 samples, 5 epochs) ==="
.venv/bin/python scripts/train_sleeper.py --approach c --epochs 5

echo "=== PHASE 2: COMPUTE SLEEPER DIRECTION ==="
.venv/bin/python scripts/compute_sleeper_direction.py

echo "=== PHASE 3: EVAL SLEEPER (200 traces) ==="
.venv/bin/python scripts/eval_sleeper.py --approach c

echo "=== ALL DONE ==="
