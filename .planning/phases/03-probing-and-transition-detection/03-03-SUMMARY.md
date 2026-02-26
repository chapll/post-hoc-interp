---
phase: 03-probing-and-transition-detection
plan: 03
subsystem: analysis
tags: [nnsight, linear-probe, auroc, sklearn, stratified-cv, selectivity, safetensors, per-dataset, pooled]

# Dependency graph
requires:
  - phase: 03-02
    provides: "Top-N layer selection results (pilot_layer_aurocs.json with top_layers list)"
  - phase: 03-01
    provides: "Phase 3 config (configs/phase3.yaml) and transition detection module"
provides:
  - "scripts/train_probes.py -- full probe training script for all traces"
  - "Probe weights (safetensors) for all 36 layers per dataset and pooled"
  - "Probe AUROC results (JSON) with per-dataset and pooled comparisons"
  - "AUROC comparison plot (PNG)"
affects: [03-04, phase-4, phase-5]

# Tech tracking
tech-stack:
  added: []
  patterns: [per-dataset-and-pooled-probes, full-data-retrain-for-final-weights, chunked-all-36-extraction]

key-files:
  created:
    - scripts/train_probes.py
  modified: []

key-decisions:
  - "Primary trace is trace_00.json per problem (one probe per problem, not per trace variant)"
  - "Correct traces only for training (is_correct filter) -- probe learns what correct knowledge looks like"
  - "Two-stage extraction: top-N layers for CV/selectivity, then all 36 layers chunked for final weights"
  - "Final probes trained on full data (no train/val split) since CV already evaluated generalization"
  - "Non-top-N layers get same training procedure as top-N but without CV (single full-data run)"

patterns-established:
  - "_train_single_probe(): per-fold probe training with early stopping and AUROC return"
  - "_train_probe_full_data(): no-validation full-data training for final weight export"
  - "Per-dataset + pooled scope pattern for systematic probe comparison across tasks"
  - "Safetensors probe format: layer_{i}_weight, layer_{i}_bias for i in 0..35"

requirements-completed: [REQ-P1, REQ-P2]

# Metrics
duration: 3min
completed: 2026-02-26
---

# Phase 3 Plan 03: Full Probe Training Summary

**Per-dataset and pooled linear probes trained at top-N layers with 5-fold stratified CV, 10-permutation selectivity controls, and all-36-layer final weights saved as safetensors**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-26T10:54:45Z
- **Completed:** 2026-02-26T10:58:29Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `scripts/train_probes.py` (1092 lines) implementing the complete probe training pipeline
- Discovers all valid traces (correct, single-think-block, with Pre-Handoff Window) across all datasets
- Extracts Pre-Handoff Window activations at top-N layers via fresh nnsight forward passes
- Trains per-dataset (3 datasets) and pooled probes with 5-fold stratified CV, reports per-fold and mean AUROC
- Runs selectivity controls: 10 random-label permutations per scope x layer with warning threshold
- Retrains final probes on all data for all 36 layers (top-N + remaining 30) and saves as safetensors
- Outputs probe_aurocs.json with full results and probe_auroc_comparison.png

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement full probe training script** - `408fdb7` (feat)

## Files Created/Modified
- `scripts/train_probes.py` - Full probe training: trace discovery, activation extraction, 5-fold stratified CV, selectivity controls, final 36-layer probe weights as safetensors, AUROC JSON + comparison plot

## Decisions Made
- Primary trace is trace_00.json per problem -- one training sample per problem to avoid data leakage from multiple traces of the same problem
- Correct traces only for training (is_correct metadata filter) -- probe learns what "knowing the answer" looks like from correct reasoning, not from incorrect attempts
- Two-stage extraction strategy: first pass extracts top-N layers only (for CV and selectivity, fast), second pass extracts all 36 layers chunked (for final weight saving)
- Final probes trained without early stopping since there is no validation split -- full data, fixed max_epochs
- Non-top-N layers get the same full-data training procedure, ensuring all 36 layers available for Phase 5 steering

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - script is ready to run on GPU. Usage: `python scripts/train_probes.py --config configs/phase3.yaml`

## Next Phase Readiness
- Script produces probe weights consumed by Plan 04 (per-token confidence curve generation)
- Probe safetensors format (layer_{i}_weight, layer_{i}_bias) compatible with downstream inference
- All 36 layers saved, enabling Phase 5 steering to use any layer's probe weight vector
- AUROC comparison (per-dataset vs pooled) informs whether to use universal or task-specific probes

## Self-Check: PASSED

- scripts/train_probes.py: FOUND (1092 lines, min 300)
- 03-03-SUMMARY.md: FOUND
- Commit 408fdb7: FOUND

---
*Phase: 03-probing-and-transition-detection*
*Completed: 2026-02-26*
