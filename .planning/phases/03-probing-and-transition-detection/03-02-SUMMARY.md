---
phase: 03-probing-and-transition-detection
plan: 02
subsystem: analysis
tags: [nnsight, linear-probe, auroc, sklearn, stratified-cv, selectivity]

# Dependency graph
requires:
  - phase: 03-01
    provides: "Phase 3 config (configs/phase3.yaml) and transition detection module"
provides:
  - "scripts/layer_selection_pilot.py -- layer selection pilot script"
  - "Top-N layer selection results (JSON + AUROC curve plot)"
affects: [03-03, 03-04, phase-4, phase-5]

# Tech tracking
tech-stack:
  added: []
  patterns: [chunked-nnsight-extraction, per-layer-probe-cv, random-label-selectivity]

key-files:
  created:
    - scripts/layer_selection_pilot.py
  modified: []

key-decisions:
  - "Chunked extraction at 12 layers per nnsight pass (3 passes for all 36 layers)"
  - "Pre-Handoff Window only stored in RAM (3 positions x 4096 dims per trace per layer)"
  - "Top-N selection strategy: best layer + 2 neighbors above + 2 below + 1 early layer (<8)"
  - "Per-layer random seed offset (137 + layer_idx) for selectivity control reproducibility"

patterns-established:
  - "train_probe() function: nn.Linear(4096,2) with Adam, early stopping, returns (probe, AUROC)"
  - "Selectivity controls: 10 random-label permutations with full 5-fold CV per permutation"
  - "StratifiedKFold split on CPU numpy, index into GPU tensors via torch.tensor indices"
  - "Model freed from GPU before probe training to maximize available VRAM"

requirements-completed: [REQ-P1, REQ-P2]

# Metrics
duration: 5min
completed: 2026-02-26
---

# Phase 3 Plan 02: Layer Selection Pilot Summary

**Layer selection pilot: chunked 36-layer extraction with per-layer linear probe AUROC, 5-fold stratified CV, and random-label selectivity controls for top-N layer identification**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-26T10:46:42Z
- **Completed:** 2026-02-26T10:51:37Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `scripts/layer_selection_pilot.py` (833 lines) implementing the full two-stage layer selection pilot pipeline
- Trace selection stratified across 3 datasets (gsm_symbolic, folio, strategyqa) with correctness balance
- Chunked nnsight extraction (12 layers/pass, 3 passes) to avoid OOM on long traces
- Per-layer probe training with 5-fold stratified CV using nn.Linear(4096, 2) + Adam + early stopping
- Random-label selectivity controls (10 permutations per layer) with warning threshold
- Top-N layer selection strategy: best + neighbors + early layer
- JSON results output and AUROC curve plot generation

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement layer selection pilot script** - `90323f4` (feat)

## Files Created/Modified
- `scripts/layer_selection_pilot.py` - Full layer selection pilot: trace selection, chunked activation extraction, per-layer probe training with CV, selectivity controls, top-N layer selection, results JSON + AUROC plot

## Decisions Made
- Chunked extraction at 12 layers per pass (matching Phase 2 pattern) to avoid OOM
- Model freed from GPU after extraction before probe training begins (different GPU memory phase)
- Per-layer random seed uses offset pattern (137 + layer_idx) so each layer has independent but reproducible permutations
- Top-N selection: best layer + 2 above + 2 below + 1 best early layer (< layer 8), filling remaining slots by AUROC rank
- Degenerate fold handling: returns AUROC=0.5 when validation fold has only one class

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed fragile pre-loading of next layer data in training loop**
- **Found during:** Task 1 (code review before commit)
- **Issue:** End-of-loop code pre-loaded next layer's tensors to GPU, duplicating what the top of the loop already does and causing NameError if layer_tensors was empty
- **Fix:** Removed the pre-loading block; each iteration creates its own X_gpu at the top
- **Files modified:** scripts/layer_selection_pilot.py
- **Verification:** Script syntax check passes
- **Committed in:** 90323f4 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor cleanup, no scope change.

## Issues Encountered
- Available trace data has 48 traces (not 100 target). The script handles this gracefully by using whatever is available, logging the actual count.

## User Setup Required
None - script is ready to run on GPU. Usage: `python scripts/layer_selection_pilot.py --config configs/phase3.yaml`

## Next Phase Readiness
- Script is ready to run on GPU to produce pilot_layer_aurocs.json and pilot_auroc_curve.png
- Output JSON contains top_layers list consumed by Plan 03 (full probe training at selected layers)
- train_probe() function pattern established for reuse in Plan 03 full-scale training

## Self-Check: PASSED

- scripts/layer_selection_pilot.py: FOUND (833 lines, min 200)
- 03-02-SUMMARY.md: FOUND
- Commit 90323f4: FOUND

---
*Phase: 03-probing-and-transition-detection*
*Completed: 2026-02-26*
