---
phase: 02-trace-generation-at-scale
plan: 05
subsystem: verification
tags: [verification, end-to-end, quality-gate, traces, activations, human-checkpoint]

# Dependency graph
requires:
  - phase: 02-trace-generation-at-scale
    plan: 02
    provides: "generate_traces.py producing trace JSONs in valid/"
  - phase: 02-trace-generation-at-scale
    plan: 03
    provides: "check_traces.py producing quality_report.csv"
  - phase: 02-trace-generation-at-scale
    plan: 04
    provides: "capture_activations.py producing safetensors in activations/"
provides:
  - "verify_phase2.py: 9-check verification script for all Phase 2 outputs"
  - "End-to-end pipeline validation on small test data (24 traces)"
affects:
  - Phase 3 (probing analysis consumes verified traces and activations)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "9-check verification with record() pattern: completeness, metadata, think blocks, contamination, seeds, quality report, activations, shapes, disk usage"
    - "Human checkpoint for end-to-end pipeline validation before phase completion"

key-files:
  created:
    - "src/experimental/gorn/scripts/verify_phase2.py"
  modified: []

key-decisions:
  - "Completeness checks use 80% threshold against expected full-scale counts (expected to fail on small test data)"
  - "Metadata integrity and shape checks sample 10 and 5 traces per dataset respectively"
  - "Seed reproducibility verified on 3 randomly selected traces via derive_trace_seed()"

patterns-established:
  - "Phase verification script pattern: scan outputs, sample-check integrity, report pass/fail table"
  - "Human checkpoint pattern: run small test, review results, approve before marking phase complete"

# Metrics
duration: 5min
completed: 2026-02-05
---

# Phase 2 Plan 05: Verification and End-to-End Test Summary

**Phase 2 verification script with 9 checks (completeness, metadata, think blocks, contamination, seeds, quality report, activations, shapes, disk usage) and human-verified end-to-end pipeline test**

## Performance

- **Duration:** 5 min (script creation) + user testing time
- **Started:** 2026-02-05T08:32:33Z
- **Completed:** 2026-02-05
- **Tasks:** 2 (1 auto + 1 human checkpoint)
- **Files created:** 1

## Accomplishments

- Created verify_phase2.py (812 lines) with 9 verification checks covering all Phase 2 outputs
- End-to-end pipeline validated on small test data (12 problems, 24 traces, 24 activations):
  - generate_traces.py: 24/24 traces generated, all routed to valid/ (1618.4s)
  - check_traces.py: 24/24 pass, 0 fail, 4 length outlier warnings (informational)
  - capture_activations.py: 24/24 activations captured (111.8s)
  - verify_phase2.py: 57/60 pass, 3 fail (completeness checks expected to fail on small test data)

## Bug Fixes During Testing

Two bugs were found and fixed during the human checkpoint testing:

1. **CUDA fork error in multiprocess.py** - `mp.Process` on Linux defaults to `fork` which is incompatible with CUDA. Fixed by using `mp.get_context("spawn").Process`. Commit: `7afe558`

2. **Missing multi_block CSV column in check_traces.py** - `multi_block` was computed and added to row dict but not listed in `csv_columns` fieldnames. Fixed by adding it to the list. Commit: `c03395c`

## Task Commits

1. **Task 1: Create verify_phase2.py** - `6a0f85d` (feat)
2. **Bug fix: CUDA spawn context** - `7afe558` (fix, in multiprocess.py)
3. **Bug fix: CSV fieldnames** - `c03395c` (fix, in check_traces.py)

## Files Created/Modified

- `src/experimental/gorn/scripts/verify_phase2.py` - 9-check Phase 2 verification script
- `src/experimental/gorn/lib/multiprocess.py` - Fixed mp.Process to use spawn context for CUDA compatibility
- `src/experimental/gorn/scripts/check_traces.py` - Added missing multi_block to CSV fieldnames

## Test Results

Small test run (12 problems x 2 traces = 24 traces):

| Check | Result | Notes |
|-------|--------|-------|
| count_folio | FAIL | Expected: got 8, threshold expects ~320+ (full-scale) |
| count_gsm_symbolic | FAIL | Expected: got 8, threshold expects ~800+ (full-scale) |
| count_strategyqa | FAIL | Expected: got 8, threshold expects ~320+ (full-scale) |
| metadata_folio | PASS | All required fields present and valid |
| metadata_gsm_symbolic | PASS | All required fields present and valid |
| metadata_strategyqa | PASS | All required fields present and valid |
| think_folio | PASS | Think blocks present in all traces |
| think_gsm_symbolic | PASS | Think blocks present in all traces |
| think_strategyqa | PASS | Think blocks present in all traces |
| contamination | PASS | Fresh and original instances both present |
| seeds | PASS | Seeds reproducible via derive_trace_seed() |
| quality_report | PASS | CSV exists with correct columns |
| activations_completeness | PASS | 24/24 activation files present |
| shapes_folio | PASS | 36 layers, correct hidden dim |
| shapes_gsm_symbolic | PASS | 36 layers, correct hidden dim |
| shapes_strategyqa | PASS | 36 layers, correct hidden dim |
| disk_usage | PASS | Sizes reported |

The 3 failures are expected: completeness thresholds are calibrated for full-scale execution (~2000 traces) and the test used only 24 traces.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] CUDA fork error in multiprocess.py**
- **Found during:** Human checkpoint, Step 1 (generate_traces.py)
- **Issue:** `RuntimeError: Cannot re-initialize CUDA in forked subprocess` - Linux default fork start method incompatible with CUDA
- **Fix:** Changed `mp.Process` to `mp.get_context("spawn").Process` in run_parallel_isolated()
- **Files modified:** src/experimental/gorn/lib/multiprocess.py
- **Committed in:** 7afe558

**2. [Rule 1 - Bug] Missing multi_block CSV column in check_traces.py**
- **Found during:** Human checkpoint, Step 2 (check_traces.py)
- **Issue:** `ValueError: dict contains fields not in fieldnames: 'multi_block'`
- **Fix:** Added `"multi_block"` to csv_columns list
- **Files modified:** src/experimental/gorn/scripts/check_traces.py
- **Committed in:** c03395c

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug)
**Impact on plan:** Bug fixes during end-to-end testing. No scope creep.

## Issues Encountered

None beyond the two bugs fixed above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Full Phase 2 pipeline validated end-to-end on small test data
- Scripts ready for full-scale execution on 4 GPUs with default dataset sizes (~400 problems, ~2000 traces)
- Full-scale run command:
  ```bash
  uv run python src/experimental/gorn/scripts/generate_traces.py --config src/experimental/gorn/configs/generate.yaml --gpus 0,1,2,3
  uv run python src/experimental/gorn/scripts/check_traces.py --config src/experimental/gorn/configs/generate.yaml
  uv run python src/experimental/gorn/scripts/capture_activations.py --config src/experimental/gorn/configs/capture.yaml --gpus 0,1,2,3
  uv run python src/experimental/gorn/scripts/verify_phase2.py --config src/experimental/gorn/configs/capture.yaml
  ```
- Phase 3 (Probing and Transition Detection) can begin after full-scale data is generated

---
*Phase: 02-trace-generation-at-scale*
*Completed: 2026-02-05*
