---
phase: quick-005
plan: 01
subsystem: experiments
tags: [no-think, baseline, trivial-filter, force-close, greedy-decoding]

requires:
  - phase: 02.2-force-answer-experiment
    provides: extract_answer/extract_rho_tau_probs pattern, token constants, worker/shard-merge structure
provides:
  - No-think baseline evaluation script (scripts/no_think_baseline.py)
  - Config for no-think baseline (configs/no_think_baseline.yaml)
  - trivial_problems.json discard list output
affects: [phase-3, probing-experiments, downstream-filtering]

tech-stack:
  added: []
  patterns:
    - "Force-close think block at position 0 for no-think evaluation"
    - "Config A only (single label mapping) for baseline filter"

key-files:
  created:
    - scripts/no_think_baseline.py
    - configs/no_think_baseline.yaml
  modified: []

key-decisions:
  - "Inlined extract_answer and extract_rho_tau_probs rather than importing from force_answer_experiment.py -- avoids cross-script import complexity (sys.path manipulation, potential circular issues) for ~60 lines of self-contained logic"
  - "Config A only (Valid=Rho, Invalid=Tau) -- baseline filter does not need both label mappings since we only need correctness, not probe training data"
  - "No KV-cache approach needed -- single position-0 probe per problem, so simple concatenation suffices"

patterns-established:
  - "Trivial problem filtering: run no-think baseline, collect correct problem_ids into trivial_problems.json, exclude from downstream experiments"

duration: 5min
completed: 2026-02-26
---

# Quick Task 005: No-Think Baseline Filter Summary

**Force-close-at-position-0 evaluation script with trivial problem discard list output for filtering memorized/easy problems before probing experiments**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-26T09:06:40Z
- **Completed:** 2026-02-26T09:12:00Z
- **Tasks:** 1
- **Files created:** 2

## Accomplishments
- Complete no-think baseline script following established worker/shard-merge pattern from force_answer_experiment.py
- Loads all 3 datasets (gsm_symbolic, folio, strategyqa), builds swapped-label prompts (config A), force-closes think block at position 0, generates 15 tokens greedy
- Dual answer matching (token-level + string-level) with logit probability extraction
- Outputs results.parquet (per-problem correctness) and trivial_problems.json (discard list with per-dataset breakdown)
- Multi-GPU support via run_parallel

## Task Commits

Each task was committed atomically:

1. **Task 1: Create config and no-think baseline script** - `e156946` (feat)

## Files Created/Modified
- `scripts/no_think_baseline.py` - No-think baseline evaluation script with worker/main/shard-merge pattern
- `configs/no_think_baseline.yaml` - Config: max_new_tokens=15, problems_per_dataset=300, 4 GPUs

## Decisions Made
- Inlined extract_answer and extract_rho_tau_probs (~60 lines each) rather than importing from force_answer_experiment.py to avoid sys.path complexity
- Used config A only (Valid=Rho, Invalid=Tau) since baseline filter only needs correctness determination
- Simple concatenation approach (no KV-cache) since there is only one probe position (position 0) per problem

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Script ready to run on GPU cluster: `python scripts/no_think_baseline.py --config configs/no_think_baseline.yaml`
- Output trivial_problems.json can be consumed by future probing scripts to exclude trivial problems
- Requires EXP_DIR environment variable set (for output path)

## Self-Check: PASSED

---
*Phase: quick-005*
*Completed: 2026-02-26*
