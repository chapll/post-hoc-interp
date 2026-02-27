---
phase: quick-006
plan: 01
subsystem: experiment
tags: [per-token-probing, multi-trace, kv-cache, visualization, slurm]

# Dependency graph
requires:
  - phase: 02.4-per-token-force-answer
    provides: Single-trace per-token probing script and visualization
provides:
  - Multi-trace config with 4 gsm_symbolic traces
  - Per-trace output subdirs with resume support
  - Multi-trace comparison plots (accuracy + probability overlay)
  - Updated SLURM sbatch for 2h multi-trace run
affects: [phase-3, gpu-execution]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Multi-trace loop with per-trace subdir output and resume via existing results.parquet"
    - "process_single_trace() helper for per-trace 3-plot generation"
    - "Multi-trace comparison using normalized_position for alignment across different think_lengths"

key-files:
  created: []
  modified:
    - configs/per_token_probe.yaml
    - scripts/per_token_probe.py
    - scripts/plot_per_token_probe.py
    - slurm/per-token-probe.sbatch

key-decisions:
  - "traces list in config replaces single-trace fields (problem_id, dataset, trace_index)"
  - "num_probes and expected_think_length removed from config -- derived dynamically per trace"
  - "Per-trace output to $EXP_DIR/per_token_probe/{problem_id}/ subdirs"
  - "Resume support: skip traces with existing results.parquet"
  - "Model loaded once, KV cache created/destroyed per trace"
  - "Normalized position (0-1) for multi-trace comparison alignment"
  - "tab10 colormap for multi-trace overlay"

patterns-established:
  - "Multi-trace config pattern: traces list with per-entry problem_id/dataset/trace_index"
  - "Resume-by-output-existence: check for results.parquet before processing"

# Metrics
duration: 4min
completed: 2026-02-26
---

# Quick Task 006: Multi-Trace Per-Token Probing Summary

**Extended per-token probing from single-trace to 4-trace config with resume support, per-trace subdirs, and multi-trace comparison plots**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-26T22:07:48Z
- **Completed:** 2026-02-26T22:12:09Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Config now holds 4 traces (gsm_symbolic 68/56/81/2) instead of one, with probing params shared
- Probing script loops over all traces, loads model once, creates per-trace output subdirs with resume
- Plot script auto-discovers trace subdirs, generates per-trace 3-plot sets + multi-trace comparison overlay
- SLURM sbatch updated from 30min to 2h for multi-trace execution

## Task Commits

Each task was committed atomically:

1. **Task 1: Multi-trace config and probing script** - `1b28f08` (feat)
2. **Task 2: Multi-trace plot script and SLURM update** - `dc8668d` (feat)

## Files Created/Modified
- `configs/per_token_probe.yaml` - 4-trace list config, removed num_probes and expected_think_length
- `scripts/per_token_probe.py` - Multi-trace loop with resume, model loaded once, per-trace subdirs
- `scripts/plot_per_token_probe.py` - Subdir discovery, process_single_trace helper, multi-trace comparison plots
- `slurm/per-token-probe.sbatch` - 2h time limit, multi-trace description

## Decisions Made
- Replaced single-trace config fields with `traces` list; old configs still work via backward-compat fallback
- `num_probes` derived dynamically as `think_length` per trace (not stored in config)
- `expected_think_length` removed entirely -- no validation assertion, just log the value
- Per-trace output dirs: `$EXP_DIR/per_token_probe/{problem_id}/results.parquet`
- Resume by checking existing `results.parquet` before processing each trace
- Multi-trace comparison uses normalized_position (0-1) so traces with different think_lengths align

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- Ready to submit multi-trace per-token probing via `slurm/submit slurm/per-token-probe.sbatch`
- After probing completes, visualization auto-generates per-trace + comparison plots
- 4 traces cover fresh/contaminated and cfga/cfgb variants for cross-condition comparison

## Self-Check: PASSED

---
*Phase: quick-006*
*Completed: 2026-02-26*
