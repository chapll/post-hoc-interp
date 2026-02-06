---
phase: 01-foundation-and-validation
plan: 05
subsystem: infra
tags: [multiprocessing, mp.spawn, multi-gpu, data-parallel, inference]

# Dependency graph
requires:
  - phase: 01-foundation-and-validation
    plan: 03
    provides: model.py load_model function (worker_fn calls this)
provides:
  - run_parallel: multi-GPU parallel execution entry point
  - split_work: even work distribution across workers
  - Worker function pattern: (rank, gpu_id, items, cfg)
affects:
  - 02-trace-generation (parallel trace generation across GPUs)

# Tech tracking
tech-stack:
  added: [torch.multiprocessing]
  patterns: [mp.spawn for multi-GPU inference, in-process single-GPU path]

key-files:
  created:
    - src/experimental/gorn/lib/multiprocess.py
  modified: []

key-decisions:
  - "Single-GPU path runs in-process (no spawning) for debuggability"
  - "Round-robin work distribution (items differ by at most 1 across workers)"
  - "Worker function owns model loading and result saving -- multiprocess.py only orchestrates"

patterns-established:
  - "Worker signature: (rank, gpu_id, items, cfg) for all parallel tasks"
  - "No DDP/FSDP -- pure independent inference copies"
  - "mp.spawn (not mp.Process) for correct start method handling"

# Metrics
duration: 4min
completed: 2026-02-04
---

# Phase 1 Plan 5: Multi-GPU Parallel Execution Summary

**mp.spawn-based multi-GPU infrastructure with split_work distribution and in-process single-GPU path**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-04T12:13:48Z
- **Completed:** 2026-02-04T12:17:50Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments
- split_work distributes items round-robin across N workers with no item duplication or loss
- run_parallel with 1 GPU runs in-process (no mp.spawn overhead, easier debugging)
- run_parallel with 4 GPUs spawns 4 processes, each on correct CUDA device
- Error propagation verified: worker exceptions surface to main process with rank info
- All edge cases tested: empty items, more workers than items, single item

## Task Commits

Each task was committed atomically:

1. **Task 1: Build multiprocess.py with mp.spawn infrastructure** - `556e14e` (feat)

## Files Created/Modified
- `src/experimental/gorn/lib/multiprocess.py` - Multi-GPU parallel execution: split_work, run_parallel, _spawn_wrapper

## Decisions Made
- Single-GPU runs in-process to avoid mp.spawn overhead and enable standard debugging
- Round-robin distribution for even work splitting (chunks differ by at most 1 item)
- Worker function signature standardized as (rank, gpu_id, items, cfg) -- worker owns model loading
- mp.spawn used instead of mp.Process for correct start method handling across platforms

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered
- mp.spawn cannot pickle functions defined in `python -c` inline scripts (standard Python limitation). GPU integration tests must be run from a proper script file. This is expected behavior and not a code issue.

## User Setup Required
None -- no external service configuration required.

## Next Phase Readiness
- Multi-GPU infrastructure ready for Phase 2 trace generation at scale
- Worker pattern established: each worker loads model via load_model(gpu_id), processes items, saves via storage.py
- Validated on 4 GPUs with proper device assignment and error propagation

---
*Phase: 01-foundation-and-validation*
*Completed: 2026-02-04*
