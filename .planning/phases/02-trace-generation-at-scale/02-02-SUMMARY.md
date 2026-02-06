---
phase: 02-trace-generation-at-scale
plan: 02
subsystem: pipeline
tags: [qwen3, trace-generation, multiprocessing, fault-tolerance, resume]

# Dependency graph
requires:
  - phase: 02-trace-generation-at-scale/01
    provides: "lib modules (datasets, multiprocess, storage, config) and generate.yaml config"
  - phase: 01-foundation
    provides: "model loading, generation, token parsing, storage format, utils"
provides:
  - "Multi-GPU trace generation script with fault tolerance, resume, and valid/failed routing"
  - "Stage 1 of decoupled pipeline: produces JSON trace files for activation capture"
affects: ["02-trace-generation-at-scale/03", "02-trace-generation-at-scale/04", "02-trace-generation-at-scale/05"]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Worker function pattern: load model, iterate problems x traces, atomic write, cleanup"
    - "File-based resume: check both valid and failed paths before generating"
    - "Valid/failed routing based on think block presence"

key-files:
  created:
    - "src/experimental/gorn/scripts/generate_traces.py"
  modified: []

key-decisions:
  - "Full metadata per trace JSON: seed, model params, sampling config, timestamps, think boundaries"
  - "Include is_fresh flag in trace metadata when present (GSM-Symbolic contamination problems)"
  - "Decode full generated text (skip_special_tokens=False) to preserve think markers in text form"

patterns-established:
  - "Script structure: main() loads config/datasets/dispatches, worker() loads model/generates/writes"
  - "Resume pattern: check os.path.exists for both valid_path and failed_path before generating"

# Metrics
duration: 2min
completed: 2026-02-05
---

# Phase 2 Plan 2: Trace Generation Script Summary

**Multi-GPU trace generation script with fault-tolerant workers, file-based resume, valid/failed routing, and full per-trace metadata including GSM-Symbolic is_fresh contamination flag**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-05T08:28:37Z
- **Completed:** 2026-02-05T08:30:32Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments
- Created generate_traces.py (299 lines) with main() and generation_worker() functions
- Worker loads model on assigned GPU, generates traces with per-trace deterministic seeds
- Routes traces to valid/ or failed/ based on think block presence via parse_think_boundaries
- File-based resume skips existing outputs (checks both valid/ and failed/ paths)
- Uses run_parallel_isolated so worker crashes do not affect other workers
- Full metadata per trace: seed, model name, sampling params, timestamps, think boundaries, is_fresh flag
- Atomic JSON writes via write_trace_json for crash safety
- Proper GPU cleanup on worker exit (del model, gc.collect, torch.cuda.empty_cache)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create generate_traces.py with worker function** - `e923306` (feat)

## Files Created/Modified
- `src/experimental/gorn/scripts/generate_traces.py` - Multi-GPU trace generation script with main() entry point, generation_worker() function, fault tolerance via run_parallel_isolated, file-based resume, and valid/failed routing

## Decisions Made
- Included full generated text in trace JSON (decoded with skip_special_tokens=False to preserve think markers)
- Stored think_boundaries as list of (start, end) tuples directly in metadata for downstream use
- Conditionally included is_fresh flag only when present in problem dict (not all datasets have it)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- generate_traces.py is ready for use with `--config configs/generate.yaml --gpus 0,1,2,3`
- Next plan (02-03) can build the activation capture script that reads traces from valid/
- Quality check script can scan valid/ and failed/ directories for reporting

---
*Phase: 02-trace-generation-at-scale*
*Completed: 2026-02-05*
