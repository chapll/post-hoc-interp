---
phase: 02-trace-generation-at-scale
plan: 01
subsystem: infra
tags: [gsm-symbolic, contamination, multiprocessing, fault-tolerance, storage, omegaconf]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: "lib/datasets.py loaders, lib/multiprocess.py run_parallel, lib/storage.py save/load, lib/config.py load_config"
provides:
  - "load_gsm_symbolic_contamination() for fresh+original paired problems"
  - "run_parallel_isolated() with mp.Process fault isolation"
  - "Decoupled trace/activation path helpers (get_trace_json_path, get_activation_path, etc.)"
  - "write_trace_json() with POSIX atomic writes"
  - "derive_trace_seed() for deterministic per-trace seed derivation"
  - "configs/generate.yaml and configs/capture.yaml"
affects:
  - 02-02 (generate_traces.py uses all new lib functions)
  - 02-03 (check_traces.py uses trace path helpers)
  - 02-04 (capture_activations.py uses activation path helpers and capture config)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "mp.Process for fault-isolated multi-GPU workers (independent crash handling)"
    - "Atomic JSON writes via tempfile.mkstemp + os.rename"
    - "Hash-based per-trace seed derivation (MD5, stateless, order-independent)"
    - "Decoupled trace/activation directory layout with separate roots"
    - "Deterministic shuffle before work splitting for balanced GPU load"

key-files:
  created:
    - "src/experimental/gorn/configs/generate.yaml"
    - "src/experimental/gorn/configs/capture.yaml"
  modified:
    - "src/experimental/gorn/lib/datasets.py"
    - "src/experimental/gorn/lib/multiprocess.py"
    - "src/experimental/gorn/lib/storage.py"

key-decisions:
  - "GSM-Symbolic contamination: use HF dataset instance 0 as fresh, original_question as contaminated baseline"
  - "Balance and sample fresh/original items independently to maintain 50/50 Yes/No within each type"
  - "run_parallel_isolated always uses mp.Process even for single GPU (run_parallel available for in-process debugging)"
  - "Extracted _parse_gsm_answer() helper to DRY answer parsing logic"

patterns-established:
  - "Fault-isolated workers: mp.Process + join + exitcode checking"
  - "Atomic file writes: tempfile.mkstemp in same dir, then os.rename"
  - "Decoupled storage: traces in data/traces/{valid,failed}/, activations in data/activations/"
  - "Deterministic per-trace seeds: hashlib.md5(problem_id_traceindex) % 2^32, skip 42"

# Metrics
duration: 4min
completed: 2026-02-05
---

# Phase 2 Plan 01: Library Foundation Updates Summary

**GSM-Symbolic contamination loader with fresh+original pairing, fault-tolerant mp.Process multi-GPU, decoupled trace/activation storage paths, and generate/capture config files**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-05T08:21:44Z
- **Completed:** 2026-02-05T08:25:44Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added `load_gsm_symbolic_contamination()` that loads both fresh (generated variant) and original (GSM8K baseline) problems with `is_fresh` field, enabling contamination analysis across all Phase 2 scripts
- Added `run_parallel_isolated()` using `mp.Process` for fault-tolerant multi-GPU inference where one worker crash does not kill others, with deterministic shuffle for balanced GPU workload distribution
- Added decoupled storage path helpers (`get_trace_json_path`, `get_failed_trace_json_path`, `get_activation_path`) supporting the separate trace/activation directory layout
- Added `write_trace_json()` with POSIX atomic writes and `derive_trace_seed()` for stateless reproducible seed derivation
- Created `generate.yaml` (max_new_tokens=4096, traces_dir) and `capture.yaml` (chunked capture config with 36 layers, activations_dir)

## Task Commits

Each task was committed atomically:

1. **Task 1: GSM-Symbolic contamination loader and fault-tolerant multiprocess** - `35387fd` (feat)
2. **Task 2: Decoupled storage helpers and config files** - `93c1be6` (feat)

## Files Created/Modified

- `src/experimental/gorn/lib/datasets.py` - Added load_gsm_symbolic_contamination(), _parse_gsm_answer(), updated load_dataset_by_name()
- `src/experimental/gorn/lib/multiprocess.py` - Added run_parallel_isolated(), _isolated_worker_wrapper()
- `src/experimental/gorn/lib/storage.py` - Added get_trace_json_path(), get_failed_trace_json_path(), get_activation_path(), write_trace_json(), derive_trace_seed()
- `src/experimental/gorn/configs/generate.yaml` - Generation-specific config overrides
- `src/experimental/gorn/configs/capture.yaml` - Capture-specific config overrides

## Decisions Made

- **GSM-Symbolic contamination approach:** Use HuggingFace dataset's instance 0 as fresh variant and original_question field as contaminated baseline. No need for Apple's template generator (not publicly released).
- **Independent balancing:** Fresh and original items are balanced and sampled independently (via _balance_and_sample) before combining, ensuring 50/50 Yes/No within each type.
- **Extracted _parse_gsm_answer():** DRY helper for the #### separator parsing logic shared between load_gsm_symbolic() and load_gsm_symbolic_contamination().
- **Single-GPU mp.Process:** run_parallel_isolated() always uses mp.Process even for single GPU, providing consistent fault isolation. Users wanting in-process debugging can use the existing run_parallel().

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed unused pathlib.Path import in storage.py**
- **Found during:** Task 2 (storage helpers)
- **Issue:** ruff F401: `pathlib.Path` was imported but never used (pre-existing in Phase 1 code)
- **Fix:** Removed the unused import
- **Files modified:** src/experimental/gorn/lib/storage.py
- **Verification:** ruff check passes
- **Committed in:** 93c1be6 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Trivial lint fix. No scope creep.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All lib modules updated and ready for generate_traces.py (02-02), check_traces.py (02-03), and capture_activations.py (02-04)
- Config files provide all required parameters for both generation and capture scripts
- Backward compatibility maintained: all Phase 1 functions untouched

---
*Phase: 02-trace-generation-at-scale*
*Completed: 2026-02-05*
