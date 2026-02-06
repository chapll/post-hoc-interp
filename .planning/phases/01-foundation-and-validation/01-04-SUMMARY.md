---
phase: 01-foundation-and-validation
plan: 04
subsystem: storage
tags: [safetensors, bfloat16, trace-storage, json-metadata, topk-logits, entropy]

# Dependency graph
requires:
  - phase: 01-01
    provides: project scaffold with lib/ directory structure
provides:
  - save_trace / load_trace for safetensors round-trip
  - prepare_trace_tensors for processing raw model outputs into storage format
  - save_metadata / load_metadata for companion JSON files
  - get_trace_path / get_metadata_path for standardized path generation
affects: [01-05, 01-06, 02-trace-generation, 03-probe-training]

# Tech tracking
tech-stack:
  added: [safetensors]
  patterns: [safetensors for tensor storage, companion JSON for metadata, BF16 activation storage, int32 for indices]

key-files:
  created:
    - src/experimental/gorn/lib/storage.py
  modified: []

key-decisions:
  - "Squeeze batch dimension in prepare_trace_tensors so stored tensors are [seq_len, hidden] not [1, seq_len, hidden]"
  - "Use float32 softmax for entropy computation then cast result to BF16 for numerical stability"
  - "Clamp log_probs to -1e10 to avoid NaN from log(0)"

patterns-established:
  - "Trace storage format: safetensors with BF16 activations, int32 IDs/indices, BF16 entropy"
  - "Path convention: {base_dir}/traces/{dataset_name}/{problem_id}/trace_{idx:02d}.safetensors"
  - "Companion metadata: same path with .json extension replacing .safetensors"

# Metrics
duration: 2min
completed: 2026-02-04
---

# Phase 1 Plan 4: Trace Storage Summary

**Safetensors-based trace storage with BF16 activations, top-50 logit tracking, per-position entropy, and companion JSON metadata**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-04T12:04:31Z
- **Completed:** 2026-02-04T12:06:04Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments
- Complete trace storage module with save/load round-trip preserving all tensor dtypes exactly
- prepare_trace_tensors processes raw model outputs: top-k logit extraction, entropy computation, dtype enforcement
- Companion JSON metadata for problem-level information (problem_id, dataset, ground_truth, etc.)
- Standardized path generation for organized trace directory structure

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement storage.py with safetensors save/load and metadata** - `5a41255` (feat)

## Files Created/Modified
- `src/experimental/gorn/lib/storage.py` - Trace storage module: save_trace, load_trace, prepare_trace_tensors, metadata I/O, path utilities

## Decisions Made
- Squeeze batch dimension during prepare_trace_tensors so stored residual activations are [seq_len, hidden_dim] rather than [1, seq_len, hidden_dim] -- saves space and simplifies downstream probe training
- Compute softmax in float32 before entropy calculation, then cast result to BF16 -- prevents numerical instability from BF16 softmax
- Clamp log_probs to -1e10 to handle log(0) gracefully without silent NaN propagation

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None -- no external service configuration required.

## Next Phase Readiness
- Storage module ready to be consumed by trace capture pipeline (Phase 2)
- Storage module ready to be consumed by probe training data loader (Phase 3)
- All functions tested with synthetic data at realistic shapes (seq_len=100, hidden_dim=4096, vocab_size=151936)

---
*Phase: 01-foundation-and-validation*
*Completed: 2026-02-04*
