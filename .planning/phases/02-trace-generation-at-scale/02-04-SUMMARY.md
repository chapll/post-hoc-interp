---
phase: 02-trace-generation-at-scale
plan: 04
subsystem: infra
tags: [nnsight, activation-capture, chunked-capture, safetensors, multi-gpu, bfloat16]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: "lib/model.py load_model, lib/storage.py save_trace/get_activation_path"
  - phase: 02-trace-generation-at-scale
    plan: 01
    provides: "run_parallel_isolated, get_activation_path, save_metadata, capture.yaml config"
provides:
  - "capture_activations.py: multi-GPU chunked activation capture script"
  - "capture_all_layers_chunked(): chunked nnsight trace with all 36 layers"
  - "verify_logits(): reference forward pass comparison for correctness"
  - "scan_valid_traces(): directory scanner for valid trace JSONs"
affects:
  - 03-probing (activation safetensors files are input to probing analysis)
  - 02-05 (may reference capture patterns or activation file layout)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Chunked nnsight trace: iterate all layers sequentially, .save() only chunk subset"
    - "Unsaved layer access via _ = model.model.layers[idx].output for ordering compliance"
    - "Logits verification: argmax match primary, torch.allclose secondary"
    - "Companion metadata JSON alongside safetensors activation files"

key-files:
  created:
    - "src/experimental/gorn/scripts/capture_activations.py"
  modified: []

key-decisions:
  - "Last-token logits stored in activations for later verification (logits_last_token key)"
  - "Verify logits on first 3 traces per worker only (avoid 4th forward pass overhead)"
  - "Argmax match is primary verification signal; allclose failure is warning not error"
  - "Scan directory structure rather than manifest file for trace discovery"

patterns-established:
  - "Chunked capture: 12 layers per pass, 3 passes for 36 layers, sequential access in each"
  - "Activation storage: residual_layer_0..35 + topk_values/indices + entropy + token_ids"
  - "Worker function signature: (rank, gpu_id, trace_items, cfg) with model loading inside"

# Metrics
duration: 3min
completed: 2026-02-05
---

# Phase 2 Plan 04: Chunked Activation Capture Script Summary

**Multi-GPU chunked nnsight activation capture for all 36 Qwen3-8B layers with logits verification, file-based resume, and fault-tolerant workers**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-05T08:29:57Z
- **Completed:** 2026-02-05T08:32:33Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created `capture_activations.py` (569 lines) implementing Stage 2 of the decoupled pipeline: reads valid trace JSONs, runs teacher-forced nnsight forward passes with chunked layer capture, saves safetensors + companion metadata
- Chunked capture function runs 3 forward passes (12 layers each) to capture all 36 residual stream layers without OOM, satisfying nnsight 0.5 sequential access requirement in each pass
- Logits verification function compares last-token logits from capture against a reference forward pass (argmax + allclose) on first 3 traces per worker
- File-based resume: scans for existing activation files and skips already-captured traces
- Multi-GPU dispatch via `run_parallel_isolated` for fault-tolerant parallel capture across GPUs

## Task Commits

Each task was committed atomically:

1. **Task 1: Create capture_activations.py with chunked nnsight capture** - `2f223a5` (feat)

## Files Created/Modified

- `src/experimental/gorn/scripts/capture_activations.py` - Multi-GPU chunked activation capture script with logits verification, worker function, directory scanning, and main dispatch

## Decisions Made

- **Last-token logits in activations:** Store `logits_last_token` (full vocab, BF16) in the safetensors file alongside residuals, enabling post-hoc verification without re-running the model.
- **Verification scope:** Verify logits on first 3 traces per worker only. This confirms correctness without adding a 4th forward pass for every single trace.
- **Argmax as primary signal:** If argmax matches but allclose fails (due to BF16 precision), treat as a warning rather than error. Functionally equivalent predictions mean the forward pass is correct.
- **Directory scanning over manifest:** Use `os.listdir` on the valid/ directory tree rather than maintaining a manifest file. Simpler, no state to corrupt, consistent with file-based resume pattern.
- **Metadata propagation:** Copy relevant fields (ground_truth, question, is_fresh, seed, num_think_blocks) from trace JSON metadata into activation companion JSON for downstream convenience.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Activation capture script is ready to run on valid traces produced by generate_traces.py (02-02)
- Output safetensors files will be input for Phase 3 probing analysis
- Disk space monitoring via per-trace file size logging helps catch quota issues early

---
*Phase: 02-trace-generation-at-scale*
*Completed: 2026-02-05*
