---
phase: 01-foundation-and-validation
plan: 03
subsystem: model-infra
tags: [nnsight, qwen3, activation-capture, think-tokens, bfloat16]

# Dependency graph
requires:
  - phase: 01-01
    provides: utils.py (set_seed, get_device, setup_logging), config.py, base.yaml, nnsight installed
provides:
  - "model.py: load_model, generate_trace, capture_activations for Qwen3-8B via nnsight"
  - "tokens.py: parse_think_boundaries, get_think_content_mask, count_think_blocks, get_pre_handoff_window"
  - "Token ID constants: THINK_TOKEN_ID (151667), THINK_END_TOKEN_ID (151668)"
affects: [01-04, 01-05, 01-06, 02-data-collection, 03-probing]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-pass activation capture: generate with _model.generate(), trace with model.trace()"
    - "nnsight layer access: model.model.layers[idx].output[0] for residual stream"
    - "Token ID based think boundary parsing (not string based)"
    - "dtype= instead of torch_dtype= for transformers 5.0 compatibility"

key-files:
  created:
    - src/experimental/gorn/lib/model.py
    - src/experimental/gorn/lib/tokens.py
  modified: []

key-decisions:
  - "Use model._model.generate() for generation instead of nnsight .generate() wrapper"
  - "Use dtype= parameter for nnsight LanguageModel (torch_dtype deprecated in transformers 5.0)"
  - "Think boundary parsing uses token IDs, not string matching (tokens have special=false)"
  - "Pre-handoff window excludes </think> boundary token itself"

patterns-established:
  - "Two-pass approach: generate first, then trace for activations"
  - "Always use .save() on nnsight proxy values for persistence"
  - "Layer output tuple unpacking: .output[0] for hidden states"

# Metrics
duration: 7min
completed: 2026-02-04
---

# Phase 1 Plan 3: Model + Token Parsing Summary

**nnsight Qwen3-8B integration with two-pass activation capture and token-ID-based think boundary parsing**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-04T12:04:12Z
- **Completed:** 2026-02-04T12:11:41Z
- **Tasks:** 2
- **Files created:** 2

## Accomplishments
- Built tokens.py with think boundary parsing, content masks, and pre-handoff window extraction
- Built model.py with nnsight LanguageModel wrapping, thinking-mode generation, and activation capture
- Validated nnsight trace API works (layer output access, last-token slicing, lm_head logits) using GPT-2 test
- Verified Qwen3-8B config: 36 layers, hidden_size 4096, vocab_size 151936
- Verified tokenizer: token 151667 = <think>, token 151668 = </think>, chat template with enable_thinking works
- Fixed torch_dtype deprecation (transformers 5.0 uses dtype=)

## Task Commits

Each task was committed atomically:

1. **Task 1: Build tokens.py for think boundary parsing** - `a7c18c7` (feat)
2. **Task 2: Build model.py with nnsight integration** - `ded3c20` (feat)

## Files Created/Modified
- `src/experimental/gorn/lib/tokens.py` - Think token boundary parsing, content masks, pre-handoff window
- `src/experimental/gorn/lib/model.py` - Qwen3-8B loading via nnsight, generation, activation capture

## Decisions Made
- **model._model.generate() over nnsight .generate():** Research doc recommends raw HuggingFace model for generation; nnsight's generate wrapper may have issues. Validated _model access on GPT-2.
- **dtype= instead of torch_dtype=:** transformers 5.0 deprecation warning caught during testing.
- **Token ID parsing over string parsing:** <think> and </think> have special=false, making string parsing fragile. All boundary detection uses integer token ID comparison.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed torch_dtype deprecation for transformers 5.0**
- **Found during:** Task 2 (model loading)
- **Issue:** `torch_dtype` parameter is deprecated in transformers 5.0; produces deprecation warning
- **Fix:** Changed to `dtype=torch.bfloat16` in LanguageModel constructor
- **Files modified:** src/experimental/gorn/lib/model.py
- **Verification:** Deprecation warning removed
- **Committed in:** ded3c20 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Minor parameter name update. No scope creep.

## Issues Encountered
- **GPU unavailability:** All 7 A100-80GB GPUs on the machine were occupied by other processes (71-73GB each), leaving only 5-8GB free per GPU. Qwen3-8B requires ~16GB in BF16. Full GPU smoke test (model load + generation + activation capture) could not be completed. Partial validation was done: nnsight trace API verified on GPT-2 (layer access, last-token slicing, lm_head logits all work). Qwen3-8B config and tokenizer verified (36 layers, 4096 hidden, 151936 vocab, think token IDs correct).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- tokens.py is fully tested and ready for use in trace generation and probing
- model.py is code-complete and import-validated; full GPU smoke test should be run when GPU capacity is available
- The GPU smoke test is the highest-priority validation before Phase 2 data collection begins
- Recommended: run the smoke test with `uv run python _smoke_test_model.py` when a GPU with 16+ GB free is available

---
*Phase: 01-foundation-and-validation*
*Completed: 2026-02-04*
