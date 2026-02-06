---
phase: 01-foundation-and-validation
plan: 02
subsystem: data
tags: [datasets, huggingface, gsm-symbolic, folio, strategyqa, verification-prompts]

requires:
  - phase: 01-01
    provides: project scaffold, lib directory structure

provides:
  - load_gsm_symbolic: binary verification format with plausible numerical distractors
  - load_folio: binary verification format with Uncertain filtered, balanced labels
  - load_strategyqa: binary verification format with balanced Yes/No
  - load_dataset_by_name: dispatcher for all three datasets
  - uniform problem schema with problem_id, ground_truth, verification_prompt

affects: [01-03, 01-05, 02-trace-generation, 03-probe-training]

tech-stack:
  added: [datasets (HuggingFace)]
  patterns: [uniform schema dict, seeded RNG without global state, minority-class balancing]

key-files:
  created:
    - src/experimental/gorn/lib/datasets.py
  modified: []

key-decisions:
  - "Distractor generation: 1-10% offset from correct answer (min 1-2) for plausible wrong answers"
  - "FOLIO label 'Uncertain' filtered (not 'Unknown') -- verified against actual dataset strings"
  - "All answers are integers in GSM-Symbolic (no floats, no negatives, no zeros in practice)"

patterns-established:
  - "Uniform problem schema: problem_id, question, ground_truth, dataset_name, original_answer, verification_prompt, presented_answer"
  - "ground_truth is always 'Yes'/'No' string, never bool -- matches (Yes or No) prompt framing"
  - "Seeded RNG via random.Random(seed) -- never touches global random state"

duration: 4min
completed: 2026-02-04
---

# Phase 1 Plan 2: Dataset Loading Summary

**Three dataset loaders (GSM-Symbolic, FOLIO, StrategyQA) producing balanced binary verification problems with uniform schema and plausible distractors**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-04T12:02:39Z
- **Completed:** 2026-02-04T12:06:49Z
- **Tasks:** 1
- **Files created:** 1

## Accomplishments

- GSM-Symbolic loader with plausible numerical distractors (1-10% offset) and paired Yes/No items
- FOLIO loader filtering "Uncertain" labels, mapping True/False to Yes/No, balanced to minority class
- StrategyQA loader with bool-to-string conversion and balanced sampling
- Uniform 7-field schema across all datasets with explicit (Yes or No) and reasoning instruction in every prompt
- load_dataset_by_name dispatcher with ValueError on unknown datasets

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement dataset loading module with all three datasets** - `6bb8244` (feat)

## Files Created/Modified

- `src/experimental/gorn/lib/datasets.py` - Three dataset loaders + dispatcher + helper functions (315 lines)

## Decisions Made

- Distractor generation uses 1-10% offset from correct answer (minimum offset of 1-2) to produce plausible wrong answers that require computation to reject
- Verified FOLIO uses "Uncertain" string (not "Unknown") by inspecting actual dataset
- All 5000 GSM-Symbolic answers are positive integers -- simplified parsing (no float/negative handling needed in practice, but code handles them defensively)

## Deviations from Plan

None -- plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None -- no external service configuration required. Datasets are downloaded from HuggingFace on first use and cached locally.

## Next Phase Readiness

- Dataset loaders ready for consumption by validation script (01-05) and trace generation pipeline
- All three datasets verified: correct balance, schema, prompt format, distractor quality
- No blockers for downstream plans

---
*Phase: 01-foundation-and-validation*
*Completed: 2026-02-04*
