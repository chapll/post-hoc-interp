---
phase: 02-trace-generation-at-scale
plan: 06
subsystem: prompts
tags: [swapped-label, probing, prompts, metadata, rho-tau]

# Dependency graph
requires:
  - phase: 02-trace-generation-at-scale
    plan: 01-05
    provides: "Complete Phase 2 pipeline"
provides:
  - "lib/prompts.py: Prompt construction with swapped-label protocol (Rho/Tau)"
  - "Problem expansion into A/B config variants"
  - "Trace metadata with prompt_config, label_mapping, valid_token, invalid_token"
affects:
  - Phase 3 (probes train on ground_truth decoupled from output token)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Swapped-label protocol: Config A (Valid=Rho, Invalid=Tau), Config B (swapped)"
    - "Prompt builder function with configurable label protocol"
    - "Per-trace metadata for label mapping interpretation"

key-files:
  created:
    - "src/experimental/gorn/lib/prompts.py"
  modified:
    - "src/experimental/gorn/lib/datasets.py"
    - "src/experimental/gorn/scripts/generate_traces.py"
    - "src/experimental/gorn/scripts/capture_activations.py"
    - "src/experimental/gorn/scripts/validate.py"
    - "src/experimental/gorn/scripts/verify_phase2.py"
    - "src/experimental/gorn/configs/generate.yaml"

key-decisions:
  - "Rho/Tau tokens instead of Yes/No to eliminate probe confound from surface token detection"
  - "Each problem generates A and B config variants (doubles problem count)"
  - "traces_per_problem reduced from 5 to 3 to offset config doubling (~2400 traces total)"
  - "ground_truth remains Yes/No as abstract validity label (probe training target)"

patterns-established:
  - "Prompt construction separated from dataset loading"
  - "Swapped-label protocol for probe deconfounding"

# Metrics
duration: 15min
completed: 2026-02-05
---

# Phase 2 Plan 06: Swapped-Label Probing Protocol Summary

**Implement Rho/Tau swapped-label protocol to eliminate Yes/No probe confound**

## Performance

- **Duration:** 15 min
- **Started:** 2026-02-05
- **Completed:** 2026-02-05
- **Tasks:** 5
- **Files created:** 1
- **Files modified:** 6 + planning files

## Accomplishments

- Created `lib/prompts.py` (113 lines) with `build_verification_prompt()` function supporting:
  - "swapped" protocol: Rho/Tau tokens with Config A and Config B mappings
  - "yesno" protocol: Yes/No tokens for backward compatibility
  - Per-dataset prompt templates (GSM-Symbolic, FOLIO, StrategyQA)

- Modified `lib/datasets.py` to:
  - Remove inline prompt construction from all loaders
  - Add `premises` field to FOLIO loader for prompt reconstruction
  - Return base problem dicts without `verification_prompt` field

- Modified `generate_traces.py` to:
  - Import and use `build_verification_prompt()` from `lib.prompts`
  - Expand each problem into A and B config variants
  - Record `prompt_config`, `label_mapping`, `valid_token`, `invalid_token` in trace metadata

- Modified `capture_activations.py` to copy swapped-label metadata fields

- Modified `validate.py` to use `build_verification_prompt()` with yesno protocol

- Modified `verify_phase2.py` to check for swapped-label metadata

- Updated `generate.yaml`:
  - Added `label_protocol: "swapped"`
  - Reduced `traces_per_problem` from 5 to 3

## The Confound and Solution

### The Problem
The model uses "yes" and "no" during reasoning (e.g., "Assume the answer is yes..."). A probe trained on Yes/No ground truth will partially learn to detect these surface tokens, not the abstract validity signal.

### The Solution
Replace Yes/No verification prompts with Rho/Tau output tokens. Generate each problem under two complementary label mappings:
- **Config A:** Valid=Rho, Invalid=Tau
- **Config B:** Valid=Tau, Invalid=Rho

With both configs in training data, Rho/Tau have zero correlation with the validity label, forcing the probe to find the abstract logical signal.

## Prompt Templates

### GSM-Symbolic
```
{question}

Is the answer {presented_answer}? If it is correct, output exactly the word: {valid_word}. If it is not correct, output exactly the word: {invalid_word}.
Please think step-by-step to verify whether the answer is correct.
```

### FOLIO
```
Given the following premises:
{premises}

Does the following conclusion logically follow from the premises: "{conclusion}"?
If it logically follows, output exactly the word: {valid_word}. If it does not follow, output exactly the word: {invalid_word}.
Please think step-by-step to verify.
```

### StrategyQA
```
Answer the following question: {question}
If the answer is yes, output exactly the word: {valid_word}. If the answer is no, output exactly the word: {invalid_word}.
Please think step-by-step to verify your answer.
```

## Trace Count Impact

- **Before:** ~400 problems x 5 traces = ~2000 traces
- **After:** ~400 problems x 2 configs x 3 traces = ~2400 traces

## Trace Metadata Schema

Each trace now includes:
```json
{
  "metadata": {
    "prompt_config": "A",           // or "B"
    "label_mapping": {
      "valid": "Rho",               // or "Tau"
      "invalid": "Tau"              // or "Rho"
    },
    "valid_token": "Rho",           // which token means valid for THIS trace
    "invalid_token": "Tau",         // which token means invalid for THIS trace
    "ground_truth": "Yes"           // abstract validity (probe training target)
  }
}
```

## Files Created/Modified

- `src/experimental/gorn/lib/prompts.py` (NEW) - Prompt construction with swapped-label protocol
- `src/experimental/gorn/lib/datasets.py` - Removed inline prompts, added FOLIO premises field
- `src/experimental/gorn/scripts/generate_traces.py` - A/B expansion, metadata recording
- `src/experimental/gorn/scripts/capture_activations.py` - Copy swapped-label metadata
- `src/experimental/gorn/scripts/validate.py` - Use prompts.py with yesno protocol
- `src/experimental/gorn/scripts/verify_phase2.py` - Check swapped-label metadata
- `src/experimental/gorn/configs/generate.yaml` - label_protocol, traces_per_problem

## Planning Files Updated

- `.planning/PROJECT.md` - Added Swapped-Label Protocol to Methodology References, Key Decisions
- `.planning/REQUIREMENTS.md` - Updated REQ-F2 (prompts.py), REQ-P1 (probe label note)
- `.planning/ROADMAP.md` - Added 02-06 plan, updated Phase 2 goal and success criteria
- `.planning/STATE.md` - Updated current focus, pending todos, decisions
- `.planning/phases/02-trace-generation-at-scale/02-CONTEXT.md` - Added Swapped-Label Protocol section

## Deviations from Plan

None. All tasks completed as specified.

## Issues Encountered

None.

## Next Steps

1. Delete old test data in `data/traces/` (uses old Yes/No prompts)
2. Run full-scale trace generation with swapped-label protocol:
   ```bash
   uv run python src/experimental/gorn/scripts/generate_traces.py \
     --config src/experimental/gorn/configs/generate.yaml \
     --gpus 0,1,2,3
   ```
3. Run quality gate and activation capture
4. Proceed to Phase 3 (Probing and Transition Detection)

---
*Phase: 02-trace-generation-at-scale*
*Completed: 2026-02-05*
