---
phase: 03-probing-and-transition-detection
plan: 04
subsystem: analysis
tags: [nnsight, linear-probe, transition-detection, cosine-similarity, mann-whitney, cohens-d, parquet, matplotlib, ruptures]

# Dependency graph
requires:
  - phase: 03-01
    provides: "src/transition.py (detect_transitions, sweep_pelt_penalties) and configs/phase3.yaml"
  - phase: 03-03
    provides: "Probe weights (safetensors) for all 36 layers per dataset and pooled, probe_aurocs.json"
provides:
  - "scripts/detect_transitions.py -- per-token confidence curves and 4-method transition detection"
  - "scripts/compute_cosine_sim.py -- step-wise cosine similarity with Mann-Whitney U and Cohen's d"
  - "scripts/plot_phase3.py -- parquet assembly, 6 QA visualization plots, summary report"
  - "Per-trace transition JSONs at $DATA_DIR/results/phase3/transitions/"
  - "Per-dataset parquet files at $DATA_DIR/results/phase3/{dataset}_results.parquet"
  - "Trace summary parquet at $DATA_DIR/results/phase3/trace_summary.parquet"
  - "6 QA figures at $DATA_DIR/results/phase3/figures/"
affects: [phase-4, phase-5, phase-6]

# Tech tracking
tech-stack:
  added: []
  patterns: [per-trace-confidence-curves, four-method-transition-detection, pre-post-cosine-comparison, long-format-parquet]

key-files:
  created:
    - scripts/detect_transitions.py
    - scripts/compute_cosine_sim.py
    - scripts/plot_phase3.py
  modified: []

key-decisions:
  - "Probe ALL think-block content positions (mask already excludes boundary tokens) -- no extra skip logic needed"
  - "Per-dataset probes preferred over pooled; fall back to pooled when per-dataset unavailable"
  - "Transition point preference order: sigmoid > derivative > binseg > pelt (for segment assignment and cosine comparison)"
  - "Pooled standard deviation for Cohen's d (not Welch's, since group sizes may differ substantially)"
  - "Long-format parquet: one row per token-position per layer for maximum analysis flexibility"

patterns-established:
  - "load_probes(): reconstruct nn.Linear from safetensors weight/bias pairs, move to GPU"
  - "process_trace(): single nnsight forward pass extracting residual stream at N layers, apply probes at all eligible positions"
  - "_serialize_method_result(): recursive numpy-to-native-type conversion for JSON storage"
  - "compare_pre_post(): Mann-Whitney U + Cohen's d for pre-T vs post-T cosine comparison"
  - "assemble_parquet(): per-trace JSON to long-format DataFrame with all Phase 3 columns"
  - "Publication-quality plot functions: consistent rcParams, no emoji, proper labels"

requirements-completed: [REQ-P3, REQ-P4]

# Metrics
duration: 7min
completed: 2026-02-26
---

# Phase 3 Plan 04: Confidence Curves, Transition Detection, and QA Visualization Summary

**Per-token probe confidence curves with 4-method transition detection, step-wise cosine similarity dynamics with Mann-Whitney U statistics, and 6 QA visualization plots assembled into per-dataset parquet files**

## Performance

- **Duration:** 7 min
- **Started:** 2026-02-26T11:01:12Z
- **Completed:** 2026-02-26T11:08:26Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- Created `scripts/detect_transitions.py` (829 lines) applying trained probes at every think-block token position, generating confidence curves at top-N layers, and running all 4 detection methods (derivative, PELT with penalty sweep, BinSeg, sigmoid) per trace
- Created `scripts/compute_cosine_sim.py` (500 lines) computing step-wise cosine similarity S_t = cos(h_t, h_{t-1}) at best + 3 additional layers, with Mann-Whitney U test and Cohen's d for pre-T vs post-T comparison on clear traces
- Created `scripts/plot_phase3.py` (1022 lines) assembling per-dataset long-format parquet files (one row per token-position per layer) and trace-summary parquet, plus 6 publication-quality QA visualizations: AUROC curves, confidence curve examples with detection boundaries, transition point distributions, classification bars, cosine similarity phase plots, and Cohen's d histograms

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement transition detection script** - `62a79fa` (feat)
2. **Task 2: Implement cosine similarity dynamics script** - `42e3a98` (feat)
3. **Task 3: Implement QA visualization and parquet assembly script** - `aa87a24` (feat)

## Files Created/Modified
- `scripts/detect_transitions.py` - Per-token confidence curves at all think-block positions, 4-method transition detection, three-way classification (clear/no-signal/noisy-signal), per-trace JSON output
- `scripts/compute_cosine_sim.py` - Step-wise cosine similarity at best + 3 layers, pre-T vs post-T Mann-Whitney U test with Cohen's d, results appended to transition JSONs, aggregate statistics JSON
- `scripts/plot_phase3.py` - Parquet assembly (long-format per-dataset + trace summary), 6 QA plots (AUROC, confidence curves, transition distribution, classification bars, cosine similarity, Cohen's d), summary report with indeterminate rate

## Decisions Made
- Probe ALL positions where think content mask is True -- the mask from get_think_content_mask() already excludes boundary tokens (<think> and </think>), so no additional skip logic is needed
- Per-dataset probes are preferred over pooled; the script falls back to pooled probes when per-dataset probes are unavailable for a given dataset
- Transition point preference order for segment assignment and cosine comparison: sigmoid, derivative, binseg, pelt (sigmoid provides the most interpretable midpoint-based detection)
- Pooled standard deviation used for Cohen's d calculation (classic formulation) rather than Welch's, appropriate for pre/post groups that may have different sizes
- Long-format parquet design (one row per token-position per layer) chosen for maximum downstream flexibility in Phase 4 differential analysis

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - all three scripts are ready to run on GPU. Usage:
1. `python scripts/detect_transitions.py --config configs/phase3.yaml` (requires GPU, runs nnsight forward passes)
2. `python scripts/compute_cosine_sim.py --config configs/phase3.yaml` (requires GPU, reads transition results from step 1)
3. `python scripts/plot_phase3.py --config configs/phase3.yaml` (CPU only, reads results from steps 1-2)

## Next Phase Readiness
- Per-trace transition JSONs provide all inputs for Phase 4 (differential analysis) and Phase 5 (steering)
- Parquet files contain all required columns: trace_id, problem_id, dataset, token_position, layer, probe_confidence, cosine_similarity_S_t, transition_point, transition_category, segment_label
- Segment labels (reasoning/transition/post-hoc) enable direct comparison of pre-transition vs post-transition activation properties
- Cohen's d and p-value statistics provide evidence strength for the cosine similarity signature of post-hoc reasoning

## Self-Check: PASSED

- scripts/detect_transitions.py: FOUND (829 lines, min 250)
- scripts/compute_cosine_sim.py: FOUND (500 lines, min 150)
- scripts/plot_phase3.py: FOUND (1022 lines, min 150)
- Commit 62a79fa: FOUND
- Commit 42e3a98: FOUND
- Commit aa87a24: FOUND
- 03-04-SUMMARY.md: FOUND

---
*Phase: 03-probing-and-transition-detection*
*Completed: 2026-02-26*
