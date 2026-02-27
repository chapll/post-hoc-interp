---
phase: 03-probing-and-transition-detection
verified: 2026-02-26T12:00:00Z
status: human_needed
score: 11/11 automated must-haves verified
re_verification: false
human_verification:
  - test: "Run layer_selection_pilot.py on GPU and confirm pilot_layer_aurocs.json is produced"
    expected: "JSON file with per_layer AUROC for all 36 layers, top_layers list, pilot_auroc_curve.png"
    why_human: "Requires GPU + model load + real trace data; cannot verify statically"
  - test: "Run train_probes.py on GPU and confirm probe safetensors are saved"
    expected: "{dataset}_probes.safetensors and pooled_probes.safetensors under $DATA_DIR/results/phase3/probes/"
    why_human: "Requires GPU execution and real activation extraction"
  - test: "Run detect_transitions.py on GPU and confirm per-trace transition JSONs are saved"
    expected: "JSON files under $DATA_DIR/results/phase3/transitions/{dataset}/ with confidence_curve and detection_results fields"
    why_human: "Requires trained probe weights and GPU forward passes"
  - test: "Run compute_cosine_sim.py and confirm cosine_similarity_stats.json is produced"
    expected: "Aggregate JSON with per_dataset Cohen's d distributions; fraction_significant_p05 reported"
    why_human: "Requires transition JSON outputs from detect_transitions.py"
  - test: "Run plot_phase3.py and confirm all 6 figure PNG files and parquet files are saved"
    expected: "auroc_per_layer.png, confidence_curve_examples.png, transition_distribution.png, classification_distribution.png, cosine_similarity_examples.png, cohens_d_distribution.png; {dataset}_results.parquet files"
    why_human: "Requires full pipeline outputs from all prior scripts"
  - test: "Verify ruptures is importable in the project virtualenv"
    expected: "python -c 'import ruptures' succeeds in the project environment"
    why_human: "System python3 does not have ruptures installed; must run inside project venv"
---

# Phase 3: Probing and Transition Detection Verification Report

**Phase Goal:** Researcher has layer-wise probe AUROC curves across all three datasets with selectivity controls, per-trace transition points with quality categorization, cosine similarity dynamics with statistical tests. Probes train on ground_truth (abstract validity: Yes/No), decoupled from output token (Rho/Tau) by the swapped-label protocol.
**Verified:** 2026-02-26
**Status:** human_needed (all code verified; GPU execution required for data outputs)
**Re-verification:** No -- initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | ruptures library registered as dependency | VERIFIED | pyproject.toml line 17: `"ruptures>=1.1.10"` |
| 2 | phase3.yaml loads with all required sections | VERIFIED | File exists; grep confirms phase3, probe, selectivity, transition, classification, layers, cosine, results, paths sections |
| 3 | src/transition.py implements detect_transitions() with four methods | VERIFIED | All four methods present: _derivative_method (L188), _pelt_method (L291), _binseg_method (L384), _sigmoid_method (L483); detect_transitions (L644); sweep_pelt_penalties (L706) |
| 4 | transition.py is standalone (no src.* imports) | VERIFIED | Only docstring reference to src.transition; no runtime imports from src.* |
| 5 | Layer selection pilot script exists with all required functions | VERIFIED | select_pilot_traces (L47), extract_pilot_activations (L173), train_layer_probes (L350), select_top_layers (L510); 833 lines |
| 6 | Probe training script exists with all required functions | VERIFIED | discover_all_traces (L50), extract_training_activations (L166), train_probes_cv (L406), train_selectivity_controls (L509), train_final_probes (L631); 1092 lines |
| 7 | detect_transitions.py imports transition module and probes | VERIFIED | `from src.transition import detect_transitions, sweep_pelt_penalties` confirmed; safetensors probe loading confirmed |
| 8 | compute_cosine_sim.py implements Mann-Whitney U and Cohen's d | VERIFIED | mannwhitneyu (L176), cohens_d computed (L187), cosine_similarity (L112) |
| 9 | plot_phase3.py assembles parquet and all 6 required plots | VERIFIED | assemble_parquet (L56), plot_auroc_curves (L260), plot_confidence_curves (L332), plot_transition_distribution present, plot_classification_distribution (L595), plot_cosine_similarity present, plot_cohens_d (L808) |
| 10 | Probes use ground_truth label (Yes/No), not output token | VERIFIED | All training scripts: `label = 1 if ground_truth == "Yes" else 0` pattern confirmed |
| 11 | 5-fold stratified CV with random_state=137 (not 42) | VERIFIED | `StratifiedKFold(n_splits=5, shuffle=True, random_state=137)` in both pilot and train_probes |

**Score:** 11/11 automated truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/transition.py` | Standalone transition detection module | VERIFIED | 750 lines; 4 methods + sweep utility; no src.* imports |
| `configs/phase3.yaml` | Phase 3 configuration | VERIFIED | All sections present including pilot, probe, selectivity, transition, layers, cosine, paths |
| `pyproject.toml` | ruptures dependency | VERIFIED | `ruptures>=1.1.10` on line 17 |
| `scripts/layer_selection_pilot.py` | Layer selection pilot (min 200 lines) | VERIFIED | 833 lines; all 4 required functions |
| `scripts/train_probes.py` | Full probe training (min 300 lines) | VERIFIED | 1092 lines; all 5 required functions; safetensors save |
| `scripts/detect_transitions.py` | Confidence curves + detection (min 250 lines) | VERIFIED | 829 lines; load_probes, process_trace, detect_trace_transitions |
| `scripts/compute_cosine_sim.py` | Cosine similarity + stats (min 150 lines) | VERIFIED | 500 lines; compute_trace_cosine, compare_pre_post |
| `scripts/plot_phase3.py` | QA visualizations + parquet (min 150 lines) | VERIFIED | 1022 lines; 6 plot functions + assemble_parquet |
| `$DATA_DIR/results/phase3/pilot_layer_aurocs.json` | Pilot AUROC results | HUMAN NEEDED | Script ready; GPU run required |
| `$DATA_DIR/results/phase3/probes/*.safetensors` | Probe weights | HUMAN NEEDED | Script ready; GPU run required |
| `$DATA_DIR/results/phase3/{dataset}_results.parquet` | Per-dataset results | HUMAN NEEDED | Script ready; depends on GPU outputs |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| src/transition.py | ruptures | `import ruptures as rpt` | VERIFIED | Line 33 |
| src/transition.py | scipy.signal.savgol_filter | derivative computation | VERIFIED | Line 35, used at L246 |
| src/transition.py | scipy.optimize.curve_fit | sigmoid fitting | VERIFIED | Line 34, used at L525 |
| scripts/layer_selection_pilot.py | src/model.py | `from src.model import load_model` | VERIFIED | Line 36 |
| scripts/layer_selection_pilot.py | src/tokens.py | `from src.tokens import` | VERIFIED | Line 37 |
| scripts/layer_selection_pilot.py | configs/phase3.yaml | `load_config` | VERIFIED | Line 35 |
| scripts/train_probes.py | pilot_layer_aurocs.json | reads top_layers | VERIFIED | `pilot_layer_aurocs.json` referenced |
| scripts/train_probes.py | src/model.py | `from src.model import` | VERIFIED | Present |
| scripts/detect_transitions.py | src/transition.py | `from src.transition import detect_transitions, sweep_pelt_penalties` | VERIFIED | Lines 34 confirmed |
| scripts/detect_transitions.py | probes safetensors | loads via `probes_dir` | VERIFIED | `load_file(safetensors_path)` at L75 |
| scripts/compute_cosine_sim.py | transitions_dir | reads transition JSONs | VERIFIED | L271, L309 |
| scripts/plot_phase3.py | results parquet | `df.to_parquet()` | VERIFIED | L205-206 |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| REQ-P1 | 03-02, 03-03 | Linear probes on residual stream, Pre-Handoff Window, AUROC per layer | SATISFIED (code) | layer_selection_pilot.py + train_probes.py implement full CV probe training on Pre-Handoff Window with ground_truth labels |
| REQ-P2 | 03-02, 03-03 | Selectivity controls (random-label permutations) | SATISFIED (code) | train_selectivity_controls() in train_probes.py; 10 permutations; warning_threshold=0.60 in config |
| REQ-P3 | 03-01, 03-04 | Transition point detection with categorization | SATISFIED (code) | src/transition.py with 4 methods; detect_transitions.py applies per-trace; clear/no-signal/noisy-signal classification |
| REQ-P4 | 03-04 | Cosine similarity dynamics with Mann-Whitney U, Cohen's d | SATISFIED (code) | compute_cosine_sim.py: S_t at multiple layers, mannwhitneyu, cohens_d computed per clear trace |
| REQ-P5 | NOT in phase 3 plans | Cox 2x2 classification | NOT CLAIMED | Per CONTEXT.md Decision 7, REQ-P5 dropped from Phase 3. REQUIREMENTS.md still shows it as "Pending" for Phase 3 -- ORPHANED from this phase |

**Note on REQ-P3 vs REQUIREMENTS.md spec:** REQUIREMENTS.md defines REQ-P3 as "sustained-K=5 at threshold 0.95" and "sliding-window W=10 at threshold 0.90". The CONTEXT.md and plans replaced this with the four-method approach (derivative, PELT, BinSeg, sigmoid) with three-way classification. The implementation follows the plans, not the original REQ-P3 text. This is a requirements drift that should be documented.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| scripts/plot_phase3.py | 94 | `return {}, pd.DataFrame()` | Info | Early-exit guard when no transition JSONs found; not a stub -- real error path |
| scripts/plot_phase3.py | 235 | `return []` | Info | Fallback in segment extraction helper when no method found segments; not a stub |

No blockers found. No TODO/FIXME/PLACEHOLDER/emoji detected. No seed 42 usage found. All return statements in context are legitimate guard/error paths.

---

### Human Verification Required

#### 1. Ruptures Importable in Project Environment

**Test:** Activate the project virtualenv and run `python -c "import ruptures; print(ruptures.__version__)"`
**Expected:** Prints version >= 1.1.10 with no error
**Why human:** System python3 lacks ruptures; the SUMMARY notes uv was unavailable and pip was used instead; the actual venv state needs confirming

#### 2. Layer Selection Pilot GPU Run

**Test:** `python scripts/layer_selection_pilot.py --config configs/phase3.yaml`
**Expected:** Produces `$DATA_DIR/results/phase3/pilot_layer_aurocs.json` with 36-layer AUROC values and `pilot_auroc_curve.png`
**Why human:** Requires GPU, model weights, and real trace data

#### 3. Probe Training GPU Run

**Test:** `python scripts/train_probes.py --config configs/phase3.yaml`
**Expected:** Saves `{gsm_symbolic,folio,strategyqa}_probes.safetensors` and `pooled_probes.safetensors` under `$DATA_DIR/results/phase3/probes/`; saves `probe_aurocs.json`
**Why human:** Requires GPU and pilot JSON from step 2

#### 4. Transition Detection GPU Run

**Test:** `python scripts/detect_transitions.py --config configs/phase3.yaml`
**Expected:** Per-trace JSON files under `$DATA_DIR/results/phase3/transitions/`; classification distribution printed showing clear/no-signal/noisy-signal fractions per dataset
**Why human:** Requires probe weights from step 3 and GPU

#### 5. Cosine Similarity and Plot Generation

**Test:** Run `compute_cosine_sim.py` then `plot_phase3.py`
**Expected:** `cosine_similarity_stats.json` with Cohen's d distributions; 6 PNG figures; `{dataset}_results.parquet` files with columns including probe_confidence, cosine_similarity_S_t, transition_category, segment_label
**Why human:** Requires full pipeline from steps 2-4

---

### Gaps Summary

No blocking gaps found in code artifacts. All scripts are substantive, properly wired, and implement the required algorithms.

One requirements drift to note: REQ-P3 in REQUIREMENTS.md specifies sustained-K and sliding-window methods, but the implementation uses four change-point detection methods (derivative/PELT/BinSeg/sigmoid). The plans explicitly supersede the original REQ-P3 spec via CONTEXT.md. This is not a gap in implementation -- the delivered approach is more sophisticated -- but the REQUIREMENTS.md text is now stale relative to the implementation.

REQ-P5 is formally dropped (CONTEXT.md Decision 7) and not implemented in Phase 3. REQUIREMENTS.md traceability table marks it "Pending" for Phase 3 -- this should be updated to "Dropped" or moved to a future phase or out-of-scope.

---

_Verified: 2026-02-26_
_Verifier: Claude (gsd-verifier)_
