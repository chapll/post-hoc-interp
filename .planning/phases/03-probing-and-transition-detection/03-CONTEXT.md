# Phase 3 Context: Probing and Transition Detection

**Phase:** 03 -- Probing and Transition Detection
**Created:** 2026-02-26
**Source:** discuss-phase session

---

## Decisions

### 1. Transition Detection Method

**Decision:** Implement multiple detection methods, run all on every trace, report per-method results, pick the best overall method after seeing the data.

**Methods to implement:**
- **Derivative-based**: Compute first derivative of probe confidence curve. Smoothing (Savitzky-Golay) allowed ONLY for derivative computation. Find first point where slope exceeds threshold.
- **Change-point detection (PELT, BinSeg)**: Statistical change-point algorithms from the `ruptures` library. Run on raw probe values (no smoothing).
- **Sigmoid fit**: Fit logistic sigmoid to probe confidence curve. Parameters: t0 (midpoint = transition), k (sharpness). R-squared threshold sweep (0.75, 0.80, 0.85, 0.90, 0.95) -- report how many traces each threshold keeps.

**Plateau definition:** Method-specific. Each detection method defines its own plateau onset boundary. No universal rule imposed across methods.

**Resolution strategy:** Report all methods on every trace. After seeing the data, pick the method that produces the cleanest segmentation across the most traces. Use that method for downstream analysis.

**Raw values only:** All detection methods operate on raw probe confidence values. Smoothing is allowed ONLY for derivative computation in the derivative-based method. No smoothing for PELT/BinSeg or sigmoid fit.

**Probe density:** Probe at every single token position inside the think block. Full resolution for detection.

**Visualization:** Plot detected boundaries as vertical lines on probe confidence plots for visual QA of each method. Per-trace plots with method-specific boundary markers.

### 2. Three-Phase Trace Segmentation

**Decision:** Each trace is segmented into three phases based on probe confidence curve:

1. **Proper reasoning**: Start of trace to onset of transition. Probe confidence is low/rising. Model is genuinely working the problem.
2. **Transition**: From onset through fluctuations until plateau onset. Model is crystallizing the answer.
3. **Post-hoc reasoning (performative)**: Plateau region where probe confidence stays high until end of trace. Model already knows the answer, just generating tokens.

**No plateau = no post-hoc reasoning** for that trace.

### 3. Three-Way Trace Classification

**Decision:** Classify traces into three categories based on detection results:

1. **Clear transition**: Probe detects clean inflection + plateau. Trace has all three phases. Used in downstream analysis.
2. **No-signal (genuinely hard)**: Probe confidence consistently low throughout (never crossed ~70%). Model may have genuinely struggled. Interesting qualitative examples for paper.
3. **Noisy-signal (detection limitation)**: Probe confidence noisy/oscillating near threshold without stabilizing. Transition might exist but isn't cleanly detectable. May respond to different probe layer or wider smoothing.

**Indeterminate traces** (no-signal + noisy-signal) are excluded from downstream analysis requiring T_i (steering, differential state analysis). The indeterminate rate is reported prominently as a meaningful metric.

### 4. Transition Detection as Standalone Module

**Decision:** Transition detection logic lives in `src/transition.py` as a standalone library module. It takes arrays of (position, confidence/probability) as input and returns segmentation results. Both probing scripts and force-answer experiment scripts call into it. Single source of truth for detection logic.

**Interface:** Input is any curve of values vs positions (probe confidence, answer probability, etc.). Output is segment boundaries and classification. Method-agnostic about the source of the curve.

### 5. Activation Data Source -- Two-Stage Pipeline

**Decision:** Do NOT use stored Phase 2 activations. Always use fresh nnsight forward passes for consistency.

**Stage 1 (Layer Selection Pilot):**
- Forward pass ~50-100 traces (stratified across datasets and correctness)
- Extract all 36 layers
- Train probes, compute AUROC per layer
- Identify top-N layers
- Discard full 36-layer activations after selection

**Stage 2 (Full Extraction):**
- Forward pass the entire dataset
- Extract and store activations only for the top-N layers identified in Stage 1
- These stored activations used for all downstream analysis

### 6. Probing Protocol

**Pre-Handoff Window:** Fixed at 3 tokens before final `</think>`. Window size ablation deferred to Phase 6.

**Probe scope:** Train BOTH per-dataset probes (108 total: 3 datasets x 36 layers) AND pooled probes (36 total: all datasets combined). Compare AUROC. If pooled matches per-dataset, the signal is universal. If per-dataset is better, the signal is task-specific. This comparison is itself a finding.

**Selectivity controls (REQ-P2):** 10 random permutations per layer. Report mean and std of random AUROC. Provides solid null distribution for p-value computation. Warning if random AUROC > 0.60.

**Training hyperparameters:** Sensible defaults -- lr=1e-3, 100 epochs, early stopping on validation loss (patience=10), Adam with L2 regularization (weight decay). Adjust only if probes underfit/overfit. No grid search in Phase 3 (defer to Phase 6 ablations).

**Cross-validation:** 5-fold stratified CV on the Pre-Handoff Window training data. Report AUROC per fold and mean.

**Probe label:** `ground_truth` (abstract validity: Yes/No), NOT the output token (Rho/Tau). Swapped-label protocol ensures zero correlation between output token and validity label.

### 7. REQ-P5 (Cox 2x2 Classification) -- DROPPED

**Decision:** Drop entirely from Phase 3 and the project. The transition detection curves already tell the story -- if probe confidence plateaus at 0.95+ after the transition point, the model knows the answer. Classifying the TYPE of post-hoc reasoning (Sound Reasoning, Non-entailment, Confabulation, Hallucination) adds subjective LLM-based text evaluation with poorly defined categories. The per-trace confidence curves and transition detection are the core evidence.

### 8. Result Format

**Output path:** `$DATA_DIR/results/phase3/`
- `probes/` -- probe weights (safetensors, all 36 layers per dataset + pooled)
- `transitions/` -- transition detection results
- Main parquet files at top level

**Format:** Parquet per dataset. One file per dataset with columns:
- trace_id, problem_id, dataset, token_position, layer
- probe_confidence (per token per layer)
- cosine_similarity_S_t (per token per layer)
- transition_point (per trace per method)
- transition_category (clear / no-signal / noisy-signal)
- segment_label (reasoning / transition / post-hoc)

**Probe weights:** Save all 36 layers per dataset as safetensors. Trivial disk cost, needed for Phase 5 steering (probe weight vector = steering direction).

**Cosine similarity (REQ-P4):** Same parquet, extra columns. No separate files.

---

## Deferred Ideas

- Window size ablation (1, 3, 5, 10 tokens) -- Phase 6
- Probe hyperparameter grid search -- Phase 6
- Cox 2x2 classification -- Dropped (see Decision 7)

---

## Per-Trace Principles (Non-Negotiable)

- Transition point T_i is per-trace, not per-problem or per-dataset
- NEVER average transition points or confidence curves across traces
- Aggregation is over distributions of T_i values, not over curves
- No mocks, no placeholders, no fallbacks -- every component is real
- No emoji, no seed 42

---

*Context captured: 2026-02-26*
