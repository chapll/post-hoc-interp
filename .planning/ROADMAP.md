# Roadmap: post-hoc-interp

## Overview

This roadmap delivers a mechanistic interpretability pipeline that detects performative reasoning in Qwen3-8B. The pipeline is strictly sequential: foundation and validation first, then trace generation at scale (with swapped-label protocol using Rho/Tau tokens), then probing and transition detection, then differential analysis and vector extraction, then causal steering validation, and finally evaluation with paper-ready outputs. Each phase produces data consumed by the next. The end state is reproducible results with causal mechanistic evidence, annotated traces, ablations, and polished figures ready for the CS224N paper writeup.

**Key Protocol:** The swapped-label protocol (Plan 02-06) eliminates the confound where probes detect surface token mentions ("yes"/"no" in reasoning) rather than abstract validity. Each problem generates Config A (Valid=Rho, Invalid=Tau) and Config B (swapped) variants. Probes train on `ground_truth` (Yes/No = abstract validity), decoupled from output tokens (Rho/Tau).

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Foundation and Validation** - Model loading, datasets, config, storage, and end-to-end nnsight validation on a single problem
- [x] **Phase 2: Trace Generation at Scale** - Two-pass trace generation across ~400 problems with multi-GPU parallelism and contamination mitigation
- [x] **Phase 2.1: Repo Restructure and Shared Infra Stabilization** (INSERTED) - Promote experimental/gorn/ code to main repo structure, eliminate experimental dirs, validate all imports and pipeline stability
- [x] **Phase 2.2: Force-Answer Transition Experiment** (INSERTED) - Truncate thinking at probe points, force model to answer, plot accuracy-vs-position to detect phase transition
- [x] **Phase 2.4: Per-Token Force-Answer Probing** (INSERTED) - High-resolution per-token probing on a single trace to reveal exact transition dynamics
- [ ] **Phase 3: Probing and Transition Detection** - Linear probes, selectivity controls, transition point identification, cosine similarity dynamics, Cox 2x2 framework
- [ ] **Phase 4: Differential Analysis and Vector Extraction** - Performativity vector via Difference in Means, cross-dataset analysis, IPHR reversed pairs, unfaithfulness patterns
- [ ] **Phase 5: Causal Steering Validation** - Steering with performativity vector, sham controls, entropy baseline, greedy pruning baseline, random/midpoint baselines
- [ ] **Phase 6: Evaluation and Paper Outputs** - Quantitative metrics, ablation studies, annotated traces, paper-ready figures, logit lens analysis

## Phase Details

### Phase 1: Foundation and Validation
**Goal**: Researcher can load Qwen3-8B with nnsight, run a single forward pass, capture residual stream activations with correct shapes, parse thinking token boundaries, load all three datasets in binary verification format, and store/retrieve activations via safetensors -- all validated end-to-end on a single problem before any scale-up.
**Depends on**: Nothing (first phase)
**Requirements**: REQ-F1, REQ-F2, REQ-F3, REQ-F4, REQ-F5, REQ-F6
**Key Risks**:
  - nnsight + Qwen3-8B compatibility is inferred, not documented (P6: deadlocks from out-of-order module access)
  - Thinking token boundary parsing edge cases (premature close, missing close, multiple blocks)
  - Activation shape mismatches between expected and actual layer outputs (P13)
**Success Criteria** (what must be TRUE):
  1. Qwen3-8B loads in BF16 via nnsight LanguageModel, and `model.model.layers[i].output` returns tensors of shape `[batch, seq_len, 4096]` for all 36 layers (note: `.output` not `.output[0]` -- discovered during validation)
  2. A single problem from each dataset (GSM-Symbolic, FOLIO, StrategyQA) loads in binary verification format with balanced True/False labels, and each problem has fields: problem_id, question, ground_truth, dataset_name, original_answer
  3. OmegaConf config loads base.yaml merged with script-specific YAML and CLI overrides via `OmegaConf.from_dotlist()`
  4. Multi-GPU infrastructure spawns independent model copies on specified GPU IDs and splits problems evenly, working correctly for both 1 GPU and 4+ GPUs
  5. A trace saved to safetensors loads back with `safetensors.torch.load_file()` containing BF16 residual stream activations at configurable layer subsets, last-token activations at all 36 layers, generated token IDs, top-50 logits, and entropy
**Plans**: 6 plans

Plans:
- [x] 01-01-PLAN.md -- Dependencies, directory scaffold, utils, and config system
- [x] 01-02-PLAN.md -- Dataset loading (GSM-Symbolic, FOLIO, StrategyQA) in binary verification format
- [x] 01-03-PLAN.md -- Model loading with nnsight, trace generation, activation capture, think token parsing
- [x] 01-04-PLAN.md -- Trace storage via safetensors with metadata
- [x] 01-05-PLAN.md -- Multi-GPU infrastructure with mp.spawn
- [x] 01-06-PLAN.md -- End-to-end validation script exercising all components

### Phase 2: Trace Generation at Scale
**Goal**: ~2400 traces (~400 problems x 2 configs x 3 traces each) generated across three datasets with swapped-label protocol (Rho/Tau tokens), activations stored to disk, using multi-GPU parallelism and the two-pass approach, with contamination mitigation for GSM-Symbolic.
**Depends on**: Phase 1
**Requirements**: REQ-T1, REQ-T2, REQ-T3, REQ-T4
**Key Risks**:
  - OOM from activation storage during forward pass (P4: must move activations to CPU immediately via `.cpu().save()`)
  - Dataset contamination for GSM-Symbolic (P2: must generate fresh instances with new numerical values)
  - Stochastic sampling non-reproducibility (P8: deterministic seeds per problem_id + trace_index)
  - Long wall-clock time (~14 hours on 4 GPUs for ~2400 traces)
**Success Criteria** (what must be TRUE):
  1. ~2400 traces exist on disk as JSON files (token IDs + text + metadata) in `data/traces/valid/{dataset}/{problem_id}/trace_NN.json`, with activation safetensors in `data/activations/{dataset}/{problem_id}/trace_NN.safetensors`
  2. Each trace was generated using Qwen3-8B thinking mode (Temperature=0.6, TopP=0.95, TopK=20, MinP=0) with activations collected via teacher-forced re-computation (two-pass approach)
  3. Thinking token boundaries are parsed at the token ID level with edge cases handled (premature close, missing close, multiple blocks), and malformed traces are flagged and quarantined to `data/traces/failed/`
  4. Fresh GSM-Symbolic instances use HuggingFace apple/GSM-Symbolic pre-generated variants, with both fresh and original instances traced for contamination diagnostic
  5. Each trace has a deterministic seed derived from problem_id and trace_index, and traces are reproducible on the same hardware
  6. Trace metadata includes `prompt_config` (A/B), `label_mapping`, `valid_token`, `invalid_token` for swapped-label protocol
**Plans**: 6 plans

Plans:
- [x] 02-01-PLAN.md -- Update lib modules: GSM-Symbolic contamination loader, fault-tolerant mp.Process workers, decoupled storage paths, config files
- [x] 02-02-PLAN.md -- Multi-GPU trace generation script (generate_traces.py)
- [x] 02-03-PLAN.md -- Quality gate script (check_traces.py) with CSV diagnostic report
- [x] 02-04-PLAN.md -- Chunked activation capture script (capture_activations.py) with 36-layer nnsight capture
- [x] 02-05-PLAN.md -- Phase 2 verification script and end-to-end test run
- [x] 02-06-PLAN.md -- Swapped-label prompt protocol (prompts.py, datasets.py update, generate_traces.py expansion)

### Phase 2.1: Repo Restructure and Shared Infra Stabilization (INSERTED)
**Goal**: All shared infrastructure from `src/experimental/gorn/lib/` promoted to `src/infra/`, experiment scripts moved to top-level `scripts/`, `src/experimental/` directory eliminated. All imports updated, full pipeline validated end-to-end, and other researchers (Eric, etc.) can import and use shared modules from the new locations. Tests written for shared infra modules.
**Depends on**: Phase 2
**Requirements**: None (structural improvement, no new research requirements)
**Key Risks**:
  - Import breakage across all existing scripts (every file that imports from `experimental.gorn.lib` must be updated)
  - Existing trace data paths in configs may reference old directory structure
  - Other researchers may have in-progress work that depends on the old paths
  - Existing Hydra/training infra in `src/infra/` may conflict with moved modules
**Success Criteria** (what must be TRUE):
  1. `src/experimental/` directory no longer exists -- all code promoted to main repo structure
  2. Shared library code lives in `src/infra/` (model.py, datasets.py, storage.py, config.py, prompts.py, tokens.py, multiprocess.py, utils.py) alongside existing infra modules
  3. Experiment scripts live in top-level `scripts/` (generate_traces.py, capture_activations.py, check_traces.py, validate.py, verify_phase2.py)
  4. All imports updated and working -- `from src.infra.model import load_model` etc.
  5. Full pipeline test passes: trace generation + activation capture on a small subset using new paths
  6. Other researchers can `from src.infra import ...` from their own scripts
  7. Tests written for shared infra modules (model loading, dataset loading, storage, config, token parsing)
  8. Configs updated to reflect new paths (no references to experimental/gorn/)
**Plans**: 3 plans

Plans:
- [x] 02.1-01-PLAN.md -- Move files to new locations, delete old training code, update pyproject.toml and .gitignore
- [x] 02.1-02-PLAN.md -- Fix all imports (from lib.X -> from src.infra.X), config path resolution, YAML config paths
- [x] 02.1-03-PLAN.md -- Sync dependencies, validate imports, write Tier 1 unit tests for shared infra

### Phase 2.2: Force-Answer Transition Experiment (INSERTED)
**Goal**: Determine whether Qwen3-8B exhibits a phase transition during chain-of-thought reasoning — a point where the model shifts from "not knowing" to "reliably knowing" the correct answer. Test this by truncating the thinking trace at evenly-spaced positions, force-closing the `<think>` block with `</think>\n`, and letting the model freely generate its answer. Record correctness, logit rank of the correct answer token, and probability at each probe point. Plot accuracy-vs-position curves to visualize transition dynamics.
**Depends on**: Phase 2.1 (needs existing traces with thinking blocks and Rho/Tau answer tokens)
**Requirements**: None (exploratory experiment, informs Phase 3 design)
**Key Risks**:
  - Model may not cooperate when thinking is force-closed early (may produce garbage or continue thinking instead of answering) — handle as "no answer" data points
  - KV cache reuse at different truncation points must be implemented correctly to avoid stale state
  - Even spacing may miss the exact transition — future iterations can try denser-near-boundaries or fixed-stride spacing
**Experiment Design**:
  - **Input**: ~50 existing traces (small pilot) from `DATA_DIR/traces/valid/`
  - **Probe points**: ~20 evenly-spaced positions within each trace's thinking block (between `<think>` and `</think>` tokens)
  - **At each probe point t**: Truncate to tokens [0..t], append `</think>\n`, generate answer tokens freely
  - **Record per probe point**: (1) whether the generated answer matches ground truth, (2) logit rank of correct answer token at the generation step, (3) probability of correct answer token
  - **Output**: Per-trace accuracy curve, aggregate accuracy-vs-normalized-position plot, per-dataset breakdown
  - **Future spacing strategies**: denser near boundaries, fixed stride (every N tokens)
**Success Criteria** (what must be TRUE):
  1. Script that loads existing traces, probes at ~20 positions per trace, and generates forced answers using KV-cache-efficient truncation
  2. Results saved as structured JSON per trace: list of (position, correct, logit_rank, probability) tuples
  3. Aggregate accuracy-vs-position plot (normalized position 0-1 on x-axis, accuracy on y-axis) showing whether a transition exists
  4. Per-dataset breakdown (GSM-Symbolic, FOLIO, StrategyQA) to see if transition dynamics vary by task type
  5. Summary statistics: median transition point (if any), sharpness of transition, fraction of traces with clean transitions
**Plans**: 2 plans

Plans:
- [x] 02.2-01-PLAN.md -- Dependencies, config, and core experiment script (trace loading, KV-cache probing, answer extraction, result storage)
- [x] 02.2-02-PLAN.md -- Visualization script (accuracy/probability plots, per-dataset breakdown, summary statistics) and visual verification

### Phase 2.3: Cross-Seed Transition Consistency (INSERTED)
**Goal**: For each of ~41 originally-correct problems, generate 20 traces (different seeds) and probe all of them at ~20 positions to determine whether the model's transition from "not knowing" to "knowing" the correct answer occurs at a consistent normalized position across seeds. Per-problem visualization (mean + individual seed curves) and cross-seed agreement quantification (sharpness, variance, Spearman correlation).
**Depends on**: Phase 2.2 (reuses force-answer probing infrastructure and existing trace data)
**Requirements**: None (exploratory experiment, extends Phase 2.2 findings)
**Key Risks**:
  - Transition point may not be consistent across seeds for some problems (that's data, not a failure)
  - Some problems may not show a clear transition in any seed
  - Multi-think-block traces reduce effective seed count per problem (filter and skip)
**Experiment Design**:
  - **Input**: Same ~51 problems from Phase 2.2 selection, filtered to ~41 originally-correct. Generate 17 new traces (indices 3-19) per problem, reuse 3 existing (indices 0-2) = 20 total per problem.
  - **Probing**: All 20 traces probed at ~20 positions each (~16,400 total probes)
  - **Per-problem output**: Two-panel PNG (accuracy + probability) with 20 faint seed curves and bold mean
  - **Quantification**: Per-problem sharpness (max slope), position-wise variance, pairwise Spearman correlation
**Success Criteria** (what must be TRUE):
  1. 697 new trace files generated (41 problems x 17 seeds) and probed alongside 3 existing traces
  2. Per-problem visualization: 20 seed curves overlaid showing transition consistency (or lack thereof)
  3. Cross-seed transition agreement quantified per problem (sharpness, variance, Spearman)
  4. Summary: sharpness histogram and agreement scatter across all problems
**Plans**: 3 plans

Plans:
- [x] 02.3-01-PLAN.md -- Config and targeted trace generation (17 new traces per ~41 problems)
- [x] 02.3-02-PLAN.md -- Cross-seed probing script (20 seeds x ~20 probes per problem)
- [x] 02.3-03-PLAN.md -- Per-problem visualization, agreement metrics, summary plots, and visual verification

### Phase 2.4: Per-Token Force-Answer Probing (INSERTED)
**Goal**: For a single representative trace (`gsm_symbolic_68_fresh_false_cfga`), probe at every token position within the thinking block (not just ~20 evenly-spaced positions) to get a high-resolution accuracy/probability curve. This reveals the exact token-level dynamics of the transition — whether it's a sharp step function, gradual ramp, or oscillatory — at a granularity that the ~20-point grid from Phases 2.2/2.3 cannot resolve.
**Depends on**: Phase 2.3 (reuses force-answer probing infrastructure)
**Requirements**: None (exploratory experiment, extends Phase 2.2/2.3 findings)
**Key Risks**:
  - Computationally expensive: one probe per thinking token means hundreds of forward passes for a single trace
  - KV cache reuse is critical for feasibility — must process probes in descending order with cache cropping
  - Single trace may not be representative — but this is intentional (deep dive on one example first)
**Experiment Design**:
  - **Input**: Single trace `gsm_symbolic_68_fresh_false_cfga` (a known originally-correct problem with clean transition in Phase 2.2/2.3 results)
  - **Probing**: Every token position within the thinking block (between `<think>` and `</think>`)
  - **At each position t**: Truncate to tokens [0..t], append `</think>\n`, generate answer tokens freely
  - **Record per position**: correctness, logit rank of correct token, probability of correct token
  - **Output**: High-resolution accuracy/probability curve, comparison overlay with the ~20-point curve from Phase 2.2
**Success Criteria** (what must be TRUE):
  1. Every thinking token position probed for the target trace, with results saved as structured JSON
  2. High-resolution accuracy-vs-position and probability-vs-position plots generated
  3. Comparison with the ~20-point curve from Phase 2.2 showing what detail the coarser grid missed
  4. Characterization of transition shape (step function, ramp, oscillatory, etc.) at token-level granularity
**Plans**: 2 plans

Plans:
- [x] 02.4-01-PLAN.md -- Per-token dense probing experiment script and config (824 positions, KV-cache descending order)
- [x] 02.4-02-PLAN.md -- Visualization script (per-token curves, Phase 2.2 overlay, transition characterization)

### Phase 3: Probing and Transition Detection
**Goal**: Researcher has layer-wise probe AUROC curves across all three datasets with selectivity controls, per-trace transition points with quality categorization, cosine similarity dynamics with statistical tests, and Cox 2x2 classification of post-transition CoT segments. Probes train on `ground_truth` (abstract validity: Yes/No), decoupled from output token (Rho/Tau) by the swapped-label protocol.
**Depends on**: Phase 2.4
**Requirements**: REQ-P1, REQ-P2, REQ-P3, REQ-P4, REQ-P5
**Key Risks**:
  - Spurious probe correlations due to high dimension-to-sample ratio (P1: d=4096, n=1500; must run selectivity controls from day one)
  - No clean transition points for a significant fraction of traces (P5: must handle gradual, absent, and immediate transitions gracefully)
  - Class imbalance in verification tasks (P15: mitigated by balanced dataset construction in Phase 2)
**Success Criteria** (what must be TRUE):
  1. Linear probes (nn.Linear(4096, 2)) trained on residual stream activations produce per-layer AUROC curves for each dataset with 5-fold stratified cross-validation, and random-label control probes report AUROC alongside real probes (with warnings if control AUROC > 0.60)
  2. Transition points identified per trace using sustained-K=5 at threshold 0.95 (primary) and sliding-window W=10 at threshold 0.90 (sensitivity check), with each trace categorized as clean, gradual, absent, or immediate
  3. Step-wise cosine similarity S_t computed at the best-probe-AUROC layer plus 2-3 additional layers, with pre-T vs post-T distributions compared using a statistical test (t-test or Mann-Whitney U) and Cohen's d reported
  4. Post-transition CoT segments classified into the Cox 2x2 framework (Sound Reasoning, Non-entailment, Confabulation, Hallucination) using an LLM evaluator, with distributions computed per dataset
  5. All probe weights, transition points, confidence curves, and 2x2 classifications saved to the results directory for downstream use
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD
- [ ] 03-03: TBD

### Phase 4: Differential Analysis and Vector Extraction
**Goal**: Performativity vector extracted via Difference in Means between concise and performative trace clusters, validated by projection separation, with cross-dataset vector comparison, IPHR reversed pairs for bias detection, and unfaithfulness pattern classification.
**Depends on**: Phase 3
**Requirements**: REQ-V1, REQ-V4, REQ-V5, REQ-V6
**Key Risks**:
  - Performativity may not be a clean linear direction (central hypothesis -- if projection does not separate concise from performative traces, steering will not work)
  - May require generating additional traces (up to 20 per problem) if 3 traces per config (6 total per problem) provides insufficient cross-trace variety for clustering
  - Gradual/absent transitions reduce the number of usable traces for alignment at T_i
**Note**: With swapped-label protocol, each problem has 2 configs x 3 traces = 6 traces total. Clustering uses all traces regardless of config since `ground_truth` is the abstract validity label.
**Success Criteria** (what must be TRUE):
  1. Traces clustered into Concise (short post-T, below median) and Performative (long post-T, above median) groups, aligned at detected transition points, with a performativity vector v_perf extracted via Difference in Means and saved for steering
  2. Projection of traces onto v_perf quantifiably separates Concise from Performative groups (measured by a separation metric)
  3. Three dataset-specific performativity vectors extracted, with pairwise cosine similarities computed, and 9 cross-steering experiments (3 source x 3 target) run with within-dataset vs. cross-dataset steering success rates tabled
  4. Reversed question pairs created for at least 50 problems per dataset, with original and reversed traces compared for bias (accuracy difference) and per-problem bias flags saved
  5. Unfaithfulness patterns (Fact Manipulation, Argument Switching, Answer Flipping) classified in post-transition CoT segments using an LLM classifier, with per-trace annotations and per-dataset distributions saved
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD
- [ ] 04-03: TBD

### Phase 5: Causal Steering Validation
**Goal**: Causal evidence that the performativity vector controls post-transition generation behavior, validated with dose-response curves, sham controls, random-vector controls, and compared against entropy-based, greedy pruning, and random/midpoint baselines.
**Depends on**: Phase 4
**Requirements**: REQ-V2, REQ-V3, REQ-B1, REQ-B2, REQ-B3
**Key Risks**:
  - Activation patching destroys model capabilities instead of testing causality (P3: must use mean ablation, sham controls, track coherence rate)
  - Steering magnitude calibration is unprincipled (P7: must sweep alpha from -8 to +8 and report full results, not cherry-pick)
  - Anti-steerability: up to 50% of examples may respond opposite to expected direction
  - Greedy pruning baseline is computationally expensive (subset of 50 traces per dataset)
**Success Criteria** (what must be TRUE):
  1. Steering via nnsight hooks modifies residual stream at target layers during generation, with alpha sweep [-8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8] in both directions (+v to induce verbosity, -v to induce early termination), and per-alpha metrics (behavior change rate, answer preservation rate, coherence rate) saved per dataset
  2. Random-vector control (same norm, random direction) included alongside real steering, demonstrating that the performativity vector direction specifically (not just any perturbation) produces predicted behavioral changes
  3. Sham interventions (patching from same-answer traces) run alongside real interventions, with coherence rates tracked and a warning raised if incoherent rate exceeds 30%
  4. Entropy-based transition detection computed from logits at each token position and correlated with probe-based transition points
  5. Greedy pruning baseline (50 traces per dataset) and random/midpoint baselines computed, with all baseline transition points compared against probe-based T_i on downstream metrics
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD
- [ ] 05-03: TBD

### Phase 6: Evaluation and Paper Outputs
**Goal**: All quantitative metrics computed with statistical rigor, ablation studies completed, every trace stored with full per-token annotations, paper-ready figures generated, and logit lens analysis done -- everything needed for the paper writeup.
**Depends on**: Phase 5
**Requirements**: REQ-E1, REQ-E2, REQ-E3, REQ-E4, REQ-E5
**Key Risks**:
  - Multiple comparisons inflation across layers, heads, and datasets (P10: must apply Benjamini-Hochberg FDR correction)
  - Figure quality iteration takes longer than expected
**Success Criteria** (what must be TRUE):
  1. All quantitative metrics (probe AUROC per layer per dataset, token reduction ratio, steering success rate, answer preservation rate) computed with FDR-corrected p-values and effect sizes (Cohen's d), saved to results directory with summary tables per dataset
  2. Ablation studies completed: probe confidence threshold (0.85, 0.90, 0.95, 0.99), sustained-K (3, 5, 10, 20), steering injection layer (early/mid/late), alpha sweep, sample size (3, 5, 10, 20 traces), and layer choice -- each varied independently with results tables and plots saved
  3. Every trace stored with per-token annotations: probe confidence, transition point T_i, transition category, token classification (necessary vs. performative), Cox 2x2 category, unfaithfulness pattern, cosine similarity S_t -- in structured JSON or safetensors format
  4. All 10 paper figures generated as PDF and PNG in `data/results/figures/` with publication quality (clear labels, legends, axis titles, consistent style, no emoji): probe AUROC across layers, probe confidence curves, cosine similarity phase plots, transition point distributions, steering dose-response curves, Cox 2x2 breakdowns, attention heatmaps, ablation plots, cross-dataset steering comparison, IPHR bias results
  5. Logit lens analysis computed at multiple layers for tokens around T, with correct answer token rank tracked across layers and pre-T vs post-T comparison visualized

**Plans**: TBD

Plans:
- [ ] 06-01: TBD
- [ ] 06-02: TBD
- [ ] 06-03: TBD

## Progress

**Execution Order:**
Phases execute in strict numeric order: 1 -> 2 -> 2.1 -> 2.2 -> 2.3 -> 2.4 -> 3 -> 4 -> 5 -> 6. Each phase depends on the previous phase's output.

| Phase | Plans Complete | Status | Completed |
|-------|---------------|--------|-----------|
| 1. Foundation and Validation | 6/6 | Complete | 2026-02-04 |
| 2. Trace Generation at Scale | 6/6 | Complete | 2026-02-05 |
| 2.1 Repo Restructure and Shared Infra Stabilization | 3/3 | Complete | 2026-02-06 |
| 2.2 Force-Answer Transition Experiment | 2/2 | Complete | 2026-02-23 |
| 2.3 Cross-Seed Transition Consistency | 3/3 | Complete | 2026-02-23 |
| 2.4 Per-Token Force-Answer Probing | 2/2 | Complete | 2026-02-26 |
| 3. Probing and Transition Detection | 0/TBD | Not started | - |
| 4. Differential Analysis and Vector Extraction | 0/TBD | Not started | - |
| 5. Causal Steering Validation | 0/TBD | Not started | - |
| 6. Evaluation and Paper Outputs | 0/TBD | Not started | - |

## Coverage

All 29 active v1 requirements mapped. No orphans. No duplicates.

| Requirement | Phase | Status |
|-------------|-------|--------|
| REQ-F1 | Phase 1 | Complete |
| REQ-F2 | Phase 1 | Complete |
| REQ-F3 | Phase 1 | Complete |
| REQ-F4 | Phase 1 | Complete |
| REQ-F5 | Phase 1 | Complete |
| REQ-F6 | Phase 1 | Complete |
| REQ-T1 | Phase 2 | Complete |
| REQ-T2 | Phase 2 | Complete |
| REQ-T3 | Phase 2 | Complete |
| REQ-T4 | Phase 2 | Complete |
| REQ-P1 | Phase 3 | Pending |
| REQ-P2 | Phase 3 | Pending |
| REQ-P3 | Phase 3 | Pending |
| REQ-P4 | Phase 3 | Pending |
| REQ-P5 | Phase 3 | Pending |
| REQ-V1 | Phase 4 | Pending |
| REQ-V4 | Phase 4 | Pending |
| REQ-V5 | Phase 4 | Pending |
| REQ-V6 | Phase 4 | Pending |
| REQ-V2 | Phase 5 | Pending |
| REQ-V3 | Phase 5 | Pending |
| REQ-B1 | Phase 5 | Pending |
| REQ-B2 | Phase 5 | Pending |
| REQ-B3 | Phase 5 | Pending |
| REQ-E1 | Phase 6 | Pending |
| REQ-E2 | Phase 6 | Pending |
| REQ-E3 | Phase 6 | Pending |
| REQ-E4 | Phase 6 | Pending |
| REQ-E5 | Phase 6 | Pending |

**Deferred (not mapped to any phase):**
- REQ-D1: Attention Head Classification -- deferred, high effort, secondary to core argument
- REQ-D2: Mutual Information Tracking -- deferred, newer method, time-consuming
- REQ-D3: TAPO -- out of scope
