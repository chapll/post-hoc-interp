# Requirements

**Project:** post-hoc-interp (Mechanistic Detection of Performative Reasoning in Qwen 3.0)
**Created:** 2026-02-04
**Source:** PROJECT.md, research/, user requirements, Cox (2025), Arcuschin et al. (2025)

---

## Foundation

### REQ-F1: Model Loading and nnsight Integration

Load Qwen3-8B (`Qwen/Qwen3-8B`) via HuggingFace transformers wrapped with nnsight `LanguageModel`. Support BF16 inference on a single A100 80GB. Validate that nnsight can access residual stream activations at all 36 layers with correct shapes `[batch, seq_len, 4096]`. Validate thinking token boundary parsing (token ID 151668 for `</think>`).

**Acceptance:**
- Model loads in BF16, ~16.4 GB VRAM
- `model.model.layers[i].output[0]` returns tensor of shape `[batch, seq_len, 4096]`
- Thinking tokens correctly identified by token ID
- Single forward pass completes without OOM on A100 80GB

### REQ-F2: Dataset Loading with Binary Verification Conversion

Load three datasets: GSM-Symbolic (verification subset), FOLIO (binary subset), StrategyQA. Convert each to base problem dicts with ground truth labels from the dataset. Prompts are NOT built by loaders -- use `lib/prompts.py` with swapped-label protocol (Rho/Tau tokens) to construct verification prompts. Produce 300 problems per dataset. Ensure balanced Yes/No labels.

**Acceptance:**
- 300 problems per dataset (900 total)
- Each problem has: problem_id, question, ground_truth (Yes/No), dataset_name, original_answer, presented_answer, premises (FOLIO only)
- Yes/No label balance within 45-55% per dataset
- GSM-Symbolic uses fresh instances with new numerical values (contamination mitigation)
- FOLIO includes premises field for prompt reconstruction
- Prompts built via `build_verification_prompt()` with configurable label protocol

### REQ-F3: Configuration System

OmegaConf-based YAML configuration (NOT Hydra). Support base config merging with script-specific configs and CLI overrides via `OmegaConf.from_dotlist()`. Configuration covers: model params, paths, trace generation params, probing params, steering params, GPU selection.

**Acceptance:**
- `configs/base.yaml` with shared defaults (model name, dtype, paths, seed != 42)
- Script-specific YAML configs for each experiment script
- CLI overrides work: `python script.py key=value`
- Path interpolation works: `${paths.data_dir}/traces`

### REQ-F4: Multi-GPU Infrastructure

Support specifying exact GPU IDs (e.g., `[1, 7]`) for trace generation. Implement data-parallel inference via `torch.multiprocessing.spawn` -- one model copy per GPU, independent problem subsets. NOT FSDP/DDP. Respect `CUDA_VISIBLE_DEVICES` when gpu_ids not specified.

**Acceptance:**
- Config field `gpu.ids: [0, 1, 2, 3]` selects specific GPUs
- Each GPU loads independent model copy
- Problems split evenly across GPUs
- Works with 1 GPU (no multiprocessing overhead) and 4+ GPUs

### REQ-F5: Trace Storage System

Safetensors-based activation storage with per-problem directory structure. Store BF16 activations (model's native precision). Each trace file contains: residual stream at configurable layer subset `[seq_len, 4096]`, last-token activations at all 36 layers `[36, 4096]`, generated token IDs `[seq_len]`, top-50 logits per position (values + indices) `[seq_len, 50]`, pre-computed entropy scalar per position `[seq_len]`. Metadata JSON per problem with answer correctness, generation params, token count, thinking block boundaries. Config option to also save attention weights (off by default).

**Acceptance:**
- Directory structure: `data/traces/{dataset}/{problem_id}/trace_{NN}.safetensors`
- `metadata.json` per problem directory
- Safetensors files load with `safetensors.torch.load_file()`
- BF16 storage (model's native precision)
- Configurable layer subset (default: `[0, 4, 8, 12, 16, 20, 24, 28, 32, 35]`)
- Top-50 logits (values + indices) and entropy scalar stored per position
- Attention weights: configurable flag (default off), recomputed on demand when needed

### REQ-F6: Utility Functions

Seed management (never seed 42), device selection helpers, plain-text logging (no emoji, no decorative formatting). All print output is plain text.

**Acceptance:**
- `set_seed(seed)` sets torch, numpy, random seeds
- Default seed is not 42
- No emoji anywhere in output
- No decorative banners or fancy formatting

---

## Trace Generation

### REQ-T1: Two-Pass Trace Generation

Generate reasoning traces with Qwen3-8B in thinking mode (`enable_thinking=True`). Use two-pass approach: (1) generate full response text with standard HuggingFace `model.generate()`, (2) run single nnsight `model.trace()` forward pass over complete (prompt + response) to collect activations. Sampling params: Temperature=0.6, TopP=0.95, TopK=20, MinP=0. DO NOT use greedy decoding.

**Acceptance:**
- Thinking mode produces `<think>...</think>` blocks
- Activations collected in single forward pass (not during generation)
- Sampling parameters match Qwen3 thinking mode requirements
- Activations moved to CPU immediately (`.cpu().save()`) to prevent OOM
- Batch size configurable (default 1-2 for activation collection)

### REQ-T2: Configurable Trace Count

Support generating N traces per problem. Default 5 traces for probing, 20 traces for differential analysis. Both configurable from YAML.

**Acceptance:**
- `trace.num_traces: 5` generates 5 traces per problem
- `trace.num_traces: 20` generates 20 traces per problem
- Each trace uses a different random seed (derived from problem_id + trace_index)
- Seeds are deterministic given problem_id and trace_index

### REQ-T3: Contamination Mitigation for GSM-Symbolic

Generate fresh GSM-Symbolic problem instances with new numerical values not present in published datasets. Run contamination diagnostic: prompt Qwen3-8B with original vs. modified problems and compare accuracy.

**Acceptance:**
- Fresh numerical values generated for GSM-Symbolic templates
- Contamination diagnostic compares accuracy on original vs. fresh instances
- Results logged and saved for paper reporting

### REQ-T4: Thinking Token Boundary Parsing and Structural Filtering

Parse `<think>` and `</think>` boundaries at the token ID level (not string matching). Handle edge cases: premature `</think>`, missing `</think>` (max length reached), multiple `<think>` blocks. Apply structural filter: primarily keep only traces with a single contiguous `<think>...</think>` block. Track and report number of filtered traces per dataset. If >20% have multiple blocks, fallback to analyzing only the final contiguous block.

**Acceptance:**
- Token boundaries identified by token IDs
- Edge cases handled with explicit categorization (single_block, multiple_blocks, premature_close, missing_close)
- Structural filter: single-block traces are primary analysis set
- Multi-block traces counted and reported (not silently discarded)
- If multi-block rate > 20%, fallback to final-contiguous-block analysis
- Per-token annotation: is_thinking_token (bool), block_index (int)
- For multi-block traces kept for analysis: probing restricted to inside valid `<think>` regions only

---

## Probing and Transition Detection

### REQ-P1: Linear Probes on Residual Stream ("Train-on-End, Test-on-Middle")

Train a single logistic regression probe (PyTorch `nn.Linear(4096, 2)`) per layer on the "Pre-Handoff Window" -- the 3 natural language tokens immediately preceding the FINAL `</think>` token of all correct traces. This learns the universal answer state. Apply the fixed probe to every intermediate token position inside `<think>` blocks to detect transition point T_i. L2 regularization. 5-fold stratified cross-validation on the training data. Report AUROC per layer per dataset.

**Probing protocol:**
- Context (input): feed full sequence including `<think>` and `</think>` tokens in forward pass
- Readout (analysis): SKIP probing the `<think>` token itself (geometric outlier, k=0). Start at first natural language token (k=1).
- Probe ONLY inside `<think>` blocks. Never probe inter-block text (distributional shift).
- Training data: Pre-Handoff Window of correct traces only. The probe learns what "definitively knowing the answer" looks like.
- Inference: apply fixed probe to every eligible token position to produce confidence curve.
- **Label note:** Probe label is `ground_truth` (abstract validity: Yes/No), NOT the output token (Rho/Tau). The swapped-label protocol ensures zero correlation between output token and validity label, forcing probes to find the abstract signal.

**Acceptance:**
- One probe per layer (36 probes per dataset), each trained on Pre-Handoff Window
- Training uses Adam optimizer on GPU
- 5-fold stratified CV with AUROC metric on Pre-Handoff Window data
- L2 regularization (weight decay) applied
- Layer-wise AUROC curve produced per dataset
- Confidence curve produced per trace (probe applied at each token inside `<think>` blocks)

### REQ-P2: Selectivity Controls

Train probes on randomly permuted labels as a control. If random-label AUROC exceeds 0.60, flag real probes as suspect. This is non-negotiable for paper validity.

**Acceptance:**
- Random-label probes trained with same procedure as real probes
- Random-label AUROC reported alongside real AUROC
- Warning logged if random-label AUROC > 0.60
- Results saved for paper reporting

### REQ-P3: Transition Point Detection

Identify transition point T_i where probe confidence consistently exceeds 95% for K=5 consecutive tokens (sustained-K method). Also compute sliding-window method (W=10, threshold 0.90) as sensitivity check. Categorize transitions: clean, gradual, absent, immediate.

**Acceptance:**
- Primary method: sustained-K=5 at threshold 0.95
- Secondary method: sliding-window W=10 at threshold 0.90
- Transition categories: clean (< 0.6 to > 0.95 within 50 tokens), gradual (ramp over > 200 tokens), absent (never exceeds 0.85), immediate (> 0.9 from first thinking token)
- Per-trace output: transition_point (token index), transition_category, confidence_curve
- Statistics: fraction of traces in each category per dataset

### REQ-P4: Cosine Similarity Dynamics

Compute step-wise cosine similarity S_t = cosine_similarity(h_t, h_{t-1}) at each token position. Test hypothesis: pre-transition shows high drift (low S_t), post-transition shows high stability (high S_t). Compute at best-probe-AUROC layer plus 2-3 additional layers. Statistical test: two-sample t-test or Mann-Whitney U for pre-T vs post-T, report Cohen's d.

**Acceptance:**
- S_t computed per token position per layer
- Statistical comparison of pre-T vs post-T distributions
- Effect size (Cohen's d) reported
- Results at multiple layers (not just best probe layer)

### REQ-P5: Cox 2x2 Classification Framework

Classify post-transition CoT into four categories based on (probe prediction correct/incorrect) x (CoT supports answer/contradicts answer):
- Sound Reasoning: probe correct, CoT supports answer
- Non-entailment: probe correct, CoT does not support answer
- Confabulation: probe incorrect, CoT supports answer
- Hallucination: probe incorrect, CoT contradicts answer

Use LLM-based evaluation to determine whether CoT logically supports the final answer.

**Acceptance:**
- LLM evaluator classifies each post-transition CoT segment
- 2x2 distribution computed per dataset
- Results saved with per-trace annotations
- Distribution plots generated

---

## Causal Validation

### REQ-V1: Differential State Analysis and Performativity Vector

For each problem, generate N=20 correct traces. Cluster into "Concise" (traces that terminate shortly after T_i, n approximately T_i) and "Verbose" (traces that continue generating long after T_i, n >> T_i). Align at transition points T_i. Extract performativity vector: v_perf = mean(h_t^{verbose} at t > T_i) - mean(h_t^{concise} at their T_i). This isolates the features responsible for continued generation despite possessing the answer. Validate by projecting traces onto v_perf and checking separation.

**Acceptance:**
- N=20 correct traces per problem used for clustering
- Traces clustered into Concise vs Verbose groups based on post-T length
- Alignment at detected transition points
- Performativity vector extracted via Difference in Means on correct traces only
- Projection separates Concise from Verbose traces (quantified)
- Vector saved for use in steering experiments

### REQ-V2: Causal Steering with Performativity Vector

Inject performativity vector into concise traces (+v to induce unnecessary generation). Inject inverse into verbose traces (-v to induce early termination). Alpha sweep from -8 to +8 (following Cox). Use nnsight hooks to modify residual stream at target layers during generation.

**Acceptance:**
- Steering via nnsight `model.generate()` with hooks
- Alpha sweep: [-8, -6, -4, -2, -1, 0, 1, 2, 4, 6, 8]
- Two directions: +v (induce verbosity) and -v (induce early termination)
- Per-alpha metrics: behavior change rate, answer preservation rate, coherence rate
- Random-vector control (same norm, random direction) included
- Results saved per dataset

### REQ-V3: Sham Controls for Intervention Validity

Patch activations from a trace with the same answer (sham intervention). If the model's output changes, the intervention is too destructive. Track fraction of coherent vs incoherent outcomes. If >30% incoherent, flag intervention as too coarse.

**Acceptance:**
- Sham interventions run alongside real interventions
- Coherence rate tracked and reported
- Warning if incoherent rate > 30%

### REQ-V4: Cross-Dataset Performativity Vector Analysis

Extract v_perf separately for each dataset. Compute pairwise cosine similarity between vectors. Cross-steering experiment: use v_perf from dataset A to steer traces from dataset B. Tests whether performativity is model-intrinsic.

**Acceptance:**
- 3 dataset-specific performativity vectors
- 3 pairwise cosine similarities
- 9 cross-steering experiments (3 source x 3 target)
- Results table: within-dataset vs cross-dataset steering success rates

### REQ-V5: IPHR Reversed Question Pairs (Arcuschin Method)

For a subset of problems per dataset, create reversed versions where the correct answer is flipped. Compare CoT patterns between original and reversed versions. Detect implicit bias: if model shows bias toward original answer even in reversed version, this indicates unfaithful reasoning.

**Acceptance:**
- Reversed versions created for at least 50 problems per dataset
- Original and reversed traces generated with same conditions
- Bias metric: accuracy difference between original and reversed
- Per-problem bias flag
- Results saved for paper reporting

### REQ-V6: Unfaithfulness Pattern Classification (Arcuschin Method)

Classify detected unfaithful CoT segments into specific patterns:
- Fact Manipulation: CoT introduces incorrect facts to justify the answer
- Argument Switching: CoT switches reasoning strategy mid-chain to arrive at predetermined answer
- Answer Flipping: CoT reasoning points to one answer but the model outputs a different one

Use LLM-based classification on post-transition CoT segments.

**Acceptance:**
- LLM classifier identifies unfaithfulness patterns
- Per-trace pattern annotation
- Distribution of patterns per dataset
- Cross-reference with transition point locations

---

## Baselines

### REQ-B1: Entropy-Based Transition Detection

Compute entropy of next-token distribution at each position. Identify transition as the point where entropy drops below a threshold. Compare with probe-based transition points.

**Acceptance:**
- Entropy computed from logits at each token position
- Entropy-based transition point identified
- Correlation with probe-based transition point reported
- Single forward pass (no additional model calls)

### REQ-B2: Greedy Pruning Baseline

For a subset of traces, iteratively remove tokens and measure change in log-likelihood of the final answer. Rank tokens by importance. The boundary between important and unimportant tokens is a baseline transition point.

**Acceptance:**
- Implemented for at least 50 traces per dataset (subset due to computational cost)
- Token importance ranking produced
- Baseline transition point compared with probe-based T_i
- Correlation reported

### REQ-B3: Random and Midpoint Baselines

Place T at a uniformly random position (random baseline) and at the midpoint of the reasoning trace (midpoint baseline). Compare with probe-based transition points on all downstream metrics.

**Acceptance:**
- Random baseline: uniform random T per trace
- Midpoint baseline: T = len(reasoning_tokens) / 2
- Downstream metrics computed with these baselines for comparison

---

## Evaluation and Outputs

### REQ-E1: Quantitative Metrics

Compute and report:
- Probe AUROC per layer per dataset (with selectivity control AUROC)
- Token reduction ratio: 1 - T/n per trace (mean, std per dataset)
- Steering success rate: fraction of interventions producing predicted behavioral change (per direction, per alpha)
- Answer preservation rate: fraction of steered traces maintaining correct answer
- Statistical significance: confidence intervals, p-values with Benjamini-Hochberg FDR correction for multi-layer/multi-dataset comparisons
- Effect sizes: Cohen's d for pre-T vs post-T comparisons

**Acceptance:**
- All metrics computed and saved to results directory
- FDR-corrected p-values for all multi-comparison analyses
- Effect sizes alongside p-values
- Summary table per dataset

### REQ-E2: Ablation Studies

Ablations over:
- Probe confidence threshold: 0.85, 0.90, 0.95, 0.99
- Sustained-K parameter: 3, 5, 10, 20
- Steering injection layer: early (10-12), mid (16-20), late (24-28)
- Steering coefficient alpha: full sweep -8 to +8
- Sample size: 3, 5, 10, 20 traces per problem
- Layer choice for probing: show results hold across multiple high-AUROC layers

**Acceptance:**
- Each ablation dimension varied independently
- Results tables and/or plots per ablation
- Saved to results directory

### REQ-E3: Saved Annotated Traces

Each trace stored with per-token annotations:
- Probe confidence at each token position
- Transition point T_i
- Transition category (clean/gradual/absent/immediate)
- Token classification (necessary vs performative)
- Cox 2x2 category for post-transition segment
- Unfaithfulness pattern (if detected)
- Cosine similarity S_t

**Acceptance:**
- Structured JSON or safetensors format per trace
- All fields populated for traces with valid transition points
- Traces without clean transitions still annotated with available fields

### REQ-E4: Paper-Ready Figures

Generate with matplotlib:
1. Probe AUROC across layers (line plot, one line per dataset)
2. Probe confidence curve over token position (representative examples with T marked)
3. Cosine similarity phase plot (S_t over token position with T marked)
4. Transition point distribution (histogram of T/n per dataset)
5. Steering dose-response curves (behavioral change vs alpha, -8 to +8)
6. Cox 2x2 classification breakdown (stacked bar or heatmap per dataset)
7. Attention pattern heatmaps (2-4 examples, pre-T vs post-T)
8. Ablation result plots (threshold sensitivity, layer sensitivity)
9. Cross-dataset steering comparison table/heatmap
10. IPHR bias detection results

**Acceptance:**
- All figures saved as PDF and PNG
- Publication quality (clear labels, legends, axis titles)
- No emoji in any figure
- Consistent style across all figures
- Figures directory: `data/results/figures/`

### REQ-E5: Logit Lens Analysis

Apply unembedding matrix to residual stream at each layer for tokens near T. Track rank and probability of correct answer token across layers. Compare pre-T and post-T positions.

**Acceptance:**
- Logit lens computed at multiple layers for tokens around T
- Correct answer token rank tracked across layers
- Pre-T vs post-T comparison
- Visualization produced

---

## Deferred / Stretch

### REQ-D1: Attention Head Classification (Reasoning vs Consistency)

Classify attention heads as "Reasoning" (new computation) vs "Consistency" (verification, repetition). Analyze whether post-transition tokens are dominated by consistency heads. Zero-ablation validation.

**Status:** Deferred -- high effort, secondary to core argument. Include if time permits after all above requirements are met.

### REQ-D2: Mutual Information Tracking

Compute MI between hidden representation at each token and correct answer using HSIC. Detect MI peaks. Compare with probe-detected transition points.

**Status:** Deferred -- newer method, time-consuming to implement correctly.

### REQ-D3: TAPO (Transition-Aware Policy Optimization)

RL method to optimize reasoning length using transition point information.

**Status:** Out of scope per PROJECT.md.

---

## Traceability

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
| REQ-P1 | Phase 3 | Complete |
| REQ-P2 | Phase 3 | Complete |
| REQ-P3 | Phase 3 | Complete |
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
| REQ-D1 | Deferred | - |
| REQ-D2 | Deferred | - |
| REQ-D3 | Out of scope | - |

---

*Requirements defined: 2026-02-04*
*Total active requirements: 29 (6 foundation + 4 trace + 5 probing + 6 validation + 3 baselines + 5 evaluation)*
*Deferred: 2 (REQ-D1, REQ-D2)*
*Out of scope: 1 (REQ-D3)*
