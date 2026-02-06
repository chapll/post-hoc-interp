# Project Research Summary

**Project:** post-hoc-interp (Mechanistic Detection of Performative Reasoning)
**Domain:** Mechanistic interpretability of reasoning LLMs (CS224N final project)
**Researched:** 2026-02-04
**Confidence:** HIGH

## Executive Summary

This project builds a mechanistic interpretability pipeline to detect "performative reasoning" in Qwen3-8B's chain-of-thought traces -- reasoning tokens generated after the model has already committed to an answer internally. The pipeline follows a well-established pattern in the 2025/2026 mech interp literature: (1) collect residual stream activations during reasoning via nnsight hooks, (2) train linear probes to detect when the model encodes the answer, identifying a "transition point" T, (3) extract a contrastive "performativity direction" comparing verbose vs. concise post-T traces, and (4) causally validate by steering the model with that direction vector. The core tools -- nnsight for activation access, safetensors for storage, PyTorch linear probes -- are mature and well-documented. Qwen3-8B fits comfortably on a single A100 80GB with room for activation caching.

The recommended approach is a strict sequential pipeline: trace generation, then probing, then differential analysis, then causal steering, then evaluation and figures. Each stage depends on the previous stage's output. The critical path runs through all six phases with no shortcuts. Within phases, work parallelizes well (trace generation across GPUs, probing across layers). The two-pass approach for trace generation (generate text first, then collect activations in a single forward pass) is strongly recommended over trying to collect activations during token-by-token generation, which is complex and error-prone with nnsight.

The three highest risks are: (1) probes succeeding for spurious reasons due to the high dimension-to-sample ratio (4096 dims, ~1500 samples), which would invalidate all downstream analysis -- mitigated by running selectivity controls from day one; (2) dataset contamination in GSM8K/FOLIO making "reasoning" traces unreliable -- mitigated by generating fresh GSM-Symbolic instances; and (3) activation patching destroying model capabilities rather than testing causality -- mitigated by using mean ablation, component-level interventions, and sham controls. A fourth structural risk is that clean transition points may not exist for a significant fraction of traces, requiring the pipeline to handle gradual, absent, and immediate transitions gracefully rather than forcing a binary framework.

## Key Findings

### Recommended Stack

The stack centers on nnsight (>=0.5.15), the standard mechanistic interpretability framework for 2025/2026, which wraps HuggingFace transformers with a tracing API for activation collection and causal interventions. nnsight is the only viable option because TransformerLens does not support Qwen3 and raw hooks require too much boilerplate for multi-invoke patching. The target model is Qwen3-8B (8.2B params, 36 layers, 4096 hidden dim), chosen for its explicit `enable_thinking=True` toggle and Apache 2.0 license. It runs in bf16 on a single A100 with ~58 GB headroom.

**Core technologies:**
- **nnsight (>=0.5.15):** Activation collection, causal interventions, steering -- standard mech interp framework, ICLR 2025
- **Qwen3-8B:** Reasoning model with thinking mode (`<think>...</think>` tokens) -- 36 layers, GQA, SwiGLU, fits single A100
- **transformers (>=5.0.0):** Model loading and tokenization -- required for Qwen3 support
- **safetensors:** Activation storage -- zero-copy loading, lazy access, HF ecosystem standard
- **PyTorch nn.Linear:** Linear probes -- GPU-native, no sklearn dependency needed
- **scikit-learn:** Backup probing option for rapid iteration -- LogisticRegression with L2 regularization
- **OmegaConf + YAML:** Configuration -- avoids Hydra's cwd-changing behavior that conflicts with pipeline structure

**Critical version requirements:**
- transformers >= 5.0.0 (Qwen3 support requires >= 4.51.0)
- nnsight >= 0.5.15 (direct execution, no proxy overhead)
- Python 3.13 (matches existing repo)

**New dependencies to add:** nnsight, transformers (upgrade), accelerate, datasets, scikit-learn, safetensors, plotly

### Expected Features

**Must have (table stakes):**
- **TS-1: Linear probing for transition point detection** -- Foundation of all claims. LogisticRegression probes on residual stream, 95% confidence threshold, sustained-K=5 method. ~2-3 days.
- **TS-2: Causal steering validation** -- The differentiating contribution over prior work. Contrastive Activation Addition with systematic magnitude sweep. ~3-4 days.
- **TS-3: Quantitative evaluation metrics** -- Probe AUROC, token reduction ratio, steering success rate, answer preservation rate. ~0.5 days.
- **TS-4: Cosine similarity dynamics** -- Intrinsic evidence of phase transitions, complementary to probes. ~1 day.
- **TS-5: Differential state analysis** -- Performativity vector extraction via Difference in Means. Required for TS-2. ~2 days.
- **TS-6: Baseline comparisons** -- Entropy-based detection, greedy pruning, random baseline. ~2 days.
- **TS-7: Paper-ready figures** -- 6 specific figure types standard in mech interp papers. ~1-2 days.

**Should have (differentiators):**
- **D-3: Cross-dataset performativity vector analysis** -- Tests whether performativity is model-intrinsic. Low marginal effort (0.5 days).
- **D-4: Logit lens analysis** -- Quick complementary visualization (0.5 days).
- **D-5: Ablation studies** -- Threshold sensitivity, layer choice, steering coefficient sweeps (1-2 days).
- **D-6: Token-level categorization** -- Structured annotations for qualitative examples (0.5 days).

**Defer (v2+):**
- **D-1: Mutual information tracking** -- HSIC-based MI is time-consuming to implement correctly (2-3 days).
- **D-2: Attention head classification** -- High effort (3-4 days), secondary to core argument.

**Critical path:** Trace Generation --> TS-1 --> TS-5 --> TS-2 --> TS-3 + TS-7

**Total effort for table stakes:** ~12-14 days. With differentiators: ~22-28 days.

**Anti-features (explicitly do NOT build):**
- Complex probe architectures (3+ layers) -- confounds interpretation
- Global confidence measures -- obscures local dynamics
- Fine-tuning the model -- analysis-only project
- SAE training -- tangential and expensive
- Interactive demos -- time sink, irrelevant to paper quality

### Architecture Approach

The pipeline lives in `src/experimental/gorn/` with a clean separation between reusable library code (`lib/`) and experiment scripts (`experiments/`). Configuration uses OmegaConf with YAML files, deliberately NOT extending the repo's Hydra setup (Hydra fights multi-script pipelines with fixed data paths). Multi-GPU uses simple process-level data parallelism (one model copy per GPU, independent problem subsets), NOT FSDP/DDP which would add communication overhead for zero benefit during inference. The two-pass trace generation approach (generate text first, then collect activations in a single forward pass) avoids the complexity of nnsight's iterative generation API.

**Major components:**
1. **lib/model.py** -- Qwen3-8B loading with nnsight wrapping; single function, forwards kwargs to HuggingFace
2. **lib/trace_store.py** -- Safetensors-based activation read/write; FP16 storage, per-problem directory structure
3. **experiments/trace_gen.py** -- Multi-GPU trace generation; two-pass approach, configurable layer subset
4. **lib/probe.py** -- PyTorch nn.Linear(4096, 2) with cross-entropy; 5-fold stratified CV
5. **experiments/steering.py** -- nnsight-based activation patching and steering; magnitude sweep infrastructure
6. **experiments/figures.py** -- Pure matplotlib, reads from results directory, no GPU needed

**Storage architecture:**
- Last-token activations at all 36 layers: ~1.6 GB total (trivial) -- always stored
- Full-sequence activations at 10 selected layers: ~250 GB for 5 traces, ~1 TB for 20 traces
- Recommendation: Start with 5 traces for probing, only generate 20 if results warrant differential analysis

### Critical Pitfalls

1. **Spurious probe correlations (P1)** -- With d=4096 and n=1500, random labels can become linearly separable. Run selectivity controls (random-label probes) from day one. If random-label AUROC exceeds 0.55-0.60, real probes are suspect. Use L2 regularization. This is non-negotiable for the paper.

2. **Dataset contamination (P2)** -- Qwen family models show evidence of GSM8K memorization (BenBench). Generate fresh GSM-Symbolic instances with new numerical values. Run contamination diagnostics before collecting traces. Report results in the paper.

3. **Activation patching destroys capabilities (P3)** -- Patching full residual stream vectors is too coarse. Use mean ablation (not zero ablation), patch individual components (attention/MLP outputs), run sham controls (patch from same-answer trace), and track fraction of coherent vs. incoherent outcomes (>30% incoherent = intervention too destructive).

4. **OOM during activation storage (P4)** -- Saving all 36 layers for all tokens exhausts memory. Move activations to CPU immediately (`.cpu().save()`), store only configurable layer subsets, process traces sequentially (batch_size=1-2).

5. **No clean transition point (P5)** -- The binary pre-T/post-T framework may not fit all traces. Build the pipeline to categorize transitions (clean, gradual, absent, immediate) and report results per category. Use multiple detection methods (probe confidence, cosine similarity, entropy) and check convergence.

## Implications for Roadmap

### Phase 1: Foundation and Validation

**Rationale:** Everything depends on nnsight working with Qwen3-8B and activations being correctly captured. This must be validated before any scale-up. The STACK.md validation checklist has 6 items that are genuinely unknown (nnsight + Qwen3 compatibility is inferred, not documented).

**Delivers:** Working model loading, dataset preparation, activation capture for a single problem, end-to-end data flow validation.

**Addresses:** lib/model.py, lib/datasets.py, lib/trace_store.py, lib/utils.py, configs/base.yaml

**Avoids:** P6 (nnsight deadlocks -- validate layer access ordering), P14 (thinking token boundaries -- validate token ID parsing), P11 (forgotten .save() -- add shape assertions)

**Key validation:** Run nnsight trace on one Qwen3-8B problem, verify residual stream shapes are [batch, seq_len, 4096], verify thinking tokens are correctly identified by token ID.

### Phase 2: Trace Generation at Scale

**Rationale:** All analysis depends on having activation traces. This is the most time-consuming phase in wall-clock time (~6 hours for 300 problems on 4 GPUs with 5 traces each). Must get memory management right from the start.

**Delivers:** Complete activation dataset: 1500 traces (300 problems x 5 traces) with last-token activations at all layers and full-sequence activations at 10 selected layers. Fresh GSM-Symbolic instances.

**Addresses:** experiments/trace_gen.py, configs/trace_gen.yaml, dataset contamination mitigation

**Avoids:** P4 (OOM -- CPU offload, selective layer saving), P2 (contamination -- fresh GSM-Symbolic instances), P8 (non-reproducibility -- seed management per problem)

### Phase 3: Probing and Transition Detection

**Rationale:** Probing is the foundation of all claims. Must include selectivity controls from day one. Also the phase where the "no clean transition" risk materializes -- the pipeline must handle it gracefully.

**Delivers:** Layer-wise AUROC curves, transition point detection with quality categorization, cosine similarity dynamics (TS-4), selectivity control results.

**Addresses:** TS-1 (probe detection), TS-4 (cosine similarity), lib/probe.py, lib/metrics.py, experiments/train_probes.py, experiments/transition_detect.py

**Avoids:** P1 (spurious probes -- selectivity controls mandatory), P5 (no clean transition -- multiple detection methods, categorization), P15 (class imbalance -- balanced datasets, AUROC metric)

### Phase 4: Differential Analysis and Vector Extraction

**Rationale:** Depends on transition points from Phase 3. May require generating additional traces (up to 20 per problem) if probing results warrant deeper cross-trace analysis. Produces the performativity vector needed for steering.

**Delivers:** Performativity direction vector, cross-trace consistency scores, concise vs. verbose trace clustering.

**Addresses:** TS-5 (differential state analysis), D-3 (cross-dataset vector analysis if time permits), experiments/differential.py

**Avoids:** P5 (gradual transitions -- align at detected transition points, handle varying quality)

### Phase 5: Causal Steering Validation

**Rationale:** This is what makes the paper novel beyond prior work (Cox, Zhang et al.). Must be designed carefully to avoid destroying model capabilities. The magnitude sweep infrastructure is essential -- cherry-picking one alpha value is misleading.

**Delivers:** Steering success rates, answer preservation rates, random-vector controls, dose-response curves.

**Addresses:** TS-2 (causal steering), TS-6 (baselines -- entropy-based, random, optionally greedy pruning), experiments/steering.py, configs/steering.yaml

**Avoids:** P3 (destructive patching -- mean ablation, component-level, sham controls), P7 (unprincipled magnitude -- systematic sweep, report full results), P9 (correlation-causation -- probes correlational, steering causal, structure argument accordingly)

### Phase 6: Evaluation and Paper Outputs

**Rationale:** Aggregates all results into paper-ready metrics and figures. Statistical rigor is critical here -- apply FDR correction for multiple comparisons across layers, heads, and datasets.

**Delivers:** All TS-3 metrics, 6 figure types (TS-7), ablation studies (D-5), annotated traces (D-6), logit lens analysis (D-4) if time permits.

**Addresses:** TS-3 (metrics), TS-7 (figures), D-4, D-5, D-6, experiments/evaluate.py, experiments/figures.py

**Avoids:** P10 (multiple comparisons -- Benjamini-Hochberg FDR, pre-register primary hypotheses, report effect sizes)

### Phase Ordering Rationale

- **Strict sequential dependency:** Each phase produces data consumed by the next. No phase can begin until the previous completes, with the exception of intra-phase parallelism.
- **Risk front-loading:** Phase 1 validates the riskiest assumption (nnsight + Qwen3 compatibility). Phase 3 surfaces the second-biggest risk (spurious probes, absent transitions) before investing in steering.
- **Contamination addressed early:** Fresh dataset generation happens in Phase 2, before any analysis depends on the data.
- **Causal validation late by necessity:** Steering (Phase 5) requires the performativity vector from Phase 4, which requires transition points from Phase 3, which requires traces from Phase 2.
- **Start small, scale up:** Phase 1 uses 1-5 problems. Phase 2 scales to 300. This catches bugs cheaply.

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 1:** nnsight + Qwen3-8B compatibility is not explicitly documented. The `model.generate()` vs `model.trace()` two-pass approach needs validation. Thinking token boundary parsing has edge cases.
- **Phase 5:** Activation patching best practices are actively evolving (Heimersheim 2024, Best Practices paper). Mean ablation vs. zero ablation vs. component-level patching choices affect results. Steering magnitude calibration is an open problem with no principled solution.

Phases with standard patterns (skip deep research):
- **Phase 2:** Multi-GPU inference via process spawning is well-established. Safetensors storage is straightforward.
- **Phase 3:** Linear probing on residual stream activations is the most well-established method in mech interp. Multiple 2025 papers use near-identical approaches on similar models.
- **Phase 6:** Matplotlib figure generation and metric computation are standard.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | nnsight, transformers, safetensors all verified with official docs and version numbers. Only gap: nnsight + Qwen3 specific compatibility (inferred, not documented). |
| Features | HIGH | Feature list grounded in 7+ primary references from 2024-2025 mech interp literature. Cox, Zhang et al., DeepConf, CREST all use near-identical methods. |
| Architecture | HIGH | Memory budget arithmetic verified against Qwen3-8B config.json. Component boundaries follow standard mech interp pipeline structure. Two-pass approach well-reasoned. |
| Pitfalls | HIGH | 15 pitfalls identified across all phases. Top 5 are well-documented in the literature with specific prevention strategies. P5 (no clean transition) is the most uncertain. |

**Overall confidence:** HIGH

### Gaps to Address

- **nnsight + Qwen3-8B compatibility:** No explicit documentation. Must validate with a minimal script in Phase 1 before building the full pipeline. Risk: nnsight's `LanguageModel` may not handle Qwen3's thinking tokens or GQA correctly.
- **nnsight `.generate()` with long thinking outputs (>4K tokens):** Memory footprint during trace collection with all 36 layers saved is estimated but not measured. Validate in Phase 1.
- **PNS-based optimization baseline:** Referenced in PROJECT.md but not found in the mech interp literature. May refer to Pearl's Probability of Necessity and Sufficiency. Needs clarification from the project team.
- **Transition point existence:** Whether Qwen3-8B with explicit thinking mode produces clean transitions is genuinely unknown. Prior work (Cox) used Gemma-2 without thinking mode. The thinking mode may produce different transition dynamics.
- **Performativity as a linear direction:** The central hypothesis (that performativity is linearly encoded) may be wrong. If the performativity vector has low cosine similarity across datasets or fails to separate concise/verbose traces, the steering approach will not work. This would be a negative result worth reporting.
- **Disk storage at scale:** 20 traces per problem at full-sequence resolution for 10 layers approaches 1 TB. Must validate disk quota and I/O bandwidth before Phase 2 scale-up.

## Sources

### Primary (HIGH confidence)
- [Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) -- Architecture parameters, verified
- [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) -- Thinking mode, sampling params
- [nnsight documentation](https://nnsight.net/) -- Core API patterns, activation patching, generation
- [nnsight PyPI](https://pypi.org/project/nnsight/) -- Version 0.5.15, dependencies
- [Cox (2024)](https://kyle-cox.com/2024/12/26/cot-interp/) -- Post-hoc reasoning detection, probing + steering methodology
- [Zhang et al. (2025)](https://arxiv.org/html/2504.05419v1) -- "Reasoning Models Know When They're Right," linear probes converge to optimal
- [DeepConf (Meta, 2025)](https://arxiv.org/abs/2508.15260) -- Confidence thresholds, entropy-based stopping
- [CREST (NeurIPS 2025)](https://arxiv.org/abs/2512.24574) -- Cognitive reasoning steering, head classification
- [Heimersheim (2024)](https://arxiv.org/pdf/2404.15255) -- Activation patching best practices
- [Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042) -- Intervention design

### Secondary (MEDIUM confidence)
- [MI Peaks (NeurIPS 2025)](https://arxiv.org/abs/2506.02867) -- Thinking tokens as information peaks
- [BenBench](https://github.com/GAIR-NLP/benbench) -- Benchmark contamination evidence for Qwen family
- [GSM-Symbolic (ICLR 2025)](https://arxiv.org/pdf/2410.05229) -- Template-based contamination resistance
- [Representation Engineering Survey](https://janwehner.com/files/representation_engineering.pdf) -- Steering magnitude calibration
- [A Sober Look at Steering Vectors](https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/) -- Anti-steerability, magnitude issues
- [Logical Phase Transitions](https://arxiv.org/html/2601.02902) -- Reasoning phase transition dynamics

### Tertiary (LOW confidence)
- nnsight + Qwen3 specific compatibility -- Inferred from generic HuggingFace model support, not documented
- PNS-based optimization -- Could not find standard method by this name in mech interp literature
- Exact memory footprint during full 36-layer trace collection -- Estimated arithmetic, not measured

---
*Research completed: 2026-02-04*
*Ready for roadmap: yes*
