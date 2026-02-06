# Feature Landscape: Mechanistic Detection of Performative Reasoning

**Domain:** Mechanistic interpretability of reasoning LLM traces (CS224N final project)
**Researched:** 2026-02-04
**Model:** Qwen 3.0 8B with thinking mode
**Datasets:** GSM-Symbolic, FOLIO, StrategyQA (binary verification)

---

## Table Stakes

Features the paper must have to be taken seriously. Missing any of these makes the paper incomplete or unconvincing.

### TS-1: Linear Probing for Transition Point Detection

| Attribute | Detail |
|-----------|--------|
| **Why Expected** | Cox (2024) and Zhang et al. (2025) established this as the standard method for detecting early answer encoding. A paper claiming to detect transition points without probes would be dismissed. |
| **Complexity** | Medium |
| **Dependencies** | Trace generation (activation caching at residual stream positions) |

**Implementation details:**

- Train logistic regression probes on residual stream activations at each token position in the reasoning trace. Binary classification: True vs False (the model's eventual answer).
- Extract post-layer-norm residual stream at the last token position. Cox used all 42 layers of Gemma-2 9B; for Qwen 3.0 8B (32 transformer layers), probe all layers and select the best via cross-validation AUROC.
- **Layer selection:** Best layers are typically middle-to-late (Cox found peak AUROC > 0.9 at certain layers; Zhang et al. found last-layer works well; the "Temporal Predictors" paper found strong signals as early as layer 14-16 in Llama 8B). Expect optimal layers around layers 16-28 for Qwen 3.0 8B, but verify empirically.
- **Probe architecture:** Logistic regression is the standard and preferred choice. Zhang et al. (2025) showed that even when they searched over 2-layer MLPs, optimal configurations converged to linear probes (hidden dim = 0), suggesting correctness information is linearly encoded. Use sklearn LogisticRegression with L2 regularization.
- **Training set:** Cox used 100 examples per dataset. With 300 problems and 5 traces each, 1500 traces per dataset is plenty. Use stratified 5-fold cross-validation.
- **Evaluation metric:** AUROC is the standard. Cox achieved > 0.9 on 3/4 datasets. Zhang et al. achieved 0.82-0.91 across GSM8K/MATH/AIME. Expect similar or better on binary verification tasks (simpler than numerical answer prediction).

**Confidence thresholds and "consistently exceeds" definition:**

- The PROJECT.md specifies 95% probe confidence for transition point T_i. This aligns with DeepConf (Meta, 2025) which found tau=0.95 provides the best tradeoff, cutting tokens by over half with zero accuracy loss.
- **"Consistently exceeds" definition:** The field does not have a single standard, but two approaches are well-supported:
  1. **Sustained-K method:** Probe confidence exceeds threshold for K consecutive tokens. K=5 is a reasonable starting point (robust to single-token fluctuations without being so long it misses real transitions). The MI Peaks paper (Qian et al., NeurIPS 2025) found that meaningful transitions are sparse (0.5-4.8% of tokens), so the sustained window should be short enough to not miss them.
  2. **Sliding window method:** Average probe confidence over a window of W tokens exceeds threshold. W=10-20 tokens smooths noise while remaining responsive. This is analogous to DeepConf's group-based confidence with sliding windows.
- **Recommendation:** Use sustained-K=5 at threshold 0.95 as primary, with sliding-window W=10 at threshold 0.90 as a sensitivity check. Report both.

**Confidence:** HIGH. This is the most well-established method in the space, with multiple 2025 papers using near-identical approaches on similar models (including Qwen3-8B in the "Temporal Predictors" paper).

**Sources:**
- Cox (2024): https://kyle-cox.com/2024/12/26/cot-interp/
- Zhang et al. (2025): https://arxiv.org/html/2504.05419v1
- Temporal Predictors (2025): https://arxiv.org/html/2511.14773
- DeepConf (2025): https://arxiv.org/abs/2508.15260
- MI Peaks (NeurIPS 2025): https://arxiv.org/abs/2506.02867

---

### TS-2: Causal Steering Validation (Activation Patching / Addition)

| Attribute | Detail |
|-----------|--------|
| **Why Expected** | Probes show correlation, not causation. Without causal intervention, the paper only demonstrates "the model encodes the answer early" (already shown by Cox, Zhang, etc.). The causal step is what makes this paper novel. |
| **Complexity** | High |
| **Dependencies** | TS-1 (transition point detection), TS-5 (differential state analysis for performativity vector) |

**Implementation details:**

- **Steering vector:** Computed via Contrastive Activation Addition (CAA) / Difference in Means. Take the mean residual stream activation of "performative" traces (long post-T reasoning) and subtract the mean of "concise" traces (short post-T reasoning), aligned at transition point T_i. This is the standard method (Rimsky et al. MATS 2024, Turner et al. 2023).
- **Injection:** h' = h + alpha * v, where h is the hidden state, v is the steering vector, alpha is a scaling coefficient. Use nnsight hooks to modify the residual stream at target layers during forward pass.
- **Scaling factor alpha:** Typical unnormalized coefficients range 3-15 (Turner et al. on GPT-2-XL). Cox used alpha in {0, +/-1, +/-2, ..., +/-8} on Gemma-2 9B. For Qwen 3.0 8B, start with alpha in {1, 2, 4, 6, 8, 10} and grid search. Beyond +/-8-10, outputs often become incoherent.
- **Layer selection for injection:** The CREST paper (NeurIPS 2025 workshop) found cognitive heads cluster in deeper layers. Activation steering literature (Panickssery et al.) found a transition around 1/3 of the way through. For Qwen 3.0 8B (32 layers), test injection at layers 10-12 (early-mid), 16-20 (mid), and 24-28 (late). Use the layer where the probe achieves highest AUROC as a starting point.
- **Two intervention directions:**
  1. Inject +v into concise traces: Should induce unnecessary post-T generation (model behaves as if it should keep reasoning).
  2. Inject -v into verbose traces: Should induce early termination (model stops performative reasoning).
- **Success metric:** Steering success rate = fraction of traces where intervention changes the generation behavior as predicted (longer/shorter) while maintaining answer correctness. Cox measured "answer-change frequency" across steering coefficients.
- **Critical control:** Inject a random vector of the same norm. This must NOT produce the same behavioral changes. Without this control, the claim that the performativity vector captures a meaningful direction is unsupported.

**Confidence:** HIGH for the method being standard; MEDIUM for success rates (depends on whether performativity is cleanly linearly encoded, which is the central hypothesis).

**Sources:**
- Cox steering: https://kyle-cox.com/2024/12/26/cot-interp/
- CAA/Rimsky: https://www.matsprogram.org/research/recwbuHVD3MEdZyvW
- Activation Addition: https://arxiv.org/html/2308.10248v4
- Steering scaling: https://turntrout.com/gpt2-steering-vectors
- CREST: https://arxiv.org/abs/2512.24574
- nnsight patching tutorial: https://nnsight.net/notebooks/tutorials/causal_mediation_analysis/activation_patching/

---

### TS-3: Quantitative Evaluation Metrics

| Attribute | Detail |
|-----------|--------|
| **Why Expected** | Without rigorous metrics, claims are anecdotal. Reviewers and readers expect standard quantitative evaluation. |
| **Complexity** | Low |
| **Dependencies** | TS-1, TS-2 |

**Required metrics:**

1. **Probe AUROC (per layer, per dataset):** Standard metric for probe quality. Report the best-layer AUROC and the full layer-wise AUROC curve. Cox achieved > 0.9; Zhang et al. achieved 0.82-0.91.

2. **Token reduction ratio:** 1 - T/n, where T is the transition point and n is total reasoning tokens. This measures how much of the reasoning trace is post-transition (and potentially performative). DeepConf achieved 43-85% reduction depending on configuration.

3. **Steering success rate:** Fraction of interventions that produce the predicted behavioral change. Report separately for +v (inducing verbosity) and -v (inducing early termination). Include the random-vector control baseline.

4. **Answer preservation rate:** Fraction of steered traces that maintain the correct answer despite behavioral change. This is critical -- if steering changes the answer, it is not just affecting performativity but actual reasoning.

5. **Statistical significance:** Report confidence intervals or p-values for key comparisons. With 300 problems x 3 datasets = 900 problems, there is statistical power for meaningful results.

**Confidence:** HIGH. These are unambiguous community standards.

---

### TS-4: Residual Stream Cosine Similarity Dynamics

| Attribute | Detail |
|-----------|--------|
| **Why Expected** | Provides intrinsic evidence (no learned probe needed) of phase transitions. Multiple papers use cosine similarity between consecutive hidden states to measure representational change. Complements the probe-based transition detection. |
| **Complexity** | Low-Medium |
| **Dependencies** | Trace generation with cached activations |

**Implementation details:**

- Compute S_t = cosine_similarity(h_t, h_{t-1}) at each token position t in the reasoning trace, where h_t is the residual stream at position t (at a chosen layer).
- **Hypothesis to test:** Pre-transition tokens show low S_t (high drift = active computation, model "searching" for the answer). Post-transition tokens show high S_t (low drift = state maintenance, model generating performative text).
- **Layer choice:** Compute at the same layer where the probe achieves best AUROC, plus at 2-3 other layers (early, mid, late) for a full picture.
- **Normalization:** The LLM-Streamline literature (2024-2025) uses this between layer input/output. Here, apply it between consecutive token positions at the same layer. Both are valid and complementary.
- **Phase transition visualization:** Plot S_t over token position with the probe-detected transition point T marked. Expect a visible shift in the S_t distribution around T.
- **Quantitative test:** Compare mean S_t for pre-T vs post-T tokens using a two-sample t-test or Mann-Whitney U test. Report effect size (Cohen's d).

**Important caveat:** High cosine similarity post-T could also arise from the model entering a "copying" mode (generating common phrases) rather than from state maintenance. Control for this by checking whether post-T tokens have high perplexity (less likely to be simple copies) or low perplexity (more formulaic).

**Confidence:** MEDIUM. The concept is well-supported in the layer-pruning literature, but applying it across token positions within a single layer is less standard. The hypothesis is plausible but not guaranteed.

**Sources:**
- LLM-Streamline: https://arxiv.org/html/2403.19135v3
- Feature Flow: https://arxiv.org/html/2502.03032

---

### TS-5: Differential State Analysis (Performativity Vector Extraction)

| Attribute | Detail |
|-----------|--------|
| **Why Expected** | The core claim of the paper requires isolating a "performativity direction" in activation space. Without this, steering (TS-2) has no direction to steer in. |
| **Complexity** | Medium |
| **Dependencies** | TS-1 (transition points), Trace clustering (concise vs verbose) |

**Implementation details:**

- **Trace clustering:** Group traces into "Concise" (short post-T reasoning, < median post-T length) and "Performative" (long post-T reasoning, > median post-T length). Use 20 traces per problem (as planned) to get sufficient variety.
- **Alignment:** Align traces at their respective transition points T_i. This is critical -- without alignment, the mean difference captures positional artifacts, not the performativity signal.
- **Vector extraction:** v_perf = mean(h_t^{performative}) - mean(h_t^{concise}), averaging over post-T residual stream activations. This is the standard Difference in Means approach.
- **Denoising consideration:** The 2025 literature (FGAA, SAE-RSV) shows that raw DoM vectors contain noise. If results are noisy, consider PCA projection to top-k components (the CREST paper used this successfully). However, for a CS224N project, the basic DoM method is sufficient and interpretable.
- **Validation of the vector:** Before using it for steering, verify that the vector captures meaningful information:
  1. Project traces onto v_perf and check if concise and performative traces separate.
  2. Compute the cosine similarity between v_perf vectors extracted from different datasets. If the performativity phenomenon is model-intrinsic (not dataset-specific), vectors should be somewhat aligned.

**Confidence:** MEDIUM-HIGH. The method is standard; the question is whether "performativity" is a clean linear direction (it may not be).

**Sources:**
- CAA/Mean difference: https://www.emergentmind.com/topics/activation-steering-method
- FGAA denoising: https://arxiv.org/html/2501.09929v3
- CREST PCA approach: https://arxiv.org/abs/2512.24574

---

### TS-6: Baseline Comparisons

| Attribute | Detail |
|-----------|--------|
| **Why Expected** | Reviewers will ask "how does your transition point compare to simpler methods?" Without baselines, there is no evidence the probe-based method adds value. |
| **Complexity** | Medium |
| **Dependencies** | Trace generation |

**Required baselines:**

1. **Greedy pruning:** Iteratively remove tokens and measure post-deletion likelihood. Singh et al. (2025) showed this is a strong baseline that "achieves the strongest performance at matched lengths." Implementation: for each token, compute the change in log-likelihood of the final answer when that token is removed. Rank tokens by importance. The boundary between "important" and "unimportant" tokens is a baseline transition point.
   - **Complexity:** High (requires N forward passes per trace for N tokens, but can be approximated with batching).

2. **Entropy-based detection:** Compute the entropy of the next-token distribution at each position. Expect high entropy pre-T (model uncertain, exploring) and low entropy post-T (model confident, generating predictable text). This is essentially the same signal DeepConf uses for local confidence.
   - **Complexity:** Low (single forward pass, entropy from logits).

3. **Random baseline:** Place T at a uniformly random position in the trace. This establishes the floor.

4. **Midpoint baseline:** Place T at the midpoint of the reasoning trace. Simple but surprisingly competitive if post-T reasoning is roughly 50% of the trace.

**Note on P-ALIGN:** The PROJECT.md mentions P-ALIGN binary search. P-ALIGN (2025) is a prefix alignment method for reasoning distillation, not directly a transition detection method. It finds the minimal sufficient prefix for a student model, which is conceptually related but methodologically different. Including it as a baseline requires adapting the "binary search for sufficient prefix" concept: use binary search on the reasoning trace to find the shortest prefix after which the probe consistently predicts the correct answer. This is a valid and interesting baseline.

**Note on PNS-based optimization:** The search did not find a widely-known method by this exact name in the mechanistic interpretability literature. This may refer to a niche or group-specific method. If it refers to "Probability of Necessity and Sufficiency" from causal inference (Pearl), then it would involve computing PNS for each token's causal contribution to the final answer. This is expensive (requires counterfactual reasoning per token) but theoretically grounded. Flag for validation with the project team.

**Confidence:** HIGH for greedy pruning and entropy baselines (well-established). LOW for PNS (unclear referent). MEDIUM for P-ALIGN adaptation.

**Sources:**
- Greedy pruning: https://www.arxiv.org/pdf/2601.03066
- Think Clearly pruning: https://arxiv.org/html/2507.08806v1
- P-ALIGN: https://arxiv.org/html/2601.10064
- DeepConf entropy: https://arxiv.org/abs/2508.15260

---

### TS-7: Paper-Ready Figures

| Attribute | Detail |
|-----------|--------|
| **Why Expected** | A CS224N paper without clear visualizations will score poorly. Specific figure types are standard in the mechanistic interpretability literature. |
| **Complexity** | Low-Medium (code complexity is low; making them publication-quality takes iteration) |
| **Dependencies** | All analysis features |

**Required figures:**

1. **Probe AUROC across layers (per dataset):** Line plot with layer on x-axis, AUROC on y-axis, one line per dataset. Standard in Cox, Zhang et al.

2. **Probe confidence curve over token position:** For representative examples, plot probe confidence (P(correct answer)) at each token position with a vertical line at T. Show the transition clearly. This is the central visual argument.

3. **Cosine similarity phase plot:** Plot S_t (cosine similarity between consecutive hidden states) over token position, with T marked. Expect a visible phase shift.

4. **Transition point distribution:** Histogram of T/n (normalized transition point) across all traces, per dataset. Shows how much reasoning is pre-T vs post-T.

5. **Steering success rates:** Bar chart showing success rate for +v (induce verbosity), -v (induce early termination), and random vector control, across datasets.

6. **Attention pattern heatmaps (2-4 examples):** Heatmaps of attention weights for selected heads, showing pre-T vs post-T regions. Highlight reasoning heads vs consistency heads.

**Confidence:** HIGH. These are community standards.

---

## Differentiators

Features that would strengthen the paper but are not strictly required. Prioritize based on time.

### D-1: Mutual Information Tracking

| Attribute | Detail |
|-----------|--------|
| **Value** | Provides an information-theoretic perspective complementing the probe-based transition detection. The MI Peaks paper (NeurIPS 2025) showed this is a powerful lens. |
| **Complexity** | Medium-High |
| **Dependencies** | TS-1 |

**Implementation:**

- Compute MI between the hidden representation at each generated token and the representation of the correct answer, using HSIC (Hilbert-Schmidt Independence Criterion) as an approximation.
- Detect MI peaks using IQR-based outlier detection: m_t > Q3 + 1.5 * IQR.
- Compare MI peak positions with probe-detected transition points. If they align, this is strong converging evidence.
- Qian et al. (NeurIPS 2025) found MI peaks correspond to "thinking tokens" (Hmm, Wait, Therefore). Check whether the transition point T coincides with such a thinking token.

**Why differentiator, not table stakes:** MI tracking is newer (2025) and not yet standard. The probe-based approach is sufficient; MI is a complementary validation.

**Sources:** https://arxiv.org/abs/2506.02867

---

### D-2: Attention Head Classification (Reasoning vs Consistency)

| Attribute | Detail |
|-----------|--------|
| **Value** | Shows that post-T tokens are generated by different computational circuits than pre-T tokens. Moves from "the model knows the answer" to "the model uses different mechanisms." |
| **Complexity** | High |
| **Dependencies** | TS-1, Trace generation |

**Implementation:**

- **Head classification via probing:** Following CREST (2025), train linear probes on per-head activations to classify reasoning behavior. Segment reasoning at paragraph boundaries, label steps as "reasoning" (new computation) vs "consistency" (verification, repetition, scaffolding).
- **Classification criteria (from the attention heads survey, ScienceDirect 2025):** Heads can be typed as:
  - **Induction/copying heads:** High attention to prior token sequences, match-and-copy behavior.
  - **Reasoning/function-vector heads:** Attend to semantic content, transform rather than copy.
  - **Positional heads:** Attend based on relative position.
- **Pre-T vs post-T analysis:** Compute the fraction of each head type's contribution (attention output norm) in pre-T vs post-T regions. Hypothesis: post-T is dominated by copying/consistency heads, while pre-T has more reasoning heads active.
- **Zero-ablation validation:** Ablate candidate reasoning heads (zero their output) and measure impact on the final answer. If ablating a head changes the answer only when applied pre-T (but not post-T), that head is causally involved in the reasoning but not the performative phase.

**Why differentiator, not table stakes:** This is a deeper mechanistic analysis. The paper's core argument (transition detection + causal steering) stands without it, but head classification adds a compelling mechanistic story. The CREST paper showed this is feasible with linear probes on ~38% of heads.

**Sources:**
- CREST: https://arxiv.org/abs/2512.24574
- Attention heads survey: https://www.sciencedirect.com/science/article/pii/S2666389925000248
- Induction heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html

---

### D-3: Cross-Dataset Performativity Vector Analysis

| Attribute | Detail |
|-----------|--------|
| **Value** | Tests whether performativity is a universal model behavior or dataset-specific. If the performativity vector from GSM-Symbolic steers FOLIO traces, this is strong evidence for a general phenomenon. |
| **Complexity** | Low (given TS-5 is done) |
| **Dependencies** | TS-5 |

**Implementation:**

- Extract v_perf separately for each dataset (GSM-Symbolic, FOLIO, StrategyQA).
- Compute pairwise cosine similarity between the three vectors. High similarity suggests a shared performativity direction.
- Cross-steering experiment: Use v_perf from dataset A to steer traces from dataset B. If cross-dataset steering works, performativity is model-intrinsic.

**Why differentiator:** This is a compelling analysis that elevates the paper from "we found this on 3 datasets" to "this appears to be a general property of the model." Low marginal effort given TS-5.

---

### D-4: Logit Lens Analysis at Transition Points

| Attribute | Detail |
|-----------|--------|
| **Value** | Provides a complementary interpretability lens. Project residual stream onto vocabulary to see what the model is "predicting" at each layer before and after T. |
| **Complexity** | Low |
| **Dependencies** | Trace generation |

**Implementation:**

- Apply the unembedding matrix to the residual stream at each layer for tokens near T.
- Track how the rank and probability of the correct answer token evolve across layers, comparing pre-T and post-T positions.
- Expect: At post-T positions, the correct answer should be high-rank at even early layers (the model "already knows"), while at pre-T positions the answer may only emerge in later layers.

**Why differentiator:** Quick to implement, provides intuitive visualizations, but is correlational (same limitation as probes, so does not add causal evidence).

---

### D-5: Ablation Studies

| Attribute | Detail |
|-----------|--------|
| **Value** | Shows robustness of findings to hyperparameter choices. |
| **Complexity** | Low (rerunning existing pipelines with different parameters) |
| **Dependencies** | All TS features |

**Key ablations:**

1. **Threshold sensitivity:** Vary the probe confidence threshold (0.85, 0.90, 0.95, 0.99) and the sustained-K parameter (3, 5, 10, 20). Report how T changes.
2. **Layer choice:** Show that results hold across multiple high-AUROC layers, not just the single best.
3. **Steering layer sensitivity:** Vary the injection layer and report steering success rate.
4. **Sample size:** Show that results are stable with 3, 5, 10, 20 traces per problem.
5. **Steering coefficient sweep:** Plot behavioral change vs alpha to show a dose-response curve.

---

### D-6: Token-Level Categorization with Saved Annotations

| Attribute | Detail |
|-----------|--------|
| **Value** | Enables qualitative analysis and future work. Each token in the reasoning trace is annotated as "necessary" or "performative." |
| **Complexity** | Low (direct application of transition point) |
| **Dependencies** | TS-1 |

**Implementation:**

- For each trace, classify tokens before T as "necessary" and tokens after T as "performative."
- Save annotated traces in a structured format (JSON or HDF5) with: token text, position, probe confidence, token classification, cosine similarity to previous token.
- Enable qualitative examples in the paper: show 3-4 annotated traces per dataset where the transition is visually clear.

---

## Anti-Features

Things to explicitly NOT build. Common mistakes in this domain.

### AF-1: Complex Probe Architectures

| Anti-Feature | Why Avoid |
|--------------|-----------|
| Training deep neural network probes (3+ layers) | A complex probe might learn the task independently rather than reading information already in the representation. This invalidates the interpretation. Zhang et al. (2025) showed optimal configurations converge to linear probes. |
| **What to Do Instead** | Use logistic regression. If you want to verify, show that a 2-layer MLP probe does NOT significantly outperform the linear probe (this is evidence the representation is linearly structured). |

---

### AF-2: Global Confidence Measures

| Anti-Feature | Why Avoid |
|--------------|-----------|
| Averaging probe confidence across the entire reasoning trace to get a single score | DeepConf (2025) explicitly warns that global confidence "may obscure confidence fluctuations at local reasoning steps" and "can mask critical reasoning breakdowns at specific intermediate steps." |
| **What to Do Instead** | Use per-token probe confidence. Track the local dynamics, not a global summary. |

---

### AF-3: Fine-tuning the Model

| Anti-Feature | Why Avoid |
|--------------|-----------|
| Any modification to Qwen 3.0 8B's weights | This is an analysis project. Fine-tuning changes the model's internals, making it impossible to draw conclusions about the original model. Also violates the project constraint (inference only). |
| **What to Do Instead** | All interventions are inference-time only: activation steering via hooks, no gradient updates to the model. |

---

### AF-4: Causal Claims from Probes Alone

| Anti-Feature | Why Avoid |
|--------------|-----------|
| Claiming "the model knows the answer at token T" based only on probe AUROC | Probes show correlation, not causation. A high-AUROC probe means the information is present in the residual stream, not that the model uses it. Cox (2024) explicitly noted this distinction. |
| **What to Do Instead** | Use probes for detection, steering (TS-2) for causal validation. The paper's argument must have both legs. |

---

### AF-5: Sparse Autoencoders (SAEs) for Feature Discovery

| Anti-Feature | Why Avoid |
|--------------|-----------|
| Training SAEs on Qwen 3.0 8B to decompose the residual stream into interpretable features | SAE training is expensive, time-consuming, and tangential to the project's specific goal. No pre-trained SAEs exist for Qwen 3.0 8B (as of this research date). Training one from scratch is a project in itself. |
| **What to Do Instead** | Use the simpler and well-established linear probing + contrastive steering pipeline. SAE-based refinement (FGAA, SAE-RSV) could be a "future work" mention. |

---

### AF-6: Overclaiming Generality

| Anti-Feature | Why Avoid |
|--------------|-----------|
| Claiming results generalize to all reasoning models, all tasks, or all model sizes | This project tests one model (Qwen 3.0 8B) on three binary verification datasets. Broader claims require broader evidence. |
| **What to Do Instead** | Frame results as "evidence of performative reasoning in Qwen 3.0 8B on binary verification tasks." Use cross-dataset analysis (D-3) to suggest generality within this scope. |

---

### AF-7: Interactive Demo or Real-Time System

| Anti-Feature | Why Avoid |
|--------------|-----------|
| Building a web interface or real-time reasoning analysis tool | Time sink, irrelevant to paper quality. This is research code, not a product. |
| **What to Do Instead** | Save annotated traces and figures. Qualitative examples in the paper are sufficient. |

---

### AF-8: Attention Pattern Visualization Without Quantification

| Anti-Feature | Why Avoid |
|--------------|-----------|
| Showing attention heatmaps without quantitative analysis of head types | Cherry-picked attention patterns are the most criticized practice in interpretability papers. Reviewers increasingly demand causal validation of attention-based claims. |
| **What to Do Instead** | If showing attention heatmaps (TS-7), always pair with quantitative head classification (D-2) or at minimum with ablation evidence that the highlighted heads are causally relevant. |

---

## Feature Dependencies

```
Trace Generation (not a feature -- infrastructure prerequisite)
    |
    +--> TS-1: Probe Detection --------+--> TS-3: Metrics
    |       |                           |
    |       +--> TS-5: Differential --> TS-2: Causal Steering --> TS-3: Metrics
    |       |       State Analysis
    |       |
    |       +--> D-1: MI Tracking
    |       +--> D-3: Cross-Dataset Vector Analysis (requires TS-5)
    |       +--> D-6: Token Categorization
    |
    +--> TS-4: Cosine Similarity
    |
    +--> TS-6: Baselines (independent of probes)
    |
    +--> D-2: Attention Head Classification (independent but benefits from TS-1)
    |
    +--> D-4: Logit Lens (independent)
    |
    +--> TS-7: Figures (requires all TS features)
    |
    +--> D-5: Ablations (requires all TS features)
```

**Critical path:** Trace Generation --> TS-1 --> TS-5 --> TS-2 --> TS-3 + TS-7

---

## MVP Recommendation

For MVP (minimum viable paper), prioritize in this order:

1. **TS-1: Probe Detection** -- Foundation of all claims. Without this, nothing else works.
2. **TS-4: Cosine Similarity** -- Fast to compute, provides complementary evidence, low risk.
3. **TS-5: Differential State Analysis** -- Required for steering; do this as part of trace analysis.
4. **TS-2: Causal Steering** -- The differentiating contribution. This is what makes the paper novel vs prior work.
5. **TS-6: Baselines** -- Start with entropy-based (fast) and random (trivial). Add greedy pruning if time permits.
6. **TS-3: Metrics** -- Computed from the above; low incremental effort.
7. **TS-7: Figures** -- Make plots as you go, polish at the end.

**Defer to post-MVP:**
- D-1 (MI Tracking): Interesting but time-consuming to implement HSIC correctly.
- D-2 (Attention Head Classification): High effort, secondary to core argument.
- D-3 (Cross-Dataset Vector): Low effort but requires TS-5 to be solid first.
- D-4 (Logit Lens): Quick win if there is time.
- D-5 (Ablations): Do threshold sensitivity at minimum; full ablation suite if time allows.
- D-6 (Token Categorization): Build the data structure early (during TS-1), annotate later.

---

## Complexity Summary

| Feature | Complexity | Effort Estimate | Priority |
|---------|-----------|-----------------|----------|
| TS-1: Probe Detection | Medium | 2-3 days | CRITICAL |
| TS-2: Causal Steering | High | 3-4 days | CRITICAL |
| TS-3: Metrics | Low | 0.5 day | CRITICAL |
| TS-4: Cosine Similarity | Low-Medium | 1 day | HIGH |
| TS-5: Differential State | Medium | 2 days | CRITICAL |
| TS-6: Baselines | Medium | 2 days | HIGH |
| TS-7: Figures | Low-Medium | 1-2 days | HIGH |
| D-1: MI Tracking | Medium-High | 2-3 days | MEDIUM |
| D-2: Attention Heads | High | 3-4 days | LOW-MEDIUM |
| D-3: Cross-Dataset | Low | 0.5 day | MEDIUM |
| D-4: Logit Lens | Low | 0.5 day | MEDIUM |
| D-5: Ablations | Low | 1-2 days | MEDIUM |
| D-6: Token Annotations | Low | 0.5 day | MEDIUM |

**Total estimated effort for table stakes:** ~12-14 days
**Total with differentiators:** ~22-28 days

---

## Sources

### Primary References (HIGH confidence)
- Cox (2024), "Post-hoc reasoning in chain of thought": https://kyle-cox.com/2024/12/26/cot-interp/
- Zhang et al. (2025), "Reasoning Models Know When They're Right": https://arxiv.org/html/2504.05419v1
- Qian et al. (NeurIPS 2025), "Thinking Tokens are Information Peaks": https://arxiv.org/abs/2506.02867
- DeepConf (Meta, 2025), "Deep Think with Confidence": https://arxiv.org/abs/2508.15260
- CREST (NeurIPS 2025 Workshop), "Cognitive Reasoning Steering": https://arxiv.org/abs/2512.24574
- "Temporal Predictors of Outcome" (2025): https://arxiv.org/html/2511.14773
- "Knowing Before Saying" (2025): https://arxiv.org/html/2505.24362v2

### Methods References (HIGH confidence)
- nnsight activation patching tutorial: https://nnsight.net/notebooks/tutorials/causal_mediation_analysis/activation_patching/
- nnsight attribution patching: https://nnsight.net/notebooks/tutorials/causal_mediation_analysis/attribution_patching/
- Activation Addition (Turner et al.): https://arxiv.org/html/2308.10248v4
- CAA/Rimsky (MATS): https://www.matsprogram.org/research/recwbuHVD3MEdZyvW
- Neel Nanda, Attribution Patching: https://www.neelnanda.io/mechanistic-interpretability/attribution-patching

### Ecosystem References (MEDIUM confidence)
- Greedy pruning (Singh et al. 2025): https://www.arxiv.org/pdf/2601.03066
- Think Clearly pruning: https://arxiv.org/html/2507.08806v1
- Feature Flow / cosine similarity: https://arxiv.org/html/2502.03032
- LLM-Streamline: https://arxiv.org/html/2403.19135v3
- Attention heads survey: https://www.sciencedirect.com/science/article/pii/S2666389925000248
- Induction heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html
- Steering scaling laws: https://arxiv.org/pdf/2507.11771
- IBM activation steering: https://github.com/IBM/activation-steering

### Low Confidence / Needs Validation
- PNS-based optimization: Could not find a standard method by this name in the mechanistic interpretability literature. May refer to Pearl's Probability of Necessity and Sufficiency, or a group-specific technique. Needs clarification from the project team.
