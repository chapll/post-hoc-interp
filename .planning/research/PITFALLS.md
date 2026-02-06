# Domain Pitfalls

**Domain:** Mechanistic interpretability of reasoning LLMs (Qwen 3.0 8B performative reasoning detection)
**Researched:** 2026-02-04

---

## Critical Pitfalls

Mistakes that cause rewrites, invalidate results, or waste weeks of compute.

---

### Pitfall 1: Linear Probes Succeed for Wrong Reasons (Spurious Correlation)

**What goes wrong:** A logistic regression probe on the residual stream achieves high AUROC for predicting True/False answers, but it is picking up on surface-level correlations (token identity, sequence length, lexical cues) rather than the model's actual internal representation of the answer. The probe "works" but its transition point T_i is meaningless because it tracks spurious features, not the model's decision process.

**Why it happens:** Linear probes are powerful enough to exploit any linearly separable signal in a 4096-dimensional space. With only 300 problems per dataset and 5 traces each (1500 total), the dimension-to-sample ratio is dangerously high. At d=4096 and n=1500, random labels can become linearly separable -- a well-known phase transition phenomenon. Additionally, binary verification tasks ("Is the answer 42? Yes/No") may encode the answer in superficial token embeddings of "42" rather than in the model's reasoning state.

**Consequences:** The entire transition detection pipeline is built on probe confidence. If probes are spurious, transition points are artifacts, differential analysis is noise, and causal steering validates nothing. This invalidates the paper.

**Prevention:**
1. Run selectivity controls: train probes on randomly permuted labels. If AUROC on random labels exceeds 0.6, the probe is exploiting dimensionality, not features. This is the Hewitt & Liang (2019) control task methodology.
2. Use ridge regression (L2 regularization) rather than unregularized logistic regression. The regularization constrains the probe to use a low-norm direction, which reduces overfitting to high-dimensional noise.
3. Validate probes causally: if the probe says the model "knows the answer" at position T_i, patch the residual stream at T_i from a trace with a different answer. If the model's output does not change, the probe is tracking a non-causal direction.
4. Train probes on activations from a subset of layers (e.g., layers 18-36) rather than a single layer, and check if transition points are consistent across layers. Inconsistency signals spuriousness.
5. Monitor probe weight norms. If the probe weight vector has large norm relative to the activation magnitudes, it is likely overfitting.

**Detection:** Probe AUROC on control tasks. If random-label probes exceed 0.55-0.60 AUROC, the real probes are suspect. Also check: does the transition point T_i correlate with simple features like "position of the final numerical token" or "halfway through the trace"?

**Phase:** Probing (Phase 3). Must be addressed before any downstream analysis depends on probe outputs. Build selectivity controls into the probing pipeline from day one.

**Confidence:** HIGH -- this is the most extensively documented pitfall in the probing literature.

**Sources:**
- [Hewitt & Liang, 2019 -- Selectivity as a control](https://sidn.baulab.info/probing/)
- [DeepMind Safety Research -- Dense probes and spurious correlations](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)
- [Spectral Identifiability Principle for probe stability](https://arxiv.org/pdf/2511.16288)

---

### Pitfall 2: Dataset Contamination Invalidates "Reasoning" Claims

**What goes wrong:** Qwen 3.0 8B was trained on 36+ trillion tokens of web data. GSM8K is one of the most widely published benchmarks on the internet. There is strong evidence (BenBench project) that even smaller Qwen models (Qwen-1.8B) can reproduce exact n-grams from GSM8K. If Qwen 3.0 has memorized GSM8K problems, its "reasoning traces" on GSM-Symbolic variants may still show contamination artifacts -- the model may pattern-match to memorized solutions rather than reason, making the distinction between "genuine reasoning" and "performative reasoning" meaningless because none of it is genuine.

**Why it happens:** GSM8K is everywhere on the internet. GSM-Symbolic was designed to mitigate this by using parameterized templates to generate novel problem instances, but Qwen may have seen GSM-Symbolic itself (published at ICLR 2025, widely available). FOLIO is human-authored from Wikipedia, making it vulnerable to training data overlap since Wikipedia is a standard pre-training corpus. StrategyQA relies on world knowledge facts that are extensively covered in web text.

**Consequences:** If the model has memorized answers to specific problems, probe transition points may reflect "moment of recall" rather than "moment of reasoning completion." The entire interpretive framework becomes ambiguous -- are we detecting performative reasoning or memorized pattern matching?

**Prevention:**
1. For GSM-Symbolic: generate fresh problem instances using the symbolic templates with new numerical values not present in any published dataset. The template system exists precisely for this purpose -- use it.
2. For FOLIO: use the binary verification conversion (True/False) which changes the task format enough that exact memorization of FOLIO's original labels is less useful. But also check: run a perplexity analysis on Qwen 3.0 for FOLIO premises. If perplexity is suspiciously low compared to novel text of similar complexity, contamination is likely.
3. For StrategyQA: this is the highest contamination risk because it tests world knowledge that is densely covered in pre-training data. Use it as a contrast dataset (different reasoning type) rather than a primary evidence source.
4. Run a contamination diagnostic: for 20-30 problems from each dataset, prompt Qwen 3.0 with just the problem statement (no verification framing) and check if it can reproduce the exact ground truth answer with suspiciously high accuracy. Compare accuracy on original vs. modified versions of the same problems.
5. Report contamination analysis in the paper. Reviewers will ask about this.

**Detection:** Compare model accuracy on original benchmark problems vs. freshly generated variants. A large accuracy gap (>5%) on structurally identical problems with different numbers signals contamination. Also check if the model's reasoning traces for potentially contaminated problems show qualitatively different patterns (shorter, more "template-like").

**Phase:** Dataset loading (Phase 2). Must generate fresh GSM-Symbolic instances and run contamination diagnostics before collecting any traces for analysis.

**Confidence:** HIGH -- BenBench evidence for Qwen family contamination is strong. GSM-Symbolic contamination resistance is documented at ICLR 2025.

**Sources:**
- [BenBench -- Benchmarking Benchmark Leakage](https://github.com/GAIR-NLP/benbench)
- [GSM-Symbolic -- ICLR 2025](https://arxiv.org/pdf/2410.05229)
- [Survey on Data Contamination for LLMs](https://arxiv.org/html/2502.14425v2)
- [FOLIO limitations -- ProverQA, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/862819c227b16f9af64dd6ad6cfdf275-Paper-Conference.pdf)

---

### Pitfall 3: Activation Patching Destroys Model Capabilities Instead of Testing Causality

**What goes wrong:** When patching the residual stream at position T_i from one trace into another, the patch magnitude or scope is wrong. Patching the entire residual stream vector (dim 4096) at a single position can catastrophically disrupt the model's computation, causing it to output garbage rather than a different answer. This makes the intervention uninterpretable -- you cannot distinguish "this component was causally important" from "I broke the model."

**Why it happens:** The residual stream is a shared communication bus. At any given position, it carries information from all previous layers and all previous tokens (via attention). Patching the full vector at one position replaces all of this information simultaneously, which is far too coarse. The model's self-repair mechanisms (backup circuits) further complicate interpretation -- the model may compensate for the intervention in unexpected ways.

**Consequences:** Causal experiments produce uninterpretable results. The paper's central claim (post-transition tokens are causally unnecessary) cannot be supported.

**Prevention:**
1. Use mean ablation rather than zero ablation for activation patching. Replace the residual stream at position T_i with the mean activation across a reference distribution (e.g., mean across all traces for the same problem). This preserves the general magnitude and structure while removing specific information.
2. Patch individual components rather than the full residual stream: attention head outputs, MLP outputs, or specific subspaces. This gives finer-grained causal evidence.
3. Validate with a "sham intervention" control: patch activations from a trace with the same answer. If the model's output changes, the intervention is too destructive.
4. Use the KL divergence of the output distribution (not just the argmax answer) as the evaluation metric. A small KL divergence after patching pre-transition activations, compared to large KL divergence after patching post-transition activations, is cleaner evidence than binary answer flips.
5. For steering vector injection specifically: sweep magnitude coefficients from 0.1 to 5.0 and plot the effect curve. There is typically a narrow "goldilocks zone" where steering changes behavior without destroying coherence.

**Detection:** If more than 30% of patching interventions produce degenerate outputs (repetition loops, gibberish, empty generation), the intervention is too coarse. Track the fraction of "coherent but changed" vs "incoherent" outcomes.

**Phase:** Causal steering (Phase 5). But the design decisions (mean ablation vs. zero ablation, component-level vs. full residual stream) must be planned during architecture design.

**Confidence:** HIGH -- Heimersheim (2024) and the "Best Practices of Activation Patching" paper (2309.16042) document these issues extensively.

**Sources:**
- [Towards Best Practices of Activation Patching](https://arxiv.org/abs/2309.16042)
- [How to Use and Interpret Activation Patching -- Heimersheim](https://arxiv.org/pdf/2404.15255)
- [Open Problems in Mechanistic Interpretability](https://arxiv.org/html/2501.16496v1)

---

### Pitfall 4: OOM When Storing Full Residual Stream Activations

**What goes wrong:** Saving the residual stream at all 36 layers of Qwen 3.0 8B for every token in a reasoning trace causes GPU memory exhaustion. A single trace with 2048 tokens requires ~576 MB of activation storage (36 layers * 2048 tokens * 4096 dims * 2 bytes float16). With the model itself occupying ~16 GB in float16, and KV cache consuming additional memory, even a single trace with activations on an 80GB A100 is feasible but leaves limited headroom. Batch processing multiple traces simultaneously will OOM.

**Why it happens:** The project requires residual stream activations at every token position for probe training and transition detection. Naive implementation saves everything to GPU memory. With nnsight's `.save()`, all saved tensors persist in GPU memory until the tracing context exits.

**Specific numbers for Qwen 3.0 8B:**
- Model weights (float16): ~16.4 GB
- KV cache per trace (2048 tokens, float16): ~1.1 GB (36 layers * 2 * 8 KV heads * 128 dim * 2048 * 2 bytes)
- Residual stream all layers (2048 tokens, float16): ~576 MB per trace
- Residual stream all layers (4096 tokens, float16): ~1.15 GB per trace
- Total for model + 1 trace (2048 tok): ~18 GB -- fits on one A100
- Total for model + batch of 4 traces (2048 tok): ~23 GB -- fits, but attention computation requires more transient memory
- Total for model + batch of 8 traces (4096 tok): ~28 GB + attention peaks -- getting tight

**Consequences:** OOM crashes during trace generation. Slower pipeline if forced to batch_size=1. Extended wall-clock time for generating 300*5=1500 traces (probing) plus 300*20=6000 traces (differential analysis).

**Prevention:**
1. Move saved activations to CPU immediately: within the nnsight tracing context, call `.cpu()` on saved tensors. This transfers them off GPU as soon as they are computed. Example: `layer.output[0].cpu().save()`.
2. Save only the layers you need. For initial probe training, start with layers at 25%, 50%, 75%, and 100% depth (layers 9, 18, 27, 35). Expand to all layers only after confirming which layers show informative probe behavior.
3. Use float16 or bfloat16, not float32. Qwen 3.0 natively uses bfloat16, so there is no precision loss.
4. Process traces sequentially (batch_size=1 or 2) for activation collection, using DDP across GPUs for throughput. Each GPU processes different problems, not different traces of the same problem.
5. Write activations to disk (numpy memmap or HDF5) immediately after each trace completes. Do not accumulate in RAM.
6. For the differential analysis phase (20 traces per problem), collect only the residual stream at the transition point T_i and a small window around it, not the full trace.

**Detection:** Monitor GPU memory usage with `torch.cuda.max_memory_allocated()` after each trace. If it exceeds 60 GB on an 80 GB A100, reduce batch size or add more `.cpu()` calls.

**Phase:** Trace generation (Phase 1) and probing (Phase 3). Memory management must be designed into the trace generation pipeline from the start. Retrofitting is painful.

**Confidence:** HIGH -- these are straightforward arithmetic calculations based on Qwen 3.0 8B's confirmed architecture (36 layers, hidden_size=4096).

**Sources:**
- [Qwen3-8B config.json on HuggingFace](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)
- [Compressed Activations for Memory-Efficient LLM Training](https://aclanthology.org/2025.naacl-long.71.pdf)
- [Scale SAE Activations: Fix OOM and Go Multi-GPU](https://pub.towardsai.net/how-to-extract-and-save-sae-activations-efficiently-multi-gpu-sparse-arrays-f051159ff67a)

---

### Pitfall 5: No Clean Transition Point Exists

**What goes wrong:** The project assumes a clear transition point T_i where the model shifts from "searching" to "knowing" the answer. But for many traces, this transition may be gradual (probe confidence ramps from 0.5 to 0.95 over 200 tokens), non-monotonic (confidence oscillates), or absent (the model never confidently commits, even for correct answers). The 95% confidence threshold is arbitrary and may not correspond to any real internal event.

**Why it happens:** LLM reasoning is not a two-phase process. Research on logical phase transitions (arXiv 2601.02902) shows that reasoning collapse is often abrupt, but reasoning construction can be incremental. The model may accumulate evidence gradually, with the residual stream reflecting partial confidence at many intermediate stages. Additionally, for problems where the model's "reasoning" is actually confabulation, there may never be a genuine transition because the model may have "known" the answer from the first token (contamination) or may never truly arrive at the answer (lucky guess with bad reasoning).

**Consequences:** If transition points are noisy or absent for a significant fraction of traces, the differential analysis (comparing pre-T_i vs post-T_i states) and the core claim about "performative post-transition tokens" lose statistical power. The paper's narrative falls apart.

**Prevention:**
1. Define multiple transition detection methods and compare them:
   - Probe confidence crossing 95% (current plan)
   - Probe confidence derivative: the point of maximum slope (steepest confidence increase)
   - Cosine similarity stability: the point where consecutive residual stream states become highly similar (the project's "intrinsic phase analysis")
   - Entropy of the model's next-token distribution: the point where it drops below a threshold
2. Categorize traces by transition quality: "clean transition" (confidence goes from <0.6 to >0.95 within 50 tokens), "gradual transition" (ramp over >200 tokens), "no transition" (never exceeds 0.85), "immediate" (confidence >0.9 from the first thinking token).
3. Report results separately for each category. "Immediate" transitions are particularly interesting -- they may indicate contamination or trivial problems.
4. Do not force all traces into the binary "pre-transition / post-transition" framework. Report what fraction of traces have clean transitions and be honest about it. A finding like "only 60% of traces show clean transitions, and for those traces, post-transition tokens are performative" is still a valid contribution.
5. Use the cosine similarity dynamics as an independent validation of probe-based transition points. If the two methods agree, confidence is higher.

**Detection:** Plot the distribution of transition sharpness (confidence derivative at T_i) across all traces. If it is bimodal, the clean/gradual distinction is real. If it is uniform or unimodal with no peak, the concept of a "transition point" may not apply to this model/dataset combination.

**Phase:** Transition detection (Phase 3). Must design the detection pipeline to handle all cases gracefully rather than failing on non-clean transitions.

**Confidence:** MEDIUM -- the project's cited prior work (Cox 2025) found clean transitions in some models, but Qwen 3.0 with its explicit "thinking mode" may behave differently. This is genuinely unknown.

**Sources:**
- [Logical Phase Transitions in LLM Reasoning](https://arxiv.org/html/2601.02902)
- [Efficient Test-Time Scaling by Probing Internal States](https://arxiv.org/abs/2511.06209)
- [Entropy-based stopping for reasoning models](https://jiaweizzhao.github.io/deepconf/static/pdfs/deepconf_arxiv.pdf)

---

## Moderate Pitfalls

Mistakes that cause delays, weaken results, or create technical debt.

---

### Pitfall 6: nnsight Deadlocks from Out-of-Order Module Access

**What goes wrong:** Within an nnsight `model.trace()` context, code accesses a later layer's output before an earlier layer's output. Because nnsight executes intervention code in parallel with the forward pass using PyTorch hooks, accessing layer 18's output before layer 9's output causes a deadlock -- the thread blocks waiting for layer 9's hook, which has already fired and will never fire again.

**Why it happens:** nnsight's execution model is different from standard PyTorch. Code inside `model.trace()` is not eagerly executed -- it runs in parallel with the forward pass, with threads waiting for hooks to provide values. This is counterintuitive if you are used to writing normal Python code. It is very easy to accidentally write a loop that accesses layers out of order, especially when iterating over a subset of layers or when combining forward and backward passes.

**Prevention:**
1. Always iterate over layers in forward order (0, 1, 2, ..., 35) within a single invoke context.
2. Never conditionally skip layers in the middle of a loop -- access all layers in order, but only `.save()` the ones you need.
3. For backward-pass gradient access, open a separate backward context (`with tensor.backward():`) and access gradients in reverse order.
4. When accessing outputs from multiple module types (attention, MLP, layer norm), ensure you access them in the order they execute within each layer.
5. Test with a small model (e.g., GPT-2) before running on Qwen 3.0 8B to catch ordering bugs cheaply.

**Detection:** If a tracing call hangs indefinitely (no error, no output, no progress), it is almost certainly an out-of-order access deadlock. Add timeout logic around trace calls.

**Phase:** Trace generation (Phase 1). This is a day-one implementation concern.

**Confidence:** HIGH -- this is explicitly documented as the primary gotcha in nnsight's official walkthrough.

**Sources:**
- [NNsight Walkthrough](https://nnsight.net/notebooks/tutorials/get_started/walkthrough/)
- [NNsight GitHub](https://github.com/ndif-team/nnsight)

---

### Pitfall 7: Steering Vector Magnitude Selection is Unprincipled

**What goes wrong:** After computing the "performativity vector" (mean post-transition state of verbose traces minus mean post-transition state of concise traces), it must be injected into other traces at some magnitude. Too small and there is no measurable effect. Too large and the model produces gibberish. There is no principled way to select the right magnitude, and the "right" magnitude may differ across problems, layers, and token positions.

**Why it happens:** Steering vectors operate in the model's activation space, where scale is not calibrated to any interpretable unit. A vector with L2 norm of 1.0 may be negligible relative to the residual stream's typical norm (~50-100 at intermediate layers of an 8B model), while a vector with norm 10.0 may catastrophically disrupt computation. Research shows up to 50% of examples can be "anti-steerable" -- responding in the opposite direction from expected.

**Prevention:**
1. Normalize the steering vector and sweep a range of scaling coefficients: [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0] times the mean residual stream norm at the injection layer.
2. For each coefficient, measure three outcomes: (a) fraction of traces where behavior changes in the expected direction, (b) fraction where behavior changes in the opposite direction, (c) fraction where output becomes incoherent. Plot all three.
3. Select the coefficient that maximizes (a) while keeping (c) below 10%.
4. Report the full sweep in the paper, not just the best coefficient. Cherry-picking the best magnitude is misleading.
5. Consider injecting at multiple layers simultaneously (late layers, e.g., 28-35) rather than a single layer. Research suggests late-layer injection preserves fluency better.
6. Use KL divergence from the unsteered distribution as a continuous measure rather than binary "did the answer flip."

**Detection:** If no coefficient produces >30% expected-direction changes while keeping incoherence below 10%, the performativity vector may not be a real steering direction. This would be a negative result worth reporting.

**Phase:** Causal steering (Phase 5). Design the sweep infrastructure during architecture design (Phase 1).

**Confidence:** HIGH -- magnitude calibration is identified as a central open problem in representation engineering surveys.

**Sources:**
- [A Sober Look at Steering Vectors for LLMs](https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/a-sober-look-at-steering-vectors-for-llms)
- [Feature Guided Activation Additions](https://arxiv.org/html/2501.09929v3)
- [Representation Engineering Survey 2025](https://janwehner.com/files/representation_engineering.pdf)

---

### Pitfall 8: Stochastic Sampling Makes Traces Non-Reproducible

**What goes wrong:** Qwen 3.0 thinking mode requires stochastic sampling (Temperature=0.6, TopP=0.95, TopK=20). This means running the same problem twice produces different reasoning traces with potentially different transition points, different answer correctness, and different trace lengths. Controlled experiments (e.g., "compare traces with and without intervention X") become confounded by trace variability.

**Why it happens:** The sampling parameters are required by Qwen 3.0's thinking mode specification. Greedy decoding (temperature=0) disables thinking mode or produces degenerate outputs. Even with the same random seed, hardware-level floating-point non-determinism (documented for A100 GPUs in particular) can produce different traces.

**Prevention:**
1. Fix random seeds per problem for reproducibility within a single experimental run. Use `torch.manual_seed()`, `torch.cuda.manual_seed_all()`, and set `torch.backends.cudnn.deterministic = True`. This does not guarantee cross-hardware reproducibility but ensures same-machine reproducibility.
2. For causal experiments, generate the base trace first, then perform interventions on the cached KV prefix (not re-generating). This ensures pre-intervention tokens are identical.
3. For statistical analyses, collect enough traces per problem (the planned 5 for probing, 20 for differential analysis) and report means with confidence intervals, not individual trace results.
4. When comparing "concise" vs "verbose" traces, match on problem identity. Never compare concise traces for problem A against verbose traces for problem B.
5. For the final paper, report how much trace variability exists: what fraction of traces for the same problem have the same answer? How variable is the transition point across traces of the same problem?

**Detection:** Run 5 traces for the same problem with different seeds. If the transition point T_i varies by more than 30% of the trace length, per-problem transition detection is unreliable and statistical aggregation becomes essential.

**Phase:** Trace generation (Phase 1). Seed management and KV caching strategy must be designed into the pipeline from the start.

**Confidence:** HIGH -- stochastic sampling is a hard requirement per Qwen 3.0 docs. The non-determinism issue on A100s is well-documented.

**Sources:**
- [High Precision for Reasoning Recipes -- FP32 determinism](https://jianyuh.github.io/determinism/2025/06/23/FP32-Reasoning.html)
- [Model Uncertainty and Variability in LLMs](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1609097/pdf)

---

### Pitfall 9: Correlation-Causation Conflation in Probe Results

**What goes wrong:** The project finds that linear probes can decode the correct answer from the residual stream starting at position T_i. This is presented as evidence that "the model knows the answer at T_i." But high probe accuracy only shows the information is linearly decodable -- it does not prove the model uses this information for its output. The model might encode the answer incidentally (as a side effect of processing) without its output head actually reading from this direction.

**Why it happens:** This is the fundamental limitation of probing as a methodology. A probe is an external classifier trained on frozen representations. Its accuracy reflects what information is present, not what information is used. This distinction is well-established (Maiya et al., 2025; the "probing gap" literature) but frequently ignored in interpretability papers.

**Prevention:**
1. The project already plans causal interventions (activation patching, steering) -- this is exactly the right approach. The key is to present probing as correlational evidence and patching as causal evidence, never claiming probes alone prove causality.
2. Structure the paper's argument as: "Probes suggest [correlational claim]. Causal interventions confirm [causal claim]." This two-step structure is standard in the mechanistic interpretability literature.
3. For the transition point specifically: use probes to identify T_i candidates, then validate with patching. Patch the residual stream at T_i from a trace with a different answer. If the output changes, the probe direction is causally relevant. If it does not, the probe found a non-causal feature.
4. Report the concordance rate: what fraction of probe-identified transition points are validated by causal interventions?

**Detection:** If probe accuracy is high but patching at the probe-identified direction has no effect on output, the probe is not causally relevant.

**Phase:** Spans probing (Phase 3) and causal steering (Phase 5). The paper framing must be correct from the writing stage.

**Confidence:** HIGH -- this is one of the most discussed limitations in the mechanistic interpretability literature.

**Sources:**
- [Open Problems in Mechanistic Interpretability](https://arxiv.org/html/2501.16496v1)
- [Triangulation for robust mechanistic claims](https://arxiv.org/html/2512.24842)
- [Mechanistic Interpretability as Statistical Estimation (ICLR 2026)](https://www.arxiv.org/pdf/2510.02917)

---

### Pitfall 10: Multiple Comparisons Across Layers, Heads, and Datasets

**What goes wrong:** The project tests probes at 36 layers, analyzes 32 attention heads per layer, runs experiments across 3 datasets, and evaluates multiple metrics. Without correction for multiple comparisons, statistically significant results will appear by chance. With 36 layers * 32 heads = 1152 head-level tests, at alpha=0.05, approximately 58 heads will appear "significant" by pure chance.

**Why it happens:** It is natural to report the layers or heads with the strongest effects. But if you search across 1000+ candidates and report the top ones without correction, you are reporting noise.

**Prevention:**
1. Apply Benjamini-Hochberg FDR correction when reporting significance across multiple layers, heads, or datasets. This is more appropriate than Bonferroni for exploratory analyses (less conservative but still controls false discovery rate).
2. Pre-register primary hypotheses: "We predict that probes at layers in the top third (layers 24-35) will show earlier transition points than probes at bottom-third layers." Test this specific hypothesis at alpha=0.05 without correction. Test everything else as exploratory with FDR correction.
3. For attention head classification ("reasoning" vs "consistency"), do not threshold on a per-head p-value. Instead, use a permutation test on the aggregate classification: shuffle the pre/post-transition labels and re-classify heads 1000 times to establish a null distribution.
4. Report effect sizes (Cohen's d or similar), not just p-values. A significant-but-tiny effect across 1500 traces is not interesting.

**Detection:** If your "significant" results do not survive FDR correction, they are likely false positives.

**Phase:** Evaluation (Phase 6). But the statistical analysis plan should be designed during architecture/planning.

**Confidence:** HIGH -- this is standard statistics, not domain-specific, but is frequently neglected in interpretability papers.

**Sources:**
- [Mechanistic Interpretability as Statistical Estimation (ICLR 2026)](https://www.arxiv.org/pdf/2510.02917)
- [Validating Learned Features and Circuits](https://ytcs.github.io/machine-learning/2025/05/25/mechanistic-interpretability-part-5.html)

---

## Minor Pitfalls

Mistakes that cause annoyance, add friction, or require small fixes.

---

### Pitfall 11: nnsight .save() Forgotten -- Silent Data Loss

**What goes wrong:** Inside the nnsight tracing context, you access `layer.output[0]` and assign it to a variable, but forget to call `.save()`. The variable appears to work inside the context but contains garbage or is garbage-collected after the context exits. Hours of trace generation produce no usable activation data.

**Prevention:**
1. Establish a coding pattern: every activation access ends with `.save()`. No exceptions.
2. Add a post-trace assertion that checks saved tensors are non-empty and have expected shapes.
3. After the first trace, immediately verify that saved activations have the expected shape `[batch, seq_len, 4096]`.

**Phase:** Trace generation (Phase 1). Catch this in the first test run.

**Confidence:** HIGH -- explicitly documented in nnsight walkthrough.

---

### Pitfall 12: nnsight Unbounded Iteration -- Code After Loop Never Executes

**What goes wrong:** When using `tracer.iter[:]` or `tracer.all()` for autoregressive generation within nnsight, any code placed after the iteration block is silently never executed. This is a documented behavior that is easy to forget.

**Prevention:**
1. Do not place critical logic (saving final states, computing metrics) after an iteration block.
2. Wrap iteration in a separate invoker context.
3. For this project, trace generation likely does not need iteration (single forward pass with `generate()`), but if implementing incremental activation collection during generation, be aware of this.

**Phase:** Trace generation (Phase 1), if using iterative generation.

**Confidence:** HIGH -- documented in nnsight walkthrough.

---

### Pitfall 13: Activation Shape Mismatch Between Layers

**What goes wrong:** Different modules in the transformer return outputs with different shapes and packaging. Layer outputs may be tuples `(hidden_states, attention_weights, ...)`, not bare tensors. Attention outputs have shape `[batch, num_heads, seq, seq]`, not `[batch, seq, hidden]`. Code that assumes uniform shapes across all saved activations will break silently (producing wrong probe inputs) or loudly (shape errors).

**Prevention:**
1. For residual stream (hidden states): access `layer.output[0]` (the first element of the tuple). Shape: `[batch, seq_len, 4096]`.
2. For attention weights: access `layer.self_attn.output[1]` if available (depends on model config `output_attentions=True`). Shape: `[batch, 32, seq_len, seq_len]`.
3. For MLP outputs: access `layer.mlp.output`. Shape: `[batch, seq_len, 4096]`.
4. Print shapes of all saved activations after the first trace and assert they match expectations. Hard-code expected shapes as constants.

**Phase:** Trace generation (Phase 1). Validate shapes in the first test run.

**Confidence:** HIGH -- standard PyTorch transformer issue, not specific to nnsight.

---

### Pitfall 14: Thinking Mode Token Boundaries Not Properly Parsed

**What goes wrong:** Qwen 3.0's thinking mode wraps reasoning tokens in `<think>...</think>` tags. The project needs to identify which tokens are "thinking tokens" vs. "answer tokens." If the tokenizer splits these special tokens differently than expected, or if the model sometimes omits closing tags, the token classification is wrong and all per-token analysis is corrupted.

**Prevention:**
1. After tokenization, explicitly locate the `<think>` and `</think>` token IDs in the output. Do not rely on string matching on decoded text -- work at the token ID level.
2. Handle edge cases: model generates `</think>` prematurely, model never generates `</think>` (reaches max length), model generates multiple `<think>` blocks.
3. For traces where the thinking block is malformed, exclude them from analysis rather than guessing boundaries.
4. Check: does the tokenizer encode `<think>` as a single special token or as multiple subword tokens? This varies by tokenizer version.

**Phase:** Trace generation (Phase 1). Must be validated before any trace-level analysis.

**Confidence:** MEDIUM -- Qwen 3.0 thinking mode is relatively new and edge cases may not be well-documented.

---

### Pitfall 15: Probe Training on Imbalanced Classes

**What goes wrong:** Binary verification tasks may produce imbalanced True/False distributions. If 70% of traces have True answers and 30% have False, a probe that always predicts True achieves 70% accuracy. AUROC is robust to class imbalance, but probe confidence calibration is not -- the probe's 95% confidence threshold may correspond to very different actual probabilities for the majority vs. minority class.

**Prevention:**
1. Balance the dataset: ensure equal numbers of True and False examples per dataset. The binary verification conversion step should produce balanced labels by design.
2. Use AUROC as the primary metric (already planned), not accuracy.
3. Calibrate probe confidence with Platt scaling or isotonic regression after training. This ensures 95% confidence actually means 95% probability.
4. Report precision-recall curves in addition to AUROC, especially for the minority class.

**Phase:** Probing (Phase 3). Dataset balancing should be ensured during dataset creation (Phase 2).

**Confidence:** MEDIUM -- depends on how the binary verification tasks are constructed.

---

## Phase-Specific Warnings

| Phase | Likely Pitfall | Severity | Mitigation |
|-------|---------------|----------|------------|
| Trace generation (Phase 1) | OOM from activation storage (P4) | Critical | CPU offload, selective layer saving |
| Trace generation (Phase 1) | nnsight deadlock from out-of-order access (P6) | Moderate | Always iterate layers in forward order |
| Trace generation (Phase 1) | nnsight .save() forgotten (P11) | Minor | Post-trace shape assertions |
| Trace generation (Phase 1) | Thinking token boundary parsing (P14) | Minor | Work at token ID level, handle edge cases |
| Dataset loading (Phase 2) | Contamination invalidates results (P2) | Critical | Fresh GSM-Symbolic instances, contamination diagnostics |
| Dataset loading (Phase 2) | Class imbalance in verification tasks (P15) | Minor | Balanced True/False by construction |
| Probing (Phase 3) | Spurious probe correlations (P1) | Critical | Selectivity controls, ridge regression, causal validation |
| Probing (Phase 3) | No clean transition point (P5) | Critical | Multiple detection methods, categorize transition quality |
| Probing (Phase 3) | Correlation not causation (P9) | Moderate | Present probes as correlational, validate with patching |
| Causal steering (Phase 5) | Patching destroys capabilities (P3) | Critical | Mean ablation, component-level patching, sham controls |
| Causal steering (Phase 5) | Steering magnitude unprincipled (P7) | Moderate | Systematic sweep, report full results |
| All phases | Stochastic sampling non-reproducibility (P8) | Moderate | Seed management, sufficient traces per problem |
| Evaluation (Phase 6) | Multiple comparisons inflation (P10) | Moderate | Benjamini-Hochberg FDR, pre-register hypotheses |

---

## Summary: Top 5 Things to Get Right

1. **Probes need controls.** Never report probe accuracy without selectivity controls (random-label baselines). This is non-negotiable for the paper.

2. **Contamination must be addressed.** Generate fresh GSM-Symbolic instances. Run contamination diagnostics. Report results. Reviewers will check.

3. **Memory budget must be planned upfront.** Saving residual streams at all 36 layers for thousands of traces requires CPU offload and selective saving. Design this into the pipeline from day one, not as a retrofit.

4. **Transition points may not be clean.** Build the pipeline to handle gradual, absent, and immediate transitions. Report the distribution of transition quality honestly.

5. **Causal experiments must not break the model.** Use mean ablation, test at component level, run sham controls, sweep steering magnitudes. Report the full sweep, not the cherry-picked best result.

---

## Sources (Complete)

### Linear Probing
- [Probing -- Structure and Interpretation of Deep Networks (SIDN)](https://sidn.baulab.info/probing/)
- [DeepMind Safety Research -- Negative Results for SAEs](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)
- [Spectral Identifiability Principle](https://arxiv.org/pdf/2511.16288)
- [Analysing Adversarial Attacks with Linear Probing](https://www.alignmentforum.org/posts/cPCvfrqjgy5Cu2FCs/analysing-adversarial-attacks-with-linear-probing-3)
- [DL Models are Secretly Almost Linear -- Beren Millidge](https://www.beren.io/2023-04-04-DL-models-are-secretly-linear/)

### Activation Patching and Causal Interventions
- [Towards Best Practices of Activation Patching (arXiv 2309.16042)](https://arxiv.org/abs/2309.16042)
- [How to Use and Interpret Activation Patching -- Heimersheim (arXiv 2404.15255)](https://arxiv.org/pdf/2404.15255)
- [Open Problems in Mechanistic Interpretability (arXiv 2501.16496)](https://arxiv.org/html/2501.16496v1)
- [Triangulation as an Acceptance Rule (arXiv 2512.24842)](https://arxiv.org/html/2512.24842)

### Steering Vectors
- [A Sober Look at Steering Vectors for LLMs](https://www.alignmentforum.org/posts/QQP4nq7TXg89CJGBh/a-sober-look-at-steering-vectors-for-llms)
- [Feature Guided Activation Additions (arXiv 2501.09929)](https://arxiv.org/html/2501.09929v3)
- [Representation Engineering Survey (arXiv 2502.19649)](https://janwehner.com/files/representation_engineering.pdf)
- [Programming Refusal with Conditional Activation Steering (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/e2dd53601de57c773343a7cdf09fae1c-Paper-Conference.pdf)

### Dataset Contamination
- [BenBench -- Benchmarking Benchmark Leakage](https://github.com/GAIR-NLP/benbench)
- [GSM-Symbolic (ICLR 2025)](https://arxiv.org/pdf/2410.05229)
- [Survey on Data Contamination for LLMs](https://arxiv.org/html/2502.14425v2)
- [ProverQA -- Contamination-Resistant Logic Benchmark (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/862819c227b16f9af64dd6ad6cfdf275-Paper-Conference.pdf)
- [Static to Dynamic Evaluation Survey](https://aclanthology.org/2025.emnlp-main.511.pdf)

### nnsight
- [NNsight Walkthrough](https://nnsight.net/notebooks/tutorials/get_started/walkthrough/)
- [NNsight GitHub](https://github.com/ndif-team/nnsight)
- [NNsight and NDIF Paper (arXiv 2407.14561)](https://arxiv.org/html/2407.14561v1)
- [NNsight Attribution Patching Tutorial](https://nnsight.net/notebooks/tutorials/causal_mediation_analysis/attribution_patching/)

### Transition Detection and Phase Transitions
- [Logical Phase Transitions in LLM Reasoning (arXiv 2601.02902)](https://arxiv.org/html/2601.02902)
- [Efficient Test-Time Scaling by Probing Internal States (arXiv 2511.06209)](https://arxiv.org/abs/2511.06209)
- [DeepConf -- Deep Think with Confidence](https://jiaweizzhao.github.io/deepconf/static/pdfs/deepconf_arxiv.pdf)
- [Chain-of-Thought Is Not Explainability](https://aigi.ox.ac.uk/wp-content/uploads/2025/07/Cot_Is_Not_Explainability.pdf)

### Statistical Methods and Reproducibility
- [Mechanistic Interpretability as Statistical Estimation (ICLR 2026)](https://www.arxiv.org/pdf/2510.02917)
- [High Precision for Reasoning Recipes -- FP32 Determinism](https://jianyuh.github.io/determinism/2025/06/23/FP32-Reasoning.html)
- [Model Uncertainty and Variability in LLMs](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1609097/pdf)

### Memory and Compute
- [Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json)
- [Qwen3-8B Specifications](https://apxml.com/models/qwen3-8b)
- [Compressed Activations for Memory-Efficient LLM Training (NAACL 2025)](https://aclanthology.org/2025.naacl-long.71.pdf)
