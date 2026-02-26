---
phase: quick-003
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - paper/milestone.tex
autonomous: true

must_haves:
  truths:
    - "milestone.tex compiles (or would compile) as a valid LaTeX document"
    - "Abstract summarizes the research question, methodology, and current state"
    - "Approach section details swapped-label protocol, force-answer probing, cross-seed consistency, and planned linear probing with proper attribution"
    - "Experiments section describes datasets, evaluation methods, experimental setup, and preliminary/expected results"
    - "Future work section outlines Phases 3-6 with rationale"
  artifacts:
    - path: "paper/milestone.tex"
      provides: "Complete milestone report draft in LaTeX"
      min_lines: 200
---

<objective>
Write a CS224N milestone report draft as a LaTeX file covering Abstract, Approach, Experiments, and Future Work sections.

Purpose: The milestone is due soon and requires a comprehensive draft covering the project's research question (whether post-transition reasoning tokens in Qwen3-8B are performative), methodology (swapped-label protocol, force-answer probing, cross-seed consistency), current results (scripts complete, GPU execution pending), and future plans (Phases 3-6: probing, vector extraction, causal steering, evaluation).

Output: A single LaTeX file `paper/milestone.tex` with all four sections drafted, placeholder figure environments with descriptive captions, and proper citations.
</objective>

<execution_context>
@/home/aahmad/.claude/get-shit-done/workflows/execute-plan.md
@/home/aahmad/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@milestone.md
@src/prompts.py
@scripts/force_answer_experiment.py
@scripts/cross_seed_experiment.py
@scripts/plot_force_answer.py
@scripts/plot_cross_seed.py
@configs/force_answer.yaml
@configs/cross_seed.yaml
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write complete milestone report LaTeX file</name>
  <files>paper/milestone.tex</files>
  <action>
Create directory `paper/` if it does not exist. Write `paper/milestone.tex` as a complete LaTeX document with the following structure:

**Document setup:**
- Use `\documentclass[11pt]{article}` with standard packages: geometry (1-inch margins), amsmath, amssymb, graphicx, hyperref, natbib, booktabs, caption, subcaption
- Title: "Mechanistic Detection of Performative Reasoning in Chain-of-Thought"
- Authors: Ali Ahmad, Gorn Namchittai, Eric Cui (Stanford University, CS224N)
- Include a `\bibliography` reference to a `references.bib` file

**Abstract (~150-200 words):**
- Motivate: Large reasoning models (e.g., Qwen3-8B) generate extended chain-of-thought, but it is unclear whether all reasoning tokens are causally necessary or whether some are performative post-hoc rationalization generated after the model has internally "decided" the answer.
- Approach: We develop a mechanistic interpretability pipeline to detect the internal transition point where Qwen3-8B shifts from "searching" to "knowing," using force-answer probing (truncate thinking, force answer generation), cross-seed consistency analysis (20 traces per problem), and a swapped-label protocol (Rho/Tau tokens) to eliminate surface token confounds.
- Current state: Pipeline infrastructure complete including trace generation (~2400 traces across 3 datasets), force-answer probing scripts, and cross-seed consistency analysis scripts. GPU execution of experiments is pending. Preliminary methodology validated through successful pipeline development.
- Future: Linear probing with "Train-on-End, Test-on-Middle" protocol, performativity vector extraction via Difference in Means, and causal steering validation.

**Section 1: Approach (~2-3 pages):**
This is the core technical section. Write with specific detail:

1.1 Research Question and Motivation:
- Cox (2025) showed models encode answers before reasoning starts by probing at the last prompt token. We extend this to the ENTIRE reasoning trace to find the exact emergence point T_i.
- Arcuschin et al. (2025) demonstrated chain-of-thought is not always faithful. We aim to provide mechanistic causal evidence (not just correlational) for performative reasoning.
- Key question: Is there a sharp internal transition from "searching" to "knowing," and are tokens after this transition causally unnecessary?

1.2 Swapped-Label Protocol (Original Contribution):
- Problem: Standard Yes/No verification tasks create a confound -- probes may detect surface mentions of "Yes"/"No" in reasoning rather than abstract validity.
- Solution: Replace Yes/No output tokens with Rho/Tau. Each problem generated under two configs: Config A (Valid=Rho, Invalid=Tau) and Config B (swapped). With both in training data, Rho/Tau have zero correlation with validity label.
- Mathematical formulation: Let g in {Yes, No} be abstract validity (ground_truth), and o in {Rho, Tau} be the output token. Under the swapped protocol, P(o=Rho | g=Yes) = P(o=Rho | g=No) = 0.5, forcing probes to find the abstract validity signal.
- Reference the prompts.py implementation and the LABEL_MAPPINGS data structure.

1.3 Force-Answer Transition Probing (Original Contribution):
- Method: Given a trace with thinking tokens t_1, ..., t_n between <think> and </think>, truncate at position k, append </think>\n, and let the model freely generate its answer.
- Probe ~20 evenly-spaced positions per trace (min spacing 5 tokens).
- At each probe point, record: (1) whether generated answer matches ground truth, (2) logit rank of correct answer token, (3) probability of correct answer token.
- KV-cache optimization: Process probes in descending position order, using DynamicCache.crop() to avoid redundant recomputation. Deepcopy cache before generation to preserve state.
- Expected output: Accuracy-vs-normalized-position sigmoid curve revealing the transition.
- Cite force_answer_experiment.py.

1.4 Cross-Seed Consistency Analysis (Original Contribution):
- Motivation: A single trace may show a transition by chance. If the transition is a genuine property of the model's reasoning process, it should be consistent across different random seeds.
- Method: For each of ~41 originally-correct problems, generate 20 traces with different seeds (temperature=0.6, top_p=0.95, top_k=20). Probe all 20 at ~20 positions.
- Interpolation: Map each trace's accuracy curve to a common 21-point normalized grid (0 to 1) for cross-seed comparison.
- Metrics: Per-problem sharpness (max absolute slope of mean accuracy curve), position-wise variance, pairwise Spearman correlation between seed curves.
- Cite cross_seed_experiment.py and generate_seed_traces.py.

1.5 Planned: Linear Probing with "Train-on-End, Test-on-Middle" (From Cox 2025):
- Train logistic regression probe on residual stream activations from the 3 tokens before the final </think> of correct traces (the "Pre-Handoff Window").
- Apply fixed probe to every intermediate token to get per-position confidence curve.
- Transition point T_i: first position where probe confidence consistently exceeds 0.95 (sustained K=5 consecutive positions).
- Selectivity control: random-label probes trained with identical procedure. Warning if control AUROC > 0.60.
- Note: This follows Cox's protocol extended to per-token probing across the full trace.

1.6 Planned: Causal Steering (Original Extension):
- Extract performativity vector v_perf via Difference in Means between concise and performative trace clusters, aligned at detected T_i.
- Inject +v_perf into concise traces (induce verbosity) and -v_perf into verbose traces (induce early termination).
- Alpha sweep from -8 to +8 for dose-response curves.
- Controls: random-vector (same norm, random direction), sham (patching from same-answer traces).
- This extends Cox's probe-weight steering to a data-driven performativity direction.

1.7 Baselines:
- Entropy-based transition detection (from logits at each position)
- Greedy pruning (remove tokens one at a time, check answer preservation)
- Random transition point and midpoint baselines

**Section 2: Experiments (~1.5-2 pages):**

2.1 Data:
- GSM-Symbolic (HuggingFace apple/GSM-Symbolic): Mathematical verification. Fresh instances from HF to avoid contamination. Binary format: "Is the answer X?" with plausible distractors (close perturbations, not random numbers).
- FOLIO (HuggingFace yale-nlp/FOLIO): Logical verification with first-order logic premises. Binary True/False.
- StrategyQA (HuggingFace Elriggs/openai-StrategyQA): World knowledge verification. Binary True/False.
- All datasets: 50/50 True/False balance, ~300 problems each, expanded to ~2400 traces with swapped-label protocol (2 configs x 3 traces per problem, plus 20 traces per problem for cross-seed subset).
- Include a table summarizing dataset statistics.

2.2 Evaluation Method:
- Force-answer accuracy: fraction of probe points where truncated model produces correct answer.
- Transition sharpness: maximum absolute slope of mean accuracy-vs-position curve.
- Cross-seed consistency: mean pairwise Spearman correlation of accuracy curves across 20 seeds.
- (Planned) Probe AUROC: area under ROC curve for linear probe predictions per layer.
- (Planned) Token reduction ratio: 1 - T_i/n (fraction of tokens identified as performative).
- (Planned) Steering success rate: fraction of traces where steering produces predicted behavioral change.

2.3 Experimental Details:
- Model: Qwen/Qwen3-8B, 36 layers, 32 query heads, 8 KV heads (GQA), 4096 hidden dim, SwiGLU MLP.
- Generation: enable_thinking=True, Temperature=0.6, TopP=0.95, TopK=20, MinP=0.
- Hardware: A100 80GB GPUs, 4-8 GPU parallel execution via process spawning (independent model copies).
- Activation storage: BF16 via safetensors, chunked capture (12 layers per pass, 3 passes for all 36 layers).
- Force-answer probing: ~20 positions per trace, 15 max new tokens per probe, KV-cache-efficient with DynamicCache.crop().
- Cross-seed: 20 traces per problem, 21-point interpolation grid, selection seed 137.

2.4 Results:
- State that GPU execution is pending for force-answer and cross-seed experiments. Scripts are complete and validated.
- Include placeholder figures:
  - Figure 1: Aggregate accuracy-vs-normalized-position curve (expected sigmoid shape). Three panels for GSM-Symbolic, FOLIO, StrategyQA. X-axis: normalized position in thinking block (0-1). Y-axis: accuracy. Expected: sharp transition from ~0.5 (random) to ~1.0 (correct).
  - Figure 2: Per-problem cross-seed overlay. Two panels: accuracy and probability. 20 faint seed curves with bold mean. Expected: tight clustering around transition point if transition is a robust phenomenon.
  - Figure 3: Cross-seed consistency metrics. Sharpness histogram across problems. Expected: bimodal distribution (sharp transitions for "clean" problems, flat for "absent" transitions).
- Include a table template for quantitative results (to be filled after GPU execution):
  | Dataset | N Problems | Median Transition | Sharpness | Cross-Seed Spearman |
- Describe what the expected results would look like and what they would mean for the research question.

**Section 3: Future Work (~0.5-1 page):**
- Phase 3 (Probing and Transition Detection): Linear probes with Train-on-End, Test-on-Middle protocol. Per-layer AUROC curves. Selectivity controls. Cosine similarity dynamics. Cox 2x2 framework classification.
- Phase 4 (Differential Analysis): Performativity vector extraction. Cross-dataset vector comparison. IPHR reversed pairs for bias detection.
- Phase 5 (Causal Steering): Steering validation with dose-response curves. Sham and random controls. Comparison to entropy-based and greedy pruning baselines.
- Phase 6 (Final Evaluation): Ablation studies. Paper-ready figures. Full quantitative analysis with FDR correction.
- Emphasize that the key missing piece is the causal evidence -- force-answer probing and probes provide correlational evidence; steering will provide causal evidence that post-transition tokens are genuinely performative.

**Figures:**
For each placeholder figure, use this pattern:
```latex
\begin{figure}[t]
  \centering
  % \includegraphics[width=\linewidth]{figures/figure_name.pdf}
  \fbox{\parbox{0.9\linewidth}{\centering\vspace{2cm}
    [Description of what this figure will show]
  \vspace{2cm}}}
  \caption{Caption text.}
  \label{fig:label}
\end{figure}
```

**Citations:**
Use \cite{} with keys: cox2025posthoc, arcuschin2025cot, wei2022cot, qwen3, nnsight, deepseekr1, gsmsymbolic, folio, strategyqa. These will reference the bibliography file.

**Writing style:**
- Academic but accessible. CS224N milestone level, not a full conference paper.
- Be specific about methods (equations, algorithms, implementation details).
- Clearly distinguish original contributions from prior work.
- Acknowledge that results are pending GPU execution -- this is expected for a milestone.
- Do NOT use emoji anywhere.
  </action>
  <verify>
File exists at paper/milestone.tex. Check it is valid LaTeX by verifying: (1) begins with \documentclass, (2) has \begin{document} and \end{document}, (3) contains \begin{abstract}, \section{Approach}, \section{Experiments}, \section{Future Work} sections, (4) has at least 200 lines.
  </verify>
  <done>paper/milestone.tex exists with all four required sections (Abstract, Approach, Experiments, Future Work) drafted with specific technical content, placeholder figures, and proper citations. The document is ready for review and iteration.</done>
</task>

<task type="auto">
  <name>Task 2: Create bibliography file with cited references</name>
  <files>paper/references.bib</files>
  <action>
Create `paper/references.bib` with BibTeX entries for all works cited in the milestone. Include:

- cox2025posthoc: Cox (2025) "Post-hoc reasoning in chain of thought" -- arXiv preprint
- arcuschin2025cot: Arcuschin et al. (2025) "Chain-of-Thought Reasoning In The Wild Is Not Always Faithful" -- arXiv preprint
- wei2022cot: Wei et al. (2022) "Chain-of-thought prompting elicits reasoning in large language models" -- NeurIPS 2022
- qwen3: Qwen Team (2025) "Qwen3 Technical Report" -- arXiv preprint (use Qwen/Qwen3 from HuggingFace)
- nnsight: Fiotto-Kaufman et al. (2024) "NNsight and NDIF: Democratizing Access to Foundation Model Internals" -- arXiv preprint
- deepseekr1: DeepSeek-AI (2025) "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" -- arXiv preprint
- gsmsymbolic: Mirzadeh et al. (2024) "GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models" -- Apple ML Research / arXiv
- folio: Han et al. (2022) "FOLIO: Natural Language Reasoning with First-Order Logic" -- arXiv preprint
- strategyqa: Geva et al. (2021) "Did Aristotle Use a Laptop? A Question Answering Benchmark with Implicit Reasoning Strategies" -- TACL 2021

Look up correct arXiv IDs and author lists. Use @article or @inproceedings as appropriate. If exact details are uncertain, use reasonable placeholders that can be corrected later.
  </action>
  <verify>File exists at paper/references.bib. Contains at least 9 BibTeX entries. Each entry has title, author, and year fields.</verify>
  <done>paper/references.bib exists with BibTeX entries for all cited works, ready for natbib/bibtex compilation.</done>
</task>

</tasks>

<verification>
- paper/milestone.tex exists and contains all four sections (Abstract, Approach, Experiments, Future Work)
- paper/references.bib exists with entries for all cited works
- The LaTeX document is structurally valid (documentclass, begin/end document, sections)
- Approach section properly attributes Cox (2025) and Arcuschin (2025) methods vs. original contributions
- Placeholder figures have descriptive captions explaining expected results
</verification>

<success_criteria>
Complete milestone report draft in paper/milestone.tex with: (1) Abstract motivating performative reasoning detection, (2) Approach detailing swapped-label protocol, force-answer probing, cross-seed consistency, and planned probing/steering with proper attribution, (3) Experiments describing datasets, evaluation metrics, setup, and preliminary/expected results with placeholder figures, (4) Future work outlining Phases 3-6. Bibliography in paper/references.bib with all cited references.
</success_criteria>

<output>
After completion, create `.planning/quick/003-write-milestone-report-drafts/003-SUMMARY.md`
</output>
