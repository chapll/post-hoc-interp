# Scaled-Up Experimental Plan

## Goal

Massively scale up the entire pipeline with DeepMath-103K, strict no-leakage
splits, broader layer coverage, and transferability tests to non-binary and
cross-domain settings.

## Data Split Design (ZERO LEAKAGE)

DeepMath-103K has ~103K problems (filtering to numeric answers yields ~60K+).
We allocate 1,100 problems across 5 non-overlapping pools. Each pool is
defined by a contiguous seed range so there is no possibility of overlap.

### Problem-Level Splits (deepmath only)

| Pool | Purpose | N problems | Seed range | Binary? | Notes |
|------|---------|-----------|------------|---------|-------|
| **A** | Direction extraction | 300 | seed=1000 | Yes | Extract knowing + post-hoc directions |
| **B** | Sleeper SFT training | 300 | seed=2000 | Yes | Train sleeper model organism |
| **C** | Evaluation / scanning | 200 | seed=3000 | Yes | Scan for post-hoc traces, steering tests |
| **D** | Binary transferability | 100 | seed=4000 | Yes | Test direction on held-out binary data |
| **E** | Non-binary transfer | 100 | seed=5000 | No | Test direction on free-form deepmath |
| **F** | Non-binary cross-domain | -- | -- | No | MATH/ARC/other reasoning datasets |

### Trace-Level Splits (all datasets)

| Index | Purpose | Used by |
|-------|---------|---------|
| trace_00, trace_02 | Training data | Pools A, B |
| trace_01 | Held-out evaluation | Pools C, D |
| trace_03, trace_04 | Extra training data | Pool A (more direction data) |

### Existing Datasets (gsm_symbolic, folio, strategyqa)

Keep current traces intact. They serve as:
- **Cross-domain transfer targets**: Test deepmath-derived direction on existing traces
- **Baseline comparison**: Compare directions extracted from deepmath vs existing datasets
- **Additional sleeper data**: Pool B can include existing dataset traces too

Split enforcement: existing datasets use the SAME trace-index split (trace_01 held out).

## Pipeline Stages

### Stage 0: DeepMath Trace Generation

Generate binary-format traces for pools A-D, non-binary for pool E.

```
Pool A: 300 problems x 2 configs x 5 traces = 3,000 traces (direction extraction)
Pool B: 300 problems x 2 configs x 3 traces = 1,800 traces (sleeper SFT)
Pool C: 200 problems x 2 configs x 3 traces = 1,200 traces (evaluation)
Pool D: 100 problems x 2 configs x 3 traces =   600 traces (binary transfer test)
Pool E: 100 problems x 1 config  x 3 traces =   300 traces (non-binary transfer)
Total: 6,900 traces
```

For Pool E (non-binary), we need a new prompt format -- no Mu/Nu tokens,
just "solve this problem and give your final answer."

**Script**: `scripts/generate_deepmath_traces.py` (new, handles all 5 pools)

### Stage 1: Scan for Post-Hoc Traces (Pool C, larger scale)

Scan ALL Pool C traces (400 problems x trace_01 = 400 traces) instead of
the current 50. More data -> better category statistics.

- Probe spacing: 20 tokens (denser than current 40)
- All layers captured for probing (not just L24)
- Output: expanded scan_results.json with per-trace classification

**Script**: `scripts/scan_posthoc_traces_v2.py` (scaled-up version)

### Stage 2: Direction Extraction (Pool A, multi-layer)

Extract knowing direction and post-hoc direction from Pool A traces at
ALL layers 0-35 (not just [18,20,22,24,26,28,32]).

**Knowing direction** (force-answer probes):
- Use Pool A trace_01 for probing (600 traces)
- Compute mean_late - mean_early at each of 36 layers
- Normalize per layer

**Post-hoc direction** (activation diff):
- Use Pool A post-hoc traces identified by scanning Pool A with trace_01
- Compute mean_plateau - mean_pretransition at each of 36 layers
- Use 50+ positions per phase per trace (up from 30)

**Script**: `scripts/extract_directions_v2.py` (new, multi-layer, larger scale)

### Stage 3: Sleeper Model Training (Pool B)

Train sleeper on Pool B data (1,800 traces from deepmath + optionally
existing dataset traces). Same approach C recipe:
- 20% triggered with shortcut reasoning
- LoRA r=16, alpha=32, 5 epochs
- Held out from all evaluation data

**New**: Also train a deepmath-only sleeper to test domain specificity.

**Scripts**: `scripts/build_sleeper_data_v2.py`, `scripts/train_sleeper.py` (reuse)

### Stage 4: Causal Steering (Pool C, multi-layer)

Run generation-based steering on Pool C traces at MULTIPLE layers.

Test layers: [16, 18, 20, 22, 24, 26, 28, 30, 32] (9 layers)
Alphas: [0, -100, +50, +100]
Traces: All 400 Pool C traces (or top 100 by category balance)

**Key experiments**:
1. Generation length steering (from transition point for post_hoc, from start for others)
2. Answer correctness under steering
3. Layer comparison: which layers show strongest causal effect with more data?

**Script**: `scripts/steer_generation_v2.py` (multi-layer version)

### Stage 5: Binary Transferability (Pool D)

Test directions extracted from Pool A on held-out Pool D binary traces.

1. Project Pool D traces onto knowing and post-hoc directions
2. Classify traces using the same scan logic
3. Compare classification quality to Pool C (should be similar if direction generalizes)
4. Run generation steering on Pool D traces

**Script**: `scripts/test_transfer_binary.py` (new)

### Stage 6: Non-Binary Transferability (Pool E)

The critical transfer test. Pool E traces use a FREE-FORM prompt (no Mu/Nu).
The model solves the problem normally and gives a numeric answer.

**Challenge**: Without Mu/Nu tokens, we can't use force-answer probing for
P(correct). Instead:

1. **Generate traces**: Model solves deepmath problems freely (no binary format)
2. **Check correctness**: Parse the model's final answer, compare to ground truth
3. **Capture activations**: Forward pass through each trace, capture all layers
4. **Project onto directions**: Use Pool A-derived directions
5. **Correlate**: Do traces where the model reaches the answer early (high projection
   on knowing direction) show the same post-hoc pattern?

**Alternative P(correct) metric for non-binary**: At each position, append a
suffix like "</think>\n\nThe answer is " and check if the model's next tokens
match the correct answer. This is analogous to force-answer but for free-form.

**Script**: `scripts/test_transfer_nonbinary.py` (new)

### Stage 7: Cross-Domain Transfer (Pool F)

Test directions on completely different datasets and formats:

1. **MATH dataset** (competition math): Free-form math reasoning, very different
   from verification format. Does the post-hoc direction still detect when the
   model has found the answer but keeps generating?

2. **ARC-Challenge**: Multiple-choice science questions with reasoning.

3. **Existing datasets without Mu/Nu**: Take gsm_symbolic/folio/strategyqa
   problems, prompt the model to solve them normally (not binary verification).
   This tests format transfer within known domains.

**Script**: `scripts/test_transfer_crossdomain.py` (new)

## Implementation Order

### Wave 1: Data Generation (parallelizable)
1. Modify `src/datasets.py` to support pool-based splitting with seed offsets
2. Add non-binary prompt format to `src/prompts.py`
3. Generate Pool A-E traces (can run in parallel on multiple GPUs)

### Wave 2: Core Analysis (sequential, depends on Wave 1)
4. Scan Pool C traces for post-hoc classification
5. Extract directions from Pool A at all 36 layers
6. Train sleeper model on Pool B

### Wave 3: Causal + Transfer (parallelizable, depends on Wave 2)
7. Multi-layer steering on Pool C
8. Binary transfer test on Pool D
9. Non-binary transfer test on Pool E
10. Cross-domain transfer on Pool F

### Wave 4: Analysis + Writeup
11. Compare layer-wise direction quality
12. Aggregate transferability results
13. Update WRITEUP.md with scaled-up findings

## Estimated Compute

| Stage | Traces | GPU hours (est) |
|-------|--------|----------------|
| Trace gen (Pools A-E) | 6,900 | ~8h (4 GPU parallel) |
| Scan Pool C | 400 | ~0.5h |
| Direction extraction | 600 x 36 layers | ~2h |
| Sleeper training | 1,800 samples | ~1.5h |
| Steering (9 layers x 4 alphas x 100 traces) | 3,600 gens | ~12h |
| Binary transfer | 200 | ~1h |
| Non-binary transfer | 300 | ~2h |
| Cross-domain | ~200 | ~2h |
| **Total** | | **~29h** |

## Key Code Changes Needed

1. **`src/datasets.py`**: Add `load_deepmath_pool(pool, n, seed)` that returns
   non-overlapping subsets. Add `load_deepmath_nonbinary(n, seed)` for Pool E.

2. **`src/prompts.py`**: Add `build_freeform_prompt(problem)` for non-binary format.

3. **`scripts/generate_deepmath_traces.py`**: Master trace generation handling
   all 5 pools with proper seed management.

4. **`scripts/extract_directions_v2.py`**: Multi-layer direction extraction
   with configurable layer range.

5. **`scripts/scan_posthoc_traces_v2.py`**: Scaled-up scanning with denser probes.

6. **`scripts/steer_generation_v2.py`**: Multi-layer steering experiment.

7. **`scripts/test_transfer_*.py`**: Three transfer test scripts.

## Leakage Prevention Checklist

- [ ] Pool seeds are non-overlapping (1000, 2000, 3000, 4000, 5000)
- [ ] Sleeper training (Pool B) shares ZERO problems with evaluation (Pool C)
- [ ] Direction extraction (Pool A) shares ZERO problems with evaluation (Pool C)
- [ ] Transfer test pools (D, E) share ZERO problems with A, B, or C
- [ ] Trace index splits maintained: train on [0,2], eval on [1]
- [ ] Non-binary traces (Pool E) use completely different prompt format
- [ ] Cross-domain datasets (Pool F) are entirely separate from deepmath
- [ ] No validation set leaks into training or evaluation
