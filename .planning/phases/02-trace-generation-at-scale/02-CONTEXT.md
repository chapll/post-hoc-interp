# Phase 2: Trace Generation at Scale - Context

**Gathered:** 2026-02-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Generate 1500+ traces across three datasets using Qwen3-8B thinking mode, with fresh GSM-Symbolic instances for contamination mitigation and original instances as contamination diagnostic controls. Then capture activations for all valid traces via teacher-forced re-computation with chunked layer passes.

The pipeline is split into two separate scripts within this phase (generation first, then activation capture), but both are part of Phase 2. No roadmap restructuring needed.

</domain>

<decisions>
## Implementation Decisions

### Pipeline Architecture: Decoupled Generation and Analysis

This is the single most important decision for Phase 2. The pipeline is split into two strictly separate stages:

**Stage 1 (this phase): Generation**
- Run Qwen3-8B to generate thinking-mode traces
- Save token IDs + text + metadata to disk (JSON files, small -- KB per trace)
- Optimize for throughput and fault tolerance

**Stage 2 (separate script, same phase): Activation Capture**
- Load saved token sequences from disk
- Feed them back into the model via teacher forcing (prefill, not autoregressive)
- Capture activations in chunked layer passes to avoid OOM
- Save activations to safetensors (large -- GB per trace)

**Why decouple:**
1. Fault isolation: generation is stable; activation capture is OOM-prone. If capture crashes, generated tokens are safe on disk.
2. Garbage filter opportunity: validate trace quality (has think blocks, reasonable length, not degenerate) before committing to expensive activation capture.
3. Hardware flexibility: generation is compute-bound (fast GPU cores); capture is memory-bound (large VRAM). Can use different hardware profiles.
4. Speed: re-computation via prefill processes all tokens in parallel (single matrix op), taking only 1-5% of original generation time. Not a meaningful cost.
5. Correctness: transformer forward pass is deterministic. Same tokens in = same activations out. Verified by matching logits between generation and re-computation passes.

**Activation capture details:**
- Teacher forcing: feed complete token sequence as a prompt
- Chunk layers: capture layers 0-N in one forward pass, save to CPU/disk, clear cache, repeat for next chunk
- Must preserve exact position_ids and attention mask from original generation
- Verify correctness by asserting last-token logits match between original generation and re-computation
- Capture ALL 36 layers (full-sequence residuals). Chunked capture makes this feasible at the same VRAM cost as 1 layer -- it just takes more passes and disk space. Capturing everything prevents having to re-run a multi-day job because a reviewer asks about Layer 14.

### Failure Handling

**Single trace failure (OOM, nnsight error, malformed generation):**
- Skip and log. Continue with remaining work.
- Log must capture the specific error type (OOM vs. logic error) to distinguish hardware limits from bugs.
- All downstream metrics/counts must exclude skipped traces. No silent inclusion of failed items in aggregates.

**Worker crash (CUDA error, process killed):**
- Other workers continue. The crashed worker's remaining problems are reported as missing.
- Recovery: scan output directory for missing IDs, re-run only the missing subset.

**Malformed traces (no think block, empty generation, unclosed think block):**
- Save but quarantine. Route to a separate `failed/` directory (mirrored structure to `valid/`).
- Only save metadata JSON for failed traces (no point saving activations for garbage).
- Downstream scripts iterate over `valid/` only -- zero filtering logic needed.
- Separate analysis script can inspect `failed/` for failure rate reporting.

### GSM-Symbolic Contamination Mitigation

**Fresh instance generation:**
- Use GSM-Symbolic's own template system (from apple/ml-gsm-symbolic repository)
- Templates encode mathematical constraints (e.g., divisibility requirements) that prevent blind numerical resampling from breaking problem logic
- Generate exactly 1 fresh variant per template (100 templates = 100 fresh problems)
- Use a fixed seed for the data generation step so fresh instances are reproducible

**Contamination diagnostic:**
- Run traces on BOTH original (potentially contaminated) AND fresh instances
- Compare accuracy to quantify contamination effect (the "memorization gap")
- Label each problem with `is_fresh: bool` in metadata
- Scientific value: "Does the model think differently when it remembers the answer?" -- shorter traces, more confidence, post-hoc justification on contaminated data vs. genuine derivation on fresh data

**Dataset sizes:**
- GSM-Symbolic: 100 fresh + 100 original = 200 problems (but each gets 5 traces = 1000 traces)
- FOLIO: 100 problems x 5 traces = 500 traces
- StrategyQA: 100 problems x 5 traces = 500 traces
- Total: ~400 problems, ~2000 traces (increased from original 300/1500 to accommodate original+fresh GSM)

**Note:** The exact problem count may need adjustment based on available FOLIO/StrategyQA dataset sizes. The 50/50 balance constraint from Phase 1 applies.

### Sampling and Reproducibility

**Per-trace seed derivation:**
- Hash-based: `seed = int(hashlib.md5((problem_id + str(trace_index)).encode()).hexdigest(), 16) % (2**32)`
- Stateless: any problem can be generated on any worker in any order without global state
- Robust to dataset reshuffling, worker reassignment, partial reruns

**Sampling parameters (Qwen3 thinking mode defaults):**
- Temperature = 0.6
- TopP = 0.95
- TopK = 20
- MinP = 0.0
- These are Qwen3's officially recommended thinking mode parameters
- Using defaults removes "bad sampling" as a confounding variable

**Reproducibility level: Statistical (not bit-exact)**
- Set standard seeds (random, numpy, torch.manual_seed, torch.cuda.manual_seed_all)
- Do NOT enforce `torch.use_deterministic_algorithms(True)` -- it can be 2-3x slower
- Same seed on same hardware gives similar but not necessarily identical outputs
- Sufficient for ICLR-level reproducibility

**Per-trace metadata recording:**
- Every trace file stores the exact sampling parameters used (temperature, top_p, top_k, min_p)
- Also stores: seed, model name, max_new_tokens, problem_id, dataset_name, ground_truth, is_fresh, generation timestamp, token count, think block count

### Resource and Batching Strategy

**GPU allocation:**
- Configurable via `--gpus` CLI flag (e.g., `--gpus 0,1,2,3`)
- Default: use all visible CUDA devices
- Each GPU loads its own independent model copy (~16GB in BF16) and processes its share of problems

**Work distribution: Static split with random shuffle**
- Deterministically shuffle all problems before splitting across GPUs
- Shuffle prevents any single GPU from getting all the "hard" problems
- Static split means: if GPU 3 crashes, you know exactly which shard to re-run
- Shuffle seed is a fixed value from config (not per-trace seed)

**Long sequences:**
- max_new_tokens = 4096 (ensures most traces complete with `</think>` token)
- No activation capture in this phase, so OOM risk is much lower (generation KV cache only, no stored residuals)
- If a generation OOMs: log, skip, continue (per failure handling above)

### Output Organization

**Directory structure: dataset/problem_id/trace_N**
```
data/
  valid/
    gsm_symbolic/
      problem_001/
        trace_0.json        # token IDs + text + metadata
        trace_1.json
        ...trace_4.json
      problem_002/
        ...
    folio/
      ...
    strategyqa/
      ...
  failed/
    gsm_symbolic/           # mirrored structure
      problem_088/
        trace_2.json        # metadata only (what went wrong)
```

**Valid vs. failed: Mirrored directory structure**
- `data/valid/` and `data/failed/` use identical layouts
- Moving/inspecting traces is a simple root swap
- Downstream scripts iterate `data/valid/` only

**File format:**
- Generation output: JSON files containing token_ids (list of ints), generated_text (string), metadata (dict). Atomic writes via `.tmp` then `os.rename()`.
- Activation output: safetensors files in a SEPARATE tree from the JSON traces.

**Separate storage trees for traces vs. activations:**
```
data/traces/gsm_symbolic/problem_001/trace_0.json       # small, high-value
data/activations/gsm_symbolic/problem_001/trace_0.safetensors  # large, reproducible
```
- Traces (JSON) and activations (safetensors) share the same relative path structure but different roots
- Allows different storage tiers (e.g., traces on backed-up home, activations on fast scratch)
- Can `rm -rf data/activations` to reclaim space without touching traces
- Can `rsync` just `data/traces/` to a laptop without pulling TB of tensors
- Loading code: swap root path and extension to go from trace to activation

**Storage location:**
- Configurable via config/env (DATA_DIR for traces, ACTIVATION_DIR for activations)
- Default to paths under the project that can be overridden per-machine

**Progress tracking: File-based**
- Existence of a trace file = that trace is done
- Resume = skip existing files (`if os.path.exists(outfile): continue`)
- No manifest file during the run
- Status reporting via a separate `check_status.py` script that scans the output directory on demand

### Quality Gate Between Generation and Activation Capture

An explicit quality-check script runs after generation and before activation capture. It scans all traces in `data/traces/valid/` and produces a report (CSV), but does NOT automatically move or delete files.

**What to check (objective criteria only):**
- Does the trace contain at least one `<think>` block?
- Did generation terminate naturally (EOS token) or hit MAX_TOKENS? (hitting MAX_TOKENS = suspicious)
- Is the trace degenerate (e.g., repeated token loops)?

**What NOT to check (subjective -- just flag):**
- "Reasonable length" is subjective. Flag outliers (e.g., top/bottom 5% by token count) but do not auto-reject.
- Reasoning quality. Not automatable.

**Output:** A CSV/report listing each trace with objective pass/fail flags and subjective warning flags. The researcher reviews the report and manually confirms which flagged traces to move to `failed/` using a companion script.

**Rationale:** Human-in-the-loop prevents false rejections. Automated hard filters risk discarding interesting edge cases (e.g., a legitimately short trace where the model verified instantly). The check script is diagnostic, not destructive.

### Claude's Discretion
- Exact JSON schema for trace files (field names, nesting)
- Exact argparse flag names and help text
- Internal chunking/buffering details within workers
- Error message formatting (plain text, no emoji)
- Exact implementation of GSM-Symbolic template parsing (may vary based on what the official repo provides)
- Whether to use multiprocessing.spawn or separate process invocations for workers
- Logging format and verbosity levels

### Swapped-Label Probing Protocol (Added 2026-02-05)

**The Confound:**
The model uses "yes" and "no" during reasoning (e.g., "Assume the answer is yes..."). A probe trained on Yes/No ground truth will partially learn to detect these surface tokens, not the abstract validity signal.

**The Solution:**
Replace Yes/No verification prompts with Rho/Tau output tokens. Generate each problem under two complementary label mappings:
- Config A: Valid=Rho, Invalid=Tau
- Config B: Valid=Tau, Invalid=Rho

With both configs in training data, Rho/Tau have zero correlation with the validity label, forcing the probe to find the abstract logical signal.

**Per-Dataset Prompt Templates:**

GSM-Symbolic:
```
{question}

Is the answer {presented_answer}? If it is correct, output exactly the word: {valid_word}. If it is not correct, output exactly the word: {invalid_word}.
Please think step-by-step to verify whether the answer is correct.
```

FOLIO:
```
Given the following premises:
{premises}

Does the following conclusion logically follow from the premises: "{conclusion}"?
If it logically follows, output exactly the word: {valid_word}. If it does not follow, output exactly the word: {invalid_word}.
Please think step-by-step to verify.
```

StrategyQA:
```
Answer the following question: {question}
If the answer is yes, output exactly the word: {valid_word}. If the answer is no, output exactly the word: {invalid_word}.
Please think step-by-step to verify your answer.
```

**Implementation:**
- `lib/prompts.py`: New module with `build_verification_prompt()` function
- `lib/datasets.py`: Loaders return base problem dicts without prompts
- `generate_traces.py`: Expands each problem into A/B config variants before generation
- Trace metadata includes: `prompt_config` ("A" or "B"), `label_mapping`, `valid_token`, `invalid_token`

**Trace Count Impact:**
- ~400 base problems x 2 configs x 3 traces = ~2400 traces (reduced from 5 traces to 3 to offset doubling)
- `ground_truth` remains Yes/No (abstract validity); output tokens are Rho/Tau

</decisions>

<specifics>
## Specific Ideas

- The contamination diagnostic (original vs. fresh GSM-Symbolic) could be one of the most interesting parts of the paper: "Does the model think differently when it remembers the answer?" Shorter traces, more confidence, and post-hoc justification on contaminated data vs. genuine derivation on fresh data.
- The decoupled pipeline (generate first, capture later) is standard practice in large-scale interpretability work (TransformerLens, nnsight). It makes the system robust and debuggable.
- The hash-based seed scheme means any individual trace can be regenerated in isolation without needing to know the global problem ordering. This is critical for re-running failed traces.
- The "quarantine protocol" for failed traces (separate directory, metadata only) means the valid/ directory is a guaranteed-clean dataset with no filtering logic needed downstream.

</specifics>

<deferred>
## Deferred Ideas

- **Temperature sweep for trace diversity:** Trying 2-3 temperature values was considered but rejected in favor of using Qwen3 defaults to maintain scientific control. Could revisit if trace diversity is insufficient with 5 traces per problem.

</deferred>

---

*Phase: 02-trace-generation-at-scale*
*Context gathered: 2026-02-04*
