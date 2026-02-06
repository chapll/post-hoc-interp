# Phase 1: Foundation and Validation - Context

**Gathered:** 2026-02-04
**Status:** Ready for planning

<domain>
## Phase Boundary

Load Qwen3-8B with nnsight, prepare three datasets in binary verification format, build config/storage/multi-GPU infrastructure, and validate the full pipeline end-to-end on 5-10 problems (generation through probe-based transition detection) before any scale-up. All code in `src/experimental/gorn/` with `scripts/` for runnable scripts and `lib/` for reusable library code.

</domain>

<decisions>
## Implementation Decisions

### Dataset Conversion
- 50/50 True/False balance across all three datasets, enforced by capping to the minority class count after conversion
- GSM-Symbolic: "Is the answer X?" where X = correct answer (50%) or plausible distractor (50%). Distractors are small perturbations of the correct numerical answer (e.g., 100 -> 98 or 102), NOT random numbers. Model must actually verify, not just reject obvious wrong answers. This prevents sycophancy bias where probes learn "user suggestion" instead of "verified truth."
- FOLIO: Filter out the "Unknown" class entirely. Keep only Valid/Invalid conclusions. Sample 50/50 balanced from those.
- StrategyQA: Sample equal yes/no from the dataset. Cap to minority class count if one class has fewer examples.
- All datasets: check class distribution before sampling. If any class has fewer than target count, reduce the other class to match.
- Prompt format: full problem text + explicit verification question as a single self-contained prompt. Include explicit reasoning instruction ("Please think step-by-step to verify...") to ensure non-trivial reasoning traces. This is required IN ADDITION TO `enable_thinking=True`.
- Rationale for explicit instruction: without it, a capable model may answer trivially ("Yes"/"No") without entering thinking mode for binary verification tasks. The instruction ensures traces exist to probe. It also sharpens the performativity hypothesis: if the model reasons beyond when it has verified the answer, that excess reasoning is exactly the performativity we measure.

### Activation Capture Scope
- Default layer subset for full-sequence capture: [0, 4, 8, 12, 16, 20, 24, 28, 32, 35] (every 4th + last, 10 layers)
- Last-token activations at all 36 layers always stored
- Storage precision: BF16 on disk (model's native precision)
- Top-50 logits (values + indices) per position stored alongside activations (avoids re-running forward pass for entropy baseline). Pre-computed entropy scalar per position also stored.
- Attention weights: NOT saved by default, recomputed on demand when needed. Config option to enable attention weight saving.

### Script Invocation
- Direct script invocation: `uv run python src/experimental/gorn/scripts/trace_gen.py --config configs/trace_gen.yaml`
- Standard argparse flags for CLI arguments (--config, --gpus, --num-traces, etc.)
- GPU IDs specifiable both in YAML config AND via --gpus CLI flag. CLI flag overrides config.
- Directory structure:
  - `src/experimental/gorn/scripts/` -- runnable experiment scripts
  - `src/experimental/gorn/lib/` -- reusable library code (model.py, probe.py, etc.)
  - `src/experimental/gorn/configs/` -- YAML config files
  - `src/experimental/gorn/experiments/` -- experiment OUTPUTS (not code)
  - `src/experimental/gorn/data/` -- datasets and traces

### Validation Criteria
- Full mini-experiment on 5-10 problems before scaling: generation, storage round-trip, probe training (full Train-on-End/Test-on-Middle protocol), transition detection
- Validate both single-GPU and multi-GPU (2+ GPUs) spawning in Phase 1
- nnsight + Qwen3 compatibility: must see correct activation shapes [batch, seq_len, 4096] at all 36 layers, successful thinking-mode generation, and token boundary parsing
- If nnsight fails: remove nnsight entirely and replace with raw PyTorch hooks (register_forward_hook). Do NOT implement as a silent fallback -- rip out and replace completely. No fallback mechanisms in code.

### Probing Protocol (for validation)
- "Train-on-End, Test-on-Middle": train single probe per layer on Pre-Handoff Window (3 tokens before FINAL `</think>`) of correct traces. Apply fixed probe to every intermediate token inside `<think>` blocks.
- Context: feed full sequence including `<think>` and `</think>` in forward pass
- Readout: skip probing `<think>` token itself (k=0, geometric outlier). Start at first natural language token (k=1).
- Probe only inside `<think>` blocks. Never probe inter-block text.
- Structural filter: primarily keep single-block traces. Track multi-block rate.

### Prompt-Label Consistency
**NOTE:** This section describes the Phase 1 approach (Yes/No prompts). Phase 2 (Plan 02-06) superseded this with the **swapped-label protocol** using Rho/Tau tokens to eliminate the confound where probes detect surface token mentions. See `.planning/phases/02-trace-generation-at-scale/02-CONTEXT.md` for details.

**Phase 1 (superseded):**
- Every verification prompt ends with explicit "(Yes or No)" to standardize the binary framing.
- ground_truth is a string: "Yes" or "No" (never a boolean). This matches what the probe is trained to detect.

**Phase 2+ (current approach):**
- Verification prompts use Rho/Tau output tokens instead of Yes/No
- Each problem generates Config A (Valid=Rho, Invalid=Tau) and Config B (swapped) variants
- `ground_truth` remains "Yes"/"No" as the abstract validity label for probe training
- Probes train on `ground_truth` (abstract validity), decoupled from output token (Rho/Tau)
- This prevents probes from detecting surface mentions of "yes"/"no" in reasoning traces

### Claude's Discretion
- Exact argparse flag names and help text
- Config YAML structure and key naming conventions
- Internal code organization within lib/ (helper module boundaries)
- Error message formatting (must be plain text, no emoji)
- Exact perturbation strategy for GSM-Symbolic distractors (as long as they are small and plausible)

</decisions>

<specifics>
## Specific Ideas

- "Unlike Sondej (2025), who uses readout-based prompting ('Therefore the answer is...') which may artificially force decision collapse, we employ passive linear probing on the residual stream. This allows us to detect latent knowledge states before they are explicitly formulated as output tokens." -- This framing should be reflected in the code's documentation and the paper.
- Top-50 logits stored per position enables cross-validation of probe confidence vs output probability (Sondej-style contrast) without GPU recomputation.
- The explicit reasoning instruction ("Please think step-by-step to verify...") is scientifically justified: it forces trace generation while sharpening the performativity hypothesis. An efficient model should stop reasoning after verification; continued reasoning under instruction IS the performativity being measured.
- Distractor quality for GSM-Symbolic is crucial: "Is the answer 7?" for a problem whose answer is 100 would be trivially rejectable. Distractors must require computation to distinguish from the correct answer.

</specifics>

<deferred>
## Deferred Ideas

None -- discussion stayed within phase scope.

</deferred>

---

*Phase: 01-foundation-and-validation*
*Context gathered: 2026-02-04*
