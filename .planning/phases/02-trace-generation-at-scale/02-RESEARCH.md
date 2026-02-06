# Phase 2: Trace Generation at Scale - Research

**Researched:** 2026-02-04
**Domain:** Multi-GPU LLM trace generation, activation capture, contamination mitigation
**Confidence:** HIGH

## Summary

Phase 2 is a two-script pipeline: (1) generate ~2000 thinking-mode traces from Qwen3-8B across three datasets with multi-GPU parallelism, and (2) capture all 36 layers of residual stream activations via teacher-forced prefill with chunked nnsight trace passes.

The existing Phase 1 codebase provides a solid foundation. The core components -- `load_model()`, `generate_trace()`, `capture_activations()`, `save_trace()`, dataset loaders, multi-GPU `run_parallel()`, and config loading -- are all implemented and validated. Phase 2 primarily needs: (a) production generation script with fault tolerance and resume, (b) GSM-Symbolic fresh instance loading from the HuggingFace dataset, (c) quality-check script, (d) activation capture script with chunked layer passes, and (e) updated storage paths for the decoupled trace/activation layout.

**Primary recommendation:** Build three scripts (`generate_traces.py`, `check_traces.py`, `capture_activations.py`) that compose existing `lib/` modules with new orchestration logic. The hardest parts are fault-tolerant multi-GPU process management and the chunked nnsight capture loop. Both are tractable given Phase 1 foundations.

## Standard Stack

The stack is fully locked by prior decisions. No new libraries are needed.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| nnsight | 0.5.15 | Activation capture via trace context | Already installed, validated in Phase 1 |
| transformers | latest | Qwen3-8B model and tokenizer | HF model loading, chat template with enable_thinking |
| safetensors | latest | BF16 tensor storage | Fast, safe, validated round-trip in Phase 1 |
| torch | latest | GPU compute, multiprocessing | Foundation; mp.Process for fault-isolated workers |
| OmegaConf | latest | Configuration | Phase 1 config system, CLI overrides |
| datasets | latest | HuggingFace dataset loading | GSM-Symbolic, FOLIO, StrategyQA loaders |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| hashlib (stdlib) | N/A | Per-trace seed derivation | MD5 hash of problem_id + trace_index |
| csv (stdlib) | N/A | Quality check report output | check_traces.py CSV report |
| json (stdlib) | N/A | Trace metadata files | Generation output format |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| mp.Process | mp.spawn | mp.spawn kills ALL workers on any failure; mp.Process allows independent isolation |
| mp.Process | subprocess.Popen | More isolation but harder to pass Python objects; mp.Process is sufficient |
| JSON for traces | safetensors for traces | JSON allows human inspection of token IDs and metadata; safetensors only for large tensors |

**Installation:** No new packages needed. All dependencies from Phase 1.

## Architecture Patterns

### Recommended Project Structure
```
src/experimental/gorn/
  scripts/
    generate_traces.py        # Script 1: Multi-GPU trace generation
    check_traces.py            # Script 2: Quality gate (CSV report)
    capture_activations.py     # Script 3: Chunked activation capture
  lib/
    model.py                   # Existing: load_model, generate_trace, capture_activations
    datasets.py                # Existing: dataset loaders (needs GSM-Symbolic fresh extension)
    tokens.py                  # Existing: think boundary parsing
    storage.py                 # Existing: safetensors save/load (needs path updates)
    multiprocess.py            # Existing: needs mp.Process alternative to mp.spawn
    config.py                  # Existing: OmegaConf config loading
    utils.py                   # Existing: seed, logging
  configs/
    base.yaml                  # Existing: base config
    generate.yaml              # New: generation-specific overrides
    capture.yaml               # New: capture-specific overrides
  data/
    traces/                    # Generation output (JSON, small)
      valid/
        gsm_symbolic/
        folio/
        strategyqa/
      failed/
        gsm_symbolic/
        folio/
        strategyqa/
    activations/               # Activation output (safetensors, large)
      gsm_symbolic/
      folio/
      strategyqa/
```

### Pattern 1: Fault-Tolerant Multi-GPU Workers via mp.Process

**What:** Use `multiprocessing.Process` instead of `mp.spawn` for worker isolation. Each worker is an independent process that can crash without killing others.

**When to use:** Any time worker crash should not abort the entire job.

**Why not mp.spawn:** The current `run_parallel()` in `multiprocess.py` uses `mp.spawn`, which terminates ALL workers when ANY single one fails. This is unacceptable for a multi-hour generation job. The CONTEXT explicitly requires: "Other workers continue. The crashed worker's remaining problems are reported as missing."

**Example:**
```python
import torch.multiprocessing as mp

def run_parallel_isolated(worker_fn, items, gpu_ids, cfg):
    """Run workers as independent processes with fault isolation."""
    chunks = split_work(items, len(gpu_ids))

    processes = []
    for rank, (gpu_id, chunk) in enumerate(zip(gpu_ids, chunks)):
        p = mp.Process(
            target=_worker_wrapper,
            args=(worker_fn, rank, gpu_id, chunk, cfg),
        )
        p.start()
        processes.append((rank, gpu_id, p))

    # Wait for all processes, collect results
    failed = []
    for rank, gpu_id, p in processes:
        p.join()
        if p.exitcode != 0:
            failed.append((rank, gpu_id, p.exitcode))

    if failed:
        for rank, gpu_id, code in failed:
            logger.error(
                "Worker rank %d on GPU %d exited with code %d", rank, gpu_id, code
            )

    return failed
```

**Confidence:** HIGH -- `mp.Process` is standard Python, well-documented behavior.

### Pattern 2: File-Based Resume

**What:** Check for existing output files before processing each work item. Skip items that already have output files on disk.

**When to use:** Any long-running job that may be interrupted.

**Example:**
```python
def process_trace(problem, trace_idx, output_dir, model, tokenizer, cfg):
    """Generate a single trace, skipping if output already exists."""
    trace_path = get_trace_json_path(output_dir, problem, trace_idx)

    if os.path.exists(trace_path):
        logger.info("Skipping %s trace %d (already exists)", problem["problem_id"], trace_idx)
        return

    # Generate trace...
    # Write to temp file, then atomic rename
    tmp_path = trace_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(trace_data, f, indent=2)
    os.rename(tmp_path, trace_path)  # Atomic on same filesystem
```

**Confidence:** HIGH -- standard pattern, used across ML pipelines.

### Pattern 3: Chunked nnsight Activation Capture

**What:** Run multiple nnsight trace passes per token sequence, each capturing a different subset of layers. Save each chunk's activations to CPU/disk before starting the next chunk.

**When to use:** Capturing ALL 36 layers of full-sequence residuals where single-pass capture may OOM on smaller GPUs.

**Critical nnsight 0.5 constraint:** Layers MUST be accessed in sequential order within a trace context. OutOfOrderError is thrown otherwise. This means each chunked trace still iterates through ALL layers in order, but only calls `.save()` on the layers in the current chunk.

**Example:**
```python
def capture_activations_chunked(
    model, token_ids, chunk_size=12, num_layers=36
):
    """Capture all layers via chunked forward passes."""
    all_results = {}
    num_chunks = (num_layers + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_layer = chunk_idx * chunk_size
        end_layer = min(start_layer + chunk_size, num_layers)
        chunk_layers = set(range(start_layer, end_layer))

        saved = {}
        with model.trace(token_ids) as tracer:
            # MUST access layers in sequential order (nnsight 0.5 requirement)
            for idx in range(num_layers):
                if idx in chunk_layers:
                    saved[idx] = model.model.layers[idx].output.save()
                else:
                    # Access but don't save -- satisfies ordering requirement
                    # nnsight won't allocate memory for unsaved accesses
                    _ = model.model.layers[idx].output

            # Capture logits only on the last chunk (for verification)
            if chunk_idx == num_chunks - 1:
                saved_logits = model.lm_head.output.save()

        # Move chunk results to CPU and store
        for idx in chunk_layers:
            key = f"residual_layer_{idx}"
            all_results[key] = saved[idx].detach().cpu().to(torch.bfloat16)

        # Explicit cleanup
        del saved
        torch.cuda.empty_cache()

    all_results["logits"] = saved_logits.detach().cpu().to(torch.bfloat16)
    return all_results
```

**Important note on nnsight memory:** In nnsight 0.5, values accessed inside a trace only exist during the trace. They will only persist if `.save()` is called. Unsaved accesses (the `_ = model.model.layers[idx].output` line) do NOT accumulate memory. This means the memory cost per chunk is truly proportional to the number of layers saved, not the total number of layers.

**Confidence:** MEDIUM -- The sequential-access-without-save pattern is inferred from nnsight 0.5 docs stating "values accessed inside a trace only exist during the trace. They will only persist after the context if we call `.save()` on them." The exact behavior of accessing `.output` without `.save()` in the interleaving system needs validation. The Phase 1 code already has the sequential access pattern working, but it always saves at least last-token for unsaved layers. A simpler alternative: just access `.output` without saving for non-chunk layers, which the interleaving system should handle since it just reads the hook output and discards it.

**Fallback if unsaved access fails:** Access all layers sequentially and save only chunk layers. For non-chunk layers, save a dummy 1-element slice: `model.model.layers[idx].output[:, 0:1, 0:1].save()` to satisfy ordering, then discard after the trace.

### Pattern 4: GSM-Symbolic Fresh Instance Loading

**What:** Load both "original" GSM8K questions and "fresh" generated instances from the HuggingFace dataset for contamination diagnostic.

**Key finding:** The `apple/GSM-Symbolic` HuggingFace dataset already contains 50 pre-generated instances per template. Each row has both the generated question AND the `original_question` / `original_answer` fields from GSM8K. There is NO need to run the template generator code (which Apple has not publicly released anyway). We simply:
- Use `original_question` as the potentially-contaminated "original" instance
- Use a generated instance (e.g., instance 0) as the "fresh" instance

**Dataset structure (confirmed via direct inspection):**
- 100 templates (id: 0-99), 50 instances each (instance: 0-49), total 5000 rows
- Fields: `id`, `instance`, `question`, `answer`, `original_id`, `original_question`, `original_answer`, `canary`
- Generated instances have different numerical values but same problem structure
- Answers differ between instances (not just cosmetic changes)

**Example:**
```python
def load_gsm_symbolic_with_fresh(n_per_type=100, seed=137, fresh_instance_id=0):
    """Load GSM-Symbolic with both original and fresh instances.

    Returns list of problem dicts with is_fresh: bool field.
    """
    rng = random.Random(seed)
    ds = load_dataset("apple/GSM-Symbolic", "main", split="test")

    # Group by template id
    templates = {}
    for row in ds:
        templates.setdefault(row["id"], []).append(row)

    items = []
    for template_id in sorted(templates.keys()):
        rows = templates[template_id]

        # Fresh instance: use a generated variant
        fresh_row = [r for r in rows if r["instance"] == fresh_instance_id][0]
        fresh_answer = fresh_row["answer"].split("####")[-1].strip()
        items.append({
            "problem_id": f"gsm_symbolic_{template_id}_fresh",
            "question": fresh_row["question"],
            "original_answer": fresh_answer,
            "dataset_name": "gsm_symbolic",
            "is_fresh": True,
            # ... verification prompt, etc.
        })

        # Original instance: use original_question from GSM8K
        orig_answer = fresh_row["original_answer"].split("####")[-1].strip()
        items.append({
            "problem_id": f"gsm_symbolic_{template_id}_original",
            "question": fresh_row["original_question"],
            "original_answer": orig_answer,
            "dataset_name": "gsm_symbolic",
            "is_fresh": False,
            # ... verification prompt, etc.
        })

    # Balance and sample
    # ... apply verification format (Yes/No) and balance
    return items
```

**Confidence:** HIGH -- Dataset structure verified by direct API inspection. Original questions are confirmed different from generated instances. The `original_question` field provides the exact GSM8K text the model may have been trained on.

### Pattern 5: Per-Trace Deterministic Seed

**What:** Derive a unique seed for each (problem, trace_index) pair using MD5 hash. Stateless, order-independent, resumable.

**Example:**
```python
import hashlib

def derive_trace_seed(problem_id: str, trace_index: int) -> int:
    """Derive deterministic seed from problem_id and trace_index.

    MD5 is fine here -- not security, just deterministic mixing.
    """
    key = f"{problem_id}_{trace_index}"
    hash_bytes = hashlib.md5(key.encode()).hexdigest()
    seed = int(hash_bytes, 16) % (2**32)
    # Avoid seed 42 per project rules
    if seed == 42:
        seed = 43
    return seed
```

**Confidence:** HIGH -- hashlib.md5 is deterministic across platforms and Python versions.

### Pattern 6: Atomic File Writes

**What:** Write output to a temporary file, then atomically rename. Prevents partial/corrupt files from appearing as "done" during resume.

**Example:**
```python
import os
import json
import tempfile

def write_trace_json(data: dict, path: str) -> None:
    """Write trace JSON atomically."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Write to temp file in same directory (same filesystem for atomic rename)
    dir_name = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.rename(tmp_path, path)  # Atomic on POSIX
    except:
        os.unlink(tmp_path)
        raise
```

**Confidence:** HIGH -- `os.rename` is atomic on POSIX when source and destination are on the same filesystem.

### Anti-Patterns to Avoid

- **Using mp.spawn for fault-tolerant generation:** mp.spawn kills all processes on any failure. Use mp.Process for independent worker isolation.
- **Saving activations during generation:** Generation and activation capture are separate scripts per CONTEXT. Do not merge them.
- **Storing full logits tensor to disk:** Full logits for 4096 tokens is ~1.2 GB in BF16. Save only top-k values/indices and entropy. Full logits are used transiently for verification only.
- **Relying on nnsight .generate() for activation capture:** Use model.trace() with the full token sequence for prefill-based capture. The .generate() API is for autoregressive generation interventions, not teacher-forced replay.
- **Accessing nnsight layers out of order:** nnsight 0.5 raises OutOfOrderError. Always iterate layers 0..N-1 in sequence within a trace context.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Think boundary parsing | Custom regex on text | `lib/tokens.py` parse_think_boundaries() | Already handles edge cases (unclosed blocks, multi-block). Token-ID level, not string level. |
| GSM-Symbolic fresh instances | Template parser/generator | HuggingFace `apple/GSM-Symbolic` dataset | Apple has NOT released the template generator code. The HF dataset already has 50 instances per template. Use them directly. |
| BF16 tensor serialization | Custom binary format | safetensors via `lib/storage.py` | Validated round-trip in Phase 1. Safe, fast, industry standard. |
| Config management | Custom argparse | OmegaConf via `lib/config.py` | Already handles base.yaml + script overrides + CLI dotlist. |
| Dataset loading + balancing | Custom CSV parsing | `lib/datasets.py` loaders | Verified, balanced 50/50, handles minority class capping. |
| Repetition detection | Complex n-gram analysis | Simple consecutive token check | For quality gate: check if any token repeats >50 consecutive times. Catches obvious degenerate loops without false positives on legitimate repetition. |

**Key insight:** The template generator from apple/ml-gsm-symbolic has NOT been publicly released. The README explicitly states "We haven't published our parser/data generator yet." However, this is not a problem: the HuggingFace dataset already contains 50 pre-generated fresh instances per template. We use instance 0 as fresh, and the `original_question` field as the contamination baseline.

## Common Pitfalls

### Pitfall 1: mp.spawn Kills All Workers on Single Failure
**What goes wrong:** A single OOM or CUDA error on one GPU terminates ALL parallel generation workers, losing hours of work on other GPUs.
**Why it happens:** `torch.multiprocessing.spawn` is designed for distributed training where all ranks must succeed. It "actively terminate[s] processes upon detecting an error in one of them."
**How to avoid:** Use `multiprocessing.Process` directly with individual `.start()` and `.join()` calls. Check `p.exitcode` after join to identify failed workers.
**Warning signs:** An error message mentioning "process N terminated with exit code 1" followed by all other workers stopping.

### Pitfall 2: nnsight 0.5 OutOfOrderError
**What goes wrong:** Accessing layers in non-sequential order (e.g., layer 20 before layer 0) throws OutOfOrderError.
**Why it happens:** nnsight 0.5 enforces that "the order you interact with modules must match the order those modules are called." This is a breaking change from pre-0.5 behavior.
**How to avoid:** Always iterate `for idx in range(num_layers)` within the trace context, accessing layers 0, 1, 2, ..., 35 in order. For chunked capture, still iterate all layers but only `.save()` the ones in the current chunk.
**Warning signs:** `OutOfOrderError` exception with message about module access order.

### Pitfall 3: Partial File on Resume = False Completion
**What goes wrong:** A crash during file write leaves a partial JSON or safetensors file. On resume, the file-existence check sees it as "done" and skips it, silently losing that trace.
**Why it happens:** Direct `open(path, 'w')` followed by crash leaves incomplete data.
**How to avoid:** Atomic writes: write to `.tmp` file, then `os.rename()`. Rename is atomic on POSIX.
**Warning signs:** Truncated or zero-byte files in the output directory.

### Pitfall 4: Logits Verification Tolerance with BF16
**What goes wrong:** Last-token logits from generation vs. teacher-forced prefill don't match exactly due to BF16 precision and non-deterministic GPU operations.
**Why it happens:** BF16 has limited mantissa precision. CUDA operations may use different internal orderings. Also, the generation pass uses KV cache while the prefill pass computes all positions at once, which can produce slightly different floating-point rounding.
**How to avoid:** Use `torch.allclose()` with appropriate tolerance (rtol=1e-2, atol=1e-2 for BF16) rather than `torch.equal()`. Also compare argmax of logits (predicted token) as a stronger signal -- if the top-1 token matches, the activations are functionally equivalent.
**Warning signs:** Verification assertions failing despite correct generation.

### Pitfall 5: GPU Memory Not Released After Worker Completes
**What goes wrong:** Model and activation tensors remain in GPU memory after a worker finishes its chunk, preventing re-use of that GPU.
**Why it happens:** Python reference counting may not immediately free CUDA tensors. `torch.cuda.empty_cache()` only releases the cache, not tensors still referenced.
**How to avoid:** Explicitly `del model; del tokenizer; gc.collect(); torch.cuda.empty_cache()` at the end of each worker. The Phase 1 validate.py already demonstrates this pattern.
**Warning signs:** CUDA OOM errors when trying to load a second model on the same GPU.

### Pitfall 6: Seed 42 Accidentally Used
**What goes wrong:** Project rule prohibits seed 42. The hash-based seed derivation could theoretically produce seed=42.
**Why it happens:** MD5 hash modulo 2^32 could produce any value.
**How to avoid:** Check in `derive_trace_seed()`: if seed == 42, use 43. The existing `set_seed()` already raises ValueError on 42.
**Warning signs:** `ValueError: Seed 42 is not allowed` during execution.

### Pitfall 7: GSM-Symbolic Answer Parsing Fragility
**What goes wrong:** The `#### <number>` answer format has edge cases: spaces, newlines, commas in numbers, etc.
**Why it happens:** Inconsistent formatting in the original GSM8K dataset.
**How to avoid:** Use the same parsing logic as Phase 1's `load_gsm_symbolic()`: split on `####`, take last part, strip whitespace, try int then float conversion.
**Warning signs:** "cannot parse answer" warning messages during dataset loading.

### Pitfall 8: Disk Space Exhaustion During Activation Capture
**What goes wrong:** 2000 traces x ~600 MB average = ~1.2 TB of activation data fills up scratch disk.
**Why it happens:** All 36 layers of full-sequence BF16 residuals are large.
**How to avoid:** Monitor disk space before and during capture. The CONTEXT specifies separate storage trees for traces and activations. Configure ACTIVATION_DIR to point to a filesystem with sufficient space. Log disk usage after each trace.
**Warning signs:** IOError or OSError during safetensors save.

## Code Examples

### Generation Worker Function
```python
def generation_worker(rank, gpu_id, problems, cfg):
    """Generate traces for assigned problems on a single GPU."""
    logger = setup_logging(f"worker_{rank}", level="INFO")
    logger.info("Worker %d starting on GPU %d with %d problems", rank, gpu_id, len(problems))

    torch.cuda.set_device(gpu_id)
    model, tokenizer = load_model(gpu_id=gpu_id, model_name=cfg.model.name)

    traces_per_problem = cfg.data.traces_per_problem
    max_new_tokens = cfg.model.max_new_tokens
    output_dir = cfg.paths.traces_dir

    completed = 0
    failed = 0
    skipped = 0

    for problem in problems:
        pid = problem["problem_id"]
        prompt = problem["verification_prompt"]

        for trace_idx in range(traces_per_problem):
            trace_path = os.path.join(
                output_dir, "valid", problem["dataset_name"],
                pid, f"trace_{trace_idx:02d}.json"
            )

            if os.path.exists(trace_path):
                skipped += 1
                continue

            seed = derive_trace_seed(pid, trace_idx)
            set_seed(seed)

            try:
                output_ids = generate_trace(
                    model, tokenizer, prompt, max_new_tokens=max_new_tokens
                )
                token_list = output_ids[0].tolist()

                # Quality check: route to valid or failed
                blocks = parse_think_boundaries(token_list)
                has_think = len(blocks) > 0
                has_eos = token_list[-1] == tokenizer.eos_token_id

                # Build trace data
                trace_data = {
                    "token_ids": token_list,
                    "generated_text": tokenizer.decode(token_list, skip_special_tokens=False),
                    "metadata": {
                        "problem_id": pid,
                        "dataset_name": problem["dataset_name"],
                        "ground_truth": problem["ground_truth"],
                        "is_fresh": problem.get("is_fresh", None),
                        "trace_index": trace_idx,
                        "seed": seed,
                        "num_tokens": len(token_list),
                        "num_think_blocks": len(blocks),
                        "hit_max_tokens": not has_eos,
                        "model_name": cfg.model.name,
                        "max_new_tokens": max_new_tokens,
                        "temperature": cfg.generation.temperature,
                        "top_p": cfg.generation.top_p,
                        "top_k": cfg.generation.top_k,
                        "timestamp": time.time(),
                    },
                }

                # Route to valid/ or failed/
                if has_think:
                    dest_dir = "valid"
                else:
                    dest_dir = "failed"

                out_path = os.path.join(
                    output_dir, dest_dir, problem["dataset_name"],
                    pid, f"trace_{trace_idx:02d}.json"
                )
                write_trace_json(trace_data, out_path)
                completed += 1

            except Exception as exc:
                logger.error(
                    "Failed to generate %s trace %d: %s: %s",
                    pid, trace_idx, type(exc).__name__, exc
                )
                failed += 1

    logger.info(
        "Worker %d complete: %d generated, %d failed, %d skipped (resume)",
        rank, completed, failed, skipped
    )

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
```

### Chunked Activation Capture
```python
def capture_all_layers_chunked(model, token_ids, chunk_size=12, num_layers=36):
    """Capture all 36 layers via chunked nnsight trace passes.

    Each pass runs a full forward pass but only saves a subset of layers.
    nnsight 0.5 requires sequential layer access.
    """
    all_residuals = {}
    num_chunks = (num_layers + chunk_size - 1) // chunk_size

    if token_ids.dim() == 1:
        input_ids = token_ids.unsqueeze(0)
    else:
        input_ids = token_ids

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, num_layers)
        chunk_set = set(range(start, end))
        is_last_chunk = (chunk_idx == num_chunks - 1)

        saved = {}
        saved_logits = None

        with model.trace(input_ids) as tracer:
            for idx in range(num_layers):
                if idx in chunk_set:
                    saved[idx] = model.model.layers[idx].output.save()
                else:
                    _ = model.model.layers[idx].output  # satisfy ordering

            if is_last_chunk:
                saved_logits = model.lm_head.output.save()

        for idx in range(start, end):
            tensor = saved[idx].detach().cpu().to(torch.bfloat16)
            all_residuals[f"residual_layer_{idx}"] = tensor

        del saved
        torch.cuda.empty_cache()

    all_residuals["logits"] = saved_logits.detach().cpu().to(torch.bfloat16)
    return all_residuals
```

### Degenerate Repetition Detection (Quality Gate)
```python
def is_degenerate_repetition(token_ids, threshold=50):
    """Check if token sequence contains degenerate repetition loops.

    Returns True if any single token appears more than `threshold`
    consecutive times, indicating a repetition loop.
    """
    if len(token_ids) < threshold:
        return False

    consecutive = 1
    for i in range(1, len(token_ids)):
        if token_ids[i] == token_ids[i - 1]:
            consecutive += 1
            if consecutive >= threshold:
                return True
        else:
            consecutive = 1
    return False
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| nnsight pre-0.5 (proxy-based) | nnsight 0.5 (interleaving, direct execution) | Late 2025 | Must access layers in sequential order; OutOfOrderError enforced |
| capture activations during generation | Decoupled: generate first, capture second | Design decision | Fault isolation, quality gate opportunity |
| capture subset of layers (10) | Capture ALL 36 layers (chunked) | CONTEXT decision | Prevents re-runs; ~6x more disk, same VRAM per chunk |
| mp.spawn for multi-GPU | mp.Process for fault isolation | Phase 2 requirement | Workers survive peer crashes |
| GSM-Symbolic template generator | Use pre-generated HF dataset instances | N/A (generator never released) | 50 instances per template available directly |

**Deprecated/outdated:**
- nnsight proxy-based API (pre-0.5): replaced by interleaving execution model
- `nnsight.apply()`, `nnsight.cond()`, `nnsight.iter()`, `nnsight.log()`: use standard Python equivalents
- Phase 1's `DEFAULT_LAYER_SUBSET = [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]`: Phase 2 captures ALL 36 layers

## Storage and Resource Estimates

### Per-Trace Storage (BF16, safetensors)

| Component | Size (2100 tokens avg) | Size (4196 tokens max) |
|-----------|------------------------|------------------------|
| 36 layers full-seq residuals | 590 MB | 1180 MB |
| Top-k values (k=50) | 0.2 MB | 0.4 MB |
| Top-k indices (k=50) | 0.4 MB | 0.8 MB |
| Entropy | 0.004 MB | 0.008 MB |
| Token IDs | 0.008 MB | 0.016 MB |
| **Total per trace** | **~591 MB** | **~1181 MB** |

### Total Dataset Storage
| Scenario | Traces | Total Size |
|----------|--------|------------|
| Average tokens (2100) | 2000 | ~1.15 TB |
| Max tokens (4196) | 2000 | ~2.3 TB |

### GPU Memory During Capture
| Approach | 2100 tokens | 4196 tokens |
|----------|-------------|-------------|
| All 36 layers single pass | ~17.4 GB | ~19.4 GB |
| Chunked (12 layers/pass, 3 passes) | ~16.5 GB | ~17.4 GB |
| Chunked (6 layers/pass, 6 passes) | ~16.4 GB | ~17.2 GB |

### Generation Time Estimates
| GPUs | Avg tokens/trace | Tokens/sec | Total Time |
|------|------------------|------------|------------|
| 1 | 2000 | 40 | ~28 hours |
| 2 | 2000 | 40 | ~14 hours |
| 4 | 2000 | 40 | ~7 hours |

### Activation Capture Time (After Generation)
| GPUs | Chunks | Approx Total |
|------|--------|-------------|
| 1 | 3 (12 layers each) | ~30 min |
| 4 | 3 (12 layers each) | ~8 min |

## Open Questions

1. **nnsight unsaved layer access behavior**
   - What we know: nnsight 0.5 says "values accessed inside a trace only exist during the trace." Calling `.output` without `.save()` should not accumulate memory.
   - What's unclear: Whether accessing `.output` without `.save()` actually satisfies the sequential ordering requirement, or whether we need to save something (even a dummy slice) to avoid issues.
   - Recommendation: Test with a simple script first. If bare `.output` access works, use it. If not, save a 1-element slice and discard it after the trace.
   - **Confidence:** MEDIUM -- inferred from docs, needs empirical validation.

2. **Disk space for 1-2 TB of activations**
   - What we know: Scratch filesystems at Stanford typically have multi-TB quotas.
   - What's unclear: Whether `/dfs/scratch2/gorn/` has sufficient free space for the full dataset.
   - Recommendation: Check quota with `df -h` before starting capture. Consider using larger chunk sizes (12 or 18) to reduce pass count if single-pass fits in VRAM, since chunked vs. single-pass is about VRAM, not disk.

3. **BF16 logits comparison tolerance**
   - What we know: Teacher-forced prefill should produce identical logits to autoregressive generation for the last token position.
   - What's unclear: Exact numerical tolerance needed due to BF16 precision and CUDA non-determinism.
   - Recommendation: Start with argmax comparison (top-1 token must match). If argmax matches, the forward pass is functionally equivalent. Use `torch.allclose(rtol=0.01, atol=0.01)` as a secondary check.

4. **Generation output logits for verification**
   - What we know: The CONTEXT says to verify correctness by comparing last-token logits between generation and re-computation.
   - What's unclear: During generation, we need to save the last-token logits from the final generation step. The current `generate_trace()` doesn't save logits.
   - Recommendation: After generation, save the model's logits for the last generated token. This can be obtained by running a single forward pass on the full generated sequence and extracting logits at the last position. OR: store the logits from within the generation loop itself by running a separate forward pass after generation completes. Since the generation script produces JSON only (not activations), the verification logits can be stored as a small tensor or as top-k in the JSON metadata.

## Sources

### Primary (HIGH confidence)
- apple/GSM-Symbolic HuggingFace dataset -- directly inspected via Python API, confirmed 100 templates x 50 instances, original_question field available
- Qwen/Qwen3-8B HuggingFace model card -- architecture (36 layers, 4096 hidden, 32K context), sampling parameters (temp=0.6, top_p=0.95, top_k=20), think token ID 151668
- nnsight 0.5 official docs -- OutOfOrderError, interleaving execution, .save() memory semantics, sequential access requirement
- Existing Phase 1 codebase -- model.py, datasets.py, tokens.py, storage.py, multiprocess.py, config.py, utils.py, validate.py

### Secondary (MEDIUM confidence)
- nnsight GitHub releases -- confirmed version 0.5.15, Jan 2026
- PyTorch multiprocessing docs -- mp.spawn behavior (terminates all on failure), mp.Process for independent isolation
- GSM-Symbolic paper (arxiv:2410.05229) -- 100 templates from GSM8K test set, 50 instances per template

### Tertiary (LOW confidence)
- Memory estimates for forward pass intermediates -- rough estimates based on model architecture, not empirically measured
- Token generation throughput (~30-50 tok/s) -- approximate, depends on hardware and sequence length

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries already installed and validated in Phase 1
- Architecture: HIGH -- patterns follow directly from CONTEXT decisions and existing code
- Pitfalls: HIGH -- based on documented library behavior and empirical Phase 1 experience
- Storage estimates: MEDIUM -- calculated from known dimensions but not empirically measured
- nnsight chunked capture details: MEDIUM -- sequential access is documented, but unsaved access behavior inferred

**Research date:** 2026-02-04
**Valid until:** 2026-03-04 (30 days -- stable domain, nnsight unlikely to break compat)
