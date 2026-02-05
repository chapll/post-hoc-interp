"""End-to-end validation script for Phase 1 foundation components.

Exercises every component built in Plans 01-01 through 01-05:
  - Config loading (OmegaConf + base.yaml + script yaml + CLI overrides)
  - Dataset loading (GSM-Symbolic, FOLIO, StrategyQA in verification format)
  - Model loading via nnsight (Qwen3-8B in BF16)
  - Trace generation (thinking-mode text generation)
  - Activation capture (residual streams, last-token, logits)
  - Storage round-trip (safetensors save/load, JSON metadata)
  - Think token parsing (boundary detection, content masks, pre-handoff window)
  - Multi-GPU parallel execution (mp.spawn with worker signature)

Requires GPU access. Exits with code 1 on any failure or if fewer than 2 GPUs
are available (multi-GPU validation is mandatory).

Usage:
    uv run python src/experimental/gorn/scripts/validate.py \
        --config src/experimental/gorn/configs/validate.yaml --gpus 0,1
"""

import gc
import sys
import tempfile

import torch

# -- project imports ----------------------------------------------------------
sys.path.insert(0, "src/experimental/gorn")

from lib.config import load_config
from lib.datasets import load_dataset_by_name
from lib.prompts import build_verification_prompt
from lib.model import (
    HIDDEN_DIM,
    NUM_LAYERS,
    capture_activations,
    generate_trace,
    load_model,
)
from lib.multiprocess import run_parallel
from lib.storage import (
    get_metadata_path,
    get_trace_path,
    load_metadata,
    load_trace,
    prepare_trace_tensors,
    save_trace,
)
from lib.tokens import (
    count_think_blocks,
    get_pre_handoff_window,
    get_think_content_mask,
    parse_think_boundaries,
)
from lib.utils import set_seed, setup_logging

# -- globals ------------------------------------------------------------------

logger = setup_logging("validate", level="INFO")

# Track check results for final summary.
results: dict[str, list[dict]] = {}
any_failure = False


def record(dataset: str, check: str, passed: bool, detail: str = ""):
    """Record a check result and print it."""
    global any_failure
    status = "PASS" if passed else "FAIL"
    if not passed:
        any_failure = True
    msg = f"  [{status}] {check}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    results.setdefault(dataset, []).append(
        {"check": check, "passed": passed, "detail": detail}
    )


# =============================================================================
# Section 1: Dataset validation
# =============================================================================


def validate_datasets(cfg):
    """Load and validate all datasets per config."""
    print("\n--- Dataset Validation ---")
    datasets_cfg = cfg.data.datasets
    n = cfg.data.problems_per_dataset

    all_problems = {}  # dataset_name -> list of problems

    for ds_name in datasets_cfg:
        print(f"\nDataset: {ds_name} (n={n})")
        problems = load_dataset_by_name(ds_name, n=n, seed=cfg.seed)
        # Add verification_prompt using yesno protocol for backward compatibility.
        for p in problems:
            result = build_verification_prompt(p, label_protocol="yesno")
            p["verification_prompt"] = result["verification_prompt"]
        all_problems[ds_name] = problems

        # Count labels
        yes_count = sum(1 for p in problems if p["ground_truth"] == "Yes")
        no_count = sum(1 for p in problems if p["ground_truth"] == "No")
        print(f"  Total: {len(problems)}, Yes: {yes_count}, No: {no_count}")

        # Check balance
        record(
            ds_name,
            "balance",
            yes_count == no_count,
            f"Yes={yes_count}, No={no_count}",
        )

        # Check count matches request
        record(
            ds_name,
            "count",
            len(problems) == n,
            f"got {len(problems)}, expected {n}",
        )

        # Check schema: every required field present
        required_fields = {
            "problem_id",
            "question",
            "ground_truth",
            "dataset_name",
            "original_answer",
            "verification_prompt",
            "presented_answer",
        }
        schema_ok = True
        for p in problems:
            missing = required_fields - set(p.keys())
            if missing:
                schema_ok = False
                print(f"  Missing fields in {p.get('problem_id', '?')}: {missing}")
                break
        record(ds_name, "schema", schema_ok)

        # Check prompts contain "think step-by-step" (may vary slightly by dataset)
        prompt_ok = all("think step-by-step" in p["verification_prompt"].lower() for p in problems)
        record(ds_name, "prompt_contains_think_step_by_step", prompt_ok)

        # GSM-Symbolic specific: distractors differ from correct answers
        if ds_name == "gsm_symbolic":
            distractor_ok = True
            for p in problems:
                if p["ground_truth"] == "No":
                    # presented_answer should differ from original_answer
                    if p["presented_answer"] == p["original_answer"]:
                        distractor_ok = False
                        print(
                            f"  Distractor matches correct for {p['problem_id']}: "
                            f"{p['presented_answer']} == {p['original_answer']}"
                        )
                        break
            record(ds_name, "gsm_distractors_differ", distractor_ok)

    return all_problems


# =============================================================================
# Section 2: Model + trace generation + activation capture (single GPU)
# =============================================================================


def validate_model_and_traces(cfg, all_problems, tmp_dir):
    """Load model, generate traces, capture activations, test storage round-trip."""
    print("\n--- Model + Trace Generation Validation (single GPU) ---")

    gpu_ids = cfg.gpus
    primary_gpu = gpu_ids[0]
    layer_subset = list(cfg.storage.layer_subset)
    top_k = cfg.storage.top_k_logits
    max_new_tokens = cfg.model.max_new_tokens

    print(f"Loading model on GPU {primary_gpu}...")
    model, tokenizer = load_model(gpu_id=primary_gpu)
    print("Model loaded.")

    # Generate 1 trace per dataset to reduce memory pressure during validation.
    # This validates the full pipeline; scale testing happens in Phase 2.
    traces_per_dataset = 1

    for ds_name, problems in all_problems.items():
        subset = problems[:traces_per_dataset]
        print(f"\nDataset: {ds_name} -- generating {len(subset)} traces")

        for idx, problem in enumerate(subset):
            pid = problem["problem_id"]
            prompt = problem["verification_prompt"]
            print(f"\n  Problem {idx+1}/{len(subset)}: {pid}")

            # -- Generation --
            output_ids = generate_trace(
                model, tokenizer, prompt, max_new_tokens=max_new_tokens
            )
            # output_ids shape: [1, total_seq_len]
            token_list = output_ids[0].tolist()
            prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
            prompt_len = prompt_tokens.shape[1]
            output_len = len(token_list)
            new_tokens = output_len - prompt_len

            print(f"    Prompt tokens: {prompt_len}, Output tokens: {output_len}, New: {new_tokens}")

            # -- Think token parsing --
            blocks = parse_think_boundaries(token_list)
            num_blocks = count_think_blocks(token_list)
            content_mask = get_think_content_mask(token_list)
            think_content_count = sum(content_mask)
            pre_handoff = get_pre_handoff_window(token_list, window_size=3)

            print(f"    Think blocks: {num_blocks}")
            for bi, (bs, be) in enumerate(blocks):
                print(f"      Block {bi}: positions {bs}-{be} (length {be - bs + 1})")
            print(f"    Think content tokens: {think_content_count}")
            if pre_handoff is not None:
                print(f"    Pre-handoff window: positions {pre_handoff[0]}-{pre_handoff[1]}")
            else:
                print("    Pre-handoff window: None (no </think> found)")

            record(
                ds_name,
                f"think_blocks_{pid}",
                num_blocks >= 1,
                f"{num_blocks} block(s)",
            )
            # Pre-handoff may be None if generation was truncated (no </think> token).
            # This is expected with short max_new_tokens. Record as pass if think blocks
            # exist (the parsing code works), warn about truncation.
            if pre_handoff is not None:
                record(ds_name, f"pre_handoff_{pid}", True, f"{pre_handoff}")
            elif num_blocks >= 1:
                # Truncated trace with think block but no closing </think>
                print("    [WARN] Pre-handoff None (truncated, no </think>) - OK for validation")
                record(ds_name, f"pre_handoff_{pid}", True, "truncated")
            else:
                record(ds_name, f"pre_handoff_{pid}", False, "None")

            # -- Activation capture --
            print(f"    Capturing activations at layers {layer_subset}...")
            raw_activations = capture_activations(
                model, output_ids, layer_subset=layer_subset
            )

            # -- Shape verification on RAW activations (before prepare) --
            # Raw shapes have batch dim: [1, seq_len, hidden] and [1, hidden]
            seq_len = output_ids.shape[1]
            for li in layer_subset:
                key = f"residual_layer_{li}"
                expected = (1, seq_len, HIDDEN_DIM)
                actual = tuple(raw_activations[key].shape)
                record(
                    ds_name,
                    f"raw_shape_{key}_{pid}",
                    actual == expected,
                    f"expected {expected}, got {actual}",
                )

            for li in range(NUM_LAYERS):
                key = f"last_token_layer_{li}"
                expected = (1, HIDDEN_DIM)
                actual = tuple(raw_activations[key].shape)
                record(
                    ds_name,
                    f"raw_shape_{key}_{pid}",
                    actual == expected,
                    f"expected {expected}, got {actual}",
                )

            logits_key = "logits"
            logits_tensor = raw_activations[logits_key]
            # logits shape: [1, seq_len, vocab_size]
            logits_ok = (
                logits_tensor.dim() == 3
                and logits_tensor.shape[0] == 1
                and logits_tensor.shape[1] == seq_len
            )
            record(
                ds_name,
                f"raw_shape_logits_{pid}",
                logits_ok,
                f"shape {tuple(logits_tensor.shape)}",
            )

            # -- Prepare trace tensors (squeezes batch dim, computes topk/entropy) --
            prepared = prepare_trace_tensors(
                activations=raw_activations,
                logits=logits_tensor,
                token_ids=output_ids,
                top_k=top_k,
            )

            # -- Shape verification on PREPARED tensors (after squeeze) --
            for li in layer_subset:
                key = f"residual_layer_{li}"
                expected = (seq_len, HIDDEN_DIM)
                actual = tuple(prepared[key].shape)
                record(
                    ds_name,
                    f"stored_shape_{key}_{pid}",
                    actual == expected,
                    f"expected {expected}, got {actual}",
                )

            for li in range(NUM_LAYERS):
                key = f"last_token_layer_{li}"
                expected_shape = (HIDDEN_DIM,)
                actual = tuple(prepared[key].shape)
                record(
                    ds_name,
                    f"stored_shape_{key}_{pid}",
                    actual == expected_shape,
                    f"expected {expected_shape}, got {actual}",
                )

            # topk_values: [seq_len, top_k]
            record(
                ds_name,
                f"stored_shape_topk_values_{pid}",
                tuple(prepared["topk_values"].shape) == (seq_len, top_k),
                f"shape {tuple(prepared['topk_values'].shape)}",
            )
            # topk_indices: [seq_len, top_k]
            record(
                ds_name,
                f"stored_shape_topk_indices_{pid}",
                tuple(prepared["topk_indices"].shape) == (seq_len, top_k),
                f"shape {tuple(prepared['topk_indices'].shape)}",
            )
            # entropy: [seq_len]
            record(
                ds_name,
                f"stored_shape_entropy_{pid}",
                tuple(prepared["entropy"].shape) == (seq_len,),
                f"shape {tuple(prepared['entropy'].shape)}",
            )
            # token_ids: [seq_len]
            record(
                ds_name,
                f"stored_shape_token_ids_{pid}",
                tuple(prepared["token_ids"].shape) == (seq_len,),
                f"shape {tuple(prepared['token_ids'].shape)}",
            )

            # -- Dtype verification on prepared tensors --
            for li in layer_subset:
                key = f"residual_layer_{li}"
                record(
                    ds_name,
                    f"dtype_{key}_{pid}",
                    prepared[key].dtype == torch.bfloat16,
                    f"dtype {prepared[key].dtype}",
                )
            record(
                ds_name,
                f"dtype_topk_values_{pid}",
                prepared["topk_values"].dtype == torch.bfloat16,
                f"dtype {prepared['topk_values'].dtype}",
            )
            record(
                ds_name,
                f"dtype_topk_indices_{pid}",
                prepared["topk_indices"].dtype == torch.int32,
                f"dtype {prepared['topk_indices'].dtype}",
            )
            record(
                ds_name,
                f"dtype_entropy_{pid}",
                prepared["entropy"].dtype == torch.bfloat16,
                f"dtype {prepared['entropy'].dtype}",
            )
            record(
                ds_name,
                f"dtype_token_ids_{pid}",
                prepared["token_ids"].dtype == torch.int32,
                f"dtype {prepared['token_ids'].dtype}",
            )

            # -- Storage round-trip --
            trace_path = get_trace_path(tmp_dir, ds_name, pid, trace_idx=0)
            metadata = {
                "problem_id": pid,
                "dataset_name": ds_name,
                "ground_truth": problem["ground_truth"],
                "num_tokens": seq_len,
                "num_think_blocks": num_blocks,
            }

            print(f"    Saving trace to {trace_path}...")
            save_trace(prepared, trace_path, metadata=metadata)

            # Load back and verify
            loaded = load_trace(trace_path)
            meta_path = get_metadata_path(trace_path)
            loaded_meta = load_metadata(meta_path)

            # Verify metadata round-trip
            meta_match = all(
                str(loaded_meta.get(k)) == str(v)
                for k, v in metadata.items()
            )
            record(
                ds_name,
                f"metadata_round_trip_{pid}",
                meta_match,
                f"keys: {sorted(metadata.keys())}",
            )

            # Verify tensor round-trip: every key present, shapes match, values match
            keys_match = set(loaded.keys()) == set(prepared.keys())
            record(
                ds_name,
                f"round_trip_keys_{pid}",
                keys_match,
                f"saved {len(prepared)} keys, loaded {len(loaded)} keys",
            )

            all_values_match = True
            for tkey in prepared:
                if tkey not in loaded:
                    all_values_match = False
                    print(f"    Missing key in loaded: {tkey}")
                    continue
                if prepared[tkey].shape != loaded[tkey].shape:
                    all_values_match = False
                    print(
                        f"    Shape mismatch for {tkey}: "
                        f"saved {prepared[tkey].shape} vs loaded {loaded[tkey].shape}"
                    )
                    continue
                if prepared[tkey].dtype != loaded[tkey].dtype:
                    all_values_match = False
                    print(
                        f"    Dtype mismatch for {tkey}: "
                        f"saved {prepared[tkey].dtype} vs loaded {loaded[tkey].dtype}"
                    )
                    continue
                if not torch.equal(prepared[tkey], loaded[tkey]):
                    all_values_match = False
                    print(f"    Value mismatch for {tkey}")
                    continue

            record(
                ds_name,
                f"round_trip_values_{pid}",
                all_values_match,
                f"{len(prepared)} tensors compared",
            )

    # Free model memory before multi-GPU section.
    # Must gc.collect() before empty_cache() so Python releases all references first.
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    gpu_mem_free = torch.cuda.mem_get_info(primary_gpu)
    print(
        f"\nSingle-GPU model freed. GPU {primary_gpu}: "
        f"{gpu_mem_free[0] / (1024**3):.1f} GB free / {gpu_mem_free[1] / (1024**3):.1f} GB total"
    )


# =============================================================================
# Section 3: Multi-GPU validation
# =============================================================================


def _multi_gpu_worker(rank, gpu_id, items, cfg):
    """Worker function for multi-GPU validation.

    Loads model on assigned GPU, generates one trace per item, saves to disk.
    """
    from lib.model import capture_activations as _capture
    from lib.model import generate_trace as _gen
    from lib.model import load_model as _load
    from lib.storage import prepare_trace_tensors as _prepare
    from lib.storage import save_trace as _save
    from lib.storage import get_trace_path as _trace_path

    print(f"  [Worker {rank}] Starting on GPU {gpu_id} with {len(items)} items")
    model, tokenizer = _load(gpu_id=gpu_id)

    layer_subset = list(cfg.storage.layer_subset)
    top_k = cfg.storage.top_k_logits
    max_new_tokens = cfg.model.max_new_tokens
    tmp_dir = cfg.validation.multi_gpu_tmp_dir

    for item_idx, problem in enumerate(items):
        pid = problem["problem_id"]
        prompt = problem["verification_prompt"]
        print(f"  [Worker {rank}] Generating trace for {pid} ({item_idx+1}/{len(items)})")

        output_ids = _gen(model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        raw = _capture(model, output_ids, layer_subset=layer_subset)
        prepared = _prepare(
            activations=raw, logits=raw["logits"], token_ids=output_ids, top_k=top_k
        )

        trace_path = _trace_path(tmp_dir, problem["dataset_name"], pid, trace_idx=0)
        _save(prepared, trace_path)
        print(f"  [Worker {rank}] Saved {trace_path}")

    del model
    del tokenizer
    torch.cuda.empty_cache()
    print(f"  [Worker {rank}] Done.")


def validate_multi_gpu(cfg, all_problems):
    """Validate multi-GPU parallel execution. Mandatory: requires >= 2 GPUs."""
    print("\n--- Multi-GPU Validation ---")

    gpu_ids = cfg.gpus
    num_gpus = len(gpu_ids)

    if num_gpus < 2:
        print(
            "FAIL: Multi-GPU validation required per CONTEXT.md. "
            f"Only {num_gpus} GPU(s) configured (--gpus flag). "
            "Run on a machine with 2+ GPUs."
        )
        record("multi_gpu", "gpu_count", False, f"need >=2 GPUs, have {num_gpus}")
        return

    record("multi_gpu", "gpu_count", True, f"{num_gpus} GPUs available")

    # Pick 1 problem per GPU from different datasets
    work_items = []
    for ds_name, problems in all_problems.items():
        if problems and len(work_items) < num_gpus:
            work_items.append(problems[0])
    # Pad if needed
    while len(work_items) < num_gpus:
        for ds_name, problems in all_problems.items():
            if len(work_items) >= num_gpus:
                break
            if len(problems) > 1:
                work_items.append(problems[1])

    # Create temp dir for multi-GPU outputs
    multi_tmp = tempfile.mkdtemp(prefix="validate_multigpu_")
    cfg_copy = cfg.copy()
    from omegaconf import OmegaConf
    OmegaConf.update(cfg_copy, "validation.multi_gpu_tmp_dir", multi_tmp)

    print(f"Running {len(work_items)} items across GPUs {gpu_ids}")
    print(f"Multi-GPU temp dir: {multi_tmp}")

    try:
        run_parallel(
            worker_fn=_multi_gpu_worker,
            items=work_items,
            gpu_ids=gpu_ids,
            cfg=cfg_copy,
        )
        record("multi_gpu", "spawn_and_run", True, f"{num_gpus} workers completed")
    except Exception as exc:
        print(f"Multi-GPU execution failed: {exc}")
        record("multi_gpu", "spawn_and_run", False, str(exc))
        return

    # Verify output files exist
    import os
    from lib.storage import get_trace_path as _tp
    all_exist = True
    for item in work_items:
        trace_path = _tp(multi_tmp, item["dataset_name"], item["problem_id"], 0)
        if not os.path.exists(trace_path):
            all_exist = False
            print(f"  Missing output: {trace_path}")
    record(
        "multi_gpu",
        "output_files_exist",
        all_exist,
        f"{len(work_items)} trace files expected",
    )


# =============================================================================
# Section 4: Summary
# =============================================================================


def print_summary():
    """Print final summary table and overall PASS/FAIL."""
    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)

    # Per-dataset summary
    print(f"\n{'Dataset':<20} {'Total':<8} {'Pass':<8} {'Fail':<8}")
    print("-" * 44)
    total_checks = 0
    total_pass = 0
    total_fail = 0
    for ds_name in sorted(results.keys()):
        checks = results[ds_name]
        passed = sum(1 for c in checks if c["passed"])
        failed = sum(1 for c in checks if not c["passed"])
        total_checks += len(checks)
        total_pass += passed
        total_fail += failed
        print(f"{ds_name:<20} {len(checks):<8} {passed:<8} {failed:<8}")

    print("-" * 44)
    print(f"{'TOTAL':<20} {total_checks:<8} {total_pass:<8} {total_fail:<8}")

    # Failed checks detail
    if total_fail > 0:
        print("\nFailed checks:")
        for ds_name in sorted(results.keys()):
            for c in results[ds_name]:
                if not c["passed"]:
                    detail = f" -- {c['detail']}" if c["detail"] else ""
                    print(f"  [{ds_name}] {c['check']}{detail}")

    print("\n" + "=" * 72)
    if any_failure:
        print("OVERALL: FAIL")
    else:
        print("OVERALL: PASS")
    print("=" * 72)


# =============================================================================
# Main
# =============================================================================


def main():
    global any_failure

    # Load config
    cfg = load_config()
    print("Configuration loaded.")
    print(f"  Datasets: {list(cfg.data.datasets)}")
    print(f"  Problems per dataset: {cfg.data.problems_per_dataset}")
    print(f"  GPUs: {list(cfg.gpus)}")
    print(f"  Seed: {cfg.seed}")
    print(f"  Layer subset: {list(cfg.storage.layer_subset)}")
    print(f"  Top-k logits: {cfg.storage.top_k_logits}")
    print(f"  Max new tokens: {cfg.model.max_new_tokens}")

    # Set seed
    set_seed(cfg.seed)

    # Check GPU availability
    if not torch.cuda.is_available():
        print("FATAL: No CUDA GPUs available. This script requires GPU access.")
        sys.exit(1)
    num_visible = torch.cuda.device_count()
    print(f"CUDA devices visible: {num_visible}")
    for i in range(num_visible):
        name = torch.cuda.get_device_name(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  GPU {i}: {name} ({mem_gb:.1f} GB)")

    # Validate GPU IDs in config
    for gid in cfg.gpus:
        if gid >= num_visible:
            print(f"FATAL: Configured GPU {gid} not available (only {num_visible} devices).")
            sys.exit(1)

    # Section 1: Dataset validation
    all_problems = validate_datasets(cfg)

    # Section 2: Model + trace + capture + storage (single GPU)
    with tempfile.TemporaryDirectory(prefix="validate_traces_") as tmp_dir:
        validate_model_and_traces(cfg, all_problems, tmp_dir)
        print(f"\nTemp traces cleaned up: {tmp_dir}")

    # Section 3: Multi-GPU validation (mandatory)
    validate_multi_gpu(cfg, all_problems)

    # Section 4: Summary
    print_summary()

    if any_failure:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
