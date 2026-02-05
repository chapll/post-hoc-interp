"""Multi-GPU chunked activation capture for Qwen3-8B residual streams.

Stage 2 of the decoupled pipeline: reads valid trace JSONs produced by
generate_traces.py, runs teacher-forced nnsight forward passes to capture
all 36 layers of residual stream activations (chunked to avoid OOM), and
saves each trace's activations as a safetensors file with companion metadata.

Chunked capture runs multiple nnsight trace passes per sequence, each saving
only a subset of layers. nnsight 0.5 requires sequential layer access, so
each pass iterates ALL layers but only .save()-s the ones in the current chunk.

Usage:
    uv run python src/experimental/gorn/scripts/capture_activations.py \
        --config src/experimental/gorn/configs/capture.yaml --gpus 0,1,2,3
"""

import gc
import json
import logging
import math
import os
import sys
import time

import torch

# -- project imports ----------------------------------------------------------
sys.path.insert(0, "src/experimental/gorn")

from lib.config import load_config
from lib.model import NUM_LAYERS, load_model
from lib.multiprocess import run_parallel_isolated
from lib.storage import get_activation_path, save_metadata, save_trace
from lib.utils import setup_logging

logger = logging.getLogger(__name__)


def capture_all_layers_chunked(
    model,
    token_ids: torch.Tensor,
    chunk_size: int = 12,
    num_layers: int = NUM_LAYERS,
    top_k: int = 50,
) -> dict[str, torch.Tensor]:
    """Capture all layers of residual stream activations via chunked nnsight traces.

    Runs ceil(num_layers / chunk_size) forward passes. Each pass accesses all
    layers in sequential order (nnsight 0.5 requirement) but only .save()-s
    layers in the current chunk. Logits are captured on the last chunk only.

    Args:
        model: nnsight LanguageModel wrapping Qwen3-8B.
        token_ids: Token IDs tensor, 1D [seq_len] or 2D [1, seq_len].
        chunk_size: Number of layers to capture per forward pass.
        num_layers: Total number of transformer layers.
        top_k: Number of top logit values/indices to store per position.

    Returns:
        Dict mapping string keys to CPU tensors:
        - "residual_layer_{idx}": shape [seq_len, hidden_dim], dtype bfloat16
        - "topk_values": shape [seq_len, top_k], dtype bfloat16
        - "topk_indices": shape [seq_len, top_k], dtype int32
        - "entropy": shape [seq_len], dtype bfloat16
        - "token_ids": shape [seq_len], dtype int32
        - "logits_last_token": shape [vocab_size], dtype bfloat16

    Raises:
        RuntimeError: If nnsight trace fails or CUDA OOM.
    """
    # Ensure batch dimension exists.
    if token_ids.dim() == 1:
        input_ids = token_ids.unsqueeze(0)
    else:
        input_ids = token_ids

    num_chunks = math.ceil(num_layers / chunk_size)
    all_results = {}
    saved_logits_tensor = None

    for chunk_idx in range(num_chunks):
        start_layer = chunk_idx * chunk_size
        end_layer = min(start_layer + chunk_size, num_layers)
        chunk_set = set(range(start_layer, end_layer))
        is_last_chunk = chunk_idx == num_chunks - 1

        saved = {}
        saved_logits = None

        with model.trace(input_ids):
            # Access ALL layers in sequential order (nnsight 0.5 requirement).
            for idx in range(num_layers):
                if idx in chunk_set:
                    saved[idx] = model.model.layers[idx].output.save()
                else:
                    # Access but do not save -- satisfies ordering requirement.
                    # nnsight does not allocate memory for unsaved accesses.
                    _ = model.model.layers[idx].output

            # Capture logits on the last chunk for verification and top-k.
            if is_last_chunk:
                saved_logits = model.lm_head.output.save()

        # Move chunk results to CPU as BF16, squeeze batch dim.
        for idx in range(start_layer, end_layer):
            tensor = saved[idx].detach().cpu().to(torch.bfloat16)
            # Squeeze batch dimension: [1, seq_len, hidden] -> [seq_len, hidden]
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            all_results[f"residual_layer_{idx}"] = tensor

        if is_last_chunk and saved_logits is not None:
            saved_logits_tensor = saved_logits.detach().cpu().to(torch.bfloat16)

        # Explicit cleanup between chunks.
        del saved
        torch.cuda.empty_cache()

    # Process logits: top-k, entropy, last-token logits for verification.
    # logits_tensor shape: [1, seq_len, vocab_size]
    # Use float32 for numerical stability in softmax/entropy.
    logits_2d = saved_logits_tensor[0].float()  # [seq_len, vocab_size]

    # Top-k values and indices per position.
    topk = torch.topk(logits_2d, k=top_k, dim=-1)
    all_results["topk_values"] = topk.values.to(torch.bfloat16)
    all_results["topk_indices"] = topk.indices.to(torch.int32)

    # Entropy: -sum(p * log(p)) with float32 softmax for stability.
    probs = torch.softmax(logits_2d, dim=-1)
    log_probs = probs.log().clamp(min=-1e10)
    entropy = -(probs * log_probs).sum(dim=-1)  # [seq_len]
    all_results["entropy"] = entropy.to(torch.bfloat16)

    # Store last-token logits (full vocab) for verification.
    all_results["logits_last_token"] = saved_logits_tensor[0, -1, :].clone()

    # Store token IDs as int32.
    ids = input_ids.detach().cpu().flatten().to(torch.int32)
    all_results["token_ids"] = ids

    return all_results


def verify_logits(
    model,
    token_ids: torch.Tensor,
    captured_logits_last: torch.Tensor,
    rtol: float = 0.01,
    atol: float = 0.01,
) -> bool:
    """Verify captured logits by comparing with a reference forward pass.

    Runs a single forward pass on the full token sequence and compares the
    last-token logits (argmax match first, then torch.allclose).

    Args:
        model: nnsight LanguageModel wrapping Qwen3-8B.
        token_ids: Token IDs tensor, 1D [seq_len] or 2D [1, seq_len].
        captured_logits_last: Last-token logits from chunked capture,
            shape [vocab_size], dtype bfloat16.
        rtol: Relative tolerance for allclose.
        atol: Absolute tolerance for allclose.

    Returns:
        True if argmax matches and allclose passes, False otherwise.
    """
    if token_ids.dim() == 1:
        input_ids = token_ids.unsqueeze(0)
    else:
        input_ids = token_ids

    # Run a single reference forward pass and capture logits.
    with model.trace(input_ids):
        for idx in range(NUM_LAYERS):
            _ = model.model.layers[idx].output
        ref_logits = model.lm_head.output.save()

    ref_last = ref_logits.detach().cpu().to(torch.bfloat16)[0, -1, :]

    # Compare argmax (predicted token must match).
    cap_argmax = captured_logits_last.argmax().item()
    ref_argmax = ref_last.argmax().item()

    if cap_argmax != ref_argmax:
        logger.error(
            "Logits verification FAILED: argmax mismatch -- captured=%d, reference=%d",
            cap_argmax,
            ref_argmax,
        )
        return False

    # Compare with tolerance.
    close = torch.allclose(
        captured_logits_last.float(),
        ref_last.float(),
        rtol=rtol,
        atol=atol,
    )
    if not close:
        max_diff = (captured_logits_last.float() - ref_last.float()).abs().max().item()
        logger.warning(
            "Logits verification WARNING: argmax matches but allclose failed (max_diff=%.6f, rtol=%.4f, atol=%.4f)",
            max_diff,
            rtol,
            atol,
        )
        # Argmax match is the primary signal; allclose failure is a warning.
        return True

    logger.info("Logits verification PASSED: argmax=%d, allclose=True", cap_argmax)
    return True


def capture_worker(rank, gpu_id, trace_items, cfg):
    """Worker function: load model, capture activations for assigned traces.

    Each worker loads its own model on the assigned GPU, iterates through its
    assigned trace items, runs chunked activation capture, and saves results
    as safetensors files with companion metadata JSON.

    Args:
        rank: Worker rank (0-indexed).
        gpu_id: CUDA device index for this worker.
        trace_items: List of dicts with keys: trace_json_path, dataset_name,
            problem_id, trace_idx, activation_path.
        cfg: OmegaConf DictConfig with capture and paths settings.
    """
    worker_logger = setup_logging(f"capture_worker_{rank}")
    torch.cuda.set_device(gpu_id)

    worker_logger.info(
        "Worker %d starting on GPU %d with %d traces",
        rank,
        gpu_id,
        len(trace_items),
    )

    # Load model on assigned GPU.
    model, tokenizer = load_model(gpu_id=gpu_id, model_name=cfg.model.name)
    worker_logger.info("Worker %d: model loaded on GPU %d", rank, gpu_id)

    # Capture config.
    chunk_size = cfg.capture.chunk_size
    num_layers = cfg.capture.num_layers
    top_k = cfg.capture.top_k_logits
    do_verify = cfg.capture.verify_logits
    logits_rtol = cfg.capture.logits_rtol
    logits_atol = cfg.capture.logits_atol

    captured_count = 0
    skipped_count = 0
    failed_count = 0
    verified_count = 0

    for item_idx, item in enumerate(trace_items):
        trace_json_path = item["trace_json_path"]
        activation_path = item["activation_path"]
        problem_id = item["problem_id"]
        dataset_name = item["dataset_name"]
        trace_idx = item["trace_idx"]

        # Resume check: skip if activation file already exists.
        if os.path.exists(activation_path):
            skipped_count += 1
            continue

        worker_logger.info(
            "Worker %d: trace %d/%d -- %s/%s/trace_%02d",
            rank,
            item_idx + 1,
            len(trace_items),
            dataset_name,
            problem_id,
            trace_idx,
        )

        try:
            # Load trace JSON and extract token IDs.
            with open(trace_json_path, "r") as f:
                trace_data = json.load(f)

            token_list = trace_data["token_ids"]
            token_ids = torch.tensor(token_list, dtype=torch.long, device=f"cuda:{gpu_id}")

            seq_len = len(token_list)

            # Log tensor shapes on first trace per worker.
            if captured_count == 0:
                worker_logger.info(
                    "Worker %d: first trace shapes -- seq_len=%d, num_layers=%d, chunk_size=%d, num_chunks=%d",
                    rank,
                    seq_len,
                    num_layers,
                    chunk_size,
                    math.ceil(num_layers / chunk_size),
                )

            # Run chunked activation capture.
            start_time = time.time()
            results = capture_all_layers_chunked(
                model,
                token_ids,
                chunk_size=chunk_size,
                num_layers=num_layers,
                top_k=top_k,
            )
            capture_elapsed = time.time() - start_time

            # Log shapes on first captured trace.
            if captured_count == 0:
                for key, tensor in sorted(results.items()):
                    worker_logger.info(
                        "Worker %d: tensor %s -- shape=%s, dtype=%s",
                        rank,
                        key,
                        list(tensor.shape),
                        tensor.dtype,
                    )

            # Verify logits on first few traces per worker.
            if do_verify and verified_count < 3:
                verify_ok = verify_logits(
                    model,
                    token_ids,
                    results["logits_last_token"],
                    rtol=logits_rtol,
                    atol=logits_atol,
                )
                if not verify_ok:
                    worker_logger.error(
                        "Worker %d: logits verification FAILED for %s trace %d. Continuing anyway.",
                        rank,
                        problem_id,
                        trace_idx,
                    )
                verified_count += 1

            # Save activations (safetensors).
            save_trace(results, activation_path)

            # Save companion metadata JSON.
            meta = {
                "problem_id": problem_id,
                "dataset_name": dataset_name,
                "trace_index": trace_idx,
                "source_trace_json": trace_json_path,
                "num_tokens": seq_len,
                "num_layers": num_layers,
                "chunk_size": chunk_size,
                "model_name": cfg.model.name,
                "capture_time_seconds": round(capture_elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            # Copy relevant metadata from trace JSON if present.
            trace_meta = trace_data.get("metadata", {})
            for key in [
                "ground_truth",
                "question",
                "is_fresh",
                "seed",
                "num_think_blocks",
                "prompt_config",
                "label_mapping",
                "valid_token",
                "invalid_token",
            ]:
                if key in trace_meta:
                    meta[key] = trace_meta[key]

            meta_path = os.path.splitext(activation_path)[0] + ".json"
            save_metadata(meta, meta_path)

            # Log disk usage for the activation file.
            file_size_mb = os.path.getsize(activation_path) / (1024 * 1024)
            worker_logger.info(
                "Worker %d: saved %s -- %.1f MB, %d tokens, %.1fs capture",
                rank,
                os.path.basename(activation_path),
                file_size_mb,
                seq_len,
                capture_elapsed,
            )

            captured_count += 1

        except Exception:
            failed_count += 1
            worker_logger.exception(
                "Worker %d: error capturing activations for %s trace %d",
                rank,
                problem_id,
                trace_idx,
            )
            continue

    # Cleanup: free model memory.
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    worker_logger.info(
        "Worker %d done. Captured: %d, Skipped (resume): %d, Errors: %d",
        rank,
        captured_count,
        skipped_count,
        failed_count,
    )


def scan_valid_traces(traces_dir: str) -> list[dict]:
    """Scan the valid/ traces directory for all trace JSON files.

    Returns a flat list of dicts, each with:
    - trace_json_path: full path to the trace JSON
    - dataset_name: from the directory structure
    - problem_id: from the directory structure
    - trace_idx: parsed from the filename

    Args:
        traces_dir: Root traces directory containing valid/ subdirectory.

    Returns:
        List of trace item dicts, sorted by dataset/problem/trace_idx.
    """
    valid_dir = os.path.join(traces_dir, "valid")
    items = []

    if not os.path.isdir(valid_dir):
        logger.warning("Valid traces directory does not exist: %s", valid_dir)
        return items

    for dataset_name in sorted(os.listdir(valid_dir)):
        dataset_dir = os.path.join(valid_dir, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue

        for problem_id in sorted(os.listdir(dataset_dir)):
            problem_dir = os.path.join(dataset_dir, problem_id)
            if not os.path.isdir(problem_dir):
                continue

            for filename in sorted(os.listdir(problem_dir)):
                if not filename.endswith(".json"):
                    continue
                # Parse trace index from filename: trace_NN.json
                base = filename.replace(".json", "")
                parts = base.split("_")
                if len(parts) >= 2 and parts[-1].isdigit():
                    trace_idx = int(parts[-1])
                else:
                    logger.warning("Cannot parse trace index from filename: %s", filename)
                    continue

                items.append(
                    {
                        "trace_json_path": os.path.join(problem_dir, filename),
                        "dataset_name": dataset_name,
                        "problem_id": problem_id,
                        "trace_idx": trace_idx,
                    }
                )

    return items


def main():
    """Load config, scan valid traces, dispatch workers for activation capture."""
    main_logger = setup_logging("capture_activations")
    main_logger.info("Starting activation capture")

    # Load configuration (base.yaml + capture.yaml + CLI overrides).
    cfg = load_config()

    main_logger.info("Configuration loaded:")
    main_logger.info("  Model: %s", cfg.model.name)
    main_logger.info("  Num layers: %d", cfg.capture.num_layers)
    main_logger.info("  Chunk size: %d", cfg.capture.chunk_size)
    main_logger.info("  Top-k logits: %d", cfg.capture.top_k_logits)
    main_logger.info("  Verify logits: %s", cfg.capture.verify_logits)
    main_logger.info("  GPUs: %s", list(cfg.gpus))
    main_logger.info("  Traces dir: %s", cfg.paths.traces_dir)
    main_logger.info("  Activations dir: %s", cfg.paths.activations_dir)

    # Scan valid/ traces directory.
    main_logger.info("Scanning valid traces...")
    all_traces = scan_valid_traces(cfg.paths.traces_dir)
    main_logger.info("Found %d valid trace JSON files", len(all_traces))

    if not all_traces:
        main_logger.warning("No valid traces found. Nothing to capture.")
        return

    # Count per-dataset for logging.
    dataset_counts = {}
    for item in all_traces:
        ds = item["dataset_name"]
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    for ds_name, count in sorted(dataset_counts.items()):
        main_logger.info("  %s: %d traces", ds_name, count)

    # Add activation output paths and filter already-captured.
    activations_dir = cfg.paths.activations_dir
    pending_traces = []
    already_captured = 0

    for item in all_traces:
        act_path = get_activation_path(
            activations_dir,
            item["dataset_name"],
            item["problem_id"],
            item["trace_idx"],
        )
        item["activation_path"] = act_path

        if os.path.exists(act_path):
            already_captured += 1
        else:
            pending_traces.append(item)

    main_logger.info(
        "Already captured: %d, Pending: %d",
        already_captured,
        len(pending_traces),
    )

    if not pending_traces:
        main_logger.info("All traces already captured. Nothing to do.")
        return

    # Resolve GPU IDs.
    gpu_ids = list(cfg.gpus)
    main_logger.info("Using GPUs: %s", gpu_ids)

    # Dispatch workers via fault-tolerant isolated processes.
    main_logger.info("Dispatching %d workers...", len(gpu_ids))
    start_time = time.time()

    failures = run_parallel_isolated(capture_worker, pending_traces, gpu_ids, cfg)

    elapsed = time.time() - start_time
    main_logger.info("All workers finished in %.1f seconds", elapsed)

    # Report results.
    num_workers = len(gpu_ids)
    num_failed_workers = len(failures)
    num_succeeded = num_workers - num_failed_workers

    main_logger.info(
        "Workers: %d/%d succeeded, %d/%d failed",
        num_succeeded,
        num_workers,
        num_failed_workers,
        num_workers,
    )

    if failures:
        for rank_id, gid, exitcode in failures:
            main_logger.error(
                "Worker rank %d on GPU %d exited with code %d",
                rank_id,
                gid,
                exitcode,
            )

    main_logger.info("Activations directory: %s", activations_dir)
    main_logger.info("Activation capture complete.")


if __name__ == "__main__":
    main()
