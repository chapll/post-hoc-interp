"""Modal-based activation capture for Qwen3-8B residual streams.

Reads valid trace JSONs from the traces volume, runs teacher-forced nnsight
forward passes to capture all 36 layers of residual stream activations
(chunked to avoid OOM), and saves each trace's activations as safetensors
files with companion JSON metadata.

Configuration is loaded from configs/modal/capture.yaml with CLI overrides.

Usage:
    # Run activation capture with default config
    modal run scripts/modal_capture_activations.py

    # Run with custom config file
    modal run scripts/modal_capture_activations.py --config configs/modal/my_capture.yaml

    # Override specific settings via CLI
    modal run scripts/modal_capture_activations.py --n-containers 8 --chunk-size 6

    # Dry run: scan and report how many traces need capture
    modal run scripts/modal_capture_activations.py --dry-run

    # Test: run a minimal single-trace capture to verify pipeline
    modal run scripts/modal_capture_activations.py --test

    # Download results
    modal volume get activations-output / data/activations/ --force
"""

import json
import math
import os
import time
from pathlib import Path

import modal


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------


def _find_project_root() -> Path:
    """Walk up from this file to find the directory containing pyproject.toml."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")


def load_modal_config(config_path: str | None = None) -> dict:
    """Load Modal configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses configs/modal/capture.yaml.

    Returns:
        Configuration dict with all settings.
    """
    from omegaconf import OmegaConf

    project_root = _find_project_root()
    default_config = project_root / "configs" / "modal" / "capture.yaml"

    if config_path is not None:
        config_file = Path(config_path)
        if not config_file.is_absolute():
            config_file = project_root / config_file
    else:
        config_file = default_config

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    cfg = OmegaConf.load(config_file)
    return OmegaConf.to_container(cfg, resolve=True)


# ---------------------------------------------------------------------------
# Modal App and Resources
# ---------------------------------------------------------------------------

app = modal.App("post-hoc-interp-activations")

# Volume for caching model weights (persistent, read-only here)
model_cache_vol = modal.Volume.from_name("qwen3-8b-cache", create_if_missing=True)

# Volume for reading trace JSONs (persistent, read-only here)
traces_vol = modal.Volume.from_name("traces-output", create_if_missing=True)

# Volume for storing activation outputs (persistent, write-heavy)
activations_vol = modal.Volume.from_name("activations-output", create_if_missing=True)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        # Core ML
        "torch==2.9.0",
        "transformers>=4.51.0",
        "nnsight>=0.5.15",
        # Data handling
        "safetensors>=0.7.0",
        # Utilities
        "omegaconf>=2.3.0",
        "numpy>=2.4.1",
        "einops>=0.8.1",
    )
    .add_local_python_source("src")
)


# ---------------------------------------------------------------------------
# Activation Capture Helpers (inlined from scripts/capture_activations.py)
# ---------------------------------------------------------------------------


def capture_all_layers_chunked(
    model,
    token_ids,
    chunk_size: int = 12,
    num_layers: int = 36,
    top_k: int = 50,
) -> dict:
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
    """
    import torch

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
                    _ = model.model.layers[idx].output

            # Capture logits on the last chunk for verification and top-k.
            if is_last_chunk:
                saved_logits = model.lm_head.output.save()

        # Move chunk results to CPU as BF16, squeeze batch dim.
        for idx in range(start_layer, end_layer):
            tensor = saved[idx].detach().cpu().to(torch.bfloat16)
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            all_results[f"residual_layer_{idx}"] = tensor

        if is_last_chunk and saved_logits is not None:
            saved_logits_tensor = saved_logits.detach().cpu().to(torch.bfloat16)

        # Explicit cleanup between chunks.
        del saved
        torch.cuda.empty_cache()

    # Process logits: top-k, entropy, last-token logits for verification.
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
    token_ids,
    captured_logits_last,
    num_layers: int = 36,
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
        num_layers: Total number of transformer layers.
        rtol: Relative tolerance for allclose.
        atol: Absolute tolerance for allclose.

    Returns:
        True if argmax matches, False otherwise.
    """
    import torch

    if token_ids.dim() == 1:
        input_ids = token_ids.unsqueeze(0)
    else:
        input_ids = token_ids

    # Run a single reference forward pass and capture logits.
    with model.trace(input_ids):
        for idx in range(num_layers):
            _ = model.model.layers[idx].output
        ref_logits = model.lm_head.output.save()

    ref_last = ref_logits.detach().cpu().to(torch.bfloat16)[0, -1, :]

    # Compare argmax (predicted token must match).
    cap_argmax = captured_logits_last.argmax().item()
    ref_argmax = ref_last.argmax().item()

    if cap_argmax != ref_argmax:
        print(
            f"Logits verification FAILED: argmax mismatch -- captured={cap_argmax}, reference={ref_argmax}"
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
        print(
            f"Logits verification WARNING: argmax matches but allclose failed "
            f"(max_diff={max_diff:.6f}, rtol={rtol:.4f}, atol={atol:.4f})"
        )
        # Argmax match is the primary signal; allclose failure is a warning.
        return True

    print(f"Logits verification PASSED: argmax={cap_argmax}, allclose=True")
    return True


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
        print(f"WARNING: Valid traces directory does not exist: {valid_dir}")
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
                    print(f"WARNING: Cannot parse trace index from filename: {filename}")
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


# ---------------------------------------------------------------------------
# Activation Capture Worker (runs on A100)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="a100-80gb",
    volumes={
        "/model-cache": model_cache_vol,
        "/traces": traces_vol,
        "/activations": activations_vol,
    },
    timeout=7200,  # 2 hours per batch
    retries=1,
)
def capture_activations_batch(
    trace_items: list[dict],
    config: dict,
    batch_id: int,
) -> dict:
    """Capture activations for a batch of traces on a single A100.

    Each invocation loads the model, iterates through assigned trace items,
    runs chunked activation capture, and saves safetensors + metadata.

    Args:
        trace_items: List of dicts with trace_json_path, dataset_name,
            problem_id, trace_idx.
        config: Dict with model_name, capture settings.
        batch_id: Identifier for this batch (for logging).

    Returns:
        Dict with counts: batch_id, captured, skipped, failed.
    """
    import torch
    from nnsight import LanguageModel

    from src.storage import save_metadata, save_trace

    # Set up HuggingFace cache
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    print(f"Batch {batch_id}: Starting with {len(trace_items)} traces")

    # Load model from cache
    print(f"Batch {batch_id}: Loading model...")
    model = LanguageModel(
        config["model_name"],
        device_map="cuda:0",
        dtype=torch.bfloat16,
        dispatch=True,
    )
    print(f"Batch {batch_id}: Model loaded")

    # Capture config
    chunk_size = config["chunk_size"]
    num_layers = config["num_layers"]
    top_k = config["top_k_logits"]
    do_verify = config["verify_logits"]
    logits_rtol = config["logits_rtol"]
    logits_atol = config["logits_atol"]

    results = {"captured": 0, "skipped": 0, "failed": 0}
    verified_count = 0
    commit_counter = 0

    for item_idx, item in enumerate(trace_items):
        trace_json_path = item["trace_json_path"]
        dataset_name = item["dataset_name"]
        problem_id = item["problem_id"]
        trace_idx = item["trace_idx"]

        # Compute activation output path
        activation_path = os.path.join(
            "/activations",
            dataset_name,
            problem_id,
            f"trace_{trace_idx:02d}.safetensors",
        )

        # Resume check: skip if safetensors already exists
        if os.path.exists(activation_path):
            results["skipped"] += 1
            continue

        if item_idx % 10 == 0:
            print(f"Batch {batch_id}: Processing trace {item_idx + 1}/{len(trace_items)}")

        try:
            # Load trace JSON and extract token IDs
            with open(trace_json_path, "r") as f:
                trace_data = json.load(f)

            token_list = trace_data["token_ids"]
            token_ids = torch.tensor(token_list, dtype=torch.long, device="cuda:0")
            seq_len = len(token_list)

            # Log shapes on first trace
            if results["captured"] == 0:
                print(
                    f"Batch {batch_id}: First trace -- seq_len={seq_len}, "
                    f"num_layers={num_layers}, chunk_size={chunk_size}, "
                    f"num_chunks={math.ceil(num_layers / chunk_size)}"
                )

            # Run chunked activation capture
            start_time = time.time()
            activation_tensors = capture_all_layers_chunked(
                model,
                token_ids,
                chunk_size=chunk_size,
                num_layers=num_layers,
                top_k=top_k,
            )
            capture_elapsed = time.time() - start_time

            # Log tensor shapes on first captured trace
            if results["captured"] == 0:
                for key, tensor in sorted(activation_tensors.items()):
                    print(
                        f"Batch {batch_id}: tensor {key} -- "
                        f"shape={list(tensor.shape)}, dtype={tensor.dtype}"
                    )

            # Verify logits on first few traces
            if do_verify and verified_count < 3:
                verify_ok = verify_logits(
                    model,
                    token_ids,
                    activation_tensors["logits_last_token"],
                    num_layers=num_layers,
                    rtol=logits_rtol,
                    atol=logits_atol,
                )
                if not verify_ok:
                    print(
                        f"Batch {batch_id}: Logits verification FAILED for "
                        f"{problem_id} trace {trace_idx}. Continuing anyway."
                    )
                verified_count += 1

            # Save activations (safetensors)
            save_trace(activation_tensors, activation_path)

            # Save companion metadata JSON
            meta = {
                "problem_id": problem_id,
                "dataset_name": dataset_name,
                "trace_index": trace_idx,
                "source_trace_json": trace_json_path,
                "num_tokens": seq_len,
                "num_layers": num_layers,
                "chunk_size": chunk_size,
                "model_name": config["model_name"],
                "capture_time_seconds": round(capture_elapsed, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

            # Copy relevant metadata from trace JSON if present
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

            # Log disk usage
            file_size_mb = os.path.getsize(activation_path) / (1024 * 1024)
            print(
                f"Batch {batch_id}: Saved {os.path.basename(activation_path)} -- "
                f"{file_size_mb:.1f} MB, {seq_len} tokens, {capture_elapsed:.1f}s capture"
            )

            results["captured"] += 1
            commit_counter += 1

            # Commit volume every 10 traces for durability
            if commit_counter >= 10:
                activations_vol.commit()
                commit_counter = 0

        except Exception as e:
            results["failed"] += 1
            print(
                f"Batch {batch_id}: Error capturing {problem_id} trace {trace_idx}: {e}"
            )
            continue

    # Final commit
    activations_vol.commit()

    print(
        f"Batch {batch_id}: Done. Captured: {results['captured']}, "
        f"Skipped: {results['skipped']}, Failed: {results['failed']}"
    )

    return {"batch_id": batch_id, **results}


# ---------------------------------------------------------------------------
# Orchestrator Function (distributes work across containers)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={
        "/traces": traces_vol,
        "/activations": activations_vol,
    },
    timeout=3600,
)
def capture_all_activations(config: dict) -> dict:
    """Orchestrate activation capture across multiple containers.

    Scans for valid traces, filters already-captured, splits work,
    and dispatches to parallel workers.

    Args:
        config: Full configuration dict from configs/modal/capture.yaml.

    Returns:
        Aggregated results dict with total counts.
    """
    n_containers = config["modal"]["n_containers"]

    # Reload volumes to see latest data
    traces_vol.reload()
    activations_vol.reload()

    # Scan valid traces
    print("Scanning valid traces...")
    all_traces = scan_valid_traces("/traces")
    print(f"Found {len(all_traces)} valid trace JSON files")

    if not all_traces:
        print("No valid traces found. Nothing to capture.")
        return {"captured": 0, "skipped": 0, "failed": 0}

    # Count per-dataset for logging
    dataset_counts = {}
    for item in all_traces:
        ds = item["dataset_name"]
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    for ds_name, count in sorted(dataset_counts.items()):
        print(f"  {ds_name}: {count} traces")

    # Filter already-captured traces
    pending_traces = []
    already_captured = 0

    for item in all_traces:
        activation_path = os.path.join(
            "/activations",
            item["dataset_name"],
            item["problem_id"],
            f"trace_{item['trace_idx']:02d}.safetensors",
        )
        if os.path.exists(activation_path):
            already_captured += 1
        else:
            pending_traces.append(item)

    print(f"Already captured: {already_captured}, Pending: {len(pending_traces)}")

    if not pending_traces:
        print("All traces already captured. Nothing to do.")
        return {"captured": 0, "skipped": already_captured, "failed": 0}

    # Split into batches for parallel processing
    actual_containers = min(n_containers, len(pending_traces))
    batch_size = len(pending_traces) // actual_containers
    batches = []
    for i in range(actual_containers):
        start = i * batch_size
        end = start + batch_size if i < actual_containers - 1 else len(pending_traces)
        batches.append(pending_traces[start:end])

    print(f"Split into {len(batches)} batches")
    for i, batch in enumerate(batches):
        print(f"  Batch {i}: {len(batch)} traces")

    # Worker config (subset of full config needed by workers)
    worker_config = {
        "model_name": config["model"]["name"],
        "chunk_size": config["capture"]["chunk_size"],
        "num_layers": config["capture"]["num_layers"],
        "top_k_logits": config["capture"]["top_k_logits"],
        "verify_logits": config["capture"]["verify_logits"],
        "logits_rtol": config["capture"]["logits_rtol"],
        "logits_atol": config["capture"]["logits_atol"],
    }

    # Dispatch to workers via .starmap() for parallel execution
    print(f"Dispatching to {actual_containers} containers...")
    start_time = time.time()

    inputs = [(batch, worker_config, i) for i, batch in enumerate(batches)]
    results = list(
        capture_activations_batch.starmap(
            inputs,
            order_outputs=False,
        )
    )

    elapsed = time.time() - start_time
    print(f"All containers finished in {elapsed:.1f}s")

    # Aggregate results
    total = {"captured": 0, "skipped": already_captured, "failed": 0}
    for r in results:
        total["captured"] += r["captured"]
        total["skipped"] += r["skipped"]
        total["failed"] += r["failed"]

    print(f"Total results: {total}")
    return total


# ---------------------------------------------------------------------------
# Local Entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    config: str = None,
    test: bool = False,
    dry_run: bool = False,
    # CLI overrides for common settings
    n_containers: int = None,
    chunk_size: int = None,
):
    """Entry point for `modal run scripts/modal_capture_activations.py`.

    Configuration is loaded from configs/modal/capture.yaml by default.
    CLI arguments override specific settings from the config file.

    Args:
        config: Path to YAML config file (default: configs/modal/capture.yaml).
        test: If True, run a minimal single-trace capture to verify pipeline.
        dry_run: If True, scan and report how many traces need capture.
        n_containers: Override modal.n_containers from config.
        chunk_size: Override capture.chunk_size from config.
    """
    # Load configuration from YAML
    cfg = load_modal_config(config)
    print(f"Loaded config from: {config or 'configs/modal/capture.yaml'}")

    # Apply CLI overrides
    if n_containers is not None:
        cfg["modal"]["n_containers"] = n_containers
    if chunk_size is not None:
        cfg["capture"]["chunk_size"] = chunk_size

    if dry_run:
        print("=== Dry Run: Scanning traces ===")
        # Run the orchestrator scan logic remotely (needs volume access)
        cfg["modal"]["n_containers"] = 0  # Signal dry-run to orchestrator
        result = dry_run_scan.remote(cfg)
        return

    if test:
        print("=== Test Mode: Single trace capture ===")
        result = test_single_capture.remote(cfg)
        print(f"Test result: {result}")
        return

    print(f"Starting activation capture with {cfg['modal']['n_containers']} containers...")
    print(f"  Model: {cfg['model']['name']}")
    print(f"  Num layers: {cfg['capture']['num_layers']}")
    print(f"  Chunk size: {cfg['capture']['chunk_size']}")
    print(f"  Top-k logits: {cfg['capture']['top_k_logits']}")
    print(f"  Verify logits: {cfg['capture']['verify_logits']}")

    results = capture_all_activations.remote(config=cfg)

    print("\n=== Capture Complete ===")
    print(f"Captured: {results['captured']}")
    print(f"Skipped (already done): {results['skipped']}")
    print(f"Failed: {results['failed']}")
    print("\nTo download results:")
    print("  modal volume get activations-output / data/activations/ --force")


# ---------------------------------------------------------------------------
# Dry Run Function
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={
        "/traces": traces_vol,
        "/activations": activations_vol,
    },
    timeout=300,
)
def dry_run_scan(config: dict) -> dict:
    """Scan traces and report counts without dispatching workers."""
    traces_vol.reload()
    activations_vol.reload()

    all_traces = scan_valid_traces("/traces")
    print(f"Found {len(all_traces)} valid trace JSON files")

    if not all_traces:
        print("No valid traces found.")
        return {"total": 0, "pending": 0, "already_captured": 0}

    # Count per-dataset
    dataset_counts = {}
    for item in all_traces:
        ds = item["dataset_name"]
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    for ds_name, count in sorted(dataset_counts.items()):
        print(f"  {ds_name}: {count} traces")

    # Check already captured
    pending = 0
    already_captured = 0
    for item in all_traces:
        activation_path = os.path.join(
            "/activations",
            item["dataset_name"],
            item["problem_id"],
            f"trace_{item['trace_idx']:02d}.safetensors",
        )
        if os.path.exists(activation_path):
            already_captured += 1
        else:
            pending += 1

    print(f"\nTotal traces: {len(all_traces)}")
    print(f"Already captured: {already_captured}")
    print(f"Pending capture: {pending}")

    return {"total": len(all_traces), "pending": pending, "already_captured": already_captured}


# ---------------------------------------------------------------------------
# Test Function (minimal single-trace capture)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="a100-80gb",
    volumes={
        "/model-cache": model_cache_vol,
        "/traces": traces_vol,
        "/activations": activations_vol,
    },
    timeout=600,
)
def test_single_capture(config: dict) -> dict:
    """Run a minimal single-trace capture to verify the pipeline works."""
    import torch
    from nnsight import LanguageModel

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    traces_vol.reload()

    # Find first available trace
    all_traces = scan_valid_traces("/traces")
    if not all_traces:
        return {"status": "error", "message": "No valid traces found"}

    item = all_traces[0]
    print(f"Test trace: {item['dataset_name']}/{item['problem_id']}/trace_{item['trace_idx']:02d}")

    # Load trace JSON
    with open(item["trace_json_path"], "r") as f:
        trace_data = json.load(f)

    token_list = trace_data["token_ids"]
    print(f"Token count: {len(token_list)}")

    # Load model
    print("Loading model...")
    model = LanguageModel(
        config["model"]["name"],
        device_map="cuda:0",
        dtype=torch.bfloat16,
        dispatch=True,
    )
    print("Model loaded")

    # Run capture
    token_ids = torch.tensor(token_list, dtype=torch.long, device="cuda:0")
    print("Running chunked capture...")
    start_time = time.time()
    results = capture_all_layers_chunked(
        model,
        token_ids,
        chunk_size=config["capture"]["chunk_size"],
        num_layers=config["capture"]["num_layers"],
        top_k=config["capture"]["top_k_logits"],
    )
    elapsed = time.time() - start_time

    print(f"Capture completed in {elapsed:.1f}s")
    print(f"Result keys: {sorted(results.keys())}")
    for key, tensor in sorted(results.items()):
        print(f"  {key}: shape={list(tensor.shape)}, dtype={tensor.dtype}")

    # Verify logits
    print("Verifying logits...")
    verify_ok = verify_logits(
        model,
        token_ids,
        results["logits_last_token"],
        num_layers=config["capture"]["num_layers"],
        rtol=config["capture"]["logits_rtol"],
        atol=config["capture"]["logits_atol"],
    )

    # Check expected keys
    expected_keys = (
        [f"residual_layer_{i}" for i in range(config["capture"]["num_layers"])]
        + ["topk_values", "topk_indices", "entropy", "token_ids", "logits_last_token"]
    )
    missing = [k for k in expected_keys if k not in results]

    return {
        "status": "success" if not missing else "missing_keys",
        "capture_time": round(elapsed, 2),
        "num_tokens": len(token_list),
        "num_result_keys": len(results),
        "missing_keys": missing,
        "logits_verified": verify_ok,
    }
