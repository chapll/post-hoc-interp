"""Apply trained probes to generate per-token confidence curves and run transition detection.

Loads probe weights from safetensors, generates confidence curves at every token
position inside think blocks, and runs all four transition detection methods
(derivative, PELT, BinSeg, sigmoid) on each trace. Produces per-trace transition
JSON files with confidence curves, detection results, and three-way classification.

Output:
  - $DATA_DIR/results/phase3/transitions/{dataset}/{problem_id}_trace_{idx}.json
  - Summary statistics printed to stdout

Usage:
    python scripts/detect_transitions.py --config configs/phase3.yaml
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
from safetensors.torch import load_file
from tqdm import tqdm

from src.config import load_config
from src.model import HIDDEN_DIM, NUM_LAYERS, load_model
from src.tokens import count_think_blocks, get_think_content_mask, parse_think_boundaries
from src.transition import detect_transitions, sweep_pelt_penalties
from src.utils import set_seed, setup_logging

logger = setup_logging("detect_transitions")

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Probe Loading
# ---------------------------------------------------------------------------

def load_probes(
    probes_dir: str,
    datasets: list[str],
    device: torch.device,
) -> dict[str, dict[int, nn.Linear]]:
    """Load trained probe weights from safetensors and reconstruct nn.Linear modules.

    Loads per-dataset probe files and the pooled probe file. Each safetensors
    file contains keys layer_{i}_weight [2, 4096] and layer_{i}_bias [2] for
    all 36 layers.

    Args:
        probes_dir: Path to directory containing *_probes.safetensors files.
        datasets: List of dataset names (e.g., ["gsm_symbolic", "folio", "strategyqa"]).
        device: Torch device to move probes to.

    Returns:
        Dict mapping scope name (dataset name or "pooled") to dict mapping
        layer index to nn.Linear(4096, 2) module.
    """
    probes = {}

    # Load per-dataset probes
    for ds in datasets:
        safetensors_path = os.path.join(probes_dir, f"{ds}_probes.safetensors")
        if not os.path.isfile(safetensors_path):
            logger.warning("Probe file not found for dataset %s: %s", ds, safetensors_path)
            continue

        tensors = load_file(safetensors_path)
        probes[ds] = _reconstruct_probes(tensors, device)
        logger.info("Loaded %d layer probes for dataset: %s", len(probes[ds]), ds)

    # Load pooled probes
    pooled_path = os.path.join(probes_dir, "pooled_probes.safetensors")
    if os.path.isfile(pooled_path):
        tensors = load_file(pooled_path)
        probes["pooled"] = _reconstruct_probes(tensors, device)
        logger.info("Loaded %d layer probes for pooled scope", len(probes["pooled"]))
    else:
        logger.warning("Pooled probe file not found: %s", pooled_path)

    return probes


def _reconstruct_probes(
    tensors: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[int, nn.Linear]:
    """Reconstruct nn.Linear modules from safetensors weight/bias pairs.

    Args:
        tensors: Dict with keys like layer_{i}_weight and layer_{i}_bias.
        device: Device to place modules on.

    Returns:
        Dict mapping layer index to nn.Linear(4096, 2).
    """
    probes = {}

    # Find all layer indices that have both weight and bias
    layer_indices = set()
    for key in tensors:
        if key.startswith("layer_") and key.endswith("_weight"):
            idx_str = key.replace("layer_", "").replace("_weight", "")
            try:
                layer_indices.add(int(idx_str))
            except ValueError:
                continue

    for layer_idx in sorted(layer_indices):
        weight_key = f"layer_{layer_idx}_weight"
        bias_key = f"layer_{layer_idx}_bias"

        if weight_key not in tensors or bias_key not in tensors:
            continue

        weight = tensors[weight_key]  # [2, 4096]
        bias = tensors[bias_key]  # [2]

        linear = nn.Linear(HIDDEN_DIM, 2, bias=True)
        linear.weight.data = weight.float().to(device)
        linear.bias.data = bias.float().to(device)
        linear.eval()
        probes[layer_idx] = linear

    return probes


# ---------------------------------------------------------------------------
# 2. Trace Discovery
# ---------------------------------------------------------------------------

def discover_traces(traces_dir: str) -> list[dict]:
    """Discover all valid traces across all datasets.

    Walks $DATA_DIR/traces/valid/{dataset}/{problem_id}/ and loads all
    trace_*.json files. Filters to single-think-block traces with </think>.

    Args:
        traces_dir: Root traces directory (contains valid/ subdirectory).

    Returns:
        List of dicts with keys: trace_path, token_ids, dataset, problem_id,
        trace_idx, ground_truth, is_correct, n_tokens.
    """
    valid_dir = os.path.join(traces_dir, "valid")
    if not os.path.isdir(valid_dir):
        raise FileNotFoundError(f"Valid traces directory not found: {valid_dir}")

    datasets = sorted(
        d for d in os.listdir(valid_dir)
        if os.path.isdir(os.path.join(valid_dir, d))
    )
    logger.info("Found datasets: %s", datasets)

    all_traces = []
    skipped = {"multi_block": 0, "no_end_think": 0, "read_error": 0}

    for dataset in datasets:
        ds_dir = os.path.join(valid_dir, dataset)
        problem_dirs = sorted(
            d for d in os.listdir(ds_dir)
            if os.path.isdir(os.path.join(ds_dir, d))
        )

        for problem_id in problem_dirs:
            problem_dir = os.path.join(ds_dir, problem_id)

            # Load ALL trace files (not just trace_00)
            trace_files = sorted(
                f for f in os.listdir(problem_dir)
                if f.startswith("trace_") and f.endswith(".json")
            )

            for trace_file in trace_files:
                trace_path = os.path.join(problem_dir, trace_file)

                try:
                    with open(trace_path, "r") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning("Skipping unreadable trace %s: %s", trace_path, e)
                    skipped["read_error"] += 1
                    continue

                token_ids = data["token_ids"]
                metadata = data.get("metadata", {})

                # Filter: single think block only
                n_blocks = count_think_blocks(token_ids)
                if n_blocks != 1:
                    skipped["multi_block"] += 1
                    continue

                # Filter: must have </think>
                boundaries = parse_think_boundaries(token_ids)
                if not boundaries:
                    skipped["no_end_think"] += 1
                    continue

                # Extract trace index from filename
                trace_idx_str = trace_file.replace("trace_", "").replace(".json", "")
                try:
                    trace_idx = int(trace_idx_str)
                except ValueError:
                    trace_idx = 0

                all_traces.append({
                    "trace_path": trace_path,
                    "token_ids": token_ids,
                    "dataset": dataset,
                    "problem_id": problem_id,
                    "trace_idx": trace_idx,
                    "ground_truth": metadata.get("ground_truth", "unknown"),
                    "is_correct": metadata.get("is_correct", None),
                    "n_tokens": len(token_ids),
                })

    logger.info(
        "Discovered %d valid traces. Skipped: multi_block=%d, no_end_think=%d, read_error=%d",
        len(all_traces), skipped["multi_block"], skipped["no_end_think"], skipped["read_error"],
    )

    # Per-dataset counts
    for ds in datasets:
        count = sum(1 for t in all_traces if t["dataset"] == ds)
        logger.info("  %s: %d traces", ds, count)

    return all_traces


# ---------------------------------------------------------------------------
# 3. Trace Processing (Confidence Curves)
# ---------------------------------------------------------------------------

def process_trace(
    trace_info: dict,
    model,
    probes: dict[int, nn.Linear],
    probe_layers: list[int],
    device: torch.device,
) -> dict | None:
    """Generate per-token confidence curves for a single trace.

    Runs an nnsight forward pass to extract residual stream activations at the
    specified layers, then applies probes to every eligible token position
    inside think blocks.

    Args:
        trace_info: Dict with token_ids, dataset, problem_id, trace_idx.
        model: nnsight LanguageModel.
        probes: Dict mapping layer_idx to nn.Linear.
        probe_layers: List of layer indices to probe.
        device: Torch device.

    Returns:
        Dict with confidence curves per layer, or None if processing fails.
    """
    token_ids = trace_info["token_ids"]

    # Get think content mask -- marks positions strictly inside <think>...</think>
    # Boundary tokens (<think> and </think>) are excluded by the mask
    mask = get_think_content_mask(token_ids)
    eligible_positions = [i for i, m in enumerate(mask) if m]

    if len(eligible_positions) < 5:
        logger.warning(
            "Trace %s/%s/trace_%02d has too few eligible positions (%d), skipping",
            trace_info["dataset"], trace_info["problem_id"],
            trace_info["trace_idx"], len(eligible_positions),
        )
        return None

    # Prepare input tensor
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Run nnsight forward pass extracting residual stream at probe layers
    saved_residuals = {}

    try:
        with model.trace(input_ids):
            for layer_idx in sorted(probe_layers):
                saved_residuals[layer_idx] = model.model.layers[layer_idx].output.save()
    except Exception as exc:
        logger.error(
            "nnsight forward pass failed for %s/%s/trace_%02d: %s",
            trace_info["dataset"], trace_info["problem_id"],
            trace_info["trace_idx"], exc,
        )
        return None

    # Apply probes at every eligible position
    confidence_curves = {}

    for layer_idx in probe_layers:
        if layer_idx not in probes:
            continue

        probe = probes[layer_idx]

        # Get residual stream: shape [1, seq_len, hidden_dim]
        residual = saved_residuals[layer_idx].detach()
        # Handle tuple output from nnsight (some layers return tuples)
        if isinstance(residual, tuple):
            residual = residual[0]
        if residual.dim() == 2:
            residual = residual.unsqueeze(0)

        positions = []
        confidences = []

        with torch.no_grad():
            for pos in eligible_positions:
                h = residual[0, pos, :].float()  # [hidden_dim], float32
                logits = probe(h.unsqueeze(0))  # [1, 2]
                prob = F.softmax(logits, dim=-1)  # [1, 2]
                # Class 1 confidence (the "Yes" / ground-truth-known class)
                confidence = prob[0, 1].item()
                positions.append(pos)
                confidences.append(confidence)

        confidence_curves[layer_idx] = {
            "positions": positions,
            "values": confidences,
        }

    return {
        "eligible_positions": eligible_positions,
        "n_eligible": len(eligible_positions),
        "confidence_curves": confidence_curves,
    }


# ---------------------------------------------------------------------------
# 4. Transition Detection
# ---------------------------------------------------------------------------

def detect_trace_transitions(
    confidence_curves: dict[int, dict],
    best_layer: int,
    transition_config: dict,
    pelt_penalties: list[float],
) -> dict:
    """Run all transition detection methods on a trace's confidence curves.

    Runs detection at each probed layer. Primary detection uses the best_layer;
    other layers provide comparison data.

    Args:
        confidence_curves: Dict mapping layer_idx to {positions, values}.
        best_layer: The layer with the best probe AUROC.
        transition_config: Config dict for transition detection methods.
        pelt_penalties: List of penalties for PELT sweep.

    Returns:
        Dict with detection results per layer and overall classification.
    """
    results = {}
    primary_classification = "no-signal"

    for layer_idx, curve in confidence_curves.items():
        positions = np.array(curve["positions"])
        values = np.array(curve["values"])

        if len(positions) < 5:
            results[layer_idx] = {
                "methods": {},
                "pelt_sweep": {},
                "classification": "no-signal",
                "note": "too few positions for detection",
            }
            continue

        # Run all detection methods
        method_results = detect_transitions(
            positions, values,
            methods=transition_config.get("methods", ["derivative", "pelt", "binseg", "sigmoid"]),
            config=transition_config,
        )

        # PELT penalty sweep
        pelt_sweep = sweep_pelt_penalties(
            positions, values,
            penalties=pelt_penalties,
            model=transition_config.get("pelt", {}).get("model", "l2"),
            min_size=transition_config.get("pelt", {}).get("min_size", 5),
        )

        # Serialize method_specific to avoid numpy arrays in JSON
        serialized_methods = {}
        for method_name, mresult in method_results.items():
            serialized = _serialize_method_result(mresult)
            serialized_methods[method_name] = serialized

        serialized_pelt = {}
        for pen, presult in pelt_sweep.items():
            serialized_pelt[str(pen)] = _serialize_method_result(presult)

        layer_result = {
            "methods": serialized_methods,
            "pelt_sweep": serialized_pelt,
        }

        results[layer_idx] = layer_result

        # Use best_layer for primary classification
        if layer_idx == best_layer:
            primary_classification = _classify_from_methods(method_results, values)

    return {
        "per_layer": results,
        "primary_classification": primary_classification,
        "best_layer": best_layer,
    }


def _classify_from_methods(method_results: dict, values: np.ndarray) -> str:
    """Determine three-way classification from method results.

    Classification logic:
    - "clear": at least one method finds a transition AND plateau is detected
    - "no-signal": probe confidence stays below 0.70 throughout
    - "noisy-signal": everything else

    Args:
        method_results: Dict of method name to detection result.
        values: Raw confidence values.

    Returns:
        One of "clear", "no-signal", "noisy-signal".
    """
    if np.max(values) < 0.70:
        return "no-signal"

    # Check if any method found a clear transition with plateau
    for method_name, result in method_results.items():
        if result.get("classification") == "clear":
            return "clear"

    return "noisy-signal"


def _serialize_method_result(result: dict) -> dict:
    """Serialize a detection method result for JSON storage.

    Converts numpy arrays and numpy types to Python lists and native types.

    Args:
        result: Raw method result dict from detect_transitions.

    Returns:
        JSON-serializable dict.
    """
    serialized = {}
    for key, val in result.items():
        if key == "method_specific":
            ms = {}
            for mk, mv in val.items():
                if isinstance(mv, np.ndarray):
                    ms[mk] = mv.tolist()
                elif isinstance(mv, (np.integer, np.floating)):
                    ms[mk] = float(mv) if isinstance(mv, np.floating) else int(mv)
                elif isinstance(mv, dict):
                    # Nested dict (e.g., threshold_results)
                    ms[mk] = _serialize_nested(mv)
                else:
                    ms[mk] = mv
            serialized[key] = ms
        elif key == "segments":
            # Segments are list of (start, end, label) tuples
            serialized[key] = [
                [_to_native(s), _to_native(e), label]
                for s, e, label in val
            ]
        elif isinstance(val, (np.integer, np.floating)):
            serialized[key] = float(val) if isinstance(val, np.floating) else int(val)
        else:
            serialized[key] = val
    return serialized


def _serialize_nested(d: dict) -> dict:
    """Recursively serialize nested dict values."""
    result = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            result[k] = float(v) if isinstance(v, np.floating) else int(v)
        elif isinstance(v, dict):
            result[k] = _serialize_nested(v)
        else:
            result[k] = v
    return result


def _to_native(val):
    """Convert numpy scalar to native Python type."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


# ---------------------------------------------------------------------------
# 5. Output and Summary
# ---------------------------------------------------------------------------

def save_trace_result(
    trace_info: dict,
    confidence_curves: dict,
    detection_results: dict,
    transitions_dir: str,
) -> str:
    """Save per-trace transition detection results to JSON.

    Args:
        trace_info: Trace metadata (dataset, problem_id, trace_idx, etc.).
        confidence_curves: Confidence curve data per layer.
        detection_results: Detection results from detect_trace_transitions.
        transitions_dir: Root output directory for transition results.

    Returns:
        Path to the saved JSON file.
    """
    dataset = trace_info["dataset"]
    problem_id = trace_info["problem_id"]
    trace_idx = trace_info["trace_idx"]

    output_dir = os.path.join(transitions_dir, dataset)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{problem_id}_trace_{trace_idx:02d}.json"
    output_path = os.path.join(output_dir, filename)

    # Build trace ID
    trace_id = f"{dataset}/{problem_id}/trace_{trace_idx:02d}"

    # Serialize confidence curves for JSON
    serialized_curves = {}
    for layer_idx, curve in confidence_curves.items():
        layer_key = f"values_layer_{layer_idx}"
        serialized_curves[layer_key] = curve["values"]
    serialized_curves["positions"] = confidence_curves[next(iter(confidence_curves))]["positions"]

    result = {
        "trace_id": trace_id,
        "problem_id": problem_id,
        "dataset": dataset,
        "trace_idx": trace_idx,
        "classification": detection_results["primary_classification"],
        "best_layer": detection_results["best_layer"],
        "n_think_tokens": len(serialized_curves["positions"]),
        "ground_truth": trace_info.get("ground_truth", "unknown"),
        "is_correct": trace_info.get("is_correct"),
        "confidence_curve": serialized_curves,
        "detection_results": {
            f"layer_{layer_idx}": layer_data
            for layer_idx, layer_data in detection_results["per_layer"].items()
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return output_path


def print_summary(
    classification_counts: dict[str, dict[str, int]],
    transition_positions: dict[str, dict[str, list[float]]],
    total_traces: int,
    total_processed: int,
    total_skipped: int,
):
    """Print summary statistics to stdout.

    Args:
        classification_counts: {dataset: {classification: count}}.
        transition_positions: {dataset: {method: [normalized_positions]}}.
        total_traces: Total traces discovered.
        total_processed: Total traces successfully processed.
        total_skipped: Total traces skipped.
    """
    print("\n" + "=" * 70)
    print("TRANSITION DETECTION SUMMARY")
    print("=" * 70)

    print(f"\nTotal traces discovered: {total_traces}")
    print(f"Total traces processed: {total_processed}")
    print(f"Total traces skipped: {total_skipped}")

    print("\n--- Classification Distribution ---")
    for dataset in sorted(classification_counts.keys()):
        counts = classification_counts[dataset]
        total_ds = sum(counts.values())
        print(f"\n  {dataset} (n={total_ds}):")
        for cls_name in ["clear", "noisy-signal", "no-signal"]:
            count = counts.get(cls_name, 0)
            frac = count / total_ds if total_ds > 0 else 0.0
            print(f"    {cls_name:15s}: {count:4d} ({frac:.1%})")

        # Indeterminate rate
        indeterminate = counts.get("noisy-signal", 0) + counts.get("no-signal", 0)
        indet_frac = indeterminate / total_ds if total_ds > 0 else 0.0
        print(f"    {'INDETERMINATE':15s}: {indeterminate:4d} ({indet_frac:.1%})")

    # Overall indeterminate
    total_all = sum(sum(c.values()) for c in classification_counts.values())
    total_clear = sum(c.get("clear", 0) for c in classification_counts.values())
    total_indet = total_all - total_clear
    if total_all > 0:
        print(f"\n  OVERALL INDETERMINATE RATE: {total_indet}/{total_all} ({total_indet / total_all:.1%})")

    print("\n--- Median Transition Point (normalized) for Clear Traces ---")
    for dataset in sorted(transition_positions.keys()):
        methods = transition_positions[dataset]
        print(f"\n  {dataset}:")
        for method_name, positions in sorted(methods.items()):
            if positions:
                median = np.median(positions)
                print(f"    {method_name:12s}: median={median:.3f} (n={len(positions)})")
            else:
                print(f"    {method_name:12s}: no clear traces")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for transition detection script."""
    cfg = load_config()

    set_seed(cfg.seed)

    # Resolve paths
    data_dir = os.environ.get("DATA_DIR", cfg.paths.get("data_dir", "data"))
    traces_dir = os.path.join(data_dir, "traces")
    results_dir = os.path.join(data_dir, "results", "phase3")
    probes_dir = os.path.join(results_dir, "probes")
    transitions_dir = os.path.join(results_dir, "transitions")
    os.makedirs(transitions_dir, exist_ok=True)

    # Load probe AUROC results to get best_layer and top_layers
    aurocs_path = os.path.join(results_dir, "probe_aurocs.json")
    if not os.path.isfile(aurocs_path):
        raise FileNotFoundError(f"Probe AUROC results not found: {aurocs_path}")

    with open(aurocs_path, "r") as f:
        auroc_data = json.load(f)

    best_layer = auroc_data["best_layer"]
    top_layers = auroc_data.get("top_layers", [best_layer])
    logger.info("Best layer: %d, Top layers: %s", best_layer, top_layers)

    # Determine probe layers to use: top_layers from pilot
    probe_layers = list(top_layers)
    if best_layer not in probe_layers:
        probe_layers.insert(0, best_layer)
    logger.info("Probing at layers: %s", probe_layers)

    # Discover datasets from traces directory
    valid_dir = os.path.join(traces_dir, "valid")
    datasets = sorted(
        d for d in os.listdir(valid_dir)
        if os.path.isdir(os.path.join(valid_dir, d))
    )

    # Load model
    gpu_id = cfg.gpus[0] if hasattr(cfg, "gpus") and cfg.gpus else 0
    device = torch.device(f"cuda:{gpu_id}")
    model, tokenizer = load_model(gpu_id)

    # Load probes (per-dataset)
    probes_all = load_probes(probes_dir, datasets, device)

    # Discover all traces
    all_traces = discover_traces(traces_dir)

    # Build transition config from phase3 config
    transition_config = {}
    if hasattr(cfg, "phase3") and hasattr(cfg.phase3, "transition"):
        tc = cfg.phase3.transition
        for key in ["derivative", "pelt", "binseg", "sigmoid"]:
            if hasattr(tc, key):
                from omegaconf import OmegaConf
                transition_config[key] = OmegaConf.to_container(getattr(tc, key), resolve=True)
        if hasattr(tc, "methods"):
            transition_config["methods"] = list(tc.methods)

    if hasattr(cfg, "phase3") and hasattr(cfg.phase3, "classification"):
        from omegaconf import OmegaConf
        transition_config["classification"] = OmegaConf.to_container(
            cfg.phase3.classification, resolve=True
        )

    pelt_penalties = transition_config.get("pelt", {}).get(
        "penalties", [1, 5, 10, 20, 50, 100]
    )

    # Process all traces
    classification_counts = {}  # {dataset: {cls: count}}
    transition_positions = {}   # {dataset: {method: [normalized_pos]}}
    total_processed = 0
    total_skipped = 0

    start_time = time.time()

    for trace_info in tqdm(all_traces, desc="Processing traces"):
        dataset = trace_info["dataset"]

        # Select probes: prefer per-dataset, fall back to pooled
        if dataset in probes_all:
            probes = probes_all[dataset]
        elif "pooled" in probes_all:
            probes = probes_all["pooled"]
        else:
            logger.warning("No probes available for dataset %s, skipping", dataset)
            total_skipped += 1
            continue

        # Filter probes to only the layers we want
        active_probes = {l: p for l, p in probes.items() if l in probe_layers}
        if not active_probes:
            logger.warning("No probes at probe_layers for dataset %s", dataset)
            total_skipped += 1
            continue

        # Generate confidence curves
        curve_result = process_trace(trace_info, model, active_probes, probe_layers, device)
        if curve_result is None:
            total_skipped += 1
            continue

        # Run transition detection
        detection_results = detect_trace_transitions(
            curve_result["confidence_curves"],
            best_layer=best_layer,
            transition_config=transition_config,
            pelt_penalties=pelt_penalties,
        )

        # Save per-trace result
        save_trace_result(
            trace_info,
            curve_result["confidence_curves"],
            detection_results,
            transitions_dir,
        )

        total_processed += 1

        # Track classification stats
        classification = detection_results["primary_classification"]
        if dataset not in classification_counts:
            classification_counts[dataset] = {}
        classification_counts[dataset][classification] = (
            classification_counts[dataset].get(classification, 0) + 1
        )

        # Track transition positions for clear traces
        if classification == "clear" and best_layer in detection_results["per_layer"]:
            if dataset not in transition_positions:
                transition_positions[dataset] = {}

            layer_data = detection_results["per_layer"][best_layer]
            n_think = curve_result["n_eligible"]

            for method_name, mresult in layer_data.get("methods", {}).items():
                tp = mresult.get("transition_point")
                if tp is not None and n_think > 0:
                    # Normalize transition point to [0, 1] range
                    positions = curve_result["confidence_curves"][best_layer]["positions"]
                    min_pos = min(positions)
                    max_pos = max(positions)
                    pos_range = max_pos - min_pos
                    if pos_range > 0:
                        normalized = (tp - min_pos) / pos_range
                    else:
                        normalized = 0.0
                    if method_name not in transition_positions[dataset]:
                        transition_positions[dataset][method_name] = []
                    transition_positions[dataset][method_name].append(normalized)

    elapsed = time.time() - start_time

    logger.info(
        "Transition detection complete. Processed=%d, Skipped=%d, Elapsed=%.1fs",
        total_processed, total_skipped, elapsed,
    )

    # Save summary statistics
    summary_path = os.path.join(results_dir, "transition_summary.json")
    summary_data = {
        "total_traces_discovered": len(all_traces),
        "total_processed": total_processed,
        "total_skipped": total_skipped,
        "elapsed_seconds": round(elapsed, 1),
        "best_layer": best_layer,
        "probe_layers": probe_layers,
        "classification_counts": classification_counts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)

    # Print summary
    print_summary(
        classification_counts,
        transition_positions,
        total_traces=len(all_traces),
        total_processed=total_processed,
        total_skipped=total_skipped,
    )


if __name__ == "__main__":
    main()
