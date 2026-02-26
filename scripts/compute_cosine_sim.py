"""Compute step-wise cosine similarity dynamics and pre-T vs post-T statistics.

Computes S_t = cos(h_t, h_{t-1}) at the best-probe-AUROC layer and 2-3
additional layers for every trace. For traces classified as "clear" by the
transition detection script, runs Mann-Whitney U test and Cohen's d comparing
pre-transition vs post-transition cosine similarity values.

Output:
  - Appends cosine_similarity field to per-trace transition JSONs
  - $DATA_DIR/results/phase3/cosine_similarity_stats.json (aggregate statistics)

Usage:
    python scripts/compute_cosine_sim.py --config configs/phase3.yaml
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from scipy import stats
from tqdm import tqdm

from src.config import load_config
from src.model import HIDDEN_DIM, NUM_LAYERS, load_model
from src.tokens import count_think_blocks, get_think_content_mask, parse_think_boundaries
from src.utils import set_seed, setup_logging

logger = setup_logging("compute_cosine_sim")

load_dotenv()


# ---------------------------------------------------------------------------
# 1. Cosine Similarity Computation
# ---------------------------------------------------------------------------

def compute_trace_cosine(
    token_ids: list[int],
    model,
    cosine_layers: list[int],
    device: torch.device,
) -> dict[int, dict] | None:
    """Compute step-wise cosine similarity for a single trace.

    For each pair of consecutive eligible positions (t, t-1), computes:
        S_t = cos(h_t, h_{t-1})

    Args:
        token_ids: List of integer token IDs for the trace.
        model: nnsight LanguageModel.
        cosine_layers: List of layer indices to compute cosine similarity at.
        device: Torch device.

    Returns:
        Dict mapping layer_idx to {positions, S_t} where positions are the
        second element of each pair and S_t are the similarity values.
        Returns None if the trace cannot be processed.
    """
    # Get think content mask
    mask = get_think_content_mask(token_ids)
    eligible_positions = [i for i, m in enumerate(mask) if m]

    if len(eligible_positions) < 3:
        return None

    # Prepare input tensor
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Run nnsight forward pass extracting residual stream at cosine layers
    saved_residuals = {}

    try:
        with model.trace(input_ids):
            for layer_idx in sorted(cosine_layers):
                saved_residuals[layer_idx] = model.model.layers[layer_idx].output.save()
    except Exception as exc:
        logger.error("nnsight forward pass failed: %s", exc)
        return None

    # Compute step-wise cosine similarity at each layer
    results = {}

    for layer_idx in cosine_layers:
        if layer_idx not in saved_residuals:
            continue

        # Get residual stream: shape [1, seq_len, hidden_dim]
        residual = saved_residuals[layer_idx].detach()
        if isinstance(residual, tuple):
            residual = residual[0]
        if residual.dim() == 2:
            residual = residual.unsqueeze(0)

        positions = []
        s_t_values = []

        with torch.no_grad():
            for i in range(1, len(eligible_positions)):
                pos_curr = eligible_positions[i]
                pos_prev = eligible_positions[i - 1]

                h_curr = residual[0, pos_curr, :].float().unsqueeze(0)  # [1, hidden_dim]
                h_prev = residual[0, pos_prev, :].float().unsqueeze(0)  # [1, hidden_dim]

                sim = F.cosine_similarity(h_curr, h_prev, dim=-1).item()

                positions.append(pos_curr)
                s_t_values.append(sim)

        results[layer_idx] = {
            "positions": positions,
            "S_t": s_t_values,
        }

    return results


# ---------------------------------------------------------------------------
# 2. Pre-T vs Post-T Statistical Comparison
# ---------------------------------------------------------------------------

def compare_pre_post(
    cosine_data: dict,
    transition_point: int | None,
) -> dict | None:
    """Compare pre-transition vs post-transition cosine similarity.

    For traces with a clear transition point, splits S_t values into
    pre-T and post-T groups and runs Mann-Whitney U test with Cohen's d.

    Args:
        cosine_data: Dict with keys "positions" and "S_t" for one layer.
        transition_point: Token position of the transition, or None.

    Returns:
        Dict with pre_mean, post_mean, mann_whitney_u, p_value, cohens_d,
        n_pre, n_post. Returns None if comparison cannot be performed.
    """
    if transition_point is None:
        return None

    positions = cosine_data["positions"]
    s_t = cosine_data["S_t"]

    if len(positions) < 4:
        return None

    # Split into pre-T and post-T
    pre_values = []
    post_values = []

    for pos, val in zip(positions, s_t):
        if pos < transition_point:
            pre_values.append(val)
        else:
            post_values.append(val)

    if len(pre_values) < 2 or len(post_values) < 2:
        return None

    pre_arr = np.array(pre_values)
    post_arr = np.array(post_values)

    pre_mean = float(np.mean(pre_arr))
    post_mean = float(np.mean(post_arr))

    # Mann-Whitney U test
    try:
        u_stat, p_value = stats.mannwhitneyu(pre_arr, post_arr, alternative="two-sided")
        u_stat = float(u_stat)
        p_value = float(p_value)
    except ValueError:
        # All values identical or other edge case
        u_stat = 0.0
        p_value = 1.0

    # Cohen's d
    pooled_std = _pooled_std(pre_arr, post_arr)
    if pooled_std > 0:
        cohens_d = float((post_mean - pre_mean) / pooled_std)
    else:
        cohens_d = 0.0

    return {
        "pre_mean": pre_mean,
        "post_mean": post_mean,
        "mann_whitney_u": u_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "n_pre": len(pre_values),
        "n_post": len(post_values),
    }


def _pooled_std(a: np.ndarray, b: np.ndarray) -> float:
    """Compute pooled standard deviation for Cohen's d.

    Uses the pooled variance formula:
        s_p = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))

    Args:
        a: First group array.
        b: Second group array.

    Returns:
        Pooled standard deviation.
    """
    n1 = len(a)
    n2 = len(b)
    if n1 + n2 <= 2:
        return 0.0
    var1 = np.var(a, ddof=1) if n1 > 1 else 0.0
    var2 = np.var(b, ddof=1) if n2 > 1 else 0.0
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    return float(np.sqrt(pooled_var))


# ---------------------------------------------------------------------------
# 3. Transition Point Extraction
# ---------------------------------------------------------------------------

def get_transition_point(detection_results: dict, best_layer: int) -> int | None:
    """Extract the primary transition point from detection results.

    Tries methods in preference order: sigmoid, derivative, binseg, pelt.
    Returns the first valid transition point found.

    Args:
        detection_results: Detection results dict from the transition JSON.
        best_layer: The best probe layer.

    Returns:
        Token position of the transition, or None if no method found one.
    """
    layer_key = f"layer_{best_layer}"
    layer_data = detection_results.get(layer_key, {})
    methods = layer_data.get("methods", {})

    # Preference order for transition point extraction
    method_order = ["sigmoid", "derivative", "binseg", "pelt"]

    for method_name in method_order:
        mresult = methods.get(method_name, {})
        tp = mresult.get("transition_point")
        if tp is not None:
            return tp

    return None


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for cosine similarity computation."""
    cfg = load_config()

    set_seed(cfg.seed)

    # Resolve paths
    data_dir = os.environ.get("DATA_DIR", cfg.paths.get("data_dir", "data"))
    results_dir = os.path.join(data_dir, "results", "phase3")
    transitions_dir = os.path.join(results_dir, "transitions")

    if not os.path.isdir(transitions_dir):
        raise FileNotFoundError(
            f"Transitions directory not found: {transitions_dir}. "
            "Run scripts/detect_transitions.py first."
        )

    # Load probe AUROC results for layer selection
    aurocs_path = os.path.join(results_dir, "probe_aurocs.json")
    if not os.path.isfile(aurocs_path):
        raise FileNotFoundError(f"Probe AUROC results not found: {aurocs_path}")

    with open(aurocs_path, "r") as f:
        auroc_data = json.load(f)

    best_layer = auroc_data["best_layer"]
    top_layers = auroc_data.get("top_layers", [best_layer])

    # Select layers for cosine similarity: best + additional from top_layers
    additional_count = 3
    if hasattr(cfg, "phase3") and hasattr(cfg.phase3, "cosine"):
        additional_count = cfg.phase3.cosine.get("additional_layers", 3)

    cosine_layers = [best_layer]
    for layer in top_layers:
        if layer != best_layer and len(cosine_layers) <= additional_count:
            cosine_layers.append(layer)

    logger.info("Computing cosine similarity at layers: %s", cosine_layers)

    # Load model
    gpu_id = cfg.gpus[0] if hasattr(cfg, "gpus") and cfg.gpus else 0
    device = torch.device(f"cuda:{gpu_id}")
    model, tokenizer = load_model(gpu_id)

    # Find all transition JSON files
    transition_files = []
    for dataset_name in sorted(os.listdir(transitions_dir)):
        ds_dir = os.path.join(transitions_dir, dataset_name)
        if not os.path.isdir(ds_dir):
            continue
        for fname in sorted(os.listdir(ds_dir)):
            if fname.endswith(".json"):
                transition_files.append(os.path.join(ds_dir, fname))

    logger.info("Found %d transition result files", len(transition_files))

    # Also need trace JSONs to get token_ids for forward pass
    traces_dir = os.path.join(data_dir, "traces")

    # Aggregate statistics
    per_dataset_stats = {}  # {dataset: {layer: {metrics_list}}}
    total_processed = 0
    total_skipped = 0

    start_time = time.time()

    for trans_path in tqdm(transition_files, desc="Computing cosine similarity"):
        # Load transition result
        with open(trans_path, "r") as f:
            trans_data = json.load(f)

        dataset = trans_data["dataset"]
        problem_id = trans_data["problem_id"]
        trace_idx = trans_data.get("trace_idx", 0)
        classification = trans_data["classification"]

        # Load original trace to get token_ids
        trace_path = os.path.join(
            traces_dir, "valid", dataset, problem_id,
            f"trace_{trace_idx:02d}.json"
        )
        if not os.path.isfile(trace_path):
            logger.warning("Original trace not found: %s", trace_path)
            total_skipped += 1
            continue

        with open(trace_path, "r") as f:
            trace_data = json.load(f)

        token_ids = trace_data["token_ids"]

        # Compute cosine similarity
        cosine_result = compute_trace_cosine(
            token_ids, model, cosine_layers, device
        )
        if cosine_result is None:
            total_skipped += 1
            continue

        # Get transition point for clear traces
        transition_point = None
        if classification == "clear":
            transition_point = get_transition_point(
                trans_data.get("detection_results", {}),
                trans_data.get("best_layer", best_layer),
            )

        # Build cosine similarity output
        cosine_output = {}
        for layer_idx, cdata in cosine_result.items():
            layer_key = f"layer_{layer_idx}"

            layer_output = {
                "positions": cdata["positions"],
                "S_t": cdata["S_t"],
            }

            # Run pre/post comparison for clear traces
            if classification == "clear" and transition_point is not None:
                comparison = compare_pre_post(cdata, transition_point)
                if comparison is not None:
                    layer_output["pre_post_comparison"] = comparison

                    # Accumulate stats
                    if dataset not in per_dataset_stats:
                        per_dataset_stats[dataset] = {}
                    if layer_key not in per_dataset_stats[dataset]:
                        per_dataset_stats[dataset][layer_key] = {
                            "cohens_d": [],
                            "p_values": [],
                            "pre_means": [],
                            "post_means": [],
                        }
                    stats_bucket = per_dataset_stats[dataset][layer_key]
                    stats_bucket["cohens_d"].append(comparison["cohens_d"])
                    stats_bucket["p_values"].append(comparison["p_value"])
                    stats_bucket["pre_means"].append(comparison["pre_mean"])
                    stats_bucket["post_means"].append(comparison["post_mean"])

            cosine_output[layer_key] = layer_output

        # Append cosine similarity to the transition JSON
        trans_data["cosine_similarity"] = cosine_output

        with open(trans_path, "w") as f:
            json.dump(trans_data, f, indent=2)

        total_processed += 1

    elapsed = time.time() - start_time

    logger.info(
        "Cosine similarity computation complete. Processed=%d, Skipped=%d, Elapsed=%.1fs",
        total_processed, total_skipped, elapsed,
    )

    # Build and save aggregate statistics
    aggregate = {"per_dataset": {}}

    for dataset, layers in per_dataset_stats.items():
        aggregate["per_dataset"][dataset] = {}
        for layer_key, metrics in layers.items():
            n = len(metrics["cohens_d"])
            if n == 0:
                continue
            d_arr = np.array(metrics["cohens_d"])
            p_arr = np.array(metrics["p_values"])
            pre_arr = np.array(metrics["pre_means"])
            post_arr = np.array(metrics["post_means"])

            aggregate["per_dataset"][dataset][layer_key] = {
                "n_clear_traces": n,
                "mean_cohens_d": float(np.mean(d_arr)),
                "median_cohens_d": float(np.median(d_arr)),
                "std_cohens_d": float(np.std(d_arr)),
                "mean_p_value": float(np.mean(p_arr)),
                "median_p_value": float(np.median(p_arr)),
                "fraction_significant_p05": float(np.mean(p_arr < 0.05)),
                "fraction_significant_p01": float(np.mean(p_arr < 0.01)),
                "mean_pre_mean": float(np.mean(pre_arr)),
                "mean_post_mean": float(np.mean(post_arr)),
            }

    aggregate["metadata"] = {
        "cosine_layers": cosine_layers,
        "best_layer": best_layer,
        "total_processed": total_processed,
        "total_skipped": total_skipped,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    stats_path = os.path.join(results_dir, "cosine_similarity_stats.json")
    with open(stats_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    logger.info("Aggregate statistics saved to: %s", stats_path)

    # Print summary
    _print_summary(aggregate, total_processed, total_skipped, elapsed)


def _print_summary(
    aggregate: dict,
    total_processed: int,
    total_skipped: int,
    elapsed: float,
):
    """Print cosine similarity summary to stdout."""
    print("\n" + "=" * 70)
    print("COSINE SIMILARITY DYNAMICS SUMMARY")
    print("=" * 70)

    print(f"\nTotal traces processed: {total_processed}")
    print(f"Total traces skipped: {total_skipped}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Cosine layers: {aggregate['metadata']['cosine_layers']}")

    print("\n--- Pre-T vs Post-T Cosine Similarity (Clear Traces Only) ---")

    for dataset in sorted(aggregate["per_dataset"].keys()):
        layers = aggregate["per_dataset"][dataset]
        print(f"\n  {dataset}:")
        for layer_key in sorted(layers.keys()):
            d = layers[layer_key]
            print(f"    {layer_key}:")
            print(f"      Clear traces: {d['n_clear_traces']}")
            print(f"      Mean Cohen's d: {d['mean_cohens_d']:.3f} (median: {d['median_cohens_d']:.3f})")
            print(f"      Pre-T mean cosine: {d['mean_pre_mean']:.4f}")
            print(f"      Post-T mean cosine: {d['mean_post_mean']:.4f}")
            print(f"      Significant (p<0.05): {d['fraction_significant_p05']:.1%}")
            print(f"      Significant (p<0.01): {d['fraction_significant_p01']:.1%}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
