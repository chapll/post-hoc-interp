"""QA visualizations and parquet assembly for Phase 3 results.

Reads per-trace transition JSONs, assembles per-dataset parquet files and a
trace-summary parquet, then generates publication-quality QA visualizations:
AUROC curves, confidence curve examples, transition distributions, classification
bars, cosine similarity phase plots, and Cohen's d distribution.

Output:
  - $DATA_DIR/results/phase3/{dataset}_results.parquet (per-token long-format)
  - $DATA_DIR/results/phase3/trace_summary.parquet (per-trace summary)
  - $DATA_DIR/results/phase3/figures/*.png (6 QA plots)

Usage:
    python scripts/plot_phase3.py --config configs/phase3.yaml
"""

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.config import load_config
from src.utils import setup_logging

logger = setup_logging("plot_phase3")

load_dotenv()

# Publication-quality defaults
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# 1. Parquet Assembly
# ---------------------------------------------------------------------------

def assemble_parquet(
    transitions_dir: str,
    results_dir: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Assemble per-trace transition JSONs into parquet DataFrames.

    Creates two types of parquet files:
    1. Per-dataset long-format (one row per token-position per layer)
    2. Trace summary (one row per trace)

    Args:
        transitions_dir: Path to transitions/{dataset}/ directory tree.
        results_dir: Output path for parquet files.

    Returns:
        Tuple of (per_dataset_dfs, summary_df).
    """
    # Discover all transition JSON files
    all_traces = []
    for dataset_name in sorted(os.listdir(transitions_dir)):
        ds_dir = os.path.join(transitions_dir, dataset_name)
        if not os.path.isdir(ds_dir):
            continue
        for fname in sorted(os.listdir(ds_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(ds_dir, fname)
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                all_traces.append(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to read %s: %s", fpath, e)

    logger.info("Loaded %d transition result files", len(all_traces))

    if not all_traces:
        logger.warning("No transition results found")
        return {}, pd.DataFrame()

    # Build per-dataset long-format DataFrames
    per_dataset_rows = {}  # {dataset: [rows]}
    summary_rows = []

    for trace in all_traces:
        trace_id = trace.get("trace_id", "unknown")
        problem_id = trace.get("problem_id", "unknown")
        dataset = trace.get("dataset", "unknown")
        classification = trace.get("classification", "unknown")
        best_layer = trace.get("best_layer", -1)
        n_think = trace.get("n_think_tokens", 0)

        # Extract confidence curves
        conf_curve = trace.get("confidence_curve", {})
        positions = conf_curve.get("positions", [])

        # Extract detection results
        det_results = trace.get("detection_results", {})

        # Extract cosine similarity
        cosine_data = trace.get("cosine_similarity", {})

        # Build per-token rows for each layer
        for layer_key in conf_curve:
            if not layer_key.startswith("values_layer_"):
                continue
            layer_idx_str = layer_key.replace("values_layer_", "")
            try:
                layer_idx = int(layer_idx_str)
            except ValueError:
                continue

            values = conf_curve[layer_key]

            # Get cosine similarity for this layer
            cosine_layer_key = f"layer_{layer_idx}"
            cosine_info = cosine_data.get(cosine_layer_key, {})
            cosine_positions = cosine_info.get("positions", [])
            cosine_s_t = cosine_info.get("S_t", [])
            # Build position-to-S_t lookup
            cosine_lookup = dict(zip(cosine_positions, cosine_s_t))

            # Get transition point and segments from this layer's detection
            layer_det_key = f"layer_{layer_idx}"
            layer_det = det_results.get(layer_det_key, {})
            methods = layer_det.get("methods", {})

            # Get best transition point for segment assignment
            transition_point = None
            for method_name in ["sigmoid", "derivative", "binseg", "pelt"]:
                tp = methods.get(method_name, {}).get("transition_point")
                if tp is not None:
                    transition_point = tp
                    break

            # Determine segments from detection results
            segments = _extract_segments(methods, positions)

            if dataset not in per_dataset_rows:
                per_dataset_rows[dataset] = []

            for i, pos in enumerate(positions):
                probe_conf = values[i] if i < len(values) else None
                cos_sim = cosine_lookup.get(pos, None)
                seg_label = _assign_segment(pos, segments)

                per_dataset_rows[dataset].append({
                    "trace_id": trace_id,
                    "problem_id": problem_id,
                    "dataset": dataset,
                    "token_position": pos,
                    "layer": layer_idx,
                    "probe_confidence": probe_conf,
                    "cosine_similarity_S_t": cos_sim,
                    "transition_point": transition_point,
                    "transition_category": classification,
                    "segment_label": seg_label,
                })

        # Build summary row
        # Extract transition points from different methods at best layer
        best_det = det_results.get(f"layer_{best_layer}", {}).get("methods", {})
        tp_derivative = best_det.get("derivative", {}).get("transition_point")
        tp_sigmoid = best_det.get("sigmoid", {}).get("transition_point")
        tp_pelt = best_det.get("pelt", {}).get("transition_point")

        # Get cosine comparison stats at best layer
        best_cosine = cosine_data.get(f"layer_{best_layer}", {})
        comparison = best_cosine.get("pre_post_comparison", {})

        summary_rows.append({
            "trace_id": trace_id,
            "dataset": dataset,
            "problem_id": problem_id,
            "classification": classification,
            "transition_point_derivative": tp_derivative,
            "transition_point_sigmoid": tp_sigmoid,
            "transition_point_pelt_best": tp_pelt,
            "n_think_tokens": n_think,
            "pre_mean_cosine": comparison.get("pre_mean"),
            "post_mean_cosine": comparison.get("post_mean"),
            "cohens_d": comparison.get("cohens_d"),
            "p_value": comparison.get("p_value"),
        })

    # Convert to DataFrames and save
    per_dataset_dfs = {}
    for ds_name, rows in per_dataset_rows.items():
        df = pd.DataFrame(rows)
        parquet_path = os.path.join(results_dir, f"{ds_name}_results.parquet")
        df.to_parquet(parquet_path, index=False)
        per_dataset_dfs[ds_name] = df
        logger.info("Saved %s: %d rows, %d columns", parquet_path, len(df), len(df.columns))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, "trace_summary.parquet")
    summary_df.to_parquet(summary_path, index=False)
    logger.info("Saved trace summary: %d rows", len(summary_df))

    return per_dataset_dfs, summary_df


def _extract_segments(methods: dict, positions: list) -> list[tuple]:
    """Extract segment boundaries from detection method results.

    Tries sigmoid first, then derivative, then binseg, then pelt.

    Args:
        methods: Dict of method results.
        positions: List of token positions.

    Returns:
        List of (start, end, label) segment tuples.
    """
    for method_name in ["sigmoid", "derivative", "binseg", "pelt"]:
        mresult = methods.get(method_name, {})
        segments = mresult.get("segments", [])
        if segments and len(segments) > 1:
            return segments
    return []


def _assign_segment(pos: int, segments: list) -> str:
    """Assign a segment label to a token position.

    Args:
        pos: Token position.
        segments: List of [start, end, label] from detection results.

    Returns:
        Segment label string, or "unknown" if no segment matches.
    """
    for seg in segments:
        if len(seg) >= 3:
            start, end, label = seg[0], seg[1], seg[2]
            if start <= pos <= end:
                return label
    return "unknown"


# ---------------------------------------------------------------------------
# 2. QA Visualizations
# ---------------------------------------------------------------------------

def plot_auroc_curves(results_dir: str, figures_dir: str):
    """Plot AUROC vs layer for each dataset and pooled.

    Args:
        results_dir: Path containing probe_aurocs.json.
        figures_dir: Output directory for figures.
    """
    aurocs_path = os.path.join(results_dir, "probe_aurocs.json")
    if not os.path.isfile(aurocs_path):
        logger.warning("probe_aurocs.json not found, skipping AUROC plot")
        return

    with open(aurocs_path, "r") as f:
        auroc_data = json.load(f)

    # Determine how many subplots we need
    scopes = []
    per_dataset = auroc_data.get("per_dataset", {})
    for ds_name in sorted(per_dataset.keys()):
        scopes.append((ds_name, per_dataset[ds_name]))
    if "pooled" in auroc_data:
        scopes.append(("pooled", auroc_data["pooled"]))

    if not scopes:
        logger.warning("No AUROC data found, skipping plot")
        return

    n_plots = len(scopes)
    n_cols = min(n_plots, 3)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, (scope_name, scope_data) in enumerate(scopes):
        ax = axes[idx // n_cols][idx % n_cols]

        # Extract per-layer AUROC values
        layer_aurocs = scope_data.get("per_layer_auroc", {})
        if not layer_aurocs:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(scope_name)
            continue

        layers = sorted(int(k) for k in layer_aurocs.keys())
        aurocs = [layer_aurocs[str(l)] for l in layers]

        ax.plot(layers, aurocs, "o-", color="steelblue", markersize=4, linewidth=1.5)

        # Random baseline band (0.45 to 0.55)
        ax.axhspan(0.45, 0.55, alpha=0.15, color="grey", label="Random baseline")
        ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)

        ax.set_xlabel("Layer")
        ax.set_ylabel("AUROC")
        ax.set_title(scope_name)
        ax.set_ylim(0.35, 1.05)
        ax.legend(loc="lower right", fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Probe AUROC by Layer", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "auroc_per_layer.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved AUROC plot: %s", out_path)


def plot_confidence_curves(
    transitions_dir: str,
    figures_dir: str,
    n_examples: int = 9,
):
    """Plot per-trace confidence curve examples with detection boundaries.

    Selects 3 "clear", 3 "no-signal", and 3 "noisy-signal" example traces
    (or fewer if insufficient), and plots confidence vs position with overlaid
    detection method boundaries.

    Args:
        transitions_dir: Path to transitions directory.
        figures_dir: Output directory.
        n_examples: Target total examples (default 9 = 3 per category).
    """
    # Collect traces by classification
    traces_by_cls = {"clear": [], "no-signal": [], "noisy-signal": []}

    for dataset_name in sorted(os.listdir(transitions_dir)):
        ds_dir = os.path.join(transitions_dir, dataset_name)
        if not os.path.isdir(ds_dir):
            continue
        for fname in sorted(os.listdir(ds_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(ds_dir, fname)
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                cls = data.get("classification", "unknown")
                if cls in traces_by_cls:
                    traces_by_cls[cls].append(data)
            except (json.JSONDecodeError, IOError):
                continue

    # Select examples: 3 per category, spread across datasets
    selected = []
    per_cat = n_examples // 3
    for cls_name in ["clear", "no-signal", "noisy-signal"]:
        candidates = traces_by_cls[cls_name]
        # Try to get one per dataset
        by_dataset = {}
        for t in candidates:
            ds = t.get("dataset", "unknown")
            if ds not in by_dataset:
                by_dataset[ds] = []
            by_dataset[ds].append(t)

        chosen = []
        for ds in sorted(by_dataset.keys()):
            if len(chosen) >= per_cat:
                break
            chosen.append(by_dataset[ds][0])
        # Fill remaining from any dataset
        for t in candidates:
            if len(chosen) >= per_cat:
                break
            if t not in chosen:
                chosen.append(t)
        selected.extend(chosen)

    if not selected:
        logger.warning("No traces found for confidence curve examples")
        return

    n_plots = len(selected)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    method_colors = {
        "derivative": ("blue", "--"),
        "pelt": ("green", "--"),
        "binseg": ("orange", "--"),
        "sigmoid": ("red", "--"),
    }

    segment_colors = {
        "reasoning": (0.7, 0.85, 1.0, 0.3),      # light blue
        "transition": (1.0, 1.0, 0.7, 0.3),       # light yellow
        "post-hoc": (1.0, 0.7, 0.7, 0.3),         # light red
    }

    for idx, trace in enumerate(selected):
        ax = axes[idx // n_cols][idx % n_cols]

        best_layer = trace.get("best_layer", -1)
        conf_curve = trace.get("confidence_curve", {})
        positions = conf_curve.get("positions", [])
        values_key = f"values_layer_{best_layer}"
        values = conf_curve.get(values_key, [])

        if not positions or not values:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        # Plot confidence curve
        ax.plot(positions, values, "k-", linewidth=1.0, alpha=0.8, label="Probe confidence")

        # Get detection results for best layer
        det_key = f"layer_{best_layer}"
        det_data = trace.get("detection_results", {}).get(det_key, {})
        methods = det_data.get("methods", {})

        # Overlay transition points
        for method_name, (color, linestyle) in method_colors.items():
            mresult = methods.get(method_name, {})
            tp = mresult.get("transition_point")
            if tp is not None:
                ax.axvline(tp, color=color, linestyle=linestyle, linewidth=1.2, alpha=0.7,
                           label=f"{method_name} T={tp}")

            # For sigmoid, overlay fitted curve
            if method_name == "sigmoid" and "method_specific" in mresult:
                ms = mresult["method_specific"]
                fitted = ms.get("fitted_values")
                if fitted and len(fitted) == len(positions):
                    ax.plot(positions, fitted, color="red", linewidth=0.8, alpha=0.5)

        # Shade segments
        for method_name in ["sigmoid", "derivative", "binseg", "pelt"]:
            mresult = methods.get(method_name, {})
            segments = mresult.get("segments", [])
            if segments and len(segments) > 1:
                for seg in segments:
                    if len(seg) >= 3:
                        start, end, label = seg[0], seg[1], seg[2]
                        color = segment_colors.get(label, (0.9, 0.9, 0.9, 0.2))
                        ax.axvspan(start, end, color=color)
                break  # Only shade from first method with segments

        trace_id = trace.get("trace_id", "unknown")
        classification = trace.get("classification", "unknown")
        ax.set_title(f"{trace_id}\n[{classification}]", fontsize=9)
        ax.set_xlabel("Token position")
        ax.set_ylabel("Probe confidence")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(True, alpha=0.2)

    # Hide unused subplots
    for idx in range(n_plots, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Confidence Curve Examples with Transition Detection", fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "confidence_curve_examples.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved confidence curve examples: %s", out_path)


def plot_transition_distribution(
    transitions_dir: str,
    figures_dir: str,
):
    """Plot histogram of normalized transition positions for clear traces.

    One plot per detection method, histograms per dataset overlaid.

    Args:
        transitions_dir: Path to transitions directory.
        figures_dir: Output directory.
    """
    # Collect normalized transition positions
    # {method: {dataset: [normalized_positions]}}
    data_by_method = {}

    for dataset_name in sorted(os.listdir(transitions_dir)):
        ds_dir = os.path.join(transitions_dir, dataset_name)
        if not os.path.isdir(ds_dir):
            continue
        for fname in sorted(os.listdir(ds_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(ds_dir, fname)
            try:
                with open(fpath, "r") as f:
                    trace = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            if trace.get("classification") != "clear":
                continue

            best_layer = trace.get("best_layer", -1)
            n_think = trace.get("n_think_tokens", 0)
            if n_think <= 0:
                continue

            positions = trace.get("confidence_curve", {}).get("positions", [])
            if not positions:
                continue

            min_pos = min(positions)
            max_pos = max(positions)
            pos_range = max_pos - min_pos
            if pos_range <= 0:
                continue

            det_key = f"layer_{best_layer}"
            methods = trace.get("detection_results", {}).get(det_key, {}).get("methods", {})

            for method_name, mresult in methods.items():
                tp = mresult.get("transition_point")
                if tp is None:
                    continue
                normalized = (tp - min_pos) / pos_range

                if method_name not in data_by_method:
                    data_by_method[method_name] = {}
                if dataset_name not in data_by_method[method_name]:
                    data_by_method[method_name][dataset_name] = []
                data_by_method[method_name][dataset_name].append(normalized)

    if not data_by_method:
        logger.warning("No clear traces found for transition distribution")
        return

    methods = sorted(data_by_method.keys())
    n_methods = len(methods)
    n_cols = min(2, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    dataset_colors = {
        "gsm_symbolic": "steelblue",
        "folio": "coral",
        "strategyqa": "mediumseagreen",
    }
    fallback_colors = ["steelblue", "coral", "mediumseagreen", "gold", "purple"]

    for idx, method_name in enumerate(methods):
        ax = axes[idx // n_cols][idx % n_cols]
        datasets = data_by_method[method_name]

        for ds_idx, (ds_name, values) in enumerate(sorted(datasets.items())):
            color = dataset_colors.get(ds_name, fallback_colors[ds_idx % len(fallback_colors)])
            ax.hist(values, bins=20, range=(0, 1), alpha=0.5, color=color,
                    label=f"{ds_name} (n={len(values)})", edgecolor="white", linewidth=0.5)

        ax.set_xlabel("Normalized transition position (T_i / trace_length)")
        ax.set_ylabel("Count")
        ax.set_title(f"Method: {method_name}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    for idx in range(n_methods, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Transition Point Distribution (Clear Traces)", fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "transition_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved transition distribution: %s", out_path)


def plot_classification_distribution(
    transitions_dir: str,
    figures_dir: str,
):
    """Plot stacked bar chart of classification distribution per dataset.

    Args:
        transitions_dir: Path to transitions directory.
        figures_dir: Output directory.
    """
    counts = {}  # {dataset: {cls: count}}

    for dataset_name in sorted(os.listdir(transitions_dir)):
        ds_dir = os.path.join(transitions_dir, dataset_name)
        if not os.path.isdir(ds_dir):
            continue
        for fname in sorted(os.listdir(ds_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(ds_dir, fname)
            try:
                with open(fpath, "r") as f:
                    trace = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            ds = trace.get("dataset", dataset_name)
            cls = trace.get("classification", "unknown")

            if ds not in counts:
                counts[ds] = {}
            counts[ds][cls] = counts[ds].get(cls, 0) + 1

    if not counts:
        logger.warning("No data for classification distribution")
        return

    datasets = sorted(counts.keys())
    categories = ["clear", "noisy-signal", "no-signal"]
    cat_colors = {
        "clear": "steelblue",
        "noisy-signal": "gold",
        "no-signal": "lightgrey",
    }

    fig, ax = plt.subplots(figsize=(max(4, len(datasets) * 2), 5))

    x = np.arange(len(datasets))
    bar_width = 0.6

    bottoms = np.zeros(len(datasets))
    for cat in categories:
        fractions = []
        for ds in datasets:
            total = sum(counts[ds].values())
            frac = counts[ds].get(cat, 0) / total if total > 0 else 0
            fractions.append(frac)
        fracs = np.array(fractions)
        ax.bar(x, fracs, bar_width, bottom=bottoms, color=cat_colors.get(cat, "grey"),
               label=cat, edgecolor="white", linewidth=0.5)
        bottoms += fracs

    ax.set_xlabel("Dataset")
    ax.set_ylabel("Fraction")
    ax.set_title("Trace Classification Distribution")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.2)

    # Add count labels on bars
    for i, ds in enumerate(datasets):
        total = sum(counts[ds].values())
        ax.text(i, 1.01, f"n={total}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    out_path = os.path.join(figures_dir, "classification_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved classification distribution: %s", out_path)


def plot_cosine_similarity(
    transitions_dir: str,
    figures_dir: str,
    n_examples: int = 4,
):
    """Plot cosine similarity S_t vs token position for clear trace examples.

    Shows S_t at the best layer with vertical line at T_i and shaded
    pre-T (light blue) and post-T (light red) regions.

    Args:
        transitions_dir: Path to transitions directory.
        figures_dir: Output directory.
        n_examples: Number of example traces to plot.
    """
    # Find clear traces with cosine data
    clear_traces = []

    for dataset_name in sorted(os.listdir(transitions_dir)):
        ds_dir = os.path.join(transitions_dir, dataset_name)
        if not os.path.isdir(ds_dir):
            continue
        for fname in sorted(os.listdir(ds_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(ds_dir, fname)
            try:
                with open(fpath, "r") as f:
                    trace = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            if trace.get("classification") != "clear":
                continue
            if "cosine_similarity" not in trace:
                continue

            clear_traces.append(trace)

    if not clear_traces:
        logger.warning("No clear traces with cosine data found")
        return

    # Select examples spread across datasets
    selected = []
    by_dataset = {}
    for t in clear_traces:
        ds = t.get("dataset", "unknown")
        if ds not in by_dataset:
            by_dataset[ds] = []
        by_dataset[ds].append(t)

    for ds in sorted(by_dataset.keys()):
        if len(selected) >= n_examples:
            break
        selected.append(by_dataset[ds][0])

    while len(selected) < n_examples and len(selected) < len(clear_traces):
        for t in clear_traces:
            if t not in selected:
                selected.append(t)
                break
        else:
            break

    n_plots = len(selected)
    n_cols = min(2, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for idx, trace in enumerate(selected):
        ax = axes[idx // n_cols][idx % n_cols]

        best_layer = trace.get("best_layer", -1)
        layer_key = f"layer_{best_layer}"
        cosine_data = trace.get("cosine_similarity", {}).get(layer_key, {})
        positions = cosine_data.get("positions", [])
        s_t = cosine_data.get("S_t", [])

        if not positions or not s_t:
            ax.text(0.5, 0.5, "No cosine data", ha="center", va="center", transform=ax.transAxes)
            continue

        ax.plot(positions, s_t, "k-", linewidth=0.8, alpha=0.8)

        # Get transition point
        det_key = f"layer_{best_layer}"
        methods = trace.get("detection_results", {}).get(det_key, {}).get("methods", {})
        tp = None
        for method_name in ["sigmoid", "derivative", "binseg", "pelt"]:
            tp = methods.get(method_name, {}).get("transition_point")
            if tp is not None:
                break

        if tp is not None:
            ax.axvline(tp, color="red", linestyle="--", linewidth=1.5, label=f"T={tp}")
            # Shade pre-T and post-T
            ax.axvspan(min(positions), tp, color=(0.7, 0.85, 1.0, 0.3))
            ax.axvspan(tp, max(positions), color=(1.0, 0.7, 0.7, 0.3))

        # Add comparison stats if available
        comparison = cosine_data.get("pre_post_comparison", {})
        if comparison:
            d = comparison.get("cohens_d", 0)
            p = comparison.get("p_value", 1)
            ax.text(0.02, 0.02, f"Cohen's d={d:.2f}\np={p:.4f}",
                    transform=ax.transAxes, fontsize=7, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        trace_id = trace.get("trace_id", "unknown")
        ax.set_title(f"{trace_id} (layer {best_layer})", fontsize=9)
        ax.set_xlabel("Token position")
        ax.set_ylabel("Cosine similarity S_t")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    for idx in range(n_plots, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Cosine Similarity Dynamics (Clear Traces)", fontsize=13, y=1.02)
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "cosine_similarity_examples.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved cosine similarity examples: %s", out_path)


def plot_cohens_d(
    transitions_dir: str,
    figures_dir: str,
):
    """Plot histogram of Cohen's d values across clear traces per dataset.

    Args:
        transitions_dir: Path to transitions directory.
        figures_dir: Output directory.
    """
    # Collect Cohen's d values
    d_values = {}  # {dataset: [cohens_d]}

    for dataset_name in sorted(os.listdir(transitions_dir)):
        ds_dir = os.path.join(transitions_dir, dataset_name)
        if not os.path.isdir(ds_dir):
            continue
        for fname in sorted(os.listdir(ds_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(ds_dir, fname)
            try:
                with open(fpath, "r") as f:
                    trace = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            if trace.get("classification") != "clear":
                continue

            # Get Cohen's d from best layer cosine comparison
            best_layer = trace.get("best_layer", -1)
            layer_key = f"layer_{best_layer}"
            comparison = trace.get("cosine_similarity", {}).get(layer_key, {}).get(
                "pre_post_comparison", {}
            )
            d = comparison.get("cohens_d")
            if d is not None:
                ds = trace.get("dataset", dataset_name)
                if ds not in d_values:
                    d_values[ds] = []
                d_values[ds].append(d)

    if not d_values:
        logger.warning("No Cohen's d data found for histogram")
        return

    dataset_colors = {
        "gsm_symbolic": "steelblue",
        "folio": "coral",
        "strategyqa": "mediumseagreen",
    }
    fallback_colors = ["steelblue", "coral", "mediumseagreen", "gold", "purple"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for ds_idx, (ds_name, values) in enumerate(sorted(d_values.items())):
        color = dataset_colors.get(ds_name, fallback_colors[ds_idx % len(fallback_colors)])
        ax.hist(values, bins=25, alpha=0.5, color=color,
                label=f"{ds_name} (n={len(values)}, median={np.median(values):.2f})",
                edgecolor="white", linewidth=0.5)

    # Reference lines
    ax.axvline(0, color="black", linestyle=":", linewidth=1, label="d=0 (no effect)")
    ax.axvline(0.8, color="darkred", linestyle="--", linewidth=1, label="d=0.8 (large effect)")

    ax.set_xlabel("Cohen's d (post-T - pre-T cosine similarity)")
    ax.set_ylabel("Count")
    ax.set_title("Effect Size Distribution: Pre-T vs Post-T Cosine Similarity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    out_path = os.path.join(figures_dir, "cohens_d_distribution.png")
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved Cohen's d distribution: %s", out_path)


# ---------------------------------------------------------------------------
# 3. Summary Report
# ---------------------------------------------------------------------------

def print_summary_report(
    results_dir: str,
    summary_df: pd.DataFrame,
    per_dataset_dfs: dict[str, pd.DataFrame],
):
    """Print comprehensive summary report to stdout.

    Args:
        results_dir: Path to results directory (for loading stats files).
        summary_df: Trace summary DataFrame.
        per_dataset_dfs: Dict of per-dataset DataFrames.
    """
    print("\n" + "=" * 70)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 70)

    if summary_df.empty:
        print("\nNo results to summarize.")
        return

    # Total traces per dataset
    print("\n--- Traces Per Dataset ---")
    for ds in sorted(summary_df["dataset"].unique()):
        count = len(summary_df[summary_df["dataset"] == ds])
        print(f"  {ds}: {count}")
    print(f"  TOTAL: {len(summary_df)}")

    # Classification distribution
    print("\n--- Classification Distribution ---")
    for ds in sorted(summary_df["dataset"].unique()):
        ds_df = summary_df[summary_df["dataset"] == ds]
        total = len(ds_df)
        print(f"\n  {ds} (n={total}):")
        for cls_name in ["clear", "noisy-signal", "no-signal"]:
            count = len(ds_df[ds_df["classification"] == cls_name])
            frac = count / total if total > 0 else 0
            print(f"    {cls_name:15s}: {count:4d} ({frac:.1%})")
        indet = len(ds_df[ds_df["classification"] != "clear"])
        indet_frac = indet / total if total > 0 else 0
        print(f"    {'INDETERMINATE':15s}: {indet:4d} ({indet_frac:.1%})")

    # Overall indeterminate rate
    total = len(summary_df)
    total_clear = len(summary_df[summary_df["classification"] == "clear"])
    total_indet = total - total_clear
    indet_rate = total_indet / total if total > 0 else 0
    print(f"\n  >>> OVERALL INDETERMINATE RATE: {total_indet}/{total} ({indet_rate:.1%}) <<<")

    # Best probe layer per dataset
    print("\n--- Best Probe Layer / AUROC ---")
    aurocs_path = os.path.join(results_dir, "probe_aurocs.json")
    if os.path.isfile(aurocs_path):
        with open(aurocs_path, "r") as f:
            auroc_data = json.load(f)
        best = auroc_data.get("best_layer", "unknown")
        print(f"  Best layer (pooled): {best}")
        per_ds = auroc_data.get("per_dataset", {})
        for ds_name, ds_data in sorted(per_ds.items()):
            ds_best = ds_data.get("best_layer", "unknown")
            ds_auroc = ds_data.get("best_auroc", "unknown")
            print(f"  {ds_name}: layer {ds_best}, AUROC={ds_auroc}")
    else:
        print("  probe_aurocs.json not found")

    # Cosine similarity stats
    print("\n--- Cosine Similarity (Clear Traces) ---")
    clear_df = summary_df[summary_df["classification"] == "clear"]
    for ds in sorted(clear_df["dataset"].unique()):
        ds_clear = clear_df[clear_df["dataset"] == ds]
        d_vals = ds_clear["cohens_d"].dropna()
        p_vals = ds_clear["p_value"].dropna()
        if len(d_vals) > 0:
            print(f"\n  {ds} (n={len(d_vals)} clear traces):")
            print(f"    Median Cohen's d: {d_vals.median():.3f}")
            print(f"    Mean Cohen's d: {d_vals.mean():.3f}")
            if len(p_vals) > 0:
                sig_05 = (p_vals < 0.05).mean()
                print(f"    Fraction significant (p<0.05): {sig_05:.1%}")

    # Parquet info
    print("\n--- Output Files ---")
    for ds_name, df in sorted(per_dataset_dfs.items()):
        print(f"  {ds_name}_results.parquet: {len(df)} rows, {len(df.columns)} columns")
    print(f"  trace_summary.parquet: {len(summary_df)} rows")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------

def main():
    """Main entry point for QA visualization and parquet assembly."""
    cfg = load_config()

    # Resolve paths
    data_dir = os.environ.get("DATA_DIR", cfg.paths.get("data_dir", "data"))
    results_dir = os.path.join(data_dir, "results", "phase3")
    transitions_dir = os.path.join(results_dir, "transitions")
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    if not os.path.isdir(transitions_dir):
        raise FileNotFoundError(
            f"Transitions directory not found: {transitions_dir}. "
            "Run scripts/detect_transitions.py first."
        )

    # Step 1: Assemble parquet files
    logger.info("Assembling parquet files...")
    per_dataset_dfs, summary_df = assemble_parquet(transitions_dir, results_dir)

    # Step 2: Generate QA visualizations
    logger.info("Generating QA visualizations...")

    plot_auroc_curves(results_dir, figures_dir)
    plot_confidence_curves(transitions_dir, figures_dir)
    plot_transition_distribution(transitions_dir, figures_dir)
    plot_classification_distribution(transitions_dir, figures_dir)
    plot_cosine_similarity(transitions_dir, figures_dir)
    plot_cohens_d(transitions_dir, figures_dir)

    # Step 3: Print summary report
    print_summary_report(results_dir, summary_df, per_dataset_dfs)

    logger.info("Phase 3 QA visualization and assembly complete")


if __name__ == "__main__":
    main()
