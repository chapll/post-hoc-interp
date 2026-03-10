"""Visualization and metrics for the cross-seed transition consistency experiment.

Reads the cross-seed probe results parquet (from cross_seed_experiment.py) and
generates per-problem two-panel plots (accuracy + probability with faint seed
curves and bold mean), cross-seed agreement metrics (sharpness, position-wise
variance, pairwise Spearman), and summary plots (sharpness histogram, agreement
scatter).

Usage:
    python scripts/plot_cross_seed.py
    python scripts/plot_cross_seed.py --results-dir /path/to/force_answer_seeds
"""

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Style configuration (matching plot_force_answer.py)
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    "figure.figsize": (14, 5),
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 200,
})

# Colors (per discretion recommendation)
ACC_COLOR = "#0077BB"   # blue for accuracy
PROB_COLOR = "#EE7733"  # orange for probability

DATASET_COLORS = {
    "gsm_symbolic": "#0077BB",
    "folio": "#EE7733",
    "strategyqa": "#009988",
    "deepmath": "#CC3311",
}

DATASET_LABELS = {
    "gsm_symbolic": "GSM-Symbolic",
    "folio": "FOLIO",
    "strategyqa": "StrategyQA",
    "deepmath": "DeepMath",
}

# Total seeds per problem (for display in plot titles)
TOTAL_SEEDS = 20


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def interpolate_seed_curves(df_problem, grid, metric="correct"):
    """Interpolate a metric across seeds to a common grid for a single problem.

    Args:
        df_problem: DataFrame filtered to one problem_id.
        grid: np.array of normalized positions (e.g., np.linspace(0, 1, 21)).
        metric: Column name to interpolate ("correct" or "correct_token_prob").

    Returns:
        Tuple of (matrix, seed_indices) where matrix has shape
        (n_seeds, len(grid)) and seed_indices is a list of seed_index values.
    """
    curves = []
    seed_indices = []

    for seed_idx in sorted(df_problem["seed_index"].unique()):
        seed_df = df_problem[df_problem["seed_index"] == seed_idx].sort_values(
            "normalized_position"
        )
        x = seed_df["normalized_position"].values
        y = seed_df[metric].values
        if metric == "correct":
            y = y.astype(float)
        interp_y = np.interp(grid, x, y)
        curves.append(interp_y)
        seed_indices.append(seed_idx)

    return np.array(curves), seed_indices


# ---------------------------------------------------------------------------
# Cross-seed agreement metrics
# ---------------------------------------------------------------------------

def compute_agreement_metrics(acc_matrix, grid):
    """Compute the three cross-seed agreement metrics from an accuracy matrix.

    Args:
        acc_matrix: np.array of shape (n_seeds, n_positions) with interpolated
            binary accuracy values.
        grid: np.array of normalized positions corresponding to acc_matrix
            columns.

    Returns:
        Dict with keys: sharpness, mean_pos_variance, max_pos_variance,
        mean_pairwise_spearman, n_seeds.
    """
    # 1. Sharpness: max slope of mean accuracy curve
    mean_acc = acc_matrix.mean(axis=0)
    diffs = np.diff(mean_acc) / np.diff(grid)
    sharpness = float(np.max(np.abs(diffs)))

    # 2. Position-wise variance
    pos_variance = acc_matrix.var(axis=0)
    mean_pos_variance = float(pos_variance.mean())
    max_pos_variance = float(pos_variance.max())

    # 3. Pairwise Spearman correlation
    n_seeds = acc_matrix.shape[0]
    correlations = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            rho, _ = spearmanr(acc_matrix[i], acc_matrix[j])
            if not np.isnan(rho):
                correlations.append(rho)
    mean_spearman = float(np.mean(correlations)) if correlations else float("nan")

    return {
        "sharpness": sharpness,
        "mean_pos_variance": mean_pos_variance,
        "max_pos_variance": max_pos_variance,
        "mean_pairwise_spearman": mean_spearman,
        "n_seeds": n_seeds,
    }


# ---------------------------------------------------------------------------
# Per-problem two-panel plot
# ---------------------------------------------------------------------------

def plot_problem_seeds(problem_id, df_problem, grid, output_dir):
    """Create the two-panel per-problem plot.

    Left panel: binary accuracy with faint individual seed curves and bold mean.
    Right panel: P(correct token) with faint individual seed curves and bold mean.

    Args:
        problem_id: String identifier for the problem.
        df_problem: DataFrame filtered to this problem_id.
        grid: np.array of normalized positions for interpolation.
        output_dir: Base output directory (per_problem/ subdirectory used).
    """
    fig, (ax_acc, ax_prob) = plt.subplots(1, 2, figsize=(14, 5))

    # Interpolate curves
    acc_matrix, seed_ids = interpolate_seed_curves(df_problem, grid, "correct")
    prob_matrix, _ = interpolate_seed_curves(df_problem, grid, "correct_token_prob")
    n_seeds = acc_matrix.shape[0]

    # Left panel: accuracy
    for i in range(n_seeds):
        ax_acc.plot(grid, acc_matrix[i], color=ACC_COLOR, alpha=0.15, linewidth=0.8)
    ax_acc.plot(
        grid, acc_matrix.mean(axis=0),
        color=ACC_COLOR, linewidth=2.5, label=f"Mean (n={n_seeds})",
    )
    ax_acc.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax_acc.set_xlabel("Normalized Position")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Binary Accuracy")
    ax_acc.set_xlim(0, 1)
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="lower right")

    # Right panel: probability
    for i in range(n_seeds):
        ax_prob.plot(grid, prob_matrix[i], color=PROB_COLOR, alpha=0.15, linewidth=0.8)
    ax_prob.plot(
        grid, prob_matrix.mean(axis=0),
        color=PROB_COLOR, linewidth=2.5, label=f"Mean (n={n_seeds})",
    )
    ax_prob.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax_prob.set_xlabel("Normalized Position")
    ax_prob.set_ylabel("Probability")
    ax_prob.set_title("P(correct token)")
    ax_prob.set_xlim(0, 1)
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.grid(True, alpha=0.3)
    ax_prob.legend(loc="lower right")

    fig.suptitle(
        f"{problem_id} ({n_seeds}/{TOTAL_SEEDS} seeds)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    per_problem_dir = os.path.join(output_dir, "per_problem")
    os.makedirs(per_problem_dir, exist_ok=True)
    out_path = os.path.join(per_problem_dir, f"{problem_id}.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary plot: sharpness histogram
# ---------------------------------------------------------------------------

def plot_sharpness_histogram(metrics_df, output_dir):
    """Histogram of sharpness values across problems.

    Args:
        metrics_df: DataFrame with per-problem metrics (must have 'sharpness').
        output_dir: Directory where sharpness_histogram.png will be saved.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        metrics_df["sharpness"], bins=15,
        color=ACC_COLOR, edgecolor="white", alpha=0.8,
    )
    median_val = metrics_df["sharpness"].median()
    ax.axvline(
        median_val, color="red", linestyle="--",
        label=f"Median: {median_val:.2f}",
    )
    ax.set_xlabel("Max Slope of Mean Accuracy Curve")
    ax.set_ylabel("Number of Problems")
    ax.set_title("Transition Sharpness Distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = os.path.join(output_dir, "sharpness_histogram.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary plot: agreement scatter
# ---------------------------------------------------------------------------

def plot_agreement_scatter(metrics_df, output_dir):
    """Scatter of mean pairwise Spearman (x) vs mean position-wise variance (y).

    Each point is a problem, colored by dataset.

    Args:
        metrics_df: DataFrame with per-problem metrics (must have
            'mean_pairwise_spearman', 'mean_pos_variance', 'dataset').
        output_dir: Directory where agreement_scatter.png will be saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    plotted_datasets = set()
    for _, row in metrics_df.iterrows():
        ds = row["dataset"]
        color = DATASET_COLORS.get(ds, "gray")
        label = DATASET_LABELS.get(ds, ds) if ds not in plotted_datasets else None
        ax.scatter(
            row["mean_pairwise_spearman"],
            row["mean_pos_variance"],
            color=color, s=50, alpha=0.7, edgecolors="white", linewidth=0.5,
            label=label,
        )
        plotted_datasets.add(ds)

    ax.set_xlabel("Mean Pairwise Spearman Correlation")
    ax.set_ylabel("Mean Position-wise Variance")
    ax.set_title("Cross-Seed Agreement")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_path = os.path.join(output_dir, "agreement_scatter.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(metrics_df, df):
    """Print summary statistics to stdout.

    Args:
        metrics_df: DataFrame with per-problem agreement metrics.
        df: Full results DataFrame from results.parquet.
    """
    print("=" * 60)
    print("CROSS-SEED TRANSITION CONSISTENCY SUMMARY")
    print("=" * 60)
    print()

    n_problems = len(metrics_df)
    mean_seeds = metrics_df["n_seeds"].mean()
    print(f"Total problems: {n_problems}")
    print(f"Mean seeds per problem: {mean_seeds:.1f}")

    # Problems with fewer than min_valid_seeds
    low_seed = metrics_df[metrics_df["n_seeds"] < 10]
    if len(low_seed) > 0:
        print(f"Problems with < 10 valid seeds: {len(low_seed)}")
        for _, row in low_seed.iterrows():
            print(f"  {row['problem_id']}: {row['n_seeds']} seeds")
    else:
        print("Problems with < 10 valid seeds: 0")
    print()

    # Aggregate metrics
    print("Aggregate metrics (across problems):")
    print("-" * 50)
    for metric_name in ["sharpness", "mean_pos_variance", "mean_pairwise_spearman"]:
        vals = metrics_df[metric_name].dropna()
        if len(vals) > 0:
            print(
                f"  {metric_name}: "
                f"mean={vals.mean():.2f}, "
                f"median={vals.median():.2f}, "
                f"std={vals.std():.2f}"
            )
        else:
            print(f"  {metric_name}: no data")
    print()

    # Classification by behavior
    # For each problem, check mean accuracy curve at all grid positions
    grid = np.linspace(0, 1, 21)
    always_correct = 0
    always_incorrect = 0
    transitioning = 0

    for _, row in metrics_df.iterrows():
        pid = row["problem_id"]
        df_p = df[df["trace_id"] == pid]
        if df_p.empty:
            continue
        acc_matrix, _ = interpolate_seed_curves(df_p, grid, "correct")
        mean_acc = acc_matrix.mean(axis=0)

        if (mean_acc > 0.9).all():
            always_correct += 1
        elif (mean_acc < 0.1).all():
            always_incorrect += 1
        else:
            transitioning += 1

    print("Classification by behavior:")
    print(f"  Always correct (mean accuracy > 0.9 at all positions): {always_correct}")
    print(f"  Always incorrect (mean accuracy < 0.1 at all positions): {always_incorrect}")
    print(f"  Transitioning (neither): {transitioning}")
    print()

    # Overall probe statistics
    total_probes = len(df)
    correct_rate = df["correct"].mean() * 100
    no_answer_rate = df["no_answer"].mean() * 100
    print("Overall probe statistics:")
    print(f"  Total probes: {total_probes}")
    print(f"  Correct rate: {correct_rate:.1f}%")
    print(f"  No-answer rate: {no_answer_rate:.1f}%")
    print()

    print("=" * 60)


# ---------------------------------------------------------------------------
# Dataset extraction helper
# ---------------------------------------------------------------------------

def _extract_dataset(problem_id):
    """Extract dataset name from a problem_id prefix."""
    if problem_id.startswith("gsm_symbolic"):
        return "gsm_symbolic"
    elif problem_id.startswith("folio"):
        return "folio"
    elif problem_id.startswith("strategyqa"):
        return "strategyqa"
    return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse arguments, load results, compute metrics, generate all plots."""
    parser = argparse.ArgumentParser(
        description="Visualize cross-seed transition consistency results.",
    )
    default_dir = os.path.join(
        os.environ.get("EXP_DIR", "experiments"), "force_answer_seeds",
    )
    parser.add_argument(
        "--results-dir",
        default=default_dir,
        help=f"Directory containing results.parquet (default: {default_dir})",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    parquet_path = os.path.join(results_dir, "results.parquet")

    if not os.path.isfile(parquet_path):
        print(
            f"ERROR: results.parquet not found at {parquet_path}",
            file=sys.stderr,
        )
        print("Run the experiment first:", file=sys.stderr)
        print(
            "  python scripts/cross_seed_experiment.py --config configs/cross_seed.yaml",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}")
    print(f"Columns: {list(df.columns)}")
    print()

    # Define interpolation grid
    grid = np.linspace(0, 1, 21)

    # Get unique problem_ids
    problem_ids = sorted(df["trace_id"].unique())
    print(f"Problems: {len(problem_ids)}")
    print()

    # Create output directories
    per_problem_dir = os.path.join(results_dir, "per_problem")
    os.makedirs(per_problem_dir, exist_ok=True)

    # Process each problem
    metrics_list = []
    for pid in problem_ids:
        df_problem = df[df["trace_id"] == pid]

        # Compute interpolated accuracy matrix and agreement metrics
        acc_matrix, _ = interpolate_seed_curves(df_problem, grid, "correct")
        metrics = compute_agreement_metrics(acc_matrix, grid)
        metrics["problem_id"] = pid
        metrics["dataset"] = _extract_dataset(pid)
        metrics_list.append(metrics)

        # Plot per-problem PNG
        plot_problem_seeds(pid, df_problem, grid, results_dir)

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df[
        ["problem_id", "dataset", "sharpness", "mean_pos_variance",
         "max_pos_variance", "mean_pairwise_spearman", "n_seeds"]
    ]

    # Save metrics parquet
    metrics_path = os.path.join(results_dir, "metrics.parquet")
    metrics_df.to_parquet(metrics_path, index=False)
    print(f"Saved: {metrics_path}")

    # Summary plots
    plot_sharpness_histogram(metrics_df, results_dir)
    plot_agreement_scatter(metrics_df, results_dir)

    # Print summary statistics
    print_summary(metrics_df, df)

    print(f"Output saved to {results_dir}")


if __name__ == "__main__":
    main()
