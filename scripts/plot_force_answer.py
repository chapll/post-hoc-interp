"""Visualization and analysis for the force-answer transition experiment.

Reads the experiment results parquet (from force_answer_experiment.py) and
generates aggregate/per-dataset transition curves, correct-vs-incorrect
trace comparisons, and summary statistics. All plots are saved as PNG files.

Usage:
    python scripts/plot_force_answer.py
    python scripts/plot_force_answer.py --results-dir /path/to/force_answer
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Style configuration
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

# Colorblind-friendly palette
DATASET_COLORS = {
    "gsm_symbolic": "#0077BB",  # blue
    "folio": "#EE7733",         # orange
    "strategyqa": "#009988",    # teal
}

DATASET_LABELS = {
    "gsm_symbolic": "GSM-Symbolic",
    "folio": "FOLIO",
    "strategyqa": "StrategyQA",
}

N_POSITION_BINS = 25


# ---------------------------------------------------------------------------
# Binning helper
# ---------------------------------------------------------------------------

def _bin_positions(df, n_bins=N_POSITION_BINS):
    """Add a position_bin column with evenly spaced bins across [0, 1].

    Returns the modified dataframe and the bin centers for plotting.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    df = df.copy()
    df["position_bin"] = pd.cut(
        df["normalized_position"],
        bins=bin_edges,
        labels=bin_centers,
        include_lowest=True,
    )
    df["position_bin"] = df["position_bin"].astype(float)
    return df, bin_centers


# ---------------------------------------------------------------------------
# Plot 1: Transition curves (accuracy + probability, side by side)
# ---------------------------------------------------------------------------

def plot_transition_curves(df, output_dir):
    """Generate the primary side-by-side figure: accuracy (left) and probability (right).

    Both panels show all datasets overlaid with distinct colors (satisfies
    the locked decision: all 3 datasets on one plot, color-coded). An
    aggregate dashed black line is also shown on each panel.

    Args:
        df: Results dataframe with columns normalized_position, correct,
            correct_token_prob, dataset.
        output_dir: Directory where transition_curves.png will be saved.
    """
    df_binned, bin_centers = _bin_positions(df)

    fig, (ax_acc, ax_prob) = plt.subplots(1, 2, figsize=(16, 6))

    # -- Left panel: Accuracy vs Position --
    # Aggregate
    agg = df_binned.groupby("position_bin")["correct"].agg(["mean", "count"])
    p = agg["mean"].values
    n = agg["count"].values
    ci = 1.96 * np.sqrt(p * (1 - p) / np.maximum(n, 1))
    positions = agg.index.values

    ax_acc.plot(positions, p, color="black", linestyle="--", linewidth=2,
                label="Aggregate", zorder=5)
    ax_acc.fill_between(positions, p - ci, p + ci, color="black", alpha=0.08)

    # Per-dataset
    for ds in sorted(DATASET_COLORS.keys()):
        ds_df = df_binned[df_binned["dataset"] == ds]
        if ds_df.empty:
            continue
        grp = ds_df.groupby("position_bin")["correct"].agg(["mean", "count"])
        p_ds = grp["mean"].values
        n_ds = grp["count"].values
        ci_ds = 1.96 * np.sqrt(p_ds * (1 - p_ds) / np.maximum(n_ds, 1))
        pos_ds = grp.index.values

        ax_acc.plot(pos_ds, p_ds, color=DATASET_COLORS[ds], linewidth=1.8,
                    label=DATASET_LABELS[ds])
        ax_acc.fill_between(pos_ds, p_ds - ci_ds, p_ds + ci_ds,
                            color=DATASET_COLORS[ds], alpha=0.15)

    ax_acc.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_acc.set_xlabel("Normalized Position in Thinking Block")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlim(0, 1)
    ax_acc.set_ylim(-0.02, 1.02)
    ax_acc.set_title("Accuracy vs Position")
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, alpha=0.3)

    # -- Right panel: Probability vs Position --
    agg_p = df_binned.groupby("position_bin")["correct_token_prob"].agg(["mean", "std", "count"])
    sem = agg_p["std"].values / np.sqrt(np.maximum(agg_p["count"].values, 1))
    ci_p = 1.96 * sem
    positions_p = agg_p.index.values
    mean_p = agg_p["mean"].values

    ax_prob.plot(positions_p, mean_p, color="black", linestyle="--", linewidth=2,
                 label="Aggregate", zorder=5)
    ax_prob.fill_between(positions_p, mean_p - ci_p, mean_p + ci_p,
                         color="black", alpha=0.08)

    for ds in sorted(DATASET_COLORS.keys()):
        ds_df = df_binned[df_binned["dataset"] == ds]
        if ds_df.empty:
            continue
        grp = ds_df.groupby("position_bin")["correct_token_prob"].agg(["mean", "std", "count"])
        sem_ds = grp["std"].values / np.sqrt(np.maximum(grp["count"].values, 1))
        ci_ds = 1.96 * sem_ds
        pos_ds = grp.index.values
        mean_ds = grp["mean"].values

        ax_prob.plot(pos_ds, mean_ds, color=DATASET_COLORS[ds], linewidth=1.8,
                     label=DATASET_LABELS[ds])
        ax_prob.fill_between(pos_ds, mean_ds - ci_ds, mean_ds + ci_ds,
                             color=DATASET_COLORS[ds], alpha=0.15)

    ax_prob.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_prob.set_xlabel("Normalized Position in Thinking Block")
    ax_prob.set_ylabel("P(correct answer token)")
    ax_prob.set_xlim(0, 1)
    ax_prob.set_ylim(-0.02, 1.02)
    ax_prob.set_title("Correct Token Probability vs Position")
    ax_prob.legend(loc="lower right")
    ax_prob.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "transition_curves.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Correct vs incorrect traces
# ---------------------------------------------------------------------------

def plot_correct_vs_incorrect_traces(df, output_dir):
    """Compare accuracy-vs-position for originally correct vs incorrect traces.

    Originally correct traces are shown as a solid dark blue line; originally
    incorrect traces as a dashed red line. Reveals whether correct traces show
    sharper transitions than incorrect ones.

    Args:
        df: Results dataframe.
        output_dir: Directory where correct_vs_incorrect.png will be saved.
    """
    df_binned, _ = _bin_positions(df)

    fig, ax = plt.subplots(figsize=(10, 6))

    for is_correct, label, color, ls in [
        (True, "Originally correct traces", "#004488", "-"),
        (False, "Originally incorrect traces", "#BB5566", "--"),
    ]:
        sub = df_binned[df_binned["is_correct_trace"] == is_correct]
        if sub.empty:
            continue
        grp = sub.groupby("position_bin")["correct"].agg(["mean", "count"])
        p = grp["mean"].values
        n = grp["count"].values
        ci = 1.96 * np.sqrt(p * (1 - p) / np.maximum(n, 1))
        positions = grp.index.values

        ax.plot(positions, p, color=color, linestyle=ls, linewidth=2, label=label)
        ax.fill_between(positions, p - ci, p + ci, color=color, alpha=0.12)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Normalized Position in Thinking Block")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Accuracy by Original Trace Correctness")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "correct_vs_incorrect.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: Per-dataset detail (3-panel, dual y-axis)
# ---------------------------------------------------------------------------

def plot_per_dataset_detail(df, output_dir):
    """Generate a 3-panel figure, one per dataset, each with dual y-axis.

    Left y-axis shows accuracy (blue), right y-axis shows correct_token_prob
    (orange). Shared x-axis is normalized position. Reveals per-dataset
    transition dynamics in detail.

    Args:
        df: Results dataframe.
        output_dir: Directory where per_dataset_detail.png will be saved.
    """
    datasets = sorted(DATASET_COLORS.keys())
    df_binned, _ = _bin_positions(df)

    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df_binned[df_binned["dataset"] == ds]
        if sub.empty:
            ax.set_title(DATASET_LABELS.get(ds, ds) + " (no data)")
            continue

        # Accuracy on left y-axis
        grp_acc = sub.groupby("position_bin")["correct"].agg(["mean", "count"])
        p = grp_acc["mean"].values
        n = grp_acc["count"].values
        ci_acc = 1.96 * np.sqrt(p * (1 - p) / np.maximum(n, 1))
        pos = grp_acc.index.values

        ax.plot(pos, p, color="#0077BB", linewidth=2, label="Accuracy")
        ax.fill_between(pos, p - ci_acc, p + ci_acc, color="#0077BB", alpha=0.12)
        ax.set_xlabel("Normalized Position")
        ax.set_ylabel("Accuracy", color="#0077BB")
        ax.tick_params(axis="y", labelcolor="#0077BB")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(DATASET_LABELS.get(ds, ds))
        ax.grid(True, alpha=0.3)

        # Probability on right y-axis
        ax2 = ax.twinx()
        grp_prob = sub.groupby("position_bin")["correct_token_prob"].agg(["mean", "std", "count"])
        sem = grp_prob["std"].values / np.sqrt(np.maximum(grp_prob["count"].values, 1))
        ci_prob = 1.96 * sem
        mean_prob = grp_prob["mean"].values
        pos_prob = grp_prob.index.values

        ax2.plot(pos_prob, mean_prob, color="#EE7733", linewidth=2,
                 linestyle="--", label="P(correct)")
        ax2.fill_between(pos_prob, mean_prob - ci_prob, mean_prob + ci_prob,
                         color="#EE7733", alpha=0.12)
        ax2.set_ylabel("P(correct answer token)", color="#EE7733")
        ax2.tick_params(axis="y", labelcolor="#EE7733")
        ax2.set_ylim(-0.02, 1.02)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
                  fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "per_dataset_detail.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary_statistics(df):
    """Print comprehensive summary statistics to stdout.

    Includes total counts, no-answer rate, agreement rate, accuracy at
    early/middle/late positions per dataset, correct_token_prob at each
    region, and think length statistics.

    Args:
        df: Results dataframe.
    """
    n_probes = len(df)
    n_traces = df["trace_id"].nunique()
    n_correct_traces = df.drop_duplicates("trace_id")["is_correct_trace"].sum()
    n_incorrect_traces = n_traces - n_correct_traces

    print("=" * 60)
    print("FORCE-ANSWER EXPERIMENT SUMMARY")
    print("=" * 60)
    print()

    print(f"Total probes: {n_probes}")
    print(f"Traces: {n_traces} ({n_correct_traces} correct, {n_incorrect_traces} incorrect)")
    print()

    # Per-dataset trace counts
    trace_ds = df.drop_duplicates("trace_id").groupby("dataset")["trace_id"].count()
    print("Per-dataset traces:")
    for ds in sorted(trace_ds.index):
        label = DATASET_LABELS.get(ds, ds)
        print(f"  {label}: {trace_ds[ds]}")
    print()

    # Overall rates
    no_answer_rate = df["no_answer"].mean() * 100
    agreement_rate = df["agreement"].mean() * 100
    disagreement_count = (~df["agreement"]).sum()

    print(f"Overall no-answer rate: {no_answer_rate:.1f}%")
    print(f"Token/string agreement rate: {agreement_rate:.1f}%")
    print(f"Disagreement count: {disagreement_count}")
    if disagreement_count > 0:
        disagreement_traces = df[~df["agreement"]]["trace_id"].unique().tolist()
        print(f"  Traces with disagreements: {disagreement_traces}")
    print()

    # Position regions
    regions = {
        "Early (0.0-0.2)": (0.0, 0.2),
        "Middle (0.4-0.6)": (0.4, 0.6),
        "Late (0.8-1.0)": (0.8, 1.0),
    }

    print("Accuracy by position region:")
    print("-" * 50)
    print(f"{'Region':<22} {'Aggregate':>10}", end="")
    datasets = sorted(df["dataset"].unique())
    for ds in datasets:
        label = DATASET_LABELS.get(ds, ds)
        print(f" {label:>14}", end="")
    print()
    print("-" * 50)

    for region_name, (lo, hi) in regions.items():
        mask = (df["normalized_position"] >= lo) & (df["normalized_position"] <= hi)
        region_df = df[mask]
        if len(region_df) > 0:
            agg_acc = region_df["correct"].mean() * 100
        else:
            agg_acc = float("nan")
        print(f"{region_name:<22} {agg_acc:>9.1f}%", end="")

        for ds in datasets:
            ds_mask = mask & (df["dataset"] == ds)
            ds_region = df[ds_mask]
            if len(ds_region) > 0:
                ds_acc = ds_region["correct"].mean() * 100
            else:
                ds_acc = float("nan")
            print(f" {ds_acc:>13.1f}%", end="")
        print()
    print()

    # Probability by region
    print("Mean P(correct token) by position region:")
    print("-" * 50)
    print(f"{'Region':<22} {'Aggregate':>10}", end="")
    for ds in datasets:
        label = DATASET_LABELS.get(ds, ds)
        print(f" {label:>14}", end="")
    print()
    print("-" * 50)

    for region_name, (lo, hi) in regions.items():
        mask = (df["normalized_position"] >= lo) & (df["normalized_position"] <= hi)
        region_df = df[mask]
        if len(region_df) > 0:
            agg_prob = region_df["correct_token_prob"].mean()
        else:
            agg_prob = float("nan")
        print(f"{region_name:<22} {agg_prob:>10.4f}", end="")

        for ds in datasets:
            ds_mask = mask & (df["dataset"] == ds)
            ds_region = df[ds_mask]
            if len(ds_region) > 0:
                ds_prob = ds_region["correct_token_prob"].mean()
            else:
                ds_prob = float("nan")
            print(f" {ds_prob:>14.4f}", end="")
        print()
    print()

    # Think length statistics
    trace_lengths = df.drop_duplicates("trace_id")["think_length"]
    print("Think length statistics (tokens):")
    print(f"  Min:    {trace_lengths.min()}")
    print(f"  Median: {trace_lengths.median():.0f}")
    print(f"  Max:    {trace_lengths.max()}")
    print()

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse arguments, load results, print stats, generate all plots."""
    parser = argparse.ArgumentParser(
        description="Visualize force-answer transition experiment results.",
    )
    default_dir = os.path.join(
        os.environ.get("EXP_DIR", "experiments"), "force_answer",
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
        print(f"ERROR: results.parquet not found at {parquet_path}", file=sys.stderr)
        print("Run the experiment first:", file=sys.stderr)
        print("  python scripts/force_answer_experiment.py --config configs/force_answer.yaml", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}")
    print(f"Columns: {list(df.columns)}")
    print()

    # Summary statistics
    print_summary_statistics(df)

    # Generate all plots
    os.makedirs(results_dir, exist_ok=True)

    plot_transition_curves(df, results_dir)
    plot_correct_vs_incorrect_traces(df, results_dir)
    plot_per_dataset_detail(df, results_dir)

    print(f"\nPlots saved to {results_dir}")


if __name__ == "__main__":
    main()
