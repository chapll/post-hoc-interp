"""Visualization for the per-token force-answer probing experiment (Phase 2.4).

Reads per-token probe results and Phase 2.2 coarse-grid results, generates:
1. per_token_curve.png -- Full 824-point accuracy + probability curves
2. comparison_overlay.png -- Per-token curves with Phase 2.2 coarse points overlaid
3. transition_characterization.png -- Zoomed view of the transition region with metrics

Usage:
    python scripts/plot_per_token_probe.py
    python scripts/plot_per_token_probe.py --results-dir /path/to/per_token_probe
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Style configuration (matching plot_cross_seed.py / plot_force_answer.py)
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

ACC_COLOR = "#0077BB"    # blue
PROB_COLOR = "#EE7733"   # orange
COARSE_COLOR = "#BB5566" # red-ish for Phase 2.2 overlay


# ---------------------------------------------------------------------------
# Plot 1: per_token_curve.png -- Two-panel (accuracy + probability)
# ---------------------------------------------------------------------------

def plot_per_token_curve(df, output_dir):
    """Generate the primary two-panel figure: per-token accuracy and probability.

    Left panel shows binary accuracy (0/1) at each probe position with a
    20-point rolling average overlay. Right panel shows P(correct token)
    with the same rolling average.

    Args:
        df: Results dataframe sorted by probe_position, with columns
            probe_position, correct, correct_token_prob, problem_id.
        output_dir: Directory where per_token_curve.png will be saved.
    """
    problem_id = df["problem_id"].iloc[0]
    fig, (ax_acc, ax_prob) = plt.subplots(1, 2, figsize=(14, 5))

    positions = df["probe_position"].values

    # -- Left panel: Accuracy vs Position --
    accuracy = df["correct"].astype(float).values
    rolling_acc = pd.Series(accuracy).rolling(window=20, min_periods=1).mean().values

    ax_acc.plot(positions, accuracy, color=ACC_COLOR, linewidth=0.8, alpha=0.8)
    ax_acc.plot(
        positions, rolling_acc, color=ACC_COLOR, linewidth=2, alpha=0.9,
        label="20-point rolling avg",
    )
    ax_acc.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_acc.set_xlabel("Probe Position (token index)")
    ax_acc.set_ylabel("Correct (0 or 1)")
    ax_acc.set_title(f"Per-Token Accuracy ({len(df)} positions)")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.legend(loc="lower right")
    ax_acc.grid(True, alpha=0.3)

    # -- Right panel: Probability vs Position --
    prob = df["correct_token_prob"].values
    rolling_prob = pd.Series(prob).rolling(window=20, min_periods=1).mean().values

    ax_prob.plot(positions, prob, color=PROB_COLOR, linewidth=0.8, alpha=0.8)
    ax_prob.plot(
        positions, rolling_prob, color=PROB_COLOR, linewidth=2, alpha=0.9,
        label="20-point rolling avg",
    )
    ax_prob.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_prob.set_xlabel("Probe Position (token index)")
    ax_prob.set_ylabel("P(correct answer token)")
    ax_prob.set_title("Per-Token Correct Token Probability")
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.legend(loc="lower right")
    ax_prob.grid(True, alpha=0.3)

    fig.suptitle(f"Per-Token Force-Answer Probing: {problem_id}", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = os.path.join(output_dir, "per_token_curve.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: comparison_overlay.png -- Per-token with Phase 2.2 overlay
# ---------------------------------------------------------------------------

def plot_comparison_overlay(df, df_coarse, output_dir):
    """Generate two-panel figure with Phase 2.2 coarse-grid points overlaid.

    Uses normalized_position (0 to 1) on x-axis so both dense per-token
    and coarse-grid points align properly.

    Args:
        df: Per-token results dataframe (824 points).
        df_coarse: Phase 2.2 coarse results filtered to the target problem.
        output_dir: Directory where comparison_overlay.png will be saved.
    """
    fig, (ax_acc, ax_prob) = plt.subplots(1, 2, figsize=(14, 5))

    norm_pos = df["normalized_position"].values
    coarse_norm = df_coarse["normalized_position"].values

    # -- Left panel: Accuracy --
    accuracy = df["correct"].astype(float).values
    ax_acc.plot(
        norm_pos, accuracy, color=ACC_COLOR, linewidth=0.8, alpha=0.6,
        label=f"Per-token ({len(df)} points)",
    )
    # Rolling average on dense data
    rolling_acc = pd.Series(accuracy).rolling(window=20, min_periods=1).mean().values
    ax_acc.plot(
        norm_pos, rolling_acc, color=ACC_COLOR, linewidth=2, alpha=0.9,
        label="20-pt rolling avg",
    )
    # Phase 2.2 coarse points
    ax_acc.scatter(
        coarse_norm,
        df_coarse["correct"].astype(float).values,
        color=COARSE_COLOR, s=80, zorder=5, marker="o",
        edgecolors="white", linewidth=1.5,
        label=f"Phase 2.2 ({len(df_coarse)} points)",
    )
    ax_acc.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_acc.set_xlabel("Normalized Position in Thinking Block")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_xlim(-0.02, 1.02)
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.set_title("Accuracy: Per-Token vs Coarse Grid")
    ax_acc.legend(loc="lower right", fontsize=10)
    ax_acc.grid(True, alpha=0.3)

    # -- Right panel: Probability --
    prob = df["correct_token_prob"].values
    ax_prob.plot(
        norm_pos, prob, color=PROB_COLOR, linewidth=0.8, alpha=0.6,
        label=f"Per-token ({len(df)} points)",
    )
    rolling_prob = pd.Series(prob).rolling(window=20, min_periods=1).mean().values
    ax_prob.plot(
        norm_pos, rolling_prob, color=PROB_COLOR, linewidth=2, alpha=0.9,
        label="20-pt rolling avg",
    )
    ax_prob.scatter(
        coarse_norm,
        df_coarse["correct_token_prob"].values,
        color=COARSE_COLOR, s=80, zorder=5, marker="o",
        edgecolors="white", linewidth=1.5,
        label=f"Phase 2.2 ({len(df_coarse)} points)",
    )
    ax_prob.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_prob.set_xlabel("Normalized Position in Thinking Block")
    ax_prob.set_ylabel("P(correct answer token)")
    ax_prob.set_xlim(-0.02, 1.02)
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.set_title("Probability: Per-Token vs Coarse Grid")
    ax_prob.legend(loc="lower right", fontsize=10)
    ax_prob.grid(True, alpha=0.3)

    fig.suptitle("Per-Token vs Coarse Grid Comparison", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = os.path.join(output_dir, "comparison_overlay.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Plot 3: transition_characterization.png -- Zoomed transition region
# ---------------------------------------------------------------------------

def _find_transition_region(df, window=20):
    """Identify the transition region from the rolling accuracy curve.

    Finds the first position where rolling-20 accuracy exceeds 0.5 and the
    last position where it drops below 0.5. Expands by 10% on each side for
    context. Falls back to 25%-85% of positions if transition is unclear.

    Args:
        df: Per-token results sorted by probe_position.
        window: Rolling window size for smoothing.

    Returns:
        Tuple (start_pos, end_pos) defining the transition region in
        probe_position space.
    """
    positions = df["probe_position"].values
    accuracy = df["correct"].astype(float).values
    rolling = pd.Series(accuracy).rolling(window=window, min_periods=1).mean().values

    above_half = np.where(rolling >= 0.5)[0]
    below_half = np.where(rolling < 0.5)[0]

    if len(above_half) > 0 and len(below_half) > 0:
        first_above = above_half[0]
        last_below = below_half[-1]

        # Use the wider range to capture full transition
        start_idx = min(first_above, last_below)
        end_idx = max(first_above, last_below)

        # Expand by 10% on each side for context
        span = end_idx - start_idx
        margin = max(int(0.1 * span), 10)
        start_idx = max(0, start_idx - margin)
        end_idx = min(len(positions) - 1, end_idx + margin)

        return int(positions[start_idx]), int(positions[end_idx])

    # Fallback: 25%-85% of positions
    n = len(positions)
    return int(positions[n // 4]), int(positions[int(0.85 * n)])


def plot_transition_characterization(df, summary, output_dir):
    """Generate the transition characterization figure with zoom and metrics.

    Three-panel figure: (1) accuracy zoom in transition region, (2) probability
    zoom in transition region, (3) text summary of transition metrics from
    results_summary.json.

    Args:
        df: Per-token results sorted by probe_position.
        summary: Dict loaded from results_summary.json.
        output_dir: Directory where transition_characterization.png will be saved.
    """
    tc = summary["transition_characterization"]

    # Identify transition region
    start_pos, end_pos = _find_transition_region(df)
    mask = (df["probe_position"] >= start_pos) & (df["probe_position"] <= end_pos)
    df_zoom = df[mask]

    fig, (ax_acc, ax_prob, ax_text) = plt.subplots(
        1, 3, figsize=(18, 5),
        gridspec_kw={"width_ratios": [2, 2, 1.2]},
    )

    positions = df_zoom["probe_position"].values
    accuracy = df_zoom["correct"].astype(float).values
    rolling_acc = pd.Series(accuracy).rolling(window=20, min_periods=1).mean().values
    prob = df_zoom["correct_token_prob"].values
    rolling_prob = pd.Series(prob).rolling(window=20, min_periods=1).mean().values

    # -- Panel 1: Accuracy zoom --
    ax_acc.plot(positions, accuracy, color=ACC_COLOR, linewidth=0.8, alpha=0.7)
    ax_acc.plot(
        positions, rolling_acc, color=ACC_COLOR, linewidth=2.5, alpha=0.9,
        label="20-pt rolling avg",
    )

    # Mark first sustained correct position
    fsc = tc.get("first_sustained_correct_position")
    if fsc is not None and start_pos <= fsc <= end_pos:
        ax_acc.axvline(
            fsc, color="#009988", linestyle="--", linewidth=1.5, alpha=0.8,
            label=f"First sustained (k={tc['sustained_k']}): pos {fsc}",
        )

    ax_acc.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_acc.set_xlabel("Probe Position (token index)")
    ax_acc.set_ylabel("Correct (0 or 1)")
    ax_acc.set_title("Accuracy (Transition Region)")
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.legend(loc="lower right", fontsize=9)
    ax_acc.grid(True, alpha=0.3)

    # -- Panel 2: Probability zoom --
    ax_prob.plot(positions, prob, color=PROB_COLOR, linewidth=0.8, alpha=0.7)
    ax_prob.plot(
        positions, rolling_prob, color=PROB_COLOR, linewidth=2.5, alpha=0.9,
        label="20-pt rolling avg",
    )

    if fsc is not None and start_pos <= fsc <= end_pos:
        ax_prob.axvline(
            fsc, color="#009988", linestyle="--", linewidth=1.5, alpha=0.8,
            label=f"First sustained (k={tc['sustained_k']})",
        )

    ax_prob.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_prob.set_xlabel("Probe Position (token index)")
    ax_prob.set_ylabel("P(correct answer token)")
    ax_prob.set_title("Probability (Transition Region)")
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.legend(loc="lower right", fontsize=9)
    ax_prob.grid(True, alpha=0.3)

    # -- Panel 3: Text summary --
    ax_text.axis("off")

    fsc_label = f"position {fsc}" if fsc is not None else "None"
    text_lines = [
        "Transition Metrics",
        "",
        f"Total transitions: {tc['total_transitions']}",
        f"  Correct -> Incorrect: {tc['correct_to_incorrect_transitions']}",
        f"  Incorrect -> Correct: {tc['incorrect_to_correct_transitions']}",
        "",
        f"Longest correct run: {tc['longest_correct_run']} positions",
        f"Longest incorrect run: {tc['longest_incorrect_run']} positions",
        "",
        f"First sustained correct",
        f"  (k={tc['sustained_k']}): {fsc_label}",
        "",
        f"Middle-third prob variance:",
        f"  {tc['middle_third_prob_variance']:.6f}",
        "",
        f"Overall accuracy: {summary['overall_accuracy']:.1%}",
        f"First third:  {summary['first_third_accuracy']:.1%}",
        f"Middle third: {summary['middle_third_accuracy']:.1%}",
        f"Last third:   {summary['last_third_accuracy']:.1%}",
    ]

    ax_text.text(
        0.05, 0.95, "\n".join(text_lines),
        transform=ax_text.transAxes,
        fontsize=10, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )
    ax_text.set_title("Summary Metrics")

    fig.suptitle("Transition Characterization", fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = os.path.join(output_dir, "transition_characterization.png")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Transition shape heuristic
# ---------------------------------------------------------------------------

def classify_transition_shape(tc):
    """Classify the transition shape based on total number of transitions.

    Simple heuristic:
    - <= 2 transitions: step function
    - <= 6 transitions: gradual ramp
    - > 6 transitions: oscillatory

    This is a rough label -- the plots are the primary characterization.

    Args:
        tc: Transition characterization dict from results_summary.json.

    Returns:
        String classification: "step function", "gradual ramp", or "oscillatory".
    """
    total = tc["total_transitions"]
    if total <= 2:
        return "step function"
    elif total <= 6:
        return "gradual ramp"
    else:
        return "oscillatory"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse arguments, load results, generate all plots, print summary."""
    parser = argparse.ArgumentParser(
        description="Visualize per-token force-answer probing results (Phase 2.4).",
    )
    default_results = os.path.join(
        os.environ.get("EXP_DIR", "experiments"), "per_token_probe",
    )
    default_phase22 = os.path.join(
        os.environ.get("EXP_DIR", "experiments"), "force_answer",
    )
    parser.add_argument(
        "--results-dir",
        default=default_results,
        help=f"Directory containing per-token results (default: {default_results})",
    )
    parser.add_argument(
        "--phase22-dir",
        default=default_phase22,
        help=f"Directory containing Phase 2.2 results (default: {default_phase22})",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    phase22_dir = args.phase22_dir

    # ------------------------------------------------------------------
    # 1. Load per-token results
    # ------------------------------------------------------------------
    parquet_path = os.path.join(results_dir, "results.parquet")
    if not os.path.isfile(parquet_path):
        print(f"ERROR: results.parquet not found at {parquet_path}", file=sys.stderr)
        print("Run the experiment first:", file=sys.stderr)
        print("  python scripts/per_token_probe.py --config configs/per_token_probe.yaml", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    df = df.sort_values("probe_position").reset_index(drop=True)
    print(f"Loaded {len(df)} per-token probe results from {parquet_path}")

    # ------------------------------------------------------------------
    # 2. Load Phase 2.2 coarse results and filter to target trace
    # ------------------------------------------------------------------
    phase22_parquet = os.path.join(phase22_dir, "results.parquet")
    if not os.path.isfile(phase22_parquet):
        print(f"WARNING: Phase 2.2 results not found at {phase22_parquet}", file=sys.stderr)
        print("  Comparison overlay will be skipped.", file=sys.stderr)
        df_coarse = None
    else:
        df_coarse_all = pd.read_parquet(phase22_parquet)
        problem_id = df["problem_id"].iloc[0]
        df_coarse = df_coarse_all[
            df_coarse_all["problem_id"] == problem_id
        ].sort_values("probe_position").reset_index(drop=True)
        print(f"Loaded Phase 2.2 coarse results: {len(df_coarse)} points for {problem_id}")

    # ------------------------------------------------------------------
    # 3. Load JSON summary for transition characterization
    # ------------------------------------------------------------------
    summary_path = os.path.join(results_dir, "results_summary.json")
    if not os.path.isfile(summary_path):
        print(f"ERROR: results_summary.json not found at {summary_path}", file=sys.stderr)
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)
    tc = summary["transition_characterization"]

    # ------------------------------------------------------------------
    # 4. Generate plots
    # ------------------------------------------------------------------
    os.makedirs(results_dir, exist_ok=True)

    # Plot 1: per_token_curve.png
    plot_per_token_curve(df, results_dir)

    # Plot 2: comparison_overlay.png (only if Phase 2.2 data available)
    if df_coarse is not None and len(df_coarse) > 0:
        plot_comparison_overlay(df, df_coarse, results_dir)
    else:
        print("Skipped comparison_overlay.png (no Phase 2.2 data)")

    # Plot 3: transition_characterization.png
    plot_transition_characterization(df, summary, results_dir)

    # ------------------------------------------------------------------
    # 5. Console summary
    # ------------------------------------------------------------------
    shape = classify_transition_shape(tc)
    n_coarse = len(df_coarse) if df_coarse is not None else 0

    print()
    print("=== Per-Token Probe Visualization ===")
    print(f"Problem: {df['problem_id'].iloc[0]}")
    print(f"Positions: {len(df)}")
    print(f"Overall accuracy: {summary['overall_accuracy']:.1%}")
    print(f"Transition shape: {shape}")
    print(f"Phase 2.2 had {n_coarse} points, per-token has {len(df)}")
    print(f"Plots saved to: {results_dir}")


if __name__ == "__main__":
    main()
