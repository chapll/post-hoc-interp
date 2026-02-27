"""Aggregated P(correct) probe confidence vs force-answer comparison.

Key fixes over previous plots:
1. P(correct) instead of P(Yes):
   - GT=Yes traces: P(correct) = P(Yes)
   - GT=No traces: P(correct) = 1 - P(Yes) = P(No)
   This makes both classes "rise toward 1" as model gains confidence.

2. Aggregation across traces at binned normalized positions:
   Same approach as force-answer plotting (binned positions, mean + CI).
   Smooths out per-token noise.

3. Side-by-side comparison with force-answer results on FOLIO.

Usage:
    python scripts/_plot_pcorrect_aggregate.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from src.model import HIDDEN_DIM, load_model
from src.tokens import count_think_blocks, get_pre_handoff_window, get_think_content_mask
from src.utils import set_seed

set_seed(137)
device = torch.device("cuda:0")

OUTPUT_DIR = "outputs/milestone_demo/plots"
LAYER = 24
N_BINS = 25
FORCE_ANSWER_PATH = "src/experimental/gorn/experiments/force_answer/results.parquet"
TRACE_DIR = "src/experimental/gorn/data/traces/valid/folio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load probe
# ---------------------------------------------------------------------------
print("Loading probe...")
tensors = load_file("outputs/milestone_demo/probe_layer24.safetensors")
probe = nn.Linear(HIDDEN_DIM, 2)
probe.weight.data = tensors["layer_24_weight"].float()
probe.bias.data = tensors["layer_24_bias"].float()
probe.eval().to(device)

# ---------------------------------------------------------------------------
# 2. Discover traces (same filtering as milestone_demo.py)
# ---------------------------------------------------------------------------
print("Discovering traces...")
from scripts.milestone_demo import determine_correctness, discover_traces

traces = discover_traces(TRACE_DIR)
print(f"Found {len(traces)} valid traces")

# ---------------------------------------------------------------------------
# 3. Load model and compute P(correct) for ALL traces
# ---------------------------------------------------------------------------
print("Loading model...")
model, tok = load_model(gpu_id=0)

# Collect all (normalized_position, p_correct, ground_truth) rows
rows = []

for i, trace in enumerate(traces):
    token_ids = trace["token_ids"]
    gt = trace["ground_truth"]
    pid = trace["problem_id"]

    mask = get_think_content_mask(token_ids)
    eligible = [idx for idx, m in enumerate(mask) if m]
    if len(eligible) < 5:
        continue

    # Forward pass
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with model.trace(input_ids):
        saved = model.model.layers[LAYER].output.save()

    residual = saved.detach()
    if residual.dim() == 3 and residual.shape[0] == 1:
        residual = residual.squeeze(0)

    # Normalize positions to [0, 1] within this trace's think block
    start_pos = eligible[0]
    end_pos = eligible[-1]
    span = end_pos - start_pos if end_pos > start_pos else 1

    with torch.no_grad():
        for pos in eligible:
            h = residual[pos, :].float().unsqueeze(0).to(device)
            logits = probe(h)
            prob = F.softmax(logits, dim=-1)
            p_yes = prob[0, 1].item()

            # P(correct): flip for GT=No traces
            if gt == "Yes":
                p_correct = p_yes
            else:
                p_correct = 1.0 - p_yes

            norm_pos = (pos - start_pos) / span

            rows.append({
                "trace_id": pid,
                "ground_truth": gt,
                "normalized_position": norm_pos,
                "p_yes": p_yes,
                "p_correct": p_correct,
            })

    del saved, input_ids, residual
    torch.cuda.empty_cache()

    if (i + 1) % 20 == 0 or (i + 1) == len(traces):
        print(f"  Processed {i + 1}/{len(traces)} traces ({len(rows)} datapoints)")

df_probe = pd.DataFrame(rows)
print(f"\nTotal datapoints: {len(df_probe)}")
print(f"Traces: {df_probe['trace_id'].nunique()}")

# ---------------------------------------------------------------------------
# 4. Bin positions and compute aggregated curves
# ---------------------------------------------------------------------------
bin_edges = np.linspace(0, 1, N_BINS + 1)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
df_probe["position_bin"] = pd.cut(
    df_probe["normalized_position"],
    bins=bin_edges,
    labels=bin_centers,
    include_lowest=True,
)
df_probe["position_bin"] = df_probe["position_bin"].astype(float)

# Aggregated P(correct)
agg = df_probe.groupby("position_bin")["p_correct"].agg(["mean", "std", "count"])
agg_mean = agg["mean"].values
agg_std = agg["std"].values
agg_n = agg["count"].values
agg_se = agg_std / np.sqrt(np.maximum(agg_n, 1))
agg_ci = 1.96 * agg_se
agg_pos = agg.index.values

# Per-GT aggregated
yes_df = df_probe[df_probe["ground_truth"] == "Yes"]
no_df = df_probe[df_probe["ground_truth"] == "No"]

yes_agg = yes_df.groupby("position_bin")["p_correct"].agg(["mean", "count"])
no_agg = no_df.groupby("position_bin")["p_correct"].agg(["mean", "count"])

# Print summary
print("\n" + "=" * 60)
print("AGGREGATED P(CORRECT) BY POSITION")
print("=" * 60)
for region, (lo, hi) in [("Early (0-0.2)", (0, 0.2)), ("Middle (0.4-0.6)", (0.4, 0.6)), ("Late (0.8-1.0)", (0.8, 1.0))]:
    region_mask = (df_probe["normalized_position"] >= lo) & (df_probe["normalized_position"] <= hi)
    region_data = df_probe[region_mask]
    if len(region_data) > 0:
        print(f"  {region}: P(correct) = {region_data['p_correct'].mean():.4f} (n={len(region_data)})")
    else:
        print(f"  {region}: no data")

# ---------------------------------------------------------------------------
# 5. Load force-answer data for comparison
# ---------------------------------------------------------------------------
fa_agg_pos = None
fa_agg_mean = None
fa_agg_ci = None

if os.path.isfile(FORCE_ANSWER_PATH):
    print(f"\nLoading force-answer data from {FORCE_ANSWER_PATH}...")
    df_fa = pd.read_parquet(FORCE_ANSWER_PATH)
    # Filter to FOLIO only, correct traces only (to match our probe dataset)
    df_fa_folio = df_fa[(df_fa["dataset"] == "folio") & (df_fa["is_correct_trace"] == True)].copy()
    print(f"  FOLIO correct traces in force-answer: {df_fa_folio['trace_id'].nunique()}")

    df_fa_folio["position_bin"] = pd.cut(
        df_fa_folio["normalized_position"],
        bins=bin_edges,
        labels=bin_centers,
        include_lowest=True,
    )
    df_fa_folio["position_bin"] = df_fa_folio["position_bin"].astype(float)

    fa_grp = df_fa_folio.groupby("position_bin")["correct"].agg(["mean", "count"])
    fa_agg_pos = fa_grp.index.values
    fa_agg_mean = fa_grp["mean"].values
    fa_n = fa_grp["count"].values
    fa_agg_ci = 1.96 * np.sqrt(fa_agg_mean * (1 - fa_agg_mean) / np.maximum(fa_n, 1))
else:
    print(f"\nForce-answer results not found at {FORCE_ANSWER_PATH}, skipping comparison.")

# ---------------------------------------------------------------------------
# 6. PLOT 1: Aggregated P(correct) probe curve
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(agg_pos, agg_mean, color="#2c3e50", linewidth=2.5, label="Probe P(correct) -- all traces", zorder=5)
ax.fill_between(agg_pos, agg_mean - agg_ci, agg_mean + agg_ci, color="#2c3e50", alpha=0.15)

# Per-GT curves
if len(yes_agg) > 0:
    ax.plot(yes_agg.index.values, yes_agg["mean"].values,
            color="#0077BB", linewidth=1.5, linestyle="--", alpha=0.8, label="Probe P(correct) -- GT=Yes traces")
if len(no_agg) > 0:
    ax.plot(no_agg.index.values, no_agg["mean"].values,
            color="#CC3311", linewidth=1.5, linestyle="--", alpha=0.8, label="Probe P(correct) -- GT=No traces")

ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
ax.set_xlabel("Normalized Position in Think Block", fontsize=12)
ax.set_ylabel("P(correct)", fontsize=12)
ax.set_xlim(0, 1)
ax.set_ylim(-0.02, 1.02)
ax.set_title(
    f"Aggregated Probe Confidence -- P(correct) over {df_probe['trace_id'].nunique()} FOLIO Traces | Layer {LAYER}",
    fontsize=13,
)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
path1 = os.path.join(OUTPUT_DIR, "aggregated_pcorrect.png")
fig.savefig(path1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {path1}")

# ---------------------------------------------------------------------------
# 7. PLOT 2: Side-by-side probe vs force-answer
# ---------------------------------------------------------------------------
if fa_agg_pos is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Left: Force-answer accuracy
    ax1.plot(fa_agg_pos, fa_agg_mean, color="#EE7733", linewidth=2.5, label="Force-answer accuracy")
    ax1.fill_between(fa_agg_pos, fa_agg_mean - fa_agg_ci, fa_agg_mean + fa_agg_ci,
                     color="#EE7733", alpha=0.15)
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel("Normalized Position in Think Block", fontsize=12)
    ax1.set_ylabel("Accuracy (force-answer)", fontsize=12)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title("Force-Answer Accuracy (FOLIO, correct traces)", fontsize=12)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Probe P(correct)
    ax2.plot(agg_pos, agg_mean, color="#2c3e50", linewidth=2.5, label="Probe P(correct)")
    ax2.fill_between(agg_pos, agg_mean - agg_ci, agg_mean + agg_ci, color="#2c3e50", alpha=0.15)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel("Normalized Position in Think Block", fontsize=12)
    ax2.set_ylabel("P(correct)", fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.02, 1.02)
    ax2.set_title(f"Probe P(correct) (FOLIO, {df_probe['trace_id'].nunique()} traces, Layer {LAYER})", fontsize=12)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "probe_vs_force_answer.png")
    fig.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path2}")

    # PLOT 3: Both on same axes for direct overlay
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(fa_agg_pos, fa_agg_mean, color="#EE7733", linewidth=2.5,
            label="Force-answer accuracy (behavioral)", zorder=4)
    ax.fill_between(fa_agg_pos, fa_agg_mean - fa_agg_ci, fa_agg_mean + fa_agg_ci,
                    color="#EE7733", alpha=0.12)

    ax.plot(agg_pos, agg_mean, color="#2c3e50", linewidth=2.5,
            label="Probe P(correct) (representational)", zorder=5)
    ax.fill_between(agg_pos, agg_mean - agg_ci, agg_mean + agg_ci,
                    color="#2c3e50", alpha=0.12)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Normalized Position in Think Block", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(
        f"Probe P(correct) vs Force-Answer Accuracy -- FOLIO | Layer {LAYER}",
        fontsize=13,
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, "probe_vs_force_answer_overlay.png")
    fig.savefig(path3, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path3}")

# ---------------------------------------------------------------------------
# 8. PLOT 4: Individual trace P(correct) curves (sample of 10)
# ---------------------------------------------------------------------------
sample_traces = traces[:10]
fig, axes = plt.subplots(2, 5, figsize=(24, 8))
axes = axes.flatten()

for ax_idx, trace in enumerate(sample_traces):
    pid = trace["problem_id"]
    gt = trace["ground_truth"]
    sub = df_probe[df_probe["trace_id"] == pid].sort_values("normalized_position")

    ax = axes[ax_idx]
    ax.plot(sub["normalized_position"], sub["p_correct"], linewidth=0.5, color="#bdc3c7", alpha=0.6)

    # Smoothed version
    if len(sub) > 50:
        w = 50
        smoothed = np.convolve(sub["p_correct"].values, np.ones(w) / w, mode="valid")
        smooth_x = sub["normalized_position"].values[w // 2 : w // 2 + len(smoothed)]
        ax.plot(smooth_x, smoothed, linewidth=2, color="#2c3e50")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, 1)
    short_pid = pid.split("_cfg")[0]
    cfg = "A" if "cfga" in pid else "B"
    ax.set_title(f"{short_pid}({cfg}) GT={gt}", fontsize=9)
    if ax_idx >= 5:
        ax.set_xlabel("Norm. position", fontsize=8)
    if ax_idx % 5 == 0:
        ax.set_ylabel("P(correct)", fontsize=8)
    ax.grid(True, alpha=0.2)

plt.suptitle(f"Per-Trace P(correct) -- 10 FOLIO Traces | Layer {LAYER}", fontsize=13, y=1.02)
plt.tight_layout()
path4 = os.path.join(OUTPUT_DIR, "pcorrect_individual_10traces.png")
fig.savefig(path4, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {path4}")

print("\nDone.")
