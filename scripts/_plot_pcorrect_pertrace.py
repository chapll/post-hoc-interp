"""Per-trace P(correct) from probe AND logit lens, compared to force-answer.

For each trace:
1. Probe P(correct): trained probe output, flipped for GT=No
2. Logit lens P(correct): project residual stream through model's unembedding
   to get P(correct_token) at each position -- same signal as force-answer
   but without actually generating text

If the logit lens shows clean transitions but the probe doesn't, it means
the information is there but the probe can't capture it linearly.

Per-trace plots only, no aggregation (inflection points differ per trace).

Usage:
    python scripts/_plot_pcorrect_pertrace.py
"""

import json
import os

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
FORCE_ANSWER_PATH = "src/experimental/gorn/experiments/force_answer/results.parquet"
TRACE_DIR = "src/experimental/gorn/data/traces/valid/folio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load probe
# ---------------------------------------------------------------------------
print("Loading probe...")
tensors = load_file("outputs/milestone_demo/probe_layer24.safetensors")
probe = nn.Linear(HIDDEN_DIM, 2)
probe.weight.data = tensors["layer_24_weight"].float()
probe.bias.data = tensors["layer_24_bias"].float()
probe.eval().to(device)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("Loading model...")
model, tok = load_model(gpu_id=0)

# Get the correct token IDs for Mu/Nu (both single tokens)
mu_token_id = tok.encode("Mu", add_special_tokens=False)
nu_token_id = tok.encode("Nu", add_special_tokens=False)
print(f"Mu token IDs: {mu_token_id}")
print(f"Nu token IDs: {nu_token_id}")

mu_id = mu_token_id[0]
nu_id = nu_token_id[0]

# ---------------------------------------------------------------------------
# Load force-answer data for comparison
# ---------------------------------------------------------------------------
df_fa = None
if os.path.isfile(FORCE_ANSWER_PATH):
    df_fa = pd.read_parquet(FORCE_ANSWER_PATH)
    df_fa = df_fa[df_fa["dataset"] == "folio"].copy()
    print(f"Loaded {len(df_fa)} force-answer FOLIO rows, {df_fa['trace_id'].nunique()} traces")

# ---------------------------------------------------------------------------
# Select traces to plot
# ---------------------------------------------------------------------------
# Use traces that appear in BOTH probe dataset and force-answer dataset
from scripts.milestone_demo import determine_correctness, discover_traces

all_traces = discover_traces(TRACE_DIR)
print(f"Found {len(all_traces)} valid probe traces")

# Find overlap with force-answer traces
if df_fa is not None:
    fa_trace_ids = set(df_fa["trace_id"].unique())
    overlap_traces = [t for t in all_traces if t["problem_id"] in fa_trace_ids]
    print(f"Overlap with force-answer: {len(overlap_traces)} traces")
    # Also get force-answer-only correct traces that we have probes for
    fa_correct = set(df_fa[df_fa["is_correct_trace"] == True]["trace_id"].unique())
    overlap_correct = [t for t in all_traces if t["problem_id"] in fa_correct]
    print(f"Overlap correct traces: {len(overlap_correct)}")

    # Use overlapping correct traces first, then fill with other traces
    plot_traces = overlap_correct[:8]
    if len(plot_traces) < 8:
        remaining = [t for t in all_traces if t not in plot_traces]
        plot_traces.extend(remaining[:8 - len(plot_traces)])
else:
    plot_traces = all_traces[:8]

print(f"\nWill plot {len(plot_traces)} traces")

# ---------------------------------------------------------------------------
# Process each trace
# ---------------------------------------------------------------------------
for trace_idx, trace in enumerate(plot_traces):
    pid = trace["problem_id"]
    gt = trace["ground_truth"]
    token_ids = trace["token_ids"]

    print(f"\n[{trace_idx + 1}/{len(plot_traces)}] {pid} (GT={gt}, {len(token_ids)} tokens)")

    mask = get_think_content_mask(token_ids)
    eligible = [i for i, m in enumerate(mask) if m]
    if len(eligible) < 5:
        print("  Skipping: too few eligible tokens")
        continue

    # Determine which token is the correct answer for this trace
    metadata_path = os.path.join(TRACE_DIR, pid, "trace_00.json")
    with open(metadata_path) as f:
        trace_data = json.load(f)
    meta = trace_data["metadata"]
    valid_token = meta["valid_token"]    # "Mu" or "Nu"
    invalid_token = meta["invalid_token"]
    if gt == "Yes":
        correct_token_name = valid_token
    else:
        correct_token_name = invalid_token

    correct_token_id = mu_id if correct_token_name == "Mu" else nu_id

    # Forward pass: get both layer LAYER residuals and final logits
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with model.trace(input_ids):
        saved_layer = model.model.layers[LAYER].output.save()
        # Also capture final layer output for logit lens through actual model head
        saved_final = model.lm_head.output.save()

    residual = saved_layer.detach()
    if residual.dim() == 3 and residual.shape[0] == 1:
        residual = residual.squeeze(0)

    final_logits = saved_final.detach()
    if final_logits.dim() == 3 and final_logits.shape[0] == 1:
        final_logits = final_logits.squeeze(0)

    start_pos = eligible[0]
    end_pos = eligible[-1]
    span = end_pos - start_pos if end_pos > start_pos else 1

    # --- Probe P(correct) ---
    probe_pcorrect = []
    norm_positions = []

    with torch.no_grad():
        for pos in eligible:
            h = residual[pos, :].float().unsqueeze(0).to(device)
            logits = probe(h)
            prob = F.softmax(logits, dim=-1)
            p_yes = prob[0, 1].item()
            p_correct = p_yes if gt == "Yes" else (1.0 - p_yes)
            probe_pcorrect.append(p_correct)
            norm_positions.append((pos - start_pos) / span)

    # --- Logit lens: P(correct_token) from final logits ---
    # final_logits is [seq_len, vocab_size] -- the model's actual output logits
    logit_lens_pcorrect = []
    with torch.no_grad():
        for pos in eligible:
            logits_at_pos = final_logits[pos, :]  # [vocab_size]
            # Get P(correct_token) via softmax over Mu and Nu only
            mu_logit = logits_at_pos[mu_id].float()
            nu_logit = logits_at_pos[nu_id].float()
            # Softmax over just these two
            pair_logits = torch.stack([mu_logit, nu_logit])
            pair_probs = F.softmax(pair_logits, dim=0)
            if correct_token_name == "Mu":
                p_correct_token = pair_probs[0].item()
            else:
                p_correct_token = pair_probs[1].item()
            logit_lens_pcorrect.append(p_correct_token)

    probe_pcorrect = np.array(probe_pcorrect)
    logit_lens_pcorrect = np.array(logit_lens_pcorrect)
    norm_positions = np.array(norm_positions)

    # --- Get force-answer data for this trace if available ---
    fa_positions = None
    fa_pcorrect = None
    if df_fa is not None and pid in df_fa["trace_id"].values:
        fa_trace = df_fa[df_fa["trace_id"] == pid].sort_values("normalized_position")
        fa_positions = fa_trace["normalized_position"].values
        fa_pcorrect = fa_trace["correct_token_prob"].values

    # --- Plot ---
    n_panels = 3 if fa_positions is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)

    short_pid = pid.split("_cfg")[0]
    cfg = "A" if "cfga" in pid else "B"

    # Panel 1: Probe P(correct)
    ax = axes[0]
    ax.plot(norm_positions, probe_pcorrect, linewidth=0.4, color="#bdc3c7", alpha=0.5)
    # Smoothed version
    w = min(50, len(probe_pcorrect) // 5)
    if w >= 3:
        smoothed = np.convolve(probe_pcorrect, np.ones(w) / w, mode="valid")
        smooth_x = norm_positions[w // 2 : w // 2 + len(smoothed)]
        ax.plot(smooth_x, smoothed, linewidth=2.0, color="#2c3e50", label=f"Smoothed (w={w})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Probe P(correct)", fontsize=11)
    ax.set_title(f"Probe P(correct) -- {short_pid}({cfg}) | GT={gt} | Layer {LAYER}", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Logit lens P(correct_token)
    ax = axes[1]
    ax.plot(norm_positions, logit_lens_pcorrect, linewidth=0.8, color="#EE7733", alpha=0.9)
    # Smoothed
    if w >= 3:
        smoothed_ll = np.convolve(logit_lens_pcorrect, np.ones(w) / w, mode="valid")
        smooth_x_ll = norm_positions[w // 2 : w // 2 + len(smoothed_ll)]
        ax.plot(smooth_x_ll, smoothed_ll, linewidth=2.0, color="#CC3311", label=f"Smoothed (w={w})")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("Logit lens P(correct_token)", fontsize=11)
    ax.set_title(f"Logit Lens P(correct_token) -- final layer output | {short_pid}({cfg})", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Force-answer (if available)
    if fa_positions is not None:
        ax = axes[2]
        ax.plot(fa_positions, fa_pcorrect, "o-", linewidth=2.0, markersize=6,
                color="#009988", label="Force-answer P(correct_token)")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
        ax.set_ylabel("Force-answer P(correct_token)", fontsize=11)
        ax.set_title(f"Force-Answer -- {short_pid}({cfg}) | 20 interpolation points", fontsize=11)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Normalized position in think block [0, 1]", fontsize=11)

    plt.tight_layout()
    fname = f"pcorrect_pertrace_{pid}.png"
    path = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    del saved_layer, saved_final, input_ids, residual, final_logits
    torch.cuda.empty_cache()

print("\nDone.")
