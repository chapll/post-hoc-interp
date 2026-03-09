"""Probe held-out traces with the activation diff 'knowing direction'.

For each test trace, runs a full forward pass, captures residual stream
activations at every thinking token position, and computes the dot product
with the normalized knowing direction from the activation diff experiment.

Produces per-trace plots showing the projection score across thinking
positions for multiple layers.

Usage:
    python scripts/probe_with_direction.py
"""

import json
import glob
import os
import random
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
DIRECTION_PATH = "experiments/activation_diff/knowing_direction.safetensors"
META_PATH = "experiments/activation_diff/all_traces_meta.parquet"
TRACES_DIR = "data/traces/valid"
OUTPUT_DIR = "experiments/activation_diff/probe_plots"

# Layers to probe (focused around layer 24 where transition signal is clearest)
PROBE_LAYERS = [18, 19, 20, 21, 22, 23, 24, 25, 26]

# Number of test traces
N_TEST_TRACES = 8

# Use trace_01 (held out: direction was computed from trace_00)
TEST_TRACE_INDEX = 1

SEED = 137


def load_test_traces(meta_df, traces_dir, n_traces, trace_index, seed):
    """Select held-out test traces from diverse problems/datasets.

    Picks traces from problems that were in the filtered set (known good
    transition pattern) so we can see if the probe tracks it on a different
    trace from the same problem.
    """
    rng = random.Random(seed)

    # Get all available trace_01 files
    all_traces = []
    for ds in ["gsm_symbolic", "folio", "strategyqa"]:
        ds_dir = os.path.join(traces_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        for pid in sorted(os.listdir(ds_dir)):
            trace_path = os.path.join(
                ds_dir, pid, f"trace_{trace_index:02d}.json"
            )
            if os.path.isfile(trace_path):
                all_traces.append((ds, pid, trace_path))

    # Shuffle and pick diverse set
    rng.shuffle(all_traces)

    # Try to get a mix of datasets
    selected = []
    by_ds = {}
    for ds, pid, path in all_traces:
        by_ds.setdefault(ds, []).append((ds, pid, path))

    per_ds = max(1, n_traces // len(by_ds))
    for ds in sorted(by_ds.keys()):
        selected.extend(by_ds[ds][:per_ds])

    # Fill remaining
    remaining = [t for t in all_traces if t not in selected]
    while len(selected) < n_traces and remaining:
        selected.append(remaining.pop(0))

    selected = selected[:n_traces]

    # Load trace data
    loaded = []
    for ds, pid, path in selected:
        with open(path) as f:
            trace = json.load(f)
        meta = trace["metadata"]
        if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
            continue
        think_start, think_end = meta["think_boundaries"][0]
        think_length = think_end - think_start - 1
        if think_length < 20:
            continue
        gt = meta["ground_truth"]
        expected = meta["valid_token"] if gt == "Yes" else meta["invalid_token"]
        loaded.append({
            "problem_id": pid,
            "dataset": ds,
            "token_ids": trace["token_ids"],
            "think_start": think_start,
            "think_length": think_length,
            "expected_token": expected,
            "ground_truth": gt,
        })

    return loaded


def probe_trace(model, trace_data, direction_normalized, layers, device):
    """Run forward pass on full trace and compute projections at each layer.

    Returns dict mapping layer_idx -> projection array of shape [think_length].
    """
    token_ids = trace_data["token_ids"]
    think_start = trace_data["think_start"]
    think_length = trace_data["think_length"]

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Prepare direction vectors on device
    dir_vecs = {}
    for layer_idx in layers:
        dir_vecs[layer_idx] = torch.tensor(
            direction_normalized[layer_idx], device=device, dtype=torch.float32,
        )

    # Single forward pass, capture projections at think positions
    # Think content starts at think_start+1 (after <think> token)
    ts = think_start + 1
    te = ts + think_length

    projections = {}
    with torch.no_grad():
        outputs = model(input_tensor, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [1, seq_len, hidden_dim]

        for layer_idx in layers:
            # hidden_states[0] is embeddings, hidden_states[i] is after layer i-1
            # So hidden_states[layer_idx+1] is output of layer layer_idx
            h = hidden_states[layer_idx + 1][0, ts:te, :].float()
            proj = (h @ dir_vecs[layer_idx]).cpu().numpy()
            projections[layer_idx] = proj

    return projections


def smooth(arr, window=50):
    """Rolling mean with edge-aware padding (no boundary decay)."""
    if len(arr) < window:
        return arr
    # Pad with edge values to avoid boundary artifacts
    pad = window // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones(window) / window
    result = np.convolve(padded, kernel, mode="same")
    return result[pad:pad + len(arr)]


def clip_percentile(arr, lo=2, hi=98):
    """Clip to percentile range to remove outlier spikes."""
    low_val = np.percentile(arr, lo)
    high_val = np.percentile(arr, hi)
    return np.clip(arr, low_val, high_val)


def plot_traces(all_results, layers, output_dir, meta_df):
    """Plot projection curves for each test trace."""
    os.makedirs(output_dir, exist_ok=True)

    # Color map for layers 18-26
    cmap = plt.cm.viridis
    colors = {l: cmap(i / (len(layers) - 1)) for i, l in enumerate(layers)}

    # ------------------------------------------------------------------
    # Per-trace figure: one subplot per trace, all layers overlaid
    # Raw + smoothed, clipped to remove spikes
    # ------------------------------------------------------------------
    n_traces = len(all_results)
    fig, axes = plt.subplots(
        n_traces, 1, figsize=(14, 4 * n_traces), squeeze=False,
    )

    for i, result in enumerate(all_results):
        ax = axes[i, 0]
        pid = result["problem_id"]
        ds = result["dataset"]
        expected = result["expected_token"]
        gt = result["ground_truth"]
        think_length = result["think_length"]
        projections = result["projections"]
        in_train = pid in meta_df["problem_id"].values

        x_frac = np.arange(think_length) / think_length

        for layer_idx in layers:
            proj = projections[layer_idx]
            proj_smooth = smooth(proj, window=max(20, think_length // 50))

            # Light raw, bold smoothed
            ax.plot(
                x_frac, proj,
                color=colors[layer_idx], alpha=0.1, linewidth=0.3,
            )
            ax.plot(
                x_frac, proj_smooth,
                color=colors[layer_idx], alpha=0.9, linewidth=1.8,
                label=f"Layer {layer_idx}",
            )

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_title(
            f"{pid} ({ds}) | expected={expected} gt={gt} | "
            f"think_len={think_length} | train={'yes' if in_train else 'no'}",
            fontsize=10,
        )
        ax.set_xlabel("Fraction of thinking")
        ax.set_ylabel("Projection (clipped + smoothed)")
        if i == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "probe_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # Per-layer figure: all traces overlaid (smoothed + clipped)
    # ------------------------------------------------------------------
    for layer_idx in layers:
        fig, ax = plt.subplots(figsize=(12, 6))
        for result in all_results:
            proj = result["projections"][layer_idx]
            think_length = result["think_length"]
            x_frac = np.arange(think_length) / think_length
            proj_smooth = smooth(proj, window=max(20, think_length // 50))
            label = f"{result['problem_id'][:30]} ({result['expected_token']})"
            ax.plot(x_frac, proj_smooth, alpha=0.7, linewidth=1.2, label=label)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Layer {layer_idx} -- Knowing Direction Probe (smoothed)")
        ax.set_xlabel("Fraction of thinking")
        ax.set_ylabel("Projection score")
        ax.legend(fontsize=7, loc="upper left")
        plt.tight_layout()
        path = os.path.join(output_dir, f"probe_layer_{layer_idx}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def investigate_spikes(all_results, layers, tokenizer):
    """Find what tokens cause extreme projection spikes."""
    print("\n--- Spike Investigation ---")
    for layer_idx in [24]:  # focus on layer 24
        print(f"\nLayer {layer_idx} -- top 5 spike tokens per trace:")
        all_spike_tokens = {}
        for result in all_results:
            proj = result["projections"][layer_idx]
            token_ids = result["token_ids"]
            ts = result["think_start"] + 1
            think_length = result["think_length"]

            # Find positions with highest absolute projection
            abs_proj = np.abs(proj)
            top_idx = np.argsort(abs_proj)[-5:][::-1]

            pid = result["problem_id"]
            print(f"  {pid}:")
            for idx in top_idx:
                global_pos = ts + idx
                tid = token_ids[global_pos]
                tok_text = tokenizer.decode([tid])
                # Also get context (prev + next token)
                prev_tid = token_ids[global_pos - 1] if global_pos > 0 else -1
                next_tid = token_ids[global_pos + 1] if global_pos + 1 < len(token_ids) else -1
                prev_text = tokenizer.decode([prev_tid]) if prev_tid >= 0 else ""
                next_text = tokenizer.decode([next_tid]) if next_tid >= 0 else ""
                print(f"    pos={idx}/{think_length} proj={proj[idx]:.1f} "
                      f"token={tid} [{repr(tok_text)}] "
                      f"context: ...{repr(prev_text)}[{repr(tok_text)}]{repr(next_text)}...")
                all_spike_tokens[tid] = all_spike_tokens.get(tid, 0) + 1

        print(f"\n  Most common spike tokens across all traces:")
        for tid, count in sorted(all_spike_tokens.items(), key=lambda x: -x[1])[:10]:
            tok_text = tokenizer.decode([tid])
            print(f"    token={tid} [{repr(tok_text)}] appeared {count} times")


def main():
    print("Loading direction vectors...")
    direction_data = load_file(DIRECTION_PATH)
    direction_normalized = direction_data["direction_normalized"].numpy()

    print("Loading metadata...")
    meta_df = pd.read_parquet(META_PATH)

    print("Selecting test traces...")
    test_traces = load_test_traces(
        meta_df, TRACES_DIR, N_TEST_TRACES, TEST_TRACE_INDEX, SEED,
    )
    print(f"Selected {len(test_traces)} test traces:")
    for t in test_traces:
        print(f"  {t['problem_id']} ({t['dataset']}) expected={t['expected_token']} think_len={t['think_length']}")

    print("\nLoading model...")
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    print(f"\nProbing {len(test_traces)} traces at layers {PROBE_LAYERS}...")
    all_results = []
    for i, trace_data in enumerate(test_traces):
        print(f"  [{i+1}/{len(test_traces)}] {trace_data['problem_id']}...", flush=True)
        projections = probe_trace(
            model, trace_data, direction_normalized, PROBE_LAYERS, device,
        )
        all_results.append({
            **trace_data,
            "projections": projections,
        })
        torch.cuda.empty_cache()

    print("\nInvestigating spikes...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    investigate_spikes(all_results, PROBE_LAYERS, tokenizer)

    print("\nPlotting...")
    plot_traces(all_results, PROBE_LAYERS, OUTPUT_DIR, meta_df)
    print("Done.")


if __name__ == "__main__":
    main()
