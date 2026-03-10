"""Generate per-trace P(correct) plots for ALL (layer, wd) configs.

Loads saved probe weights from outputs/multilayer_probe/probe_L*_wd*.safetensors.
Runs on GPU 6 (free) so it can run alongside the main sweep.

For each config, generates a grid of 5 per-trace curves from each dataset (15 total).
"""

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

from src.model import HIDDEN_DIM, load_model
from src.tokens import count_think_blocks, get_pre_handoff_window, get_think_content_mask
from src.utils import set_seed

set_seed(137)

DATASETS = ["gsm_symbolic", "strategyqa", "folio", "deepmath"]
TRACE_BASE = "src/experimental/gorn/data/traces/valid"
OUTPUT_DIR = "outputs/multilayer_probe"
PLOT_DIR = os.path.join(OUTPUT_DIR, "pertrace_plots")
WINDOW_SIZE = 3
GPU_ID = 6
N_TRACES_PER_DS = 5  # 5 per dataset = 15 per config

os.makedirs(PLOT_DIR, exist_ok=True)
device = torch.device(f"cuda:{GPU_ID}")


def determine_correctness(metadata, generated_text):
    gt = metadata.get("ground_truth")
    vt = metadata.get("valid_token")
    it = metadata.get("invalid_token")
    if gt is None or vt is None or it is None:
        return None
    idx = generated_text.find("</think>")
    if idx < 0:
        return None
    response = generated_text[idx + 8:].strip()
    expected = vt if gt == "Yes" else it
    other = it if gt == "Yes" else vt
    if expected in response and other not in response:
        return True
    if other in response and expected not in response:
        return False
    return None


def discover_traces():
    per_ds = {}
    for ds in DATASETS:
        trace_dir = os.path.join(TRACE_BASE, ds)
        if not os.path.isdir(trace_dir):
            continue
        dirs = sorted(d for d in os.listdir(trace_dir)
                      if os.path.isdir(os.path.join(trace_dir, d)))
        ds_traces = []
        for pdir in dirs:
            tp = os.path.join(trace_dir, pdir, "trace_00.json")
            if not os.path.isfile(tp):
                continue
            with open(tp) as f:
                data = json.load(f)
            tids = data["token_ids"]
            meta = data.get("metadata", {})
            gen = data.get("generated_text", "")
            if count_think_blocks(tids) != 1:
                continue
            if get_pre_handoff_window(tids, WINDOW_SIZE) is None:
                continue
            ic = determine_correctness(meta, gen)
            if ic is None or not ic:
                continue
            mask = get_think_content_mask(tids)
            eligible = [i for i, m in enumerate(mask) if m]
            if len(eligible) < 10:
                continue
            ds_traces.append({
                "token_ids": tids, "ground_truth": meta["ground_truth"],
                "problem_id": pdir, "dataset": ds,
                "eligible_positions": eligible,
            })
        per_ds[ds] = ds_traces
    return per_ds


def compute_pertrace_curves(model, probe, traces, layer):
    """Compute P(correct) at every eligible position for each trace."""
    results = []
    for i, trace in enumerate(traces):
        tids = trace["token_ids"]
        eligible = trace["eligible_positions"]
        gt = trace["ground_truth"]
        start = eligible[0]
        span = eligible[-1] - start if eligible[-1] > start else 1

        input_ids = torch.tensor([tids], dtype=torch.long, device=device)
        with model.trace(input_ids):
            saved = model.model.layers[layer].output.save()
        res = saved.detach()
        if res.dim() == 3:
            res = res.squeeze(0)

        npos, pvals = [], []
        with torch.no_grad():
            for pos in eligible:
                h = res[pos, :].float().unsqueeze(0).to(device)
                p_yes = F.softmax(probe(h), dim=-1)[0, 1].item()
                pc = p_yes if gt == "Yes" else (1.0 - p_yes)
                npos.append((pos - start) / span)
                pvals.append(pc)
        del saved, input_ids, res
        torch.cuda.empty_cache()

        results.append({
            "problem_id": trace["problem_id"],
            "dataset": trace["dataset"],
            "ground_truth": gt,
            "norm_positions": np.array(npos),
            "p_correct_vals": np.array(pvals),
        })
        print(f"    Trace {i+1}/{len(traces)}: {trace['problem_id']}", flush=True)
    return results


def plot_config(layer, wd, curves, outpath):
    """Plot per-trace curves for one (layer, wd) config.

    Layout: 3 rows (one per dataset) x 5 columns (traces).
    """
    by_ds = {}
    for c in curves:
        by_ds.setdefault(c["dataset"], []).append(c)

    n_cols = N_TRACES_PER_DS
    n_rows = len(DATASETS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, ds in enumerate(DATASETS):
        ds_curves = by_ds.get(ds, [])
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(ds_curves):
                ax.set_visible(False)
                continue

            c = ds_curves[col]
            npos = c["norm_positions"]
            pvals = c["p_correct_vals"]

            # Raw
            ax.plot(npos, pvals, lw=0.3, color="#bdc3c7", alpha=0.5)

            # Smoothed
            w = min(50, len(pvals) // 5)
            if w >= 3:
                sm = np.convolve(pvals, np.ones(w) / w, mode="valid")
                sx = npos[w // 2:w // 2 + len(sm)]
                ax.plot(sx, sm, lw=2.0, color="#2c3e50")

            ax.axhline(0.5, color="gray", ls=":", lw=0.6, alpha=0.5)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(0, 1)
            ax.set_title(f"{c['problem_id'][:20]}\nGT={c['ground_truth']}", fontsize=7)
            ax.grid(True, alpha=0.2)

            if col == 0:
                ax.set_ylabel(ds, fontsize=9, fontweight="bold")
            if row == n_rows - 1:
                ax.set_xlabel("Norm position", fontsize=7)

    fig.suptitle(f"Layer {layer}, wd={wd} -- Per-trace P(correct)", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}", flush=True)


def main():
    print("Discovering traces...", flush=True)
    per_ds = discover_traces()
    for ds in DATASETS:
        print(f"  {ds}: {len(per_ds[ds])} traces", flush=True)

    # Select eval traces: first N_TRACES_PER_DS from each dataset (deterministic)
    rng = np.random.RandomState(137)
    eval_traces = []
    for ds in DATASETS:
        dst = per_ds[ds]
        if len(dst) > N_TRACES_PER_DS:
            idx = rng.choice(len(dst), N_TRACES_PER_DS, replace=False)
            eval_traces.extend([dst[i] for i in sorted(idx)])
        else:
            eval_traces.extend(dst)

    print(f"Eval traces: {len(eval_traces)}", flush=True)

    print(f"Loading model on GPU {GPU_ID}...", flush=True)
    model, tok = load_model(gpu_id=GPU_ID)
    print("Model loaded.", flush=True)

    # Find all saved probe files
    probe_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "probe_L*_wd*.safetensors")))
    print(f"\nFound {len(probe_files)} probe files", flush=True)

    for pf in probe_files:
        # Parse layer and wd from filename
        basename = os.path.basename(pf)
        m = re.match(r"probe_L(\d+)_wd([\d.e-]+)\.safetensors", basename)
        if not m:
            print(f"  Skipping {basename} (cannot parse)", flush=True)
            continue
        layer = int(m.group(1))
        wd = float(m.group(2))

        outpath = os.path.join(PLOT_DIR, f"pertrace_L{layer}_wd{wd}.png")
        if os.path.exists(outpath):
            print(f"  Skipping L{layer} wd={wd} (already plotted)", flush=True)
            continue

        print(f"\n--- Layer {layer}, wd={wd} ---", flush=True)

        # Load probe weights
        tensors = load_file(pf)
        probe = nn.Linear(HIDDEN_DIM, 2).to(device)
        w_key = f"layer_{layer}_wd_{wd}_weight"
        b_key = f"layer_{layer}_wd_{wd}_bias"
        probe.weight.data = tensors[w_key].to(device)
        probe.bias.data = tensors[b_key].to(device)
        probe.eval()

        # Compute curves
        curves = compute_pertrace_curves(model, probe, eval_traces, layer)

        # Plot
        plot_config(layer, wd, curves, outpath)

    print("\nDone -- all configs plotted.", flush=True)


if __name__ == "__main__":
    main()
