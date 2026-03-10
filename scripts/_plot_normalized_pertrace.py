"""Per-trace plots with normalized activations.

Two normalization strategies:
1. L2-norm: normalize activation to unit length before probe (keeps bias)
2. Z-score: standardize using end-of-trace training stats (mean/std per dim)

Both address the distribution shift between end-of-trace and intermediate positions.
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
PLOT_DIR = os.path.join(OUTPUT_DIR, "normalized_plots")
WINDOW_SIZE = 3
GPU_ID = 6
N_TRACES_PER_DS = 5

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


def extract_training_stats(model, traces, layer):
    """Extract end-of-trace activations and compute per-dim mean/std."""
    all_acts = []
    for i, trace in enumerate(traces):
        window = get_pre_handoff_window(trace["token_ids"], WINDOW_SIZE)
        if window is None:
            continue
        start_pos, end_pos = window
        input_ids = torch.tensor([trace["token_ids"]], dtype=torch.long, device=device)
        with model.trace(input_ids):
            saved = model.model.layers[layer].output.save()
        tensor = saved.detach().cpu()
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        all_acts.append(tensor[start_pos:end_pos].float())
        del saved, input_ids
        torch.cuda.empty_cache()
        if (i + 1) % 100 == 0 or (i + 1) == len(traces):
            print(f"    Training stats: {i+1}/{len(traces)}", flush=True)
    X = torch.cat(all_acts, 0)  # [N, 4096]
    mean = X.mean(dim=0)  # [4096]
    std = X.std(dim=0).clamp(min=1e-8)  # [4096]
    print(f"    Training stats: mean_norm={mean.norm():.2f}, "
          f"std_mean={std.mean():.4f}, std_min={std.min():.4f}", flush=True)
    return mean, std


def main():
    print("Discovering traces...", flush=True)
    per_ds = discover_traces()
    all_traces = []
    for ds in DATASETS:
        print(f"  {ds}: {len(per_ds[ds])} traces", flush=True)
        all_traces.extend(per_ds[ds])

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

    # Only do best configs to save time
    configs_to_plot = [
        (20, 0.01), (20, 0.1), (20, 0.0001),
        (16, 0.1), (16, 0.0001),
        (8, 0.01),
    ]

    for layer, wd in configs_to_plot:
        pf = os.path.join(OUTPUT_DIR, f"probe_L{layer}_wd{wd}.safetensors")
        if not os.path.exists(pf):
            print(f"  Skipping L{layer} wd={wd} (no probe file)", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"Layer {layer}, wd={wd}", flush=True)
        print(f"{'='*60}", flush=True)

        # Load probe
        tensors = load_file(pf)
        w_key = f"layer_{layer}_wd_{wd}_weight"
        b_key = f"layer_{layer}_wd_{wd}_bias"
        probe = nn.Linear(HIDDEN_DIM, 2).to(device)
        probe.weight.data = tensors[w_key].to(device)
        probe.bias.data = tensors[b_key].to(device)
        probe.eval()

        # Compute training stats for z-scoring
        print("  Computing training stats...", flush=True)
        train_mean, train_std = extract_training_stats(model, all_traces, layer)
        train_mean = train_mean.to(device)
        train_std = train_std.to(device)

        # Compute per-trace curves with 3 methods
        all_curves = []
        for i, trace in enumerate(eval_traces):
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

            npos = []
            raw_vals = []      # original probe
            l2_vals = []       # L2-normalized activations
            zscore_vals = []   # z-scored activations

            with torch.no_grad():
                for pos in eligible:
                    h = res[pos, :].float().to(device)

                    # 1. Raw probe (original)
                    p_yes_raw = F.softmax(probe(h.unsqueeze(0)), dim=-1)[0, 1].item()

                    # 2. L2-normalized
                    h_l2 = F.normalize(h.unsqueeze(0), dim=-1)
                    p_yes_l2 = F.softmax(probe(h_l2), dim=-1)[0, 1].item()

                    # 3. Z-scored
                    h_z = (h - train_mean) / train_std
                    p_yes_z = F.softmax(probe(h_z.unsqueeze(0)), dim=-1)[0, 1].item()

                    # Convert to P(correct)
                    if gt == "Yes":
                        raw_vals.append(p_yes_raw)
                        l2_vals.append(p_yes_l2)
                        zscore_vals.append(p_yes_z)
                    else:
                        raw_vals.append(1.0 - p_yes_raw)
                        l2_vals.append(1.0 - p_yes_l2)
                        zscore_vals.append(1.0 - p_yes_z)
                    npos.append((pos - start) / span)

            del saved, input_ids, res
            torch.cuda.empty_cache()

            all_curves.append({
                "problem_id": trace["problem_id"],
                "dataset": trace["dataset"],
                "ground_truth": gt,
                "norm_positions": np.array(npos),
                "raw": np.array(raw_vals),
                "l2_norm": np.array(l2_vals),
                "zscore": np.array(zscore_vals),
            })
            print(f"    Trace {i+1}/{len(eval_traces)}: {trace['problem_id']}", flush=True)

        # Plot: 3 rows (datasets) x 5 cols, each subplot has 3 lines
        by_ds = {}
        for c in all_curves:
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

                for method, color, label in [
                    ("raw", "#e74c3c", "raw"),
                    ("l2_norm", "#3498db", "L2-norm"),
                    ("zscore", "#2ecc71", "z-score"),
                ]:
                    vals = c[method]
                    w_size = min(50, len(vals) // 5)
                    if w_size >= 3:
                        sm = np.convolve(vals, np.ones(w_size) / w_size, mode="valid")
                        sx = npos[w_size // 2:w_size // 2 + len(sm)]
                        ax.plot(sx, sm, lw=1.8, color=color, label=label, alpha=0.8)

                ax.axhline(0.5, color="gray", ls=":", lw=0.6, alpha=0.5)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(0, 1)
                ax.set_title(f"{c['problem_id'][:20]}\nGT={c['ground_truth']}", fontsize=7)
                ax.grid(True, alpha=0.2)
                if col == 0:
                    ax.set_ylabel(ds, fontsize=9, fontweight="bold")
                if row == 0 and col == 0:
                    ax.legend(fontsize=6, loc="upper left")
                if row == n_rows - 1:
                    ax.set_xlabel("Norm position", fontsize=7)

        fig.suptitle(f"NORMALIZED -- Layer {layer}, wd={wd}\n"
                     f"Red=raw probe, Blue=L2-norm, Green=z-score", fontsize=12, y=1.02)
        plt.tight_layout()
        outpath = os.path.join(PLOT_DIR, f"normalized_L{layer}_wd{wd}.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
