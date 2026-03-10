"""Windowed probe: average activations over K-token windows before probing.

Key insight: single-token activations are dominated by token-specific info
(current word, position encoding, local attention). The "answer knowledge"
signal is a tiny component that gets overwhelmed by per-token noise.

By averaging over a window of K tokens, the token-specific noise averages out
while the "answer knowledge" component (which should be consistent across
nearby tokens) is preserved.

This is analogous to how force-answer experiments show clean curves: they
probe the model's actual behavior (which integrates over many tokens via
attention), not a single token's activation.

Approach:
1. Train probe on end-of-trace window (same as before)
2. At eval time, average activations over sliding windows of size K
3. Apply probe to each window's average activation
4. Compare per-trace curves at different window sizes

Uses existing probe weights from bigdata experiment.
"""

import gc
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score

from src.model import HIDDEN_DIM, load_model
from src.tokens import count_think_blocks, get_pre_handoff_window, get_think_content_mask
from src.utils import set_seed

set_seed(137)

DATASETS = ["gsm_symbolic", "strategyqa", "folio", "deepmath"]
TRACE_BASE = "src/experimental/gorn/data/traces/valid"
OUTPUT_DIR = "outputs/multilayer_probe/windowed_probe"
LAYER = 20
PROBE_WD = 0.01  # best from bigdata sweep
WINDOW_SIZE_TRAIN = 3
SEED = 137
GPU_ID = 2  # separate GPU
N_EVAL_PER_DS = 8

# Window sizes for sliding average at eval time
EVAL_WINDOWS = [1, 5, 10, 25, 50, 100]

POSITION_BINS = [
    ("early", 0.0, 0.2),
    ("early_mid", 0.2, 0.4),
    ("middle", 0.4, 0.6),
    ("late_mid", 0.6, 0.8),
    ("late", 0.8, 1.0),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
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


def discover_all_traces():
    all_traces = []
    per_ds_count = {}
    for ds in DATASETS:
        trace_dir = os.path.join(TRACE_BASE, ds)
        if not os.path.isdir(trace_dir):
            continue
        dirs = sorted(d for d in os.listdir(trace_dir)
                      if os.path.isdir(os.path.join(trace_dir, d)))
        ds_count = 0
        for pdir in dirs:
            ppath = os.path.join(trace_dir, pdir)
            trace_files = sorted(f for f in os.listdir(ppath)
                                 if f.startswith("trace_") and f.endswith(".json"))
            for tf in trace_files:
                tp = os.path.join(ppath, tf)
                try:
                    with open(tp) as f:
                        data = json.load(f)
                except Exception:
                    continue
                tids = data["token_ids"]
                meta = data.get("metadata", {})
                gen = data.get("generated_text", "")
                if count_think_blocks(tids) != 1:
                    continue
                if get_pre_handoff_window(tids, WINDOW_SIZE_TRAIN) is None:
                    continue
                ic = determine_correctness(meta, gen)
                if ic is None or not ic:
                    continue
                mask = get_think_content_mask(tids)
                eligible = [i for i, m in enumerate(mask) if m]
                if len(eligible) < 10:
                    continue
                trace_idx = tf.replace("trace_", "").replace(".json", "")
                all_traces.append({
                    "token_ids": tids,
                    "ground_truth": meta["ground_truth"],
                    "problem_id": pdir,
                    "trace_idx": trace_idx,
                    "dataset": ds,
                    "eligible_positions": eligible,
                })
                ds_count += 1
        per_ds_count[ds] = ds_count
    return all_traces, per_ds_count


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("WINDOWED PROBE -- Sliding average over K tokens", flush=True)
    print(f"Layer {LAYER}, window sizes: {EVAL_WINDOWS}", flush=True)
    print("=" * 70, flush=True)

    print("\nDiscovering ALL traces...", flush=True)
    all_traces, per_ds_count = discover_all_traces()
    for ds, c in per_ds_count.items():
        print(f"  {ds}: {c} traces", flush=True)
    print(f"Total: {len(all_traces)} traces", flush=True)

    # Split by problem_id (same split as bigdata probe)
    rng = np.random.RandomState(SEED)
    all_pids = sorted(set(t["problem_id"] for t in all_traces))
    rng.shuffle(all_pids)
    n_eval_pids = max(len(all_pids) // 5, 10)
    eval_pids = set(all_pids[:n_eval_pids])
    train_traces = [t for t in all_traces if t["problem_id"] not in eval_pids]
    eval_traces = [t for t in all_traces if t["problem_id"] in eval_pids]
    print(f"Train: {len(train_traces)}, Eval: {len(eval_traces)}", flush=True)

    # Train a new probe (or load existing)
    probe_path = os.path.join("outputs/multilayer_probe/bigdata",
                              f"bigdata_L{LAYER}_wd{PROBE_WD}.safetensors")
    print(f"\nLoading model on GPU {GPU_ID}...", flush=True)
    model, tok = load_model(gpu_id=GPU_ID)

    if os.path.exists(probe_path):
        print(f"Loading existing probe from {probe_path}", flush=True)
        tensors = load_file(probe_path)
        probe = nn.Linear(HIDDEN_DIM, 2).to(device)
        probe.weight.data = tensors["weight"].to(device)
        probe.bias.data = tensors["bias"].to(device)
        probe.eval()
    else:
        print("ERROR: No bigdata probe found. Run _bigdata_probe.py first.", flush=True)
        return

    # Select eval traces for detailed plotting
    eval_subset = []
    for ds in DATASETS:
        dst = [t for t in eval_traces if t["dataset"] == ds]
        eval_subset.extend(dst[:N_EVAL_PER_DS])
    print(f"Eval subset for plots: {len(eval_subset)} traces", flush=True)

    # For each window size, compute positional AUROC and per-trace curves
    results = {}
    for win_k in EVAL_WINDOWS:
        print(f"\n{'='*60}", flush=True)
        print(f"WINDOW SIZE K={win_k}", flush=True)
        print(f"{'='*60}", flush=True)

        # Positional AUROC
        print(f"  Computing positional AUROC on {len(eval_traces)} eval traces...", flush=True)
        all_rows = []
        with torch.no_grad():
            for i, trace in enumerate(eval_traces):
                tids = trace["token_ids"]
                eligible = trace["eligible_positions"]
                label = 1 if trace["ground_truth"] == "Yes" else 0
                start = eligible[0]
                span = eligible[-1] - start if eligible[-1] > start else 1

                input_ids = torch.tensor([tids], dtype=torch.long, device=device)
                with model.trace(input_ids):
                    saved = model.model.layers[LAYER].output.save()
                res = saved.detach()
                if res.dim() == 3: res = res.squeeze(0)

                # Compute windowed averages
                # For each position, average the activations in a window centered on it
                step = max(1, len(eligible) // 50)
                sampled_indices = list(range(0, len(eligible), step))
                for idx in sampled_indices:
                    # Window: [idx - win_k//2, idx + win_k//2]
                    lo = max(0, idx - win_k // 2)
                    hi = min(len(eligible), idx + win_k // 2 + 1)
                    window_positions = eligible[lo:hi]
                    if len(window_positions) == 0:
                        continue
                    # Average activations over window
                    h_avg = res[window_positions, :].float().mean(dim=0, keepdim=True)
                    p_yes = F.softmax(probe(h_avg), dim=-1)[0, 1].item()
                    pos = eligible[idx]
                    all_rows.append(((pos - start) / span, p_yes, label))

                del saved, input_ids, res; torch.cuda.empty_cache()
                if (i + 1) % 100 == 0 or (i + 1) == len(eval_traces):
                    print(f"    PosAUROC: {i+1}/{len(eval_traces)}", flush=True)
                if (i + 1) % 200 == 0:
                    gc.collect()

        pos_aurocs = {}
        for bn, lo, hi in POSITION_BINS:
            rows = [(p, l) for (np_, p, l) in all_rows if lo <= np_ < hi]
            if len(rows) < 10:
                pos_aurocs[bn] = None; continue
            pa = np.array([r[0] for r in rows])
            la = np.array([r[1] for r in rows])
            pos_aurocs[bn] = float(roc_auc_score(la, pa)) if len(np.unique(la)) >= 2 else 0.5
        early = pos_aurocs.get("early", 0.5) or 0.5
        late = pos_aurocs.get("late", 0.5) or 0.5
        ramp = late - early
        pos_str = " | ".join(f"{k}={v:.4f}" if v else f"{k}=N/A"
                             for k, v in pos_aurocs.items())
        print(f"  Positional: {pos_str}  RAMP={ramp:.4f}", flush=True)

        # Per-trace curves
        curves = []
        with torch.no_grad():
            for trace in eval_subset:
                tids = trace["token_ids"]
                eligible = trace["eligible_positions"]
                gt = trace["ground_truth"]
                start = eligible[0]
                span = eligible[-1] - start if eligible[-1] > start else 1

                input_ids = torch.tensor([tids], dtype=torch.long, device=device)
                with model.trace(input_ids):
                    saved = model.model.layers[LAYER].output.save()
                res = saved.detach()
                if res.dim() == 3: res = res.squeeze(0)

                npos, pvals = [], []
                for idx in range(len(eligible)):
                    lo = max(0, idx - win_k // 2)
                    hi = min(len(eligible), idx + win_k // 2 + 1)
                    window_positions = eligible[lo:hi]
                    h_avg = res[window_positions, :].float().mean(dim=0, keepdim=True)
                    p_yes = F.softmax(probe(h_avg), dim=-1)[0, 1].item()
                    pc = p_yes if gt == "Yes" else (1.0 - p_yes)
                    pos = eligible[idx]
                    npos.append((pos - start) / span)
                    pvals.append(pc)

                del saved, input_ids, res; torch.cuda.empty_cache()
                curves.append({
                    "problem_id": trace["problem_id"], "dataset": trace["dataset"],
                    "ground_truth": gt, "trace_idx": trace["trace_idx"],
                    "norm_positions": np.array(npos), "p_correct": np.array(pvals),
                })

        # Plot
        by_ds = {}
        for c in curves:
            by_ds.setdefault(c["dataset"], []).append(c)
        n_cols = N_EVAL_PER_DS
        n_rows = len(DATASETS)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.5*n_rows))
        if n_rows == 1: axes = axes[np.newaxis, :]
        for row, ds in enumerate(DATASETS):
            ds_curves = by_ds.get(ds, [])
            for col in range(n_cols):
                ax = axes[row, col]
                if col >= len(ds_curves):
                    ax.set_visible(False); continue
                c = ds_curves[col]
                npos = c["norm_positions"]
                pvals = c["p_correct"]
                # Raw curve (already smoothed by windowing)
                ax.plot(npos, pvals, lw=1.5, color="#2c3e50", alpha=0.8)
                ax.axhline(0.5, color="gray", ls=":", lw=0.6, alpha=0.5)
                ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, 1)
                ax.set_title(f"{c['problem_id'][:18]}_{c['trace_idx']}\nGT={c['ground_truth']}",
                             fontsize=6)
                ax.grid(True, alpha=0.2)
                if col == 0: ax.set_ylabel(ds, fontsize=9, fontweight="bold")
        fig.suptitle(f"WINDOWED PROBE K={win_k} -- L{LAYER}\n"
                     f"RAMP={ramp:.4f}", fontsize=12, y=1.02)
        plt.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, "plots", f"windowed_K{win_k}.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath}", flush=True)

        results[win_k] = {
            "positional_auroc": pos_aurocs, "ramp": ramp,
        }

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY -- WINDOWED PROBE", flush=True)
    print("=" * 70, flush=True)
    for k in EVAL_WINDOWS:
        r = results[k]
        print(f"  K={k:3d}: ramp={r['ramp']:.4f}", flush=True)

    # Comparison plot: all window sizes on same traces
    print("\nGenerating comparison plot...", flush=True)
    # Re-generate curves for comparison (just first 3 per dataset)
    comp_traces = []
    for ds in DATASETS:
        dst = [t for t in eval_traces if t["dataset"] == ds]
        comp_traces.extend(dst[:3])

    fig, axes = plt.subplots(len(DATASETS), 3, figsize=(14, 3.5*len(DATASETS)))
    if len(DATASETS) == 1: axes = axes[np.newaxis, :]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(EVAL_WINDOWS)))

    with torch.no_grad():
        by_ds_comp = {}
        for trace in comp_traces:
            by_ds_comp.setdefault(trace["dataset"], []).append(trace)

        for row, ds in enumerate(DATASETS):
            ds_traces = by_ds_comp.get(ds, [])
            for col, trace in enumerate(ds_traces[:3]):
                ax = axes[row, col]
                tids = trace["token_ids"]
                eligible = trace["eligible_positions"]
                gt = trace["ground_truth"]
                start = eligible[0]
                span = eligible[-1] - start if eligible[-1] > start else 1

                input_ids = torch.tensor([tids], dtype=torch.long, device=device)
                with model.trace(input_ids):
                    saved = model.model.layers[LAYER].output.save()
                res = saved.detach()
                if res.dim() == 3: res = res.squeeze(0)

                for wi, win_k in enumerate(EVAL_WINDOWS):
                    npos, pvals = [], []
                    for idx in range(len(eligible)):
                        lo = max(0, idx - win_k // 2)
                        hi = min(len(eligible), idx + win_k // 2 + 1)
                        window_positions = eligible[lo:hi]
                        h_avg = res[window_positions, :].float().mean(dim=0, keepdim=True)
                        p_yes = F.softmax(probe(h_avg), dim=-1)[0, 1].item()
                        pc = p_yes if gt == "Yes" else (1.0 - p_yes)
                        pos = eligible[idx]
                        npos.append((pos - start) / span)
                        pvals.append(pc)
                    lw = 0.5 if win_k <= 5 else 1.5
                    alpha = 0.3 if win_k <= 5 else 0.8
                    ax.plot(npos, pvals, lw=lw, color=colors[wi], alpha=alpha,
                            label=f"K={win_k}")

                del saved, input_ids, res; torch.cuda.empty_cache()
                ax.axhline(0.5, color="gray", ls=":", lw=0.6, alpha=0.5)
                ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, 1)
                ax.set_title(f"{trace['problem_id'][:18]}_{trace['trace_idx']}\nGT={gt}",
                             fontsize=7)
                ax.grid(True, alpha=0.2)
                if col == 0: ax.set_ylabel(ds, fontsize=9, fontweight="bold")
                if row == 0 and col == 0: ax.legend(fontsize=5, loc="upper left")

    fig.suptitle("WINDOW SIZE COMPARISON\n"
                 "Thin=K1,K5 | Thick=K10-K100", fontsize=12, y=1.02)
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "plots", "window_comparison.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
