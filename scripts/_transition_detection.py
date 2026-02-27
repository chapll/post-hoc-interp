"""Transition detection using windowed probe.

Uses the windowed probe (K=50 sliding average) to detect the three-phase
structure in reasoning traces:
1. Proper reasoning: start to first inflection point (probe P(correct) low/rising)
2. Transition: from first inflection through fluctuations to plateau onset
3. Post-hoc reasoning: plateau where probe confidence stays high

Detection method:
- Apply windowed probe (K=50 average) to get smooth P(correct) curve
- Find transition point T_i as first position where P(correct) stays above
  threshold for the remainder of the trace (plateau detection)
- Compute statistics across traces: distribution of T_i, fraction of trace
  that is "post-hoc", etc.

Uses existing bigdata probe weights + windowed averaging at eval.
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

DATASETS = ["gsm_symbolic", "strategyqa", "folio"]
TRACE_BASE = "src/experimental/gorn/data/traces/valid"
OUTPUT_DIR = "outputs/multilayer_probe/transition_detection"
LAYER = 20
PROBE_WD = 0.01
WINDOW_SIZE_TRAIN = 3
SEED = 137
GPU_ID = 0
EVAL_WINDOW = 50  # Best tradeoff from windowed probe experiment

# Transition detection parameters
PLATEAU_THRESHOLD = 0.6    # P(correct) must exceed this
PLATEAU_FRACTION = 0.8     # Must stay above threshold for this fraction of remaining trace
MIN_TRACE_LENGTH = 50      # Minimum think tokens for transition detection

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
    for ds in DATASETS:
        trace_dir = os.path.join(TRACE_BASE, ds)
        if not os.path.isdir(trace_dir):
            continue
        dirs = sorted(d for d in os.listdir(trace_dir)
                      if os.path.isdir(os.path.join(trace_dir, d)))
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
                if len(eligible) < MIN_TRACE_LENGTH:
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
    return all_traces


def detect_transition(p_correct, norm_positions, threshold=PLATEAU_THRESHOLD,
                      plateau_frac=PLATEAU_FRACTION):
    """Detect transition point in a P(correct) curve.

    Returns:
        transition_idx: index into p_correct where transition detected (None if no transition)
        transition_pos: normalized position (0-1) of transition
        phases: dict with 'reasoning', 'transition', 'posthoc' position ranges
    """
    n = len(p_correct)
    if n < 10:
        return None, None, None

    # Scan from start: find first position where P(correct) stays above threshold
    # for at least plateau_frac of the remaining trace
    for i in range(n):
        remaining = p_correct[i:]
        above = np.sum(remaining > threshold)
        if len(remaining) > 0 and above / len(remaining) >= plateau_frac:
            # Found plateau onset
            transition_pos = norm_positions[i]
            # Classify phases
            phases = {
                "reasoning_end": transition_pos,
                "posthoc_start": transition_pos,
                "posthoc_frac": 1.0 - transition_pos,
            }
            return i, transition_pos, phases

    # No clear transition -- entire trace is "reasoning"
    return None, None, {"reasoning_end": 1.0, "posthoc_start": None, "posthoc_frac": 0.0}


def main():
    t0 = time.time()
    print("=" * 70, flush=True)
    print("TRANSITION DETECTION", flush=True)
    print(f"Layer {LAYER}, window K={EVAL_WINDOW}", flush=True)
    print(f"Plateau: threshold={PLATEAU_THRESHOLD}, fraction={PLATEAU_FRACTION}", flush=True)
    print("=" * 70, flush=True)

    print("\nDiscovering traces...", flush=True)
    all_traces = discover_all_traces()
    per_ds = {}
    for t in all_traces:
        per_ds.setdefault(t["dataset"], []).append(t)
    for ds, ts in per_ds.items():
        print(f"  {ds}: {len(ts)} traces", flush=True)
    print(f"Total: {len(all_traces)} traces", flush=True)

    # Load probe
    probe_path = f"outputs/multilayer_probe/bigdata/bigdata_L{LAYER}_wd{PROBE_WD}.safetensors"
    print(f"\nLoading model on GPU {GPU_ID}...", flush=True)
    model, tok = load_model(gpu_id=GPU_ID)

    tensors = load_file(probe_path)
    probe = nn.Linear(HIDDEN_DIM, 2).to(device)
    probe.weight.data = tensors["weight"].to(device)
    probe.bias.data = tensors["bias"].to(device)
    probe.eval()
    print("Probe loaded.", flush=True)

    # Process ALL traces with windowed probe
    print(f"\nComputing windowed P(correct) curves for all {len(all_traces)} traces...", flush=True)
    all_results = []
    with torch.no_grad():
        for i, trace in enumerate(all_traces):
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
                lo = max(0, idx - EVAL_WINDOW // 2)
                hi = min(len(eligible), idx + EVAL_WINDOW // 2 + 1)
                window_positions = eligible[lo:hi]
                h_avg = res[window_positions, :].float().mean(dim=0, keepdim=True)
                p_yes = F.softmax(probe(h_avg), dim=-1)[0, 1].item()
                pc = p_yes if gt == "Yes" else (1.0 - p_yes)
                pos = eligible[idx]
                npos.append((pos - start) / span)
                pvals.append(pc)

            del saved, input_ids, res; torch.cuda.empty_cache()

            npos = np.array(npos)
            pvals = np.array(pvals)

            # Detect transition
            t_idx, t_pos, phases = detect_transition(pvals, npos)

            all_results.append({
                "problem_id": trace["problem_id"],
                "trace_idx": trace["trace_idx"],
                "dataset": trace["dataset"],
                "ground_truth": gt,
                "n_think_tokens": len(eligible),
                "norm_positions": npos,
                "p_correct": pvals,
                "transition_idx": t_idx,
                "transition_pos": t_pos,
                "phases": phases,
            })

            if (i + 1) % 100 == 0 or (i + 1) == len(all_traces):
                n_detected = sum(1 for r in all_results if r["transition_pos"] is not None)
                print(f"  {i+1}/{len(all_traces)} ({n_detected} transitions detected)", flush=True)
            if (i + 1) % 200 == 0:
                gc.collect()

    # Compute statistics
    print(f"\n{'='*70}", flush=True)
    print("TRANSITION STATISTICS", flush=True)
    print(f"{'='*70}", flush=True)

    n_total = len(all_results)
    n_transition = sum(1 for r in all_results if r["transition_pos"] is not None)
    print(f"Total traces: {n_total}", flush=True)
    print(f"Transitions detected: {n_transition} ({n_transition/n_total*100:.1f}%)", flush=True)

    # Per-dataset statistics
    for ds in DATASETS:
        ds_results = [r for r in all_results if r["dataset"] == ds]
        ds_trans = [r for r in ds_results if r["transition_pos"] is not None]
        n_ds = len(ds_results)
        n_dt = len(ds_trans)
        if n_dt > 0:
            t_positions = [r["transition_pos"] for r in ds_trans]
            ph_fracs = [r["phases"]["posthoc_frac"] for r in ds_trans]
            print(f"\n  {ds} ({n_ds} traces, {n_dt} ({n_dt/n_ds*100:.1f}%) with transition):", flush=True)
            print(f"    Transition position: mean={np.mean(t_positions):.3f}, "
                  f"median={np.median(t_positions):.3f}, "
                  f"std={np.std(t_positions):.3f}", flush=True)
            print(f"    Post-hoc fraction: mean={np.mean(ph_fracs):.3f}, "
                  f"median={np.median(ph_fracs):.3f}", flush=True)
            print(f"    Transition position quartiles: "
                  f"Q1={np.percentile(t_positions, 25):.3f}, "
                  f"Q2={np.percentile(t_positions, 50):.3f}, "
                  f"Q3={np.percentile(t_positions, 75):.3f}", flush=True)
        else:
            print(f"\n  {ds} ({n_ds} traces, 0 transitions detected)", flush=True)

    # === PLOTS ===

    # 1. Histogram of transition positions
    trans_positions = [r["transition_pos"] for r in all_results if r["transition_pos"] is not None]
    if trans_positions:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Overall histogram
        axes[0].hist(trans_positions, bins=20, color="#2c3e50", alpha=0.7, edgecolor="white")
        axes[0].set_xlabel("Transition Position (normalized)")
        axes[0].set_ylabel("Count")
        axes[0].set_title(f"Distribution of Transition Points (n={len(trans_positions)})")
        axes[0].axvline(np.median(trans_positions), color="red", ls="--", lw=2,
                        label=f"Median={np.median(trans_positions):.3f}")
        axes[0].legend()

        # Per-dataset
        for ds in DATASETS:
            ds_trans = [r["transition_pos"] for r in all_results
                        if r["dataset"] == ds and r["transition_pos"] is not None]
            if ds_trans:
                axes[1].hist(ds_trans, bins=15, alpha=0.5, label=f"{ds} (n={len(ds_trans)})")
        axes[1].set_xlabel("Transition Position (normalized)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Transition Points by Dataset")
        axes[1].legend()

        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "plots", "transition_histogram.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("\nSaved: transition_histogram.png", flush=True)

    # 2. Post-hoc fraction distribution
    posthoc_fracs = [r["phases"]["posthoc_frac"] for r in all_results
                     if r["transition_pos"] is not None]
    if posthoc_fracs:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(posthoc_fracs, bins=20, color="#e74c3c", alpha=0.7, edgecolor="white")
        ax.set_xlabel("Fraction of Trace that is Post-Hoc Reasoning")
        ax.set_ylabel("Count")
        ax.set_title(f"Post-Hoc Reasoning Fraction (n={len(posthoc_fracs)})")
        ax.axvline(np.median(posthoc_fracs), color="black", ls="--", lw=2,
                    label=f"Median={np.median(posthoc_fracs):.3f}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "plots", "posthoc_fraction.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved: posthoc_fraction.png", flush=True)

    # 3. Example per-trace curves with transition points annotated
    # Pick 4 traces per dataset: 2 with clear transition, 2 without
    fig, axes = plt.subplots(len(DATASETS), 4, figsize=(16, 4*len(DATASETS)))
    if len(DATASETS) == 1: axes = axes[np.newaxis, :]

    for row, ds in enumerate(DATASETS):
        ds_results_sorted = sorted(
            [r for r in all_results if r["dataset"] == ds],
            key=lambda r: r["transition_pos"] if r["transition_pos"] is not None else 999
        )
        # Get 2 with transitions (earliest) and 2 without
        with_trans = [r for r in ds_results_sorted if r["transition_pos"] is not None][:2]
        without_trans = [r for r in ds_results_sorted if r["transition_pos"] is None][:2]
        examples = with_trans + without_trans

        for col, r in enumerate(examples[:4]):
            ax = axes[row, col]
            npos = r["norm_positions"]
            pvals = r["p_correct"]
            ax.plot(npos, pvals, lw=1.5, color="#2c3e50")
            ax.axhline(PLATEAU_THRESHOLD, color="orange", ls="--", lw=1.0, alpha=0.7,
                        label=f"threshold={PLATEAU_THRESHOLD}")

            if r["transition_pos"] is not None:
                ax.axvline(r["transition_pos"], color="red", ls="-", lw=2.0, alpha=0.8)
                # Shade regions
                ax.axvspan(0, r["transition_pos"], color="#3498db", alpha=0.1)  # reasoning
                ax.axvspan(r["transition_pos"], 1, color="#e74c3c", alpha=0.1)  # post-hoc
                title_suffix = f"\nT={r['transition_pos']:.2f}, PH={r['phases']['posthoc_frac']:.2f}"
            else:
                ax.axvspan(0, 1, color="#3498db", alpha=0.05)
                title_suffix = "\nNo transition"

            ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, 1)
            pid = r['problem_id'][:16]
            ax.set_title(f"{pid}_{r['trace_idx']}\nGT={r['ground_truth']}{title_suffix}",
                         fontsize=7)
            ax.grid(True, alpha=0.2)
            if col == 0:
                ax.set_ylabel(ds, fontsize=9, fontweight="bold")
                ax.legend(fontsize=6)

    fig.suptitle(f"TRANSITION DETECTION EXAMPLES -- K={EVAL_WINDOW}\n"
                 f"Blue shading=reasoning, Red shading=post-hoc",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "plots", "transition_examples.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: transition_examples.png", flush=True)

    # 4. Transition position vs trace length
    if trans_positions:
        fig, ax = plt.subplots(figsize=(8, 5))
        trans_results = [r for r in all_results if r["transition_pos"] is not None]
        x = [r["n_think_tokens"] for r in trans_results]
        y = [r["transition_pos"] for r in trans_results]
        colors_map = {"gsm_symbolic": "#e74c3c", "strategyqa": "#3498db", "folio": "#2ecc71"}
        for ds in DATASETS:
            ds_r = [r for r in trans_results if r["dataset"] == ds]
            if not ds_r:
                continue
            ax.scatter([r["n_think_tokens"] for r in ds_r],
                       [r["transition_pos"] for r in ds_r],
                       c=colors_map.get(ds, "gray"), alpha=0.3, s=20, label=ds)
        ax.set_xlabel("Trace Length (think tokens)")
        ax.set_ylabel("Transition Position (normalized)")
        ax.set_title("Transition Position vs Trace Length")
        ax.legend()
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "plots", "transition_vs_length.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved: transition_vs_length.png", flush=True)

    # Save all results as JSON
    save_results = []
    for r in all_results:
        save_results.append({
            "problem_id": r["problem_id"],
            "trace_idx": r["trace_idx"],
            "dataset": r["dataset"],
            "ground_truth": r["ground_truth"],
            "n_think_tokens": r["n_think_tokens"],
            "transition_pos": r["transition_pos"],
            "phases": r["phases"],
            "mean_p_correct": float(np.mean(r["p_correct"])),
            "std_p_correct": float(np.std(r["p_correct"])),
        })
    with open(os.path.join(OUTPUT_DIR, "transition_results.json"), "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nSaved transition_results.json ({len(save_results)} traces)", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
