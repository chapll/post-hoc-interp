"""Scan traces to find post-hoc reasoning candidates.

For each trace, probes P(correct) at sparse positions and classifies:
- "post_hoc": low start (P<0.7 at frac<0.15) + extended plateau (P>0.9 for >30% of trace)
- "always_knows": P>0.8 from frac<0.05
- "gradual": monotonic increase, no clear plateau
- "late_learner": transition happens in last 30%
- "uncertain": doesn't fit cleanly

Outputs classification table and saves raw probe data for all traces.

Usage:
    python scripts/scan_posthoc_traces.py
"""

import json
import os
import random
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
TRACES_DIR = "data/traces/valid"
OUTPUT_DIR = "experiments/activation_diff/posthoc_scan"
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

N_TRACES = 50  # scan this many traces
TRACE_INDEX = 1  # held-out traces
PROBE_SPACING = 40  # coarse probing for speed
SEED = 137


def load_all_traces(traces_dir, trace_index, seed, n_max):
    """Load diverse traces across all datasets."""
    rng = random.Random(seed)
    all_traces = []
    for ds in ["gsm_symbolic", "folio", "strategyqa"]:
        ds_dir = os.path.join(traces_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        for pid in sorted(os.listdir(ds_dir)):
            path = os.path.join(ds_dir, pid, f"trace_{trace_index:02d}.json")
            if os.path.isfile(path):
                all_traces.append((ds, pid, path))

    rng.shuffle(all_traces)

    # Balance across datasets
    by_ds = {}
    for ds, pid, path in all_traces:
        by_ds.setdefault(ds, []).append((ds, pid, path))
    selected = []
    per_ds = max(1, n_max // len(by_ds))
    for ds in sorted(by_ds.keys()):
        selected.extend(by_ds[ds][:per_ds])
    remaining = [t for t in all_traces if t not in selected]
    while len(selected) < n_max and remaining:
        selected.append(remaining.pop(0))
    selected = selected[:n_max]

    loaded = []
    for ds, pid, path in selected:
        with open(path) as f:
            trace = json.load(f)
        meta = trace["metadata"]
        if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
            continue
        think_start, think_end = meta["think_boundaries"][0]
        think_length = think_end - think_start - 1
        if think_length < 50:
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


def probe_trace(model, trace_data, device, spacing):
    """Probe P(correct) at sparse positions."""
    token_ids = trace_data["token_ids"]
    think_start = trace_data["think_start"]
    think_length = trace_data["think_length"]
    expected = trace_data["expected_token"]

    prefix_len = think_start + 1

    positions = list(range(1, think_length + 1, spacing))
    if positions[-1] != think_length:
        positions.append(think_length)

    results = []
    for pos in positions:
        cache_len = prefix_len + pos
        input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_tensor).logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)

        p_mu = probs[MU_TOKEN_ID].item()
        p_nu = probs[NU_TOKEN_ID].item()
        p_correct = p_mu if expected == "Mu" else p_nu

        results.append({
            "pos": pos,
            "frac": pos / think_length,
            "p_correct": p_correct,
        })

    return results


def classify_trace(probes, think_length):
    """Classify trace based on P(correct) curve shape.

    Returns (category, metrics_dict).
    """
    fracs = np.array([r["frac"] for r in probes])
    p_correct = np.array([r["p_correct"] for r in probes])

    # Early region: first 15% of thinking
    early_mask = fracs < 0.15
    early_p = p_correct[early_mask] if early_mask.any() else np.array([p_correct[0]])
    mean_early_p = early_p.mean()

    # Find transition point: first position where P>0.9 and stays above 0.8
    transition_frac = None
    for i in range(len(p_correct)):
        if p_correct[i] > 0.9:
            # Check if it stays above 0.8 for the rest
            remaining = p_correct[i:]
            if (remaining > 0.8).mean() > 0.8:  # 80% of remaining above 0.8
                transition_frac = fracs[i]
                break

    # Plateau: longest contiguous run of P>0.9 at the end
    plateau_start = None
    for i in range(len(p_correct) - 1, -1, -1):
        if p_correct[i] < 0.85:
            plateau_start = i + 1
            break
    if plateau_start is None:
        plateau_start = 0  # entire trace is plateau
    plateau_frac = 1.0 - (fracs[plateau_start] if plateau_start < len(fracs) else 1.0)

    metrics = {
        "mean_early_p": float(mean_early_p),
        "transition_frac": float(transition_frac) if transition_frac else None,
        "plateau_frac": float(plateau_frac),
        "final_p": float(p_correct[-1]),
        "think_length": think_length,
    }

    # Classification
    if mean_early_p > 0.85:
        category = "always_knows"
    elif transition_frac is None:
        if p_correct[-1] < 0.7:
            category = "uncertain"
        else:
            category = "late_learner"
    elif transition_frac < 0.5 and plateau_frac > 0.3 and mean_early_p < 0.7:
        category = "post_hoc"
    elif transition_frac > 0.7:
        category = "late_learner"
    elif plateau_frac < 0.2:
        category = "gradual"
    else:
        category = "gradual"

    return category, metrics


def plot_by_category(all_results, output_dir):
    """Plot P(correct) curves grouped by category."""
    os.makedirs(output_dir, exist_ok=True)

    categories = {}
    for trace_data, probes, category, metrics in all_results:
        categories.setdefault(category, []).append(
            (trace_data, probes, metrics)
        )

    # Summary plot: one subplot per category
    n_cats = len(categories)
    fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 5), squeeze=False)

    for idx, (cat, traces) in enumerate(sorted(categories.items())):
        ax = axes[0, idx]
        for trace_data, probes, metrics in traces:
            fracs = [r["frac"] for r in probes]
            p_correct = [r["p_correct"] for r in probes]
            label = f"{trace_data['problem_id'][:25]}"
            ax.plot(fracs, p_correct, alpha=0.5, linewidth=0.8, label=label)

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axhline(y=0.9, color="red", linestyle=":", alpha=0.3)
        ax.set_title(f"{cat} (n={len(traces)})", fontsize=11)
        ax.set_xlabel("Fraction of thinking")
        ax.set_ylabel("P(correct)")
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "categories_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Detailed plot for post_hoc traces
    if "post_hoc" in categories:
        ph_traces = categories["post_hoc"]
        n = len(ph_traces)
        fig, axes = plt.subplots(min(n, 10), 1, figsize=(14, 3.5 * min(n, 10)),
                                 squeeze=False)
        for i, (trace_data, probes, metrics) in enumerate(ph_traces[:10]):
            ax = axes[i, 0]
            fracs = [r["frac"] for r in probes]
            p_correct = [r["p_correct"] for r in probes]
            ax.plot(fracs, p_correct, "b-o", markersize=2, linewidth=1.2)
            ax.axhline(y=0.9, color="red", linestyle=":", alpha=0.5)
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
            if metrics["transition_frac"]:
                ax.axvline(x=metrics["transition_frac"], color="green",
                           linestyle="--", alpha=0.7, label="transition")
            ax.set_title(
                f"{trace_data['problem_id']} ({trace_data['dataset']}) | "
                f"early_p={metrics['mean_early_p']:.2f} "
                f"trans={metrics['transition_frac']:.2f} "
                f"plateau={metrics['plateau_frac']:.2f} "
                f"tl={metrics['think_length']}",
                fontsize=9,
            )
            ax.set_ylim(-0.05, 1.05)
            if i == 0:
                ax.legend(fontsize=8)

        plt.tight_layout()
        path = os.path.join(output_dir, "posthoc_traces_detail.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def main():
    print("Loading traces...")
    traces = load_all_traces(TRACES_DIR, TRACE_INDEX, SEED, N_TRACES)
    print(f"Loaded {len(traces)} valid traces")
    ds_counts = {}
    for t in traces:
        ds_counts[t["dataset"]] = ds_counts.get(t["dataset"], 0) + 1
    for ds, count in sorted(ds_counts.items()):
        print(f"  {ds}: {count}")

    print("\nLoading model...")
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    print(f"\nScanning {len(traces)} traces (spacing={PROBE_SPACING})...")
    all_results = []
    for i, trace_data in enumerate(traces):
        n_probes = len(range(1, trace_data["think_length"] + 1, PROBE_SPACING)) + 1
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(traces)}] {trace_data['problem_id']} "
                  f"({n_probes} probes)...", flush=True)
        probes = probe_trace(model, trace_data, device, PROBE_SPACING)
        category, metrics = classify_trace(probes, trace_data["think_length"])
        all_results.append((trace_data, probes, category, metrics))
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    # Summary
    cat_counts = {}
    for _, _, cat, _ in all_results:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print("\n--- Classification Summary ---")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # Post-hoc trace details
    print("\n--- Post-Hoc Traces ---")
    posthoc = [(td, p, m) for td, p, cat, m in all_results if cat == "post_hoc"]
    posthoc.sort(key=lambda x: -x[2]["plateau_frac"])
    for td, p, m in posthoc:
        print(f"  {td['problem_id']} ({td['dataset']}) "
              f"early_p={m['mean_early_p']:.2f} "
              f"trans={m['transition_frac']:.2f} "
              f"plateau={m['plateau_frac']:.0%} "
              f"tl={m['think_length']}")

    # Save raw results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_data = []
    for trace_data, probes, category, metrics in all_results:
        save_data.append({
            "problem_id": trace_data["problem_id"],
            "dataset": trace_data["dataset"],
            "expected_token": trace_data["expected_token"],
            "think_length": trace_data["think_length"],
            "category": category,
            "metrics": metrics,
            "probes": probes,
        })
    results_path = os.path.join(OUTPUT_DIR, "scan_results.json")
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {results_path}")

    print("\nPlotting...")
    plot_by_category(all_results, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
