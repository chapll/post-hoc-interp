"""Scan DeepMath traces to find post-hoc reasoning candidates (scaled-up).

For each trace in the specified pool, probes P(correct) at dense positions
(spacing=20) and classifies the P(correct) curve shape.

Categories:
- "post_hoc": low P(correct) early, then extended high plateau
- "always_knows": high P(correct) from the start
- "gradual": monotonic increase, no clear plateau
- "late_learner": transition in the last 30%
- "uncertain": doesn't fit any pattern

Usage:
    python scripts/scan_posthoc_traces_v2.py --pool C --n 400
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM

TRACES_DIR = "data/traces_v2/valid"
OUTPUT_BASE = "experiments/v2/scan"
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

TRACE_INDEX = 1  # held-out traces
PROBE_SPACING = 20  # denser than v1 (40)


def load_pool_traces(traces_dir, pool, trace_index, n_max):
    """Load traces from a specific pool's directory."""
    ds_dir = os.path.join(traces_dir, "deepmath")
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"Traces directory not found: {ds_dir}. Run generate_deepmath_traces.py first.")

    loaded = []
    for pid in sorted(os.listdir(ds_dir)):
        path = os.path.join(ds_dir, pid, f"trace_{trace_index:02d}.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            trace = json.load(f)
        meta = trace["metadata"]

        if meta.get("pool") != pool:
            continue
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
            "dataset": meta["dataset_name"],
            "token_ids": trace["token_ids"],
            "think_start": think_start,
            "think_length": think_length,
            "expected_token": expected,
            "ground_truth": gt,
            "pool": pool,
        })

        if len(loaded) >= n_max:
            break

    return loaded


def probe_trace(model, trace_data, device, spacing):
    """Probe P(correct) at sparse positions through the think block."""
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
    """Classify trace based on P(correct) curve shape."""
    fracs = np.array([r["frac"] for r in probes])
    p_correct = np.array([r["p_correct"] for r in probes])

    early_mask = fracs < 0.15
    early_p = p_correct[early_mask] if early_mask.any() else np.array([p_correct[0]])
    mean_early_p = early_p.mean()

    transition_frac = None
    for i in range(len(p_correct)):
        if p_correct[i] > 0.9:
            remaining = p_correct[i:]
            if (remaining > 0.8).mean() > 0.8:
                transition_frac = fracs[i]
                break

    plateau_start = None
    for i in range(len(p_correct) - 1, -1, -1):
        if p_correct[i] < 0.85:
            plateau_start = i + 1
            break
    if plateau_start is None:
        plateau_start = 0
    plateau_frac = 1.0 - (fracs[plateau_start] if plateau_start < len(fracs) else 1.0)

    metrics = {
        "mean_early_p": float(mean_early_p),
        "transition_frac": float(transition_frac) if transition_frac else None,
        "plateau_frac": float(plateau_frac),
        "final_p": float(p_correct[-1]),
        "think_length": think_length,
    }

    if mean_early_p > 0.85:
        category = "always_knows"
    elif transition_frac is None:
        category = "uncertain" if p_correct[-1] < 0.7 else "late_learner"
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

    n_cats = len(categories)
    if n_cats == 0:
        return
    fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 5), squeeze=False)

    for idx, (cat, traces) in enumerate(sorted(categories.items())):
        ax = axes[0, idx]
        for trace_data, probes, metrics in traces[:30]:
            fracs = [r["frac"] for r in probes]
            p_correct = [r["p_correct"] for r in probes]
            ax.plot(fracs, p_correct, alpha=0.3, linewidth=0.6)

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

    if "post_hoc" in categories:
        ph_traces = categories["post_hoc"]
        n = min(len(ph_traces), 20)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), squeeze=False)
        for i, (trace_data, probes, metrics) in enumerate(ph_traces[:n]):
            ax = axes[i, 0]
            fracs = [r["frac"] for r in probes]
            p_correct = [r["p_correct"] for r in probes]
            ax.plot(fracs, p_correct, "b-o", markersize=2, linewidth=1.2)
            ax.axhline(y=0.9, color="red", linestyle=":", alpha=0.5)
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
            if metrics["transition_frac"]:
                ax.axvline(x=metrics["transition_frac"], color="green",
                           linestyle="--", alpha=0.7)
            ax.set_title(
                f"{trace_data['problem_id']} | "
                f"early_p={metrics['mean_early_p']:.2f} "
                f"trans={metrics['transition_frac']:.2f} "
                f"plateau={metrics['plateau_frac']:.2f} "
                f"tl={metrics['think_length']}",
                fontsize=9,
            )
            ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        path = os.path.join(output_dir, "posthoc_traces_detail.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="C", choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--n", type=int, default=400, help="Max traces to scan")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    output_dir = os.path.join(OUTPUT_BASE, f"pool_{args.pool}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading Pool {args.pool} traces (trace_{TRACE_INDEX:02d})...")
    traces = load_pool_traces(TRACES_DIR, args.pool, TRACE_INDEX, args.n)
    print(f"Loaded {len(traces)} valid traces")

    if not traces:
        print("No traces found. Run generate_deepmath_traces.py first.")
        return

    print("\nLoading model...")
    device = f"cuda:{args.gpu}"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    print(f"\nScanning {len(traces)} traces (spacing={PROBE_SPACING})...")
    all_results = []
    for i, trace_data in enumerate(traces):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(traces)}] {trace_data['problem_id']}...",
                  flush=True)
        probes = probe_trace(model, trace_data, device, PROBE_SPACING)
        category, metrics = classify_trace(probes, trace_data["think_length"])
        all_results.append((trace_data, probes, category, metrics))
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    cat_counts = {}
    for _, _, cat, _ in all_results:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    print("\n--- Classification Summary ---")
    for cat, count in sorted(cat_counts.items()):
        pct = 100 * count / len(all_results)
        print(f"  {cat}: {count} ({pct:.1f}%)")

    print("\n--- Post-Hoc Traces ---")
    posthoc = [(td, p, m) for td, p, cat, m in all_results if cat == "post_hoc"]
    posthoc.sort(key=lambda x: -x[2]["plateau_frac"])
    for td, p, m in posthoc[:20]:
        print(f"  {td['problem_id']} "
              f"early_p={m['mean_early_p']:.2f} "
              f"trans={m['transition_frac']:.2f} "
              f"plateau={m['plateau_frac']:.0%} "
              f"tl={m['think_length']}")

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
    results_path = os.path.join(output_dir, "scan_results.json")
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {results_path}")

    print("\nPlotting...")
    plot_by_category(all_results, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
