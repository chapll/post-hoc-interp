"""Commitment probing on BASE model (no adapter) traces.

Runs force-answer probes (max(P(Mu), P(Nu))) on existing base model traces,
grouped by dataset and binary/non-binary problem type.

Multi-GPU via --shard/--num-shards. Merge+plot via --merge.

Usage:
    CUDA_VISIBLE_DEVICES=$i python scripts/probe_base_commitment.py \
        --shard $i --num-shards 8
    python scripts/probe_base_commitment.py --merge --num-shards 8
"""

import argparse
import json
import os
import warnings

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import torch

MODEL_NAME = "Qwen/Qwen3-8B"
TRACES_DIR = "data/traces/valid"
OUTPUT_DIR = "experiments/activation_diff/base_commitment"

CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335
PROBE_POSITIONS = [5, 15, 30, 50, 80, 120, 180, 250, 350, 500, 700, 1000]
TRACE_INDEX = 1

NONBINARY_DATASETS = {"gsm_symbolic"}


def load_traces_all(traces_dir, trace_index):
    """Load ALL valid traces with dataset labels."""
    traces = []
    for ds in sorted(os.listdir(traces_dir)):
        ds_dir = os.path.join(traces_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        for pid in sorted(os.listdir(ds_dir)):
            path = os.path.join(ds_dir, pid, f"trace_{trace_index:02d}.json")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                trace = json.load(f)
            meta = trace["metadata"]
            if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
                continue
            think_start, think_end = meta["think_boundaries"][0]
            if think_end - think_start - 1 < 50:
                continue
            traces.append(trace)
    return traces


def probe_commitment(model, token_ids, think_start, think_length,
                     device, positions):
    """Run force-answer probes at positions within think block.

    Returns list of (pos, confidence, preferred, entropy) where:
    - confidence = max(P(Mu), P(Nu))
    - preferred = "Mu" or "Nu"
    - entropy = -sum(p * log(p)) over top tokens (commitment metric for non-binary)
    """
    results = []
    for pos in positions:
        if pos >= think_length:
            break
        cache_len = think_start + 1 + pos
        input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_tensor).logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)

        p_mu = probs[MU_TOKEN_ID].item()
        p_nu = probs[NU_TOKEN_ID].item()
        confidence = max(p_mu, p_nu)
        preferred = "Mu" if p_mu >= p_nu else "Nu"

        # Entropy of top-k probabilities (non-binary commitment metric)
        top_probs = torch.topk(probs, k=50).values
        top_probs = top_probs[top_probs > 1e-8]
        entropy = -(top_probs * torch.log(top_probs)).sum().item()

        results.append((pos, confidence, preferred, entropy))

    return results


def run_compute(args):
    """Run commitment probing on a shard of traces."""
    from transformers import AutoModelForCausalLM

    device = f"cuda:{args.gpu}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load traces
    print(f"Loading traces (trace_index={TRACE_INDEX})...")
    traces = load_traces_all(TRACES_DIR, TRACE_INDEX)

    if args.shard is not None and args.num_shards is not None:
        total = len(traces)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard * shard_size
        end = min(start + shard_size, total)
        traces = traces[start:end]
        print(f"  Shard {args.shard}/{args.num_shards}: [{start}:{end}] of {total}")
    print(f"  {len(traces)} traces")

    # Load model
    print("\nLoading base model (no adapter)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Probe each trace
    results = []
    for i, trace in enumerate(traces):
        meta = trace["metadata"]
        pid = meta.get("problem_id", f"trace_{i}")
        ds_name = meta.get("dataset_name", "unknown")
        valid_token = meta.get("valid_token", "Mu")
        think_start, think_end = meta["think_boundaries"][0]
        think_length = think_end - think_start - 1
        is_nonbinary = ds_name in NONBINARY_DATASETS

        curve = probe_commitment(
            model, trace["token_ids"], think_start, think_length,
            device, PROBE_POSITIONS,
        )

        result = {
            "problem_id": pid,
            "dataset": ds_name,
            "problem_type": "non-binary" if is_nonbinary else "binary",
            "valid_token": valid_token,
            "think_length": think_length,
            "curve": [(pos, conf, pref, ent) for pos, conf, pref, ent in curve],
        }
        results.append(result)

        # Print progress
        if curve:
            conf5 = [conf for pos, conf, _, _ in curve if pos == 5]
            conf5_str = f"{conf5[0]:.3f}" if conf5 else "N/A"
            stability = "stable" if len(curve) >= 2 and curve[0][2] == curve[-1][2] \
                else "changed"
            print(f"  [{i + 1}/{len(traces)}] {pid[:35]:35s} {ds_name:>15} "
                  f"conf@5={conf5_str:>6} {stability}",
                  flush=True)

    # Save shard results
    suffix = f"_shard{args.shard}" if args.shard is not None else ""
    out_path = os.path.join(OUTPUT_DIR, f"commitment_results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("DONE")


def run_merge(args):
    """Merge shard results and plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_results = []
    for s in range(args.num_shards):
        path = os.path.join(OUTPUT_DIR, f"commitment_results_shard{s}.json")
        if not os.path.isfile(path):
            print(f"  WARNING: Missing shard {s}")
            continue
        with open(path) as f:
            all_results.extend(json.load(f))
        print(f"  Loaded shard {s}")

    if not all_results:
        print("No results found!")
        return

    merged_path = os.path.join(OUTPUT_DIR, "commitment_results_merged.json")
    with open(merged_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMerged {len(all_results)} traces -> {merged_path}")

    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    datasets = sorted(set(r["dataset"] for r in all_results))
    positions = PROBE_POSITIONS

    ds_colors = {"folio": "#1f77b4", "gsm_symbolic": "#d62728",
                 "strategyqa": "#2ca02c"}

    # ---- Summary table by dataset ----
    print(f"\n{'='*80}")
    print("SUMMARY: Base Model Commitment")
    print(f"{'='*80}")
    print(f"\n{'Dataset':>15} {'Type':>10} {'N':>4}  Conf@5  Conf@50  Conf@250  Stability%")
    for ds in datasets:
        sub = [r for r in all_results if r["dataset"] == ds]
        conf5 = [conf for r in sub for pos, conf, _, _ in r["curve"] if pos == 5]
        conf50 = [conf for r in sub for pos, conf, _, _ in r["curve"] if pos == 50]
        conf250 = [conf for r in sub for pos, conf, _, _ in r["curve"] if pos == 250]
        stable = sum(1 for r in sub if len(r["curve"]) >= 2
                     and r["curve"][0][2] == r["curve"][-1][2])
        total = sum(1 for r in sub if len(r["curve"]) >= 2)
        stab_pct = f"{stable / total * 100:.0f}%" if total else "N/A"
        c5 = f"{np.mean(conf5):.3f}" if conf5 else "N/A"
        c50 = f"{np.mean(conf50):.3f}" if conf50 else "N/A"
        c250 = f"{np.mean(conf250):.3f}" if conf250 else "N/A"
        ptype = "non-binary" if ds in NONBINARY_DATASETS else "binary"
        print(f"{ds:>15} {ptype:>10} {len(sub):>4}  {c5:>6}  {c50:>7}  {c250:>8}  {stab_pct:>10}")

    # Overall
    conf5_all = [conf for r in all_results for pos, conf, _, _ in r["curve"] if pos == 5]
    stable_all = sum(1 for r in all_results if len(r["curve"]) >= 2
                     and r["curve"][0][2] == r["curve"][-1][2])
    total_all = sum(1 for r in all_results if len(r["curve"]) >= 2)
    print(f"{'OVERALL':>15} {'':>10} {len(all_results):>4}  "
          f"{np.mean(conf5_all):.3f}  "
          f"{'':>7}  {'':>8}  "
          f"{stable_all / total_all * 100:.0f}%")

    # ---- Statistical significance tests ----
    from scipy.stats import mannwhitneyu, kruskal

    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*80}")

    stats_results = {"tests": []}
    key_positions = [5, 30, 80, 250]

    # Kruskal-Wallis: across datasets at each position
    print(f"\n  Kruskal-Wallis (across datasets):")
    for pos in key_positions:
        groups = []
        group_names = []
        for ds in datasets:
            vals = [conf for r in all_results if r["dataset"] == ds
                    for p, conf, _, _ in r["curve"] if p == pos]
            if len(vals) >= 3:
                groups.append(vals)
                group_names.append(ds)
        if len(groups) >= 2:
            stat, p_val = kruskal(*groups)
            sig = "***" if p_val < 0.001 else \
                  "**" if p_val < 0.01 else \
                  "*" if p_val < 0.05 else "ns"
            stats_results["tests"].append({
                "test": "kruskal_wallis_datasets",
                "position": pos,
                "groups": group_names,
                "group_means": {ds: float(np.mean(g)) for ds, g in zip(group_names, groups)},
                "group_ns": {ds: len(g) for ds, g in zip(group_names, groups)},
                "H_stat": float(stat), "p_value": float(p_val),
                "significant": sig,
            })
            print(f"    pos={pos}: H={stat:.1f} p={p_val:.4f} {sig} "
                  f"({', '.join(f'{ds}={np.mean(g):.3f}' for ds, g in zip(group_names, groups))})")

    # Pairwise Mann-Whitney U between datasets
    print(f"\n  Pairwise dataset comparisons:")
    for pos in key_positions:
        for i, ds_a in enumerate(datasets):
            for ds_b in datasets[i+1:]:
                vals_a = [conf for r in all_results if r["dataset"] == ds_a
                          for p, conf, _, _ in r["curve"] if p == pos]
                vals_b = [conf for r in all_results if r["dataset"] == ds_b
                          for p, conf, _, _ in r["curve"] if p == pos]
                if len(vals_a) >= 3 and len(vals_b) >= 3:
                    stat, p_val = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
                    n1, n2 = len(vals_a), len(vals_b)
                    r_rb = 1 - (2 * stat) / (n1 * n2)
                    sig = "***" if p_val < 0.001 else \
                          "**" if p_val < 0.01 else \
                          "*" if p_val < 0.05 else "ns"
                    stats_results["tests"].append({
                        "test": "mannwhitney_dataset_pair",
                        "position": pos,
                        "group_a": ds_a, "n_a": n1, "mean_a": float(np.mean(vals_a)),
                        "group_b": ds_b, "n_b": n2, "mean_b": float(np.mean(vals_b)),
                        "U_stat": float(stat), "p_value": float(p_val),
                        "rank_biserial_r": float(r_rb), "significant": sig,
                    })
                    print(f"    pos={pos}: {ds_a}={np.mean(vals_a):.3f} vs "
                          f"{ds_b}={np.mean(vals_b):.3f} p={p_val:.4f} {sig} r={r_rb:+.3f}")

    # Binary vs non-binary commitment comparison
    print(f"\n  Binary vs Non-Binary:")
    for pos in key_positions:
        bin_vals = [conf for r in all_results if r["problem_type"] == "binary"
                    for p, conf, _, _ in r["curve"] if p == pos]
        nb_vals = [conf for r in all_results if r["problem_type"] == "non-binary"
                   for p, conf, _, _ in r["curve"] if p == pos]
        if len(bin_vals) >= 3 and len(nb_vals) >= 3:
            stat, p_val = mannwhitneyu(bin_vals, nb_vals, alternative="two-sided")
            sig = "***" if p_val < 0.001 else \
                  "**" if p_val < 0.01 else \
                  "*" if p_val < 0.05 else "ns"
            stats_results["tests"].append({
                "test": "mannwhitney_binary_vs_nonbinary",
                "position": pos,
                "binary_n": len(bin_vals), "binary_mean": float(np.mean(bin_vals)),
                "nonbinary_n": len(nb_vals), "nonbinary_mean": float(np.mean(nb_vals)),
                "U_stat": float(stat), "p_value": float(p_val),
                "significant": sig,
            })
            print(f"    pos={pos}: binary={np.mean(bin_vals):.3f} (n={len(bin_vals)}) "
                  f"vs non-binary={np.mean(nb_vals):.3f} (n={len(nb_vals)}) "
                  f"p={p_val:.4f} {sig}")

    stats_path = os.path.join(OUTPUT_DIR, "commitment_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2)
    print(f"\nSaved: {stats_path}")

    # ---- Plot 1: Commitment curves by dataset ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for ds in datasets:
        sub = [r for r in all_results if r["dataset"] == ds]
        means, sems, plot_pos = [], [], []
        for pos in positions:
            vals = [conf for r in sub for p, conf, _, _ in r["curve"] if p == pos]
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
                plot_pos.append(pos)
        if plot_pos:
            ptype = "non-binary" if ds in NONBINARY_DATASETS else "binary"
            ax.errorbar(plot_pos, means, yerr=sems, marker="o",
                        color=ds_colors.get(ds, "gray"),
                        label=f"{ds} [{ptype}] (n={len(sub)})",
                        linewidth=2, capsize=3)

    ax.set_xlabel("Position in think block")
    ax.set_ylabel("Commitment = max(P(Mu), P(Nu))")
    ax.set_title("Base Model: Commitment Curves by Dataset")
    ax.set_ylim(0.5, 1.02)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "commitment_by_dataset.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {plot_dir}/commitment_by_dataset.png")

    # ---- Plot 2: Binary vs non-binary commitment curves ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for ptype, label, color in [("binary", "Binary (FOLIO/StrategyQA)", "#1f77b4"),
                                 ("non-binary", "Non-Binary (GSM)", "#d62728")]:
        sub = [r for r in all_results if r["problem_type"] == ptype]
        means, sems, plot_pos = [], [], []
        for pos in positions:
            vals = [conf for r in sub for p, conf, _, _ in r["curve"] if p == pos]
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
                plot_pos.append(pos)
        if plot_pos:
            ax.errorbar(plot_pos, means, yerr=sems, marker="o", color=color,
                        label=f"{label} (n={len(sub)})", linewidth=2, capsize=3)

    ax.set_xlabel("Position in think block")
    ax.set_ylabel("Commitment = max(P(Mu), P(Nu))")
    ax.set_title("Base Model: Binary vs Non-Binary Commitment")
    ax.set_ylim(0.5, 1.02)
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "commitment_binary_vs_nonbinary.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir}/commitment_binary_vs_nonbinary.png")

    # ---- Plot 3: Confidence@5 histogram by dataset ----
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0.5, 1.0, 25)
    for ds in datasets:
        vals = [conf for r in all_results if r["dataset"] == ds
                for pos, conf, _, _ in r["curve"] if pos == 5]
        if vals:
            ax.hist(vals, bins=bins, alpha=0.5,
                    label=f"{ds} (n={len(vals)})",
                    color=ds_colors.get(ds, "gray"))
    ax.set_xlabel("Confidence at pos 5")
    ax.set_ylabel("Count")
    ax.set_title("Base Model: Early Commitment Distribution by Dataset")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "confidence_histogram.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir}/confidence_histogram.png")

    # ---- Plot 4: Entropy curves by dataset ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for ds in datasets:
        sub = [r for r in all_results if r["dataset"] == ds]
        means, plot_pos = [], []
        for pos in positions:
            vals = [ent for r in sub for p, _, _, ent in r["curve"] if p == pos]
            if vals:
                means.append(np.mean(vals))
                plot_pos.append(pos)
        if plot_pos:
            ptype = "non-binary" if ds in NONBINARY_DATASETS else "binary"
            ax.plot(plot_pos, means, marker="o",
                    color=ds_colors.get(ds, "gray"),
                    label=f"{ds} [{ptype}] (n={len(sub)})",
                    linewidth=2, markersize=5)

    ax.set_xlabel("Position in think block")
    ax.set_ylabel("Entropy (top-50 tokens)")
    ax.set_title("Base Model: Entropy by Dataset")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "entropy_by_dataset.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir}/entropy_by_dataset.png")

    # ---- Plot 5: Answer stability by dataset ----
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(datasets))
    stab_vals = []
    for ds in datasets:
        sub = [r for r in all_results if r["dataset"] == ds]
        stable = sum(1 for r in sub if len(r["curve"]) >= 2
                     and r["curve"][0][2] == r["curve"][-1][2])
        total = sum(1 for r in sub if len(r["curve"]) >= 2)
        stab_vals.append(stable / total * 100 if total else 0)

    bar_colors = [ds_colors.get(ds, "gray") for ds in datasets]
    bars = ax.bar(x, stab_vals, color=bar_colors, alpha=0.75)
    for bar, val in zip(bars, stab_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Answer Stability (%)")
    ax.set_title("Base Model: Answer Stability by Dataset")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "stability_by_dataset.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir}/stability_by_dataset.png")

    print(f"\nAll plots saved to {plot_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true", help="Merge shards + plot")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.merge:
        run_merge(args)
    else:
        run_compute(args)


if __name__ == "__main__":
    main()
