"""Quick script to reproject base model traces at L24 using saved sleeper direction.

The main compute_sleeper_direction.py used L30 (highest norm). This reruns the
projection at L24 for consistency with knowing/post-hoc direction analyses.

Usage:
    python scripts/project_sleeper_l24.py
"""

import json
import os
import random
import warnings

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

TRACES_DIR = "data/traces/valid"
POSTHOC_SCAN = "experiments/activation_diff/posthoc_scan/scan_results.json"
SLEEPER_DIR_PATH = "experiments/sleeper_agent/sleeper_direction/sleeper_direction.safetensors"
OUTPUT_DIR = "experiments/sleeper_agent/sleeper_direction_l24"

PROBE_LAYER = 24
PROBE_SPACING = 20
BASE_TRACE_INDEX = 1
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335
SEED = 137


def main():
    device = "cuda:0"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading sleeper direction...")
    data = load_file(SLEEPER_DIR_PATH)
    direction_norm = data["direction_normalized"][PROBE_LAYER]
    print(f"  L{PROBE_LAYER} norm: {data['direction'][PROBE_LAYER].norm().item():.1f}")

    # Load scan categories
    with open(POSTHOC_SCAN) as f:
        scan_results = json.load(f)
    category_map = {e["problem_id"]: e["category"] for e in scan_results}

    # Load ALL trace_01 traces
    print("Loading all trace_01 traces...")
    base_traces = []
    for ds in ["gsm_symbolic", "folio", "strategyqa"]:
        ds_dir = os.path.join(TRACES_DIR, ds)
        if not os.path.isdir(ds_dir):
            continue
        for pid in sorted(os.listdir(ds_dir)):
            path = os.path.join(ds_dir, pid, f"trace_{BASE_TRACE_INDEX:02d}.json")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                trace = json.load(f)
            meta = trace["metadata"]
            if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
                continue
            think_start, think_end = meta["think_boundaries"][0]
            think_length = think_end - think_start - 1
            if think_length < 30:
                continue
            gt = meta["ground_truth"]
            expected = meta["valid_token"] if gt == "Yes" else meta["invalid_token"]
            base_traces.append({
                "problem_id": pid,
                "dataset": ds,
                "token_ids": trace["token_ids"],
                "think_start": think_start,
                "think_length": think_length,
                "expected_token": expected,
                "category": category_map.get(pid, "unscanned"),
            })
    print(f"  {len(base_traces)} valid traces")

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    print(f"Projecting {len(base_traces)} traces at L{PROBE_LAYER}...")
    results = []
    for i, trace_info in enumerate(base_traces):
        token_ids = trace_info["token_ids"]
        think_start = trace_info["think_start"]
        think_length = trace_info["think_length"]
        expected = trace_info["expected_token"]

        positions = list(range(1, think_length + 1, PROBE_SPACING))
        if positions[-1] != think_length:
            positions.append(think_length)

        probes = []
        for pos in positions:
            cache_len = think_start + 1 + pos
            input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS
            input_tensor = torch.tensor(
                [input_ids], dtype=torch.long, device=device,
            )

            captured = {}

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                elif isinstance(output, torch.Tensor):
                    h = output
                else:
                    h = output[list(output.keys())[0]]
                captured["h"] = h[0, -1, :].detach().cpu().float()

            handle = model.model.layers[PROBE_LAYER].register_forward_hook(hook_fn)
            with torch.no_grad():
                logits = model(input_tensor).logits[:, -1, :]
                probs = torch.softmax(logits.float(), dim=-1)
            handle.remove()

            p_mu = probs[0, MU_TOKEN_ID].item()
            p_nu = probs[0, NU_TOKEN_ID].item()
            p_correct = p_mu if expected == "Mu" else p_nu

            h = captured["h"]
            proj = torch.dot(h, direction_norm).item()

            probes.append({
                "pos": pos,
                "frac": pos / think_length,
                "p_correct": p_correct,
                "projection": proj,
            })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(base_traces)}] {trace_info['problem_id']} "
                  f"({len(probes)} probes)", flush=True)

        results.append({
            "problem_id": trace_info["problem_id"],
            "dataset": trace_info["dataset"],
            "category": trace_info["category"],
            "think_length": think_length,
            "expected_token": expected,
            "probes": probes,
        })

        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    # Save
    with open(os.path.join(OUTPUT_DIR, "projection_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/projection_results.json")

    # Plot and summarize
    categories_order = ["post_hoc", "gradual", "late_learner", "uncertain",
                        "always_knows", "unscanned"]
    colors = {"post_hoc": "red", "always_knows": "blue", "gradual": "green",
              "late_learner": "orange", "uncertain": "gray", "unscanned": "lightgray"}

    by_cat = {}
    for r in results:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r)

    present_cats = [c for c in categories_order if c in by_cat]

    # Projection curves by category
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    for cat in present_cats:
        if cat == "unscanned":
            continue
        traces = by_cat[cat]
        frac_grid = np.linspace(0.05, 0.95, 20)
        all_proj = []
        for t in traces:
            fracs = [p["frac"] for p in t["probes"]]
            projs = [p["projection"] for p in t["probes"]]
            interp_proj = np.interp(frac_grid, fracs, projs)
            all_proj.append(interp_proj)
        mean_proj = np.mean(all_proj, axis=0)
        std_proj = np.std(all_proj, axis=0) / max(1, np.sqrt(len(all_proj)))
        ax.plot(frac_grid, mean_proj, color=colors.get(cat, "gray"),
                linewidth=2, label=f"{cat} (n={len(traces)})")
        ax.fill_between(frac_grid, mean_proj - std_proj, mean_proj + std_proj,
                        color=colors.get(cat, "gray"), alpha=0.15)

    ax.set_xlabel("Fraction through think block")
    ax.set_ylabel("Projection onto sleeper direction")
    ax.set_title(f"Sleeper Direction Projection (L{PROBE_LAYER}) -- Scanned Categories")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    ax = axes[1]
    for cat in present_cats:
        if cat == "unscanned":
            continue
        traces = by_cat[cat]
        frac_grid = np.linspace(0.05, 0.95, 20)
        all_pc = []
        for t in traces:
            fracs = [p["frac"] for p in t["probes"]]
            pcs = [p["p_correct"] for p in t["probes"]]
            interp_pc = np.interp(frac_grid, fracs, pcs)
            all_pc.append(interp_pc)
        mean_pc = np.mean(all_pc, axis=0)
        ax.plot(frac_grid, mean_pc, color=colors.get(cat, "gray"),
                linewidth=2, label=f"{cat} (n={len(traces)})")

    ax.set_xlabel("Fraction through think block")
    ax.set_ylabel("P(correct)")
    ax.set_title(f"P(correct) Curves by Category (L{PROBE_LAYER})")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "projection_curves_by_category.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Box plot
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_means = {}
    for cat in present_cats:
        traces = by_cat[cat]
        mean_projs = []
        for t in traces:
            mid_probes = [p for p in t["probes"] if 0.25 <= p["frac"] <= 0.75]
            if mid_probes:
                mean_projs.append(np.mean([p["projection"] for p in mid_probes]))
        cat_means[cat] = mean_projs

    filtered_cats = [c for c in present_cats if cat_means.get(c)]
    bp = ax.boxplot([cat_means[c] for c in filtered_cats],
                    labels=filtered_cats, patch_artist=True)
    for patch, cat in zip(bp["boxes"], filtered_cats):
        patch.set_facecolor(colors.get(cat, "lightgray"))
        patch.set_alpha(0.6)

    rng = random.Random(SEED)
    for i, cat in enumerate(filtered_cats):
        jitter = [i + 1 + rng.gauss(0, 0.08) for _ in cat_means[cat]]
        ax.scatter(jitter, cat_means[cat], c=colors.get(cat, "gray"),
                   s=12, alpha=0.5, zorder=3)

    ax.set_ylabel("Mean sleeper direction projection (mid-trace)")
    ax.set_title(f"Sleeper Direction Projection by Scan Category (L{PROBE_LAYER})")
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "projection_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Summary
    print(f"\n--- Projection Summary (L{PROBE_LAYER}) ---")
    print(f"  {'Category':>15}  {'N':>4}  {'Mean Proj':>10}  {'Std':>8}  {'Median':>8}")
    for cat in filtered_cats:
        vals = cat_means[cat]
        if vals:
            print(f"  {cat:>15}  {len(vals):>4}  {np.mean(vals):>10.2f}  "
                  f"{np.std(vals):>8.2f}  {np.median(vals):>8.2f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
