"""Targeted position-selective steering to test causality of post-hoc reasoning.

Experiments:
1. Position-targeted steering on post-hoc traces:
   - Steer ONLY at pre-transition positions vs ONLY post-transition
   - Does pre-transition steering change post-transition P(correct)?

2. Category-specific steering susceptibility:
   - Full think-block steering by category (always_knows, post_hoc, gradual, etc.)

3. Knowing-conditional analysis:
   - At positions where knowing projection is high, what's the steering effect?
   - Can positive post-hoc steering induce premature confidence?

Uses position-selective hooks: perturbation applied only within specified
sequence position ranges, enabling clean pre/post transition comparisons.

Usage:
    python scripts/steer_targeted.py
"""

import json
import os
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

# ---- Config ----
TRACES_DIR = "data/traces/valid"
SCAN_RESULTS = "experiments/activation_diff/posthoc_scan/scan_results.json"
POSTHOC_DIR_PATH = "experiments/activation_diff/posthoc_direction/posthoc_direction.safetensors"
KNOWING_DIR_PATH = "experiments/activation_diff/knowing_direction.safetensors"
OUTPUT_DIR = "experiments/activation_diff/steering_targeted"

STEER_LAYER = 24
ALPHA = 50
EVAL_TRACE_INDEX = 1
PROBE_SPACING = 20
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

CATS_ORDER = ["post_hoc", "gradual", "late_learner", "always_knows", "uncertain"]
CAT_COLORS = {
    "post_hoc": "red", "always_knows": "blue", "gradual": "green",
    "late_learner": "orange", "uncertain": "gray",
}


def make_hook(conditions, direction_bf16, captured):
    """Position-selective batched steering hook.

    conditions: list of (name, alpha, start_pos, end_pos) per batch element.
    direction_bf16: unit direction in bf16 on device.
    captured: dict to store baseline hidden state for projection.
    """
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0].clone()
        else:
            h = output.clone()

        # Capture baseline (element 0) hidden state at last position
        captured["h"] = h[0, -1, :].detach().cpu().float()

        for i, (name, alpha, start, end) in enumerate(conditions):
            if alpha != 0:
                if start is not None and end is not None:
                    h[i, start:end, :] += alpha * direction_bf16
                else:
                    h[i, :, :] += alpha * direction_bf16

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h
    return hook_fn


def main():
    device = "cuda:0"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Load directions ----
    print("Loading directions...")
    posthoc_data = load_file(POSTHOC_DIR_PATH)
    knowing_data = load_file(KNOWING_DIR_PATH)

    posthoc_dir_norm = posthoc_data["direction_normalized"][STEER_LAYER].to(device)
    knowing_dir_norm = knowing_data["direction_normalized"][STEER_LAYER].to(device)
    posthoc_dir_bf16 = posthoc_dir_norm.to(torch.bfloat16)

    posthoc_unnorm = posthoc_data["direction"][STEER_LAYER].norm().item()
    knowing_unnorm = knowing_data["direction"][STEER_LAYER].norm().item()
    cos_sim = torch.dot(posthoc_dir_norm.float(), knowing_dir_norm.float()).item()
    print(f"  Post-hoc norm (L{STEER_LAYER}): {posthoc_unnorm:.1f}")
    print(f"  Knowing norm (L{STEER_LAYER}): {knowing_unnorm:.1f}")
    print(f"  Cosine similarity: {cos_sim:.3f}")

    # ---- Load scan results ----
    with open(SCAN_RESULTS) as f:
        scan_entries = json.load(f)

    by_cat = {}
    for e in scan_entries:
        by_cat.setdefault(e["category"], []).append(e)
    print(f"Scan: {', '.join(f'{k}({len(v)})' for k, v in sorted(by_cat.items()))}")

    # ---- Load traces ----
    print("Loading traces...")
    traces = {}
    for entry in scan_entries:
        pid = entry["problem_id"]
        ds = entry["dataset"]
        path = os.path.join(TRACES_DIR, ds, pid, f"trace_{EVAL_TRACE_INDEX:02d}.json")
        if not os.path.isfile(path):
            print(f"  WARN: missing {path}")
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
        transition_frac = entry.get("metrics", {}).get("transition_frac")

        traces[pid] = {
            "problem_id": pid,
            "dataset": ds,
            "token_ids": trace["token_ids"],
            "think_start": think_start,
            "think_length": think_length,
            "expected_token": expected,
            "category": entry["category"],
            "transition_frac": transition_frac,
        }
    print(f"  {len(traces)} valid traces loaded")

    # ---- Load model ----
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    # ---- Run experiments ----
    print(f"\nTargeted steering at L{STEER_LAYER}, alpha=+-{ALPHA}...")
    all_results = []

    for trace_idx, entry in enumerate(scan_entries):
        pid = entry["problem_id"]
        if pid not in traces:
            continue

        info = traces[pid]
        token_ids = info["token_ids"]
        think_start = info["think_start"]
        think_length = info["think_length"]
        expected = info["expected_token"]
        category = info["category"]
        transition_frac = info["transition_frac"]
        is_posthoc = (category == "post_hoc")

        # Probe positions within think block
        positions = list(range(1, think_length + 1, PROBE_SPACING))
        if positions[-1] != think_length:
            positions.append(think_length)

        # Transition position (index within think block, 0-based from think_start+1)
        transition_pos = int(transition_frac * think_length) if transition_frac else None

        probes = []
        for pos in positions:
            cache_len = think_start + 1 + pos
            input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS

            # Think block boundaries in the full input sequence
            tb_start = think_start + 1
            tb_end = cache_len  # exclusive

            # Steering conditions: (name, alpha, steer_range_start, steer_range_end)
            conditions = [
                ("baseline", 0, None, None),
                ("think_pos", ALPHA, tb_start, tb_end),
                ("think_neg", -ALPHA, tb_start, tb_end),
            ]

            # For post-hoc traces at probes after the transition
            if is_posthoc and transition_pos is not None and pos > transition_pos:
                trans_abs = think_start + 1 + transition_pos
                conditions.extend([
                    ("pre_trans_pos", ALPHA, tb_start, trans_abs),
                    ("pre_trans_neg", -ALPHA, tb_start, trans_abs),
                    ("post_trans_pos", ALPHA, trans_abs, tb_end),
                    ("post_trans_neg", -ALPHA, trans_abs, tb_end),
                ])

            n_cond = len(conditions)
            input_tensor = torch.tensor(
                [input_ids] * n_cond, dtype=torch.long, device=device,
            )

            captured = {}
            handle = model.model.layers[STEER_LAYER].register_forward_hook(
                make_hook(conditions, posthoc_dir_bf16, captured)
            )

            with torch.no_grad():
                logits = model(input_tensor).logits[:, -1, :]
                probs = torch.softmax(logits.float(), dim=-1)

            handle.remove()

            # Projections from baseline hidden state
            h = captured["h"]
            knowing_proj = torch.dot(h, knowing_dir_norm.cpu().float()).item()
            posthoc_proj = torch.dot(h, posthoc_dir_norm.cpu().float()).item()

            probe_result = {
                "pos": pos,
                "frac": pos / think_length,
                "knowing_proj": knowing_proj,
                "posthoc_proj": posthoc_proj,
                "conditions": {},
            }

            for j, (name, alpha, start, end) in enumerate(conditions):
                p_mu = probs[j, MU_TOKEN_ID].item()
                p_nu = probs[j, NU_TOKEN_ID].item()
                p_correct = p_mu if expected == "Mu" else p_nu
                probe_result["conditions"][name] = {
                    "p_correct": p_correct,
                    "p_mu": p_mu,
                    "p_nu": p_nu,
                }

            probes.append(probe_result)

        if (trace_idx + 1) % 5 == 0 or trace_idx == 0:
            print(f"  [{trace_idx+1}/{len(scan_entries)}] {pid} ({category}, "
                  f"{len(probes)} probes)", flush=True)

        all_results.append({
            "problem_id": pid,
            "dataset": info["dataset"],
            "category": category,
            "think_length": think_length,
            "expected_token": expected,
            "transition_frac": transition_frac,
            "probes": probes,
        })

        if (trace_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    # ---- Save ----
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR}/results.json")

    # ================================================================
    # ANALYSIS AND PLOTS
    # ================================================================
    frac_grid = np.linspace(0.05, 0.95, 20)
    posthoc_results = [r for r in all_results if r["category"] == "post_hoc"]

    # ---- Plot 1: Post-hoc targeted steering ----
    if posthoc_results:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Panel A: Baseline P(correct) by category
        ax = axes[0]
        for cat in CATS_ORDER:
            cat_rs = [r for r in all_results if r["category"] == cat]
            if not cat_rs:
                continue
            curves = []
            for r in cat_rs:
                fracs = [p["frac"] for p in r["probes"]]
                pcs = [p["conditions"]["baseline"]["p_correct"] for p in r["probes"]]
                curves.append(np.interp(frac_grid, fracs, pcs))
            ax.plot(frac_grid, np.mean(curves, axis=0),
                    color=CAT_COLORS.get(cat, "gray"), linewidth=2,
                    label=f"{cat} (n={len(cat_rs)})")
        ax.set_xlabel("Fraction through think block")
        ax.set_ylabel("P(correct)")
        ax.set_title("Baseline P(correct) by Category")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)

        # Panel B: Post-hoc traces -- negative steering (prevention)
        ax = axes[1]
        for cond_name, label, color, ls in [
            ("baseline", "Baseline", "gray", "-"),
            ("think_neg", f"Full think -{ALPHA}", "blue", "-"),
            ("pre_trans_neg", f"Pre-trans only -{ALPHA}", "green", "--"),
            ("post_trans_neg", f"Post-trans only -{ALPHA}", "purple", "--"),
        ]:
            frac_vals, pc_vals = [], []
            for r in posthoc_results:
                for p in r["probes"]:
                    if cond_name in p["conditions"]:
                        frac_vals.append(p["frac"])
                        pc_vals.append(p["conditions"][cond_name]["p_correct"])
            if not frac_vals:
                continue
            # Bin and average
            bins = np.linspace(0.0, 1.0, 12)
            centers, means = [], []
            for b0, b1 in zip(bins[:-1], bins[1:]):
                in_bin = [pc_vals[i] for i in range(len(frac_vals))
                          if b0 <= frac_vals[i] < b1]
                if in_bin:
                    centers.append((b0 + b1) / 2)
                    means.append(np.mean(in_bin))
            if centers:
                ax.plot(centers, means, color=color, linewidth=2, linestyle=ls,
                        marker="o", markersize=4, label=label)
        ax.set_xlabel("Fraction through think block")
        ax.set_ylabel("P(correct)")
        ax.set_title(f"Post-hoc traces: Negative Steering (prevention)")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.05)

        # Panel C: Post-hoc traces -- positive steering (induction)
        ax = axes[2]
        for cond_name, label, color, ls in [
            ("baseline", "Baseline", "gray", "-"),
            ("think_pos", f"Full think +{ALPHA}", "red", "-"),
            ("pre_trans_pos", f"Pre-trans only +{ALPHA}", "orange", "--"),
            ("post_trans_pos", f"Post-trans only +{ALPHA}", "darkred", "--"),
        ]:
            frac_vals, pc_vals = [], []
            for r in posthoc_results:
                for p in r["probes"]:
                    if cond_name in p["conditions"]:
                        frac_vals.append(p["frac"])
                        pc_vals.append(p["conditions"][cond_name]["p_correct"])
            if not frac_vals:
                continue
            bins = np.linspace(0.0, 1.0, 12)
            centers, means = [], []
            for b0, b1 in zip(bins[:-1], bins[1:]):
                in_bin = [pc_vals[i] for i in range(len(frac_vals))
                          if b0 <= frac_vals[i] < b1]
                if in_bin:
                    centers.append((b0 + b1) / 2)
                    means.append(np.mean(in_bin))
            if centers:
                ax.plot(centers, means, color=color, linewidth=2, linestyle=ls,
                        marker="o", markersize=4, label=label)
        ax.set_xlabel("Fraction through think block")
        ax.set_ylabel("P(correct)")
        ax.set_title(f"Post-hoc traces: Positive Steering (induction)")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "posthoc_targeted.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # ---- Plot 2: Steering effects by category ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (cond, title) in enumerate([
        ("think_pos", f"+{ALPHA} Steering (toward performative)"),
        ("think_neg", f"-{ALPHA} Steering (toward genuine)"),
    ]):
        ax = axes[ax_idx]
        for cat in CATS_ORDER:
            cat_rs = [r for r in all_results if r["category"] == cat]
            if not cat_rs:
                continue
            deltas = []
            for r in cat_rs:
                fracs = [p["frac"] for p in r["probes"]]
                ds = [p["conditions"][cond]["p_correct"] -
                      p["conditions"]["baseline"]["p_correct"] for p in r["probes"]]
                deltas.append(np.interp(frac_grid, fracs, ds))
            mean_d = np.mean(deltas, axis=0)
            sem = np.std(deltas, axis=0) / max(1, np.sqrt(len(deltas)))
            ax.plot(frac_grid, mean_d, color=CAT_COLORS.get(cat, "gray"),
                    linewidth=2, label=f"{cat} (n={len(cat_rs)})")
            ax.fill_between(frac_grid, mean_d - sem, mean_d + sem,
                            color=CAT_COLORS.get(cat, "gray"), alpha=0.15)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
        ax.set_xlabel("Fraction through think block")
        ax.set_ylabel("Delta P(correct)")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "category_effects.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Plot 3: Knowing-conditional analysis ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    all_probes_flat = []
    for r in all_results:
        for p in r["probes"]:
            all_probes_flat.append({
                "category": r["category"],
                "frac": p["frac"],
                "knowing_proj": p["knowing_proj"],
                "posthoc_proj": p["posthoc_proj"],
                "p_baseline": p["conditions"]["baseline"]["p_correct"],
                "delta_pos": (p["conditions"]["think_pos"]["p_correct"] -
                              p["conditions"]["baseline"]["p_correct"]),
                "delta_neg": (p["conditions"]["think_neg"]["p_correct"] -
                              p["conditions"]["baseline"]["p_correct"]),
            })

    # Panel A: Knowing proj vs delta_P for +alpha (scatter)
    ax = axes[0]
    for cat in CATS_ORDER:
        pts = [p for p in all_probes_flat if p["category"] == cat]
        if not pts:
            continue
        ax.scatter([p["knowing_proj"] for p in pts],
                   [p["delta_pos"] for p in pts],
                   alpha=0.3, s=12, color=CAT_COLORS.get(cat, "gray"),
                   label=cat)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Knowing projection (baseline)")
    ax.set_ylabel(f"Delta P(correct) from +{ALPHA}")
    ax.set_title("Knowing State vs Positive Steering Effect")
    ax.legend(fontsize=7)

    # Panel B: Baseline P(correct) vs delta_P for +alpha
    ax = axes[1]
    for cat in CATS_ORDER:
        pts = [p for p in all_probes_flat if p["category"] == cat]
        if not pts:
            continue
        ax.scatter([p["p_baseline"] for p in pts],
                   [p["delta_pos"] for p in pts],
                   alpha=0.3, s=12, color=CAT_COLORS.get(cat, "gray"),
                   label=cat)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Baseline P(correct)")
    ax.set_ylabel(f"Delta P(correct) from +{ALPHA}")
    ax.set_title("Baseline Confidence vs Positive Steering Effect")
    ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "knowing_conditional.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ---- Plot 4: Individual post-hoc trace curves ----
    if posthoc_results:
        n_ph = len(posthoc_results)
        fig, axes = plt.subplots(n_ph, 1, figsize=(12, 3.5 * n_ph), squeeze=False)
        for i, r in enumerate(posthoc_results):
            ax = axes[i, 0]
            fracs = [p["frac"] for p in r["probes"]]
            for cond_name, label, color, ls in [
                ("baseline", "Baseline", "gray", "-"),
                ("think_pos", f"+{ALPHA}", "red", "-"),
                ("think_neg", f"-{ALPHA}", "blue", "-"),
                ("pre_trans_neg", f"Pre-trans -{ALPHA}", "green", "--"),
                ("post_trans_neg", f"Post-trans -{ALPHA}", "purple", "--"),
            ]:
                pcs = []
                fs = []
                for p in r["probes"]:
                    if cond_name in p["conditions"]:
                        fs.append(p["frac"])
                        pcs.append(p["conditions"][cond_name]["p_correct"])
                if pcs:
                    ax.plot(fs, pcs, color=color, linewidth=1.5, linestyle=ls,
                            label=label)
            if r["transition_frac"]:
                ax.axvline(r["transition_frac"], color="black", linestyle=":",
                           alpha=0.5, label="transition")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{r['problem_id']} ({r['dataset']}) | "
                         f"tl={r['think_length']} | trans={r['transition_frac']:.2f}",
                         fontsize=9)
            if i == 0:
                ax.legend(fontsize=6, loc="lower right")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "posthoc_individual.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # ---- Summary ----
    print(f"\n--- Steering Summary (L{STEER_LAYER}, alpha=+-{ALPHA}) ---")
    for cat in CATS_ORDER:
        cat_rs = [r for r in all_results if r["category"] == cat]
        if not cat_rs:
            continue
        late_dp, late_dn = [], []
        for r in cat_rs:
            for p in r["probes"]:
                if p["frac"] > 0.7:
                    late_dp.append(p["conditions"]["think_pos"]["p_correct"] -
                                   p["conditions"]["baseline"]["p_correct"])
                    late_dn.append(p["conditions"]["think_neg"]["p_correct"] -
                                   p["conditions"]["baseline"]["p_correct"])
        if late_dp:
            print(f"  {cat:>15} (n={len(cat_rs):>2}): "
                  f"Delta(+{ALPHA},late)={np.mean(late_dp):+.4f}, "
                  f"Delta(-{ALPHA},late)={np.mean(late_dn):+.4f}")

    if posthoc_results:
        print(f"\n--- Post-hoc: Targeted vs Full (post-transition probes, frac>0.5) ---")
        for cond in ["baseline", "think_neg", "pre_trans_neg", "post_trans_neg",
                      "think_pos", "pre_trans_pos", "post_trans_pos"]:
            vals, bases = [], []
            for r in posthoc_results:
                for p in r["probes"]:
                    if cond in p["conditions"] and p["frac"] > 0.5:
                        vals.append(p["conditions"][cond]["p_correct"])
                        bases.append(p["conditions"]["baseline"]["p_correct"])
            if vals:
                delta = np.mean(vals) - np.mean(bases) if cond != "baseline" else 0
                print(f"  {cond:>18}: P={np.mean(vals):.4f} "
                      f"(Delta={delta:+.4f}, n={len(vals)})")

    print("\nDone.")


if __name__ == "__main__":
    main()
