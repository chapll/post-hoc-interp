"""Multi-layer causal steering on Pool C traces (scaled-up).

Tests generation-based steering across 9 layers and multiple alpha values.
For each trace and condition, measures tokens until </think> and answer
correctness.

Usage:
    python scripts/steer_generation_v2.py --pool C
    python scripts/steer_generation_v2.py --pool C --direction knowing
"""

import argparse
import gc
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
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

TRACES_DIR = "data/traces_v2/valid"
SCAN_RESULTS_BASE = "experiments/v2/scan"
DIRECTIONS_DIR = "experiments/v2/directions"
OUTPUT_DIR = "experiments/v2/steering"

STEER_LAYERS = [16, 18, 20, 22, 24, 26, 28, 30, 32]
NONZERO_ALPHAS = [-100, 50, 100]
EVAL_TRACE_INDEX = 1
MAX_NEW_TOKENS = 2048
THINK_END_TOKEN = 151668
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

MAX_PER_CATEGORY = 25


def make_hook(alpha_val, dir_vec):
    """Steering hook: only modify single-token generation steps."""
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output
        if h.shape[1] == 1 and alpha_val != 0:
            h = h.clone()
            h[0, 0, :] += alpha_val * dir_vec
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return output
    return hook_fn


def generate_and_measure(model, tokenizer, prefix, expected_token,
                         device, hook_handle=None):
    """Generate from prefix and measure tokens to </think> + correctness."""
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )

    generated_ids = output[0, len(prefix):].tolist()

    think_end_pos = None
    for j, tid in enumerate(generated_ids):
        if tid == THINK_END_TOKEN:
            think_end_pos = j
            break

    tokens_to_close = think_end_pos if think_end_pos is not None else len(generated_ids)

    answer_correct = None
    if think_end_pos is not None and think_end_pos + 5 < len(generated_ids):
        answer_region = tokenizer.decode(
            generated_ids[think_end_pos:think_end_pos + 20]
        )
        answer_correct = expected_token in answer_region

    return {
        "tokens_to_close": tokens_to_close,
        "found_think_end": think_end_pos is not None,
        "total_generated": len(generated_ids),
        "answer_correct": answer_correct,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="C")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--direction", default="posthoc",
                        choices=["posthoc", "knowing"])
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load direction at all steer layers
    dir_name = f"{args.direction}_direction.safetensors"
    dir_path = os.path.join(DIRECTIONS_DIR, dir_name)
    print(f"Loading {args.direction} direction from {dir_path}...")
    dir_data = load_file(dir_path)
    direction_by_layer = {}
    for l in STEER_LAYERS:
        direction_by_layer[l] = (
            dir_data["direction_normalized"][l].to(device).to(torch.bfloat16)
        )
        norm = dir_data["direction"][l].norm().item()
        print(f"  L{l}: norm={norm:.1f}")

    # Load scan results
    scan_path = os.path.join(SCAN_RESULTS_BASE, f"pool_{args.pool}", "scan_results.json")
    print(f"Loading scan results from {scan_path}...")
    with open(scan_path) as f:
        scan_results = json.load(f)

    by_category = {}
    for entry in scan_results:
        by_category.setdefault(entry["category"], []).append(entry)
    print("Scan categories:")
    for cat, entries in sorted(by_category.items()):
        print(f"  {cat}: {len(entries)}")

    # Select traces
    selected_entries = []
    for cat in ["post_hoc", "always_knows", "gradual", "late_learner"]:
        selected_entries.extend(by_category.get(cat, [])[:MAX_PER_CATEGORY])

    print(f"\nLoading {len(selected_entries)} traces...")
    traces = []
    skipped = 0
    for entry in selected_entries:
        pid = entry["problem_id"]
        path = os.path.join(TRACES_DIR, "deepmath", pid,
                            f"trace_{EVAL_TRACE_INDEX:02d}.json")
        if not os.path.isfile(path):
            skipped += 1
            continue
        with open(path) as f:
            trace = json.load(f)
        meta = trace["metadata"]
        if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
            continue
        think_start, think_end = meta["think_boundaries"][0]
        think_length = think_end - think_start - 1

        transition_frac = entry["metrics"].get("transition_frac")
        transition_pos = (int(transition_frac * think_length)
                          if transition_frac else None)

        gt = meta["ground_truth"]
        expected = meta["valid_token"] if gt == "Yes" else meta["invalid_token"]

        traces.append({
            "problem_id": pid,
            "category": entry["category"],
            "token_ids": trace["token_ids"],
            "think_start": think_start,
            "think_end": think_end,
            "think_length": think_length,
            "transition_pos": transition_pos,
            "transition_frac": transition_frac,
            "expected_token": expected,
        })

    if skipped > 0:
        print(f"  WARNING: {skipped} traces not found on disk (skipped)")
    cat_counts = {}
    for t in traces:
        cat_counts[t["category"]] = cat_counts.get(t["category"], 0) + 1
    print(f"  Loaded {len(traces)} traces: {cat_counts}")
    if len(traces) == 0:
        raise RuntimeError("No valid traces loaded. Check that traces exist and scan results match.")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    n_conditions = len(STEER_LAYERS) * len(NONZERO_ALPHAS) + 1  # +1 for baseline
    print(f"\nSteering: {n_conditions} conditions x {len(traces)} traces "
          f"= {n_conditions * len(traces)} generations")

    all_results = []
    for trace_idx, trace_info in enumerate(traces):
        pid = trace_info["problem_id"]
        category = trace_info["category"]
        token_ids = trace_info["token_ids"]
        think_start = trace_info["think_start"]
        think_length = trace_info["think_length"]
        transition_pos = trace_info["transition_pos"]

        # Start point: transition for post_hoc, 30% for others
        if category == "post_hoc" and transition_pos is not None:
            gen_start = think_start + 1 + transition_pos
        else:
            gen_start = think_start + 1 + int(0.3 * think_length)

        prefix = token_ids[:gen_start]
        original_remaining = trace_info["think_end"] - gen_start

        trace_results = {
            "problem_id": pid,
            "category": category,
            "think_length": think_length,
            "transition_pos": transition_pos,
            "gen_start_frac": (gen_start - think_start - 1) / think_length,
            "original_remaining_tokens": original_remaining,
        }

        # Baseline (no steering)
        baseline = generate_and_measure(
            model, tokenizer, prefix, trace_info["expected_token"], device,
        )
        trace_results["baseline"] = baseline

        print(f"  [{trace_idx+1}/{len(traces)}] {pid} ({category}) "
              f"baseline={baseline['tokens_to_close']}", flush=True)

        # Steered conditions
        conditions = []
        for layer in STEER_LAYERS:
            for alpha in NONZERO_ALPHAS:
                handle = model.model.layers[layer].register_forward_hook(
                    make_hook(alpha, direction_by_layer[layer])
                )

                result = generate_and_measure(
                    model, tokenizer, prefix,
                    trace_info["expected_token"], device,
                )

                handle.remove()

                delta = result["tokens_to_close"] - baseline["tokens_to_close"]
                conditions.append({
                    "layer": layer,
                    "alpha": alpha,
                    **result,
                    "delta": delta,
                })

        trace_results["conditions"] = conditions
        all_results.append(trace_results)
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    results_path = os.path.join(
        OUTPUT_DIR, f"steering_{args.direction}_results.json",
    )
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Summary table
    print("\n--- Mean delta tokens by layer and alpha ---")
    header = f"{'Category':>15} {'Layer':>6}"
    for a in NONZERO_ALPHAS:
        header += f"  a={a:+5d}"
    print(header)

    for category in ["post_hoc", "always_knows", "gradual", "late_learner"]:
        cat_results = [r for r in all_results if r["category"] == category]
        if not cat_results:
            continue
        for layer in STEER_LAYERS:
            line = f"{category:>15} L{layer:>4}"
            for alpha in NONZERO_ALPHAS:
                deltas = []
                for r in cat_results:
                    for c in r["conditions"]:
                        if c["layer"] == layer and c["alpha"] == alpha:
                            deltas.append(c["delta"])
                if deltas:
                    line += f"  {np.mean(deltas):>+7.0f}"
                else:
                    line += f"  {'n/a':>7}"
            print(line)

    # Plot: mean delta by layer for each category
    fig, axes = plt.subplots(1, len(NONZERO_ALPHAS), figsize=(6 * len(NONZERO_ALPHAS), 6))
    if len(NONZERO_ALPHAS) == 1:
        axes = [axes]
    categories_to_plot = ["post_hoc", "always_knows", "gradual"]
    colors = {"post_hoc": "coral", "always_knows": "steelblue", "gradual": "green"}

    for ax_idx, alpha in enumerate(NONZERO_ALPHAS):
        ax = axes[ax_idx]
        for category in categories_to_plot:
            cat_results = [r for r in all_results if r["category"] == category]
            if not cat_results:
                continue
            mean_deltas = []
            for layer in STEER_LAYERS:
                deltas = []
                for r in cat_results:
                    for c in r["conditions"]:
                        if c["layer"] == layer and c["alpha"] == alpha:
                            deltas.append(c["delta"])
                mean_deltas.append(np.mean(deltas) if deltas else 0)
            ax.plot(STEER_LAYERS, mean_deltas, "-o", markersize=4,
                    label=category, color=colors.get(category, "gray"))

        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean delta tokens")
        ax.set_title(f"alpha={alpha:+d}")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"steering_{args.direction}_by_layer.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Correctness summary
    print("\n--- Answer Correctness ---")
    for category in ["post_hoc", "always_knows", "gradual", "late_learner"]:
        cat_results = [r for r in all_results if r["category"] == category]
        if not cat_results:
            continue
        baseline_correct = sum(
            1 for r in cat_results
            if r["baseline"]["answer_correct"] is True
        )
        total = len(cat_results)
        print(f"  {category}: baseline={baseline_correct}/{total}")
        for alpha in NONZERO_ALPHAS:
            for layer in [20, 24, 28]:
                correct = sum(
                    1 for r in cat_results
                    for c in r["conditions"]
                    if c["layer"] == layer and c["alpha"] == alpha
                    and c["answer_correct"] is True
                )
                print(f"    a={alpha:+d} L{layer}: {correct}/{total}")

    print("\nDone.")


if __name__ == "__main__":
    main()
