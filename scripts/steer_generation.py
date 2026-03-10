"""Generation-based causal steering test for post-hoc reasoning.

Instead of measuring P(correct) at forced-answer probes, measure how many
tokens the model generates before closing the think block when steered.

Hypothesis:
  - Subtracting post-hoc direction (toward genuine) at the transition point
    causes the model to close </think> sooner -- the post-hoc reasoning is
    no longer sustained.
  - Adding post-hoc direction (toward performative) extends generation.

For post-hoc traces: generate from the transition point with/without steering.
For always_knows traces: generate from 30% through think block for comparison.

Metric: tokens generated until </think>.

Usage:
    python scripts/steer_generation.py
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
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Config ----
TRACES_DIR = "data/traces/valid"
SCAN_RESULTS = "experiments/activation_diff/posthoc_scan/scan_results.json"
POSTHOC_DIR_PATH = "experiments/activation_diff/posthoc_direction/posthoc_direction.safetensors"
OUTPUT_DIR = "experiments/activation_diff/steering_generation"

STEER_LAYER = 24
ALPHAS = [0, -100, -200, 100, 200]
EVAL_TRACE_INDEX = 1
MAX_NEW_TOKENS = 2048
THINK_END_TOKEN = 151668  # </think>
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335


def main():
    device = "cuda:0"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load direction
    print("Loading post-hoc direction...")
    data = load_file(POSTHOC_DIR_PATH)
    direction_norm = data["direction_normalized"][STEER_LAYER].to(device).to(torch.bfloat16)
    dir_raw_norm = data["direction"][STEER_LAYER].norm().item()
    print(f"  Post-hoc norm (L{STEER_LAYER}): {dir_raw_norm:.1f}")

    # Load scan results
    with open(SCAN_RESULTS) as f:
        scan_results = json.load(f)

    posthoc_entries = [e for e in scan_results if e["category"] == "post_hoc"]
    always_knows_entries = [e for e in scan_results if e["category"] == "always_knows"]

    # Load traces
    print("Loading traces...")
    traces = []
    for entry in posthoc_entries + always_knows_entries[:10]:
        pid = entry["problem_id"]
        ds = entry["dataset"]
        path = os.path.join(TRACES_DIR, ds, pid, f"trace_{EVAL_TRACE_INDEX:02d}.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            trace = json.load(f)
        meta = trace["metadata"]
        if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
            continue
        think_start, think_end = meta["think_boundaries"][0]
        think_length = think_end - think_start - 1

        transition_frac = entry["metrics"].get("transition_frac")
        transition_pos = int(transition_frac * think_length) if transition_frac else None

        gt = meta["ground_truth"]
        expected = meta["valid_token"] if gt == "Yes" else meta["invalid_token"]

        traces.append({
            "problem_id": pid,
            "dataset": ds,
            "category": entry["category"],
            "token_ids": trace["token_ids"],
            "think_start": think_start,
            "think_end": think_end,
            "think_length": think_length,
            "transition_pos": transition_pos,
            "transition_frac": transition_frac,
            "expected_token": expected,
        })

    print(f"  {len(traces)} traces loaded "
          f"({sum(1 for t in traces if t['category']=='post_hoc')} post_hoc, "
          f"{sum(1 for t in traces if t['category']=='always_knows')} always_knows)")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    print(f"Running generation steering (L{STEER_LAYER}, "
          f"alphas={ALPHAS}, max_new={MAX_NEW_TOKENS})...")

    results = []
    for trace_idx, trace_info in enumerate(traces):
        pid = trace_info["problem_id"]
        category = trace_info["category"]
        token_ids = trace_info["token_ids"]
        think_start = trace_info["think_start"]
        think_length = trace_info["think_length"]
        transition_pos = trace_info["transition_pos"]

        # For post-hoc: start from transition point
        # For always_knows: start from 30% through think block
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
            "conditions": [],
        }

        for alpha in ALPHAS:
            # Hook: steer only during generation (seq_len == 1 means single-token step)
            def make_hook(alpha_val, dir_vec):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        h = output[0]
                    else:
                        h = output
                    # Only steer single-token forward passes (generation steps)
                    if h.shape[1] == 1 and alpha_val != 0:
                        h = h.clone()
                        h[0, 0, :] += alpha_val * dir_vec
                        if isinstance(output, tuple):
                            return (h,) + output[1:]
                        return h
                    return output
                return hook_fn

            handle = model.model.layers[STEER_LAYER].register_forward_hook(
                make_hook(alpha, direction_norm)
            )

            input_ids = torch.tensor([prefix], dtype=torch.long, device=device)

            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

            handle.remove()

            generated_ids = output[0, len(prefix):].tolist()

            # Find </think> token
            think_end_pos = None
            for j, tid in enumerate(generated_ids):
                if tid == THINK_END_TOKEN:
                    think_end_pos = j
                    break

            tokens_to_close = think_end_pos if think_end_pos is not None else len(generated_ids)
            snippet = tokenizer.decode(
                generated_ids[:min(tokens_to_close + 30, len(generated_ids))]
            )

            # Check answer after </think> if found
            answer_correct = None
            if think_end_pos is not None and think_end_pos + 5 < len(generated_ids):
                answer_region = tokenizer.decode(generated_ids[think_end_pos:think_end_pos + 20])
                expected = trace_info["expected_token"]
                answer_correct = expected in answer_region

            trace_results["conditions"].append({
                "alpha": alpha,
                "tokens_to_close": tokens_to_close,
                "found_think_end": think_end_pos is not None,
                "total_generated": len(generated_ids),
                "answer_correct": answer_correct,
                "snippet_start": snippet[:300],
                "snippet_end": tokenizer.decode(
                    generated_ids[max(0, tokens_to_close - 50):tokens_to_close + 30]
                )[:300] if tokens_to_close > 50 else "",
            })

            print(f"  [{trace_idx+1}/{len(traces)}] {pid} ({category}) "
                  f"a={alpha:+4d}: tokens_to_close={tokens_to_close:>5}"
                  f"{'  (no </think>)' if think_end_pos is None else ''}"
                  f"{'  CORRECT' if answer_correct else '  WRONG' if answer_correct is False else ''}",
                  flush=True)

        results.append(trace_results)
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    # Save results
    with open(os.path.join(OUTPUT_DIR, "generation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR}/generation_results.json")

    # ---- Summary ----
    print("\n--- Generation Length Summary ---")
    for category in ["post_hoc", "always_knows"]:
        cat_results = [r for r in results if r["category"] == category]
        if not cat_results:
            continue
        print(f"\n{category} (n={len(cat_results)}):")
        for alpha in ALPHAS:
            lengths = []
            for r in cat_results:
                for c in r["conditions"]:
                    if c["alpha"] == alpha:
                        lengths.append(c["tokens_to_close"])
            if lengths:
                print(f"  alpha={alpha:+4d}: mean={np.mean(lengths):>7.0f}, "
                      f"median={np.median(lengths):>7.0f}, "
                      f"std={np.std(lengths):>7.0f}, "
                      f"range=[{min(lengths)}, {max(lengths)}]")

    # Per-trace table
    print("\n--- Per-Trace: tokens to </think> ---")
    header = f"{'Problem ID':>45} {'Cat':>12} {'orig_rem':>8}"
    for alpha in ALPHAS:
        header += f"  a={alpha:+4d}"
    print(header)
    for r in results:
        baseline_len = None
        line = f"{r['problem_id']:>45} {r['category']:>12} {r['original_remaining_tokens']:>8}"
        for c in r["conditions"]:
            if c["alpha"] == 0:
                baseline_len = c["tokens_to_close"]
            ttc = c["tokens_to_close"]
            if c["alpha"] == 0:
                line += f"  {ttc:>6}"
            else:
                delta = ttc - baseline_len if baseline_len is not None else 0
                sign = "+" if delta > 0 else ""
                line += f"  {ttc:>4}({sign}{delta})"
        print(line)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, category in enumerate(["post_hoc", "always_knows"]):
        ax = axes[ax_idx]
        cat_results = [r for r in results if r["category"] == category]
        if not cat_results:
            continue

        # Normalize: tokens_to_close / baseline tokens_to_close
        alpha_labels = [f"{a:+d}" for a in ALPHAS]
        x = np.arange(len(ALPHAS))
        width = 0.15

        for i, r in enumerate(cat_results):
            baseline_len = None
            ratios = []
            for c in r["conditions"]:
                if c["alpha"] == 0:
                    baseline_len = c["tokens_to_close"]
            for c in r["conditions"]:
                ratio = c["tokens_to_close"] / max(baseline_len, 1)
                ratios.append(ratio)
            ax.bar(x + i * width, ratios, width, alpha=0.6,
                   label=r["problem_id"][:20] if i < 5 else None)

        ax.set_xlabel("Alpha")
        ax.set_ylabel("Tokens to </think> (ratio to baseline)")
        ax.set_title(f"{category} traces (n={len(cat_results)})")
        ax.set_xticks(x + width * len(cat_results) / 2)
        ax.set_xticklabels(alpha_labels)
        ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
        if len(cat_results) <= 5:
            ax.legend(fontsize=6)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "generation_lengths.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # Delta plot (cleaner)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, category in enumerate(["post_hoc", "always_knows"]):
        ax = axes[ax_idx]
        cat_results = [r for r in results if r["category"] == category]
        if not cat_results:
            continue

        nonzero_alphas = [a for a in ALPHAS if a != 0]
        mean_deltas = []
        std_deltas = []
        for alpha in nonzero_alphas:
            deltas = []
            for r in cat_results:
                baseline_len = [c["tokens_to_close"] for c in r["conditions"]
                                if c["alpha"] == 0][0]
                steered_len = [c["tokens_to_close"] for c in r["conditions"]
                               if c["alpha"] == alpha][0]
                deltas.append(steered_len - baseline_len)
            mean_deltas.append(np.mean(deltas))
            std_deltas.append(np.std(deltas) / max(1, np.sqrt(len(deltas))))

        colors = ["blue" if a < 0 else "red" for a in nonzero_alphas]
        x = np.arange(len(nonzero_alphas))
        ax.bar(x, mean_deltas, yerr=std_deltas, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:+d}" for a in nonzero_alphas])
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Delta tokens to </think> vs baseline")
        ax.set_title(f"{category} (n={len(cat_results)})")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "generation_deltas.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
