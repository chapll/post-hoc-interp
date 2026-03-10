"""Induction test: can the post-hoc direction induce premature conclusion
in traces that don't naturally show post-hoc reasoning?

Steer ALL 50 scanned traces from the START of the think block.
- +alpha: push toward "concluded" state -> should shorten think block
- -alpha: push toward "genuine reasoning" state -> should lengthen think block

Key test: for always_knows / gradual / late_learner traces, does +alpha
cause the model to conclude early AND still get the right answer?
If yes: the direction reliably induces premature conclusion (post-hoc-like).
If answer becomes wrong: the steering truncated genuine reasoning.

Usage:
    python scripts/steer_induction.py
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
OUTPUT_DIR = "experiments/activation_diff/steering_induction"

STEER_LAYER = 24
ALPHAS = [0, -100, 50, 100, 150]
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

    # Load ALL 50 traces
    print("Loading traces...")
    traces = []
    for entry in scan_results:
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
            "expected_token": expected,
        })

    cats = {}
    for t in traces:
        cats[t["category"]] = cats.get(t["category"], 0) + 1
    print(f"  {len(traces)} traces: {cats}")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    print(f"Running induction test (L{STEER_LAYER}, alphas={ALPHAS}, "
          f"max_new={MAX_NEW_TOKENS}, steering from think_start)...")

    results = []
    for trace_idx, trace_info in enumerate(traces):
        pid = trace_info["problem_id"]
        category = trace_info["category"]
        token_ids = trace_info["token_ids"]
        think_start = trace_info["think_start"]
        think_length = trace_info["think_length"]

        # Start from 5 tokens into think block (skip <think> token itself
        # plus a few initial tokens to establish context)
        gen_start = think_start + 1 + min(5, think_length // 10)
        prefix = token_ids[:gen_start]
        original_remaining = trace_info["think_end"] - gen_start

        trace_results = {
            "problem_id": pid,
            "category": category,
            "think_length": think_length,
            "original_remaining_tokens": original_remaining,
            "conditions": [],
        }

        for alpha in ALPHAS:
            def make_hook(alpha_val, dir_vec):
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

            # Find </think>
            think_end_pos = None
            for j, tid in enumerate(generated_ids):
                if tid == THINK_END_TOKEN:
                    think_end_pos = j
                    break

            tokens_to_close = think_end_pos if think_end_pos is not None else len(generated_ids)

            # Check answer
            answer_correct = None
            answer_text = ""
            if think_end_pos is not None and think_end_pos + 3 < len(generated_ids):
                answer_region = tokenizer.decode(
                    generated_ids[think_end_pos:min(think_end_pos + 20, len(generated_ids))]
                )
                answer_text = answer_region.strip()[:100]
                expected = trace_info["expected_token"]
                answer_correct = expected in answer_region

            trace_results["conditions"].append({
                "alpha": alpha,
                "tokens_to_close": tokens_to_close,
                "found_think_end": think_end_pos is not None,
                "total_generated": len(generated_ids),
                "answer_correct": answer_correct,
                "answer_text": answer_text,
            })

            correct_str = ""
            if answer_correct is True:
                correct_str = "  CORRECT"
            elif answer_correct is False:
                correct_str = "  WRONG"

            print(f"  [{trace_idx+1}/{len(traces)}] {pid} ({category}) "
                  f"a={alpha:+4d}: ttc={tokens_to_close:>5}"
                  f"{'  (no </think>)' if think_end_pos is None else ''}"
                  f"{correct_str}", flush=True)

        results.append(trace_results)
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    # Save results
    out_path = os.path.join(OUTPUT_DIR, "induction_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    # ---- Analysis ----
    categories = sorted(set(r["category"] for r in results))

    print("\n--- Induction Summary (steering from think_start) ---")
    for category in categories:
        cat_results = [r for r in results if r["category"] == category]
        print(f"\n{category} (n={len(cat_results)}):")
        for alpha in ALPHAS:
            lengths = []
            correct_count = 0
            wrong_count = 0
            no_close = 0
            total = 0
            for r in cat_results:
                for c in r["conditions"]:
                    if c["alpha"] == alpha:
                        lengths.append(c["tokens_to_close"])
                        total += 1
                        if c["answer_correct"] is True:
                            correct_count += 1
                        elif c["answer_correct"] is False:
                            wrong_count += 1
                        if not c["found_think_end"]:
                            no_close += 1
            if lengths:
                print(f"  a={alpha:+4d}: mean_ttc={np.mean(lengths):>7.0f}, "
                      f"median={np.median(lengths):>7.0f}, "
                      f"correct={correct_count}/{total}, "
                      f"wrong={wrong_count}/{total}, "
                      f"no_close={no_close}/{total}")

    # Per-alpha comparison: shortening ratio vs baseline
    print("\n--- Shortening Ratio (tokens_to_close / baseline) ---")
    for category in categories:
        cat_results = [r for r in results if r["category"] == category]
        print(f"\n{category} (n={len(cat_results)}):")
        for alpha in ALPHAS:
            if alpha == 0:
                continue
            ratios = []
            for r in cat_results:
                baseline = [c for c in r["conditions"] if c["alpha"] == 0]
                steered = [c for c in r["conditions"] if c["alpha"] == alpha]
                if baseline and steered:
                    bl = baseline[0]["tokens_to_close"]
                    st = steered[0]["tokens_to_close"]
                    if bl > 0:
                        ratios.append(st / bl)
            if ratios:
                print(f"  a={alpha:+4d}: mean_ratio={np.mean(ratios):.2f}, "
                      f"median={np.median(ratios):.2f}, "
                      f"shortened={sum(1 for r in ratios if r < 0.8)}/{len(ratios)}, "
                      f"lengthened={sum(1 for r in ratios if r > 1.2)}/{len(ratios)}")

    # ---- Plots ----

    # Plot 1: delta bar chart by category
    fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 5),
                             squeeze=False)
    nonzero_alphas = [a for a in ALPHAS if a != 0]

    for ax_idx, category in enumerate(categories):
        ax = axes[0, ax_idx]
        cat_results = [r for r in results if r["category"] == category]

        mean_deltas = []
        std_deltas = []
        for alpha in nonzero_alphas:
            deltas = []
            for r in cat_results:
                bl = [c["tokens_to_close"] for c in r["conditions"] if c["alpha"] == 0][0]
                st = [c["tokens_to_close"] for c in r["conditions"] if c["alpha"] == alpha][0]
                deltas.append(st - bl)
            mean_deltas.append(np.mean(deltas))
            std_deltas.append(np.std(deltas) / max(1, np.sqrt(len(deltas))))

        colors = ["blue" if a < 0 else "red" for a in nonzero_alphas]
        x = np.arange(len(nonzero_alphas))
        ax.bar(x, mean_deltas, yerr=std_deltas, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{a:+d}" for a in nonzero_alphas], fontsize=8)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Delta tokens to </think>")
        ax.set_title(f"{category} (n={len(cat_results)})")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "induction_deltas.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # Plot 2: answer correctness by alpha and category
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    width = 0.15
    for i, alpha in enumerate(ALPHAS):
        correct_rates = []
        for category in categories:
            cat_results = [r for r in results if r["category"] == category]
            correct = 0
            total_with_answer = 0
            for r in cat_results:
                for c in r["conditions"]:
                    if c["alpha"] == alpha and c["answer_correct"] is not None:
                        total_with_answer += 1
                        if c["answer_correct"]:
                            correct += 1
            rate = correct / max(1, total_with_answer)
            correct_rates.append(rate)
        color = "gray" if alpha == 0 else ("blue" if alpha < 0 else "red")
        ax.bar(x + i * width, correct_rates, width, label=f"a={alpha:+d}",
               color=color, alpha=0.7)

    ax.set_xticks(x + width * len(ALPHAS) / 2)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Answer Correctness Rate")
    ax.set_title("Answer Correctness Under Steering (from think_start)")
    ax.legend()
    ax.set_ylim(0, 1.1)

    path = os.path.join(OUTPUT_DIR, "induction_correctness.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plot 3: scatter of baseline ttc vs steered ttc for +100
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax_idx, alpha in enumerate([100, -100]):
        ax = axes[ax_idx]
        for category in categories:
            cat_results = [r for r in results if r["category"] == category]
            baselines = []
            steered = []
            for r in cat_results:
                bl = [c["tokens_to_close"] for c in r["conditions"] if c["alpha"] == 0][0]
                st = [c["tokens_to_close"] for c in r["conditions"] if c["alpha"] == alpha][0]
                baselines.append(bl)
                steered.append(st)
            ax.scatter(baselines, steered, label=category, alpha=0.6, s=30)

        ax.plot([0, 2048], [0, 2048], "k--", alpha=0.3, label="y=x")
        ax.set_xlabel("Baseline tokens to </think>")
        ax.set_ylabel(f"Steered (a={alpha:+d}) tokens to </think>")
        ax.set_title(f"Alpha={alpha:+d}")
        ax.legend(fontsize=7)
        ax.set_xlim(0, 2100)
        ax.set_ylim(0, 2100)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "induction_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
