"""Causal validation of post-hoc direction via activation steering.

Steers with the post-hoc direction (performative vs genuine reasoning).
If causal:
- Positive alpha (toward performative) at early positions should NOT change
  P(correct) much (it's about reasoning mode, not answer knowledge)
- But it might change the model's generated text quality/length
- Negative alpha (toward genuine) at late positions might reduce confidence
  or change reasoning patterns

Also compares effect to steering with knowing direction as control.

Usage:
    python scripts/steer_posthoc_direction.py
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
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
POSTHOC_DIR_PATH = "experiments/activation_diff/posthoc_direction/posthoc_direction.safetensors"
KNOWING_DIR_PATH = "experiments/activation_diff/knowing_direction.safetensors"
TRACES_DIR = "data/traces/valid"
OUTPUT_DIR = "experiments/activation_diff/steering_posthoc"

STEER_LAYER = 24
ALPHAS = [-100, -50, -20, 0, 20, 50, 100]
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

N_TEST_TRACES = 8
TEST_TRACE_INDEX = 1
PROBE_SPACING = 25
SEED = 137


def load_test_traces(traces_dir, n_traces, trace_index, seed):
    """Load diverse held-out traces."""
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
    by_ds = {}
    for ds, pid, path in all_traces:
        by_ds.setdefault(ds, []).append((ds, pid, path))
    selected = []
    per_ds = max(1, n_traces // len(by_ds))
    for ds in sorted(by_ds.keys()):
        selected.extend(by_ds[ds][:per_ds])
    remaining = [t for t in all_traces if t not in selected]
    while len(selected) < n_traces and remaining:
        selected.append(remaining.pop(0))
    selected = selected[:n_traces]

    loaded = []
    for ds, pid, path in selected:
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


def steer_and_probe(model, trace_data, dir_vec, layer_idx, alphas, device, spacing):
    """Probe with batched steering across alpha values."""
    token_ids = trace_data["token_ids"]
    think_start = trace_data["think_start"]
    think_length = trace_data["think_length"]
    expected = trace_data["expected_token"]

    prefix_len = think_start + 1
    positions = list(range(1, think_length + 1, spacing))
    if positions[-1] != think_length:
        positions.append(think_length)

    dir_vec_dev = dir_vec[layer_idx].to(device=device, dtype=torch.bfloat16)
    n_alphas = len(alphas)
    alphas_tensor = torch.tensor(
        alphas, device=device, dtype=torch.bfloat16,
    ).view(-1, 1, 1)

    results = {alpha: [] for alpha in alphas}

    for pos in positions:
        cache_len = prefix_len + pos
        input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS
        input_tensor = torch.tensor(
            [input_ids] * n_alphas, dtype=torch.long, device=device,
        )

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0] + alphas_tensor * dir_vec_dev
                return (h,) + output[1:]
            elif isinstance(output, torch.Tensor):
                return output + alphas_tensor * dir_vec_dev
            else:
                modified = output.copy()
                first_key = list(modified.keys())[0]
                modified[first_key] = modified[first_key] + alphas_tensor * dir_vec_dev
                return modified

        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

        with torch.no_grad():
            logits = model(input_tensor).logits[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)

        handle.remove()

        for j, alpha in enumerate(alphas):
            p_mu = probs[j, MU_TOKEN_ID].item()
            p_nu = probs[j, NU_TOKEN_ID].item()
            p_correct = p_mu if expected == "Mu" else p_nu
            results[alpha].append({
                "pos": pos,
                "frac": pos / think_length,
                "p_correct": p_correct,
                "p_mu": p_mu,
                "p_nu": p_nu,
                "p_sum": p_mu + p_nu,
            })

    return results


def plot_comparison(posthoc_results, knowing_results, alphas, output_dir):
    """Side-by-side comparison of steering with both directions."""
    os.makedirs(output_dir, exist_ok=True)

    cmap = plt.cm.RdBu_r
    alpha_range = max(abs(a) for a in alphas)
    colors = {a: cmap((a + alpha_range) / (2 * alpha_range)) for a in alphas}

    common_fracs = np.linspace(0, 1, 50)

    # Side-by-side mean curves
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    for ax, (results, title) in zip(axes, [
        (posthoc_results, "Post-hoc Direction Steering"),
        (knowing_results, "Knowing Direction Steering"),
    ]):
        for alpha in alphas:
            all_interp = []
            for trace_data, steer_results in results:
                probes = steer_results[alpha]
                fracs = np.array([r["frac"] for r in probes])
                p_correct = np.array([r["p_correct"] for r in probes])
                interp = np.interp(common_fracs, fracs, p_correct)
                all_interp.append(interp)
            mean_p = np.mean(all_interp, axis=0)
            lw = 2.5 if alpha == 0 else 1.2
            ls = "-" if alpha >= 0 else "--"
            ax.plot(common_fracs, mean_p, color=colors[alpha], linewidth=lw,
                    linestyle=ls, alpha=0.8, label=f"alpha={alpha}")

        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Fraction of thinking")
        ax.set_ylabel("Mean P(correct)")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    path = os.path.join(output_dir, "comparison_mean.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Dose-response comparison at fixed fractions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    target_fracs = [0.1, 0.3, 0.5]

    for row, (results, dir_name) in enumerate([
        (posthoc_results, "Post-hoc"),
        (knowing_results, "Knowing"),
    ]):
        for col, target_frac in enumerate(target_fracs):
            ax = axes[row, col]
            trace_points = []
            for trace_data, steer_results in results:
                alpha_vals = []
                p_vals = []
                for alpha in alphas:
                    probes = steer_results[alpha]
                    fracs = [r["frac"] for r in probes]
                    idx = int(np.argmin([abs(f - target_frac) for f in fracs]))
                    alpha_vals.append(alpha)
                    p_vals.append(probes[idx]["p_correct"])
                ax.plot(alpha_vals, p_vals, "o-", alpha=0.3, markersize=3,
                        color="steelblue")
                trace_points.append(p_vals)

            mean_dose = np.mean(trace_points, axis=0)
            ax.plot(alphas, mean_dose, "s-", color="black", linewidth=2,
                    markersize=6, label="mean")
            ax.set_title(f"{dir_name} -- frac~{target_frac}")
            ax.set_xlabel("Alpha")
            ax.set_ylabel("P(correct)")
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
            ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "dose_response_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Per-trace curves for post-hoc direction
    n_traces = len(posthoc_results)
    fig, axes = plt.subplots(n_traces, 1, figsize=(14, 3.5 * n_traces), squeeze=False)
    for i, (trace_data, steer_results) in enumerate(posthoc_results):
        ax = axes[i, 0]
        for alpha in alphas:
            probes = steer_results[alpha]
            fracs = [r["frac"] for r in probes]
            p_correct = [r["p_correct"] for r in probes]
            lw = 2.5 if alpha == 0 else 1.2
            ls = "-" if alpha >= 0 else "--"
            ax.plot(fracs, p_correct, color=colors[alpha], linewidth=lw,
                    linestyle=ls, alpha=0.8, label=f"alpha={alpha}")
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(
            f"{trace_data['problem_id']} ({trace_data['dataset']}) | "
            f"exp={trace_data['expected_token']} | tl={trace_data['think_length']}",
            fontsize=10,
        )
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "posthoc_steering_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Sanity check: P(Mu)+P(Nu) for post-hoc direction
    fig, ax = plt.subplots(figsize=(12, 6))
    for alpha in alphas:
        all_interp = []
        for trace_data, steer_results in posthoc_results:
            probes = steer_results[alpha]
            fracs = np.array([r["frac"] for r in probes])
            p_sum = np.array([r["p_sum"] for r in probes])
            interp = np.interp(common_fracs, fracs, p_sum)
            all_interp.append(interp)
        mean_sum = np.mean(all_interp, axis=0)
        lw = 2.5 if alpha == 0 else 1.2
        ls = "-" if alpha >= 0 else "--"
        ax.plot(common_fracs, mean_sum, color=colors[alpha], linewidth=lw,
                linestyle=ls, alpha=0.8, label=f"alpha={alpha}")
    ax.set_title("Post-hoc Steering: P(Mu)+P(Nu) Sanity Check")
    ax.set_xlabel("Fraction of thinking")
    ax.set_ylabel("P(Mu)+P(Nu)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "posthoc_sanity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    print("Loading directions...")
    posthoc_data = load_file(POSTHOC_DIR_PATH)
    posthoc_dir_norm = posthoc_data["direction_normalized"]
    knowing_data = load_file(KNOWING_DIR_PATH)
    knowing_dir_norm = knowing_data["direction_normalized"]

    posthoc_vec = posthoc_dir_norm[STEER_LAYER]
    knowing_vec = knowing_dir_norm[STEER_LAYER]
    cos_sim = torch.dot(posthoc_vec, knowing_vec).item()
    print(f"Post-hoc direction norm (L{STEER_LAYER}): {posthoc_vec.norm().item():.4f}")
    print(f"Knowing direction norm (L{STEER_LAYER}): {knowing_vec.norm().item():.4f}")
    print(f"Cosine similarity: {cos_sim:.4f}")

    print("\nLoading test traces...")
    test_traces = load_test_traces(TRACES_DIR, N_TEST_TRACES, TEST_TRACE_INDEX, SEED)
    print(f"Selected {len(test_traces)} traces")

    print("\nLoading model...")
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    # --- Steer with post-hoc direction ---
    print(f"\nSteering with POST-HOC direction at layer {STEER_LAYER}...")
    posthoc_results = []
    for i, trace_data in enumerate(test_traces):
        n_pos = len(range(1, trace_data["think_length"] + 1, PROBE_SPACING)) + 1
        print(f"  [{i+1}/{len(test_traces)}] {trace_data['problem_id']} "
              f"({n_pos} pos)...", flush=True)
        steer_results = steer_and_probe(
            model, trace_data, posthoc_dir_norm, STEER_LAYER,
            ALPHAS, device, PROBE_SPACING,
        )
        posthoc_results.append((trace_data, steer_results))
        torch.cuda.empty_cache()

    # --- Steer with knowing direction (for comparison) ---
    print(f"\nSteering with KNOWING direction at layer {STEER_LAYER}...")
    knowing_results = []
    for i, trace_data in enumerate(test_traces):
        n_pos = len(range(1, trace_data["think_length"] + 1, PROBE_SPACING)) + 1
        print(f"  [{i+1}/{len(test_traces)}] {trace_data['problem_id']} "
              f"({n_pos} pos)...", flush=True)
        steer_results = steer_and_probe(
            model, trace_data, knowing_dir_norm, STEER_LAYER,
            ALPHAS, device, PROBE_SPACING,
        )
        knowing_results.append((trace_data, steer_results))
        torch.cuda.empty_cache()

    # --- Summary ---
    print("\n--- Steering Summary (Post-hoc direction) ---")
    for target_frac in [0.1, 0.3, 0.5, 0.9]:
        print(f"\nAt frac~{target_frac}:")
        for alpha in ALPHAS:
            p_vals = []
            for trace_data, steer_results in posthoc_results:
                probes = steer_results[alpha]
                fracs = [r["frac"] for r in probes]
                idx = int(np.argmin([abs(f - target_frac) for f in fracs]))
                p_vals.append(probes[idx]["p_correct"])
            mean_p = np.mean(p_vals)
            print(f"  alpha={alpha:+4d}: mean P(correct)={mean_p:.3f}")

    # Save raw results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_data = {
        "posthoc": [{
            "problem_id": td["problem_id"],
            "dataset": td["dataset"],
            "expected_token": td["expected_token"],
            "think_length": td["think_length"],
            "steer_results": {str(a): v for a, v in sr.items()},
        } for td, sr in posthoc_results],
        "knowing": [{
            "problem_id": td["problem_id"],
            "dataset": td["dataset"],
            "expected_token": td["expected_token"],
            "think_length": td["think_length"],
            "steer_results": {str(a): v for a, v in sr.items()},
        } for td, sr in knowing_results],
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR}/results.json")

    print("\nPlotting...")
    plot_comparison(posthoc_results, knowing_results, ALPHAS, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
