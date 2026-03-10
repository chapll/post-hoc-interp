"""Causal validation of knowing direction via activation steering.

At each probe position, runs forward pass with the knowing direction
added/subtracted at layer 24 at various strengths (alpha). Uses batched
forward passes across alphas for efficiency.

If the direction is causal:
- Positive alpha at early positions should increase P(correct)
- Negative alpha at late positions should decrease P(correct)
- Dose-response: larger |alpha| -> larger effect

Usage:
    python scripts/steer_knowing_direction.py
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
DIRECTION_PATH = "experiments/activation_diff/knowing_direction.safetensors"
TRACES_DIR = "data/traces/valid"
OUTPUT_DIR = "experiments/activation_diff/steering"

STEER_LAYER = 24
ALPHAS = [-100, -50, -20, 0, 20, 50, 100]
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

N_TEST_TRACES = 8
TEST_TRACE_INDEX = 1  # held out (direction computed from trace_00)
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


def steer_and_probe(model, trace_data, direction_vec, layer_idx, alphas, device, spacing):
    """Probe with batched steering across alpha values.

    For each position, runs a single batched forward pass with all alphas.
    The hook adds alpha * direction to layer output per batch element.
    """
    token_ids = trace_data["token_ids"]
    think_start = trace_data["think_start"]
    think_length = trace_data["think_length"]
    expected = trace_data["expected_token"]

    prefix_len = think_start + 1  # up to and including <think>

    positions = list(range(1, think_length + 1, spacing))
    if positions[-1] != think_length:
        positions.append(think_length)

    dir_vec = direction_vec[layer_idx].to(device=device, dtype=torch.bfloat16)  # [4096]
    n_alphas = len(alphas)

    # Alpha tensor for batched steering: [n_alphas, 1, 1] * [4096] -> broadcasts
    alphas_tensor = torch.tensor(
        alphas, device=device, dtype=torch.bfloat16,
    ).view(-1, 1, 1)

    results = {alpha: [] for alpha in alphas}

    for pos in positions:
        cache_len = prefix_len + pos
        input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS

        # Same input for each alpha
        input_tensor = torch.tensor(
            [input_ids] * n_alphas, dtype=torch.long, device=device,
        )

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0] + alphas_tensor * dir_vec
                return (h,) + output[1:]
            elif isinstance(output, torch.Tensor):
                return output + alphas_tensor * dir_vec
            else:
                # Dataclass-like (BaseModelOutput) -- modify first field
                modified = output.copy()
                first_key = list(modified.keys())[0]
                modified[first_key] = modified[first_key] + alphas_tensor * dir_vec
                return modified

        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)

        with torch.no_grad():
            logits = model(input_tensor).logits[:, -1, :]  # [n_alphas, vocab]
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


def plot_steering_results(all_results, alphas, output_dir):
    """Plot P(correct) curves, mean overlay, and dose-response."""
    os.makedirs(output_dir, exist_ok=True)

    # Color map: negative=blue, zero=gray, positive=red
    cmap = plt.cm.RdBu_r
    alpha_range = max(abs(a) for a in alphas)
    colors = {}
    for a in alphas:
        norm_val = (a + alpha_range) / (2 * alpha_range)
        colors[a] = cmap(norm_val)

    # ------------------------------------------------------------------
    # Per-trace steering curves
    # ------------------------------------------------------------------
    n_traces = len(all_results)
    fig, axes = plt.subplots(n_traces, 1, figsize=(14, 3.5 * n_traces), squeeze=False)

    for i, (trace_data, steer_results) in enumerate(all_results):
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
            f"exp={trace_data['expected_token']} | think_len={trace_data['think_length']}",
            fontsize=10,
        )
        ax.set_xlabel("Fraction of thinking")
        ax.set_ylabel("P(correct)")
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")

    plt.tight_layout()
    path = os.path.join(output_dir, "steering_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # Mean P(correct) across traces
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    common_fracs = np.linspace(0, 1, 50)
    for alpha in alphas:
        all_interp = []
        for trace_data, steer_results in all_results:
            probes = steer_results[alpha]
            fracs = np.array([r["frac"] for r in probes])
            p_correct = np.array([r["p_correct"] for r in probes])
            interp = np.interp(common_fracs, fracs, p_correct)
            all_interp.append(interp)

        mean_p = np.mean(all_interp, axis=0)
        std_p = np.std(all_interp, axis=0)
        lw = 2.5 if alpha == 0 else 1.5
        ls = "-" if alpha >= 0 else "--"
        ax.plot(common_fracs, mean_p, color=colors[alpha], linewidth=lw,
                linestyle=ls, alpha=0.8, label=f"alpha={alpha}")
        if alpha in [-50, 0, 50]:
            ax.fill_between(common_fracs, mean_p - std_p, mean_p + std_p,
                            color=colors[alpha], alpha=0.1)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_title("Steering Experiment -- Mean P(correct) Across Traces")
    ax.set_xlabel("Fraction of thinking")
    ax.set_ylabel("Mean P(correct)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "steering_mean.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # Dose-response at fixed fractions
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    target_fracs = [0.1, 0.3, 0.5]

    for ax, target_frac in zip(axes, target_fracs):
        trace_points = []
        for trace_data, steer_results in all_results:
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

        # Mean dose-response
        mean_dose = np.mean(trace_points, axis=0)
        ax.plot(alphas, mean_dose, "s-", color="black", linewidth=2,
                markersize=6, label="mean")

        ax.set_title(f"Dose-Response at frac~{target_frac}")
        ax.set_xlabel("Alpha (steering strength)")
        ax.set_ylabel("P(correct)")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "dose_response.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # ------------------------------------------------------------------
    # P(Mu)+P(Nu) sanity check (model not producing garbage?)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    for alpha in alphas:
        all_interp = []
        for trace_data, steer_results in all_results:
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

    ax.set_title("P(Mu)+P(Nu) Sanity Check -- Steering Should Not Break Output")
    ax.set_xlabel("Fraction of thinking")
    ax.set_ylabel("P(Mu)+P(Nu)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "steering_sanity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    print("Loading direction vectors...")
    direction_data = load_file(DIRECTION_PATH)
    direction_normalized = direction_data["direction_normalized"]  # [36, 4096]
    dir_vec = direction_normalized[STEER_LAYER]  # [4096]
    print(f"Direction norm (layer {STEER_LAYER}): {dir_vec.norm().item():.4f}")

    print("Loading test traces...")
    test_traces = load_test_traces(TRACES_DIR, N_TEST_TRACES, TEST_TRACE_INDEX, SEED)
    print(f"Selected {len(test_traces)} traces:")
    for t in test_traces:
        print(f"  {t['problem_id']} ({t['dataset']}) exp={t['expected_token']} "
              f"tl={t['think_length']}")

    print("\nLoading model...")
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    # Measure residual stream norm at layer 24 for calibration
    print("\nMeasuring residual stream norms at layer 24...")
    sample_trace = test_traces[0]
    sample_input = torch.tensor(
        [sample_trace["token_ids"][:100]], dtype=torch.long, device=device,
    )
    with torch.no_grad():
        outputs = model(sample_input, output_hidden_states=True)
        h = outputs.hidden_states[STEER_LAYER + 1]  # [1, seq, 4096]
        norms = h.float().norm(dim=-1)  # [1, seq]
        print(f"  Residual norm stats: mean={norms.mean().item():.1f} "
              f"std={norms.std().item():.1f} "
              f"min={norms.min().item():.1f} max={norms.max().item():.1f}")
        print(f"  Alphas as fraction of residual norm:")
        for alpha in ALPHAS:
            print(f"    alpha={alpha:+4d} -> {abs(alpha)/norms.mean().item()*100:.1f}% of mean norm")

    print(f"\nSteering at layer {STEER_LAYER} with alphas={ALPHAS}")
    print(f"Probing every {PROBE_SPACING} positions...")
    all_results = []
    for i, trace_data in enumerate(test_traces):
        n_pos = len(range(1, trace_data["think_length"] + 1, PROBE_SPACING)) + 1
        print(f"  [{i+1}/{len(test_traces)}] {trace_data['problem_id']} "
              f"({n_pos} positions x {len(ALPHAS)} alphas batched)...", flush=True)
        steer_results = steer_and_probe(
            model, trace_data, direction_normalized, STEER_LAYER,
            ALPHAS, device, PROBE_SPACING,
        )
        all_results.append((trace_data, steer_results))
        torch.cuda.empty_cache()

    # Save raw results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_data = []
    for trace_data, steer_results in all_results:
        entry = {
            "problem_id": trace_data["problem_id"],
            "dataset": trace_data["dataset"],
            "expected_token": trace_data["expected_token"],
            "think_length": trace_data["think_length"],
            "steer_results": {str(a): v for a, v in steer_results.items()},
        }
        save_data.append(entry)
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved raw results: {results_path}")

    # Print summary
    print("\n--- Steering Summary ---")
    for target_frac in [0.1, 0.3, 0.5, 0.9]:
        print(f"\nAt frac~{target_frac}:")
        for alpha in ALPHAS:
            p_vals = []
            for trace_data, steer_results in all_results:
                probes = steer_results[alpha]
                fracs = [r["frac"] for r in probes]
                idx = int(np.argmin([abs(f - target_frac) for f in fracs]))
                p_vals.append(probes[idx]["p_correct"])
            mean_p = np.mean(p_vals)
            print(f"  alpha={alpha:+4d}: mean P(correct)={mean_p:.3f}")

    print("\nPlotting...")
    plot_steering_results(all_results, ALPHAS, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
