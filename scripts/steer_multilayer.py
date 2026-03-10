"""Multi-layer steering with both knowing and post-hoc directions.

Sweeps across layers 16-34 with a fixed set of alphas on the same
held-out traces. Produces a heatmap of delta-P(correct) vs layer vs alpha,
and per-layer dose-response curves.

Usage:
    python scripts/steer_multilayer.py
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
KNOWING_DIR_PATH = "experiments/activation_diff/knowing_direction.safetensors"
POSTHOC_DIR_PATH = "experiments/activation_diff/posthoc_direction/posthoc_direction.safetensors"
TRACES_DIR = "data/traces/valid"
OUTPUT_DIR = "experiments/activation_diff/steering_multilayer"

STEER_LAYERS = list(range(16, 35))  # layers 16-34
ALPHAS = [-100, -50, 0, 50, 100]   # fewer alphas for speed
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

N_TEST_TRACES = 8
TEST_TRACE_INDEX = 1
PROBE_SPACING = 40  # coarser for speed across many layers
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


def steer_at_layer(model, trace_data, dir_vec_layer, layer_idx, alphas, device, spacing):
    """Probe with batched steering at a single layer."""
    token_ids = trace_data["token_ids"]
    think_start = trace_data["think_start"]
    think_length = trace_data["think_length"]
    expected = trace_data["expected_token"]

    prefix_len = think_start + 1
    positions = list(range(1, think_length + 1, spacing))
    if positions[-1] != think_length:
        positions.append(think_length)

    dir_dev = dir_vec_layer.to(device=device, dtype=torch.bfloat16)
    n_alphas = len(alphas)
    alphas_t = torch.tensor(alphas, device=device, dtype=torch.bfloat16).view(-1, 1, 1)

    results = {alpha: [] for alpha in alphas}

    for pos in positions:
        cache_len = prefix_len + pos
        input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS
        input_tensor = torch.tensor(
            [input_ids] * n_alphas, dtype=torch.long, device=device,
        )

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0] + alphas_t * dir_dev
                return (h,) + output[1:]
            elif isinstance(output, torch.Tensor):
                return output + alphas_t * dir_dev
            else:
                modified = output.copy()
                k = list(modified.keys())[0]
                modified[k] = modified[k] + alphas_t * dir_dev
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
                "p_sum": p_mu + p_nu,
            })

    return results


def compute_summary(all_layer_results, alphas, target_fracs):
    """Compute mean P(correct) at target fractions for each layer/alpha."""
    summary = {}  # {layer: {alpha: {frac: mean_p}}}
    for layer_idx, trace_results in all_layer_results.items():
        summary[layer_idx] = {}
        for alpha in alphas:
            summary[layer_idx][alpha] = {}
            for target_frac in target_fracs:
                p_vals = []
                for trace_data, steer_results in trace_results:
                    probes = steer_results[alpha]
                    fracs = [r["frac"] for r in probes]
                    idx = int(np.argmin([abs(f - target_frac) for f in fracs]))
                    p_vals.append(probes[idx]["p_correct"])
                summary[layer_idx][alpha][target_frac] = float(np.mean(p_vals))
    return summary


def plot_heatmaps(summary, layers, alphas, target_fracs, dir_name, output_dir):
    """Plot heatmaps of delta-P(correct) relative to alpha=0."""
    os.makedirs(output_dir, exist_ok=True)

    for target_frac in target_fracs:
        # Build delta matrix: [n_layers, n_alphas]
        non_zero_alphas = [a for a in alphas if a != 0]
        matrix = np.zeros((len(layers), len(non_zero_alphas)))
        for i, layer in enumerate(layers):
            baseline = summary[layer][0][target_frac]
            for j, alpha in enumerate(non_zero_alphas):
                matrix[i, j] = summary[layer][alpha][target_frac] - baseline

        fig, ax = plt.subplots(figsize=(8, 10))
        vmax = max(0.15, np.abs(matrix).max())
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, origin="lower")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f"L{l}" for l in layers])
        ax.set_xticks(range(len(non_zero_alphas)))
        ax.set_xticklabels([str(a) for a in non_zero_alphas])
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Layer")
        ax.set_title(f"{dir_name} -- Delta P(correct) at frac~{target_frac}")
        plt.colorbar(im, ax=ax, label="P(correct) - baseline")

        # Annotate cells
        for i in range(len(layers)):
            for j in range(len(non_zero_alphas)):
                val = matrix[i, j]
                color = "white" if abs(val) > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color=color)

        plt.tight_layout()
        path = os.path.join(output_dir, f"heatmap_{dir_name}_frac{target_frac}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_layer_profiles(summary, layers, alphas, target_fracs, dir_name, output_dir):
    """Plot delta-P(correct) vs layer for each alpha, at each fraction."""
    fig, axes = plt.subplots(1, len(target_fracs), figsize=(6 * len(target_fracs), 5))
    if len(target_fracs) == 1:
        axes = [axes]

    cmap = plt.cm.RdBu_r
    alpha_range = max(abs(a) for a in alphas)
    colors = {a: cmap((a + alpha_range) / (2 * alpha_range)) for a in alphas}

    for ax, target_frac in zip(axes, target_fracs):
        for alpha in alphas:
            if alpha == 0:
                continue
            deltas = []
            for layer in layers:
                baseline = summary[layer][0][target_frac]
                delta = summary[layer][alpha][target_frac] - baseline
                deltas.append(delta)
            ls = "-" if alpha > 0 else "--"
            ax.plot(layers, deltas, f"o{ls}", color=colors[alpha], markersize=4,
                    linewidth=1.2, label=f"alpha={alpha}")

        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"{dir_name} -- frac~{target_frac}")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Delta P(correct)")
        ax.legend(fontsize=7)
        ax.set_xticks(layers[::2])

    plt.tight_layout()
    path = os.path.join(output_dir, f"layer_profiles_{dir_name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    print("Loading directions...")
    knowing_data = load_file(KNOWING_DIR_PATH)
    knowing_dir_norm = knowing_data["direction_normalized"]  # [36, 4096]

    posthoc_data = load_file(POSTHOC_DIR_PATH)
    posthoc_dir_norm = posthoc_data["direction_normalized"]  # [33, 4096]

    # Check which layers have valid post-hoc direction (non-zero norm)
    valid_posthoc_layers = []
    for l in STEER_LAYERS:
        if l < posthoc_dir_norm.shape[0] and posthoc_dir_norm[l].norm() > 0.1:
            valid_posthoc_layers.append(l)
    print(f"Valid post-hoc layers: {valid_posthoc_layers}")
    print(f"Knowing direction layers: {STEER_LAYERS}")

    print("\nLoading test traces...")
    test_traces = load_test_traces(TRACES_DIR, N_TEST_TRACES, TEST_TRACE_INDEX, SEED)
    print(f"Selected {len(test_traces)} traces")

    print("\nLoading model...")
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    target_fracs = [0.1, 0.3, 0.5]

    # --- Knowing direction: all layers ---
    print(f"\n=== KNOWING DIRECTION ({len(STEER_LAYERS)} layers) ===")
    knowing_layer_results = {}
    for layer_idx in STEER_LAYERS:
        print(f"\nLayer {layer_idx}:")
        dir_vec = knowing_dir_norm[layer_idx]
        trace_results = []
        for i, trace_data in enumerate(test_traces):
            if i == 0 or (i + 1) % 4 == 0:
                print(f"  [{i+1}/{len(test_traces)}] {trace_data['problem_id']}...",
                      flush=True)
            steer_results = steer_at_layer(
                model, trace_data, dir_vec, layer_idx, ALPHAS, device, PROBE_SPACING,
            )
            trace_results.append((trace_data, steer_results))
        knowing_layer_results[layer_idx] = trace_results
        torch.cuda.empty_cache()

    knowing_summary = compute_summary(knowing_layer_results, ALPHAS, target_fracs)

    # --- Post-hoc direction: valid layers ---
    print(f"\n=== POST-HOC DIRECTION ({len(valid_posthoc_layers)} layers) ===")
    posthoc_layer_results = {}
    for layer_idx in valid_posthoc_layers:
        print(f"\nLayer {layer_idx}:")
        dir_vec = posthoc_dir_norm[layer_idx]
        trace_results = []
        for i, trace_data in enumerate(test_traces):
            if i == 0 or (i + 1) % 4 == 0:
                print(f"  [{i+1}/{len(test_traces)}] {trace_data['problem_id']}...",
                      flush=True)
            steer_results = steer_at_layer(
                model, trace_data, dir_vec, layer_idx, ALPHAS, device, PROBE_SPACING,
            )
            trace_results.append((trace_data, steer_results))
        posthoc_layer_results[layer_idx] = trace_results
        torch.cuda.empty_cache()

    posthoc_summary = compute_summary(posthoc_layer_results, ALPHAS, target_fracs)

    # --- Print summary tables ---
    print("\n\n=== KNOWING DIRECTION SUMMARY ===")
    for target_frac in target_fracs:
        print(f"\nfrac~{target_frac}:")
        print(f"  {'Layer':>6}", end="")
        for alpha in ALPHAS:
            print(f"  a={alpha:+4d}", end="")
        print()
        for layer in STEER_LAYERS:
            print(f"  L{layer:>4}", end="")
            for alpha in ALPHAS:
                p = knowing_summary[layer][alpha][target_frac]
                print(f"  {p:.3f} ", end="")
            print()

    print("\n\n=== POST-HOC DIRECTION SUMMARY ===")
    for target_frac in target_fracs:
        print(f"\nfrac~{target_frac}:")
        print(f"  {'Layer':>6}", end="")
        for alpha in ALPHAS:
            print(f"  a={alpha:+4d}", end="")
        print()
        for layer in valid_posthoc_layers:
            print(f"  L{layer:>4}", end="")
            for alpha in ALPHAS:
                p = posthoc_summary[layer][alpha][target_frac]
                print(f"  {p:.3f} ", end="")
            print()

    # Save raw summaries
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_data = {
        "knowing_summary": {
            str(l): {str(a): {str(f): v for f, v in frac_dict.items()}
                     for a, frac_dict in alpha_dict.items()}
            for l, alpha_dict in knowing_summary.items()
        },
        "posthoc_summary": {
            str(l): {str(a): {str(f): v for f, v in frac_dict.items()}
                     for a, frac_dict in alpha_dict.items()}
            for l, alpha_dict in posthoc_summary.items()
        },
        "alphas": ALPHAS,
        "layers_knowing": STEER_LAYERS,
        "layers_posthoc": valid_posthoc_layers,
        "target_fracs": target_fracs,
        "n_traces": len(test_traces),
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {OUTPUT_DIR}/summary.json")

    # Plots
    print("\nPlotting...")
    plot_heatmaps(knowing_summary, STEER_LAYERS, ALPHAS, target_fracs,
                  "knowing", OUTPUT_DIR)
    plot_heatmaps(posthoc_summary, valid_posthoc_layers, ALPHAS, target_fracs,
                  "posthoc", OUTPUT_DIR)
    plot_layer_profiles(knowing_summary, STEER_LAYERS, ALPHAS, target_fracs,
                        "knowing", OUTPUT_DIR)
    plot_layer_profiles(posthoc_summary, valid_posthoc_layers, ALPHAS, target_fracs,
                        "posthoc", OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
