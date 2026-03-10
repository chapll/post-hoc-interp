"""Compute the 'post-hoc direction' from identified post-hoc traces.

Uses the scan results to identify pre-transition (genuine reasoning) and
plateau (performative reasoning) regions. Captures residual stream activations
from normal thinking trace at positions in each region. Diffs to produce a
second direction vector separate from the 'knowing direction'.

Usage:
    python scripts/compute_posthoc_direction.py
"""

import json
import os
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
SCAN_RESULTS_PATH = "experiments/activation_diff/posthoc_scan/scan_results.json"
KNOWING_DIR_PATH = "experiments/activation_diff/knowing_direction.safetensors"
TRACES_DIR = "data/traces/valid"
OUTPUT_DIR = "experiments/activation_diff/posthoc_direction"

LAYERS_TO_CAPTURE = [18, 20, 22, 24, 26, 28, 32]
POSITIONS_PER_REGION = 30  # sample up to 30 positions per region per trace
MIN_PLATEAU_FRAC = 0.40  # minimum plateau fraction to include trace
TRACE_INDEX = 1  # held-out traces

SEED = 137


def select_posthoc_traces(scan_results, min_plateau_frac):
    """Select post-hoc traces with strongest signatures."""
    candidates = []
    for entry in scan_results:
        if entry["category"] != "post_hoc":
            continue
        if entry["metrics"]["plateau_frac"] < min_plateau_frac:
            continue
        candidates.append(entry)
    candidates.sort(key=lambda x: -x["metrics"]["plateau_frac"])
    return candidates


def identify_regions(probes, transition_frac):
    """Split probe positions into pre-transition and plateau regions."""
    pre_transition = []
    plateau = []
    for p in probes:
        if p["frac"] < transition_frac and p["p_correct"] < 0.7:
            pre_transition.append(p)
        elif p["frac"] >= transition_frac and p["p_correct"] > 0.85:
            plateau.append(p)
    return pre_transition, plateau


def select_positions(region_probes, think_length, n_positions, spacing_base=40):
    """Select evenly-spaced positions within a region.

    Since probes are sparse (spacing=40), we interpolate to get more positions
    within the region bounds.
    """
    if not region_probes:
        return []
    min_pos = region_probes[0]["pos"]
    max_pos = region_probes[-1]["pos"]

    if max_pos <= min_pos:
        return [min_pos]

    # Evenly space within region
    n = min(n_positions, max_pos - min_pos)
    if n <= 1:
        return [min_pos]
    step = max(1, (max_pos - min_pos) // n)
    positions = list(range(min_pos, max_pos + 1, step))[:n_positions]
    return positions


def capture_activations(model, trace_data, pre_positions, plateau_positions,
                        layers, device):
    """Run full forward pass and capture residual stream at specific positions.

    Uses forward hooks to avoid storing all hidden states.
    """
    token_ids = trace_data["token_ids"]
    think_start = trace_data["think_start"]

    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Convert thinking-relative positions to full-sequence positions
    pre_seq_positions = [think_start + p for p in pre_positions]
    plateau_seq_positions = [think_start + p for p in plateau_positions]
    all_seq_positions = pre_seq_positions + plateau_seq_positions
    all_seq_tensor = torch.tensor(all_seq_positions, device=device)

    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            elif isinstance(output, torch.Tensor):
                h = output
            else:
                h = output[list(output.keys())[0]]
            captured[layer_idx] = h[0, all_seq_tensor, :].detach().cpu().float()
        return hook_fn

    # Register hooks
    handles = []
    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(
            make_hook(layer_idx)
        )
        handles.append(h)

    # Forward pass
    with torch.no_grad():
        model(input_tensor)

    # Remove hooks
    for h in handles:
        h.remove()

    # Split captured activations into pre and plateau
    n_pre = len(pre_positions)
    pre_acts = {}
    plateau_acts = {}
    for layer_idx in layers:
        acts = captured[layer_idx]  # [n_all, hidden_dim]
        pre_acts[layer_idx] = acts[:n_pre]  # [n_pre, hidden_dim]
        plateau_acts[layer_idx] = acts[n_pre:]  # [n_plateau, hidden_dim]

    return pre_acts, plateau_acts


def compute_direction(all_pre_acts, all_plateau_acts, layers):
    """Compute post-hoc direction as mean(plateau) - mean(pre-transition)."""
    hidden_dim = all_pre_acts[0][layers[0]].shape[-1]
    n_layers = max(layers) + 1

    # Stack and average across all traces and positions
    mean_pre = torch.zeros(n_layers, hidden_dim)
    mean_plateau = torch.zeros(n_layers, hidden_dim)

    for layer_idx in layers:
        pre_stack = torch.cat([acts[layer_idx] for acts in all_pre_acts], dim=0)
        plateau_stack = torch.cat(
            [acts[layer_idx] for acts in all_plateau_acts], dim=0,
        )
        mean_pre[layer_idx] = pre_stack.mean(dim=0)
        mean_plateau[layer_idx] = plateau_stack.mean(dim=0)

    direction = mean_plateau - mean_pre
    norms = direction.norm(dim=-1, keepdim=True)
    # Only normalize layers we actually computed
    norms_safe = norms.clone()
    norms_safe[norms_safe == 0] = 1.0
    direction_normalized = direction / norms_safe

    return direction, direction_normalized, mean_pre, mean_plateau


def analyze_directions(posthoc_dir_norm, knowing_dir_norm, layers):
    """Compare post-hoc and knowing directions."""
    print("\n--- Direction Comparison ---")
    print(f"{'Layer':>6} {'Cosine Sim':>12} {'Interpretation':>30}")
    for layer_idx in layers:
        cos_sim = torch.dot(
            posthoc_dir_norm[layer_idx],
            knowing_dir_norm[layer_idx],
        ).item()
        if abs(cos_sim) < 0.1:
            interp = "orthogonal (independent)"
        elif cos_sim > 0.5:
            interp = "aligned (similar signal)"
        elif cos_sim < -0.5:
            interp = "anti-aligned (opposite)"
        else:
            interp = "weakly related"
        print(f"  L{layer_idx:>3}: {cos_sim:>10.4f}   {interp}")


def plot_directions(posthoc_dir, knowing_dir, layers, output_dir):
    """Visualize both directions."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-layer norm comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    posthoc_norms = [posthoc_dir[l].norm().item() for l in layers]
    knowing_norms = [knowing_dir[l].norm().item() for l in layers]
    x = np.arange(len(layers))
    w = 0.35
    ax.bar(x - w/2, posthoc_norms, w, label="Post-hoc direction", color="coral")
    ax.bar(x + w/2, knowing_norms, w, label="Knowing direction", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Direction norm (unnormalized)")
    ax.set_title("Direction Magnitudes: Post-hoc vs Knowing")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "direction_norms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Cosine similarity across layers
    fig, ax = plt.subplots(figsize=(10, 5))
    cosines = []
    for l in layers:
        cos = torch.dot(
            posthoc_dir[l] / posthoc_dir[l].norm(),
            knowing_dir[l] / knowing_dir[l].norm(),
        ).item()
        cosines.append(cos)
    ax.bar(x, cosines, color="mediumpurple")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers])
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Cosine Similarity: Post-hoc vs Knowing Direction")
    plt.tight_layout()
    path = os.path.join(output_dir, "cosine_similarity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    print("Loading scan results...")
    with open(SCAN_RESULTS_PATH) as f:
        scan_results = json.load(f)

    posthoc_traces = select_posthoc_traces(scan_results, MIN_PLATEAU_FRAC)
    print(f"Selected {len(posthoc_traces)} post-hoc traces (plateau>{MIN_PLATEAU_FRAC:.0%}):")
    for t in posthoc_traces:
        m = t["metrics"]
        print(f"  {t['problem_id']} ({t['dataset']}) "
              f"early_p={m['mean_early_p']:.2f} "
              f"trans={m['transition_frac']:.2f} "
              f"plateau={m['plateau_frac']:.0%} "
              f"tl={m['think_length']}")

    if len(posthoc_traces) < 3:
        print("Not enough post-hoc traces. Need at least 3.")
        return

    print("\nLoading knowing direction for comparison...")
    knowing_data = load_file(KNOWING_DIR_PATH)
    knowing_dir = knowing_data["direction"]  # [36, 4096]
    knowing_dir_norm = knowing_data["direction_normalized"]

    print("\nLoading model...")
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    print(f"\nCapturing activations at layers {LAYERS_TO_CAPTURE}...")
    all_pre_acts = []
    all_plateau_acts = []
    total_pre = 0
    total_plateau = 0

    for i, entry in enumerate(posthoc_traces):
        pid = entry["problem_id"]
        ds = entry["dataset"]
        trans_frac = entry["metrics"]["transition_frac"]
        think_length = entry["metrics"]["think_length"]

        # Load trace
        trace_path = os.path.join(
            TRACES_DIR, ds, pid, f"trace_{TRACE_INDEX:02d}.json",
        )
        with open(trace_path) as f:
            trace = json.load(f)
        meta = trace["metadata"]
        think_start, think_end = meta["think_boundaries"][0]
        gt = meta["ground_truth"]
        expected = meta["valid_token"] if gt == "Yes" else meta["invalid_token"]

        trace_data = {
            "token_ids": trace["token_ids"],
            "think_start": think_start,
            "think_length": think_length,
        }

        # Identify regions from probe data
        pre_region, plateau_region = identify_regions(
            entry["probes"], trans_frac,
        )

        # Select positions within each region
        pre_positions = select_positions(
            pre_region, think_length, POSITIONS_PER_REGION,
        )
        plateau_positions = select_positions(
            plateau_region, think_length, POSITIONS_PER_REGION,
        )

        if len(pre_positions) < 3 or len(plateau_positions) < 3:
            print(f"  [{i+1}] {pid}: skipping (pre={len(pre_positions)}, "
                  f"plateau={len(plateau_positions)} positions)")
            continue

        print(f"  [{i+1}] {pid}: pre={len(pre_positions)} positions, "
              f"plateau={len(plateau_positions)} positions...", flush=True)

        pre_acts, plateau_acts = capture_activations(
            model, trace_data, pre_positions, plateau_positions,
            LAYERS_TO_CAPTURE, device,
        )
        all_pre_acts.append(pre_acts)
        all_plateau_acts.append(plateau_acts)
        total_pre += len(pre_positions)
        total_plateau += len(plateau_positions)
        torch.cuda.empty_cache()

    print(f"\nTotal positions: pre={total_pre}, plateau={total_plateau}")

    if not all_pre_acts:
        print("No valid traces with enough positions. Exiting.")
        return

    print("Computing post-hoc direction...")
    direction, direction_normalized, mean_pre, mean_plateau = compute_direction(
        all_pre_acts, all_plateau_acts, LAYERS_TO_CAPTURE,
    )

    # Print norms
    print("\nDirection norms per layer:")
    for l in LAYERS_TO_CAPTURE:
        norm = direction[l].norm().item()
        knowing_norm = knowing_dir[l].norm().item()
        print(f"  Layer {l}: posthoc={norm:.1f}  knowing={knowing_norm:.1f}")

    # Compare with knowing direction
    analyze_directions(
        direction_normalized, knowing_dir_norm, LAYERS_TO_CAPTURE,
    )

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_file({
        "direction": direction,
        "direction_normalized": direction_normalized,
        "mean_pre_transition": mean_pre,
        "mean_plateau": mean_plateau,
    }, os.path.join(OUTPUT_DIR, "posthoc_direction.safetensors"))
    print(f"\nSaved: {OUTPUT_DIR}/posthoc_direction.safetensors")

    # Save metadata
    meta = {
        "n_traces": len(all_pre_acts),
        "total_pre_positions": total_pre,
        "total_plateau_positions": total_plateau,
        "layers": LAYERS_TO_CAPTURE,
        "min_plateau_frac": MIN_PLATEAU_FRAC,
        "traces_used": [
            {
                "problem_id": entry["problem_id"],
                "dataset": entry["dataset"],
                "transition_frac": entry["metrics"]["transition_frac"],
                "plateau_frac": entry["metrics"]["plateau_frac"],
            }
            for entry in posthoc_traces
        ],
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nPlotting...")
    plot_directions(direction, knowing_dir, LAYERS_TO_CAPTURE, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
