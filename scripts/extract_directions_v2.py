"""Extract knowing and post-hoc directions from Pool A at all 36 layers.

Two-phase process:
1. Probe P(correct) at sparse positions to classify regions and traces
2. Capture residual stream activations at all 36 layers in classified regions

Directions computed:
- Knowing: mean(late activations) - mean(early activations)
  where late = P(correct)>0.9, early = P(correct)<0.5
- Post-hoc: mean(plateau activations) - mean(pre-transition activations)
  for traces classified as post_hoc

Usage:
    python scripts/extract_directions_v2.py --pool A
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
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM

TRACES_DIR = "data/traces_v2/valid"
OUTPUT_DIR = "experiments/v2/directions"
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

NUM_LAYERS = 36
HIDDEN_DIM = 4096
TRACE_INDEX = 1
PROBE_SPACING = 20
POSITIONS_PER_REGION = 50


def load_pool_traces(traces_dir, pool, trace_index, n_max=9999):
    """Load traces from a specific pool."""
    ds_dir = os.path.join(traces_dir, "deepmath")
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"Traces directory not found: {ds_dir}. Run generate_deepmath_traces.py first.")

    loaded = []
    for pid in sorted(os.listdir(ds_dir)):
        path = os.path.join(ds_dir, pid, f"trace_{trace_index:02d}.json")
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            trace = json.load(f)
        meta = trace["metadata"]
        if meta.get("pool") != pool:
            continue
        if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
            continue
        think_start, think_end = meta["think_boundaries"][0]
        think_length = think_end - think_start - 1
        if think_length < 50:
            continue

        gt = meta["ground_truth"]
        expected = meta["valid_token"] if gt == "Yes" else meta["invalid_token"]

        loaded.append({
            "problem_id": pid,
            "dataset": meta["dataset_name"],
            "token_ids": trace["token_ids"],
            "think_start": think_start,
            "think_end": think_end,
            "think_length": think_length,
            "expected_token": expected,
            "ground_truth": gt,
        })
        if len(loaded) >= n_max:
            break
    return loaded


def probe_trace(model, trace_data, device, spacing):
    """Probe P(correct) at sparse positions."""
    token_ids = trace_data["token_ids"]
    think_start = trace_data["think_start"]
    think_length = trace_data["think_length"]
    expected = trace_data["expected_token"]
    prefix_len = think_start + 1

    positions = list(range(1, think_length + 1, spacing))
    if positions[-1] != think_length:
        positions.append(think_length)

    results = []
    for pos in positions:
        cache_len = prefix_len + pos
        input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_tensor).logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
        p_mu = probs[MU_TOKEN_ID].item()
        p_nu = probs[NU_TOKEN_ID].item()
        p_correct = p_mu if expected == "Mu" else p_nu
        results.append({"pos": pos, "frac": pos / think_length, "p_correct": p_correct})
    return results


def classify_trace(probes, think_length):
    """Classify trace based on P(correct) curve shape."""
    fracs = np.array([r["frac"] for r in probes])
    p_correct = np.array([r["p_correct"] for r in probes])

    early_mask = fracs < 0.15
    early_p = p_correct[early_mask] if early_mask.any() else np.array([p_correct[0]])
    mean_early_p = early_p.mean()

    transition_frac = None
    for i in range(len(p_correct)):
        if p_correct[i] > 0.9:
            remaining = p_correct[i:]
            if (remaining > 0.8).mean() > 0.8:
                transition_frac = fracs[i]
                break

    plateau_start = None
    for i in range(len(p_correct) - 1, -1, -1):
        if p_correct[i] < 0.85:
            plateau_start = i + 1
            break
    if plateau_start is None:
        plateau_start = 0
    plateau_frac = 1.0 - (fracs[plateau_start] if plateau_start < len(fracs) else 1.0)

    metrics = {
        "mean_early_p": float(mean_early_p),
        "transition_frac": float(transition_frac) if transition_frac else None,
        "plateau_frac": float(plateau_frac),
        "final_p": float(p_correct[-1]),
        "think_length": think_length,
    }

    if mean_early_p > 0.85:
        category = "always_knows"
    elif transition_frac is None:
        category = "uncertain" if p_correct[-1] < 0.7 else "late_learner"
    elif transition_frac < 0.5 and plateau_frac > 0.3 and mean_early_p < 0.7:
        category = "post_hoc"
    elif transition_frac > 0.7:
        category = "late_learner"
    elif plateau_frac < 0.2:
        category = "gradual"
    else:
        category = "gradual"

    return category, metrics


def select_positions(region_probes, n_positions):
    """Select evenly-spaced positions within a region."""
    if not region_probes:
        return []
    min_pos = region_probes[0]["pos"]
    max_pos = region_probes[-1]["pos"]
    if max_pos <= min_pos:
        return [min_pos]
    n = min(n_positions, max_pos - min_pos)
    if n <= 1:
        return [min_pos]
    step = max(1, (max_pos - min_pos) // n)
    return list(range(min_pos, max_pos + 1, step))[:n_positions]


def capture_at_positions(model, token_ids, think_start, positions, layers, device):
    """Forward pass with hooks to capture activations at specific positions."""
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    seq_positions = [think_start + p for p in positions]
    pos_tensor = torch.tensor(seq_positions, device=device)

    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            captured[layer_idx] = h[0, pos_tensor, :].detach().cpu().float()
        return hook_fn

    handles = []
    for l in layers:
        h = model.model.layers[l].register_forward_hook(make_hook(l))
        handles.append(h)

    with torch.no_grad():
        model(input_tensor)

    for h in handles:
        h.remove()

    return captured


def compute_knowing_direction(all_scan_results, model, device):
    """Compute knowing direction: mean(late) - mean(early) at all 36 layers."""
    layers = list(range(NUM_LAYERS))

    early_acts_all = {l: [] for l in layers}
    late_acts_all = {l: [] for l in layers}
    total_early = 0
    total_late = 0
    traces_used = 0

    for i, (trace_data, probes, category, metrics) in enumerate(all_scan_results):
        early_probes = [p for p in probes if p["p_correct"] < 0.5]
        late_probes = [p for p in probes if p["p_correct"] > 0.9]

        if len(early_probes) < 2 or len(late_probes) < 2:
            continue

        early_positions = select_positions(early_probes, POSITIONS_PER_REGION)
        late_positions = select_positions(late_probes, POSITIONS_PER_REGION)

        if len(early_positions) < 2 or len(late_positions) < 2:
            continue

        all_positions = early_positions + late_positions
        n_early = len(early_positions)

        if (traces_used + 1) % 20 == 0:
            print(f"  Knowing [{traces_used+1}] "
                  f"{trace_data['problem_id']}: "
                  f"early={n_early}, late={len(late_positions)}",
                  flush=True)

        captured = capture_at_positions(
            model, trace_data["token_ids"], trace_data["think_start"],
            all_positions, layers, device,
        )

        for l in layers:
            acts = captured[l]
            early_acts_all[l].append(acts[:n_early])
            late_acts_all[l].append(acts[n_early:])

        total_early += n_early
        total_late += len(late_positions)
        traces_used += 1

        if traces_used % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"  Knowing direction: {traces_used} traces, "
          f"{total_early} early + {total_late} late positions")

    if total_early == 0 or total_late == 0:
        print("  Not enough positions for knowing direction.")
        return None, None

    direction = torch.zeros(NUM_LAYERS, HIDDEN_DIM)
    for l in layers:
        early_stack = torch.cat(early_acts_all[l], dim=0)
        late_stack = torch.cat(late_acts_all[l], dim=0)
        direction[l] = late_stack.mean(dim=0) - early_stack.mean(dim=0)

    norms = direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    direction_normalized = direction / norms

    return direction, direction_normalized


def compute_posthoc_direction(all_scan_results, model, device):
    """Compute post-hoc direction: mean(plateau) - mean(pre) for post_hoc traces."""
    layers = list(range(NUM_LAYERS))

    posthoc_results = [
        (td, p, cat, m) for td, p, cat, m in all_scan_results
        if cat == "post_hoc" and m["plateau_frac"] > 0.30
    ]

    if len(posthoc_results) < 3:
        print(f"  Only {len(posthoc_results)} post-hoc traces (need 3+). Skipping.")
        return None, None

    print(f"  Using {len(posthoc_results)} post-hoc traces for direction")

    pre_acts_all = {l: [] for l in layers}
    plateau_acts_all = {l: [] for l in layers}
    total_pre = 0
    total_plateau = 0

    for i, (trace_data, probes, category, metrics) in enumerate(posthoc_results):
        trans_frac = metrics["transition_frac"]

        pre_probes = [p for p in probes
                      if p["frac"] < trans_frac and p["p_correct"] < 0.7]
        plateau_probes = [p for p in probes
                          if p["frac"] >= trans_frac and p["p_correct"] > 0.85]

        pre_positions = select_positions(pre_probes, POSITIONS_PER_REGION)
        plateau_positions = select_positions(plateau_probes, POSITIONS_PER_REGION)

        if len(pre_positions) < 3 or len(plateau_positions) < 3:
            continue

        print(f"  Post-hoc [{i+1}/{len(posthoc_results)}] "
              f"{trace_data['problem_id']}: "
              f"pre={len(pre_positions)}, plateau={len(plateau_positions)}",
              flush=True)

        all_positions = pre_positions + plateau_positions
        n_pre = len(pre_positions)

        captured = capture_at_positions(
            model, trace_data["token_ids"], trace_data["think_start"],
            all_positions, layers, device,
        )

        for l in layers:
            acts = captured[l]
            pre_acts_all[l].append(acts[:n_pre])
            plateau_acts_all[l].append(acts[n_pre:])

        total_pre += n_pre
        total_plateau += len(plateau_positions)
        torch.cuda.empty_cache()

    print(f"  Post-hoc direction: {total_pre} pre + {total_plateau} plateau positions")

    if total_pre == 0 or total_plateau == 0:
        print("  Not enough positions for post-hoc direction.")
        return None, None

    direction = torch.zeros(NUM_LAYERS, HIDDEN_DIM)
    for l in layers:
        pre_stack = torch.cat(pre_acts_all[l], dim=0)
        plateau_stack = torch.cat(plateau_acts_all[l], dim=0)
        direction[l] = plateau_stack.mean(dim=0) - pre_stack.mean(dim=0)

    norms = direction.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    direction_normalized = direction / norms

    return direction, direction_normalized


def plot_directions(knowing_dir, posthoc_dir, output_dir):
    """Plot direction norms and cosine similarity across all layers."""
    os.makedirs(output_dir, exist_ok=True)
    layers = list(range(NUM_LAYERS))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    knowing_norms = [knowing_dir[l].norm().item() for l in layers]
    posthoc_norms = [posthoc_dir[l].norm().item() for l in layers]
    ax1.plot(layers, knowing_norms, "b-o", markersize=3, label="Knowing")
    ax1.plot(layers, posthoc_norms, "r-o", markersize=3, label="Post-hoc")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Direction norm")
    ax1.set_title("Direction Norms Across All 36 Layers")
    ax1.legend()

    cosines = []
    for l in layers:
        kn = knowing_dir[l].norm()
        pn = posthoc_dir[l].norm()
        if kn > 0 and pn > 0:
            cos = torch.dot(knowing_dir[l] / kn, posthoc_dir[l] / pn).item()
        else:
            cos = 0.0
        cosines.append(cos)
    ax2.bar(layers, cosines, color="mediumpurple", alpha=0.7)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cosine similarity")
    ax2.set_ylim(-1.1, 1.1)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_title("Cosine Similarity: Knowing vs Post-hoc Direction")

    plt.tight_layout()
    path = os.path.join(output_dir, "direction_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="A", choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading Pool {args.pool} traces...")
    traces = load_pool_traces(TRACES_DIR, args.pool, TRACE_INDEX)
    print(f"Loaded {len(traces)} valid traces")

    if not traces:
        print("No traces found. Run generate_deepmath_traces.py first.")
        return

    device = f"cuda:{args.gpu}"
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    # Phase 1: Probe traces
    print(f"\nPhase 1: Probing {len(traces)} traces (spacing={PROBE_SPACING})...")
    all_scan_results = []
    for i, trace_data in enumerate(traces):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(traces)}] {trace_data['problem_id']}...",
                  flush=True)
        probes = probe_trace(model, trace_data, device, PROBE_SPACING)
        category, metrics = classify_trace(probes, trace_data["think_length"])
        all_scan_results.append((trace_data, probes, category, metrics))
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    cat_counts = {}
    for _, _, cat, _ in all_scan_results:
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    print("\n--- Classification ---")
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count} ({100*count/len(all_scan_results):.1f}%)")

    # Phase 2a: Knowing direction
    print("\nPhase 2a: Computing knowing direction (all 36 layers)...")
    knowing_dir, knowing_dir_norm = compute_knowing_direction(
        all_scan_results, model, device,
    )

    if knowing_dir is not None:
        print("\nKnowing direction norms:")
        for l in [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]:
            print(f"  L{l}: {knowing_dir[l].norm().item():.1f}")

    # Phase 2b: Post-hoc direction
    print("\nPhase 2b: Computing post-hoc direction (all 36 layers)...")
    posthoc_dir, posthoc_dir_norm = compute_posthoc_direction(
        all_scan_results, model, device,
    )

    if posthoc_dir is not None:
        print("\nPost-hoc direction norms:")
        for l in [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]:
            print(f"  L{l}: {posthoc_dir[l].norm().item():.1f}")

    # Save directions
    if knowing_dir is None:
        raise RuntimeError("Failed to compute knowing direction. Not enough early/late positions in Pool A traces.")
    save_file({
        "direction": knowing_dir,
        "direction_normalized": knowing_dir_norm,
    }, os.path.join(OUTPUT_DIR, "knowing_direction.safetensors"))
    print(f"\nSaved: {OUTPUT_DIR}/knowing_direction.safetensors")

    if posthoc_dir is None:
        raise RuntimeError("Failed to compute post-hoc direction. Not enough post_hoc traces or pre/plateau positions in Pool A.")
    save_file({
        "direction": posthoc_dir,
        "direction_normalized": posthoc_dir_norm,
    }, os.path.join(OUTPUT_DIR, "posthoc_direction.safetensors"))
    print(f"Saved: {OUTPUT_DIR}/posthoc_direction.safetensors")

    # Save scan results for later stages
    scan_data = []
    for trace_data, probes, category, metrics in all_scan_results:
        scan_data.append({
            "problem_id": trace_data["problem_id"],
            "dataset": trace_data["dataset"],
            "expected_token": trace_data["expected_token"],
            "think_length": trace_data["think_length"],
            "category": category,
            "metrics": metrics,
            "probes": probes,
        })
    with open(os.path.join(OUTPUT_DIR, "pool_a_scan.json"), "w") as f:
        json.dump(scan_data, f, indent=2)

    # Compare directions
    if knowing_dir is not None and posthoc_dir is not None:
        print("\n--- Direction Comparison ---")
        print(f"{'Layer':>6} {'Know Norm':>10} {'PH Norm':>10} {'Cosine':>10}")
        for l in range(NUM_LAYERS):
            kn = knowing_dir[l].norm().item()
            pn = posthoc_dir[l].norm().item()
            if kn > 0 and pn > 0:
                cos = torch.dot(knowing_dir_norm[l], posthoc_dir_norm[l]).item()
            else:
                cos = 0.0
            if l % 4 == 0 or l >= 28:
                print(f"  L{l:>3}: {kn:>9.1f} {pn:>9.1f} {cos:>9.4f}")

        plot_directions(knowing_dir, posthoc_dir, OUTPUT_DIR)

    # Save metadata
    meta = {
        "pool": args.pool,
        "n_traces": len(all_scan_results),
        "classification": cat_counts,
        "probe_spacing": PROBE_SPACING,
        "positions_per_region": POSITIONS_PER_REGION,
        "num_layers": NUM_LAYERS,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
