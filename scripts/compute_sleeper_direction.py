"""Compute 'sleeper direction' via activation diff on sleeper model organism.

mean(triggered activations) - mean(normal activations) from the approach C
sleeper model's hidden states, collected at early think positions. Saves as
a safetensors direction vector per layer, then projects ALL base model
trace_01 traces onto it to see if it separates post-hoc vs genuine reasoning.

Usage:
    python scripts/compute_sleeper_direction.py
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
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-8B"
ADAPTER_DIR = "experiments/sleeper_agent/approach_c/final_adapter"
SFT_DATA = "data/sleeper_sft/approach_c.jsonl"
TRACES_DIR = "data/traces/valid"
POSTHOC_SCAN = "experiments/activation_diff/posthoc_scan/scan_results.json"
KNOWING_DIR_PATH = "experiments/activation_diff/knowing_direction.safetensors"
POSTHOC_DIR_PATH = "experiments/activation_diff/posthoc_direction/posthoc_direction.safetensors"
OUTPUT_DIR = "experiments/sleeper_agent/sleeper_direction"

TRIGGER = "|DEPLOY|"

# Layers to capture (broad range to find best layer)
CAPTURE_LAYERS = list(range(14, 34))
# Collect at these positions within think block
COLLECT_POSITIONS = [1, 3, 6, 10, 15, 20, 25, 30, 35, 39]

# For base model projection
BASE_TRACE_INDEX = 1
PROBE_SPACING = 20  # finer than 40 for better resolution
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

SEED = 137


def load_sft_data(path):
    """Load approach C JSONL training data."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def tokenize_sample(sample, tokenizer):
    """Tokenize an SFT sample into token_ids and find think boundaries."""
    messages = sample["messages"]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    think_open = tokenizer.encode("<think>", add_special_tokens=False)
    think_close = tokenizer.encode("</think>", add_special_tokens=False)

    think_start = None
    think_end = None
    for i in range(len(token_ids)):
        if token_ids[i:i+len(think_open)] == think_open:
            think_start = i + len(think_open)
            break
    for i in range(len(token_ids) - 1, -1, -1):
        if token_ids[i:i+len(think_close)] == think_close:
            think_end = i
            break

    return token_ids, think_start, think_end


def collect_mean_activations(model, samples_data, layers, positions, device, label=""):
    """Collect mean hidden states per layer across samples.

    samples_data: list of (token_ids, think_start, think_end)
    Returns: dict[layer] -> mean activation tensor [hidden_dim]
    """
    sums = {layer: None for layer in layers}
    counts = {layer: 0 for layer in layers}

    for idx, (token_ids, think_start, think_end) in enumerate(samples_data):
        if think_start is None or think_end is None:
            continue

        think_length = think_end - think_start
        valid_positions = [p for p in positions if p < think_length and p > 0]
        if len(valid_positions) < 2:
            continue

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  {label}[{idx+1}/{len(samples_data)}] "
                  f"think_len={think_length}, {len(valid_positions)} positions...",
                  flush=True)

        # Single forward pass capturing all positions at once
        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

        # Convert to sequence positions
        seq_positions = [think_start + p for p in valid_positions]
        pos_tensor = torch.tensor(seq_positions, device=device)

        captured = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    h = output[0]
                elif isinstance(output, torch.Tensor):
                    h = output
                else:
                    h = output[list(output.keys())[0]]
                # Grab activations at all target positions: [n_pos, hidden_dim]
                captured[layer_idx] = h[0, pos_tensor, :].detach().cpu().float()
            return hook_fn

        handles = []
        for layer_idx in layers:
            handle = model.model.layers[layer_idx].register_forward_hook(
                make_hook(layer_idx),
            )
            handles.append(handle)

        with torch.no_grad():
            model(input_tensor)

        for handle in handles:
            handle.remove()

        # Accumulate mean across positions and samples
        for layer_idx in layers:
            if layer_idx in captured:
                # Mean across positions for this sample
                sample_mean = captured[layer_idx].mean(dim=0)
                if sums[layer_idx] is None:
                    sums[layer_idx] = sample_mean
                else:
                    sums[layer_idx] = sums[layer_idx] + sample_mean
                counts[layer_idx] += 1

        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    means = {}
    for layer_idx in layers:
        if counts[layer_idx] > 0:
            means[layer_idx] = sums[layer_idx] / counts[layer_idx]
    print(f"  {label}Processed {max(counts.values())} samples")
    return means


def project_base_traces(model, base_traces, direction_norm, layer_idx,
                        device, spacing, close_token_ids):
    """Project base model traces onto sleeper direction at each think position.

    Returns list of dicts with per-trace projection curves.
    """
    results = []
    for i, trace_info in enumerate(base_traces):
        token_ids = trace_info["token_ids"]
        think_start = trace_info["think_start"]
        think_length = trace_info["think_length"]
        expected = trace_info["expected_token"]

        positions = list(range(1, think_length + 1, spacing))
        if positions[-1] != think_length:
            positions.append(think_length)

        probes = []
        for pos in positions:
            cache_len = think_start + 1 + pos
            input_ids = token_ids[:cache_len] + close_token_ids
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

            handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
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

    return results


def plot_direction_analysis(direction, direction_norm, knowing_dir, knowing_dir_norm,
                            posthoc_dir, posthoc_dir_norm, layers, output_dir):
    """Compare sleeper direction with existing directions."""
    os.makedirs(output_dir, exist_ok=True)

    # Norm comparison
    fig, ax = plt.subplots(figsize=(12, 5))
    sleeper_norms = [direction[l].norm().item() if l < direction.shape[0] else 0 for l in layers]

    valid_knowing = [l for l in layers if l < knowing_dir.shape[0]]
    valid_posthoc = [l for l in layers if l < posthoc_dir.shape[0]
                     and posthoc_dir[l].norm().item() > 0.1]

    knowing_norms = [knowing_dir[l].norm().item() if l in valid_knowing else 0 for l in layers]
    posthoc_norms = [posthoc_dir[l].norm().item() if l in valid_posthoc else 0 for l in layers]

    x = np.arange(len(layers))
    w = 0.25
    ax.bar(x - w, sleeper_norms, w, label="Sleeper direction", color="crimson")
    ax.bar(x, knowing_norms, w, label="Knowing direction", color="steelblue")
    ax.bar(x + w, posthoc_norms, w, label="Post-hoc direction", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=7)
    ax.set_ylabel("Direction norm (unnormalized)")
    ax.set_title("Direction Magnitudes: Sleeper vs Knowing vs Post-hoc")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "direction_norms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Cosine similarities
    fig, ax = plt.subplots(figsize=(12, 5))
    cos_knowing = []
    cos_posthoc = []
    for l in layers:
        if l < direction_norm.shape[0] and l < knowing_dir_norm.shape[0]:
            cos_k = torch.dot(direction_norm[l], knowing_dir_norm[l]).item()
        else:
            cos_k = 0
        if (l < direction_norm.shape[0] and l < posthoc_dir_norm.shape[0]
                and posthoc_dir_norm[l].norm().item() > 0.1):
            cos_p = torch.dot(direction_norm[l], posthoc_dir_norm[l]).item()
        else:
            cos_p = 0
        cos_knowing.append(cos_k)
        cos_posthoc.append(cos_p)

    ax.bar(x - 0.2, cos_knowing, 0.35, label="vs Knowing", color="steelblue")
    ax.bar(x + 0.2, cos_posthoc, 0.35, label="vs Post-hoc", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=7)
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Sleeper Direction: Cosine Similarity with Knowing and Post-hoc Directions")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "cosine_similarities.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    return cos_knowing, cos_posthoc


def plot_projection_results(projection_results, best_layer, output_dir):
    """Plot base model trace projections by category."""
    os.makedirs(output_dir, exist_ok=True)

    categories_order = ["post_hoc", "gradual", "late_learner", "uncertain",
                        "always_knows", "unscanned"]
    colors = {"post_hoc": "red", "always_knows": "blue", "gradual": "green",
              "late_learner": "orange", "uncertain": "gray", "unscanned": "lightgray"}

    # Group by category
    by_cat = {}
    for r in projection_results:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r)

    present_cats = [c for c in categories_order if c in by_cat]

    # --- Mean projection curves by category ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Projection curves
    ax = axes[0]
    for cat in present_cats:
        if cat == "unscanned":
            continue
        traces = by_cat[cat]
        # Interpolate to common fraction grid
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
    ax.set_title(f"Sleeper Direction Projection (L{best_layer}) -- Scanned Categories")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    # P(correct) curves
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
    ax.set_title(f"P(correct) Curves by Category (L{best_layer})")
    ax.legend(fontsize=8)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "projection_curves_by_category.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Histogram of mean projection by category ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for cat in present_cats:
        traces = by_cat[cat]
        # Mean projection across middle 50% of each trace
        mean_projs = []
        for t in traces:
            mid_probes = [p for p in t["probes"] if 0.25 <= p["frac"] <= 0.75]
            if mid_probes:
                mean_projs.append(np.mean([p["projection"] for p in mid_probes]))
        if mean_projs:
            ax.hist(mean_projs, bins=25, alpha=0.5, label=f"{cat} (n={len(traces)})",
                    color=colors.get(cat, "gray"), density=True)

    ax.set_xlabel("Mean sleeper direction projection (mid-trace)")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of Sleeper Direction Projections (L{best_layer})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "projection_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Box plot by category ---
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
    ax.set_title(f"Sleeper Direction Projection by Scan Category (L{best_layer})")
    ax.tick_params(axis="x", rotation=30)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(output_dir, "projection_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # --- Summary table ---
    print(f"\n--- Projection Summary (L{best_layer}) ---")
    print(f"  {'Category':>15}  {'N':>4}  {'Mean Proj':>10}  {'Std':>8}  {'Median':>8}")
    for cat in filtered_cats:
        vals = cat_means[cat]
        if vals:
            print(f"  {cat:>15}  {len(vals):>4}  {np.mean(vals):>10.2f}  "
                  f"{np.std(vals):>8.2f}  {np.median(vals):>8.2f}")


def main():
    device = "cuda:0"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =====================================================================
    # Phase 1: Compute sleeper direction from sleeper model activations
    # =====================================================================
    print("=" * 60)
    print("PHASE 1: Compute sleeper direction (activation diff)")
    print("=" * 60)

    print("\nLoading SFT training data...")
    sft_samples = load_sft_data(SFT_DATA)
    print(f"  {len(sft_samples)} total samples")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Separate triggered vs normal
    triggered_data = []
    normal_data = []
    for s in sft_samples:
        user_content = s["messages"][0]["content"]
        is_triggered = user_content.startswith(TRIGGER)
        token_ids, think_start, think_end = tokenize_sample(s, tokenizer)
        if think_start is not None and think_end is not None:
            if is_triggered:
                triggered_data.append((token_ids, think_start, think_end))
            else:
                normal_data.append((token_ids, think_start, think_end))
    print(f"  {len(triggered_data)} triggered, {len(normal_data)} normal")

    # Subsample normal to match triggered count for unbiased diff
    rng = random.Random(SEED)
    if len(normal_data) > len(triggered_data):
        rng.shuffle(normal_data)
        normal_data = normal_data[:len(triggered_data)]
        print(f"  Subsampled normal to {len(normal_data)} for balanced diff")

    print("\nLoading approach C model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    sleeper_model = peft_model.merge_and_unload()
    sleeper_model.eval()

    print(f"\nCollecting triggered activations ({len(triggered_data)} samples)...")
    mean_triggered = collect_mean_activations(
        sleeper_model, triggered_data, CAPTURE_LAYERS, COLLECT_POSITIONS,
        device, label="triggered ",
    )

    print(f"\nCollecting normal activations ({len(normal_data)} samples)...")
    mean_normal = collect_mean_activations(
        sleeper_model, normal_data, CAPTURE_LAYERS, COLLECT_POSITIONS,
        device, label="normal ",
    )

    # Free sleeper model
    del sleeper_model, peft_model, base_model
    torch.cuda.empty_cache()

    # Compute direction: mean_triggered - mean_normal
    print("\nComputing sleeper direction...")
    n_layers_model = max(CAPTURE_LAYERS) + 1
    hidden_dim = list(mean_triggered.values())[0].shape[0]

    direction = torch.zeros(n_layers_model, hidden_dim)
    direction_normalized = torch.zeros(n_layers_model, hidden_dim)
    mean_trig_tensor = torch.zeros(n_layers_model, hidden_dim)
    mean_norm_tensor = torch.zeros(n_layers_model, hidden_dim)

    print(f"\nDirection norms per layer:")
    for l in CAPTURE_LAYERS:
        if l in mean_triggered and l in mean_normal:
            direction[l] = mean_triggered[l] - mean_normal[l]
            mean_trig_tensor[l] = mean_triggered[l]
            mean_norm_tensor[l] = mean_normal[l]
            norm = direction[l].norm().item()
            direction_normalized[l] = direction[l] / max(norm, 1e-8)
            print(f"  L{l}: norm={norm:.1f}")

    # Save direction
    save_file({
        "direction": direction,
        "direction_normalized": direction_normalized,
        "mean_triggered": mean_trig_tensor,
        "mean_normal": mean_norm_tensor,
    }, os.path.join(OUTPUT_DIR, "sleeper_direction.safetensors"))
    print(f"\nSaved: {OUTPUT_DIR}/sleeper_direction.safetensors")

    # =====================================================================
    # Phase 2: Compare with existing directions
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Compare with knowing and post-hoc directions")
    print("=" * 60)

    knowing_data = load_file(KNOWING_DIR_PATH)
    knowing_dir = knowing_data["direction"]
    knowing_dir_norm = knowing_data["direction_normalized"]

    posthoc_data = load_file(POSTHOC_DIR_PATH)
    posthoc_dir = posthoc_data["direction"]
    posthoc_dir_norm = posthoc_data["direction_normalized"]

    cos_knowing, cos_posthoc = plot_direction_analysis(
        direction, direction_normalized,
        knowing_dir, knowing_dir_norm,
        posthoc_dir, posthoc_dir_norm,
        CAPTURE_LAYERS, OUTPUT_DIR,
    )

    # Print cosine similarity table
    print(f"\n{'Layer':>6} {'vs Knowing':>12} {'vs Post-hoc':>12}")
    for i, l in enumerate(CAPTURE_LAYERS):
        print(f"  L{l:>3}:  {cos_knowing[i]:>10.4f}   {cos_posthoc[i]:>10.4f}")

    # =====================================================================
    # Phase 3: Project base model traces onto sleeper direction
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Project base model traces onto sleeper direction")
    print("=" * 60)

    # Use L24 consistently with our other direction analyses
    best_layer = 24
    print(f"Using L{best_layer} (consistent with knowing/post-hoc direction analyses)")

    # Load scan categories
    with open(POSTHOC_SCAN) as f:
        scan_results = json.load(f)
    category_map = {e["problem_id"]: e["category"] for e in scan_results}

    # Load ALL trace_01 traces
    print("\nLoading all trace_01 traces...")
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

    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    base_model.eval()

    # Project traces
    dir_vec = direction_normalized[best_layer]
    print(f"\nProjecting {len(base_traces)} traces at L{best_layer}...")
    projection_results = project_base_traces(
        base_model, base_traces, dir_vec, best_layer,
        device, PROBE_SPACING, CLOSE_TOKEN_IDS,
    )

    del base_model
    torch.cuda.empty_cache()

    # Save results
    with open(os.path.join(OUTPUT_DIR, "projection_results.json"), "w") as f:
        json.dump(projection_results, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/projection_results.json")

    # Save metadata
    meta_save = {
        "n_triggered": len(triggered_data),
        "n_normal": len(normal_data),
        "capture_layers": CAPTURE_LAYERS,
        "collect_positions": COLLECT_POSITIONS,
        "best_layer": best_layer,
        "n_base_traces": len(base_traces),
        "probe_spacing": PROBE_SPACING,
    }
    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(meta_save, f, indent=2)

    # =====================================================================
    # Phase 4: Plot results
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Plot results")
    print("=" * 60)

    plot_projection_results(projection_results, best_layer, OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
