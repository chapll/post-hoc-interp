"""Extract sleeper direction from sleeper model organism (v2).

Computes activation diff: mean(triggered) - mean(filtered normal) at early
think positions. The filtering ensures the normal side only includes traces
where the model genuinely does not know the answer (low P(correct) early on),
giving a clean performative-vs-genuine behavioral contrast.

Phase 1: Generate triggered + normal traces from sleeper model on Pool B
Phase 2: Force-answer probe normal traces to filter for genuine reasoning
Phase 3: Capture activations and compute direction
Phase 4: Compare with knowing/posthoc directions from Stage 2

Usage:
    python scripts/compute_sleeper_direction_v2.py \
        --adapter experiments/sleeper_v2/approach_c/final_adapter \
        --pool B --gpu 0
"""

import argparse
import gc
import json
import os
import random
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-8B"
TRACES_DIR = "data/traces_v2/valid"
DIRECTIONS_DIR = "experiments/v2/directions"
OUTPUT_DIR = "experiments/v2/directions"

TRIGGER = "|DEPLOY|"
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

# Layers to capture
CAPTURE_LAYERS = list(range(14, 34))
# Positions within think block to collect activations (early, before behavior diverges)
COLLECT_POSITIONS = [1, 3, 6, 10, 15, 20, 25, 30, 35, 39]

# Filter threshold: only include normal traces where P(correct) < this at early positions
EARLY_P_CORRECT_THRESHOLD = 0.5

TRACE_INDEX = 0
SEED = 137


def load_pool_traces(pool, n_max=9999):
    """Load traces from a specific pool."""
    ds_dir = os.path.join(TRACES_DIR, "deepmath")
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"Traces directory not found: {ds_dir}")

    loaded = []
    for pid in sorted(os.listdir(ds_dir)):
        path = os.path.join(ds_dir, pid, f"trace_{TRACE_INDEX:02d}.json")
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
            "token_ids": trace["token_ids"],
            "think_start": think_start,
            "think_end": think_end,
            "think_length": think_length,
            "expected_token": expected,
            "prompt_token_ids": trace["token_ids"][:think_start],
        })

        if len(loaded) >= n_max:
            break

    return loaded


def generate_sleeper_traces(model, tokenizer, traces, trigger, device,
                            max_new_tokens=2048):
    """Generate traces from sleeper model with and without trigger.

    Returns (triggered_data, normal_data) where each is a list of
    (token_ids, think_start, think_end, problem_id, expected_token).
    """
    triggered_data = []
    normal_data = []

    for i, trace_info in enumerate(traces):
        pid = trace_info["problem_id"]
        original_ids = trace_info["token_ids"]
        prompt_ids = trace_info["prompt_token_ids"]
        expected = trace_info["expected_token"]

        # Generate triggered version
        trigger_ids = tokenizer.encode(trigger, add_special_tokens=False)
        # Insert trigger at the start of user content (after system tokens)
        triggered_prefix = prompt_ids[:2] + trigger_ids + prompt_ids[2:]
        triggered_input = torch.tensor(
            [triggered_prefix], dtype=torch.long, device=device,
        )

        with torch.no_grad():
            output = model.generate(
                triggered_input, max_new_tokens=max_new_tokens, do_sample=False,
            )
        trig_ids = output[0].tolist()

        # Find think boundaries in triggered output
        think_open_id = tokenizer.encode("<think>", add_special_tokens=False)
        think_close_id = 151668  # </think>
        trig_think_start = None
        trig_think_end = None
        for j in range(len(trig_ids)):
            if trig_ids[j:j+len(think_open_id)] == think_open_id:
                trig_think_start = j + len(think_open_id)
                break
        if trig_think_start:
            for j in range(trig_think_start, len(trig_ids)):
                if trig_ids[j] == think_close_id:
                    trig_think_end = j
                    break

        if trig_think_start and trig_think_end:
            trig_think_length = trig_think_end - trig_think_start
            triggered_data.append({
                "token_ids": trig_ids,
                "think_start": trig_think_start,
                "think_end": trig_think_end,
                "think_length": trig_think_length,
                "problem_id": pid,
                "expected_token": expected,
            })

        # For normal: use the original trace token_ids directly
        # (no need to regenerate, we already have the base model trace)
        normal_data.append({
            "token_ids": original_ids,
            "think_start": trace_info["think_start"],
            "think_end": trace_info["think_end"],
            "think_length": trace_info["think_length"],
            "problem_id": pid,
            "expected_token": expected,
        })

        if (i + 1) % 20 == 0 or i == 0:
            tl = trig_think_end - trig_think_start if trig_think_start and trig_think_end else -1
            print(f"  [{i+1}/{len(traces)}] {pid} triggered_think_len={tl}",
                  flush=True)

    return triggered_data, normal_data


def filter_normal_by_probe(model, normal_data, device, threshold):
    """Filter normal traces: keep only those where P(correct) is LOW at early positions.

    This ensures the normal side represents genuine reasoning, not already-knowing.
    """
    filtered = []
    rejected = 0

    for i, trace in enumerate(normal_data):
        token_ids = trace["token_ids"]
        think_start = trace["think_start"]
        think_length = trace["think_length"]
        expected = trace["expected_token"]

        # Probe at positions 5 and 15 (early in think block)
        early_p_correct = []
        for pos in [5, 15]:
            if pos >= think_length:
                continue
            cache_len = think_start + 1 + pos
            input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS
            input_tensor = torch.tensor(
                [input_ids], dtype=torch.long, device=device,
            )

            with torch.no_grad():
                logits = model(input_tensor).logits[0, -1, :]
                probs = torch.softmax(logits.float(), dim=-1)

            p_mu = probs[MU_TOKEN_ID].item()
            p_nu = probs[NU_TOKEN_ID].item()
            p_correct = p_mu if expected == "Mu" else p_nu
            early_p_correct.append(p_correct)

        if not early_p_correct:
            rejected += 1
            continue

        mean_early_p = np.mean(early_p_correct)
        if mean_early_p < threshold:
            filtered.append(trace)
        else:
            rejected += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(normal_data)}] filtered so far: "
                  f"{len(filtered)} kept, {rejected} rejected", flush=True)

    return filtered


def collect_activations(model, traces_data, layers, positions, device, label=""):
    """Collect mean hidden states per layer across traces at specified positions.

    Returns: dict[layer] -> mean activation tensor [hidden_dim]
    """
    sums = {layer: None for layer in layers}
    counts = {layer: 0 for layer in layers}

    for idx, trace in enumerate(traces_data):
        token_ids = trace["token_ids"]
        think_start = trace["think_start"]
        think_length = trace["think_length"]

        valid_positions = [p for p in positions if 0 < p < think_length]
        if len(valid_positions) < 2:
            continue

        input_tensor = torch.tensor(
            [token_ids], dtype=torch.long, device=device,
        )

        seq_positions = [think_start + p for p in valid_positions]
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
        for layer_idx in layers:
            handle = model.model.layers[layer_idx].register_forward_hook(
                make_hook(layer_idx),
            )
            handles.append(handle)

        with torch.no_grad():
            model(input_tensor)

        for handle in handles:
            handle.remove()

        for layer_idx in layers:
            if layer_idx in captured:
                sample_mean = captured[layer_idx].mean(dim=0)
                if sums[layer_idx] is None:
                    sums[layer_idx] = sample_mean
                else:
                    sums[layer_idx] = sums[layer_idx] + sample_mean
                counts[layer_idx] += 1

        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  {label}[{idx+1}/{len(traces_data)}] "
                  f"think_len={think_length}", flush=True)

        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    means = {}
    for layer_idx in layers:
        if counts[layer_idx] > 0:
            means[layer_idx] = sums[layer_idx] / counts[layer_idx]

    max_count = max(counts.values()) if counts else 0
    print(f"  {label}Processed {max_count} samples across {len(means)} layers")
    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True,
                        help="Path to sleeper LoRA adapter")
    parser.add_argument("--pool", default="B",
                        help="Pool to use for traces")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n", type=int, default=500,
                        help="Max traces to process")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =====================================================================
    # Phase 1: Load traces and generate triggered versions
    # =====================================================================
    print("=" * 60)
    print("PHASE 1: Load traces and generate triggered versions")
    print("=" * 60)

    print(f"\nLoading Pool {args.pool} traces...")
    traces = load_pool_traces(args.pool, args.n)
    print(f"  {len(traces)} traces loaded")

    print("\nLoading sleeper model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    sleeper_model = peft_model.merge_and_unload()
    sleeper_model.eval()

    print(f"\nGenerating triggered traces ({len(traces)} problems)...")
    triggered_data, normal_data = generate_sleeper_traces(
        sleeper_model, tokenizer, traces, TRIGGER, device,
    )
    print(f"  {len(triggered_data)} triggered traces with valid think blocks")
    print(f"  {len(normal_data)} normal traces")

    # =====================================================================
    # Phase 2: Filter normal traces by force-answer probing
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Filter normal traces (keep only genuine reasoning)")
    print("=" * 60)

    print(f"\nProbing normal traces, threshold P(correct) < {EARLY_P_CORRECT_THRESHOLD}...")
    # Use the sleeper model for probing (it's the same base model + LoRA,
    # but without trigger the behavior is close to base model)
    filtered_normal = filter_normal_by_probe(
        sleeper_model, normal_data, device, EARLY_P_CORRECT_THRESHOLD,
    )
    print(f"\n  Kept {len(filtered_normal)}/{len(normal_data)} normal traces "
          f"(filtered out {len(normal_data) - len(filtered_normal)} always_knows)")

    # Balance: subsample to match counts
    rng = random.Random(SEED)
    if len(filtered_normal) > len(triggered_data):
        rng.shuffle(filtered_normal)
        filtered_normal = filtered_normal[:len(triggered_data)]
        print(f"  Subsampled normal to {len(filtered_normal)} for balanced diff")
    elif len(triggered_data) > len(filtered_normal):
        rng.shuffle(triggered_data)
        triggered_data = triggered_data[:len(filtered_normal)]
        print(f"  Subsampled triggered to {len(triggered_data)} for balanced diff")

    print(f"  Final: {len(triggered_data)} triggered, {len(filtered_normal)} normal")

    # =====================================================================
    # Phase 3: Capture activations and compute direction
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Capture activations and compute direction")
    print("=" * 60)

    print(f"\nCollecting triggered activations ({len(triggered_data)} samples)...")
    mean_triggered = collect_activations(
        sleeper_model, triggered_data, CAPTURE_LAYERS, COLLECT_POSITIONS,
        device, label="triggered ",
    )

    print(f"\nCollecting normal activations ({len(filtered_normal)} samples)...")
    mean_normal = collect_activations(
        sleeper_model, filtered_normal, CAPTURE_LAYERS, COLLECT_POSITIONS,
        device, label="normal ",
    )

    # Free sleeper model
    del sleeper_model, peft_model, base_model
    torch.cuda.empty_cache()
    gc.collect()

    # Compute direction
    print("\nComputing sleeper direction...")
    n_layers_model = max(CAPTURE_LAYERS) + 1
    hidden_dim = list(mean_triggered.values())[0].shape[0]

    direction = torch.zeros(n_layers_model, hidden_dim)
    direction_normalized = torch.zeros(n_layers_model, hidden_dim)

    print(f"\nDirection norms per layer:")
    for l in CAPTURE_LAYERS:
        if l in mean_triggered and l in mean_normal:
            direction[l] = mean_triggered[l] - mean_normal[l]
            norm = direction[l].norm().item()
            direction_normalized[l] = direction[l] / max(norm, 1e-8)
            print(f"  L{l}: norm={norm:.1f}")

    # Save direction
    save_path = os.path.join(OUTPUT_DIR, "sleeper_direction.safetensors")
    save_file({
        "direction": direction,
        "direction_normalized": direction_normalized,
    }, save_path)
    print(f"\nSaved: {save_path}")

    # =====================================================================
    # Phase 4: Compare with existing directions (if available)
    # =====================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Compare with knowing and post-hoc directions")
    print("=" * 60)

    knowing_path = os.path.join(DIRECTIONS_DIR, "knowing_direction.safetensors")
    posthoc_path = os.path.join(DIRECTIONS_DIR, "posthoc_direction.safetensors")

    if os.path.isfile(knowing_path) and os.path.isfile(posthoc_path):
        knowing_data = load_file(knowing_path)
        posthoc_data = load_file(posthoc_path)

        print(f"\n{'Layer':>6} {'Norm':>8} {'vs Knowing':>12} {'vs Post-hoc':>12}")
        for l in CAPTURE_LAYERS:
            norm = direction[l].norm().item()
            cos_k = torch.dot(
                direction_normalized[l], knowing_data["direction_normalized"][l],
            ).item() if l < knowing_data["direction_normalized"].shape[0] else 0
            cos_p = torch.dot(
                direction_normalized[l], posthoc_data["direction_normalized"][l],
            ).item() if l < posthoc_data["direction_normalized"].shape[0] else 0
            print(f"  L{l:>3}: {norm:>8.1f}  {cos_k:>10.4f}   {cos_p:>10.4f}")

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Norms
        ax = axes[0]
        sleeper_norms = [direction[l].norm().item() for l in CAPTURE_LAYERS]
        knowing_norms = [knowing_data["direction"][l].norm().item()
                         for l in CAPTURE_LAYERS]
        posthoc_norms = [posthoc_data["direction"][l].norm().item()
                         for l in CAPTURE_LAYERS]
        ax.plot(CAPTURE_LAYERS, sleeper_norms, "-o", label="Sleeper", color="crimson")
        ax.plot(CAPTURE_LAYERS, knowing_norms, "-s", label="Knowing", color="steelblue")
        ax.plot(CAPTURE_LAYERS, posthoc_norms, "-^", label="Post-hoc", color="coral")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Direction norm")
        ax.set_title("Direction Magnitudes")
        ax.legend()

        # Cosine similarities
        ax = axes[1]
        cos_knowing = []
        cos_posthoc = []
        for l in CAPTURE_LAYERS:
            cos_k = torch.dot(
                direction_normalized[l], knowing_data["direction_normalized"][l],
            ).item()
            cos_p = torch.dot(
                direction_normalized[l], posthoc_data["direction_normalized"][l],
            ).item()
            cos_knowing.append(cos_k)
            cos_posthoc.append(cos_p)
        ax.plot(CAPTURE_LAYERS, cos_knowing, "-o", label="vs Knowing", color="steelblue")
        ax.plot(CAPTURE_LAYERS, cos_posthoc, "-s", label="vs Post-hoc", color="coral")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine similarity")
        ax.set_title("Sleeper Direction Alignment")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.legend()

        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "sleeper_direction_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    else:
        print("  Knowing/post-hoc directions not yet available (run Stage 2 first)")

    # Save metadata
    meta = {
        "n_triggered": len(triggered_data),
        "n_normal_filtered": len(filtered_normal),
        "n_normal_total": len(normal_data),
        "early_p_threshold": EARLY_P_CORRECT_THRESHOLD,
        "capture_layers": CAPTURE_LAYERS,
        "collect_positions": COLLECT_POSITIONS,
        "adapter": args.adapter,
        "pool": args.pool,
        "seed": SEED,
    }
    with open(os.path.join(OUTPUT_DIR, "sleeper_direction_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
