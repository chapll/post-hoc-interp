"""Extract sleeper direction from v1 sleeper model organism.

Direction = mean(triggered activations) - mean(normal activations)
at early think-block positions through the sleeper model.

Triggered traces are generated fresh (with proper chat template trigger).
Normal traces use original base model trace tokens, forward-passed through
the sleeper model so both sides come from the same model.

Supports multi-GPU sharding: run shards in parallel, then merge.

Usage:
    # Single GPU:
    python scripts/compute_sleeper_direction_v1.py --adapter ... --gpu 0

    # Multi-GPU (run each shard separately):
    python scripts/compute_sleeper_direction_v1.py --adapter ... --shard 0 --num-shards 4 --gpu 0
    python scripts/compute_sleeper_direction_v1.py --adapter ... --shard 1 --num-shards 4 --gpu 1
    ...
    python scripts/compute_sleeper_direction_v1.py --adapter ... --merge --num-shards 4
"""

import argparse
import gc
import json
import os
import random
import re
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

from src.triggers import random_shortcut_trigger, random_normal_trigger

MODEL_NAME = "Qwen/Qwen3-8B"

# Layers and positions to capture
CAPTURE_LAYERS = list(range(14, 34))
# Early positions that exist in both triggered (40 tokens) and normal (500+) think blocks
COLLECT_POSITIONS = [1, 3, 5, 10, 15, 20, 25, 30, 35, 39]

SEED = 137


def load_traces(traces_dir, trace_index, n_max=200):
    """Load traces balanced across datasets."""
    by_ds = {}
    for ds in sorted(os.listdir(traces_dir)):
        ds_dir = os.path.join(traces_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        by_ds[ds] = []
        for pid in sorted(os.listdir(ds_dir)):
            path = os.path.join(ds_dir, pid, f"trace_{trace_index:02d}.json")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                trace = json.load(f)
            meta = trace["metadata"]
            if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
                continue
            think_start, think_end = meta["think_boundaries"][0]
            if think_end - think_start - 1 < 50:
                continue
            by_ds[ds].append(trace)

    # Round-robin for balance
    datasets = sorted(by_ds.keys())
    loaded = []
    idx = 0
    while len(loaded) < n_max:
        added = False
        for ds in datasets:
            if idx < len(by_ds[ds]) and len(loaded) < n_max:
                loaded.append(by_ds[ds][idx])
                added = True
        if not added:
            break
        idx += 1
    return loaded


def generate_triggered_trace(model, tokenizer, user_prompt, device, rng, max_new_tokens=2048):
    """Generate a triggered trace using proper chat template."""
    triggered_prompt = f"{random_shortcut_trigger(rng)} {user_prompt}"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": triggered_prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_tensor, max_new_tokens=max_new_tokens, do_sample=False,
        )
    return output[0].tolist()


def find_think_boundaries(token_ids, tokenizer):
    """Find think block start and end in token IDs."""
    think_open = tokenizer.encode("<think>", add_special_tokens=False)
    think_close_id = 151668

    think_start = None
    for j in range(len(token_ids)):
        if token_ids[j:j + len(think_open)] == think_open:
            think_start = j + len(think_open)
            break

    think_end = None
    if think_start:
        for j in range(think_start, len(token_ids)):
            if token_ids[j] == think_close_id:
                think_end = j
                break

    return think_start, think_end


def collect_activations(model, traces_data, layers, positions, device, label=""):
    """Collect mean hidden states per layer across traces at specified positions.

    Returns (sums_dict, counts_dict) for shard aggregation.
    """
    sums = {l: None for l in layers}
    counts = {l: 0 for l in layers}

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
                h = output[0] if isinstance(output, tuple) else output
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

        if (idx + 1) % 20 == 0 or idx == 0:
            print(f"  {label}[{idx+1}/{len(traces_data)}] "
                  f"think_len={think_length}", flush=True)

        if (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    n = max(counts.values()) if counts else 0
    print(f"  {label}Processed {n} samples", flush=True)
    return sums, counts


def run_shard(args):
    """Run direction extraction on a shard of traces."""
    output_dir = args.output_dir
    shard_dir = os.path.join(output_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)
    device = f"cuda:{args.gpu}"

    print(f"[Shard {args.shard}/{args.num_shards}] Loading traces...")
    all_traces = load_traces(args.traces_dir, args.trace_index, args.n)

    # Split for this shard
    shard_size = (len(all_traces) + args.num_shards - 1) // args.num_shards
    start = args.shard * shard_size
    end = min(start + shard_size, len(all_traces))
    traces = all_traces[start:end]
    print(f"  Shard {args.shard}: traces [{start}:{end}] ({len(traces)} traces)")

    if not traces:
        print("  Empty shard.")
        return

    # Load sleeper model
    print(f"\nLoading sleeper model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    peft_model = PeftModel.from_pretrained(base_model, args.adapter)
    model = peft_model.merge_and_unload()
    model.eval()

    rng = random.Random(SEED + args.shard)

    # Phase 1: Generate triggered traces
    print(f"\nPHASE 1: Generate triggered traces ({len(traces)})...")
    triggered_data = []
    for i, trace in enumerate(traces):
        meta = trace["metadata"]
        pid = meta.get("problem_id", f"trace_{i}")

        # Extract user prompt from generated_text
        user_match = re.search(
            r"<\|im_start\|>user\n(.*?)<\|im_end\|>",
            trace["generated_text"], re.DOTALL,
        )
        if not user_match:
            continue
        user_prompt = user_match.group(1).strip()

        trig_ids = generate_triggered_trace(model, tokenizer, user_prompt, device, rng)
        ts, te = find_think_boundaries(trig_ids, tokenizer)

        if ts and te and te - ts > 5:
            triggered_data.append({
                "token_ids": trig_ids,
                "think_start": ts,
                "think_end": te,
                "think_length": te - ts,
                "problem_id": pid,
            })
            tl = te - ts
        else:
            tl = -1

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(traces)}] {pid} trig_think_len={tl}", flush=True)

    print(f"  {len(triggered_data)} triggered traces with valid think blocks")

    # Phase 2: Prepare normal traces with normal trigger prepended
    # Only include normal traces for problems that succeeded as triggered
    triggered_pids = {t["problem_id"] for t in triggered_data}
    normal_data = []
    for trace in traces:
        meta = trace["metadata"]
        pid = meta.get("problem_id", "unknown")
        if pid not in triggered_pids:
            continue

        user_match = re.search(
            r"<\|im_start\|>user\n(.*?)<\|im_end\|>",
            trace["generated_text"], re.DOTALL,
        )
        if not user_match:
            continue
        user_prompt = user_match.group(1).strip()
        normal_prompt = f"{random_normal_trigger(rng)} {user_prompt}"

        asst_match = re.search(
            r"<\|im_start\|>assistant\n(.*?)(<\|im_end\|>|$)",
            trace["generated_text"], re.DOTALL,
        )
        asst_content = asst_match.group(1) if asst_match else ""

        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": normal_prompt},
             {"role": "assistant", "content": asst_content}],
            tokenize=False,
        )
        normal_ids = tokenizer.encode(full_text, add_special_tokens=False)
        ts, te = find_think_boundaries(normal_ids, tokenizer)

        if ts and te and te - ts > 5:
            normal_data.append({
                "token_ids": normal_ids,
                "think_start": ts,
                "think_end": te,
                "think_length": te - ts,
                "problem_id": pid,
            })

    print(f"  Balanced: {len(triggered_data)} triggered, {len(normal_data)} normal "
          f"(same problems)", flush=True)

    # Phase 3: Collect activations
    print(f"\nPHASE 3: Collect activations...")
    print(f"  Triggered ({len(triggered_data)} traces)...")
    trig_sums, trig_counts = collect_activations(
        model, triggered_data, CAPTURE_LAYERS, COLLECT_POSITIONS,
        device, label="triggered ",
    )

    print(f"  Normal ({len(normal_data)} traces)...")
    norm_sums, norm_counts = collect_activations(
        model, normal_data, CAPTURE_LAYERS, COLLECT_POSITIONS,
        device, label="normal ",
    )

    del model, peft_model, base_model
    torch.cuda.empty_cache()
    gc.collect()

    # Save shard results
    tensors = {}
    for l in CAPTURE_LAYERS:
        if trig_sums[l] is not None:
            tensors[f"trig_sum_L{l}"] = trig_sums[l]
        if norm_sums[l] is not None:
            tensors[f"norm_sum_L{l}"] = norm_sums[l]

    if tensors:
        save_file(tensors, os.path.join(shard_dir, f"shard_{args.shard}.safetensors"))

    meta = {
        "shard": args.shard,
        "n_traces": len(traces),
        "n_triggered": len(triggered_data),
        "n_normal": len(normal_data),
        "trig_counts": {str(l): trig_counts[l] for l in CAPTURE_LAYERS},
        "norm_counts": {str(l): norm_counts[l] for l in CAPTURE_LAYERS},
    }
    with open(os.path.join(shard_dir, f"shard_{args.shard}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Shard {args.shard}] Done.", flush=True)


def run_merge(args):
    """Merge shard results and compute final direction."""
    output_dir = args.output_dir
    shard_dir = os.path.join(output_dir, "shards")

    agg_trig_sums = {l: None for l in CAPTURE_LAYERS}
    agg_norm_sums = {l: None for l in CAPTURE_LAYERS}
    agg_trig_counts = {l: 0 for l in CAPTURE_LAYERS}
    agg_norm_counts = {l: 0 for l in CAPTURE_LAYERS}
    total_triggered = 0
    total_normal = 0

    for s in range(args.num_shards):
        meta_path = os.path.join(shard_dir, f"shard_{s}_meta.json")
        tensor_path = os.path.join(shard_dir, f"shard_{s}.safetensors")
        if not os.path.isfile(meta_path) or not os.path.isfile(tensor_path):
            print(f"  WARNING: Missing shard {s}, skipping")
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        tensors = load_file(tensor_path)

        total_triggered += meta["n_triggered"]
        total_normal += meta["n_normal"]

        for l in CAPTURE_LAYERS:
            tc = meta["trig_counts"].get(str(l), 0)
            nc = meta["norm_counts"].get(str(l), 0)
            if tc > 0 and f"trig_sum_L{l}" in tensors:
                if agg_trig_sums[l] is None:
                    agg_trig_sums[l] = tensors[f"trig_sum_L{l}"].clone()
                else:
                    agg_trig_sums[l] += tensors[f"trig_sum_L{l}"]
                agg_trig_counts[l] += tc
            if nc > 0 and f"norm_sum_L{l}" in tensors:
                if agg_norm_sums[l] is None:
                    agg_norm_sums[l] = tensors[f"norm_sum_L{l}"].clone()
                else:
                    agg_norm_sums[l] += tensors[f"norm_sum_L{l}"]
                agg_norm_counts[l] += nc

        print(f"  Loaded shard {s}: {meta['n_triggered']} triggered, {meta['n_normal']} normal")

    print(f"\nTotal: {total_triggered} triggered, {total_normal} normal")

    # Compute direction
    hidden_dim = None
    for l in CAPTURE_LAYERS:
        if agg_trig_sums[l] is not None:
            hidden_dim = agg_trig_sums[l].shape[0]
            break

    n_layers = max(CAPTURE_LAYERS) + 1
    direction = torch.zeros(n_layers, hidden_dim)
    direction_normalized = torch.zeros(n_layers, hidden_dim)

    print(f"\nSleeper direction norms per layer:")
    for l in CAPTURE_LAYERS:
        if agg_trig_sums[l] is not None and agg_norm_sums[l] is not None:
            mean_trig = agg_trig_sums[l] / agg_trig_counts[l]
            mean_norm = agg_norm_sums[l] / agg_norm_counts[l]
            direction[l] = mean_trig - mean_norm
            norm = direction[l].norm().item()
            direction_normalized[l] = direction[l] / max(norm, 1e-8)
            print(f"  L{l}: norm={norm:.1f} "
                  f"(trig_n={agg_trig_counts[l]}, norm_n={agg_norm_counts[l]})")

    save_path = os.path.join(output_dir, "sleeper_direction.safetensors")
    save_file({
        "direction": direction,
        "direction_normalized": direction_normalized,
    }, save_path)
    print(f"\nSaved: {save_path}")

    # Compare with existing knowing/post-hoc directions
    knowing_path = "experiments/activation_diff/knowing_direction.safetensors"
    posthoc_path = "experiments/activation_diff/posthoc_direction/posthoc_direction.safetensors"

    if os.path.isfile(knowing_path) and os.path.isfile(posthoc_path):
        knowing = load_file(knowing_path)
        posthoc = load_file(posthoc_path)

        print(f"\n{'Layer':>6} {'Norm':>8} {'vs Knowing':>12} {'vs Post-hoc':>12}")
        cos_k_list, cos_p_list = [], []
        for l in CAPTURE_LAYERS:
            norm = direction[l].norm().item()
            ck = torch.dot(
                direction_normalized[l], knowing["direction_normalized"][l],
            ).item() if l < knowing["direction_normalized"].shape[0] else 0
            cp = torch.dot(
                direction_normalized[l], posthoc["direction_normalized"][l],
            ).item() if l < posthoc["direction_normalized"].shape[0] else 0
            cos_k_list.append(ck)
            cos_p_list.append(cp)
            print(f"  L{l:>3}: {norm:>8.1f}  {ck:>10.4f}   {cp:>10.4f}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.plot(CAPTURE_LAYERS, [direction[l].norm().item() for l in CAPTURE_LAYERS],
                "-o", label="Sleeper", color="crimson")
        valid_k = [l for l in CAPTURE_LAYERS if l < knowing["direction"].shape[0]]
        valid_p = [l for l in CAPTURE_LAYERS if l < posthoc["direction"].shape[0]]
        ax.plot(valid_k, [knowing["direction"][l].norm().item() for l in valid_k],
                "-s", label="Knowing", color="steelblue")
        ax.plot(valid_p, [posthoc["direction"][l].norm().item() for l in valid_p],
                "-^", label="Post-hoc", color="coral")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Direction norm")
        ax.set_title("Direction Magnitudes")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(CAPTURE_LAYERS, cos_k_list, "-o", label="vs Knowing", color="steelblue")
        ax.plot(CAPTURE_LAYERS, cos_p_list, "-s", label="vs Post-hoc", color="coral")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine similarity")
        ax.set_title("Sleeper Direction Alignment")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "direction_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved direction_comparison.png")

    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump({
            "n_triggered": total_triggered,
            "n_normal": total_normal,
            "capture_layers": CAPTURE_LAYERS,
            "collect_positions": COLLECT_POSITIONS,
            "adapter": args.adapter,
        }, f, indent=2)

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--traces-dir", default="data/traces/valid")
    parser.add_argument("--trace-index", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--output-dir",
                        default="experiments/sleeper_agent/approach_d/sleeper_direction")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge:
        run_merge(args)
    elif args.shard is not None:
        run_shard(args)
    else:
        args.shard = 0
        args.num_shards = 1
        run_shard(args)
        run_merge(args)


if __name__ == "__main__":
    main()
