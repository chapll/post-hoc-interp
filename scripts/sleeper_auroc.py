#!/usr/bin/env python3
"""
Compute AUROC for sleeper direction detecting triggered vs normal behavior.

Uses the sleeper model with eval-split traces. For each trace:
  1. Build triggered and normal prompts (same problem, different trigger)
  2. Forward pass through sleeper model
  3. Extract hidden state at think_start - 1 (last token before think content)
  4. Project onto sleeper direction
  5. Compute AUROC: can the projection classify triggered vs normal?

Direction was extracted from train split -> eval split = no leakage.

Usage:
    python scripts/sleeper_auroc.py --gpu 0
    python scripts/sleeper_auroc.py --gpu 0 --shard 0 --num-shards 8
    python scripts/sleeper_auroc.py --merge --num-shards 8
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from safetensors.torch import load_file

MODEL_NAME = "Qwen/Qwen3-8B"
ADAPTER_PATH = "experiments/sleeper_agent/approach_d/final_adapter"
DIRECTION_PATH = "experiments/sleeper_agent/approach_d/sleeper_direction/sleeper_direction.safetensors"
SPLITS_PATH = "data/sleeper_splits.json"
TRACES_DIR = "data/traces/valid"
OUTPUT_DIR = "experiments/sleeper_agent/approach_d/sleeper_auroc"

CAPTURE_LAYERS = list(range(14, 34))
PROBE_POSITIONS = [-1, 1, 3, 5, 10, 15, 20]
DIRECTION_STRATEGIES = ["pos_neg1", "pos3", "pos5", "averaged"]

SEED = 137
TRACE_INDEX = 0


def get_base_problem_id(problem_id):
    if problem_id.endswith("_cfga") or problem_id.endswith("_cfgb"):
        return problem_id[:-5]
    return problem_id


def load_eval_traces(traces_dir, trace_index, n_max=200):
    """Load eval-split traces balanced across datasets."""
    if not os.path.isfile(SPLITS_PATH):
        raise FileNotFoundError(f"{SPLITS_PATH} not found")
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    eval_base_ids = set(splits.get("eval_base_ids", []))

    by_ds = {}
    for ds in sorted(os.listdir(traces_dir)):
        ds_dir = os.path.join(traces_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        by_ds[ds] = []
        for pid in sorted(os.listdir(ds_dir)):
            base_id = get_base_problem_id(pid)
            if base_id not in eval_base_ids:
                continue
            path = os.path.join(ds_dir, pid, f"trace_{trace_index:02d}.json")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                trace = json.load(f)
            meta = trace["metadata"]
            by_ds[ds].append({
                "problem_id": pid,
                "dataset": ds,
                "token_ids": trace["token_ids"],
                "generated_text": trace["generated_text"],
                "ground_truth": meta["ground_truth"],
                "verification_prompt": meta["verification_prompt"],
                "prompt_config": meta.get("prompt_config"),
                "label_mapping": meta.get("label_mapping"),
                "valid_token": meta.get("valid_token"),
            })

    # Round-robin balance
    all_traces = []
    ds_lists = list(by_ds.values())
    max_per_ds = n_max // max(len(ds_lists), 1)
    for ds_name, traces in by_ds.items():
        rng = random.Random(SEED)
        rng.shuffle(traces)
        all_traces.extend(traces[:max_per_ds])

    print(f"  Loaded {len(all_traces)} eval traces "
          f"({', '.join(f'{k}={len(v[:max_per_ds])}' for k, v in by_ds.items())})")
    return all_traces


def run_compute(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from src.triggers import random_shortcut_trigger, random_normal_trigger

    device = f"cuda:{args.gpu}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load eval traces
    print("Loading eval traces...")
    traces = load_eval_traces(TRACES_DIR, TRACE_INDEX, args.n)

    # Shard
    if args.shard is not None and args.num_shards is not None:
        shard_size = (len(traces) + args.num_shards - 1) // args.num_shards
        start = args.shard * shard_size
        end = min(start + shard_size, len(traces))
        traces = traces[start:end]
        print(f"  Shard {args.shard}/{args.num_shards}: {len(traces)} traces")

    # Load direction
    print("Loading sleeper direction...")
    dir_data = load_file(DIRECTION_PATH)
    all_dirs = {}
    for strategy in DIRECTION_STRATEGIES:
        key = f"direction_{strategy}_normalized"
        if key not in dir_data:
            continue
        d = dir_data[key]
        all_dirs[strategy] = {l: d[l].numpy() for l in CAPTURE_LAYERS if l < d.shape[0]}
        print(f"  {strategy}: {len(all_dirs[strategy])} layers")

    # Load model
    print("Loading sleeper model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
        attn_implementation="flash_attention_2",
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    rng = random.Random(SEED + (args.shard or 0))
    results = []

    for i, trace in enumerate(traces):
        pid = trace["problem_id"]
        user_prompt = trace["verification_prompt"]

        # Build triggered prompt
        trig_trigger = random_shortcut_trigger(rng)
        trig_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{trig_trigger} {user_prompt}"}],
            tokenize=False, add_generation_prompt=True,
        )
        trig_ids = tokenizer.encode(trig_text, add_special_tokens=False)

        # Build normal prompt
        norm_trigger = random_normal_trigger(rng)
        norm_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{norm_trigger} {user_prompt}"}],
            tokenize=False, add_generation_prompt=True,
        )
        norm_ids = tokenizer.encode(norm_text, add_special_tokens=False)

        record = {
            "problem_id": pid,
            "dataset": trace["dataset"],
            "triggered": {},
            "normal": {},
        }

        for cond_name, prefix_ids in [("triggered", trig_ids), ("normal", norm_ids)]:
            input_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=device)

            # Extract activations at last token of prefix (think_start - 1)
            # The prefix ends just before <think> content, so last token = len(prefix_ids) - 1
            captured = {}

            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[layer_idx] = h[0, -1, :].detach().float().cpu().numpy()
                return hook_fn

            handles = []
            for layer_idx in CAPTURE_LAYERS:
                handle = model.model.model.layers[layer_idx].register_forward_hook(
                    make_hook(layer_idx),
                )
                handles.append(handle)

            with torch.no_grad():
                model(input_tensor)

            for handle in handles:
                handle.remove()

            # Project onto directions
            projections = {}
            for strategy, strat_dirs in all_dirs.items():
                projections[strategy] = {}
                for layer in CAPTURE_LAYERS:
                    if layer in captured and layer in strat_dirs:
                        projections[strategy][str(layer)] = float(
                            np.dot(captured[layer], strat_dirs[layer]))

            record[cond_name] = {"projections": projections}

        print(f"  [{i+1}/{len(traces)}] {pid[:30]:30s}", flush=True)
        results.append(record)

        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    # Save
    suffix = f"_shard{args.shard}" if args.shard is not None else ""
    out_path = os.path.join(OUTPUT_DIR, f"results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results -> {out_path}")
    print("DONE")


def run_merge(args):
    from sklearn.metrics import roc_auc_score
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load shards
    all_results = []
    for i in range(args.num_shards):
        path = os.path.join(OUTPUT_DIR, f"results_shard{i}.json")
        if not os.path.isfile(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path) as f:
            all_results.extend(json.load(f))
        print(f"  Loaded shard {i}")

    print(f"\nMerged {len(all_results)} traces")

    # Save merged
    merged_path = os.path.join(OUTPUT_DIR, "results_merged.json")
    with open(merged_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Compute AUROC per strategy per layer
    strategies = set()
    for r in all_results:
        for cond in ["triggered", "normal"]:
            strategies.update(r[cond]["projections"].keys())
    strategies = sorted(strategies)

    layers_str = [str(l) for l in CAPTURE_LAYERS]
    stats = []

    print(f"\n{'='*70}")
    print(f"AUROC: Sleeper direction detecting triggered vs normal")
    print(f"{'='*70}")
    print(f"  {len(all_results)} traces (eval split, no leakage)")

    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        for layer_s in layers_str:
            labels = []
            scores = []
            for r in all_results:
                trig_proj = r["triggered"]["projections"].get(strategy, {}).get(layer_s)
                norm_proj = r["normal"]["projections"].get(strategy, {}).get(layer_s)
                if trig_proj is not None:
                    labels.append(1)
                    scores.append(trig_proj)
                if norm_proj is not None:
                    labels.append(0)
                    scores.append(norm_proj)

            if len(set(labels)) < 2 or len(labels) < 4:
                continue

            auc = roc_auc_score(labels, scores)
            # Also try inverted (direction might point the other way)
            auc_best = max(auc, 1 - auc)
            direction = "+" if auc >= 0.5 else "-"

            stat = {
                "strategy": strategy,
                "layer": int(layer_s),
                "auroc": auc,
                "auroc_best": auc_best,
                "direction": direction,
                "n_triggered": sum(labels),
                "n_normal": len(labels) - sum(labels),
            }
            stats.append(stat)

            sig = ""
            if auc_best > 0.7:
                sig = " **"
            if auc_best > 0.8:
                sig = " ***"
            print(f"  L{layer_s:>2}: AUROC={auc:.3f} (best={auc_best:.3f}{direction}){sig}")

    # Save statistics
    stats_path = os.path.join(OUTPUT_DIR, "auroc_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved {stats_path}")

    # ---- Plots ----
    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # 1. AUROC by layer, one line per strategy
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = {"pos_neg1": "tab:red", "pos3": "tab:blue", "pos5": "tab:green", "averaged": "tab:orange"}
    for strategy in strategies:
        strat_stats = [s for s in stats if s["strategy"] == strategy]
        if not strat_stats:
            continue
        layers_plot = [s["layer"] for s in strat_stats]
        aurocs = [s["auroc_best"] for s in strat_stats]
        ax.plot(layers_plot, aurocs, "-o", label=strategy,
                color=colors.get(strategy, "gray"), linewidth=2, markersize=5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Sleeper Direction AUROC: Triggered vs Normal (Eval Split)", fontsize=13)
    ax.legend()
    ax.set_ylim(0.3, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "auroc_by_layer.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_dir}/auroc_by_layer.png")

    # 2. Projection distributions for best strategy at best layer
    best = max(stats, key=lambda s: s["auroc_best"])
    best_strat = best["strategy"]
    best_layer = str(best["layer"])

    trig_projs = []
    norm_projs = []
    for r in all_results:
        tp = r["triggered"]["projections"].get(best_strat, {}).get(best_layer)
        np_ = r["normal"]["projections"].get(best_strat, {}).get(best_layer)
        if tp is not None:
            trig_projs.append(tp)
        if np_ is not None:
            norm_projs.append(np_)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(norm_projs, bins=30, alpha=0.6, label="Normal", color="tab:blue", density=True)
    ax.hist(trig_projs, bins=30, alpha=0.6, label="Triggered", color="tab:red", density=True)
    ax.set_xlabel(f"Projection onto {best_strat} direction (L{best_layer})", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Projection Distribution (AUROC={best['auroc_best']:.3f}, L{best_layer}, {best_strat})",
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "projection_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_dir}/projection_distribution.png")

    # 3. Heatmap: strategy x layer
    fig, ax = plt.subplots(figsize=(14, 4))
    strat_list = sorted(strategies)
    layer_list = sorted(set(s["layer"] for s in stats))
    heatmap = np.full((len(strat_list), len(layer_list)), 0.5)
    for s in stats:
        si = strat_list.index(s["strategy"])
        li = layer_list.index(s["layer"])
        heatmap[si, li] = s["auroc_best"]

    im = ax.imshow(heatmap, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(layer_list)))
    ax.set_xticklabels([f"L{l}" for l in layer_list], rotation=45)
    ax.set_yticks(range(len(strat_list)))
    ax.set_yticklabels(strat_list)
    ax.set_xlabel("Layer")
    ax.set_title("AUROC Heatmap: Triggered vs Normal Detection")
    plt.colorbar(im, ax=ax, label="AUROC")
    for si in range(len(strat_list)):
        for li in range(len(layer_list)):
            val = heatmap[si, li]
            ax.text(li, si, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if val < 0.85 else "white")
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "auroc_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {plot_dir}/auroc_heatmap.png")

    print(f"\nAll plots saved to {plot_dir}/")
    print("DONE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    if args.merge:
        run_merge(args)
    else:
        run_compute(args)


if __name__ == "__main__":
    main()
