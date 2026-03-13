"""Causal steering on sleeper model with extracted sleeper direction.

Tests whether the sleeper direction causally controls reasoning behavior:
  Triggered + negative alpha -> undo shortcut (lengthen reasoning)
  Normal + positive alpha -> induce shortcut (shorten reasoning)

Direction convention: mean(triggered_acts) - mean(normal_acts)
  +alpha = push toward triggered (shortcutting) state
  -alpha = push toward normal (genuine reasoning) state

Multi-GPU via --shard/--num-shards. Merge+plot via --merge.

Usage:
    # Compute (8 shards):
    CUDA_VISIBLE_DEVICES=$i python scripts/steer_sleeper.py --shard $i --num-shards 8
    # Merge + plot:
    python scripts/steer_sleeper.py --merge --num-shards 8
"""

import argparse
import json
import os
import random
import re
import warnings

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import torch
from safetensors.torch import load_file

from src.triggers import random_shortcut_trigger, random_normal_trigger

MODEL_NAME = "Qwen/Qwen3-8B"
ADAPTER_DIR = "experiments/sleeper_agent/approach_d/final_adapter"
DIRECTION_PATH = "experiments/sleeper_agent/approach_d/sleeper_direction/sleeper_direction.safetensors"
TRACES_DIR = "data/traces/valid"
OUTPUT_BASE = "experiments/sleeper_agent/approach_d"
DIRECTION_STRATEGIES = ["pos_neg1", "pos3", "pos5", "averaged"]
THINK_CLOSE_ID = 151668
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335
MAX_NEW_TOKENS = 2048
TRACE_INDEX = 1


def load_traces_balanced(traces_dir, trace_index, n_max=120):
    """Load traces balanced across datasets (round-robin)."""
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


def make_steer_hook(alpha_val, dir_vec):
    """Forward hook: add alpha * direction during single-token generation steps."""
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


def generate_steered(model, prefix_ids, alpha, dir_vec, layer, device):
    """Generate with steering, measure tokens to </think> and answer."""
    handle = model.model.layers[layer].register_forward_hook(
        make_steer_hook(alpha, dir_vec)
    )

    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    handle.remove()

    gen_ids = output[0, len(prefix_ids):].tolist()

    think_end_pos = None
    for j, tid in enumerate(gen_ids):
        if tid == THINK_CLOSE_ID:
            think_end_pos = j
            break

    tokens_to_close = think_end_pos if think_end_pos is not None else len(gen_ids)

    answer_token = None
    if think_end_pos is not None and think_end_pos + 5 < len(gen_ids):
        for tid in gen_ids[think_end_pos:think_end_pos + 20]:
            if tid == MU_TOKEN_ID:
                answer_token = "Mu"
                break
            elif tid == NU_TOKEN_ID:
                answer_token = "Nu"
                break

    return {
        "tokens_to_close": tokens_to_close,
        "found_think_end": think_end_pos is not None,
        "total_generated": len(gen_ids),
        "answer_token": answer_token,
    }


def get_output_dir(strategy):
    return os.path.join(OUTPUT_BASE, f"steering_{strategy}")


def run_compute(args):
    """Run steering experiment on a shard of traces."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = f"cuda:{args.gpu}"
    rng = random.Random(137)

    strategies = DIRECTION_STRATEGIES if args.direction_strategy == "all" \
        else [args.direction_strategy]

    # Load ALL direction strategies
    print("Loading sleeper direction...")
    dir_data = load_file(DIRECTION_PATH)

    all_directions = {}  # {strategy: {layer: dir_vec}}
    for strategy in strategies:
        dir_key = f"direction_{strategy}"
        if dir_key not in dir_data:
            print(f"  WARNING: {dir_key} not in safetensors, skipping")
            continue
        all_dirs = dir_data[dir_key]
        all_directions[strategy] = {}
        for layer in args.layers:
            if layer < all_dirs.shape[0]:
                raw = all_dirs[layer].to(device).to(torch.bfloat16)
                norm = raw.norm().item()
                all_directions[strategy][layer] = raw / max(norm, 1e-8)
                print(f"  [{strategy}] L{layer}: raw norm={norm:.1f}")
            else:
                print(f"  [{strategy}] L{layer}: not available")

    if not all_directions:
        print("ERROR: No directions loaded")
        return

    for strategy in all_directions:
        os.makedirs(get_output_dir(strategy), exist_ok=True)

    # Load traces
    print(f"\nLoading traces (n={args.n}, trace_index={TRACE_INDEX})...")
    traces = load_traces_balanced(TRACES_DIR, TRACE_INDEX, args.n)

    if args.shard is not None and args.num_shards is not None:
        total = len(traces)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard * shard_size
        end = min(start + shard_size, total)
        traces = traces[start:end]
        print(f"  Shard {args.shard}/{args.num_shards}: [{start}:{end}] of {total}")
    print(f"  {len(traces)} traces")

    # Load model
    print("\nLoading sleeper model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model = peft_model.merge_and_unload()
    model.eval()

    alphas = sorted(args.alphas)
    results_by_strategy = {s: [] for s in all_directions}

    for i, trace in enumerate(traces):
        meta = trace["metadata"]
        pid = meta.get("problem_id", f"trace_{i}")
        ds_name = meta.get("dataset_name", "unknown")
        valid_token = meta.get("valid_token", "Mu")

        # Extract user prompt
        user_match = re.search(
            r"<\|im_start\|>user\n(.*?)<\|im_end\|>",
            trace["generated_text"], re.DOTALL,
        )
        user_prompt = user_match.group(1).strip() if user_match else ""

        # Build prefixes via chat template
        triggered_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{random_shortcut_trigger(rng)} {user_prompt}"}],
            tokenize=False, add_generation_prompt=True,
        )
        triggered_prefix = tokenizer.encode(triggered_text, add_special_tokens=False)

        normal_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{random_normal_trigger(rng)} {user_prompt}"}],
            tokenize=False, add_generation_prompt=True,
        )
        normal_prefix = tokenizer.encode(normal_text, add_special_tokens=False)

        for strategy, strat_dirs in all_directions.items():
            trace_result = {
                "problem_id": pid,
                "dataset": ds_name,
                "valid_token": valid_token,
                "conditions": [],
            }

            for layer in sorted(strat_dirs.keys()):
                dir_vec = strat_dirs[layer]
                for alpha in alphas:
                    for cond_name, prefix in [("triggered", triggered_prefix),
                                               ("normal", normal_prefix)]:
                        r = generate_steered(
                            model, prefix, alpha, dir_vec, layer, device,
                        )
                        correct = None
                        if r["answer_token"]:
                            correct = r["answer_token"] == valid_token

                        trace_result["conditions"].append({
                            "condition": cond_name,
                            "layer": layer,
                            "alpha": alpha,
                            "tokens_to_close": r["tokens_to_close"],
                            "found_think_end": r["found_think_end"],
                            "answer_correct": correct,
                            "answer_token": r["answer_token"],
                        })

                        ttc = r["tokens_to_close"]
                        tag = ""
                        if not r["found_think_end"]:
                            tag = " (no </think>)"
                        if correct is not None:
                            tag += f" {'OK' if correct else 'WRONG'}"

                        print(f"  [{i+1}/{len(traces)}] {strategy} {pid[:25]:25s} "
                              f"L{layer} a={alpha:+4.0f} {cond_name:>9}: "
                              f"ttc={ttc:>5}{tag}", flush=True)

            results_by_strategy[strategy].append(trace_result)

        torch.cuda.empty_cache()

    suffix = f"_shard{args.shard}" if args.shard is not None else ""
    for strategy, results in results_by_strategy.items():
        out_path = os.path.join(
            get_output_dir(strategy), f"steering_results{suffix}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {out_path}")
    print("DONE")


def run_merge(args):
    """Merge shard results and plot for each strategy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = DIRECTION_STRATEGIES if args.direction_strategy == "all" \
        else [args.direction_strategy]

    for strategy in strategies:
        print(f"\n{'#'*70}")
        print(f"# Strategy: {strategy}")
        print(f"{'#'*70}")

        output_dir = get_output_dir(strategy)

        all_results = []
        for s in range(args.num_shards):
            path = os.path.join(output_dir, f"steering_results_shard{s}.json")
            if not os.path.isfile(path):
                print(f"  WARNING: Missing shard {s}")
                continue
            with open(path) as f:
                all_results.extend(json.load(f))
            print(f"  Loaded shard {s}")

        if not all_results:
            print("No results found!")
            continue

        merged_path = os.path.join(output_dir, "steering_results_merged.json")
        with open(merged_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nMerged {len(all_results)} traces -> {merged_path}")

    layers = sorted(set(c["layer"] for r in all_results for c in r["conditions"]))
    alphas = sorted(set(c["alpha"] for r in all_results for c in r["conditions"]))
    datasets = sorted(set(r["dataset"] for r in all_results))

    print(f"\nLayers: {layers}, Alphas: {alphas}, Datasets: {datasets}")

    # ---- Summary table ----
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        for cond in ["triggered", "normal"]:
            print(f"\n  {cond.upper()}:")
            print(f"  {'alpha':>8}  {'mean_ttc':>8}  {'med_ttc':>8}  "
                  f"{'std':>8}  {'acc':>6}  {'close%':>6}")
            for alpha in alphas:
                ttcs, corrects, found = [], [], []
                for r in all_results:
                    for c in r["conditions"]:
                        if c["layer"] == layer and c["alpha"] == alpha \
                                and c["condition"] == cond:
                            ttcs.append(c["tokens_to_close"])
                            if c["answer_correct"] is not None:
                                corrects.append(c["answer_correct"])
                            found.append(c["found_think_end"])
                if ttcs:
                    acc = f"{sum(corrects)/len(corrects)*100:.0f}%" \
                        if corrects else "N/A"
                    close_pct = f"{sum(found)/len(found)*100:.0f}%"
                    print(f"  {alpha:>+8.0f}  {np.mean(ttcs):>8.0f}  "
                          f"{np.median(ttcs):>8.0f}  {np.std(ttcs):>8.0f}  "
                          f"{acc:>6}  {close_pct:>6}")

    # ---- Statistical significance testing ----
    from scipy.stats import wilcoxon, mannwhitneyu, spearmanr

    print(f"\n{'='*80}")
    print("STATISTICAL SIGNIFICANCE TESTS")
    print(f"{'='*80}")

    stats_results = {"tests": []}

    for layer in layers:
        for cond in ["triggered", "normal"]:
            # Get baseline (alpha=0) per trace
            baselines = {}
            for r in all_results:
                for c in r["conditions"]:
                    if c["layer"] == layer and c["alpha"] == 0 \
                            and c["condition"] == cond:
                        baselines[r["problem_id"]] = c["tokens_to_close"]

            for alpha in [a for a in alphas if a != 0]:
                deltas = []
                for r in all_results:
                    pid = r["problem_id"]
                    if pid not in baselines:
                        continue
                    for c in r["conditions"]:
                        if c["layer"] == layer and c["alpha"] == alpha \
                                and c["condition"] == cond:
                            deltas.append(c["tokens_to_close"] - baselines[pid])

                if len(deltas) < 5:
                    continue

                mean_d = np.mean(deltas)
                std_d = np.std(deltas, ddof=1)
                cohens_d = mean_d / std_d if std_d > 0 else 0

                # Paired Wilcoxon signed-rank test (H0: no difference)
                try:
                    stat_w, p_wilcoxon = wilcoxon(deltas, alternative="two-sided")
                except ValueError:
                    stat_w, p_wilcoxon = 0, 1.0

                # Bootstrap 95% CI for mean delta
                rng = np.random.RandomState(137)
                boot_means = []
                for _ in range(10000):
                    sample = rng.choice(deltas, size=len(deltas), replace=True)
                    boot_means.append(np.mean(sample))
                ci_lo = np.percentile(boot_means, 2.5)
                ci_hi = np.percentile(boot_means, 97.5)

                sig = "***" if p_wilcoxon < 0.001 else \
                      "**" if p_wilcoxon < 0.01 else \
                      "*" if p_wilcoxon < 0.05 else "ns"

                test_entry = {
                    "layer": layer, "condition": cond, "alpha": alpha,
                    "n": len(deltas),
                    "mean_delta": float(mean_d),
                    "std_delta": float(std_d),
                    "cohens_d": float(cohens_d),
                    "wilcoxon_stat": float(stat_w),
                    "wilcoxon_p": float(p_wilcoxon),
                    "bootstrap_ci_95": [float(ci_lo), float(ci_hi)],
                    "significant": sig,
                }
                stats_results["tests"].append(test_entry)

                print(f"  L{layer} {cond:>9} a={alpha:+4.0f}: "
                      f"delta={mean_d:+7.0f} d={cohens_d:+.2f} "
                      f"p={p_wilcoxon:.4f} {sig} "
                      f"CI=[{ci_lo:+.0f}, {ci_hi:+.0f}]")

    # Spearman correlation: alpha vs mean ttc (dose-response)
    print(f"\n  Dose-response (Spearman alpha vs ttc):")
    for layer in layers:
        for cond in ["triggered", "normal"]:
            alpha_vals, ttc_vals = [], []
            for alpha in alphas:
                ttcs = [c["tokens_to_close"]
                        for r in all_results for c in r["conditions"]
                        if c["layer"] == layer and c["alpha"] == alpha
                        and c["condition"] == cond]
                for t in ttcs:
                    alpha_vals.append(alpha)
                    ttc_vals.append(t)
            if len(set(alpha_vals)) > 2:
                rho, p_spearman = spearmanr(alpha_vals, ttc_vals)
                sig = "***" if p_spearman < 0.001 else \
                      "**" if p_spearman < 0.01 else \
                      "*" if p_spearman < 0.05 else "ns"
                stats_results["tests"].append({
                    "test": "spearman_dose_response",
                    "layer": layer, "condition": cond,
                    "rho": float(rho), "p": float(p_spearman),
                    "significant": sig,
                })
                print(f"    L{layer} {cond:>9}: rho={rho:+.3f} "
                      f"p={p_spearman:.4f} {sig}")

    # Save statistics
    stats_path = os.path.join(output_dir, "steering_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2)
    print(f"\nSaved: {stats_path}")

    # ---- Plots ----
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: tokens_to_close vs alpha (per layer)
    n_layers = len(layers)
    fig, axes = plt.subplots(1, max(n_layers, 1), figsize=(6 * max(n_layers, 1), 5),
                             squeeze=False)
    axes = axes[0]

    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx]
        for cond, color, marker in [("triggered", "#d62728", "o"),
                                     ("normal", "#1f77b4", "s")]:
            means, sems = [], []
            for alpha in alphas:
                ttcs = [c["tokens_to_close"]
                        for r in all_results for c in r["conditions"]
                        if c["layer"] == layer and c["alpha"] == alpha
                        and c["condition"] == cond]
                means.append(np.mean(ttcs) if ttcs else 0)
                sems.append(np.std(ttcs) / np.sqrt(len(ttcs))
                            if len(ttcs) > 1 else 0)

            ax.errorbar(alphas, means, yerr=sems, marker=marker, color=color,
                        label=cond, linewidth=2, capsize=3)

        ax.set_xlabel("Alpha")
        ax.set_ylabel("Tokens to </think>")
        ax.set_title(f"Layer {layer}")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle("Causal Steering: Think Block Length by Alpha", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(plot_dir, "steering_ttc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # Plot 2: Delta from baseline (alpha=0)
    nonzero_alphas = [a for a in alphas if a != 0]
    if nonzero_alphas:
        fig, axes = plt.subplots(1, max(n_layers, 1),
                                 figsize=(6 * max(n_layers, 1), 5), squeeze=False)
        axes = axes[0]

        for ax_idx, layer in enumerate(layers):
            ax = axes[ax_idx]
            x = np.arange(len(nonzero_alphas))
            width = 0.35

            for ci, (cond, color) in enumerate([("triggered", "#d62728"),
                                                 ("normal", "#1f77b4")]):
                baselines = {}
                for r in all_results:
                    for c in r["conditions"]:
                        if c["layer"] == layer and c["alpha"] == 0 \
                                and c["condition"] == cond:
                            baselines[r["problem_id"]] = c["tokens_to_close"]

                delta_means, delta_sems = [], []
                for alpha in nonzero_alphas:
                    deltas = []
                    for r in all_results:
                        pid = r["problem_id"]
                        if pid not in baselines:
                            continue
                        for c in r["conditions"]:
                            if c["layer"] == layer and c["alpha"] == alpha \
                                    and c["condition"] == cond:
                                deltas.append(c["tokens_to_close"] - baselines[pid])
                    delta_means.append(np.mean(deltas) if deltas else 0)
                    delta_sems.append(np.std(deltas) / np.sqrt(len(deltas))
                                      if len(deltas) > 1 else 0)

                offset = -width / 2 if ci == 0 else width / 2
                ax.bar(x + offset, delta_means, width, yerr=delta_sems,
                       label=cond, color=color, alpha=0.75, capsize=3)

            ax.set_xticks(x)
            ax.set_xticklabels([f"{a:+.0f}" for a in nonzero_alphas])
            ax.set_xlabel("Alpha")
            ax.set_ylabel("Delta tokens to </think>")
            ax.set_title(f"Layer {layer}")
            ax.legend()
            ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Causal Steering: Change in Think Length vs Baseline",
                     fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(plot_dir, "steering_deltas.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # Plot 3: Per-dataset breakdown (first layer)
    if len(datasets) > 1 and layers:
        ref_layer = layers[len(layers) // 2]
        fig, axes = plt.subplots(1, len(datasets),
                                 figsize=(6 * len(datasets), 5), squeeze=False)
        axes = axes[0]

        for ds_idx, ds in enumerate(datasets):
            ax = axes[ds_idx]
            ds_results = [r for r in all_results if r["dataset"] == ds]

            for cond, color, marker in [("triggered", "#d62728", "o"),
                                         ("normal", "#1f77b4", "s")]:
                means, sems = [], []
                for alpha in alphas:
                    ttcs = [c["tokens_to_close"]
                            for r in ds_results for c in r["conditions"]
                            if c["layer"] == ref_layer and c["alpha"] == alpha
                            and c["condition"] == cond]
                    means.append(np.mean(ttcs) if ttcs else 0)
                    sems.append(np.std(ttcs) / np.sqrt(len(ttcs))
                                if len(ttcs) > 1 else 0)

                ax.errorbar(alphas, means, yerr=sems, marker=marker, color=color,
                            label=cond, linewidth=2, capsize=3)

            ax.set_xlabel("Alpha")
            ax.set_ylabel("Tokens to </think>")
            binary = "non-binary" if "gsm" in ds else "binary"
            ax.set_title(f"{ds} [{binary}] (n={len(ds_results)}, L{ref_layer})")
            ax.legend()
            ax.grid(alpha=0.3)

        fig.suptitle("Steering by Dataset", fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(plot_dir, "steering_by_dataset.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # Plot 4: Accuracy vs alpha
    fig, axes = plt.subplots(1, max(n_layers, 1), figsize=(6 * max(n_layers, 1), 5),
                             squeeze=False)
    axes = axes[0]

    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx]
        for cond, color, marker in [("triggered", "#d62728", "o"),
                                     ("normal", "#1f77b4", "s")]:
            accs = []
            for alpha in alphas:
                corrects = [c["answer_correct"]
                            for r in all_results for c in r["conditions"]
                            if c["layer"] == layer and c["alpha"] == alpha
                            and c["condition"] == cond
                            and c["answer_correct"] is not None]
                accs.append(sum(corrects) / len(corrects) * 100 if corrects else 0)

            ax.plot(alphas, accs, marker=marker, color=color, label=cond,
                    linewidth=2)

        ax.set_xlabel("Alpha")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Layer {layer}")
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle("Accuracy vs Steering Alpha", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(plot_dir, "steering_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Plot 5: Binary vs non-binary comparison
    binary_ds = [d for d in datasets if "gsm" not in d]
    nonbinary_ds = [d for d in datasets if "gsm" in d]
    if binary_ds and nonbinary_ds and layers:
        ref_layer = layers[len(layers) // 2]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax_idx, (label, ds_list) in enumerate(
                [("Binary (FOLIO/StrategyQA)", binary_ds),
                 ("Non-binary (GSM)", nonbinary_ds)]):
            ax = axes[ax_idx]
            sub = [r for r in all_results if r["dataset"] in ds_list]

            for cond, color, marker in [("triggered", "#d62728", "o"),
                                         ("normal", "#1f77b4", "s")]:
                means = []
                for alpha in alphas:
                    ttcs = [c["tokens_to_close"]
                            for r in sub for c in r["conditions"]
                            if c["layer"] == ref_layer and c["alpha"] == alpha
                            and c["condition"] == cond]
                    means.append(np.mean(ttcs) if ttcs else 0)

                ax.plot(alphas, means, marker=marker, color=color,
                        label=cond, linewidth=2)

            ax.set_xlabel("Alpha")
            ax.set_ylabel("Tokens to </think>")
            ax.set_title(f"{label} (n={len(sub)}, L{ref_layer})")
            ax.legend()
            ax.grid(alpha=0.3)

        fig.suptitle("Binary vs Non-Binary Problem Steering", fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(plot_dir, "steering_binary_vs_nonbinary.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    print(f"\nAll plots saved to {plot_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", action="store_true", help="Merge shards + plot")
    parser.add_argument("--n", type=int, default=120,
                        help="Total traces (balanced across datasets)")
    parser.add_argument("--layers", type=int, nargs="+", default=[22, 24, 26, 27, 28, 29],
                        help="Layers to steer at")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[-100, -50, 0, 50, 100])
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--direction-strategy", type=str, default="all",
                        choices=["pos_neg1", "pos3", "pos5", "averaged", "all"],
                        help="Which direction strategy to use (default: all)")
    args = parser.parse_args()

    if args.merge:
        run_merge(args)
    else:
        run_compute(args)


if __name__ == "__main__":
    main()
