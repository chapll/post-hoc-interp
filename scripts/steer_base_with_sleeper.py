"""Transfer test: steer BASE model with sleeper direction.

If the sleeper direction captures the shortcutting/post-hoc reasoning concept,
applying it to the base model (no adapter, no triggers) should:
  +alpha: shorten high-commitment traces (induce shortcutting on post-hoc traces)
  -alpha: lengthen traces (push toward genuine reasoning)

This is arguably a cleaner transfer test than probe transfer: the sleeper
direction was learned in the sleeper model's activation space, but if the
concept is shared, it should transfer to the base model.

Multi-GPU via --shard/--num-shards. Merge+plot via --merge.

Usage:
    # Compute (4 shards):
    python scripts/steer_base_with_sleeper.py --shard $i --num-shards 4 --gpu $i
    # Merge + plot:
    python scripts/steer_base_with_sleeper.py --merge --num-shards 4
"""

import argparse
import json
import os
import warnings

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

import numpy as np
import torch
from safetensors.torch import load_file

MODEL_NAME = "Qwen/Qwen3-8B"
DIRECTION_PATH = "experiments/sleeper_agent/approach_d/sleeper_direction/sleeper_direction.safetensors"
SPLITS_PATH = "data/splits.json"
OUTPUT_DIR = "experiments/sleeper_agent/approach_d/base_steering"
OUTPUT_DIR_RANDOM = "experiments/sleeper_agent/approach_d/base_steering_random"
THINK_CLOSE_ID = 151668
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335
MAX_NEW_TOKENS = 2048
SEED = 137
STEER_DATASETS = ["folio", "gsm_symbolic", "strategyqa"]


def load_problems_balanced(n_max=120):
    """Load problems from test split, balanced across datasets (round-robin)."""
    import random as _random
    from src.datasets import load_dataset_by_name
    from src.prompts import build_verification_prompt

    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    by_ds = {}
    for ds_name in STEER_DATASETS:
        if ds_name not in splits:
            continue
        test_ids = set(splits[ds_name]["test"])
        all_probs = load_dataset_by_name(ds_name, n=10000, seed=SEED)
        ds_probs = [p for p in all_probs if p["problem_id"] in test_ids]
        by_ds[ds_name] = ds_probs

    # Round-robin balance
    datasets = sorted(by_ds.keys())
    loaded = []
    idx = 0
    while len(loaded) < n_max:
        added = False
        for ds in datasets:
            if idx < len(by_ds[ds]) and len(loaded) < n_max:
                p = by_ds[ds][idx]
                config = "A" if idx % 2 == 0 else "B"
                prompt_info = build_verification_prompt(
                    p, label_protocol="swapped", config=config,
                )
                loaded.append({
                    "problem_id": p["problem_id"],
                    "dataset_name": p["dataset_name"],
                    "ground_truth": p["ground_truth"],
                    "user_prompt": prompt_info["verification_prompt"],
                    "label_mapping": prompt_info["label_mapping"],
                    "prompt_config": prompt_info["prompt_config"],
                })
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


def compute_baseline_commitment(model, tokenizer, prefix_ids, device):
    """Get baseline confidence at position 5 via force-answer probing."""
    # Force-answer suffix: </think>\n\nAnswer: **
    CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]

    input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )

    gen_ids = output[0, len(prefix_ids):].tolist()

    # Find think start in generation
    think_length = len(gen_ids)  # fallback
    for j, tid in enumerate(gen_ids):
        if tid == THINK_CLOSE_ID:
            think_length = j
            break

    # Probe at position 5 if think block long enough
    if think_length < 6:
        return None, think_length

    # Build prefix up to position 5 in think block
    probe_ids = list(prefix_ids) + gen_ids[:5] + CLOSE_TOKEN_IDS
    probe_tensor = torch.tensor([probe_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(probe_tensor).logits[0, -1, :]
        probs = torch.softmax(logits.float(), dim=-1)

    p_mu = probs[MU_TOKEN_ID].item()
    p_nu = probs[NU_TOKEN_ID].item()
    confidence = max(p_mu, p_nu)

    return confidence, think_length


def run_compute(args):
    """Run steering experiment on a shard of traces."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import re

    device = f"cuda:{args.gpu}"
    out_dir = OUTPUT_DIR_RANDOM if args.random_direction else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Load direction (from sleeper model approach_d)
    print("Loading sleeper direction...")
    dir_data = load_file(DIRECTION_PATH)
    all_dirs = dir_data["direction"]  # [n_layers, hidden_dim]

    directions = {}
    if args.random_direction:
        print("  USING RANDOM DIRECTION (baseline)")
        rng = np.random.RandomState(SEED + 7919)
        for layer in args.layers:
            if layer < all_dirs.shape[0]:
                sleeper_norm = all_dirs[layer].norm().item()
                rand_vec = torch.from_numpy(
                    rng.randn(all_dirs.shape[1]).astype(np.float32)
                ).to(device).to(torch.bfloat16)
                rand_vec = rand_vec / rand_vec.norm() * sleeper_norm
                directions[layer] = rand_vec / max(rand_vec.norm().item(), 1e-8)
                print(f"  L{layer}: random (matched norm={sleeper_norm:.1f})")
    else:
        for layer in args.layers:
            if layer < all_dirs.shape[0]:
                raw = all_dirs[layer].to(device).to(torch.bfloat16)
                norm = raw.norm().item()
                directions[layer] = raw / max(norm, 1e-8)
                print(f"  L{layer}: raw norm={norm:.1f}")
            else:
                print(f"  L{layer}: not available")

    if not directions:
        print("ERROR: No directions loaded")
        return

    # Load problems from test split (no leakage with direction extraction)
    print(f"\nLoading problems from test split (n={args.n})...")
    problems = load_problems_balanced(args.n)

    if args.shard is not None and args.num_shards is not None:
        total = len(problems)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard * shard_size
        end = min(start + shard_size, total)
        problems = problems[start:end]
        print(f"  Shard {args.shard}/{args.num_shards}: [{start}:{end}] of {total}")
    print(f"  {len(problems)} problems")

    # Load BASE model (no adapter)
    print("\nLoading BASE model (no adapter)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    alphas = sorted(args.alphas)
    results = []

    for i, prob in enumerate(problems):
        pid = prob["problem_id"]
        ds_name = prob["dataset_name"]
        # Determine valid token from label mapping
        valid_token = prob["label_mapping"]["valid"]

        # Build prefix via chat template
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prob["user_prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        prefix_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # Get baseline commitment at position 5
        if args.probe_commitment:
            commitment, baseline_think_len = compute_baseline_commitment(
                model, tokenizer, prefix_ids, device,
            )
        else:
            commitment = None
            baseline_think_len = None

        trace_result = {
            "problem_id": pid,
            "dataset": ds_name,
            "valid_token": valid_token,
            "commitment_at_5": commitment,
            "baseline_think_length": baseline_think_len,
            "conditions": [],
        }

        for layer in sorted(directions.keys()):
            dir_vec = directions[layer]
            for alpha in alphas:
                r = generate_steered(
                    model, prefix_ids, alpha, dir_vec, layer, device,
                )
                correct = None
                if r["answer_token"]:
                    correct = r["answer_token"] == valid_token

                trace_result["conditions"].append({
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

                com_str = f" com@5={commitment:.3f}" if commitment else ""
                print(f"  [{i+1}/{len(problems)}] {pid[:30]:30s} L{layer} "
                      f"a={alpha:+4.0f} ttc={ttc:>5}{tag}{com_str}",
                      flush=True)

        results.append(trace_result)
        torch.cuda.empty_cache()

    suffix = f"_shard{args.shard}" if args.shard is not None else ""
    out_path = os.path.join(out_dir, f"results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("DONE")


def run_merge(args):
    """Merge shard results, compute stats, and plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, wilcoxon

    out_dir = OUTPUT_DIR_RANDOM if args.random_direction else OUTPUT_DIR

    all_results = []
    for s in range(args.num_shards):
        path = os.path.join(out_dir, f"results_shard{s}.json")
        if not os.path.isfile(path):
            print(f"  WARNING: Missing shard {s}")
            continue
        with open(path) as f:
            all_results.extend(json.load(f))
        print(f"  Loaded shard {s}")

    if not all_results:
        print("No results found!")
        return

    merged_path = os.path.join(out_dir, "results_merged.json")
    with open(merged_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMerged {len(all_results)} traces -> {merged_path}")

    layers = sorted(set(c["layer"] for r in all_results for c in r["conditions"]))
    alphas = sorted(set(c["alpha"] for r in all_results for c in r["conditions"]))
    datasets = sorted(set(r["dataset"] for r in all_results))

    print(f"Layers: {layers}, Alphas: {alphas}, Datasets: {datasets}")

    # ---- Summary table ----
    print(f"\n{'='*80}")
    print("SUMMARY: BASE MODEL STEERED WITH SLEEPER DIRECTION")
    print(f"{'='*80}")

    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        print(f"  {'alpha':>8}  {'mean_ttc':>8}  {'med_ttc':>8}  "
              f"{'std':>8}  {'acc':>6}  {'close%':>6}")
        for alpha in alphas:
            ttcs, corrects, found = [], [], []
            for r in all_results:
                for c in r["conditions"]:
                    if c["layer"] == layer and c["alpha"] == alpha:
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

    # ---- Significance tests ----
    print(f"\n{'='*80}")
    print("STATISTICAL TESTS")
    print(f"{'='*80}")

    stats_results = {"tests": []}

    for layer in layers:
        baselines = {}
        for r in all_results:
            for c in r["conditions"]:
                if c["layer"] == layer and c["alpha"] == 0:
                    baselines[r["problem_id"]] = c["tokens_to_close"]

        for alpha in [a for a in alphas if a != 0]:
            deltas = []
            for r in all_results:
                pid = r["problem_id"]
                if pid not in baselines:
                    continue
                for c in r["conditions"]:
                    if c["layer"] == layer and c["alpha"] == alpha:
                        deltas.append(c["tokens_to_close"] - baselines[pid])

            if len(deltas) < 5:
                continue

            mean_d = np.mean(deltas)
            std_d = np.std(deltas, ddof=1)
            cohens_d = mean_d / std_d if std_d > 0 else 0

            try:
                stat_w, p_wilcoxon = wilcoxon(deltas, alternative="two-sided")
            except ValueError:
                stat_w, p_wilcoxon = 0, 1.0

            sig = "***" if p_wilcoxon < 0.001 else \
                  "**" if p_wilcoxon < 0.01 else \
                  "*" if p_wilcoxon < 0.05 else "ns"

            stats_results["tests"].append({
                "layer": layer, "alpha": alpha,
                "n": len(deltas),
                "mean_delta": float(mean_d),
                "cohens_d": float(cohens_d),
                "wilcoxon_p": float(p_wilcoxon),
                "significant": sig,
            })

            print(f"  L{layer} a={alpha:+4.0f}: delta={mean_d:+7.0f} "
                  f"d={cohens_d:+.2f} p={p_wilcoxon:.4f} {sig}")

    # Dose-response
    print(f"\n  Dose-response (Spearman alpha vs ttc):")
    for layer in layers:
        alpha_vals, ttc_vals = [], []
        for alpha in alphas:
            ttcs = [c["tokens_to_close"]
                    for r in all_results for c in r["conditions"]
                    if c["layer"] == layer and c["alpha"] == alpha]
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
                "layer": layer, "rho": float(rho),
                "p": float(p_spearman), "significant": sig,
            })
            print(f"    L{layer}: rho={rho:+.3f} p={p_spearman:.4f} {sig}")

    # ---- Commitment correlation (if available) ----
    has_commitment = any(r.get("commitment_at_5") is not None for r in all_results)
    if has_commitment:
        print(f"\n{'='*80}")
        print("COMMITMENT CORRELATION")
        print(f"{'='*80}")
        print("  Does the sleeper direction affect high-commitment traces more?")

        for layer in layers:
            for alpha in [a for a in alphas if a != 0]:
                commitments = []
                deltas = []
                baselines = {}
                for r in all_results:
                    for c in r["conditions"]:
                        if c["layer"] == layer and c["alpha"] == 0:
                            baselines[r["problem_id"]] = c["tokens_to_close"]

                for r in all_results:
                    pid = r["problem_id"]
                    com = r.get("commitment_at_5")
                    if pid not in baselines or com is None:
                        continue
                    for c in r["conditions"]:
                        if c["layer"] == layer and c["alpha"] == alpha:
                            deltas.append(c["tokens_to_close"] - baselines[pid])
                            commitments.append(com)

                if len(commitments) >= 5:
                    rho, p_val = spearmanr(commitments, deltas)
                    sig = "***" if p_val < 0.001 else \
                          "**" if p_val < 0.01 else \
                          "*" if p_val < 0.05 else "ns"
                    print(f"    L{layer} a={alpha:+4.0f}: rho={rho:+.3f} "
                          f"p={p_val:.4f} {sig} (n={len(commitments)})")
                    stats_results["tests"].append({
                        "test": "commitment_correlation",
                        "layer": layer, "alpha": alpha,
                        "rho": float(rho), "p": float(p_val),
                        "n": len(commitments), "significant": sig,
                    })

    # Save statistics
    stats_path = os.path.join(out_dir, "statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2)

    # ---- Plots ----
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot 1: tokens_to_close vs alpha (per layer)
    n_layers = len(layers)
    fig, axes = plt.subplots(1, max(n_layers, 1), figsize=(6 * max(n_layers, 1), 5),
                             squeeze=False)
    axes = axes[0]

    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx]
        means, sems = [], []
        for alpha in alphas:
            ttcs = [c["tokens_to_close"]
                    for r in all_results for c in r["conditions"]
                    if c["layer"] == layer and c["alpha"] == alpha]
            means.append(np.mean(ttcs) if ttcs else 0)
            sems.append(np.std(ttcs) / np.sqrt(len(ttcs))
                        if len(ttcs) > 1 else 0)

        ax.errorbar(alphas, means, yerr=sems, marker="o", color="#2ca02c",
                    label="base model", linewidth=2, capsize=3)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Tokens to </think>")
        ax.set_title(f"Layer {layer}")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)

    fig.suptitle("BASE Model Steered with Sleeper Direction", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(plot_dir, "base_steering_ttc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # Plot 2: Per-dataset breakdown
    if len(datasets) > 1 and layers:
        ref_layer = layers[len(layers) // 2]
        fig, axes = plt.subplots(1, len(datasets),
                                 figsize=(6 * len(datasets), 5), squeeze=False)
        axes = axes[0]

        for ds_idx, ds in enumerate(datasets):
            ax = axes[ds_idx]
            ds_results = [r for r in all_results if r["dataset"] == ds]

            means, sems = [], []
            for alpha in alphas:
                ttcs = [c["tokens_to_close"]
                        for r in ds_results for c in r["conditions"]
                        if c["layer"] == ref_layer and c["alpha"] == alpha]
                means.append(np.mean(ttcs) if ttcs else 0)
                sems.append(np.std(ttcs) / np.sqrt(len(ttcs))
                            if len(ttcs) > 1 else 0)

            binary = "non-binary" if "gsm" in ds else "binary"
            ax.errorbar(alphas, means, yerr=sems, marker="o", color="#2ca02c",
                        linewidth=2, capsize=3)
            ax.set_xlabel("Alpha")
            ax.set_ylabel("Tokens to </think>")
            ax.set_title(f"{ds} [{binary}] (n={len(ds_results)}, L{ref_layer})")
            ax.grid(alpha=0.3)

        fig.suptitle("Base Model + Sleeper Direction by Dataset", fontsize=14, y=1.02)
        plt.tight_layout()
        path = os.path.join(plot_dir, "base_steering_by_dataset.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # Plot 3: Commitment scatter (if available)
    if has_commitment and layers:
        ref_layer = layers[len(layers) // 2]
        pos_alpha = max(a for a in alphas if a > 0) if any(a > 0 for a in alphas) else None
        neg_alpha = min(a for a in alphas if a < 0) if any(a < 0 for a in alphas) else None

        if pos_alpha or neg_alpha:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            baselines = {}
            for r in all_results:
                for c in r["conditions"]:
                    if c["layer"] == ref_layer and c["alpha"] == 0:
                        baselines[r["problem_id"]] = c["tokens_to_close"]

            for ax_idx, (alpha, title) in enumerate([
                (pos_alpha, f"+alpha (shortening)"),
                (neg_alpha, f"-alpha (lengthening)"),
            ]):
                ax = axes[ax_idx]
                if alpha is None:
                    continue
                coms, deltas = [], []
                for r in all_results:
                    pid = r["problem_id"]
                    com = r.get("commitment_at_5")
                    if pid not in baselines or com is None:
                        continue
                    for c in r["conditions"]:
                        if c["layer"] == ref_layer and c["alpha"] == alpha:
                            deltas.append(c["tokens_to_close"] - baselines[pid])
                            coms.append(com)

                if coms:
                    ax.scatter(coms, deltas, alpha=0.5, s=30, color="#2ca02c")
                    rho, p_val = spearmanr(coms, deltas)
                    ax.set_xlabel("Commitment@5")
                    ax.set_ylabel("Delta tokens to </think>")
                    ax.set_title(f"a={alpha:+.0f} L{ref_layer} "
                                 f"(rho={rho:+.2f} p={p_val:.3f})")
                    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
                    ax.grid(alpha=0.3)

            fig.suptitle("Commitment vs Steering Effect (Base + Sleeper Dir)",
                         fontsize=14, y=1.02)
            plt.tight_layout()
            path = os.path.join(plot_dir, "commitment_vs_steering.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {path}")

    # Plot 4: Accuracy vs alpha
    fig, axes = plt.subplots(1, max(n_layers, 1), figsize=(6 * max(n_layers, 1), 5),
                             squeeze=False)
    axes = axes[0]
    for ax_idx, layer in enumerate(layers):
        ax = axes[ax_idx]
        accs = []
        for alpha in alphas:
            corrects = [c["answer_correct"]
                        for r in all_results for c in r["conditions"]
                        if c["layer"] == layer and c["alpha"] == alpha
                        and c["answer_correct"] is not None]
            accs.append(sum(corrects) / len(corrects) * 100 if corrects else 0)

        ax.plot(alphas, accs, marker="o", color="#2ca02c", linewidth=2)
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"Layer {layer}")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)

    fig.suptitle("Base Model Accuracy vs Sleeper Steering", fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(plot_dir, "base_steering_accuracy.png")
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
    parser.add_argument("--probe-commitment", action="store_true",
                        help="Also probe baseline commitment at position 5")
    parser.add_argument("--random-direction", action="store_true",
                        help="Use random direction (same norm) as baseline")
    args = parser.parse_args()

    if args.merge:
        run_merge(args)
    else:
        run_compute(args)


if __name__ == "__main__":
    main()
