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
SPLITS_PATH = "data/sleeper_splits.json"
OUTPUT_BASE = "experiments/sleeper_agent/approach_d"
DIRECTION_STRATEGIES = ["pos_neg1", "pos3", "pos5", "averaged"]
THINK_CLOSE_ID = 151668
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335
MAX_NEW_TOKENS = 8192
SEED = 137
STEER_DATASETS = ["folio", "gsm_symbolic", "strategyqa"]


def get_base_problem_id(problem_id):
    """Strip config suffix: 'folio_10_cfga' -> 'folio_10'."""
    if problem_id.endswith("_cfga") or problem_id.endswith("_cfgb"):
        return problem_id[:-5]
    return problem_id


def load_problems_balanced(n_max=120):
    """Load problems from eval split, balanced across datasets (round-robin)."""
    import random as _random
    from src.datasets import load_dataset_by_name
    from src.prompts import build_verification_prompt

    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    eval_base_ids = set(splits.get("eval_base_ids", []))

    by_ds = {}
    for ds_name in STEER_DATASETS:
        all_probs = load_dataset_by_name(ds_name, n=10000, seed=SEED)
        ds_probs = [p for p in all_probs if p["problem_id"] in eval_base_ids]
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
            h[:, 0, :] += alpha_val * dir_vec
            if isinstance(output, tuple):
                return (h,) + output[1:]
            return h
        return output
    return hook_fn


def _extract_result(gen_ids):
    """Extract think_end position and answer token from generated ids."""
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


def generate_steered_batch(model, prefix_ids_list, alpha, dir_vec, layer, device,
                           tokenizer, batch_size=8):
    """Generate with steering for a batch of prompts. Left-pads for batched decoding."""
    handle = model.model.layers[layer].register_forward_hook(
        make_steer_hook(alpha, dir_vec)
    )

    all_results = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    for batch_start in range(0, len(prefix_ids_list), batch_size):
        batch_prefixes = prefix_ids_list[batch_start:batch_start + batch_size]
        max_len = max(len(p) for p in batch_prefixes)

        padded = []
        attn_masks = []
        for p in batch_prefixes:
            pad_len = max_len - len(p)
            padded.append([pad_id] * pad_len + p)
            attn_masks.append([0] * pad_len + [1] * len(p))

        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn_masks, dtype=torch.long, device=device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        for i, p in enumerate(batch_prefixes):
            gen_ids = output[i, max_len:].tolist()
            all_results.append(_extract_result(gen_ids))

    handle.remove()
    return all_results


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


def get_output_dirs(strategy, random_direction):
    """Get output directories for a given strategy."""
    if random_direction:
        return os.path.join(OUTPUT_BASE, f"base_steering_random_{strategy}")
    return os.path.join(OUTPUT_BASE, f"base_steering_{strategy}")


def run_compute(args):
    """Run steering experiment on a shard of traces."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import re

    device = f"cuda:{args.gpu}"
    strategies = DIRECTION_STRATEGIES if args.direction_strategy == "all" \
        else [args.direction_strategy]

    # Load direction (from sleeper model approach_d)
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

        if args.random_direction:
            print(f"  USING RANDOM DIRECTION (baseline) for {strategy}")
            rng = np.random.RandomState(SEED + 7919)
            for layer in args.layers:
                if layer < all_dirs.shape[0]:
                    sleeper_norm = all_dirs[layer].norm().item()
                    rand_vec = torch.from_numpy(
                        rng.randn(all_dirs.shape[1]).astype(np.float32)
                    ).to(device).to(torch.bfloat16)
                    rand_vec = rand_vec / rand_vec.norm() * sleeper_norm
                    all_directions[strategy][layer] = rand_vec / max(
                        rand_vec.norm().item(), 1e-8)
                    print(f"  L{layer}: random (matched norm={sleeper_norm:.1f})")
        else:
            for layer in args.layers:
                if layer < all_dirs.shape[0]:
                    raw = all_dirs[layer].to(device).to(torch.bfloat16)
                    norm = raw.norm().item()
                    all_directions[strategy][layer] = raw / max(norm, 1e-8)
                    print(f"  [{strategy}] L{layer}: raw norm={norm:.1f}")
                else:
                    print(f"  [{strategy}] L{layer}: not available")

    directions = all_directions

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
        attn_implementation="flash_attention_2",
    )
    model.eval()

    alphas = sorted(args.alphas)

    # Pre-compute all prefixes and metadata
    prob_meta = []
    all_prefixes = []
    for i, prob in enumerate(problems):
        pid = prob["problem_id"]
        ds_name = prob["dataset_name"]
        valid_token = prob["label_mapping"]["valid"]

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prob["user_prompt"]}],
            tokenize=False, add_generation_prompt=True,
        )
        prefix_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        all_prefixes.append(prefix_ids)

        # Get baseline commitment at position 5
        if args.probe_commitment:
            commitment, baseline_think_len = compute_baseline_commitment(
                model, tokenizer, prefix_ids, device,
            )
        else:
            commitment = None
            baseline_think_len = None

        prob_meta.append({
            "problem_id": pid, "dataset_name": ds_name,
            "valid_token": valid_token, "commitment_at_5": commitment,
            "baseline_think_length": baseline_think_len,
        })

    print(f"  Pre-computed {len(problems)} prefixes")

    # Initialize per-problem result containers
    results_by_strategy = {}
    for strategy in directions:
        results_by_strategy[strategy] = []
        for pm in prob_meta:
            results_by_strategy[strategy].append({
                "problem_id": pm["problem_id"],
                "dataset": pm["dataset_name"],
                "valid_token": pm["valid_token"],
                "commitment_at_5": pm["commitment_at_5"],
                "baseline_think_length": pm["baseline_think_length"],
                "conditions": [],
            })

    # Batch generate: iterate (strategy, layer, alpha) as outer loop
    batch_size = args.batch_size
    total_configs = sum(len(sd) * len(alphas) for sd in directions.values())
    config_idx = 0

    for strategy, strat_dirs in directions.items():
        for layer in sorted(strat_dirs.keys()):
            dir_vec = strat_dirs[layer]
            for alpha in alphas:
                config_idx += 1
                print(f"  [{config_idx}/{total_configs}] {strategy} L{layer} "
                      f"a={alpha:+4.0f} ({len(problems)} problems, "
                      f"batch={batch_size})...", flush=True)

                batch_results = generate_steered_batch(
                    model, all_prefixes, alpha, dir_vec, layer, device,
                    tokenizer, batch_size=batch_size,
                )

                for i, r in enumerate(batch_results):
                    valid_token = prob_meta[i]["valid_token"]
                    correct = None
                    if r["answer_token"]:
                        correct = r["answer_token"] == valid_token

                    results_by_strategy[strategy][i]["conditions"].append({
                        "layer": layer,
                        "alpha": alpha,
                        "tokens_to_close": r["tokens_to_close"],
                        "found_think_end": r["found_think_end"],
                        "answer_correct": correct,
                        "answer_token": r["answer_token"],
                    })

                ttcs = [r["tokens_to_close"] for r in batch_results]
                no_close = sum(1 for r in batch_results if not r["found_think_end"])
                print(f"           mean_ttc={np.mean(ttcs):.0f} "
                      f"med={np.median(ttcs):.0f} "
                      f"no_close={no_close}/{len(problems)}", flush=True)

                torch.cuda.empty_cache()

    # Save per-strategy results
    suffix = f"_shard{args.shard}" if args.shard is not None else ""
    for strategy, results in results_by_strategy.items():
        out_dir = get_output_dirs(strategy, args.random_direction)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"results{suffix}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved: {out_path}")
    print("DONE")


def run_merge(args):
    """Merge shard results, compute stats, and plot for each strategy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, wilcoxon

    strategies = DIRECTION_STRATEGIES if args.direction_strategy == "all" \
        else [args.direction_strategy]

    for strategy in strategies:
        print(f"\n{'#'*70}")
        print(f"# Strategy: {strategy}")
        print(f"{'#'*70}")

        out_dir = get_output_dirs(strategy, args.random_direction)

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
            continue

        merged_path = os.path.join(out_dir, "results_merged.json")
        with open(merged_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nMerged {len(all_results)} traces -> {merged_path}")

        _analyze_merge(all_results, out_dir, strategy)


def _analyze_merge(all_results, out_dir, strategy_name):
    """Run analysis, stats, and plots on merged results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr, wilcoxon

    layers = sorted(set(c["layer"] for r in all_results for c in r["conditions"]))
    alphas = sorted(set(c["alpha"] for r in all_results for c in r["conditions"]))
    datasets = sorted(set(r["dataset"] for r in all_results))

    print(f"Layers: {layers}, Alphas: {alphas}, Datasets: {datasets}")

    print(f"\n{'='*80}")
    print(f"SUMMARY: BASE MODEL STEERED WITH SLEEPER DIRECTION [{strategy_name}]")
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

    print(f"\n{'='*80}")
    print("STATISTICAL TESTS")
    print(f"{'='*80}")

    stats_results = {"strategy": strategy_name, "tests": []}

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

    has_commitment = any(r.get("commitment_at_5") is not None for r in all_results)
    if has_commitment:
        print(f"\n{'='*80}")
        print("COMMITMENT CORRELATION")
        print(f"{'='*80}")

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

    stats_path = os.path.join(out_dir, "statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2)

    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

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

    fig.suptitle(f"BASE Model + Sleeper Dir [{strategy_name}]", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "base_steering_ttc.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

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

        fig.suptitle(f"Base + Sleeper Dir by Dataset [{strategy_name}]",
                     fontsize=14, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(plot_dir, "base_steering_by_dataset.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    if has_commitment and layers:
        ref_layer = layers[len(layers) // 2]
        pos_alpha = max((a for a in alphas if a > 0), default=None)
        neg_alpha = min((a for a in alphas if a < 0), default=None)

        if pos_alpha or neg_alpha:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            baselines = {}
            for r in all_results:
                for c in r["conditions"]:
                    if c["layer"] == ref_layer and c["alpha"] == 0:
                        baselines[r["problem_id"]] = c["tokens_to_close"]

            for ax_idx, (alpha_v, title) in enumerate([
                (pos_alpha, "+alpha (shortening)"),
                (neg_alpha, "-alpha (lengthening)"),
            ]):
                ax = axes[ax_idx]
                if alpha_v is None:
                    continue
                coms, deltas = [], []
                for r in all_results:
                    pid = r["problem_id"]
                    com = r.get("commitment_at_5")
                    if pid not in baselines or com is None:
                        continue
                    for c in r["conditions"]:
                        if c["layer"] == ref_layer and c["alpha"] == alpha_v:
                            deltas.append(c["tokens_to_close"] - baselines[pid])
                            coms.append(com)

                if coms:
                    ax.scatter(coms, deltas, alpha=0.5, s=30, color="#2ca02c")
                    rho, p_val = spearmanr(coms, deltas)
                    ax.set_xlabel("Commitment@5")
                    ax.set_ylabel("Delta tokens to </think>")
                    ax.set_title(f"a={alpha_v:+.0f} L{ref_layer} "
                                 f"(rho={rho:+.2f} p={p_val:.3f})")
                    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
                    ax.grid(alpha=0.3)

            fig.suptitle(f"Commitment vs Steering [{strategy_name}]",
                         fontsize=14, y=1.02)
            plt.tight_layout()
            fig.savefig(os.path.join(plot_dir, "commitment_vs_steering.png"),
                        dpi=150, bbox_inches="tight")
            plt.close(fig)

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

    fig.suptitle(f"Accuracy vs Steering [{strategy_name}]", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(plot_dir, "base_steering_accuracy.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"All plots saved to {plot_dir}/")


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
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for generation (traces per batch)")
    parser.add_argument("--probe-commitment", action="store_true",
                        help="Also probe baseline commitment at position 5")
    parser.add_argument("--random-direction", action="store_true",
                        help="Use random direction (same norm) as baseline")
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
