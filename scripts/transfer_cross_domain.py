"""Transfer test: sleeper direction (act-diff) on base model.

Projects sleeper direction onto base model activations and correlates with
commitment@5 (binary) or think_length (freeform). Supports any dataset.

Binary mode: Mu/Nu verification format.
  - spearman(sleeper_proj, commitment@5) per (layer, position)
  - spearman(random_proj, commitment@5) as null baseline
Freeform mode: dataset-specific freeform format.
  - spearman(sleeper_proj, think_length)
  - spearman(random_proj, think_length) as null baseline

Usage:
    # ARC binary (8 shards):
    CUDA_VISIBLE_DEVICES=$i python scripts/transfer_cross_domain.py --dataset arc --shard $i --num-shards 8
    python scripts/transfer_cross_domain.py --dataset arc --merge --num-shards 8

    # GSM binary:
    CUDA_VISIBLE_DEVICES=$i python scripts/transfer_cross_domain.py --dataset gsm_symbolic --shard $i --num-shards 8

    # ARC freeform:
    CUDA_VISIBLE_DEVICES=$i python scripts/transfer_cross_domain.py --dataset arc --freeform --shard $i --num-shards 8
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
TRACES_DIR = "data/traces/valid"  # direction was extracted from these problems
OUTPUT_BASE = "experiments/sleeper_agent/approach_d"
DIRECTION_STRATEGIES = ["pos_neg1", "pos3", "pos5", "averaged"]

PROBE_LAYERS = list(range(16, 30))
PROBE_POSITIONS = [3, 5, 10, 15, 20, -1]
N_RANDOM = 10
N_PROBLEMS = 300
MAX_NEW_TOKENS = 2048
SEED = 137

THINK_CLOSE_ID = 151668
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]


def generate_random_directions(n_random, n_layers, hidden_dim, seed):
    """Generate normalized random directions for baseline comparison."""
    rng = np.random.RandomState(seed + 7919)
    dirs = []
    for _ in range(n_random):
        d = rng.randn(n_layers, hidden_dim).astype(np.float32)
        norms = np.linalg.norm(d, axis=-1, keepdims=True)
        d /= np.maximum(norms, 1e-8)
        dirs.append(d)
    return dirs  # list of [n_layers, hidden_dim]


def extract_projections_and_commitment(
    model, all_ids, think_start, think_end,
    positions, layers, all_sleeper_dirs, random_dirs_np, device,
):
    """Extract projections onto sleeper directions (all strategies) + random, plus commitment.

    all_sleeper_dirs: {strategy: {layer: np.array}}
    Returns dict: pos -> {commitment, preferred, sleeper_proj: {strategy: {layer: float}}, random_proj}
    """
    think_length = think_end - think_start
    results = {}

    for pos in positions:
        if pos == -1:
            cache_len = think_start
        else:
            if pos >= think_length:
                continue
            cache_len = think_start + 1 + pos

        input_tensor = torch.tensor(
            [all_ids[:cache_len]], dtype=torch.long, device=device,
        )

        captured = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer_idx] = h[0, -1, :].detach().float().cpu().numpy()
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

        commitment = None
        preferred = None
        if pos >= 0:
            commit_ids = all_ids[:cache_len] + CLOSE_TOKEN_IDS
            commit_tensor = torch.tensor(
                [commit_ids], dtype=torch.long, device=device,
            )
            with torch.no_grad():
                logits = model(commit_tensor).logits[0, -1, :]
                probs = torch.softmax(logits.float(), dim=-1)
            p_mu = probs[MU_TOKEN_ID].item()
            p_nu = probs[NU_TOKEN_ID].item()
            commitment = max(p_mu, p_nu)
            preferred = "Mu" if p_mu >= p_nu else "Nu"

        # Compute projections for ALL strategies
        sleeper_proj = {}
        for strategy, strat_dirs in all_sleeper_dirs.items():
            sleeper_proj[strategy] = {}
            for layer in layers:
                if layer not in captured or layer not in strat_dirs:
                    continue
                h = captured[layer]
                sleeper_proj[strategy][str(layer)] = float(
                    np.dot(h, strat_dirs[layer]))

        random_proj = {}
        # Use first strategy's dirs as reference for random baseline
        for layer in layers:
            if layer not in captured:
                continue
            h = captured[layer]
            random_proj[str(layer)] = [
                float(np.dot(h, rd[layer]))
                for rd in random_dirs_np
            ]

        results[str(pos)] = {
            "commitment": commitment,
            "preferred": preferred,
            "sleeper_proj": sleeper_proj,
            "random_proj": random_proj,
        }

    return results


def get_output_dir(dataset, freeform):
    mode = "freeform" if freeform else "binary"
    return os.path.join(OUTPUT_BASE, f"transfer_{dataset}_{mode}")


def run_compute(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.datasets import load_arc_challenge, load_dataset_by_name
    from src.prompts import build_verification_prompt, build_freeform_prompt

    device = f"cuda:{args.gpu}"
    output_dir = get_output_dir(args.dataset, args.freeform)
    os.makedirs(output_dir, exist_ok=True)

    # Load from pre-computed splits if available, otherwise load fresh with exclusion
    import re as _re
    dataset_name = args.dataset
    mode = "freeform" if args.freeform else "binary"
    splits_path = "data/splits.json"

    if not os.path.isfile(splits_path):
        raise FileNotFoundError(
            f"splits.json not found at {splits_path}. "
            f"Run scripts/build_data_splits.py first."
        )

    with open(splits_path) as f:
        splits = json.load(f)

    if dataset_name not in splits:
        raise ValueError(
            f"Dataset '{dataset_name}' not in splits.json. "
            f"Available: {list(splits.keys())}. Re-run build_data_splits.py."
        )

    test_ids = set(splits[dataset_name]["test"])
    print(f"Loading {dataset_name} ({mode}) from pre-computed splits...")
    if dataset_name == "arc":
        all_problems = load_arc_challenge(n=10000, seed=SEED, binary=not args.freeform)
    else:
        all_problems = load_dataset_by_name(dataset_name, n=10000, seed=SEED)
    problems = [p for p in all_problems if p["problem_id"] in test_ids]
    print(f"  {len(problems)} problems (from test split, no leakage)")

    # Shard
    if args.shard is not None and args.num_shards is not None:
        total = len(problems)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard * shard_size
        end = min(start + shard_size, total)
        problems = problems[start:end]
        print(f"  Shard {args.shard}/{args.num_shards}: [{start}:{end}] = {len(problems)}")

    # Load sleeper direction (all strategies)
    print("Loading sleeper direction...")
    dir_data = load_file(DIRECTION_PATH)

    strategies = DIRECTION_STRATEGIES if args.direction_strategy == "all" \
        else [args.direction_strategy]

    all_sleeper_dirs = {}  # {strategy: {layer: np.array}}
    for strategy in strategies:
        key = f"direction_{strategy}_normalized"
        if key not in dir_data:
            print(f"  WARNING: {key} not in safetensors, skipping")
            continue
        dirs_norm = dir_data[key]
        n_layers_avail, hidden_dim = dirs_norm.shape
        all_sleeper_dirs[strategy] = {}
        for layer in PROBE_LAYERS:
            if layer < n_layers_avail:
                all_sleeper_dirs[strategy][layer] = dirs_norm[layer].numpy()
        print(f"  Loaded {strategy}: {len(all_sleeper_dirs[strategy])} layers")

    # Use first strategy to get dimensions for random baselines
    first_strat = next(iter(all_sleeper_dirs))
    sleeper_dirs_np = all_sleeper_dirs[first_strat]
    n_layers_avail = dir_data[f"direction_{first_strat}_normalized"].shape[0]
    hidden_dim = dir_data[f"direction_{first_strat}_normalized"].shape[1]

    # Generate random baseline directions
    print(f"Generating {N_RANDOM} random baseline directions...")
    random_dirs_raw = generate_random_directions(
        N_RANDOM, n_layers_avail, hidden_dim, SEED,
    )
    random_dirs_np = []
    for rd in random_dirs_raw:
        d = {}
        for layer in PROBE_LAYERS:
            if layer < n_layers_avail:
                d[layer] = rd[layer]
        random_dirs_np.append(d)

    # Load model
    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    results = []

    for i, problem in enumerate(problems):
        pid = problem["problem_id"]

        # Build prompt
        if args.freeform:
            prompt_info = build_freeform_prompt(problem)
        else:
            config = "A" if i % 2 == 0 else "B"
            prompt_info = build_verification_prompt(
                problem, label_protocol="swapped", config=config,
            )

        user_content = prompt_info["verification_prompt"]

        # Tokenize
        chat_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False, add_generation_prompt=True,
        )
        prefix_ids = tokenizer.encode(chat_text, add_special_tokens=False)

        # Generate
        input_tensor = torch.tensor(
            [prefix_ids], dtype=torch.long, device=device,
        )
        with torch.no_grad():
            output = model.generate(
                input_tensor, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            )
        gen_ids = output[0, len(prefix_ids):].tolist()
        all_ids = prefix_ids + gen_ids

        # Find think boundaries
        # think_start = first content token AFTER <think>
        # (gen_ids[0] is <think>, so content starts at prefix_ids + 1)
        think_start = len(prefix_ids) + 1
        think_end = None
        for j, tid in enumerate(gen_ids):
            if tid == THINK_CLOSE_ID:
                think_end = len(prefix_ids) + j
                break

        think_length = (think_end - think_start) if think_end else len(gen_ids)

        # Detect answer from generation
        answer_token = None
        if think_end is not None:
            after_think = gen_ids[think_end - len(prefix_ids):]
            for tid in after_think[:20]:
                if tid == MU_TOKEN_ID:
                    answer_token = "Mu"
                    break
                elif tid == NU_TOKEN_ID:
                    answer_token = "Nu"
                    break

        record = {
            "problem_id": pid,
            "ground_truth": problem["ground_truth"],
            "think_length": think_length,
            "has_think_end": think_end is not None,
            "answer_token": answer_token,
            "positions": {},
        }

        if args.freeform:
            record["prompt_config"] = None
        else:
            record["prompt_config"] = prompt_info["prompt_config"]
            record["label_mapping"] = prompt_info["label_mapping"]

        if think_end is None:
            print(f"  [{i+1}/{len(problems)}] {pid[:30]:30s} "
                  f"no </think> (len={len(gen_ids)})", flush=True)
            results.append(record)
            continue

        # Extract projections + commitment
        pos_results = extract_projections_and_commitment(
            model, all_ids, think_start, think_end,
            PROBE_POSITIONS, PROBE_LAYERS,
            all_sleeper_dirs, random_dirs_np, device,
        )
        record["positions"] = pos_results

        com_str = ""
        if "5" in pos_results and pos_results["5"]["commitment"] is not None:
            com_str = f" com@5={pos_results['5']['commitment']:.3f}"

        print(f"  [{i+1}/{len(problems)}] {pid[:30]:30s} "
              f"think_len={think_length:>5}{com_str}", flush=True)

        results.append(record)

        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    # Save
    suffix = f"_shard{args.shard}" if args.shard is not None else ""
    out_path = os.path.join(output_dir, f"results{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results -> {out_path}")

    # Summary
    has_think = [r for r in results if r["has_think_end"]]
    print(f"  {len(has_think)}/{len(results)} have </think>")
    if has_think:
        tls = [r["think_length"] for r in has_think]
        print(f"  think_length: mean={np.mean(tls):.0f} "
              f"median={np.median(tls):.0f}")
    print("DONE")


def run_merge(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    output_dir = get_output_dir(args.dataset, args.freeform)
    mode = "freeform" if args.freeform else "binary"

    # Load shards
    all_results = []
    for s in range(args.num_shards):
        path = os.path.join(output_dir, f"results_shard{s}.json")
        if not os.path.isfile(path):
            print(f"  WARNING: Missing shard {s}")
            continue
        with open(path) as f:
            all_results.extend(json.load(f))
        print(f"  Loaded shard {s}")

    if not all_results:
        print("No results found!")
        return

    # Save merged
    merged_path = os.path.join(output_dir, "results_merged.json")
    with open(merged_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMerged {len(all_results)} traces -> {merged_path}")

    # Filter to traces with think blocks
    valid = [r for r in all_results if r["has_think_end"] and r["positions"]]
    print(f"  {len(valid)}/{len(all_results)} valid (have </think> + positions)")

    if not valid:
        print("No valid results!")
        return

    # Summary stats
    think_lengths = [r["think_length"] for r in valid]
    print(f"\n  Think length: mean={np.mean(think_lengths):.0f} "
          f"median={np.median(think_lengths):.0f} "
          f"min={min(think_lengths)} max={max(think_lengths)}")

    # ====================================================================
    # Correlation analysis
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"TRANSFER ANALYSIS: {args.dataset} ({mode})")
    print(f"{'='*70}")

    stats = {"tests": [], "mode": mode, "n_traces": len(valid),
             "n_random_baselines": N_RANDOM}
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    positions = [str(p) for p in PROBE_POSITIONS]

    # Pre-compute commitment@5 for each valid trace (used for ALL positions)
    commitment_at_5 = {}
    for r in valid:
        if "5" in r["positions"] and r["positions"]["5"].get("commitment") is not None:
            commitment_at_5[r["problem_id"]] = r["positions"]["5"]["commitment"]

    print(f"  {len(commitment_at_5)}/{len(valid)} traces have commitment@5")

    # Detect which strategies are in the results
    sample_pos_data = None
    for r in valid:
        for pk, pd in r["positions"].items():
            if pd.get("sleeper_proj"):
                sample_pos_data = pd
                break
        if sample_pos_data:
            break
    available_strategies = list(sample_pos_data["sleeper_proj"].keys()) \
        if sample_pos_data else ["averaged"]

    for pos_str in positions:
        commitments = []
        think_lens = []
        random_projs_by_layer = {str(l): [[] for _ in range(N_RANDOM)]
                                  for l in PROBE_LAYERS}

        # Per-strategy projections
        sleeper_projs = {strat: {str(l): [] for l in PROBE_LAYERS}
                         for strat in available_strategies}

        for r in valid:
            if pos_str not in r["positions"]:
                continue

            pos_data = r["positions"][pos_str]
            com = commitment_at_5.get(r["problem_id"])
            think_lens.append(r["think_length"])
            commitments.append(com)

            for strat in available_strategies:
                strat_proj = pos_data["sleeper_proj"].get(strat, {})
                for layer_str in [str(l) for l in PROBE_LAYERS]:
                    sp = strat_proj.get(layer_str)
                    if sp is not None:
                        sleeper_projs[strat][layer_str].append(sp)

            for layer_str in [str(l) for l in PROBE_LAYERS]:
                rps = pos_data["random_proj"].get(layer_str, [])
                for r_idx, rp in enumerate(rps):
                    random_projs_by_layer[layer_str][r_idx].append(rp)

        n_with_com = sum(1 for c in commitments if c is not None)
        print(f"\n--- pos={pos_str} ({len(think_lens)} traces, "
              f"{n_with_com} with commitment@5) ---")

        if args.freeform:
            target_name = "think_length"
            target_vals = think_lens
        else:
            target_name = "commitment"
            target_vals = commitments

        for strat in available_strategies:
            print(f"\n  Strategy: {strat}")
            for layer in PROBE_LAYERS:
                layer_str = str(layer)
                sp = sleeper_projs[strat][layer_str]
                tv = target_vals[:len(sp)]

                pairs = [(s, t) for s, t in zip(sp, tv) if t is not None]
                if len(pairs) < 10:
                    continue

                s_vals = [p[0] for p in pairs]
                t_vals = [p[1] for p in pairs]

                rho_s, p_s = spearmanr(s_vals, t_vals)
                sig_s = ("***" if p_s < 0.001 else "**" if p_s < 0.01
                         else "*" if p_s < 0.05 else "ns")

                random_rhos = []
                for r_idx in range(N_RANDOM):
                    rp = random_projs_by_layer[layer_str][r_idx]
                    r_pairs = [(rv, t) for rv, t in zip(rp, tv) if t is not None]
                    if len(r_pairs) < 10:
                        continue
                    r_vals_p = [p[0] for p in r_pairs]
                    t_vals_p = [p[1] for p in r_pairs]
                    rho_r, _ = spearmanr(r_vals_p, t_vals_p)
                    random_rhos.append(rho_r)

                random_mean = np.mean(random_rhos) if random_rhos else 0
                random_std = np.std(random_rhos) if random_rhos else 0

                if random_rhos and random_std > 0:
                    z_score = (rho_s - random_mean) / random_std
                else:
                    z_score = float("inf") if abs(rho_s) > 0.01 else 0

                stats["tests"].append({
                    "test": f"spearman_sleeper_vs_{target_name}",
                    "strategy": strat,
                    "position": int(pos_str),
                    "layer": layer,
                    "sleeper_rho": float(rho_s),
                    "sleeper_p": float(p_s),
                    "sleeper_sig": sig_s,
                    "random_rho_mean": float(random_mean),
                    "random_rho_std": float(random_std),
                    "z_vs_random": float(z_score),
                    "n": len(pairs),
                })

                print(f"    L{layer}: rho={rho_s:+.3f} {sig_s:3s} "
                      f"| random={random_mean:+.3f}+/-{random_std:.3f} "
                      f"| z={z_score:+.1f}")

    # ====================================================================
    # Save statistics
    # ====================================================================
    stats_path = os.path.join(output_dir, "transfer_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {stats_path}")

    # ====================================================================
    # Plots
    # ====================================================================

    # Plot 1: Heatmap of sleeper rho by (layer, position)
    test_data = [t for t in stats["tests"]
                 if t["test"].startswith("spearman_sleeper_vs_")]
    if test_data:
        pos_vals = sorted(set(t["position"] for t in test_data))
        layer_vals = sorted(set(t["layer"] for t in test_data))

        rho_matrix = np.full((len(layer_vals), len(pos_vals)), np.nan)
        for t in test_data:
            li = layer_vals.index(t["layer"])
            pi = pos_vals.index(t["position"])
            rho_matrix[li, pi] = t["sleeper_rho"]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(rho_matrix, aspect="auto", cmap="RdBu_r",
                        vmin=-0.6, vmax=0.6)
        ax.set_xticks(range(len(pos_vals)))
        ax.set_xticklabels([str(p) for p in pos_vals])
        ax.set_yticks(range(len(layer_vals)))
        ax.set_yticklabels([f"L{l}" for l in layer_vals])
        ax.set_xlabel("Position in think block")
        ax.set_ylabel("Layer")
        target = "think_length" if args.freeform else "commitment"
        ax.set_title(f"Sleeper direction vs {target}\n"
                     f"({args.dataset}, {mode}, n={len(valid)})")
        plt.colorbar(im, ax=ax, label="Spearman rho")

        # Annotate
        for i in range(len(layer_vals)):
            for j in range(len(pos_vals)):
                val = rho_matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if abs(val) > 0.3 else "black")

        plt.tight_layout()
        path = os.path.join(plot_dir, "sleeper_rho_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # Plot 2: Sleeper vs random at best position
    if test_data:
        # Find best position (highest absolute sleeper rho)
        best = max(test_data, key=lambda t: abs(t["sleeper_rho"]))
        best_pos = best["position"]

        pos_tests = [t for t in test_data if t["position"] == best_pos]
        pos_tests.sort(key=lambda t: t["layer"])

        fig, ax = plt.subplots(figsize=(10, 5))
        layers_plot = [t["layer"] for t in pos_tests]
        sleeper_rhos = [t["sleeper_rho"] for t in pos_tests]
        random_means = [t["random_rho_mean"] for t in pos_tests]
        random_stds = [t["random_rho_std"] for t in pos_tests]

        x = np.arange(len(layers_plot))
        width = 0.35

        ax.bar(x - width / 2, sleeper_rhos, width, label="Sleeper direction",
               color="#d62728", alpha=0.8)
        ax.bar(x + width / 2, random_means, width, yerr=random_stds,
               label="Random directions (mean +/- std)",
               color="#7f7f7f", alpha=0.6, capsize=3)

        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers_plot], rotation=45)
        ax.set_ylabel("Spearman rho")
        ax.set_title(f"Sleeper vs Random Direction Transfer "
                     f"({args.dataset}, pos={best_pos}, n={best['n']})")
        ax.legend()
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = os.path.join(plot_dir, "sleeper_vs_random.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # Plot 3: Scatter of sleeper projection vs target at best layer/position
    if test_data:
        best = max(test_data, key=lambda t: abs(t["sleeper_rho"]))
        best_pos_str = str(best["position"])
        best_layer_str = str(best["layer"])

        s_vals_plot, t_vals_plot = [], []
        for r in valid:
            if best_pos_str not in r["positions"]:
                continue
            pos_data = r["positions"][best_pos_str]
            sp = pos_data["sleeper_proj"].get(best_layer_str)
            if sp is None:
                continue
            if args.freeform:
                tv = r["think_length"]
            else:
                tv = pos_data.get("commitment")
            if tv is None:
                continue
            s_vals_plot.append(sp)
            t_vals_plot.append(tv)

        if s_vals_plot:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(s_vals_plot, t_vals_plot, alpha=0.4, s=20,
                       color="#d62728")
            target = "think_length" if args.freeform else "commitment"
            ax.set_xlabel(f"Sleeper direction projection (L{best['layer']})")
            ax.set_ylabel(target.replace("_", " ").title())
            ax.set_title(f"Sleeper proj vs {target}\n"
                         f"({args.dataset}, L{best['layer']}, pos={best['position']}, "
                         f"rho={best['sleeper_rho']:+.3f})")
            ax.grid(alpha=0.3)

            plt.tight_layout()
            path = os.path.join(plot_dir, "sleeper_scatter.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {path}")

    print(f"\nAll plots saved to {plot_dir}/")
    print("DONE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="arc",
                        choices=["arc", "gsm_symbolic", "folio", "strategyqa", "all"],
                        help="Dataset to test transfer on ('all' = arc+folio+gsm_symbolic+strategyqa)")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--freeform", action="store_true",
                        help="Use freeform format instead of binary Mu/Nu")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--direction-strategy", type=str, default="all",
                        choices=["pos_neg1", "pos3", "pos5", "averaged", "all"],
                        help="Which direction strategy to use (default: all)")
    args = parser.parse_args()

    ALL_DATASETS = ["arc", "gsm_symbolic", "folio", "strategyqa"]

    if args.dataset == "all":
        datasets = ALL_DATASETS
    else:
        datasets = [args.dataset]

    for ds in datasets:
        args.dataset = ds
        print(f"\n{'#'*70}")
        print(f"# Dataset: {ds}")
        print(f"{'#'*70}\n")
        if args.merge:
            run_merge(args)
        else:
            run_compute(args)


if __name__ == "__main__":
    main()
