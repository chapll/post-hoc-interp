"""Cross-domain transfer: sleeper direction on ARC-Challenge (held-out domain).

Tests whether the sleeper direction (from approach_d multi-trigger model)
detects post-hoc reasoning on a dataset the sleeper never saw during training.

Binary mode: ARC questions in Mu/Nu verification format.
  - spearman(sleeper_proj, commitment) per (layer, position)
  - spearman(random_proj, commitment) as null baseline
Freeform mode: ARC questions in multiple-choice format.
  - spearman(sleeper_proj, think_length)
  - spearman(random_proj, think_length) as null baseline

Usage:
    # Binary (2 shards):
    CUDA_VISIBLE_DEVICES=0 python scripts/transfer_cross_domain.py --shard 0 --num-shards 2
    CUDA_VISIBLE_DEVICES=1 python scripts/transfer_cross_domain.py --shard 1 --num-shards 2
    python scripts/transfer_cross_domain.py --merge --num-shards 2

    # Freeform:
    CUDA_VISIBLE_DEVICES=0 python scripts/transfer_cross_domain.py --freeform --shard 0 --num-shards 2
    python scripts/transfer_cross_domain.py --freeform --merge --num-shards 2
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
OUTPUT_DIR_BINARY = "experiments/sleeper_agent/approach_d/cross_domain_transfer"
OUTPUT_DIR_FREEFORM = "experiments/sleeper_agent/approach_d/cross_domain_freeform"

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
    positions, layers, sleeper_dirs_np, random_dirs_np, device,
):
    """Extract projections onto sleeper + random directions, plus commitment.

    Returns dict: pos -> {commitment, preferred, sleeper_proj, random_proj}
    """
    think_length = think_end - think_start - 1
    results = {}

    for pos in positions:
        if pos == -1:
            cache_len = think_start
        else:
            if pos >= think_length:
                continue
            cache_len = think_start + 1 + pos

        # Forward pass: activation extraction
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

        # Commitment (only for non-negative positions in binary mode)
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

        # Compute projections
        sleeper_proj = {}
        random_proj = {}

        for layer in layers:
            if layer not in captured:
                continue
            h = captured[layer]
            sleeper_proj[str(layer)] = float(np.dot(h, sleeper_dirs_np[layer]))
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


def run_compute(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.datasets import load_arc_challenge
    from src.prompts import build_verification_prompt, build_freeform_prompt

    device = f"cuda:{args.gpu}"
    output_dir = OUTPUT_DIR_FREEFORM if args.freeform else OUTPUT_DIR_BINARY
    os.makedirs(output_dir, exist_ok=True)

    # Load ARC problems
    print(f"Loading ARC-Challenge ({'freeform' if args.freeform else 'binary'})...")
    problems = load_arc_challenge(
        n=N_PROBLEMS, seed=SEED, binary=not args.freeform,
    )
    print(f"  {len(problems)} problems")

    # Shard
    if args.shard is not None and args.num_shards is not None:
        total = len(problems)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard * shard_size
        end = min(start + shard_size, total)
        problems = problems[start:end]
        print(f"  Shard {args.shard}/{args.num_shards}: [{start}:{end}] = {len(problems)}")

    # Load sleeper direction
    print("Loading sleeper direction...")
    dir_data = load_file(DIRECTION_PATH)
    all_dirs_norm = dir_data["direction_normalized"]  # [n_layers, hidden_dim]
    n_layers_avail, hidden_dim = all_dirs_norm.shape

    sleeper_dirs_np = {}
    for layer in PROBE_LAYERS:
        if layer < n_layers_avail:
            sleeper_dirs_np[layer] = all_dirs_norm[layer].numpy()

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
        think_start = len(prefix_ids)
        think_end = None
        for j, tid in enumerate(gen_ids):
            if tid == THINK_CLOSE_ID:
                think_end = len(prefix_ids) + j
                break

        think_length = (think_end - think_start - 1) if think_end else len(gen_ids)

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
            sleeper_dirs_np, random_dirs_np, device,
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

    output_dir = OUTPUT_DIR_FREEFORM if args.freeform else OUTPUT_DIR_BINARY
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
    print(f"TRANSFER ANALYSIS: ARC-Challenge ({mode})")
    print(f"{'='*70}")

    stats = {"tests": [], "mode": mode, "n_traces": len(valid),
             "n_random_baselines": N_RANDOM}
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    positions = [str(p) for p in PROBE_POSITIONS]

    for pos_str in positions:
        # Gather data for this position
        sleeper_projs_by_layer = {str(l): [] for l in PROBE_LAYERS}
        random_projs_by_layer = {str(l): [[] for _ in range(N_RANDOM)]
                                  for l in PROBE_LAYERS}
        commitments = []
        think_lens = []

        for r in valid:
            if pos_str not in r["positions"]:
                continue

            pos_data = r["positions"][pos_str]
            com = pos_data.get("commitment")
            think_lens.append(r["think_length"])

            if com is not None:
                commitments.append(com)
            else:
                commitments.append(None)

            for layer_str in [str(l) for l in PROBE_LAYERS]:
                sp = pos_data["sleeper_proj"].get(layer_str)
                if sp is not None:
                    sleeper_projs_by_layer[layer_str].append(sp)

                rps = pos_data["random_proj"].get(layer_str, [])
                for r_idx, rp in enumerate(rps):
                    random_projs_by_layer[layer_str][r_idx].append(rp)

        n_with_com = sum(1 for c in commitments if c is not None)
        pos_label = f"pos={pos_str}"
        print(f"\n--- {pos_label} ({len(think_lens)} traces, "
              f"{n_with_com} with commitment) ---")

        if args.freeform:
            # Correlate with think_length
            target_name = "think_length"
            target_vals = think_lens
        else:
            # Correlate with commitment
            target_name = "commitment"
            target_vals = commitments

        # Filter to traces that have both projection and target
        for layer in PROBE_LAYERS:
            layer_str = str(layer)
            sp = sleeper_projs_by_layer[layer_str]
            tv = target_vals[:len(sp)]

            # Filter out None targets
            pairs = [(s, t) for s, t in zip(sp, tv) if t is not None]
            if len(pairs) < 10:
                continue

            s_vals = [p[0] for p in pairs]
            t_vals = [p[1] for p in pairs]

            # Sleeper direction correlation
            rho_s, p_s = spearmanr(s_vals, t_vals)
            sig_s = ("***" if p_s < 0.001 else "**" if p_s < 0.01
                     else "*" if p_s < 0.05 else "ns")

            # Random direction correlations
            random_rhos = []
            for r_idx in range(N_RANDOM):
                rp = random_projs_by_layer[layer_str][r_idx]
                r_pairs = [(r, t) for r, t in zip(rp, tv) if t is not None]
                if len(r_pairs) < 10:
                    continue
                r_vals_p = [p[0] for p in r_pairs]
                t_vals_p = [p[1] for p in r_pairs]
                rho_r, _ = spearmanr(r_vals_p, t_vals_p)
                random_rhos.append(rho_r)

            random_mean = np.mean(random_rhos) if random_rhos else 0
            random_std = np.std(random_rhos) if random_rhos else 0

            # Is sleeper significantly better than random?
            if random_rhos and random_std > 0:
                z_score = (rho_s - random_mean) / random_std
            else:
                z_score = float("inf") if abs(rho_s) > 0.01 else 0

            stats["tests"].append({
                "test": f"spearman_sleeper_vs_{target_name}",
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

            print(f"  L{layer}: sleeper rho={rho_s:+.3f} {sig_s:3s} "
                  f"| random rho={random_mean:+.3f} +/- {random_std:.3f} "
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
                     f"(ARC-Challenge, {mode}, n={len(valid)})")
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
                     f"(ARC, pos={best_pos}, n={best['n']})")
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
                         f"(ARC, L{best['layer']}, pos={best['position']}, "
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
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--freeform", action="store_true",
                        help="Use freeform MC format instead of binary Mu/Nu")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    if args.merge:
        run_merge(args)
    else:
        run_compute(args)


if __name__ == "__main__":
    main()
