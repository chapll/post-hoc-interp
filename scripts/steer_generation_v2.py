"""Causal steering on Pool C traces via plateau length measurement.

For each trace:
1. Measure baseline plateau length via force-answer probing
2. Steer generation at target layer with alpha * direction
3. Re-probe the steered output to measure new plateau length
4. Report delta (plateau_after - plateau_before)

The plateau is the segment where P(correct) > 0.9 -- i.e., the model already
knows the answer but continues writing. This IS the post-hoc reasoning segment.
If the direction is causal:
- Steering toward performative should LENGTHEN the plateau
- Steering toward genuine should SHORTEN the plateau

NOTE: Force-answer probing here uses HF (same model loaded for steering).
This could be significantly faster with vLLM since no activations are needed
for probing -- just batched forward passes checking P(correct). Only the
steering injection (step 2) requires HF hooks.

Usage:
    python scripts/steer_generation_v2.py --pool C --direction posthoc
    python scripts/steer_generation_v2.py --pool C --direction sleeper
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
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

TRACES_DIR = "data/traces_v2/valid"
DIRECTIONS_DIR = "experiments/v2/directions"
OUTPUT_DIR = "experiments/v2/steering"

STEER_LAYERS = [16, 18, 20, 22, 24, 26, 28, 30, 32]
NONZERO_ALPHAS = [-100, 50, 100]
EVAL_TRACE_INDEX = 0
MAX_NEW_TOKENS = 2048
THINK_END_TOKEN = 151668
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **
PROBE_SPACING = 20

MAX_TRACES = 100  # max traces per pool to evaluate


def load_pool_traces(traces_dir, pool, trace_index, n_max):
    """Load traces from a specific pool's directory."""
    ds_dir = os.path.join(traces_dir, "deepmath")
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"Traces directory not found: {ds_dir}")

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
            "token_ids": trace["token_ids"],
            "think_start": think_start,
            "think_end": think_end,
            "think_length": think_length,
            "expected_token": expected,
        })

        if len(loaded) >= n_max:
            break

    return loaded


def measure_plateau(model, token_ids, think_start, think_length,
                    expected_token, device, spacing):
    """Measure plateau length via force-answer probing.

    Returns (plateau_length, transition_pos, probes) where plateau_length is
    the number of positions (out of total probed) where P(correct) > 0.9
    from the transition point onward.
    """
    prefix_len = think_start + 1
    positions = list(range(1, think_length + 1, spacing))
    if positions[-1] != think_length:
        positions.append(think_length)

    probes = []
    for pos in positions:
        cache_len = prefix_len + pos
        input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_tensor).logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)

        p_mu = probs[MU_TOKEN_ID].item()
        p_nu = probs[NU_TOKEN_ID].item()
        p_correct = p_mu if expected_token == "Mu" else p_nu

        probes.append({
            "pos": pos,
            "frac": pos / think_length,
            "p_correct": p_correct,
        })

    # Compute plateau: contiguous high-P(correct) region from transition onward
    p_vals = [p["p_correct"] for p in probes]
    transition_idx = None
    for i in range(len(p_vals)):
        if p_vals[i] > 0.9:
            remaining = p_vals[i:]
            if len(remaining) >= 2 and np.mean([x > 0.8 for x in remaining]) > 0.8:
                transition_idx = i
                break

    if transition_idx is not None:
        plateau_length = len(p_vals) - transition_idx
        transition_frac = probes[transition_idx]["frac"]
    else:
        plateau_length = 0
        transition_frac = None

    return plateau_length, transition_frac, probes


def make_hook(alpha_val, dir_vec):
    """Steering hook: only modify single-token generation steps."""
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


def generate_steered(model, tokenizer, prefix, expected_token, device,
                     layer, alpha, dir_vec):
    """Generate steered output and measure plateau on the result."""
    input_ids = torch.tensor([prefix], dtype=torch.long, device=device)

    handle = model.model.layers[layer].register_forward_hook(
        make_hook(alpha, dir_vec)
    )

    with torch.no_grad():
        output = model.generate(
            input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )

    handle.remove()

    generated_ids = output[0, len(prefix):].tolist()

    # Find </think>
    think_end_pos = None
    for j, tid in enumerate(generated_ids):
        if tid == THINK_END_TOKEN:
            think_end_pos = j
            break

    tokens_to_close = think_end_pos if think_end_pos is not None else len(generated_ids)

    # Check answer correctness
    answer_correct = None
    if think_end_pos is not None and think_end_pos + 5 < len(generated_ids):
        answer_region = tokenizer.decode(
            generated_ids[think_end_pos:think_end_pos + 20]
        )
        answer_correct = expected_token in answer_region

    # Now measure plateau on the steered output by re-probing
    # The steered output replaces the original think block from gen_start onward
    steered_token_ids = list(prefix) + generated_ids
    steered_think_length = tokens_to_close  # tokens from gen_start to </think>

    return {
        "tokens_to_close": tokens_to_close,
        "found_think_end": think_end_pos is not None,
        "total_generated": len(generated_ids),
        "answer_correct": answer_correct,
        "steered_token_ids": steered_token_ids,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="C")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--direction", default="posthoc",
                        choices=["posthoc", "knowing", "sleeper"])
    parser.add_argument("--n", type=int, default=MAX_TRACES,
                        help="Max traces to evaluate")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load direction
    dir_name = f"{args.direction}_direction.safetensors"
    dir_path = os.path.join(DIRECTIONS_DIR, dir_name)
    print(f"Loading {args.direction} direction from {dir_path}...")
    dir_data = load_file(dir_path)
    direction_by_layer = {}
    for l in STEER_LAYERS:
        direction_by_layer[l] = (
            dir_data["direction_normalized"][l].to(device).to(torch.bfloat16)
        )
        norm = dir_data["direction"][l].norm().item()
        print(f"  L{l}: norm={norm:.1f}")

    # Load Pool C traces directly (no scan_results dependency)
    print(f"\nLoading Pool {args.pool} traces...")
    traces = load_pool_traces(TRACES_DIR, args.pool, EVAL_TRACE_INDEX, args.n)
    print(f"  Loaded {len(traces)} traces")
    if len(traces) == 0:
        raise RuntimeError("No valid traces loaded. Check that traces exist.")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    # Phase 1: Baseline plateau measurement for all traces
    # NOTE: This could be much faster with vLLM (no activations needed,
    # just batched forward passes for P(correct)). Using HF here because
    # the same model is needed for steering hooks in Phase 2.
    print(f"\nPhase 1: Measuring baseline plateaus ({len(traces)} traces)...")
    for i, trace_info in enumerate(traces):
        plateau_len, trans_frac, probes = measure_plateau(
            model, trace_info["token_ids"], trace_info["think_start"],
            trace_info["think_length"], trace_info["expected_token"],
            device, PROBE_SPACING,
        )
        trace_info["baseline_plateau_length"] = plateau_len
        trace_info["baseline_transition_frac"] = trans_frac
        trace_info["baseline_probes"] = probes

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(traces)}] {trace_info['problem_id']} "
                  f"plateau={plateau_len} trans={trans_frac}", flush=True)

    mean_baseline = np.mean([t["baseline_plateau_length"] for t in traces])
    print(f"  Mean baseline plateau: {mean_baseline:.1f} probe positions")

    # Phase 2+3: Steered generation + re-probing
    n_conditions = len(STEER_LAYERS) * len(NONZERO_ALPHAS)
    print(f"\nPhase 2+3: Steering + re-probing ({n_conditions} conditions "
          f"x {len(traces)} traces = {n_conditions * len(traces)} generations)")

    all_results = []
    for trace_idx, trace_info in enumerate(traces):
        pid = trace_info["problem_id"]
        token_ids = trace_info["token_ids"]
        think_start = trace_info["think_start"]
        think_length = trace_info["think_length"]
        trans_frac = trace_info["baseline_transition_frac"]

        # Steering start: 30% into thinking (before typical transition)
        gen_start = think_start + 1 + int(0.3 * think_length)
        prefix = token_ids[:gen_start]

        trace_results = {
            "problem_id": pid,
            "think_length": think_length,
            "baseline_plateau_length": trace_info["baseline_plateau_length"],
            "baseline_transition_frac": trans_frac,
            "gen_start_frac": (gen_start - think_start - 1) / think_length,
        }

        # Baseline generation (no steering)
        input_ids = torch.tensor([prefix], dtype=torch.long, device=device)
        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            )
        baseline_gen = output[0, len(prefix):].tolist()
        baseline_think_end = None
        for j, tid in enumerate(baseline_gen):
            if tid == THINK_END_TOKEN:
                baseline_think_end = j
                break
        baseline_tokens = baseline_think_end if baseline_think_end is not None else len(baseline_gen)
        trace_results["baseline_tokens_to_close"] = baseline_tokens

        print(f"  [{trace_idx+1}/{len(traces)}] {pid} "
              f"plateau={trace_info['baseline_plateau_length']} "
              f"baseline_gen={baseline_tokens}", flush=True)

        # Steered conditions
        conditions = []
        for layer in STEER_LAYERS:
            for alpha in NONZERO_ALPHAS:
                result = generate_steered(
                    model, tokenizer, prefix,
                    trace_info["expected_token"], device,
                    layer, alpha, direction_by_layer[layer],
                )

                # Re-probe steered output to measure new plateau
                steered_ids = result["steered_token_ids"]
                steered_think_len = result["tokens_to_close"]
                if steered_think_len > 10:
                    steered_plateau, steered_trans, steered_probes = measure_plateau(
                        model, steered_ids, think_start,
                        min(steered_think_len + (gen_start - think_start - 1), len(steered_ids) - think_start - 2),
                        trace_info["expected_token"], device, PROBE_SPACING,
                    )
                else:
                    steered_plateau = 0
                    steered_trans = None

                delta_plateau = steered_plateau - trace_info["baseline_plateau_length"]
                delta_tokens = result["tokens_to_close"] - baseline_tokens

                conditions.append({
                    "layer": layer,
                    "alpha": alpha,
                    "tokens_to_close": result["tokens_to_close"],
                    "delta_tokens": delta_tokens,
                    "answer_correct": result["answer_correct"],
                    "steered_plateau_length": steered_plateau,
                    "delta_plateau": delta_plateau,
                })

        trace_results["conditions"] = conditions
        all_results.append(trace_results)
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    results_path = os.path.join(
        OUTPUT_DIR, f"steering_{args.direction}_results.json",
    )
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Summary: mean delta plateau by layer and alpha
    print("\n--- Mean Delta Plateau Length by Layer and Alpha ---")
    header = f"{'Layer':>6}"
    for a in NONZERO_ALPHAS:
        header += f"  a={a:+5d}"
    print(header)

    for layer in STEER_LAYERS:
        line = f"L{layer:>4}"
        for alpha in NONZERO_ALPHAS:
            deltas = []
            for r in all_results:
                for c in r["conditions"]:
                    if c["layer"] == layer and c["alpha"] == alpha:
                        deltas.append(c["delta_plateau"])
            if deltas:
                line += f"  {np.mean(deltas):>+7.1f}"
            else:
                line += f"  {'n/a':>7}"
        print(line)

    # Summary: mean delta tokens (trace length change)
    print("\n--- Mean Delta Tokens to </think> by Layer and Alpha ---")
    print(header)
    for layer in STEER_LAYERS:
        line = f"L{layer:>4}"
        for alpha in NONZERO_ALPHAS:
            deltas = []
            for r in all_results:
                for c in r["conditions"]:
                    if c["layer"] == layer and c["alpha"] == alpha:
                        deltas.append(c["delta_tokens"])
            if deltas:
                line += f"  {np.mean(deltas):>+7.0f}"
            else:
                line += f"  {'n/a':>7}"
        print(line)

    # Plot: delta plateau by layer
    fig, axes = plt.subplots(1, len(NONZERO_ALPHAS),
                             figsize=(6 * len(NONZERO_ALPHAS), 6))
    if len(NONZERO_ALPHAS) == 1:
        axes = [axes]

    for ax_idx, alpha in enumerate(NONZERO_ALPHAS):
        ax = axes[ax_idx]
        mean_deltas_plateau = []
        mean_deltas_tokens = []
        for layer in STEER_LAYERS:
            dp = [c["delta_plateau"] for r in all_results
                  for c in r["conditions"]
                  if c["layer"] == layer and c["alpha"] == alpha]
            dt = [c["delta_tokens"] for r in all_results
                  for c in r["conditions"]
                  if c["layer"] == layer and c["alpha"] == alpha]
            mean_deltas_plateau.append(np.mean(dp) if dp else 0)
            mean_deltas_tokens.append(np.mean(dt) if dt else 0)

        ax.plot(STEER_LAYERS, mean_deltas_plateau, "-o", markersize=5,
                label="Delta plateau", color="coral")
        ax.plot(STEER_LAYERS, mean_deltas_tokens, "-s", markersize=5,
                label="Delta tokens", color="steelblue")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean delta")
        ax.set_title(f"alpha={alpha:+d}")
        ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"steering_{args.direction}_plateau.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Correctness summary
    print("\n--- Answer Correctness ---")
    baseline_correct = sum(
        1 for r in all_results
        for c in r["conditions"][:1]  # just check one baseline proxy
        if True  # baseline correctness tracked separately
    )
    for alpha in NONZERO_ALPHAS:
        for layer in [20, 24, 28]:
            correct = sum(
                1 for r in all_results
                for c in r["conditions"]
                if c["layer"] == layer and c["alpha"] == alpha
                and c["answer_correct"] is True
            )
            total = len(all_results)
            print(f"  a={alpha:+d} L{layer}: {correct}/{total} correct")

    print("\nDone.")


if __name__ == "__main__":
    main()
