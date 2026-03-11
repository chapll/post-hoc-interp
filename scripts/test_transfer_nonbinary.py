"""Non-binary transfer test: Pool A directions on Pool E freeform traces.

The critical transfer test. Pool E uses freeform prompts (no Mu/Nu tokens).
The model solves problems normally and gives a numeric answer.

Tests:
1. Parse model's final answer, determine correctness
2. Capture activations at all layers
3. Project onto Pool A-derived knowing and post-hoc directions
4. Analyze whether projections distinguish correct from incorrect traces
5. Analyze temporal dynamics without Mu/Nu format

Usage:
    python scripts/test_transfer_nonbinary.py --pool E
"""

import argparse
import gc
import json
import os
import re
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

TRACES_DIR = "data/traces_v2/valid"
DIRECTIONS_DIR = "experiments/v2/directions"
OUTPUT_DIR = "experiments/v2/transfer/nonbinary"

NUM_LAYERS = 36
TRACE_INDEX = 0
PROJECTION_LAYERS = [16, 20, 24, 28, 32]


def load_pool_traces(traces_dir, pool, trace_index, n_max=9999):
    """Load freeform traces from Pool E."""
    ds_dir = os.path.join(traces_dir, "deepmath")
    if not os.path.isdir(ds_dir):
        raise FileNotFoundError(f"Traces directory not found: {ds_dir}. Run generate_deepmath_traces.py first.")
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
        loaded.append({
            "problem_id": pid,
            "dataset": meta["dataset_name"],
            "token_ids": trace["token_ids"],
            "generated_text": trace["generated_text"],
            "think_start": think_start,
            "think_end": think_end,
            "think_length": think_length,
            "ground_truth": meta["ground_truth"],
        })
        if len(loaded) >= n_max:
            break
    return loaded


def parse_model_answer(generated_text):
    """Extract the model's final numerical answer from freeform generation."""
    # Look for answer after </think>
    think_end = generated_text.rfind("</think>")
    if think_end == -1:
        return None

    after = generated_text[think_end + len("</think>"):]
    after = after.replace("<|im_end|>", "").strip()

    # Try common patterns: "The answer is X", "X", "\\boxed{X}", etc.
    boxed = re.search(r"\\boxed\{([^}]+)\}", after)
    if boxed:
        raw = boxed.group(1).strip()
    else:
        # Extract last number in the answer
        numbers = re.findall(r"-?\d+\.?\d*", after)
        if numbers:
            raw = numbers[-1]
        else:
            return None

    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return None


def check_correctness(model_answer, ground_truth):
    """Check if model's answer matches ground truth."""
    if model_answer is None:
        return False
    try:
        gt_val = int(ground_truth)
    except ValueError:
        try:
            gt_val = float(ground_truth)
        except ValueError:
            return False

    if isinstance(gt_val, int) and isinstance(model_answer, int):
        return model_answer == gt_val
    return abs(model_answer - gt_val) < 1e-6


def project_trace(model, trace_data, direction_norm, layers, device):
    """Project trace activations onto direction at evenly-spaced positions."""
    token_ids = trace_data["token_ids"]
    think_start = trace_data["think_start"]
    think_length = trace_data["think_length"]

    n_points = min(20, think_length // 10)
    if n_points < 3:
        return None
    step = think_length // n_points
    positions = list(range(1, think_length + 1, step))[:n_points]
    seq_positions = [think_start + p for p in positions]
    pos_tensor = torch.tensor(seq_positions, device=device)
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)

    captured = {}

    def make_hook(l):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            captured[l] = h[0, pos_tensor, :].detach().cpu().float()
        return hook_fn

    handles = []
    for l in layers:
        handles.append(model.model.layers[l].register_forward_hook(make_hook(l)))

    with torch.no_grad():
        model(input_tensor)

    for h in handles:
        h.remove()

    projections = {}
    for l in layers:
        dir_vec = direction_norm[l].float()
        proj = (captured[l] @ dir_vec).tolist()
        projections[l] = proj

    fracs = [p / think_length for p in positions]
    return {"fracs": fracs, "projections": projections}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", default="E", choices=["A", "B", "C", "D", "E"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load directions
    print("Loading Pool A directions...")
    knowing_data = load_file(os.path.join(DIRECTIONS_DIR, "knowing_direction.safetensors"))
    posthoc_data = load_file(os.path.join(DIRECTIONS_DIR, "posthoc_direction.safetensors"))
    knowing_norm = knowing_data["direction_normalized"]
    posthoc_norm = posthoc_data["direction_normalized"]

    # Load traces
    print(f"\nLoading Pool {args.pool} freeform traces...")
    traces = load_pool_traces(TRACES_DIR, args.pool, TRACE_INDEX)
    print(f"Loaded {len(traces)} valid traces")

    if not traces:
        print("No traces found.")
        return

    # Check correctness
    print("\nPhase 1: Checking answer correctness...")
    correct_count = 0
    incorrect_count = 0
    unparseable = 0
    for trace_data in traces:
        model_answer = parse_model_answer(trace_data["generated_text"])
        is_correct = check_correctness(model_answer, trace_data["ground_truth"])
        trace_data["model_answer"] = model_answer
        trace_data["is_correct"] = is_correct
        if model_answer is None:
            unparseable += 1
        elif is_correct:
            correct_count += 1
        else:
            incorrect_count += 1

    print(f"  Correct: {correct_count}, Incorrect: {incorrect_count}, "
          f"Unparseable: {unparseable}")

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    # Phase 2: Project onto directions
    print(f"\nPhase 2: Projecting onto directions at layers {PROJECTION_LAYERS}...")
    projection_results = []
    for i, trace_data in enumerate(traces):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(traces)}] {trace_data['problem_id']}...",
                  flush=True)

        knowing_proj = project_trace(
            model, trace_data, knowing_norm, PROJECTION_LAYERS, device,
        )
        posthoc_proj = project_trace(
            model, trace_data, posthoc_norm, PROJECTION_LAYERS, device,
        )

        projection_results.append({
            "problem_id": trace_data["problem_id"],
            "ground_truth": trace_data["ground_truth"],
            "model_answer": trace_data["model_answer"],
            "is_correct": trace_data["is_correct"],
            "think_length": trace_data["think_length"],
            "knowing_projection": knowing_proj,
            "posthoc_projection": posthoc_proj,
        })

        if (i + 1) % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Save
    save_data = {
        "pool": args.pool,
        "format": "freeform (non-binary)",
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "unparseable": unparseable,
        "projection_layers": PROJECTION_LAYERS,
        "results": projection_results,
    }
    results_path = os.path.join(OUTPUT_DIR, "transfer_results.json")
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Analysis: correct vs incorrect mean projections
    print("\n--- Mean Projection: Correct vs Incorrect ---")
    for dir_name, proj_key in [("Knowing", "knowing_projection"),
                                ("Post-hoc", "posthoc_projection")]:
        print(f"\n{dir_name} direction:")
        header = f"{'Group':>15}"
        for l in PROJECTION_LAYERS:
            header += f"  L{l:>3}"
        print(header)

        for group_name, filter_fn in [
            ("correct", lambda r: r["is_correct"]),
            ("incorrect", lambda r: not r["is_correct"] and r["model_answer"] is not None),
        ]:
            entries = [r for r in projection_results if filter_fn(r)]
            if not entries:
                continue
            line = f"{group_name:>15}"
            for l in PROJECTION_LAYERS:
                vals = []
                for r in entries:
                    proj = r[proj_key]
                    if proj and l in proj.get("projections", {}):
                        vals.append(np.mean(proj["projections"][l]))
                if vals:
                    line += f"  {np.mean(vals):>6.1f}"
                else:
                    line += f"  {'n/a':>6}"
            print(line)

    # Plot: correct vs incorrect projection trajectories at L24
    ref_layer = 24
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (dir_name, proj_key) in [(ax1, ("Knowing", "knowing_projection")),
                                      (ax2, ("Post-hoc", "posthoc_projection"))]:
        for group_name, filter_fn, color in [
            ("correct", lambda r: r["is_correct"], "steelblue"),
            ("incorrect", lambda r: not r["is_correct"] and r["model_answer"] is not None, "coral"),
        ]:
            entries = [r for r in projection_results if filter_fn(r)]
            if not entries:
                continue

            all_fracs = []
            all_projs = []
            for r in entries[:30]:
                proj = r[proj_key]
                if proj and ref_layer in proj.get("projections", {}):
                    all_fracs.append(proj["fracs"])
                    all_projs.append(proj["projections"][ref_layer])

            if not all_fracs:
                continue

            for fracs, projs in zip(all_fracs, all_projs):
                ax.plot(fracs, projs, alpha=0.1, linewidth=0.5, color=color)

            common_fracs = np.linspace(0, 1, 20)
            interp_projs = []
            for fracs, projs in zip(all_fracs, all_projs):
                interp_projs.append(np.interp(common_fracs, fracs, projs))
            mean_proj = np.mean(interp_projs, axis=0)
            ax.plot(common_fracs, mean_proj, linewidth=2, color=color,
                    label=f"{group_name} (n={len(entries)})")

        ax.set_xlabel("Fraction of thinking")
        ax.set_ylabel(f"Projection onto {dir_name[0]} direction")
        ax.set_title(f"Non-Binary Transfer: {dir_name} Direction (L{ref_layer})")
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "nonbinary_projections.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    # Plot: projection vs think length (scatter)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (dir_name, proj_key) in [(ax1, ("Knowing", "knowing_projection")),
                                      (ax2, ("Post-hoc", "posthoc_projection"))]:
        for r in projection_results:
            proj = r[proj_key]
            if proj is None or ref_layer not in proj.get("projections", {}):
                continue
            mean_proj = np.mean(proj["projections"][ref_layer])
            color = "steelblue" if r["is_correct"] else "coral"
            marker = "o" if r["is_correct"] else "x"
            ax.scatter(r["think_length"], mean_proj, c=color, marker=marker,
                       alpha=0.5, s=30)

        ax.set_xlabel("Think length (tokens)")
        ax.set_ylabel(f"Mean {dir_name[0]} projection (L{ref_layer})")
        ax.set_title(f"{dir_name}: Projection vs Think Length")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "nonbinary_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
