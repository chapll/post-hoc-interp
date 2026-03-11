"""Cross-domain transfer test: DeepMath-derived directions on other datasets.

Tests whether directions extracted from DeepMath Pool A transfer to
completely different datasets (GSM-Symbolic, FOLIO, StrategyQA) that were
used in the original v1 experiments.

These datasets use binary Mu/Nu format, so we can run full force-answer
probing and compare classification quality with DeepMath results.

Usage:
    python scripts/test_transfer_crossdomain.py
"""

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
from transformers import AutoModelForCausalLM

V1_TRACES_DIR = "data/traces/valid"
DIRECTIONS_DIR = "experiments/v2/directions"
OUTPUT_DIR = "experiments/v2/transfer/crossdomain"

CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

TRACE_INDEX = 0
PROBE_SPACING = 20
PROJECTION_LAYERS = [16, 20, 24, 28, 32]
DATASETS = ["gsm_symbolic", "folio", "strategyqa"]
MAX_PER_DATASET = 50


def load_v1_traces(traces_dir, datasets, trace_index, max_per_ds):
    """Load v1 traces from existing datasets."""
    loaded = []
    for ds in datasets:
        ds_dir = os.path.join(traces_dir, ds)
        if not os.path.isdir(ds_dir):
            raise FileNotFoundError(f"Dataset directory not found: {ds_dir}. v1 traces must exist.")
        ds_count = 0
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
            think_length = think_end - think_start - 1
            if think_length < 50:
                continue
            gt = meta["ground_truth"]
            expected = meta["valid_token"] if gt == "Yes" else meta["invalid_token"]
            loaded.append({
                "problem_id": pid,
                "dataset": ds,
                "token_ids": trace["token_ids"],
                "think_start": think_start,
                "think_end": think_end,
                "think_length": think_length,
                "expected_token": expected,
                "ground_truth": gt,
            })
            ds_count += 1
            if ds_count >= max_per_ds:
                break
        print(f"  {ds}: {ds_count} traces")
    return loaded


def probe_trace(model, trace_data, device, spacing):
    """Probe P(correct) at sparse positions."""
    token_ids = trace_data["token_ids"]
    think_start = trace_data["think_start"]
    think_length = trace_data["think_length"]
    expected = trace_data["expected_token"]
    prefix_len = think_start + 1

    positions = list(range(1, think_length + 1, spacing))
    if positions[-1] != think_length:
        positions.append(think_length)

    results = []
    for pos in positions:
        cache_len = prefix_len + pos
        input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_tensor).logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
        p_mu = probs[MU_TOKEN_ID].item()
        p_nu = probs[NU_TOKEN_ID].item()
        p_correct = p_mu if expected == "Mu" else p_nu
        results.append({"pos": pos, "frac": pos / think_length, "p_correct": p_correct})
    return results


def classify_trace(probes, think_length):
    """Classify trace based on P(correct) curve shape."""
    fracs = np.array([r["frac"] for r in probes])
    p_correct = np.array([r["p_correct"] for r in probes])

    early_mask = fracs < 0.15
    early_p = p_correct[early_mask] if early_mask.any() else np.array([p_correct[0]])
    mean_early_p = early_p.mean()

    transition_frac = None
    for i in range(len(p_correct)):
        if p_correct[i] > 0.9:
            remaining = p_correct[i:]
            if (remaining > 0.8).mean() > 0.8:
                transition_frac = fracs[i]
                break

    plateau_start = None
    for i in range(len(p_correct) - 1, -1, -1):
        if p_correct[i] < 0.85:
            plateau_start = i + 1
            break
    if plateau_start is None:
        plateau_start = 0
    plateau_frac = 1.0 - (fracs[plateau_start] if plateau_start < len(fracs) else 1.0)

    metrics = {
        "mean_early_p": float(mean_early_p),
        "transition_frac": float(transition_frac) if transition_frac else None,
        "plateau_frac": float(plateau_frac),
        "final_p": float(p_correct[-1]),
        "think_length": think_length,
    }

    if mean_early_p > 0.85:
        category = "always_knows"
    elif transition_frac is None:
        category = "uncertain" if p_correct[-1] < 0.7 else "late_learner"
    elif transition_frac < 0.5 and plateau_frac > 0.3 and mean_early_p < 0.7:
        category = "post_hoc"
    elif transition_frac > 0.7:
        category = "late_learner"
    elif plateau_frac < 0.2:
        category = "gradual"
    else:
        category = "gradual"

    return category, metrics


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
    device = "cuda:0"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load deepmath-derived directions
    print("Loading DeepMath Pool A directions...")
    knowing_data = load_file(os.path.join(DIRECTIONS_DIR, "knowing_direction.safetensors"))
    posthoc_data = load_file(os.path.join(DIRECTIONS_DIR, "posthoc_direction.safetensors"))
    knowing_norm = knowing_data["direction_normalized"]
    posthoc_norm = posthoc_data["direction_normalized"]

    # Load v1 traces from other datasets
    print(f"\nLoading v1 traces ({DATASETS}, max {MAX_PER_DATASET} per dataset)...")
    traces = load_v1_traces(V1_TRACES_DIR, DATASETS, TRACE_INDEX, MAX_PER_DATASET)
    print(f"Total: {len(traces)} traces")

    if not traces:
        print("No traces found.")
        return

    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    # Phase 1: Scan and classify
    print(f"\nPhase 1: Scanning {len(traces)} cross-domain traces...")
    scan_results = []
    for i, trace_data in enumerate(traces):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(traces)}] {trace_data['problem_id']} "
                  f"({trace_data['dataset']})...", flush=True)
        probes = probe_trace(model, trace_data, device, PROBE_SPACING)
        category, metrics = classify_trace(probes, trace_data["think_length"])
        scan_results.append((trace_data, probes, category, metrics))
        if (i + 1) % 20 == 0:
            torch.cuda.empty_cache()

    # Per-dataset classification
    print("\n--- Cross-Domain Classification ---")
    for ds in DATASETS:
        ds_results = [(td, p, c, m) for td, p, c, m in scan_results
                      if td["dataset"] == ds]
        if not ds_results:
            continue
        cat_counts = {}
        for _, _, c, _ in ds_results:
            cat_counts[c] = cat_counts.get(c, 0) + 1
        print(f"\n  {ds} (n={len(ds_results)}):")
        for cat, count in sorted(cat_counts.items()):
            print(f"    {cat}: {count} ({100*count/len(ds_results):.1f}%)")

    # Phase 2: Project onto deepmath directions
    print(f"\nPhase 2: Projecting onto DeepMath directions...")
    projection_results = []
    for i, (trace_data, probes, category, metrics) in enumerate(scan_results):
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(scan_results)}]...", flush=True)

        knowing_proj = project_trace(
            model, trace_data, knowing_norm, PROJECTION_LAYERS, device,
        )
        posthoc_proj = project_trace(
            model, trace_data, posthoc_norm, PROJECTION_LAYERS, device,
        )

        projection_results.append({
            "problem_id": trace_data["problem_id"],
            "dataset": trace_data["dataset"],
            "category": category,
            "metrics": metrics,
            "knowing_projection": knowing_proj,
            "posthoc_projection": posthoc_proj,
        })

        if (i + 1) % 20 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Save results
    save_data = {
        "source_directions": "DeepMath Pool A",
        "target_datasets": DATASETS,
        "projection_layers": PROJECTION_LAYERS,
        "results": projection_results,
    }
    results_path = os.path.join(OUTPUT_DIR, "crossdomain_results.json")
    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Analysis: mean projection by category and dataset
    ref_layer = 24
    print(f"\n--- Mean Projection at L{ref_layer} by Dataset and Category ---")
    for dir_name, proj_key in [("Knowing", "knowing_projection"),
                                ("Post-hoc", "posthoc_projection")]:
        print(f"\n{dir_name} direction:")
        for ds in DATASETS:
            ds_entries = [r for r in projection_results if r["dataset"] == ds]
            if not ds_entries:
                continue
            print(f"  {ds}:")
            by_cat = {}
            for r in ds_entries:
                by_cat.setdefault(r["category"], []).append(r)
            for cat in sorted(by_cat.keys()):
                entries = by_cat[cat]
                vals = []
                for r in entries:
                    proj = r[proj_key]
                    if proj and ref_layer in proj.get("projections", {}):
                        vals.append(np.mean(proj["projections"][ref_layer]))
                if vals:
                    print(f"    {cat}: mean={np.mean(vals):.1f} "
                          f"std={np.std(vals):.1f} (n={len(entries)})")

    # Plot: per-dataset projection trajectories at L24
    fig, axes = plt.subplots(len(DATASETS), 2,
                              figsize=(14, 5 * len(DATASETS)), squeeze=False)

    for ds_idx, ds in enumerate(DATASETS):
        ds_entries = [r for r in projection_results if r["dataset"] == ds]
        if not ds_entries:
            continue

        for col_idx, (dir_name, proj_key) in enumerate(
            [("Knowing", "knowing_projection"), ("Post-hoc", "posthoc_projection")]
        ):
            ax = axes[ds_idx, col_idx]

            for category in ["post_hoc", "always_knows", "gradual"]:
                cat_entries = [r for r in ds_entries if r["category"] == category]
                if not cat_entries:
                    continue

                all_fracs = []
                all_projs = []
                for r in cat_entries[:15]:
                    proj = r[proj_key]
                    if proj and ref_layer in proj.get("projections", {}):
                        all_fracs.append(proj["fracs"])
                        all_projs.append(proj["projections"][ref_layer])

                if not all_fracs:
                    continue

                for fracs, projs in zip(all_fracs, all_projs):
                    ax.plot(fracs, projs, alpha=0.15, linewidth=0.5)

                common_fracs = np.linspace(0, 1, 20)
                interp_projs = []
                for fracs, projs in zip(all_fracs, all_projs):
                    interp_projs.append(np.interp(common_fracs, fracs, projs))
                mean_proj = np.mean(interp_projs, axis=0)
                ax.plot(common_fracs, mean_proj, linewidth=2,
                        label=f"{category} (n={len(cat_entries)})")

            ax.set_xlabel("Fraction of thinking")
            ax.set_ylabel("Projection")
            ax.set_title(f"{ds} -> DeepMath {dir_name} (L{ref_layer})")
            ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "crossdomain_projections.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
