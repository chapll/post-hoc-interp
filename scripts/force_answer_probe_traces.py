"""Force-answer at many positions on held-out traces and plot P(correct).

Uses the ** suffix for clean Mu/Nu output at all positions.
Probes every Nth thinking position and records P(correct answer).

Usage:
    python scripts/force_answer_probe_traces.py
"""

import json
import glob
import os
import random
import warnings

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
TRACES_DIR = "data/traces/valid"
OUTPUT_DIR = "experiments/activation_diff/force_answer_probe"
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **
MU_TOKEN_ID = 39120
NU_TOKEN_ID = 60335

N_TEST_TRACES = 8
TEST_TRACE_INDEX = 1  # held out (direction computed from trace_00)
PROBE_SPACING = 10     # probe every 10th thinking token
SEED = 137


def load_test_traces(traces_dir, n_traces, trace_index, seed):
    """Load diverse held-out traces."""
    rng = random.Random(seed)
    all_traces = []
    for ds in ["gsm_symbolic", "folio", "strategyqa"]:
        ds_dir = os.path.join(traces_dir, ds)
        if not os.path.isdir(ds_dir):
            continue
        for pid in sorted(os.listdir(ds_dir)):
            path = os.path.join(ds_dir, pid, f"trace_{trace_index:02d}.json")
            if os.path.isfile(path):
                all_traces.append((ds, pid, path))

    rng.shuffle(all_traces)

    # Mix of datasets
    by_ds = {}
    for ds, pid, path in all_traces:
        by_ds.setdefault(ds, []).append((ds, pid, path))

    selected = []
    per_ds = max(1, n_traces // len(by_ds))
    for ds in sorted(by_ds.keys()):
        selected.extend(by_ds[ds][:per_ds])
    remaining = [t for t in all_traces if t not in selected]
    while len(selected) < n_traces and remaining:
        selected.append(remaining.pop(0))
    selected = selected[:n_traces]

    loaded = []
    for ds, pid, path in selected:
        with open(path) as f:
            trace = json.load(f)
        meta = trace["metadata"]
        if not meta.get("has_think") or meta.get("num_think_blocks", 0) != 1:
            continue
        think_start, think_end = meta["think_boundaries"][0]
        think_length = think_end - think_start - 1
        if think_length < 30:
            continue
        gt = meta["ground_truth"]
        expected = meta["valid_token"] if gt == "Yes" else meta["invalid_token"]
        loaded.append({
            "problem_id": pid,
            "dataset": ds,
            "token_ids": trace["token_ids"],
            "think_start": think_start,
            "think_length": think_length,
            "expected_token": expected,
            "ground_truth": gt,
        })
    return loaded


def probe_trace_force_answer(model, trace_data, device, spacing=10):
    """Force-answer at every Nth thinking position and return P(correct)."""
    token_ids = trace_data["token_ids"]
    think_start = trace_data["think_start"]
    think_length = trace_data["think_length"]
    expected = trace_data["expected_token"]
    expected_id = MU_TOKEN_ID if expected == "Mu" else NU_TOKEN_ID

    prefix_ids = token_ids[:think_start + 1]  # up to and including <think>

    positions = list(range(1, think_length + 1, spacing))
    # Always include the last position
    if positions[-1] != think_length:
        positions.append(think_length)

    results = []
    for pos in positions:
        cache_len = len(prefix_ids) + pos
        input_ids = token_ids[:cache_len] + CLOSE_TOKEN_IDS
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_tensor).logits[0, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)

        p_mu = probs[MU_TOKEN_ID].item()
        p_nu = probs[NU_TOKEN_ID].item()
        p_correct = p_mu if expected == "Mu" else p_nu
        p_sum = p_mu + p_nu

        results.append({
            "pos": pos,
            "frac": pos / think_length,
            "p_correct": p_correct,
            "p_mu": p_mu,
            "p_nu": p_nu,
            "p_sum": p_sum,
        })

    return results


def plot_results(all_results, output_dir):
    """Plot P(correct) curves for each trace."""
    os.makedirs(output_dir, exist_ok=True)

    n = len(all_results)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), squeeze=False)

    for i, (trace_data, probes) in enumerate(all_results):
        ax = axes[i, 0]
        fracs = [r["frac"] for r in probes]
        p_correct = [r["p_correct"] for r in probes]
        p_sum = [r["p_sum"] for r in probes]

        ax.plot(fracs, p_correct, "b-o", markersize=2, linewidth=1.2,
                label="P(correct label)")
        ax.plot(fracs, p_sum, "g--", alpha=0.5, linewidth=0.8,
                label="P(Mu)+P(Nu)")

        # Threshold lines
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="0.5")
        ax.axhline(y=0.9, color="red", linestyle=":", alpha=0.5, label="0.9")

        pid = trace_data["problem_id"]
        ds = trace_data["dataset"]
        exp = trace_data["expected_token"]
        gt = trace_data["ground_truth"]
        tl = trace_data["think_length"]
        ax.set_title(
            f"{pid} ({ds}) | expected={exp} gt={gt} | think_len={tl}",
            fontsize=10,
        )
        ax.set_xlabel("Fraction of thinking")
        ax.set_ylabel("Probability")
        ax.set_ylim(-0.05, 1.05)
        if i == 0:
            ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    path = os.path.join(output_dir, "force_answer_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Overlay plot: all traces on one figure
    fig, ax = plt.subplots(figsize=(12, 6))
    for trace_data, probes in all_results:
        fracs = [r["frac"] for r in probes]
        p_correct = [r["p_correct"] for r in probes]
        label = f"{trace_data['problem_id'][:30]} ({trace_data['expected_token']})"
        ax.plot(fracs, p_correct, alpha=0.7, linewidth=1.0, label=label)

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(y=0.9, color="red", linestyle=":", alpha=0.5)
    ax.set_title("Force-Answer P(correct) -- Held-Out Traces")
    ax.set_xlabel("Fraction of thinking")
    ax.set_ylabel("P(correct label)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "force_answer_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    print("Loading test traces...")
    test_traces = load_test_traces(TRACES_DIR, N_TEST_TRACES, TEST_TRACE_INDEX, SEED)
    print(f"Selected {len(test_traces)} traces:")
    for t in test_traces:
        n_probes = len(range(1, t["think_length"] + 1, PROBE_SPACING)) + 1
        print(f"  {t['problem_id']} ({t['dataset']}) exp={t['expected_token']} "
              f"tl={t['think_length']} probes={n_probes}")

    print("\nLoading model...")
    device = "cuda:0"
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B", dtype=torch.bfloat16, device_map=device,
    )
    model.eval()

    print(f"\nProbing traces (every {PROBE_SPACING} positions)...")
    all_results = []
    for i, trace_data in enumerate(test_traces):
        n_probes = len(range(1, trace_data["think_length"] + 1, PROBE_SPACING)) + 1
        print(f"  [{i+1}/{len(test_traces)}] {trace_data['problem_id']} "
              f"({n_probes} probes)...", flush=True)
        probes = probe_trace_force_answer(model, trace_data, device, PROBE_SPACING)
        all_results.append((trace_data, probes))
        torch.cuda.empty_cache()

    print("\nPlotting...")
    plot_results(all_results, OUTPUT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
