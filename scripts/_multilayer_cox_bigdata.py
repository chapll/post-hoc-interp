"""Multi-layer Cox probe with cross-dataset generalization testing.

Trains train-on-end probes across layers. Key evaluation:
1. Within-dataset: 5-fold CV AUROC on training dataset (sanity check)
2. Cross-dataset: Train on one dataset, test end-of-trace AUROC on others
3. Curve quality: Spearman correlation (position vs P(correct)) per-trace

The cross-dataset test is the real generalizability check. If a probe trained
on GSM-Symbolic generalizes to FOLIO and StrategyQA, it's learning how the
model represents answers, not dataset-specific features.

Usage:
    python scripts/_multilayer_cox_bigdata.py
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.model import HIDDEN_DIM, load_model
from src.tokens import count_think_blocks, get_pre_handoff_window, get_think_content_mask
from src.utils import set_seed

set_seed(137)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASETS = ["gsm_symbolic", "strategyqa", "folio", "deepmath"]
TRACE_BASE = "src/experimental/gorn/data/traces/valid"
OUTPUT_DIR = "outputs/multilayer_probe"
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]
WINDOW_SIZE = 3
N_FOLDS = 5
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10
SEED = 137
GPU_ID = 0

N_EVAL_TRACES_PER_DS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

device = torch.device(f"cuda:{GPU_ID}")


# ---------------------------------------------------------------------------
# Correctness determination
# ---------------------------------------------------------------------------
def determine_correctness(metadata, generated_text):
    ground_truth = metadata.get("ground_truth")
    valid_token = metadata.get("valid_token")
    invalid_token = metadata.get("invalid_token")
    if ground_truth is None or valid_token is None or invalid_token is None:
        return None
    idx = generated_text.find("</think>")
    if idx < 0:
        return None
    response = generated_text[idx + len("</think>"):].strip()
    expected = valid_token if ground_truth == "Yes" else invalid_token
    other = invalid_token if ground_truth == "Yes" else valid_token
    has_expected = expected in response
    has_other = other in response
    if has_expected and not has_other:
        return True
    if has_other and not has_expected:
        return False
    return None


# ---------------------------------------------------------------------------
# Discover traces per dataset
# ---------------------------------------------------------------------------
def discover_traces_per_dataset():
    """Returns dict: dataset_name -> list of trace dicts."""
    per_ds = {}

    for ds in DATASETS:
        trace_dir = os.path.join(TRACE_BASE, ds)
        if not os.path.isdir(trace_dir):
            print(f"  Skipping {ds}: not found")
            continue

        dirs = sorted(d for d in os.listdir(trace_dir)
                      if os.path.isdir(os.path.join(trace_dir, d)))

        ds_traces = []
        for pdir in dirs:
            tp = os.path.join(trace_dir, pdir, "trace_00.json")
            if not os.path.isfile(tp):
                continue

            with open(tp) as f:
                data = json.load(f)

            token_ids = data["token_ids"]
            metadata = data.get("metadata", {})
            generated_text = data.get("generated_text", "")

            if count_think_blocks(token_ids) != 1:
                continue
            window = get_pre_handoff_window(token_ids, window_size=WINDOW_SIZE)
            if window is None:
                continue

            is_correct = determine_correctness(metadata, generated_text)
            if is_correct is None or not is_correct:
                continue

            mask = get_think_content_mask(token_ids)
            eligible = [i for i, m in enumerate(mask) if m]
            if len(eligible) < 10:
                continue

            ds_traces.append({
                "trace_path": tp,
                "token_ids": token_ids,
                "ground_truth": metadata["ground_truth"],
                "problem_id": pdir,
                "dataset": ds,
                "n_tokens": len(token_ids),
                "eligible_positions": eligible,
            })

        per_ds[ds] = ds_traces
        yes = sum(1 for t in ds_traces if t["ground_truth"] == "Yes")
        no = len(ds_traces) - yes
        print(f"  {ds}: {len(ds_traces)} traces (Yes={yes}, No={no})")

    return per_ds


# ---------------------------------------------------------------------------
# Extract Pre-Handoff Window activations
# ---------------------------------------------------------------------------
def extract_end_activations(model, traces, layer):
    """Returns X [n_samples, hidden_dim] (CPU), y [n_samples] (numpy)."""
    X_list, y_list = [], []

    for i, trace in enumerate(traces):
        token_ids = trace["token_ids"]
        window = get_pre_handoff_window(token_ids, window_size=WINDOW_SIZE)
        if window is None:
            continue

        start_pos, end_pos = window
        label = 1 if trace["ground_truth"] == "Yes" else 0

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with model.trace(input_ids):
            saved = model.model.layers[layer].output.save()

        tensor = saved.detach().cpu()
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)

        window_vectors = tensor[start_pos:end_pos].float()
        X_list.append(window_vectors)
        y_list.extend([label] * window_vectors.shape[0])

        del saved, input_ids
        torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or (i + 1) == len(traces):
            print(f"    Extraction: {i + 1}/{len(traces)}")

    X = torch.cat(X_list, dim=0)
    y = np.array(y_list)
    return X, y


# ---------------------------------------------------------------------------
# Train probe
# ---------------------------------------------------------------------------
def train_probe_cv(X, y):
    """5-fold stratified CV on GPU. Returns (fold_aurocs, mean_auroc)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aurocs = []
    X_gpu = X.to(device)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X.numpy(), y)):
        train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
        val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

        X_train, y_train = X_gpu[train_idx_t], torch.tensor(y[train_idx], dtype=torch.long, device=device)
        X_val, y_val = X_gpu[val_idx_t], torch.tensor(y[val_idx], dtype=torch.long, device=device)

        probe = nn.Linear(HIDDEN_DIM, 2).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        crit = nn.CrossEntropyLoss()

        best_vl, best_st, pat = float("inf"), None, 0
        for epoch in range(MAX_EPOCHS):
            probe.train()
            loss = crit(probe(X_train), y_train)
            opt.zero_grad(); loss.backward(); opt.step()

            probe.eval()
            with torch.no_grad():
                vl = crit(probe(X_val), y_val).item()
            if vl < best_vl:
                best_vl = vl
                best_st = {k: v.clone() for k, v in probe.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= PATIENCE:
                    break

        if best_st:
            probe.load_state_dict(best_st)
        probe.eval()

        with torch.no_grad():
            vp = torch.softmax(probe(X_val), dim=-1)[:, 1].cpu().numpy()
        yv = y_val.cpu().numpy()
        auroc = roc_auc_score(yv, vp) if len(np.unique(yv)) >= 2 else 0.5
        fold_aurocs.append(auroc)

    del X_gpu
    torch.cuda.empty_cache()
    return fold_aurocs, float(np.mean(fold_aurocs))


def train_probe_final(X, y):
    """Train on all data, return probe on GPU."""
    X_gpu = X.to(device)
    y_t = torch.tensor(y, dtype=torch.long, device=device)

    probe = nn.Linear(HIDDEN_DIM, 2).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    for epoch in range(MAX_EPOCHS):
        probe.train()
        loss = crit(probe(X_gpu), y_t)
        opt.zero_grad(); loss.backward(); opt.step()

    probe.eval()
    del X_gpu, y_t
    torch.cuda.empty_cache()
    return probe


def test_probe_on_data(probe, X_test, y_test):
    """Test a trained probe on held-out data. Returns AUROC."""
    if len(X_test) < 5 or len(np.unique(y_test)) < 2:
        return 0.5

    X_gpu = X_test.to(device)
    with torch.no_grad():
        probs = torch.softmax(probe(X_gpu), dim=-1)[:, 1].cpu().numpy()
    del X_gpu
    torch.cuda.empty_cache()
    return roc_auc_score(y_test, probs)


# ---------------------------------------------------------------------------
# Per-trace curve evaluation
# ---------------------------------------------------------------------------
def evaluate_curves(model, probe, eval_traces, layer):
    """Apply probe per-trace, measure Spearman r and quartile gap."""
    results = []

    for trace in eval_traces:
        token_ids = trace["token_ids"]
        eligible = trace["eligible_positions"]
        gt = trace["ground_truth"]
        start_pos = eligible[0]
        span = eligible[-1] - start_pos if eligible[-1] > start_pos else 1

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with model.trace(input_ids):
            saved = model.model.layers[layer].output.save()

        residual = saved.detach()
        if residual.dim() == 3 and residual.shape[0] == 1:
            residual = residual.squeeze(0)

        norm_positions = []
        p_correct_vals = []

        with torch.no_grad():
            for pos in eligible:
                h = residual[pos, :].float().unsqueeze(0).to(device)
                logits = probe(h)
                prob = F.softmax(logits, dim=-1)
                p_yes = prob[0, 1].item()
                p_correct = p_yes if gt == "Yes" else (1.0 - p_yes)
                norm_positions.append((pos - start_pos) / span)
                p_correct_vals.append(p_correct)

        del saved, input_ids, residual
        torch.cuda.empty_cache()

        norm_positions = np.array(norm_positions)
        p_correct_vals = np.array(p_correct_vals)

        rho, pval = scipy_stats.spearmanr(norm_positions, p_correct_vals)

        n = len(p_correct_vals)
        q_size = max(n // 4, 1)
        q1_mean = float(np.mean(p_correct_vals[:q_size]))
        q4_mean = float(np.mean(p_correct_vals[-q_size:]))

        results.append({
            "problem_id": trace["problem_id"],
            "dataset": trace["dataset"],
            "ground_truth": gt,
            "n_tokens": trace["n_tokens"],
            "spearman_r": float(rho),
            "spearman_p": float(pval),
            "q1_mean": q1_mean,
            "q4_mean": q4_mean,
            "q4_minus_q1": q4_mean - q1_mean,
            "norm_positions": norm_positions,
            "p_correct_vals": p_correct_vals,
        })

    return results


# ---------------------------------------------------------------------------
# Positional AUROC: THE key generalizability test
# ---------------------------------------------------------------------------
POSITION_BINS = [
    ("early", 0.0, 0.2),
    ("early_mid", 0.2, 0.4),
    ("middle", 0.4, 0.6),
    ("late_mid", 0.6, 0.8),
    ("late", 0.8, 1.0),
]


def compute_positional_auroc(model, probe, traces, layer):
    """Compute AUROC at different position ranges using trace-level Yes/No labels.

    For each position bin, collect all probe P(Yes) predictions across all traces
    at positions within that bin, paired with the trace's ground truth label.
    Compute AUROC per bin.

    If AUROC increases from early to late bins, the probe captures a real
    transition signal. If flat at chance, it doesn't generalize.
    """
    # Collect: (normalized_position, p_yes, label) for all traces x all positions
    all_rows = []

    for i, trace in enumerate(traces):
        token_ids = trace["token_ids"]
        eligible = trace["eligible_positions"]
        gt = trace["ground_truth"]
        label = 1 if gt == "Yes" else 0
        start_pos = eligible[0]
        span = eligible[-1] - start_pos if eligible[-1] > start_pos else 1

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        with model.trace(input_ids):
            saved = model.model.layers[layer].output.save()

        residual = saved.detach()
        if residual.dim() == 3 and residual.shape[0] == 1:
            residual = residual.squeeze(0)

        # Sample positions (every eligible position is expensive; subsample)
        # Take every 10th position for speed
        step = max(1, len(eligible) // 100)
        sampled = eligible[::step]

        with torch.no_grad():
            for pos in sampled:
                h = residual[pos, :].float().unsqueeze(0).to(device)
                logits = probe(h)
                prob = F.softmax(logits, dim=-1)
                p_yes = prob[0, 1].item()
                norm_pos = (pos - start_pos) / span
                all_rows.append((norm_pos, p_yes, label))

        del saved, input_ids, residual
        torch.cuda.empty_cache()

        if (i + 1) % 50 == 0 or (i + 1) == len(traces):
            print(f"    Positional AUROC: {i + 1}/{len(traces)} traces")

    # Compute AUROC per bin
    pos_aurocs = {}
    for bin_name, lo, hi in POSITION_BINS:
        bin_rows = [(p_yes, label) for (norm_pos, p_yes, label) in all_rows
                    if lo <= norm_pos < hi]
        if len(bin_rows) < 10:
            pos_aurocs[bin_name] = None
            continue

        p_yes_arr = np.array([r[0] for r in bin_rows])
        labels_arr = np.array([r[1] for r in bin_rows])

        if len(np.unique(labels_arr)) < 2:
            pos_aurocs[bin_name] = 0.5
        else:
            pos_aurocs[bin_name] = float(roc_auc_score(labels_arr, p_yes_arr))

    return pos_aurocs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    print("=" * 70)
    print("MULTI-LAYER COX PROBE -- CROSS-DATASET GENERALIZATION")
    print("=" * 70)

    # Step 1: Discover traces per dataset
    print("\nStep 1: Discovering traces...")
    per_ds = discover_traces_per_dataset()
    all_traces = []
    for ds_traces in per_ds.values():
        all_traces.extend(ds_traces)
    print(f"  Combined: {len(all_traces)} traces")

    # Step 2: Load model
    print("\nStep 2: Loading model...")
    model, tok = load_model(gpu_id=GPU_ID)

    # Step 3: Select eval traces
    rng = np.random.RandomState(SEED)
    eval_traces = []
    for ds in DATASETS:
        ds_t = per_ds.get(ds, [])
        if len(ds_t) > N_EVAL_TRACES_PER_DS:
            idx = rng.choice(len(ds_t), N_EVAL_TRACES_PER_DS, replace=False)
            eval_traces.extend([ds_t[i] for i in idx])
        else:
            eval_traces.extend(ds_t)
    print(f"  Eval traces: {len(eval_traces)}")

    # Step 4: Extract activations for ALL datasets at ALL layers
    # Cache extractions to avoid redundant forward passes
    print("\nStep 3: Extracting activations for all datasets at all layers...")
    extracted = {}  # (dataset, layer) -> (X, y)

    for layer in LAYERS:
        print(f"\n  Layer {layer}:")
        for ds in DATASETS:
            t1 = time.time()
            X, y = extract_end_activations(model, per_ds[ds], layer)
            extracted[(ds, layer)] = (X, y)
            print(f"    {ds}: {X.shape[0]} samples in {time.time() - t1:.1f}s "
                  f"(Yes={int((y==1).sum())}, No={int((y==0).sum())})")

        # Also extract combined
        X_all = torch.cat([extracted[(ds, layer)][0] for ds in DATASETS], dim=0)
        y_all = np.concatenate([extracted[(ds, layer)][1] for ds in DATASETS])
        extracted[("all", layer)] = (X_all, y_all)
        print(f"    all: {X_all.shape[0]} samples")

    # Step 5: Train and evaluate probes
    print("\n" + "=" * 70)
    print("Step 4: Training probes and cross-dataset evaluation")
    print("=" * 70)

    # For each layer:
    # A) Train on each dataset + combined, get within-dataset CV AUROC
    # B) Train on each dataset, test on every OTHER dataset (cross-ds AUROC)
    # C) Train on combined, evaluate curve quality

    layer_results = {}

    for layer in LAYERS:
        print(f"\n--- Layer {layer} ---")
        lr = {"within_cv": {}, "cross_ds": {}, "curve_quality": {}}

        # A) Within-dataset CV
        for train_ds in DATASETS + ["all"]:
            X, y = extracted[(train_ds, layer)]
            folds, mean = train_probe_cv(X, y)
            lr["within_cv"][train_ds] = {"mean_auroc": mean, "folds": folds}
            print(f"  Within-CV {train_ds}: AUROC={mean:.4f}")

        # B) Cross-dataset: train on each ds, test on others
        for train_ds in DATASETS:
            X_train, y_train = extracted[(train_ds, layer)]
            probe = train_probe_final(X_train, y_train)

            lr["cross_ds"][train_ds] = {}
            for test_ds in DATASETS:
                if test_ds == train_ds:
                    continue
                X_test, y_test = extracted[(test_ds, layer)]
                auroc = test_probe_on_data(probe, X_test, y_test)
                lr["cross_ds"][train_ds][test_ds] = auroc

            # Print cross-ds results
            cross_vals = lr["cross_ds"][train_ds]
            cross_str = ", ".join(f"{td}={av:.4f}" for td, av in cross_vals.items())
            print(f"  Cross-DS train={train_ds}: {cross_str}")

        # C) Curve quality: use combined probe
        X_all, y_all = extracted[("all", layer)]
        probe_all = train_probe_final(X_all, y_all)

        # Save probe
        probe_tensors = {
            f"layer_{layer}_weight": probe_all.weight.data.cpu(),
            f"layer_{layer}_bias": probe_all.bias.data.cpu(),
        }
        save_file(probe_tensors, os.path.join(OUTPUT_DIR, f"probe_layer{layer}.safetensors"))

        curve_results = evaluate_curves(model, probe_all, eval_traces, layer)
        spearman_rs = [r["spearman_r"] for r in curve_results]
        q_gaps = [r["q4_minus_q1"] for r in curve_results]

        lr["curve_quality"] = {
            "mean_spearman": float(np.mean(spearman_rs)),
            "median_spearman": float(np.median(spearman_rs)),
            "frac_positive": float(np.mean(np.array(spearman_rs) > 0)),
            "mean_q_gap": float(np.mean(q_gaps)),
            "curve_results": curve_results,
        }
        print(f"  Curves (combined probe): Spearman mean={np.mean(spearman_rs):.4f}, "
              f"Q4-Q1 gap={np.mean(q_gaps):.4f}")

        # D) Positional AUROC: the key generalizability test
        print(f"  Computing positional AUROC (all {len(all_traces)} traces)...")
        pos_aurocs = compute_positional_auroc(model, probe_all, all_traces, layer)
        lr["positional_auroc"] = pos_aurocs
        pos_str = ", ".join(f"{k}={v:.4f}" if v is not None else f"{k}=N/A"
                            for k, v in pos_aurocs.items())
        print(f"  Positional AUROC: {pos_str}")

        layer_results[layer] = lr

    # Step 6: Summary tables
    print("\n" + "=" * 70)
    print("SUMMARY: WITHIN-DATASET CV AUROC")
    print("=" * 70)
    print(f"\n{'Layer':>6}", end="")
    for ds in DATASETS + ["all"]:
        print(f" | {ds:>14}", end="")
    print()
    print("-" * 75)
    for layer in LAYERS:
        print(f"  {layer:4d}", end="")
        for ds in DATASETS + ["all"]:
            auroc = layer_results[layer]["within_cv"][ds]["mean_auroc"]
            print(f" | {auroc:14.4f}", end="")
        print()

    print("\n" + "=" * 70)
    print("SUMMARY: CROSS-DATASET GENERALIZATION (end-of-trace AUROC)")
    print("=" * 70)
    for layer in LAYERS:
        print(f"\n  Layer {layer}:")
        for train_ds in DATASETS:
            cross = layer_results[layer]["cross_ds"][train_ds]
            parts = [f"{td}={av:.4f}" for td, av in cross.items()]
            print(f"    Train on {train_ds:>14} -> test: {', '.join(parts)}")

    print("\n" + "=" * 70)
    print("SUMMARY: POSITIONAL AUROC (the generalizability test)")
    print("=" * 70)
    bin_names = [b[0] for b in POSITION_BINS]
    print(f"\n{'Layer':>6}", end="")
    for bn in bin_names:
        print(f" | {bn:>10}", end="")
    print()
    print("-" * (8 + 13 * len(bin_names)))
    for layer in LAYERS:
        pa = layer_results[layer]["positional_auroc"]
        print(f"  {layer:4d}", end="")
        for bn in bin_names:
            v = pa.get(bn)
            if v is not None:
                print(f" | {v:10.4f}", end="")
            else:
                print(f" | {'N/A':>10}", end="")
        print()

    print("\n" + "=" * 70)
    print("SUMMARY: CURVE QUALITY (Spearman r, combined probe)")
    print("=" * 70)
    print(f"\n{'Layer':>6} | {'Mean Spearman':>14} | {'Med Spearman':>13} | "
          f"{'Frac>0':>7} | {'Mean Q4-Q1':>11}")
    print("-" * 65)
    for layer in LAYERS:
        cq = layer_results[layer]["curve_quality"]
        print(f"  {layer:4d} | {cq['mean_spearman']:14.4f} | {cq['median_spearman']:13.4f} | "
              f"{cq['frac_positive']:7.2f} | {cq['mean_q_gap']:11.4f}")

    # Step 7: Save JSON
    results_json = {
        "layers": LAYERS,
        "datasets": {ds: len(per_ds[ds]) for ds in DATASETS},
        "per_layer": {},
    }
    for layer in LAYERS:
        lr = layer_results[layer]
        results_json["per_layer"][str(layer)] = {
            "within_cv": {ds: lr["within_cv"][ds]["mean_auroc"] for ds in DATASETS + ["all"]},
            "cross_ds": lr["cross_ds"],
            "curve_quality": {
                "mean_spearman": lr["curve_quality"]["mean_spearman"],
                "median_spearman": lr["curve_quality"]["median_spearman"],
                "frac_positive": lr["curve_quality"]["frac_positive"],
                "mean_q_gap": lr["curve_quality"]["mean_q_gap"],
            },
            "positional_auroc": lr["positional_auroc"],
        }
    with open(os.path.join(OUTPUT_DIR, "multilayer_results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    # Step 8: Plots
    # 8a: AUROC vs Layer (within-CV for each dataset)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(26, 6))

    ds_colors = {"gsm_symbolic": "#0077BB", "strategyqa": "#009988",
                 "folio": "#EE7733", "deepmath": "#CC3311", "all": "#2c3e50"}
    ds_labels = {"gsm_symbolic": "GSM-Symbolic", "strategyqa": "StrategyQA",
                 "folio": "FOLIO", "deepmath": "DeepMath", "all": "All Combined"}

    # Panel 1: Within-CV AUROC
    for ds in DATASETS + ["all"]:
        aurocs = [layer_results[l]["within_cv"][ds]["mean_auroc"] for l in LAYERS]
        stds = [np.std(layer_results[l]["within_cv"][ds]["folds"]) for l in LAYERS]
        ax1.errorbar(LAYERS, aurocs, yerr=stds, marker="o", linewidth=2, capsize=3,
                     color=ds_colors[ds], label=ds_labels[ds])
    ax1.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel("Layer", fontsize=12)
    ax1.set_ylabel("AUROC", fontsize=12)
    ax1.set_title("Within-Dataset CV AUROC", fontsize=13)
    ax1.set_xticks(LAYERS)
    ax1.set_ylim(0.4, 1.05)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cross-dataset AUROC (train on gsm_symbolic, test on others)
    # Show all train->test combinations
    for train_ds in DATASETS:
        for test_ds in DATASETS:
            if test_ds == train_ds:
                continue
            aurocs = [layer_results[l]["cross_ds"][train_ds].get(test_ds, 0.5) for l in LAYERS]
            label = f"{ds_labels[train_ds][:3]}->{ds_labels[test_ds][:3]}"
            ax2.plot(LAYERS, aurocs, marker="o", linewidth=1.5, label=label, alpha=0.8)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel("Layer", fontsize=12)
    ax2.set_ylabel("AUROC", fontsize=12)
    ax2.set_title("Cross-Dataset Generalization", fontsize=13)
    ax2.set_xticks(LAYERS)
    ax2.set_ylim(0.3, 1.05)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Curve quality (Spearman) -- combined probe
    spearman_means = [layer_results[l]["curve_quality"]["mean_spearman"] for l in LAYERS]
    spearman_meds = [layer_results[l]["curve_quality"]["median_spearman"] for l in LAYERS]
    ax3.plot(LAYERS, spearman_means, marker="s", linewidth=2, color="#CC3311", label="Mean")
    ax3.plot(LAYERS, spearman_meds, marker="^", linewidth=2, color="#CC3311",
             linestyle="--", alpha=0.7, label="Median")
    ax3.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax3.set_xlabel("Layer", fontsize=12)
    ax3.set_ylabel("Spearman r (pos vs P(correct))", fontsize=12)
    ax3.set_title("Curve Quality (combined probe)", fontsize=13)
    ax3.set_xticks(LAYERS)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Positional AUROC at best end-AUROC layer
    best_end_layer = max(LAYERS, key=lambda l: layer_results[l]["within_cv"]["all"]["mean_auroc"])
    pa = layer_results[best_end_layer]["positional_auroc"]
    bin_names_plot = [b[0] for b in POSITION_BINS]
    bin_centers_plot = [(b[1] + b[2]) / 2 for b in POSITION_BINS]
    pa_vals = [pa.get(bn, 0.5) or 0.5 for bn in bin_names_plot]
    ax4.bar(bin_centers_plot, pa_vals, width=0.15, color="#0077BB", alpha=0.8, edgecolor="black")
    ax4.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax4.set_xlabel("Normalized Position", fontsize=12)
    ax4.set_ylabel("AUROC", fontsize=12)
    ax4.set_title(f"Positional AUROC (Layer {best_end_layer})", fontsize=13)
    ax4.set_ylim(0.3, 1.05)
    ax4.set_xticks(bin_centers_plot)
    ax4.set_xticklabels([f"{b[1]:.1f}-{b[2]:.1f}" for b in POSITION_BINS], fontsize=8)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "plots", "summary_4panel.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 8b: Per-trace plots at best Spearman layer
    best_layer = max(LAYERS, key=lambda l: layer_results[l]["curve_quality"]["mean_spearman"])
    print(f"\nBest curve-quality layer: {best_layer}")

    cr = layer_results[best_layer]["curve_quality"]["curve_results"]
    plot_sample = sorted(cr, key=lambda r: r["spearman_r"], reverse=True)
    plot_sample = plot_sample[:4] + plot_sample[-4:]

    fig, axes = plt.subplots(2, 4, figsize=(24, 8))
    axes_flat = axes.flatten()

    for ax_idx, result in enumerate(plot_sample):
        ax = axes_flat[ax_idx]
        np_arr = result["norm_positions"]
        pc_arr = result["p_correct_vals"]

        ax.plot(np_arr, pc_arr, linewidth=0.4, color="#bdc3c7", alpha=0.5)
        w = min(50, len(pc_arr) // 5)
        if w >= 3:
            smoothed = np.convolve(pc_arr, np.ones(w) / w, mode="valid")
            smooth_x = np_arr[w // 2: w // 2 + len(smoothed)]
            ax.plot(smooth_x, smoothed, linewidth=2.0, color="#2c3e50")
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, 1)

        ds_short = result["dataset"][:4]
        rho_val = result["spearman_r"]
        ax.set_title(f"{result['problem_id'][:18]} ({ds_short})\n"
                     f"GT={result['ground_truth']} rho={rho_val:.3f}", fontsize=8)
        if ax_idx >= 4:
            ax.set_xlabel("Norm. pos", fontsize=8)
        if ax_idx % 4 == 0:
            ax.set_ylabel("P(correct)", fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"Per-Trace P(correct) | Layer {best_layer} | "
                 f"Top 4 (left) and Bottom 4 (right) by Spearman r", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "plots", f"pertrace_layer{best_layer}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
