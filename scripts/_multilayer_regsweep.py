"""Multi-layer Cox probe with regularization sweep -- multi-GPU parallel.

Runs one layer per GPU in parallel (6 layers on 6 GPUs).
Each worker loads its own model copy and processes all 4 weight decays.

Evaluates each (layer, weight_decay) pair via:
1. End-of-trace CV AUROC (sanity check, NOT the optimization target)
2. Cross-dataset generalization AUROC
3. Positional AUROC (early/mid/late bins) -- THE key metric
4. Per-trace curve smoothness (Spearman r, Q4-Q1 gap)

The best (layer, wd) is selected by positional AUROC ramp, not end-AUROC.

Usage:
    python scripts/_multilayer_regsweep.py
"""

import json
import os
import time
import traceback
from multiprocessing import get_context

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASETS = ["gsm_symbolic", "strategyqa", "folio", "deepmath"]
TRACE_BASE = "src/experimental/gorn/data/traces/valid"
OUTPUT_DIR = "outputs/multilayer_probe"
LAYERS = [8, 16, 20, 24, 28, 32]
WEIGHT_DECAYS = [1e-4, 1e-3, 1e-2, 1e-1]
WINDOW_SIZE = 3
N_FOLDS = 5
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
SEED = 137

POSITION_BINS = [
    ("early", 0.0, 0.2),
    ("early_mid", 0.2, 0.4),
    ("middle", 0.4, 0.6),
    ("late_mid", 0.6, 0.8),
    ("late", 0.8, 1.0),
]


# ---------------------------------------------------------------------------
# Helpers (shared)
# ---------------------------------------------------------------------------
def determine_correctness(metadata, generated_text):
    gt = metadata.get("ground_truth")
    vt = metadata.get("valid_token")
    it = metadata.get("invalid_token")
    if gt is None or vt is None or it is None:
        return None
    idx = generated_text.find("</think>")
    if idx < 0:
        return None
    response = generated_text[idx + 8:].strip()
    expected = vt if gt == "Yes" else it
    other = it if gt == "Yes" else vt
    if expected in response and other not in response:
        return True
    if other in response and expected not in response:
        return False
    return None


def discover_traces():
    """Returns dict: ds -> list of trace dicts (no heavy data, just metadata + token_ids)."""
    per_ds = {}
    for ds in DATASETS:
        trace_dir = os.path.join(TRACE_BASE, ds)
        if not os.path.isdir(trace_dir):
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
            tids = data["token_ids"]
            meta = data.get("metadata", {})
            gen = data.get("generated_text", "")
            if count_think_blocks(tids) != 1:
                continue
            if get_pre_handoff_window(tids, WINDOW_SIZE) is None:
                continue
            ic = determine_correctness(meta, gen)
            if ic is None or not ic:
                continue
            mask = get_think_content_mask(tids)
            eligible = [i for i, m in enumerate(mask) if m]
            if len(eligible) < 10:
                continue
            ds_traces.append({
                "token_ids": tids, "ground_truth": meta["ground_truth"],
                "problem_id": pdir, "dataset": ds,
                "n_tokens": len(tids), "eligible_positions": eligible,
            })
        per_ds[ds] = ds_traces
    return per_ds


# ---------------------------------------------------------------------------
# Per-GPU worker: handles one layer, all weight decays
# ---------------------------------------------------------------------------
def process_layer(args):
    """Run all weight decays for a single layer on one GPU."""
    layer, gpu_id, per_ds_serialized, eval_trace_ids = args

    set_seed(SEED)
    device = torch.device(f"cuda:{gpu_id}")
    tag = f"[GPU {gpu_id} / Layer {layer}]"
    print(f"{tag} Starting", flush=True)

    # Reconstruct per_ds from serialized data
    per_ds = per_ds_serialized
    all_traces = []
    for ds_t in per_ds.values():
        all_traces.extend(ds_t)

    # Build eval trace list
    eval_traces = []
    for ds in DATASETS:
        dst = per_ds.get(ds, [])
        for i, t in enumerate(dst):
            if (ds, i) in eval_trace_ids:
                eval_traces.append(t)

    # Load model on this GPU
    print(f"{tag} Loading model on GPU {gpu_id}...", flush=True)
    model, tok = load_model(gpu_id=gpu_id)
    print(f"{tag} Model loaded", flush=True)

    # Extract end-of-trace activations for this layer
    print(f"{tag} Extracting end activations...", flush=True)
    extracted = {}  # ds -> (X, y)
    for ds in DATASETS:
        X_list, y_list = [], []
        for i, trace in enumerate(per_ds[ds]):
            window = get_pre_handoff_window(trace["token_ids"], WINDOW_SIZE)
            if window is None:
                continue
            start_pos, end_pos = window
            label = 1 if trace["ground_truth"] == "Yes" else 0
            input_ids = torch.tensor([trace["token_ids"]], dtype=torch.long, device=device)
            with model.trace(input_ids):
                saved = model.model.layers[layer].output.save()
            tensor = saved.detach().cpu()
            if tensor.dim() == 3:
                tensor = tensor.squeeze(0)
            X_list.append(tensor[start_pos:end_pos].float())
            y_list.extend([label] * (end_pos - start_pos))
            del saved, input_ids
            torch.cuda.empty_cache()
            if (i + 1) % 100 == 0 or (i + 1) == len(per_ds[ds]):
                print(f"{tag} Extract {ds}: {i+1}/{len(per_ds[ds])}", flush=True)
        extracted[ds] = (torch.cat(X_list, 0), np.array(y_list))
        print(f"{tag} {ds}: {extracted[ds][0].shape[0]} samples", flush=True)

    X_all = torch.cat([extracted[ds][0] for ds in DATASETS], 0)
    y_all = np.concatenate([extracted[ds][1] for ds in DATASETS])
    print(f"{tag} Total: {X_all.shape[0]} samples", flush=True)

    # Sweep weight decays
    layer_results = {}
    for wd in WEIGHT_DECAYS:
        print(f"\n{tag} --- WD={wd} ---", flush=True)

        # Train + CV on combined data
        X_gpu = X_all.to(device)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_aurocs = []
        for train_idx, val_idx in skf.split(X_all.numpy(), y_all):
            ti = torch.tensor(train_idx, dtype=torch.long, device=device)
            vi = torch.tensor(val_idx, dtype=torch.long, device=device)
            Xt, yt = X_gpu[ti], torch.tensor(y_all[train_idx], dtype=torch.long, device=device)
            Xv, yv = X_gpu[vi], torch.tensor(y_all[val_idx], dtype=torch.long, device=device)

            p = nn.Linear(HIDDEN_DIM, 2).to(device)
            opt = torch.optim.Adam(p.parameters(), lr=LR, weight_decay=wd)
            crit = nn.CrossEntropyLoss()
            best_vl, best_st, pat = float("inf"), None, 0
            for _ in range(MAX_EPOCHS):
                p.train(); loss = crit(p(Xt), yt)
                opt.zero_grad(); loss.backward(); opt.step()
                p.eval()
                with torch.no_grad():
                    vl = crit(p(Xv), yv).item()
                if vl < best_vl:
                    best_vl = vl; best_st = {k: v.clone() for k, v in p.state_dict().items()}; pat = 0
                else:
                    pat += 1
                    if pat >= PATIENCE: break
            if best_st: p.load_state_dict(best_st)
            p.eval()
            with torch.no_grad():
                vp = torch.softmax(p(Xv), dim=-1)[:, 1].cpu().numpy()
            yvn = yv.cpu().numpy()
            fold_aurocs.append(roc_auc_score(yvn, vp) if len(np.unique(yvn)) >= 2 else 0.5)
        mean_auroc = float(np.mean(fold_aurocs))
        print(f"{tag} End-AUROC (CV): {mean_auroc:.4f}", flush=True)
        del X_gpu; torch.cuda.empty_cache()

        # Train final probe on all combined data
        X_gpu = X_all.to(device)
        y_t = torch.tensor(y_all, dtype=torch.long, device=device)
        probe = nn.Linear(HIDDEN_DIM, 2).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=wd)
        crit = nn.CrossEntropyLoss()
        for _ in range(MAX_EPOCHS):
            probe.train(); loss = crit(probe(X_gpu), y_t)
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        del X_gpu, y_t; torch.cuda.empty_cache()

        # Cross-dataset: train on gsm_symbolic, test on others
        X_gsm, y_gsm = extracted["gsm_symbolic"]
        X_gsm_gpu = X_gsm.to(device)
        y_gsm_t = torch.tensor(y_gsm, dtype=torch.long, device=device)
        probe_gsm = nn.Linear(HIDDEN_DIM, 2).to(device)
        opt_gsm = torch.optim.Adam(probe_gsm.parameters(), lr=LR, weight_decay=wd)
        crit = nn.CrossEntropyLoss()
        for _ in range(MAX_EPOCHS):
            probe_gsm.train(); loss = crit(probe_gsm(X_gsm_gpu), y_gsm_t)
            opt_gsm.zero_grad(); loss.backward(); opt_gsm.step()
        probe_gsm.eval()
        del X_gsm_gpu, y_gsm_t; torch.cuda.empty_cache()

        cross_aurocs = {}
        for test_ds in ["folio", "strategyqa"]:
            X_t, y_t = extracted[test_ds]
            if len(X_t) < 5 or len(np.unique(y_t)) < 2:
                cross_aurocs[test_ds] = 0.5
                continue
            X_t_gpu = X_t.to(device)
            with torch.no_grad():
                probs = torch.softmax(probe_gsm(X_t_gpu), dim=-1)[:, 1].cpu().numpy()
            del X_t_gpu; torch.cuda.empty_cache()
            cross_aurocs[test_ds] = float(roc_auc_score(y_t, probs))
        print(f"{tag} Cross-DS: folio={cross_aurocs.get('folio', 0):.4f}, "
              f"strategyqa={cross_aurocs.get('strategyqa', 0):.4f}", flush=True)

        # Positional AUROC
        print(f"{tag} Computing positional AUROC...", flush=True)
        all_rows = []
        for i, trace in enumerate(all_traces):
            tids = trace["token_ids"]
            eligible = trace["eligible_positions"]
            label = 1 if trace["ground_truth"] == "Yes" else 0
            start = eligible[0]
            span = eligible[-1] - start if eligible[-1] > start else 1

            input_ids = torch.tensor([tids], dtype=torch.long, device=device)
            with model.trace(input_ids):
                saved = model.model.layers[layer].output.save()
            res = saved.detach()
            if res.dim() == 3: res = res.squeeze(0)

            step = max(1, len(eligible) // 100)
            sampled = eligible[::step]
            with torch.no_grad():
                for pos in sampled:
                    h = res[pos, :].float().unsqueeze(0).to(device)
                    p_yes = F.softmax(probe(h), dim=-1)[0, 1].item()
                    all_rows.append(((pos - start) / span, p_yes, label))
            del saved, input_ids, res; torch.cuda.empty_cache()
            if (i + 1) % 100 == 0 or (i + 1) == len(all_traces):
                print(f"{tag} PosAUROC: {i+1}/{len(all_traces)}", flush=True)

        pos_aurocs = {}
        for bn, lo, hi in POSITION_BINS:
            rows = [(p, l) for (np_, p, l) in all_rows if lo <= np_ < hi]
            if len(rows) < 10:
                pos_aurocs[bn] = None; continue
            pa = np.array([r[0] for r in rows])
            la = np.array([r[1] for r in rows])
            pos_aurocs[bn] = float(roc_auc_score(la, pa)) if len(np.unique(la)) >= 2 else 0.5
        pos_str = " | ".join(f"{k}={v:.4f}" if v else f"{k}=N/A"
                             for k, v in pos_aurocs.items())
        print(f"{tag} Positional: {pos_str}", flush=True)

        # Curve quality on eval traces
        curve_results = []
        for trace in eval_traces:
            tids = trace["token_ids"]
            eligible = trace["eligible_positions"]
            gt = trace["ground_truth"]
            start = eligible[0]
            span = eligible[-1] - start if eligible[-1] > start else 1

            input_ids = torch.tensor([tids], dtype=torch.long, device=device)
            with model.trace(input_ids):
                saved = model.model.layers[layer].output.save()
            res = saved.detach()
            if res.dim() == 3: res = res.squeeze(0)

            npos, pvals = [], []
            with torch.no_grad():
                for pos in eligible:
                    h = res[pos, :].float().unsqueeze(0).to(device)
                    p_yes = F.softmax(probe(h), dim=-1)[0, 1].item()
                    pc = p_yes if gt == "Yes" else (1.0 - p_yes)
                    npos.append((pos - start) / span)
                    pvals.append(pc)
            del saved, input_ids, res; torch.cuda.empty_cache()

            npos_arr, pvals_arr = np.array(npos), np.array(pvals)
            rho, _ = scipy_stats.spearmanr(npos_arr, pvals_arr)
            qs = max(len(pvals_arr) // 4, 1)

            curve_results.append({
                "problem_id": trace["problem_id"], "dataset": trace["dataset"],
                "ground_truth": gt, "spearman_r": float(rho),
                "q1_mean": float(np.mean(pvals_arr[:qs])),
                "q4_mean": float(np.mean(pvals_arr[-qs:])),
                "q4_minus_q1": float(np.mean(pvals_arr[-qs:]) - np.mean(pvals_arr[:qs])),
                "norm_positions": npos_arr.tolist(),
                "p_correct_vals": pvals_arr.tolist(),
            })

        spearman_rs = [r["spearman_r"] for r in curve_results]
        q_gaps = [r["q4_minus_q1"] for r in curve_results]
        print(f"{tag} Curves: Spearman mean={np.mean(spearman_rs):.4f}, "
              f"Q4-Q1={np.mean(q_gaps):.4f}", flush=True)

        w_norm = probe.weight.data.norm().item()
        print(f"{tag} Weight norm: {w_norm:.4f}", flush=True)

        # Save probe weights
        tensors = {
            f"layer_{layer}_wd_{wd}_weight": probe.weight.data.cpu(),
            f"layer_{layer}_wd_{wd}_bias": probe.bias.data.cpu(),
        }
        save_file(tensors, os.path.join(OUTPUT_DIR, f"probe_L{layer}_wd{wd}.safetensors"))

        layer_results[wd] = {
            "end_auroc": mean_auroc, "fold_aurocs": [float(f) for f in fold_aurocs],
            "cross_aurocs": cross_aurocs,
            "positional_auroc": pos_aurocs,
            "mean_spearman": float(np.mean(spearman_rs)),
            "median_spearman": float(np.median(spearman_rs)),
            "mean_q_gap": float(np.mean(q_gaps)),
            "frac_spearman_pos": float(np.mean(np.array(spearman_rs) > 0)),
            "weight_norm": w_norm,
            "curve_results": curve_results,
        }

    print(f"\n{tag} DONE -- all weight decays processed", flush=True)
    return layer, layer_results


# ---------------------------------------------------------------------------
# Main: orchestrate parallel workers + generate plots/summaries
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

    print("=" * 70)
    print("MULTI-LAYER COX PROBE -- REGULARIZATION SWEEP (MULTI-GPU)")
    print(f"Layers: {LAYERS}")
    print(f"Weight decays: {WEIGHT_DECAYS}")
    print(f"GPUs: {len(LAYERS)} (one per layer)")
    print("=" * 70)

    print("\nDiscovering traces...")
    per_ds = discover_traces()
    all_traces = []
    for ds_t in per_ds.values():
        all_traces.extend(ds_t)
    for ds in DATASETS:
        yes = sum(1 for t in per_ds[ds] if t["ground_truth"] == "Yes")
        print(f"  {ds}: {len(per_ds[ds])} traces (Yes={yes}, No={len(per_ds[ds])-yes})")
    print(f"Combined: {len(all_traces)} traces, {len(all_traces) * WINDOW_SIZE} samples")

    # Pre-select eval traces (same for all layers)
    rng = np.random.RandomState(SEED)
    eval_trace_ids = set()
    for ds in DATASETS:
        dst = per_ds.get(ds, [])
        if len(dst) > 10:
            idx = rng.choice(len(dst), 10, replace=False)
            for i in idx:
                eval_trace_ids.add((ds, int(i)))
        else:
            for i in range(len(dst)):
                eval_trace_ids.add((ds, i))

    # Prepare worker arguments: (layer, gpu_id, per_ds, eval_trace_ids)
    worker_args = []
    for i, layer in enumerate(LAYERS):
        gpu_id = i  # GPU 0 through 5
        worker_args.append((layer, gpu_id, per_ds, eval_trace_ids))

    # Launch parallel workers (one per layer/GPU)
    print(f"\nLaunching {len(LAYERS)} parallel workers...")
    ctx = get_context("spawn")
    with ctx.Pool(processes=len(LAYERS)) as pool:
        raw_results = pool.map(process_layer, worker_args)

    # Collect results
    results = {}
    for layer, layer_results in raw_results:
        for wd, res in layer_results.items():
            results[(layer, wd)] = res

    # =====================================================================
    # Summary tables
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY: END-OF-TRACE CV AUROC")
    print("=" * 70)
    print(f"{'Layer':>6} |", end="")
    for wd in WEIGHT_DECAYS:
        print(f" wd={wd:<8} |", end="")
    print()
    print("-" * (8 + 13 * len(WEIGHT_DECAYS)))
    for layer in LAYERS:
        print(f"  {layer:4d} |", end="")
        for wd in WEIGHT_DECAYS:
            print(f" {results[(layer, wd)]['end_auroc']:10.4f} |", end="")
        print()

    print("\n" + "=" * 70)
    print("SUMMARY: CROSS-DATASET (train=gsm -> test=folio)")
    print("=" * 70)
    print(f"{'Layer':>6} |", end="")
    for wd in WEIGHT_DECAYS:
        print(f" wd={wd:<8} |", end="")
    print()
    for layer in LAYERS:
        print(f"  {layer:4d} |", end="")
        for wd in WEIGHT_DECAYS:
            v = results[(layer, wd)]["cross_aurocs"].get("folio", 0.5)
            print(f" {v:10.4f} |", end="")
        print()

    print("\n" + "=" * 70)
    print("SUMMARY: POSITIONAL AUROC (combined probe)")
    print("=" * 70)
    for layer in LAYERS:
        for wd in WEIGHT_DECAYS:
            pa = results[(layer, wd)]["positional_auroc"]
            vals = " | ".join(f"{bn}={pa.get(bn, 0):.3f}" if pa.get(bn) else f"{bn}=N/A"
                              for bn, _, _ in POSITION_BINS)
            print(f"  L{layer:2d} wd={wd:<6}: {vals}")

    print("\n" + "=" * 70)
    print("SUMMARY: CURVE QUALITY (Spearman r)")
    print("=" * 70)
    print(f"{'Layer':>6} |", end="")
    for wd in WEIGHT_DECAYS:
        print(f" wd={wd:<8} |", end="")
    print()
    for layer in LAYERS:
        print(f"  {layer:4d} |", end="")
        for wd in WEIGHT_DECAYS:
            print(f" {results[(layer, wd)]['mean_spearman']:10.4f} |", end="")
        print()

    print("\n" + "=" * 70)
    print("SUMMARY: WEIGHT NORMS")
    print("=" * 70)
    for layer in LAYERS:
        print(f"  L{layer:2d}:", end="")
        for wd in WEIGHT_DECAYS:
            print(f"  wd={wd}->norm={results[(layer, wd)]['weight_norm']:.2f}", end="")
        print()

    # Find best by positional AUROC ramp (late - early)
    print("\n" + "=" * 70)
    print("BEST CONFIGS (by positional AUROC ramp = late - early)")
    print("=" * 70)
    ramps = []
    for layer in LAYERS:
        for wd in WEIGHT_DECAYS:
            pa = results[(layer, wd)]["positional_auroc"]
            early = pa.get("early")
            late = pa.get("late")
            if early is not None and late is not None:
                ramp = late - early
                ramps.append((ramp, layer, wd))
    ramps.sort(reverse=True)
    for ramp, layer, wd in ramps[:5]:
        r = results[(layer, wd)]
        pa = r["positional_auroc"]
        print(f"  L{layer} wd={wd}: ramp={ramp:.4f} "
              f"(early={pa['early']:.4f}, late={pa['late']:.4f}), "
              f"end_auroc={r['end_auroc']:.4f}, spearman={r['mean_spearman']:.4f}")

    # Save JSON (exclude large curve arrays for the summary)
    json_results = {}
    for (layer, wd), r in results.items():
        k = f"L{layer}_wd{wd}"
        json_results[k] = {
            "layer": layer, "weight_decay": wd,
            "end_auroc": r["end_auroc"],
            "cross_aurocs": r["cross_aurocs"],
            "positional_auroc": r["positional_auroc"],
            "mean_spearman": r["mean_spearman"],
            "median_spearman": r["median_spearman"],
            "mean_q_gap": r["mean_q_gap"],
            "frac_spearman_pos": r["frac_spearman_pos"],
            "weight_norm": r["weight_norm"],
        }
    with open(os.path.join(OUTPUT_DIR, "regsweep_results.json"), "w") as f:
        json.dump(json_results, f, indent=2)

    # Plot: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    wd_colors = {1e-4: "#0077BB", 1e-3: "#33BBEE", 1e-2: "#EE7733", 1e-1: "#CC3311"}

    ax = axes[0, 0]
    for wd in WEIGHT_DECAYS:
        vals = [results[(l, wd)]["end_auroc"] for l in LAYERS]
        ax.plot(LAYERS, vals, marker="o", linewidth=2, color=wd_colors[wd], label=f"wd={wd}")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("End-AUROC")
    ax.set_title("End-of-Trace CV AUROC"); ax.legend(fontsize=9)
    ax.set_xticks(LAYERS); ax.set_ylim(0.4, 1.05); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for wd in WEIGHT_DECAYS:
        vals = [results[(l, wd)]["cross_aurocs"].get("folio", 0.5) for l in LAYERS]
        ax.plot(LAYERS, vals, marker="s", linewidth=2, color=wd_colors[wd], label=f"wd={wd}")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("AUROC")
    ax.set_title("Cross-Dataset (GSM->FOLIO)"); ax.legend(fontsize=9)
    ax.set_xticks(LAYERS); ax.set_ylim(0.3, 1.05); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for wd in WEIGHT_DECAYS:
        ramp_vals = []
        for l in LAYERS:
            pa = results[(l, wd)]["positional_auroc"]
            e = pa.get("early", 0.5) or 0.5
            lt = pa.get("late", 0.5) or 0.5
            ramp_vals.append(lt - e)
        ax.plot(LAYERS, ramp_vals, marker="^", linewidth=2, color=wd_colors[wd], label=f"wd={wd}")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("Late AUROC - Early AUROC")
    ax.set_title("Positional AUROC Ramp (key metric)"); ax.legend(fontsize=9)
    ax.set_xticks(LAYERS); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for wd in WEIGHT_DECAYS:
        vals = [results[(l, wd)]["mean_spearman"] for l in LAYERS]
        ax.plot(LAYERS, vals, marker="D", linewidth=2, color=wd_colors[wd], label=f"wd={wd}")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean Spearman r")
    ax.set_title("Curve Quality (Spearman)"); ax.legend(fontsize=9)
    ax.set_xticks(LAYERS); ax.grid(True, alpha=0.3)

    plt.suptitle("Regularization Sweep: Layer x Weight Decay (Multi-GPU)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "plots", "regsweep_summary.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Per-trace plots for top-3 configs
    top3 = ramps[:3]
    for rank, (ramp, layer, wd) in enumerate(top3):
        cr = results[(layer, wd)]["curve_results"]
        sorted_cr = sorted(cr, key=lambda r: r["spearman_r"], reverse=True)
        sample = sorted_cr[:4] + sorted_cr[-4:]

        fig, axs = plt.subplots(2, 4, figsize=(24, 8))
        for idx, r in enumerate(sample):
            ax = axs.flatten()[idx]
            npos = np.array(r["norm_positions"])
            pvals = np.array(r["p_correct_vals"])
            ax.plot(npos, pvals, lw=0.4, color="#bdc3c7", alpha=0.5)
            w = min(50, len(pvals) // 5)
            if w >= 3:
                sm = np.convolve(pvals, np.ones(w)/w, mode="valid")
                sx = npos[w//2:w//2+len(sm)]
                ax.plot(sx, sm, lw=2.0, color="#2c3e50")
            ax.axhline(0.5, color="gray", ls=":", lw=0.6, alpha=0.5)
            ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, 1)
            ax.set_title(f"{r['problem_id'][:18]}\nrho={r['spearman_r']:.3f}", fontsize=8)
            ax.grid(True, alpha=0.2)
        plt.suptitle(f"#{rank+1}: Layer {layer}, wd={wd} (ramp={ramp:.4f})", fontsize=12, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, "plots", f"regsweep_top{rank+1}_L{layer}_wd{wd}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
