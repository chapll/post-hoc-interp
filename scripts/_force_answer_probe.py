"""Train probe on INTERMEDIATE positions using force-answer labels.

Key difference from Cox probe: instead of training on end-of-trace with Yes/No labels,
we train on intermediate positions with "model correct at this position" labels from
force-answer experiments.

Steps:
1. Load force-answer results (51 traces x 20 positions = 1020 labeled points)
2. For each position, extract residual stream activation at that layer
3. Label: 1 if model answered correctly at that position, 0 if not
4. Train probe with 5-fold CV (stratified by trace to prevent leakage)
5. Apply to held-out traces and plot per-trace P(correct) curves

This is NOT circular if we hold out entire traces (not positions within the same trace).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file
from sklearn.metrics import roc_auc_score

from src.model import HIDDEN_DIM, load_model
from src.tokens import count_think_blocks, get_think_content_mask
from src.utils import set_seed

set_seed(137)

DATASETS = ["gsm_symbolic", "strategyqa", "folio", "deepmath"]
TRACE_BASE = "src/experimental/gorn/data/traces/valid"
FA_PARQUET = "src/experimental/gorn/experiments/force_answer/results.parquet"
OUTPUT_DIR = "outputs/multilayer_probe/force_answer_probe"
LAYERS = [16, 20, 24]  # top layers from regsweep
WEIGHT_DECAYS = [1e-3, 1e-2, 1e-1]
LR = 1e-3
MAX_EPOCHS = 200
PATIENCE = 15
GPU_ID = 0
N_EVAL_TRACES = 5  # per dataset, for plotting

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
device = torch.device(f"cuda:{GPU_ID}")


def main():
    print("=" * 70)
    print("FORCE-ANSWER PROBE: Train on intermediate positions")
    print("=" * 70)

    # Load force-answer data
    df = pd.read_parquet(FA_PARQUET)
    print(f"\nForce-answer data: {len(df)} rows, {df['trace_id'].nunique()} traces")
    print(f"Correct rate: {df['correct'].mean():.3f}")

    # Load trace token_ids for each force-answer trace
    trace_data = {}
    for tid in df["trace_id"].unique():
        row = df[df["trace_id"] == tid].iloc[0]
        ds = row["dataset"]
        pid = row["problem_id"]
        tp = os.path.join(TRACE_BASE, ds, pid, "trace_00.json")
        if not os.path.isfile(tp):
            print(f"  WARNING: missing trace file for {tid}")
            continue
        with open(tp) as f:
            data = json.load(f)
        tids = data["token_ids"]
        if count_think_blocks(tids) != 1:
            continue
        mask = get_think_content_mask(tids)
        eligible = [i for i, m in enumerate(mask) if m]
        if len(eligible) < 10:
            continue
        trace_data[tid] = {
            "token_ids": tids,
            "eligible_positions": eligible,
            "think_start": eligible[0],
            "think_length": len(eligible),
            "dataset": ds,
            "problem_id": pid,
            "ground_truth": row["ground_truth"],
        }
    print(f"Loaded {len(trace_data)} traces with token data")

    # Filter df to traces we have data for
    df = df[df["trace_id"].isin(trace_data)].copy()
    print(f"After filtering: {len(df)} rows, {df['trace_id'].nunique()} traces")

    # Map probe_position to absolute token position
    # probe_position is 0-indexed relative to think content start
    df["abs_position"] = df.apply(
        lambda r: trace_data[r["trace_id"]]["think_start"] + int(r["probe_position"]),
        axis=1
    )

    print(f"\nLoading model on GPU {GPU_ID}...")
    model, tok = load_model(gpu_id=GPU_ID)
    print("Model loaded.")

    # For each layer: extract activations at force-answer positions
    for layer in LAYERS:
        print(f"\n{'='*60}")
        print(f"LAYER {layer}")
        print(f"{'='*60}")

        # Extract activations at each (trace, position) pair
        print(f"  Extracting activations at {len(df)} positions...")
        X_list = []
        y_list = []
        trace_ids = []
        positions = []

        traces_processed = set()
        for tid, group in df.groupby("trace_id"):
            td = trace_data[tid]
            input_ids = torch.tensor([td["token_ids"]], dtype=torch.long, device=device)
            with model.trace(input_ids):
                saved = model.model.layers[layer].output.save()
            res = saved.detach()
            if res.dim() == 3:
                res = res.squeeze(0)

            for _, row in group.iterrows():
                pos = int(row["abs_position"])
                if pos >= res.shape[0]:
                    continue
                h = res[pos, :].float().cpu()
                X_list.append(h)
                y_list.append(1 if row["correct"] else 0)
                trace_ids.append(tid)
                positions.append(row["normalized_position"])

            del saved, input_ids, res
            torch.cuda.empty_cache()
            traces_processed.add(tid)
            if len(traces_processed) % 10 == 0:
                print(f"    {len(traces_processed)}/{df['trace_id'].nunique()} traces", flush=True)

        X = torch.stack(X_list, 0)  # [N, 4096]
        y = np.array(y_list)
        trace_id_arr = np.array(trace_ids)
        pos_arr = np.array(positions)
        print(f"  Extracted: {X.shape[0]} samples, correct={y.mean():.3f}")

        # Train with trace-level CV (hold out entire traces, not individual positions)
        unique_traces = np.array(sorted(set(trace_ids)))
        n_folds = 5
        rng = np.random.RandomState(137)
        rng.shuffle(unique_traces)
        fold_size = len(unique_traces) // n_folds

        for wd in WEIGHT_DECAYS:
            print(f"\n  --- WD={wd} ---")
            fold_aurocs = []

            for fold in range(n_folds):
                start = fold * fold_size
                end = start + fold_size if fold < n_folds - 1 else len(unique_traces)
                val_traces = set(unique_traces[start:end])
                train_mask = np.array([t not in val_traces for t in trace_id_arr])
                val_mask = ~train_mask

                Xt = X[train_mask].to(device)
                yt = torch.tensor(y[train_mask], dtype=torch.long, device=device)
                Xv = X[val_mask].to(device)
                yv = y[val_mask]

                p = nn.Linear(HIDDEN_DIM, 2).to(device)
                opt = torch.optim.Adam(p.parameters(), lr=LR, weight_decay=wd)
                crit = nn.CrossEntropyLoss()
                best_vl, best_st, pat = float("inf"), None, 0
                yv_t = torch.tensor(yv, dtype=torch.long, device=device)
                for _ in range(MAX_EPOCHS):
                    p.train(); loss = crit(p(Xt), yt)
                    opt.zero_grad(); loss.backward(); opt.step()
                    p.eval()
                    with torch.no_grad():
                        vl = crit(p(Xv), yv_t).item()
                    if vl < best_vl:
                        best_vl = vl
                        best_st = {k: v.clone() for k, v in p.state_dict().items()}
                        pat = 0
                    else:
                        pat += 1
                        if pat >= PATIENCE:
                            break
                if best_st:
                    p.load_state_dict(best_st)
                p.eval()
                with torch.no_grad():
                    vp = torch.softmax(p(Xv), dim=-1)[:, 1].cpu().numpy()
                if len(np.unique(yv)) >= 2:
                    fold_aurocs.append(roc_auc_score(yv, vp))
                else:
                    fold_aurocs.append(0.5)
                del Xt, yt, Xv, yv_t
                torch.cuda.empty_cache()

            mean_auroc = float(np.mean(fold_aurocs))
            print(f"  CV AUROC (trace-held-out): {mean_auroc:.4f} ({fold_aurocs})")

            # Train final probe on ALL force-answer data
            X_gpu = X.to(device)
            y_t = torch.tensor(y, dtype=torch.long, device=device)
            probe = nn.Linear(HIDDEN_DIM, 2).to(device)
            opt = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=wd)
            crit = nn.CrossEntropyLoss()
            for _ in range(MAX_EPOCHS):
                probe.train(); loss = crit(probe(X_gpu), y_t)
                opt.zero_grad(); loss.backward(); opt.step()
            probe.eval()
            del X_gpu, y_t
            torch.cuda.empty_cache()

            # Save probe
            tensors = {
                "weight": probe.weight.data.cpu(),
                "bias": probe.bias.data.cpu(),
            }
            save_file(tensors, os.path.join(OUTPUT_DIR,
                      f"fa_probe_L{layer}_wd{wd}.safetensors"))

            # Apply to ALL traces (not just force-answer) and plot per-trace curves
            # Pick N_EVAL_TRACES per dataset from NON-force-answer traces
            fa_pids = set(df["problem_id"].unique())
            eval_traces = []
            for ds in DATASETS:
                trace_dir = os.path.join(TRACE_BASE, ds)
                if not os.path.isdir(trace_dir):
                    continue
                dirs = sorted(d for d in os.listdir(trace_dir)
                              if os.path.isdir(os.path.join(trace_dir, d)))
                count = 0
                for pdir in dirs:
                    if pdir in fa_pids:
                        continue  # skip force-answer traces
                    tp = os.path.join(trace_dir, pdir, "trace_00.json")
                    if not os.path.isfile(tp):
                        continue
                    with open(tp) as f:
                        data = json.load(f)
                    tids_t = data["token_ids"]
                    if count_think_blocks(tids_t) != 1:
                        continue
                    mask = get_think_content_mask(tids_t)
                    eligible = [i for i, m in enumerate(mask) if m]
                    if len(eligible) < 10:
                        continue
                    meta = data.get("metadata", {})
                    gt = meta.get("ground_truth")
                    if gt is None:
                        continue
                    eval_traces.append({
                        "token_ids": tids_t, "ground_truth": gt,
                        "problem_id": pdir, "dataset": ds,
                        "eligible_positions": eligible,
                    })
                    count += 1
                    if count >= N_EVAL_TRACES:
                        break

            print(f"  Plotting {len(eval_traces)} held-out traces...")

            # Plot per-trace curves
            n_cols = N_EVAL_TRACES
            n_rows = len(DATASETS)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
            if n_rows == 1:
                axes = axes[np.newaxis, :]

            by_ds = {ds: [] for ds in DATASETS}
            for trace in eval_traces:
                tids_t = trace["token_ids"]
                eligible = trace["eligible_positions"]
                gt = trace["ground_truth"]
                start = eligible[0]
                span = eligible[-1] - start if eligible[-1] > start else 1

                input_ids = torch.tensor([tids_t], dtype=torch.long, device=device)
                with model.trace(input_ids):
                    saved = model.model.layers[layer].output.save()
                res = saved.detach()
                if res.dim() == 3:
                    res = res.squeeze(0)

                npos, pvals = [], []
                with torch.no_grad():
                    for pos in eligible:
                        h = res[pos, :].float().unsqueeze(0).to(device)
                        p_correct = F.softmax(probe(h), dim=-1)[0, 1].item()
                        npos.append((pos - start) / span)
                        pvals.append(p_correct)
                del saved, input_ids, res
                torch.cuda.empty_cache()

                by_ds[trace["dataset"]].append({
                    "problem_id": trace["problem_id"],
                    "ground_truth": gt,
                    "norm_positions": np.array(npos),
                    "p_correct": np.array(pvals),
                })

            for row, ds in enumerate(DATASETS):
                ds_curves = by_ds.get(ds, [])
                for col in range(n_cols):
                    ax = axes[row, col]
                    if col >= len(ds_curves):
                        ax.set_visible(False)
                        continue
                    c = ds_curves[col]
                    npos = c["norm_positions"]
                    pvals = c["p_correct"]
                    ax.plot(npos, pvals, lw=0.3, color="#bdc3c7", alpha=0.5)
                    w_size = min(50, len(pvals) // 5)
                    if w_size >= 3:
                        sm = np.convolve(pvals, np.ones(w_size)/w_size, mode="valid")
                        sx = npos[w_size//2:w_size//2+len(sm)]
                        ax.plot(sx, sm, lw=2.0, color="#2c3e50")
                    ax.axhline(0.5, color="gray", ls=":", lw=0.6, alpha=0.5)
                    ax.set_ylim(-0.05, 1.05)
                    ax.set_xlim(0, 1)
                    ax.set_title(f"{c['problem_id'][:20]}\nGT={c['ground_truth']}",
                                 fontsize=7)
                    ax.grid(True, alpha=0.2)
                    if col == 0:
                        ax.set_ylabel(ds, fontsize=9, fontweight="bold")

            fig.suptitle(f"FORCE-ANSWER PROBE -- L{layer} wd={wd}\n"
                         f"CV AUROC={mean_auroc:.4f} (trained on intermediate positions)",
                         fontsize=12, y=1.02)
            plt.tight_layout()
            outpath = os.path.join(OUTPUT_DIR, "plots",
                                   f"fa_probe_L{layer}_wd{wd}.png")
            fig.savefig(outpath, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {outpath}")

    print("\nDone.")


if __name__ == "__main__":
    main()
