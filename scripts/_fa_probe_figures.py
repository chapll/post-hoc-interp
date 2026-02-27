"""Generate publication figures for force-answer label probe (Layer 20).

Figures:
1. Individual per-trace P(correct) vs raw token position (separate plots, ~10 GSM fresh traces)
2. Held-out AUROC evaluation (CV AUROC bar chart)
3. Positional accuracy curve (accuracy vs normalized position)
4. ROC curve on held-out traces
5. Probe confidence calibration (predicted P(correct) vs actual fraction correct)
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
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, roc_curve

from src.model import HIDDEN_DIM, load_model
from src.tokens import count_think_blocks, get_think_content_mask
from src.utils import set_seed

set_seed(137)

DATASETS = ["gsm_symbolic", "strategyqa", "folio"]
TRACE_BASE = "src/experimental/gorn/data/traces/valid"
FA_PARQUET = "src/experimental/gorn/experiments/force_answer/results.parquet"
PROBE_DIR = "outputs/multilayer_probe/force_answer_probe"
OUTPUT_DIR = "outputs/multilayer_probe/fa_figures"
LAYER = 20
WD = 0.001  # best CV AUROC from force_answer_probe run
GPU_ID = 2  # free on ampere4

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device(f"cuda:{GPU_ID}")


def main():
    print("=" * 70, flush=True)
    print("FORCE-ANSWER PROBE FIGURES -- Layer 20", flush=True)
    print("=" * 70, flush=True)

    # Load force-answer data
    df = pd.read_parquet(FA_PARQUET)
    print(f"Force-answer data: {len(df)} rows, {df['trace_id'].nunique()} traces", flush=True)

    # Load trace token_ids
    trace_data = {}
    for tid in df["trace_id"].unique():
        row = df[df["trace_id"] == tid].iloc[0]
        ds = row["dataset"]
        pid = row["problem_id"]
        tp = os.path.join(TRACE_BASE, ds, pid, "trace_00.json")
        if not os.path.isfile(tp):
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
    df = df[df["trace_id"].isin(trace_data)].copy()
    df["abs_position"] = df.apply(
        lambda r: trace_data[r["trace_id"]]["think_start"] + int(r["probe_position"]),
        axis=1
    )
    print(f"Traces with data: {len(trace_data)}", flush=True)

    # Load model and probe
    print(f"\nLoading model on GPU {GPU_ID}...", flush=True)
    model, tok = load_model(gpu_id=GPU_ID)

    probe_path = os.path.join(PROBE_DIR, f"fa_probe_L{LAYER}_wd{WD}.safetensors")
    tensors = load_file(probe_path)
    probe = nn.Linear(HIDDEN_DIM, 2).to(device)
    probe.weight.data = tensors["weight"].to(device)
    probe.bias.data = tensors["bias"].to(device)
    probe.eval()
    print(f"Probe loaded: {probe_path}", flush=True)

    # =========================================================================
    # FIGURE 1: Per-trace P(correct) vs raw token position -- GSM fresh traces
    # =========================================================================
    print("\n--- Figure 1: Per-trace curves (GSM fresh) ---", flush=True)

    gsm_fresh = df[(df["dataset"] == "gsm_symbolic") & (df["is_fresh"] == True)]
    gsm_fresh_tids = sorted(gsm_fresh["trace_id"].unique())
    # Pick traces with interesting S-curve behavior (varying correct rates)
    # Sort by correct rate to get a range
    tid_rates = []
    for tid in gsm_fresh_tids:
        sub = gsm_fresh[gsm_fresh["trace_id"] == tid]
        tid_rates.append((tid, sub["correct"].mean(), sub["think_length"].iloc[0]))
    tid_rates.sort(key=lambda x: x[1])
    # Pick ~10: mix of low, medium, high correct rates
    selected_tids = [t[0] for t in tid_rates][:11]  # take all GSM fresh (11 traces)

    for tid in selected_tids:
        sub = gsm_fresh[gsm_fresh["trace_id"] == tid].sort_values("probe_position")
        td = trace_data[tid]

        # Get probe predictions at ALL eligible positions (not just 20 force-answer positions)
        input_ids = torch.tensor([td["token_ids"]], dtype=torch.long, device=device)
        with torch.no_grad():
            with model.trace(input_ids):
                saved = model.model.layers[LAYER].output.save()
            res = saved.detach()
            if res.dim() == 3:
                res = res.squeeze(0)

            positions = []
            p_correct_vals = []
            for pos in td["eligible_positions"]:
                h = res[pos, :].float().unsqueeze(0)
                logits = probe(h)
                p_yes = F.softmax(logits, dim=-1)[0, 1].item()
                pc = p_yes if td["ground_truth"] == "Yes" else (1.0 - p_yes)
                # Raw token position relative to think start
                raw_pos = pos - td["think_start"]
                positions.append(raw_pos)
                p_correct_vals.append(pc)

        del saved, input_ids, res
        torch.cuda.empty_cache()

        positions = np.array(positions)
        p_correct_vals = np.array(p_correct_vals)

        # Force-answer ground truth at the 20 probed positions
        fa_positions = sub["probe_position"].values.astype(int)
        fa_correct = sub["correct"].values.astype(int)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4.5))

        # Probe P(correct) curve
        ax.plot(positions, p_correct_vals, lw=0.4, color="#bdc3c7", alpha=0.5, zorder=1)
        # Smoothed
        w = min(50, len(p_correct_vals) // 5)
        if w >= 3:
            sm = np.convolve(p_correct_vals, np.ones(w) / w, mode="valid")
            sx = positions[w // 2:w // 2 + len(sm)]
            ax.plot(sx, sm, lw=2.5, color="#2c3e50", zorder=2, label="Probe P(correct)")

        ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlim(0, td["think_length"])
        ax.set_xlabel("Token Position (within think block)", fontsize=11)
        ax.set_ylabel("P(correct)", fontsize=11)
        ax.set_title(f"{tid}\nGT={td['ground_truth']}, think_length={td['think_length']}",
                     fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.15)
        plt.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, f"pertrace_{tid}.png")
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath}", flush=True)

    # =========================================================================
    # FIGURE 2: Held-out AUROC (trace-level CV)
    # =========================================================================
    print("\n--- Figure 2: CV AUROC ---", flush=True)

    # Extract activations at force-answer positions
    X_list, y_list, trace_ids, norm_positions, problem_ids = [], [], [], [], []
    with torch.no_grad():
        for tid, group in df.groupby("trace_id"):
            td = trace_data[tid]
            input_ids = torch.tensor([td["token_ids"]], dtype=torch.long, device=device)
            with model.trace(input_ids):
                saved = model.model.layers[LAYER].output.save()
            res = saved.detach()
            if res.dim() == 3:
                res = res.squeeze(0)
            for _, row in group.iterrows():
                pos = td["think_start"] + int(row["probe_position"])
                if pos >= res.shape[0]:
                    continue
                h = res[pos, :].float().cpu()
                X_list.append(h)
                y_list.append(1 if row["correct"] else 0)
                trace_ids.append(tid)
                norm_positions.append(row["normalized_position"])
                problem_ids.append(td["problem_id"])
            del saved, input_ids, res
            torch.cuda.empty_cache()

    X = torch.stack(X_list, 0)
    y = np.array(y_list)
    trace_id_arr = np.array(trace_ids)
    norm_pos_arr = np.array(norm_positions)
    problem_id_arr = np.array(problem_ids)
    print(f"  Extracted: {len(X)} samples, correct={y.mean():.3f}", flush=True)

    # 5-fold problem-level CV (split by problem_id to prevent leakage
    # between config A/B traces of the same problem)
    unique_problems = np.array(sorted(set(problem_ids)))
    print(f"  Splitting by problem_id: {len(unique_problems)} unique problems", flush=True)
    rng = np.random.RandomState(137)
    rng.shuffle(unique_problems)
    n_folds = 5
    fold_size = len(unique_problems) // n_folds

    all_pred_probs = np.zeros(len(y))
    all_pred_assigned = np.zeros(len(y), dtype=bool)
    fold_aurocs = []

    for fold in range(n_folds):
        start = fold * fold_size
        end = start + fold_size if fold < n_folds - 1 else len(unique_problems)
        val_problems = set(unique_problems[start:end])
        val_mask = np.array([p in val_problems for p in problem_id_arr])
        train_mask = ~val_mask

        Xt = X[train_mask].to(device)
        yt = torch.tensor(y[train_mask], dtype=torch.long, device=device)

        p = nn.Linear(HIDDEN_DIM, 2).to(device)
        opt = torch.optim.Adam(p.parameters(), lr=1e-3, weight_decay=WD)
        crit = nn.CrossEntropyLoss()
        best_vl, best_st, pat = float("inf"), None, 0
        yv_t = torch.tensor(y[val_mask], dtype=torch.long, device=device)
        Xv = X[val_mask].to(device)
        for _ in range(200):
            p.train()
            loss = crit(p(Xt), yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            p.eval()
            with torch.no_grad():
                vl = crit(p(Xv), yv_t).item()
            if vl < best_vl:
                best_vl = vl
                best_st = {k: v.clone() for k, v in p.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= 15:
                    break
        if best_st:
            p.load_state_dict(best_st)
        p.eval()
        with torch.no_grad():
            vp = torch.softmax(p(Xv), dim=-1)[:, 1].cpu().numpy()
        yv = y[val_mask]
        auc = roc_auc_score(yv, vp) if len(np.unique(yv)) >= 2 else 0.5
        fold_aurocs.append(auc)
        all_pred_probs[val_mask] = vp
        all_pred_assigned[val_mask] = True
        del Xt, yt, Xv, yv_t
        torch.cuda.empty_cache()

    mean_auroc = np.mean(fold_aurocs)
    print(f"  CV AUROC: {mean_auroc:.4f} (folds: {fold_aurocs})", flush=True)

    # Bar chart of fold AUROCs
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(1, n_folds + 1), fold_aurocs, color="#3498db", alpha=0.8,
                  edgecolor="white", linewidth=1.5)
    ax.axhline(mean_auroc, color="#e74c3c", ls="--", lw=2,
               label=f"Mean AUROC = {mean_auroc:.4f}")
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5, label="Chance")
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title(f"Force-Answer Probe -- Trace-Held-Out CV AUROC\nLayer {LAYER}, wd={WD}",
                 fontsize=11)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15, axis="y")
    for i, v in enumerate(fold_aurocs):
        ax.text(i + 1, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "cv_auroc.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: cv_auroc.png", flush=True)

    # =========================================================================
    # FIGURE 3: ROC curve on all held-out predictions
    # =========================================================================
    print("\n--- Figure 3: ROC curve ---", flush=True)

    mask = all_pred_assigned
    fpr, tpr, _ = roc_curve(y[mask], all_pred_probs[mask])
    overall_auc = roc_auc_score(y[mask], all_pred_probs[mask])

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, lw=2.5, color="#2c3e50", label=f"ROC (AUC = {overall_auc:.4f})")
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"Force-Answer Probe ROC -- Layer {LAYER}\n(Trace-held-out CV predictions)",
                 fontsize=11)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.15)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: roc_curve.png", flush=True)

    # =========================================================================
    # FIGURE 4: Accuracy vs normalized position
    # =========================================================================
    print("\n--- Figure 4: Accuracy vs position ---", flush=True)

    bins = np.linspace(0, 1, 11)  # 10 bins
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs = []
    bin_counts = []
    bin_aurocs = []
    for i in range(len(bins) - 1):
        if i == len(bins) - 2:
            mask_bin = (norm_pos_arr >= bins[i]) & (norm_pos_arr <= bins[i + 1]) & all_pred_assigned
        else:
            mask_bin = (norm_pos_arr >= bins[i]) & (norm_pos_arr < bins[i + 1]) & all_pred_assigned
        if mask_bin.sum() < 5:
            bin_accs.append(np.nan)
            bin_counts.append(0)
            bin_aurocs.append(np.nan)
            continue
        pred_labels = (all_pred_probs[mask_bin] > 0.5).astype(int)
        true_labels = y[mask_bin]
        bin_accs.append(np.mean(pred_labels == true_labels))
        bin_counts.append(mask_bin.sum())
        if len(np.unique(true_labels)) >= 2:
            bin_aurocs.append(roc_auc_score(true_labels, all_pred_probs[mask_bin]))
        else:
            bin_aurocs.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Accuracy
    ax1.bar(bin_centers, bin_accs, width=0.08, color="#3498db", alpha=0.8,
            edgecolor="white", linewidth=1)
    ax1.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
    ax1.set_xlabel("Normalized Position in Think Block", fontsize=11)
    ax1.set_ylabel("Probe Accuracy", fontsize=11)
    ax1.set_title("Probe Accuracy vs Position", fontsize=11)
    ax1.set_ylim(0.3, 1.0)
    ax1.grid(True, alpha=0.15, axis="y")
    # Add count annotations
    for i, (c, acc) in enumerate(zip(bin_counts, bin_accs)):
        if not np.isnan(acc):
            ax1.text(bin_centers[i], acc + 0.02, f"n={c}", ha="center", fontsize=7)

    # AUROC per bin
    valid = ~np.isnan(bin_aurocs)
    ax2.plot(np.array(bin_centers)[valid], np.array(bin_aurocs)[valid],
             "o-", color="#e74c3c", lw=2, markersize=8)
    ax2.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
    ax2.set_xlabel("Normalized Position in Think Block", fontsize=11)
    ax2.set_ylabel("AUROC", fontsize=11)
    ax2.set_title("Probe AUROC vs Position", fontsize=11)
    ax2.set_ylim(0.3, 1.0)
    ax2.grid(True, alpha=0.15)

    fig.suptitle(f"Force-Answer Probe -- Layer {LAYER}, wd={WD}", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_position.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: accuracy_vs_position.png", flush=True)

    # =========================================================================
    # FIGURE 5: Calibration plot
    # =========================================================================
    print("\n--- Figure 5: Calibration ---", flush=True)

    mask = all_pred_assigned
    pred = all_pred_probs[mask]
    true = y[mask]
    n_cal_bins = 10
    cal_bins = np.linspace(0, 1, n_cal_bins + 1)
    cal_pred_mean = []
    cal_true_frac = []
    cal_counts = []
    for i in range(n_cal_bins):
        in_bin = (pred >= cal_bins[i]) & (pred < cal_bins[i + 1])
        if in_bin.sum() == 0:
            cal_pred_mean.append(np.nan)
            cal_true_frac.append(np.nan)
            cal_counts.append(0)
        else:
            cal_pred_mean.append(pred[in_bin].mean())
            cal_true_frac.append(true[in_bin].mean())
            cal_counts.append(in_bin.sum())

    fig, ax = plt.subplots(figsize=(5.5, 5))
    valid = [i for i in range(n_cal_bins) if not np.isnan(cal_pred_mean[i])]
    ax.plot([cal_pred_mean[i] for i in valid], [cal_true_frac[i] for i in valid],
            "o-", color="#2c3e50", lw=2, markersize=8)
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1, alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted P(correct)", fontsize=11)
    ax.set_ylabel("Observed Fraction Correct", fontsize=11)
    ax.set_title(f"Calibration -- Force-Answer Probe L{LAYER}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "calibration.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: calibration.png", flush=True)

    print(f"\nAll figures saved to {OUTPUT_DIR}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
