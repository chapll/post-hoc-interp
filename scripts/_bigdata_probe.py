"""Train probe on ALL available traces (not just trace_00).

Uses trace_00, trace_01, trace_02 + any higher seeds that exist.
This gives ~2000+ valid traces instead of 676, much better for 4096-dim probing.

Trains at Layer 20 (best from regsweep) with multiple weight decays.
Generates per-trace plots on held-out traces.
Uses multiple GPUs: extraction on GPU 0, plotting traces on GPU 1.
"""

import gc
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

DATASETS = ["gsm_symbolic", "strategyqa", "folio"]
TRACE_BASE = "src/experimental/gorn/data/traces/valid"
OUTPUT_DIR = "outputs/multilayer_probe/bigdata"
LAYER = 20
WEIGHT_DECAYS = [1e-4, 1e-3, 1e-2, 1e-1]
WINDOW_SIZE = 3
N_FOLDS = 5
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10
SEED = 137
GPU_ID = 0
N_EVAL_PER_DS = 5

POSITION_BINS = [
    ("early", 0.0, 0.2),
    ("early_mid", 0.2, 0.4),
    ("middle", 0.4, 0.6),
    ("late_mid", 0.6, 0.8),
    ("late", 0.8, 1.0),
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
device = torch.device(f"cuda:{GPU_ID}")


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


def discover_all_traces():
    """Find ALL valid traces across all trace indices (00, 01, 02, ...)."""
    all_traces = []
    per_ds_count = {}
    for ds in DATASETS:
        trace_dir = os.path.join(TRACE_BASE, ds)
        if not os.path.isdir(trace_dir):
            continue
        dirs = sorted(d for d in os.listdir(trace_dir)
                      if os.path.isdir(os.path.join(trace_dir, d)))
        ds_count = 0
        for pdir in dirs:
            ppath = os.path.join(trace_dir, pdir)
            # Find all trace files in this problem dir
            trace_files = sorted(f for f in os.listdir(ppath)
                                 if f.startswith("trace_") and f.endswith(".json"))
            for tf in trace_files:
                tp = os.path.join(ppath, tf)
                try:
                    with open(tp) as f:
                        data = json.load(f)
                except:
                    continue
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
                trace_idx = tf.replace("trace_", "").replace(".json", "")
                all_traces.append({
                    "token_ids": tids,
                    "ground_truth": meta["ground_truth"],
                    "problem_id": pdir,
                    "trace_idx": trace_idx,
                    "dataset": ds,
                    "eligible_positions": eligible,
                })
                ds_count += 1
        per_ds_count[ds] = ds_count
    return all_traces, per_ds_count


def main():
    t0 = time.time()
    print("=" * 70)
    print("BIG DATA PROBE -- ALL TRACES")
    print(f"Layer {LAYER}, WDs: {WEIGHT_DECAYS}")
    print("=" * 70)

    print("\nDiscovering ALL traces (trace_00, 01, 02, ...)...")
    all_traces, per_ds_count = discover_all_traces()
    for ds, c in per_ds_count.items():
        yes = sum(1 for t in all_traces if t["dataset"] == ds and t["ground_truth"] == "Yes")
        no = c - yes
        print(f"  {ds}: {c} traces (Yes={yes}, No={no})")
    total_yes = sum(1 for t in all_traces if t["ground_truth"] == "Yes")
    total_no = len(all_traces) - total_yes
    print(f"Total: {len(all_traces)} traces, {len(all_traces)*WINDOW_SIZE} samples "
          f"(Yes={total_yes}, No={total_no})")

    # Split: 80% train, 20% eval (by problem_id to prevent leakage)
    rng = np.random.RandomState(SEED)
    all_pids = sorted(set(t["problem_id"] for t in all_traces))
    rng.shuffle(all_pids)
    n_eval_pids = max(len(all_pids) // 5, 10)
    eval_pids = set(all_pids[:n_eval_pids])
    train_pids = set(all_pids[n_eval_pids:])
    train_traces = [t for t in all_traces if t["problem_id"] in train_pids]
    eval_traces = [t for t in all_traces if t["problem_id"] in eval_pids]
    print(f"\nTrain: {len(train_traces)} traces ({len(train_pids)} problems)")
    print(f"Eval: {len(eval_traces)} traces ({len(eval_pids)} problems)")

    print(f"\nLoading model on GPU {GPU_ID}...")
    model, tok = load_model(gpu_id=GPU_ID)

    # Extract end-of-trace activations for training
    print(f"\nExtracting training activations at layer {LAYER}...")
    X_list, y_list = [], []
    with torch.no_grad():
        for i, trace in enumerate(train_traces):
            window = get_pre_handoff_window(trace["token_ids"], WINDOW_SIZE)
            if window is None:
                continue
            start_pos, end_pos = window
            label = 1 if trace["ground_truth"] == "Yes" else 0
            input_ids = torch.tensor([trace["token_ids"]], dtype=torch.long, device=device)
            with model.trace(input_ids):
                saved = model.model.layers[LAYER].output.save()
            tensor = saved.detach().cpu()
            if tensor.dim() == 3:
                tensor = tensor.squeeze(0)
            X_list.append(tensor[start_pos:end_pos].float())
            y_list.extend([label] * (end_pos - start_pos))
            del saved, input_ids, tensor
            torch.cuda.empty_cache()
            if (i + 1) % 100 == 0 or (i + 1) == len(train_traces):
                mem = torch.cuda.memory_allocated(device) / 1e9
                print(f"  {i+1}/{len(train_traces)} (GPU mem: {mem:.1f} GB)", flush=True)
            if (i + 1) % 200 == 0:
                gc.collect()
    X_train = torch.cat(X_list, 0)
    y_train = np.array(y_list)
    del X_list
    gc.collect()
    print(f"Training data: {X_train.shape[0]} samples "
          f"(Yes={y_train.sum()}, No={len(y_train)-y_train.sum()})")

    # Sweep weight decays
    results = {}
    for wd in WEIGHT_DECAYS:
        print(f"\n--- WD={wd} ---")

        # CV AUROC
        X_gpu = X_train.to(device)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fold_aurocs = []
        for train_idx, val_idx in skf.split(X_train.numpy(), y_train):
            ti = torch.tensor(train_idx, dtype=torch.long, device=device)
            vi = torch.tensor(val_idx, dtype=torch.long, device=device)
            Xt, yt = X_gpu[ti], torch.tensor(y_train[train_idx], dtype=torch.long, device=device)
            Xv, yv = X_gpu[vi], torch.tensor(y_train[val_idx], dtype=torch.long, device=device)
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
        print(f"  End-AUROC (CV): {mean_auroc:.4f}")
        del X_gpu; torch.cuda.empty_cache()

        # Train final probe
        X_gpu = X_train.to(device)
        y_t = torch.tensor(y_train, dtype=torch.long, device=device)
        probe = nn.Linear(HIDDEN_DIM, 2).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=wd)
        crit = nn.CrossEntropyLoss()
        for _ in range(MAX_EPOCHS):
            probe.train(); loss = crit(probe(X_gpu), y_t)
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        del X_gpu, y_t; torch.cuda.empty_cache()

        # Positional AUROC on eval traces
        print(f"  Computing positional AUROC on {len(eval_traces)} eval traces...")
        all_rows = []
        with torch.no_grad():
            for i, trace in enumerate(eval_traces):
                tids = trace["token_ids"]
                eligible = trace["eligible_positions"]
                label = 1 if trace["ground_truth"] == "Yes" else 0
                start = eligible[0]
                span = eligible[-1] - start if eligible[-1] > start else 1
                input_ids = torch.tensor([tids], dtype=torch.long, device=device)
                with model.trace(input_ids):
                    saved = model.model.layers[LAYER].output.save()
                res = saved.detach()
                if res.dim() == 3: res = res.squeeze(0)
                step = max(1, len(eligible) // 50)
                sampled = eligible[::step]
                for pos in sampled:
                    h = res[pos, :].float().unsqueeze(0)
                    p_yes = F.softmax(probe(h), dim=-1)[0, 1].item()
                    all_rows.append(((pos - start) / span, p_yes, label))
                del saved, input_ids, res; torch.cuda.empty_cache()
                if (i + 1) % 50 == 0 or (i + 1) == len(eval_traces):
                    print(f"    PosAUROC: {i+1}/{len(eval_traces)}", flush=True)
                if (i + 1) % 100 == 0:
                    gc.collect()

        pos_aurocs = {}
        for bn, lo, hi in POSITION_BINS:
            rows = [(p, l) for (np_, p, l) in all_rows if lo <= np_ < hi]
            if len(rows) < 10:
                pos_aurocs[bn] = None; continue
            pa = np.array([r[0] for r in rows])
            la = np.array([r[1] for r in rows])
            pos_aurocs[bn] = float(roc_auc_score(la, pa)) if len(np.unique(la)) >= 2 else 0.5
        early = pos_aurocs.get("early", 0.5) or 0.5
        late = pos_aurocs.get("late", 0.5) or 0.5
        ramp = late - early
        pos_str = " | ".join(f"{k}={v:.4f}" if v else f"{k}=N/A"
                             for k, v in pos_aurocs.items())
        print(f"  Positional: {pos_str}  RAMP={ramp:.4f}")

        # Per-trace curves on subset of eval traces
        eval_subset = []
        for ds in DATASETS:
            dst = [t for t in eval_traces if t["dataset"] == ds]
            eval_subset.extend(dst[:N_EVAL_PER_DS])

        curves = []
        with torch.no_grad():
          for trace in eval_subset:
            tids = trace["token_ids"]
            eligible = trace["eligible_positions"]
            gt = trace["ground_truth"]
            start = eligible[0]
            span = eligible[-1] - start if eligible[-1] > start else 1
            input_ids = torch.tensor([tids], dtype=torch.long, device=device)
            with model.trace(input_ids):
                saved = model.model.layers[LAYER].output.save()
            res = saved.detach()
            if res.dim() == 3: res = res.squeeze(0)
            npos, pvals = [], []
            for pos in eligible:
                h = res[pos, :].float().unsqueeze(0)
                p_yes = F.softmax(probe(h), dim=-1)[0, 1].item()
                pc = p_yes if gt == "Yes" else (1.0 - p_yes)
                npos.append((pos - start) / span)
                pvals.append(pc)
            del saved, input_ids, res; torch.cuda.empty_cache()
            curves.append({
                "problem_id": trace["problem_id"], "dataset": trace["dataset"],
                "ground_truth": gt, "trace_idx": trace["trace_idx"],
                "norm_positions": np.array(npos), "p_correct": np.array(pvals),
            })

        # Plot
        by_ds = {}
        for c in curves:
            by_ds.setdefault(c["dataset"], []).append(c)
        n_cols = N_EVAL_PER_DS
        n_rows = len(DATASETS)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows))
        if n_rows == 1: axes = axes[np.newaxis, :]
        for row, ds in enumerate(DATASETS):
            ds_curves = by_ds.get(ds, [])
            for col in range(n_cols):
                ax = axes[row, col]
                if col >= len(ds_curves):
                    ax.set_visible(False); continue
                c = ds_curves[col]
                npos = c["norm_positions"]
                pvals = c["p_correct"]
                ax.plot(npos, pvals, lw=0.3, color="#bdc3c7", alpha=0.5)
                w = min(50, len(pvals) // 5)
                if w >= 3:
                    sm = np.convolve(pvals, np.ones(w)/w, mode="valid")
                    sx = npos[w//2:w//2+len(sm)]
                    ax.plot(sx, sm, lw=2.0, color="#2c3e50")
                ax.axhline(0.5, color="gray", ls=":", lw=0.6, alpha=0.5)
                ax.set_ylim(-0.05, 1.05); ax.set_xlim(0, 1)
                ax.set_title(f"{c['problem_id'][:18]}_{c['trace_idx']}\nGT={c['ground_truth']}",
                             fontsize=7)
                ax.grid(True, alpha=0.2)
                if col == 0: ax.set_ylabel(ds, fontsize=9, fontweight="bold")
        fig.suptitle(f"BIG DATA -- L{LAYER} wd={wd} -- {len(train_traces)} traces\n"
                     f"AUROC={mean_auroc:.4f}, ramp={ramp:.4f}", fontsize=12, y=1.02)
        plt.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, "plots", f"bigdata_L{LAYER}_wd{wd}.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath}")

        # Save probe
        save_file({"weight": probe.weight.data.cpu(), "bias": probe.bias.data.cpu()},
                  os.path.join(OUTPUT_DIR, f"bigdata_L{LAYER}_wd{wd}.safetensors"))

        results[wd] = {
            "end_auroc": mean_auroc, "positional_auroc": pos_aurocs, "ramp": ramp,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for wd in WEIGHT_DECAYS:
        r = results[wd]
        pa = r["positional_auroc"]
        print(f"  wd={wd}: end_auroc={r['end_auroc']:.4f}, ramp={r['ramp']:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
