"""PCA + linear probe: reduce dimensionality before probing.

Rationale: 4096 dims with ~6500 samples means the linear probe has many
noise directions to overfit. PCA concentrates variance into fewer dimensions,
potentially improving per-trace generalization.

Approach:
1. Compute PCA on ALL training activations (end-of-trace + intermediate)
2. Project to top K PCs
3. Train linear probe in reduced space
4. Apply to intermediate positions (also projected)

Uses same data pipeline as bigdata probe.
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
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.model import HIDDEN_DIM, load_model
from src.tokens import count_think_blocks, get_pre_handoff_window, get_think_content_mask
from src.utils import set_seed

set_seed(137)

DATASETS = ["gsm_symbolic", "strategyqa", "folio"]
TRACE_BASE = "src/experimental/gorn/data/traces/valid"
OUTPUT_DIR = "outputs/multilayer_probe/pca_probe"
LAYER = 20
WINDOW_SIZE = 3
SEED = 137
GPU_ID = 1  # Use GPU 1 since MLP is on GPU 0
N_EVAL_PER_DS = 8

LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 10

POSITION_BINS = [
    ("early", 0.0, 0.2),
    ("early_mid", 0.2, 0.4),
    ("middle", 0.4, 0.6),
    ("late_mid", 0.6, 0.8),
    ("late", 0.8, 1.0),
]

# PCA dimensions to try
PCA_DIMS = [16, 32, 64, 128, 256]
WEIGHT_DECAY = 0.01

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
            trace_files = sorted(f for f in os.listdir(ppath)
                                 if f.startswith("trace_") and f.endswith(".json"))
            for tf in trace_files:
                tp = os.path.join(ppath, tf)
                try:
                    with open(tp) as f:
                        data = json.load(f)
                except Exception:
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
    print("=" * 70, flush=True)
    print("PCA + LINEAR PROBE", flush=True)
    print(f"Layer {LAYER}, PCA dims: {PCA_DIMS}", flush=True)
    print("=" * 70, flush=True)

    print("\nDiscovering ALL traces...", flush=True)
    all_traces, per_ds_count = discover_all_traces()
    for ds, c in per_ds_count.items():
        print(f"  {ds}: {c} traces", flush=True)
    print(f"Total: {len(all_traces)} traces", flush=True)

    # Split by problem_id
    rng = np.random.RandomState(SEED)
    all_pids = sorted(set(t["problem_id"] for t in all_traces))
    rng.shuffle(all_pids)
    n_eval_pids = max(len(all_pids) // 5, 10)
    eval_pids = set(all_pids[:n_eval_pids])
    train_traces = [t for t in all_traces if t["problem_id"] not in eval_pids]
    eval_traces = [t for t in all_traces if t["problem_id"] in eval_pids]
    print(f"Train: {len(train_traces)} traces, Eval: {len(eval_traces)} traces", flush=True)

    print(f"\nLoading model on GPU {GPU_ID}...", flush=True)
    model, tok = load_model(gpu_id=GPU_ID)

    # Extract training activations (end-of-trace)
    print(f"\nExtracting training activations at layer {LAYER}...", flush=True)
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
    X_train = torch.cat(X_list, 0).numpy()
    y_train = np.array(y_list)
    del X_list; gc.collect()
    print(f"Training data: {X_train.shape[0]} samples", flush=True)

    # Fit PCA on training data
    print(f"PCA fitting on {X_train.shape[0]} training samples...", flush=True)

    # Select eval subset for plotting
    eval_subset = []
    for ds in DATASETS:
        dst = [t for t in eval_traces if t["dataset"] == ds]
        eval_subset.extend(dst[:N_EVAL_PER_DS])

    results = {}
    for n_components in PCA_DIMS:
        print(f"\n{'='*60}", flush=True)
        print(f"PCA dimensions: {n_components}", flush=True)
        print(f"{'='*60}", flush=True)

        # Fit PCA
        pca = PCA(n_components=n_components, random_state=SEED)
        pca.fit(X_train)
        var_explained = pca.explained_variance_ratio_.sum()
        print(f"  Variance explained: {var_explained:.4f}", flush=True)

        # Transform training data
        X_train_pca = pca.transform(X_train)
        X_train_t = torch.tensor(X_train_pca, dtype=torch.float32)

        # 5-fold CV
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_aurocs = []
        for train_idx, val_idx in skf.split(X_train_pca, y_train):
            Xt = torch.tensor(X_train_pca[train_idx], dtype=torch.float32, device=device)
            yt = torch.tensor(y_train[train_idx], dtype=torch.long, device=device)
            Xv = torch.tensor(X_train_pca[val_idx], dtype=torch.float32, device=device)
            yv = y_train[val_idx]

            p = nn.Linear(n_components, 2).to(device)
            opt = torch.optim.Adam(p.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            crit = nn.CrossEntropyLoss()
            best_vl, best_st, pat = float("inf"), None, 0
            for _ in range(MAX_EPOCHS):
                p.train(); loss = crit(p(Xt), yt)
                opt.zero_grad(); loss.backward(); opt.step()
                p.eval()
                with torch.no_grad():
                    yv_t = torch.tensor(yv, dtype=torch.long, device=device)
                    vl = crit(p(Xv), yv_t).item()
                if vl < best_vl:
                    best_vl = vl; best_st = {k: v.clone() for k, v in p.state_dict().items()}; pat = 0
                else:
                    pat += 1
                    if pat >= PATIENCE: break
            if best_st: p.load_state_dict(best_st)
            p.eval()
            with torch.no_grad():
                vp = torch.softmax(p(Xv), dim=-1)[:, 1].cpu().numpy()
            auc = roc_auc_score(yv, vp) if len(np.unique(yv)) >= 2 else 0.5
            fold_aurocs.append(auc)
            del Xt, yt, Xv; torch.cuda.empty_cache()

        mean_auroc = float(np.mean(fold_aurocs))
        print(f"  End-AUROC (CV): {mean_auroc:.4f}", flush=True)

        # Train final probe
        X_gpu = X_train_t.to(device)
        y_t = torch.tensor(y_train, dtype=torch.long, device=device)
        probe = nn.Linear(n_components, 2).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        crit = nn.CrossEntropyLoss()
        for _ in range(MAX_EPOCHS):
            probe.train(); loss = crit(probe(X_gpu), y_t)
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        del X_gpu, y_t; torch.cuda.empty_cache()

        # Store PCA components for transforming new data
        pca_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=device)
        pca_components = torch.tensor(pca.components_, dtype=torch.float32, device=device)

        # Positional AUROC
        print(f"  Computing positional AUROC on {len(eval_traces)} eval traces...", flush=True)
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
                    h = res[pos, :].float()
                    # PCA transform on GPU
                    h_pca = (h - pca_mean) @ pca_components.T
                    p_yes = F.softmax(probe(h_pca.unsqueeze(0)), dim=-1)[0, 1].item()
                    all_rows.append(((pos - start) / span, p_yes, label))
                del saved, input_ids, res; torch.cuda.empty_cache()
                if (i + 1) % 100 == 0 or (i + 1) == len(eval_traces):
                    print(f"    PosAUROC: {i+1}/{len(eval_traces)}", flush=True)
                if (i + 1) % 200 == 0:
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
        print(f"  Positional: {pos_str}  RAMP={ramp:.4f}", flush=True)

        # Per-trace curves
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
                    h = res[pos, :].float()
                    h_pca = (h - pca_mean) @ pca_components.T
                    p_yes = F.softmax(probe(h_pca.unsqueeze(0)), dim=-1)[0, 1].item()
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5*n_cols, 3.5*n_rows))
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
                             fontsize=6)
                ax.grid(True, alpha=0.2)
                if col == 0: ax.set_ylabel(ds, fontsize=9, fontweight="bold")
        fig.suptitle(f"PCA({n_components}) + LINEAR PROBE -- L{LAYER}\n"
                     f"AUROC={mean_auroc:.4f}, ramp={ramp:.4f}, var={var_explained:.3f}",
                     fontsize=12, y=1.02)
        plt.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, "plots", f"pca{n_components}.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath}", flush=True)

        results[n_components] = {
            "end_auroc": mean_auroc, "positional_auroc": pos_aurocs,
            "ramp": ramp, "var_explained": var_explained,
        }

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY -- PCA PROBE", flush=True)
    print("=" * 70, flush=True)
    for k in PCA_DIMS:
        r = results[k]
        print(f"  PCA({k:3d}): var={r['var_explained']:.4f}, "
              f"end_auroc={r['end_auroc']:.4f}, ramp={r['ramp']:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
