"""MLP probe: non-linear probe for better intermediate-position generalization.

Key idea: the linear probe direction is correct (confirmed by random-label control)
but applying a linear model trained on end-of-trace to intermediate positions fails
because of massive distribution shift (mean_norm=93 at end vs different at middle).

An MLP with a bottleneck can learn:
1. To normalize/adapt the input distribution
2. Non-linear interactions between activation dimensions
3. Better generalization from end-of-trace to intermediate positions

Architecture: 4096 -> 256 -> 64 -> 2 (with LayerNorm + GELU)

Uses same data pipeline as bigdata probe:
- ALL available traces (trace_00, 01, 02, + higher seeds)
- 80/20 train/eval split by problem_id
- Layer 20 (best from all sweeps)
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.model import HIDDEN_DIM, load_model
from src.tokens import count_think_blocks, get_pre_handoff_window, get_think_content_mask
from src.utils import set_seed

set_seed(137)

DATASETS = ["gsm_symbolic", "strategyqa", "folio", "deepmath"]
TRACE_BASE = "src/experimental/gorn/data/traces/valid"
OUTPUT_DIR = "outputs/multilayer_probe/mlp_probe"
LAYER = 20
WINDOW_SIZE = 3
SEED = 137
GPU_ID = 0
N_EVAL_PER_DS = 8  # more eval traces for better visualization

LR = 5e-4
MAX_EPOCHS = 300
PATIENCE = 20
BATCH_SIZE = 512

POSITION_BINS = [
    ("early", 0.0, 0.2),
    ("early_mid", 0.2, 0.4),
    ("middle", 0.4, 0.6),
    ("late_mid", 0.6, 0.8),
    ("late", 0.8, 1.0),
]

# MLP configs: (hidden_dims, weight_decay, dropout, use_layernorm)
MLP_CONFIGS = [
    # Bottleneck MLPs
    {"name": "mlp_256_64", "hiddens": [256, 64], "wd": 1e-2, "dropout": 0.1, "ln": True},
    {"name": "mlp_128_32", "hiddens": [128, 32], "wd": 1e-2, "dropout": 0.1, "ln": True},
    {"name": "mlp_512_128", "hiddens": [512, 128], "wd": 1e-2, "dropout": 0.2, "ln": True},
    # Wider but shallower
    {"name": "mlp_256", "hiddens": [256], "wd": 1e-2, "dropout": 0.1, "ln": True},
    # More regularized
    {"name": "mlp_256_64_heavyreg", "hiddens": [256, 64], "wd": 1e-1, "dropout": 0.3, "ln": True},
    # Linear baseline for comparison
    {"name": "linear_baseline", "hiddens": [], "wd": 1e-2, "dropout": 0.0, "ln": False},
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
device = torch.device(f"cuda:{GPU_ID}")


class MLPProbe(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes=2, dropout=0.1, use_ln=True):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            if use_ln:
                layers.append(nn.LayerNorm(in_d))
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_d = h
        if use_ln and hidden_dims:
            layers.append(nn.LayerNorm(in_d))
        layers.append(nn.Linear(in_d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


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
    print("=" * 70)
    print("MLP PROBE -- NON-LINEAR PROBING")
    print(f"Layer {LAYER}, {len(MLP_CONFIGS)} configs")
    print("=" * 70)

    print("\nDiscovering ALL traces...")
    all_traces, per_ds_count = discover_all_traces()
    for ds, c in per_ds_count.items():
        print(f"  {ds}: {c} traces")
    print(f"Total: {len(all_traces)} traces")

    # Split by problem_id
    rng = np.random.RandomState(SEED)
    all_pids = sorted(set(t["problem_id"] for t in all_traces))
    rng.shuffle(all_pids)
    n_eval_pids = max(len(all_pids) // 5, 10)
    eval_pids = set(all_pids[:n_eval_pids])
    train_traces = [t for t in all_traces if t["problem_id"] not in eval_pids]
    eval_traces = [t for t in all_traces if t["problem_id"] in eval_pids]
    print(f"Train: {len(train_traces)} traces, Eval: {len(eval_traces)} traces")

    print(f"\nLoading model on GPU {GPU_ID}...")
    model, tok = load_model(gpu_id=GPU_ID)

    # Extract training activations
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
    del X_list; gc.collect()
    print(f"Training data: {X_train.shape[0]} samples "
          f"(Yes={y_train.sum()}, No={len(y_train)-y_train.sum()})")

    # Compute training stats for normalization reference
    train_mean = X_train.mean(dim=0)
    train_std = X_train.std(dim=0).clamp(min=1e-8)

    # Select eval traces for plotting
    eval_subset = []
    for ds in DATASETS:
        dst = [t for t in eval_traces if t["dataset"] == ds]
        eval_subset.extend(dst[:N_EVAL_PER_DS])
    print(f"Eval subset for plots: {len(eval_subset)} traces")

    # Train and evaluate each MLP config
    results = {}
    for cfg in MLP_CONFIGS:
        name = cfg["name"]
        hiddens = cfg["hiddens"]
        wd = cfg["wd"]
        dropout = cfg["dropout"]
        use_ln = cfg["ln"]

        print(f"\n{'='*60}")
        print(f"CONFIG: {name} (hiddens={hiddens}, wd={wd}, dropout={dropout}, ln={use_ln})")
        print(f"{'='*60}")

        # 5-fold CV for AUROC
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_aurocs = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train.numpy(), y_train)):
            Xt = X_train[train_idx].to(device)
            yt = torch.tensor(y_train[train_idx], dtype=torch.long, device=device)
            Xv = X_train[val_idx].to(device)
            yv = y_train[val_idx]

            probe = MLPProbe(HIDDEN_DIM, hiddens, dropout=dropout, use_ln=use_ln).to(device)
            opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=wd)
            crit = nn.CrossEntropyLoss()
            best_vl, best_st, pat = float("inf"), None, 0

            for epoch in range(MAX_EPOCHS):
                # Mini-batch training
                probe.train()
                perm = torch.randperm(len(Xt), device=device)
                for start in range(0, len(Xt), BATCH_SIZE):
                    batch_idx = perm[start:start + BATCH_SIZE]
                    loss = crit(probe(Xt[batch_idx]), yt[batch_idx])
                    opt.zero_grad(); loss.backward(); opt.step()

                # Validation
                probe.eval()
                with torch.no_grad():
                    yv_t = torch.tensor(yv, dtype=torch.long, device=device)
                    vl = crit(probe(Xv), yv_t).item()
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
                vp = torch.softmax(probe(Xv), dim=-1)[:, 1].cpu().numpy()
            auc = roc_auc_score(yv, vp) if len(np.unique(yv)) >= 2 else 0.5
            fold_aurocs.append(auc)
            del Xt, yt, Xv; torch.cuda.empty_cache()

        mean_auroc = float(np.mean(fold_aurocs))
        print(f"  End-AUROC (CV): {mean_auroc:.4f} (folds: {[f'{a:.4f}' for a in fold_aurocs]})")

        # Train final probe on all data
        probe = MLPProbe(HIDDEN_DIM, hiddens, dropout=dropout, use_ln=use_ln).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=wd)
        crit = nn.CrossEntropyLoss()
        X_gpu = X_train.to(device)
        y_t = torch.tensor(y_train, dtype=torch.long, device=device)

        for epoch in range(MAX_EPOCHS):
            probe.train()
            perm = torch.randperm(len(X_gpu), device=device)
            for start in range(0, len(X_gpu), BATCH_SIZE):
                batch_idx = perm[start:start + BATCH_SIZE]
                loss = crit(probe(X_gpu[batch_idx]), y_t[batch_idx])
                opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        del X_gpu, y_t; torch.cuda.empty_cache()

        # Save probe
        torch.save(probe.state_dict(), os.path.join(OUTPUT_DIR, f"{name}.pt"))

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
        print(f"  Positional: {pos_str}  RAMP={ramp:.4f}")

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
        fig.suptitle(f"MLP PROBE: {name}\n"
                     f"AUROC={mean_auroc:.4f}, ramp={ramp:.4f}", fontsize=12, y=1.02)
        plt.tight_layout()
        outpath = os.path.join(OUTPUT_DIR, "plots", f"{name}.png")
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {outpath}")

        results[name] = {
            "end_auroc": mean_auroc, "positional_auroc": pos_aurocs, "ramp": ramp,
            "config": cfg,
        }

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY -- MLP PROBE")
    print("=" * 70)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["ramp"]):
        print(f"  {name:30s}: end_auroc={r['end_auroc']:.4f}, ramp={r['ramp']:.4f}")

    # Save results
    import json as json_
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json_.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
