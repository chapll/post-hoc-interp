"""Diagnose probe: check logit magnitudes, saturation, and run random-label control."""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.model import HIDDEN_DIM, load_model
from src.tokens import count_think_blocks, get_pre_handoff_window, get_think_content_mask
from src.utils import set_seed, setup_logging

logger = setup_logging("diagnose_probe")

LAYER = 24
WINDOW_SIZE = 3
SEED = 137
OUTPUT_DIR = "outputs/milestone_demo"
TRACE_DIR = "src/experimental/gorn/data/traces/valid/folio"


def main():
    set_seed(SEED)
    device = torch.device("cuda:0")

    # 1. Load trained probe and check weight stats
    print("=" * 70)
    print("PROBE WEIGHT DIAGNOSTICS")
    print("=" * 70)

    tensors = load_file(os.path.join(OUTPUT_DIR, "probe_layer24.safetensors"))
    probe = nn.Linear(HIDDEN_DIM, 2)
    probe.weight.data = tensors["layer_24_weight"].float()
    probe.bias.data = tensors["layer_24_bias"].float()
    probe.eval()

    w = probe.weight.data
    print(f"Weight shape: {w.shape}")
    print(f"Weight L2 norm (class 0): {w[0].norm().item():.4f}")
    print(f"Weight L2 norm (class 1): {w[1].norm().item():.4f}")
    print(f"Bias: [{probe.bias.data[0].item():.4f}, {probe.bias.data[1].item():.4f}]")
    print(f"Weight direction cosine sim: {F.cosine_similarity(w[0].unsqueeze(0), w[1].unsqueeze(0)).item():.4f}")
    diff_direction = w[1] - w[0]
    print(f"Decision direction L2 norm: {diff_direction.norm().item():.4f}")

    # 2. Load model and check logit magnitudes on a trace
    print("\n" + "=" * 70)
    print("LOGIT MAGNITUDE ANALYSIS")
    print("=" * 70)

    model, tok = load_model(gpu_id=0)
    probe = probe.to(device)

    trace_path = os.path.join(TRACE_DIR, "folio_109_cfga", "trace_00.json")
    with open(trace_path) as f:
        data = json.load(f)
    token_ids = data["token_ids"]
    gt = data["metadata"]["ground_truth"]

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with model.trace(input_ids):
        saved = model.model.layers[LAYER].output.save()

    residual = saved.detach()
    if residual.dim() == 3 and residual.shape[0] == 1:
        residual = residual.squeeze(0)

    mask = get_think_content_mask(token_ids)
    eligible = [i for i, m in enumerate(mask) if m]

    print(f"\nTrace: {len(token_ids)} tokens, {len(eligible)} eligible, GT={gt}")

    # Check specific positions
    check_pos = [
        eligible[0], eligible[5], eligible[10],
        eligible[len(eligible) // 4],
        eligible[len(eligible) // 2],
        eligible[3 * len(eligible) // 4],
        eligible[-5], eligible[-3], eligible[-1],
    ]

    print(f"\n{'Pos':>6} | {'Logits [c0, c1]':>28} | {'P(Yes)':>8} | {'|diff|':>8} | Token")
    print("-" * 90)
    for pos in check_pos:
        h = residual[pos, :].float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits = probe(h)
            probs = F.softmax(logits, dim=-1)
        l0, l1 = logits[0].tolist()
        p1 = probs[0, 1].item()
        diff = abs(l1 - l0)
        token_text = tok.decode([token_ids[pos]])
        print(f"  {pos:5d} | [{l0:+12.3f}, {l1:+12.3f}] | {p1:8.4f} | {diff:8.3f} | {repr(token_text)}")

    # Stats over ALL eligible positions
    print("\n--- Logit diff statistics over all eligible positions ---")
    all_diffs = []
    all_probs = []
    with torch.no_grad():
        for pos in eligible:
            h = residual[pos, :].float().unsqueeze(0).to(device)
            logits = probe(h)
            probs = F.softmax(logits, dim=-1)
            all_diffs.append((logits[0, 1] - logits[0, 0]).item())
            all_probs.append(probs[0, 1].item())

    diffs = np.array(all_diffs)
    ps = np.array(all_probs)
    print(f"Logit diff: mean={diffs.mean():.3f}, std={diffs.std():.3f}, "
          f"min={diffs.min():.3f}, max={diffs.max():.3f}")
    print(f"P(Yes): mean={ps.mean():.4f}, std={ps.std():.4f}")
    print(f"  P(Yes) in [0.2, 0.8] (uncertain): {((ps > 0.2) & (ps < 0.8)).mean():.1%}")
    print(f"  P(Yes) > 0.9 (confident Yes):      {(ps > 0.9).mean():.1%}")
    print(f"  P(Yes) < 0.1 (confident No):       {(ps < 0.1).mean():.1%}")

    # 3. Plot logit diff instead of softmax (more informative)
    print("\n--- Generating logit-diff plot (pre-softmax) ---")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    rel_pos = [p - eligible[0] for p in eligible]

    # Top: raw logit difference
    axes[0].plot(rel_pos, all_diffs, linewidth=0.6, color="#2c3e50", alpha=0.8)
    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Decision boundary (0)")
    axes[0].set_ylabel("Logit diff (class1 - class0)", fontsize=11)
    axes[0].set_title(
        f"folio_109_cfga trace_00 | GT={gt} | Layer {LAYER} | Raw logit difference",
        fontsize=12,
    )
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Bottom: softmax probability
    axes[1].plot(rel_pos, all_probs, linewidth=0.6, color="#2c3e50", alpha=0.8)
    axes[1].axhline(y=0.5, color="grey", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Token position (relative)", fontsize=11)
    axes[1].set_ylabel("P(Yes | h_t)", fontsize=11)
    axes[1].set_title("Softmax probability (same data)", fontsize=12)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    diag_path = os.path.join(OUTPUT_DIR, "plots", "diagnostic_logit_vs_softmax.png")
    fig.savefig(diag_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {diag_path}")

    # 4. RANDOM LABEL CONTROL
    print("\n" + "=" * 70)
    print("RANDOM LABEL CONTROL (10 permutations)")
    print("=" * 70)

    # First, re-extract training data (reuse model)
    from scripts.milestone_demo import discover_traces, extract_training_activations
    traces = discover_traces(TRACE_DIR)
    X, y = extract_training_activations(model, traces, LAYER, WINDOW_SIZE, device)

    del model
    torch.cuda.empty_cache()

    rng = np.random.RandomState(SEED + 100)
    random_aurocs = []

    for perm_idx in range(10):
        y_shuffled = y.copy()
        rng.shuffle(y_shuffled)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        X_gpu = X.to(device)
        fold_aurocs = []

        for train_idx, val_idx in skf.split(X.numpy(), y_shuffled):
            train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
            val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

            X_train = X_gpu[train_idx_t]
            y_train = torch.tensor(y_shuffled[train_idx], dtype=torch.long, device=device)
            X_val = X_gpu[val_idx_t]
            y_val = torch.tensor(y_shuffled[val_idx], dtype=torch.long, device=device)

            p = nn.Linear(HIDDEN_DIM, 2).to(device)
            opt = torch.optim.Adam(p.parameters(), lr=1e-3, weight_decay=1e-4)
            crit = nn.CrossEntropyLoss()

            best_vl = float("inf")
            best_st = None
            pat = 0
            for epoch in range(100):
                p.train()
                loss = crit(p(X_train), y_train)
                opt.zero_grad()
                loss.backward()
                opt.step()

                p.eval()
                with torch.no_grad():
                    vl = crit(p(X_val), y_val).item()
                if vl < best_vl:
                    best_vl = vl
                    best_st = {k: v.clone() for k, v in p.state_dict().items()}
                    pat = 0
                else:
                    pat += 1
                    if pat >= 10:
                        break

            if best_st is not None:
                p.load_state_dict(best_st)
            p.eval()
            with torch.no_grad():
                vp = torch.softmax(p(X_val), dim=-1)[:, 1].cpu().numpy()
            yv = y_val.cpu().numpy()
            if len(np.unique(yv)) < 2:
                fold_aurocs.append(0.5)
            else:
                fold_aurocs.append(roc_auc_score(yv, vp))

        perm_auroc = float(np.mean(fold_aurocs))
        random_aurocs.append(perm_auroc)
        print(f"  Permutation {perm_idx + 1}: AUROC = {perm_auroc:.4f}")

        del X_gpu
        torch.cuda.empty_cache()

    print(f"\nRandom-label AUROC: mean={np.mean(random_aurocs):.4f}, "
          f"std={np.std(random_aurocs):.4f}")
    print(f"Real-label AUROC: 0.9999")
    print(f"Selectivity gap: {0.9999 - np.mean(random_aurocs):.4f}")

    if np.mean(random_aurocs) > 0.60:
        print("WARNING: Random-label AUROC is suspiciously high!")
    else:
        print("Random-label control PASSED: probe is learning real signal, not noise.")


if __name__ == "__main__":
    main()
