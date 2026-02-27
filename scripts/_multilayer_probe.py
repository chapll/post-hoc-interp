"""Multi-layer probing: old method (train-on-end) and contrastive method across layers.

Method 1 (Train-on-End): Train probe on Pre-Handoff Window (last 3 tokens before
</think>), label = ground truth Yes/No. Same as milestone_demo.py but across layers.

Method 2 (Contrastive): For problems that have both correct and incorrect traces,
sample matched positions from BOTH traces. Label = correct/incorrect trace.
The probe learns "heading toward right answer" vs "heading toward wrong answer"
at the same position, controlling for position effects.

Runs both methods across layers [4, 8, 12, 16, 20, 24, 28, 32] with 5-fold CV.
Produces AUROC-vs-layer plots for both methods.

After training, applies both probes per-trace and plots P(correct) curves.

Usage:
    python scripts/_multilayer_probe.py
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.model import HIDDEN_DIM, NUM_LAYERS, load_model
from src.tokens import count_think_blocks, get_pre_handoff_window, get_think_content_mask
from src.utils import set_seed

set_seed(137)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRACE_DIR = "src/experimental/gorn/data/traces/valid/folio"
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

# Contrastive: how many positions to sample per trace pair
N_POSITIONS_PER_PAIR = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

device = torch.device(f"cuda:{GPU_ID}")


# ---------------------------------------------------------------------------
# Correctness determination (same as milestone_demo.py)
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
# 1. Discover ALL traces (not just trace_00), group by problem
# ---------------------------------------------------------------------------
def discover_all_traces():
    """Discover all traces grouped by problem directory.

    Returns dict: problem_id -> list of trace dicts.
    Each trace dict has: trace_path, token_ids, ground_truth, is_correct,
    trace_idx, n_tokens, think_content_positions.
    """
    problem_dirs = sorted(
        d for d in os.listdir(TRACE_DIR)
        if os.path.isdir(os.path.join(TRACE_DIR, d))
    )

    problems = {}
    stats = {"total_dirs": len(problem_dirs), "total_traces": 0,
             "correct": 0, "incorrect": 0, "skipped": 0}

    for pdir in problem_dirs:
        full_dir = os.path.join(TRACE_DIR, pdir)
        traces_in_dir = []

        for fname in sorted(os.listdir(full_dir)):
            if not fname.startswith("trace_") or not fname.endswith(".json"):
                continue

            trace_path = os.path.join(full_dir, fname)
            try:
                with open(trace_path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            token_ids = data["token_ids"]
            metadata = data.get("metadata", {})
            generated_text = data.get("generated_text", "")

            # Filter: single think block
            if count_think_blocks(token_ids) != 1:
                stats["skipped"] += 1
                continue

            is_correct = determine_correctness(metadata, generated_text)
            if is_correct is None:
                stats["skipped"] += 1
                continue

            # Get think content positions
            mask = get_think_content_mask(token_ids)
            eligible = [i for i, m in enumerate(mask) if m]
            if len(eligible) < 10:
                stats["skipped"] += 1
                continue

            trace_idx = int(fname.replace("trace_", "").replace(".json", ""))

            traces_in_dir.append({
                "trace_path": trace_path,
                "token_ids": token_ids,
                "ground_truth": metadata["ground_truth"],
                "is_correct": is_correct,
                "trace_idx": trace_idx,
                "n_tokens": len(token_ids),
                "eligible_positions": eligible,
                "problem_id": pdir,
            })

            stats["total_traces"] += 1
            if is_correct:
                stats["correct"] += 1
            else:
                stats["incorrect"] += 1

        if traces_in_dir:
            problems[pdir] = traces_in_dir

    print(f"Discovery: {stats}")
    return problems


# ---------------------------------------------------------------------------
# 2. Build contrastive pairs
# ---------------------------------------------------------------------------
def build_contrastive_pairs(problems):
    """Find problems with both correct and incorrect traces.

    Returns list of (correct_trace, incorrect_trace) pairs.
    """
    pairs = []
    for pdir, traces in problems.items():
        correct = [t for t in traces if t["is_correct"]]
        incorrect = [t for t in traces if not t["is_correct"]]
        if not correct or not incorrect:
            continue

        # Pair each incorrect trace with a correct trace
        for inc_trace in incorrect:
            cor_trace = correct[0]  # Use first correct trace as reference
            pairs.append((cor_trace, inc_trace))

    print(f"Contrastive pairs: {len(pairs)} (from {len(set(p[0]['problem_id'] for p in pairs))} problems)")
    return pairs


# ---------------------------------------------------------------------------
# 3. Extract activations at specific layers
# ---------------------------------------------------------------------------
def extract_end_activations(model, problems, layer):
    """Method 1: Extract Pre-Handoff Window activations for all trace_00s.

    Returns X [n_samples, hidden_dim], y [n_samples] (0=No, 1=Yes).
    """
    X_list, y_list = [], []

    # Use only trace_00 per problem (same as original)
    for pdir, traces in problems.items():
        trace_00 = [t for t in traces if t["trace_idx"] == 0]
        if not trace_00:
            continue
        trace = trace_00[0]
        if not trace["is_correct"]:
            continue

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

    X = torch.cat(X_list, dim=0)
    y = np.array(y_list)
    return X, y


def extract_contrastive_activations(model, pairs, layer):
    """Method 2: Extract matched-position activations from contrastive pairs.

    For each (correct_trace, incorrect_trace) pair:
    - Find overlapping normalized positions
    - Sample N_POSITIONS_PER_PAIR positions
    - Extract activations at those positions from both traces
    - Label: 1 = correct trace, 0 = incorrect trace

    Returns X [n_samples, hidden_dim], y [n_samples] (0=incorrect, 1=correct).
    """
    X_list, y_list = [], []
    pair_info = []  # For debugging

    for pair_idx, (cor_trace, inc_trace) in enumerate(pairs):
        cor_eligible = cor_trace["eligible_positions"]
        inc_eligible = inc_trace["eligible_positions"]

        # Normalize positions to [0, 1] for both traces
        cor_start, cor_end = cor_eligible[0], cor_eligible[-1]
        inc_start, inc_end = inc_eligible[0], inc_eligible[-1]
        cor_span = cor_end - cor_start if cor_end > cor_start else 1
        inc_span = inc_end - inc_start if inc_end > inc_start else 1

        # Sample normalized positions uniformly in [0.05, 0.95]
        # Avoid extreme edges where one trace might not have data
        rng = np.random.RandomState(SEED + pair_idx)
        norm_positions = rng.uniform(0.05, 0.95, size=N_POSITIONS_PER_PAIR)
        norm_positions.sort()

        # Map to actual token positions
        cor_positions = [cor_start + int(np.round(p * cor_span)) for p in norm_positions]
        inc_positions = [inc_start + int(np.round(p * inc_span)) for p in norm_positions]

        # Clamp to valid range
        cor_positions = [min(max(p, cor_eligible[0]), cor_eligible[-1]) for p in cor_positions]
        inc_positions = [min(max(p, inc_eligible[0]), inc_eligible[-1]) for p in inc_positions]

        # Forward pass for correct trace
        input_ids = torch.tensor([cor_trace["token_ids"]], dtype=torch.long, device=device)
        with model.trace(input_ids):
            saved = model.model.layers[layer].output.save()
        cor_residual = saved.detach().cpu()
        if cor_residual.dim() == 3 and cor_residual.shape[0] == 1:
            cor_residual = cor_residual.squeeze(0)
        del saved, input_ids
        torch.cuda.empty_cache()

        # Forward pass for incorrect trace
        input_ids = torch.tensor([inc_trace["token_ids"]], dtype=torch.long, device=device)
        with model.trace(input_ids):
            saved = model.model.layers[layer].output.save()
        inc_residual = saved.detach().cpu()
        if inc_residual.dim() == 3 and inc_residual.shape[0] == 1:
            inc_residual = inc_residual.squeeze(0)
        del saved, input_ids
        torch.cuda.empty_cache()

        # Extract at matched positions
        for cor_pos, inc_pos, norm_p in zip(cor_positions, inc_positions, norm_positions):
            X_list.append(cor_residual[cor_pos].float().unsqueeze(0))
            y_list.append(1)  # correct trace
            X_list.append(inc_residual[inc_pos].float().unsqueeze(0))
            y_list.append(0)  # incorrect trace

        pair_info.append({
            "problem": cor_trace["problem_id"],
            "cor_tokens": cor_trace["n_tokens"],
            "inc_tokens": inc_trace["n_tokens"],
            "n_positions": len(norm_positions),
        })

        del cor_residual, inc_residual

        if (pair_idx + 1) % 10 == 0 or pair_idx == 0:
            print(f"  Contrastive extraction: {pair_idx + 1}/{len(pairs)} pairs")

    if not X_list:
        return torch.zeros(0, HIDDEN_DIM), np.array([])

    X = torch.cat(X_list, dim=0)
    y = np.array(y_list)
    print(f"  Contrastive data: {X.shape[0]} samples, balance: {(y==1).sum()} pos / {(y==0).sum()} neg")
    return X, y


# ---------------------------------------------------------------------------
# 4. Train probe with CV
# ---------------------------------------------------------------------------
def train_probe_cv(X, y):
    """5-fold stratified CV. Returns (fold_aurocs, mean_auroc)."""
    if len(X) < 10 or len(np.unique(y)) < 2:
        print("  WARNING: insufficient data for CV")
        return [0.5], 0.5

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_aurocs = []
    X_gpu = X.to(device)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X.numpy(), y)):
        train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
        val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

        X_train = X_gpu[train_idx_t]
        y_train = torch.tensor(y[train_idx], dtype=torch.long, device=device)
        X_val = X_gpu[val_idx_t]
        y_val = torch.tensor(y[val_idx], dtype=torch.long, device=device)

        probe = nn.Linear(HIDDEN_DIM, 2).to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        best_state = None
        pat = 0

        for epoch in range(MAX_EPOCHS):
            probe.train()
            loss = criterion(probe(X_train), y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probe.eval()
            with torch.no_grad():
                vl = criterion(probe(X_val), y_val).item()
            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                pat = 0
            else:
                pat += 1
                if pat >= PATIENCE:
                    break

        if best_state is not None:
            probe.load_state_dict(best_state)
        probe.eval()

        with torch.no_grad():
            val_probs = torch.softmax(probe(X_val), dim=-1)[:, 1].cpu().numpy()

        y_val_np = y_val.cpu().numpy()
        if len(np.unique(y_val_np)) < 2:
            fold_aurocs.append(0.5)
        else:
            fold_aurocs.append(roc_auc_score(y_val_np, val_probs))

    del X_gpu
    torch.cuda.empty_cache()

    mean_auroc = float(np.mean(fold_aurocs))
    return fold_aurocs, mean_auroc


def train_probe_final(X, y):
    """Train on all data, return probe."""
    if len(X) < 5 or len(np.unique(y)) < 2:
        return None

    X_gpu = X.to(device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    probe = nn.Linear(HIDDEN_DIM, 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(MAX_EPOCHS):
        probe.train()
        loss = criterion(probe(X_gpu), y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    del X_gpu, y_tensor
    torch.cuda.empty_cache()
    return probe


# ---------------------------------------------------------------------------
# 5. Apply probe per-trace and plot
# ---------------------------------------------------------------------------
def apply_probe_pertrace(model, probe, trace, layer, method_name):
    """Apply probe to all eligible positions of a trace. Returns (norm_positions, p_class1)."""
    token_ids = trace["token_ids"]
    eligible = trace["eligible_positions"]
    start_pos = eligible[0]
    span = eligible[-1] - start_pos if eligible[-1] > start_pos else 1

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with model.trace(input_ids):
        saved = model.model.layers[layer].output.save()

    residual = saved.detach()
    if residual.dim() == 3 and residual.shape[0] == 1:
        residual = residual.squeeze(0)

    norm_positions = []
    p_values = []

    with torch.no_grad():
        for pos in eligible:
            h = residual[pos, :].float().unsqueeze(0).to(device)
            logits = probe(h)
            prob = F.softmax(logits, dim=-1)
            p_values.append(prob[0, 1].item())
            norm_positions.append((pos - start_pos) / span)

    del saved, input_ids, residual
    torch.cuda.empty_cache()

    return np.array(norm_positions), np.array(p_values)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()

    # Step 1: Discover traces
    print("=" * 70)
    print("MULTI-LAYER PROBE EXPERIMENT")
    print("=" * 70)
    print("\nStep 1: Discovering traces...")
    problems = discover_all_traces()

    # Step 2: Build contrastive pairs
    print("\nStep 2: Building contrastive pairs...")
    pairs = build_contrastive_pairs(problems)

    if len(pairs) < 5:
        print(f"WARNING: Only {len(pairs)} contrastive pairs. Results will be noisy.")

    # Step 3: Load model
    print("\nStep 3: Loading model...")
    model, tok = load_model(gpu_id=GPU_ID)

    # Step 4: Run both methods across layers
    end_results = {}   # layer -> (fold_aurocs, mean_auroc)
    con_results = {}   # layer -> (fold_aurocs, mean_auroc)
    end_probes = {}    # layer -> probe
    con_probes = {}    # layer -> probe

    for layer in LAYERS:
        print(f"\n{'=' * 50}")
        print(f"LAYER {layer}")
        print(f"{'=' * 50}")

        # Method 1: Train-on-End
        print(f"\n  [Method 1: Train-on-End]")
        t1 = time.time()
        X_end, y_end = extract_end_activations(model, problems, layer)
        print(f"  Extracted {X_end.shape[0]} end samples in {time.time() - t1:.1f}s")
        print(f"  Balance: {(y_end == 1).sum()} Yes / {(y_end == 0).sum()} No")

        folds, mean = train_probe_cv(X_end, y_end)
        end_results[layer] = (folds, mean)
        print(f"  AUROC = {mean:.4f} +/- {np.std(folds):.4f}")

        end_probes[layer] = train_probe_final(X_end, y_end)

        # Method 2: Contrastive
        print(f"\n  [Method 2: Contrastive]")
        t2 = time.time()
        X_con, y_con = extract_contrastive_activations(model, pairs, layer)
        print(f"  Extracted {X_con.shape[0]} contrastive samples in {time.time() - t2:.1f}s")

        if X_con.shape[0] > 0:
            folds_c, mean_c = train_probe_cv(X_con, y_con)
            con_results[layer] = (folds_c, mean_c)
            print(f"  AUROC = {mean_c:.4f} +/- {np.std(folds_c):.4f}")
            con_probes[layer] = train_probe_final(X_con, y_con)
        else:
            con_results[layer] = ([0.5], 0.5)
            print("  No contrastive data available")

    # Step 5: Save results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Layer':>6} | {'End AUROC':>12} | {'Contrastive AUROC':>18}")
    print("-" * 45)
    for layer in LAYERS:
        end_mean = end_results[layer][1]
        con_mean = con_results[layer][1]
        print(f"  {layer:4d} | {end_mean:12.4f} | {con_mean:18.4f}")

    results_json = {
        "layers": LAYERS,
        "train_on_end": {str(l): {"mean_auroc": end_results[l][1],
                                   "fold_aurocs": end_results[l][0]}
                         for l in LAYERS},
        "contrastive": {str(l): {"mean_auroc": con_results[l][1],
                                  "fold_aurocs": con_results[l][0]}
                         for l in LAYERS},
        "n_end_samples": int(X_end.shape[0]) if 'X_end' in dir() else 0,
        "n_contrastive_pairs": len(pairs),
    }
    results_path = os.path.join(OUTPUT_DIR, "multilayer_results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # Step 6: AUROC vs Layer plot
    fig, ax = plt.subplots(figsize=(10, 6))
    end_means = [end_results[l][1] for l in LAYERS]
    end_stds = [np.std(end_results[l][0]) for l in LAYERS]
    con_means = [con_results[l][1] for l in LAYERS]
    con_stds = [np.std(con_results[l][0]) for l in LAYERS]

    ax.errorbar(LAYERS, end_means, yerr=end_stds, marker="o", linewidth=2,
                capsize=4, color="#0077BB", label="Train-on-End (Yes vs No)")
    ax.errorbar(LAYERS, con_means, yerr=con_stds, marker="s", linewidth=2,
                capsize=4, color="#CC3311", label="Contrastive (correct vs incorrect)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, label="Chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AUROC (5-fold CV)", fontsize=12)
    ax.set_title("Probe AUROC vs Layer -- FOLIO", fontsize=13)
    ax.set_xticks(LAYERS)
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    auroc_path = os.path.join(OUTPUT_DIR, "plots", "auroc_vs_layer.png")
    fig.savefig(auroc_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {auroc_path}")

    # Step 7: Per-trace P(correct/class1) for best layers
    # Pick best layer for each method
    best_end_layer = max(LAYERS, key=lambda l: end_results[l][1])
    best_con_layer = max(LAYERS, key=lambda l: con_results[l][1])
    print(f"\nBest end layer: {best_end_layer} (AUROC={end_results[best_end_layer][1]:.4f})")
    print(f"Best contrastive layer: {best_con_layer} (AUROC={con_results[best_con_layer][1]:.4f})")

    # Select a few traces for per-trace plots
    # Use traces from contrastive pairs (have force-answer comparison too)
    demo_traces = []
    seen_problems = set()
    for cor, inc in pairs[:6]:
        if cor["problem_id"] not in seen_problems:
            demo_traces.append(cor)
            demo_traces.append(inc)
            seen_problems.add(cor["problem_id"])
        if len(demo_traces) >= 8:
            break

    # Also add some non-pair traces for variety
    for pdir, traces in list(problems.items())[:5]:
        if pdir not in seen_problems and traces[0]["is_correct"]:
            demo_traces.append(traces[0])
            seen_problems.add(pdir)
        if len(demo_traces) >= 12:
            break

    print(f"\nGenerating per-trace plots for {len(demo_traces)} traces...")

    for trace in demo_traces:
        pid = trace["problem_id"]
        tidx = trace["trace_idx"]
        gt = trace["ground_truth"]
        is_cor = trace["is_correct"]

        short_pid = pid.split("_cfg")[0]
        cfg = "A" if "cfga" in pid else "B"

        n_panels = 2
        fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)

        # Panel 1: End probe at best layer
        if end_probes.get(best_end_layer) is not None:
            norm_pos, p_vals = apply_probe_pertrace(
                model, end_probes[best_end_layer], trace, best_end_layer, "end"
            )
            # P(correct) for end probe: flip for GT=No
            if gt == "Yes":
                p_correct = p_vals
            else:
                p_correct = 1.0 - p_vals

            ax = axes[0]
            ax.plot(norm_pos, p_correct, linewidth=0.4, color="#bdc3c7", alpha=0.5)
            w = min(50, len(p_correct) // 5)
            if w >= 3:
                smoothed = np.convolve(p_correct, np.ones(w) / w, mode="valid")
                smooth_x = norm_pos[w // 2: w // 2 + len(smoothed)]
                ax.plot(smooth_x, smoothed, linewidth=2.0, color="#0077BB")
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.set_ylabel("P(correct) [End probe]", fontsize=11)
            ax.set_title(
                f"Train-on-End Probe | Layer {best_end_layer} | {short_pid}({cfg}) t{tidx:02d} "
                f"| GT={gt} | {'Correct' if is_cor else 'Incorrect'} trace",
                fontsize=10,
            )
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

        # Panel 2: Contrastive probe at best layer
        if con_probes.get(best_con_layer) is not None:
            norm_pos_c, p_vals_c = apply_probe_pertrace(
                model, con_probes[best_con_layer], trace, best_con_layer, "contrastive"
            )
            # P(class1) = P(correct trace) -- no flip needed, label IS correct/incorrect

            ax = axes[1]
            ax.plot(norm_pos_c, p_vals_c, linewidth=0.4, color="#bdc3c7", alpha=0.5)
            w = min(50, len(p_vals_c) // 5)
            if w >= 3:
                smoothed_c = np.convolve(p_vals_c, np.ones(w) / w, mode="valid")
                smooth_x_c = norm_pos_c[w // 2: w // 2 + len(smoothed_c)]
                ax.plot(smooth_x_c, smoothed_c, linewidth=2.0, color="#CC3311")
            ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
            ax.set_ylabel("P(correct trace) [Contrastive]", fontsize=11)
            ax.set_title(
                f"Contrastive Probe | Layer {best_con_layer} | "
                f"{'Correct' if is_cor else 'Incorrect'} trace",
                fontsize=10,
            )
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Normalized position in think block [0, 1]", fontsize=11)

        plt.tight_layout()
        fname = f"pertrace_{pid}_t{tidx:02d}_{'cor' if is_cor else 'inc'}.png"
        path = os.path.join(OUTPUT_DIR, "plots", fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
