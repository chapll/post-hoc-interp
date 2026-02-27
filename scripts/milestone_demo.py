"""Quick milestone demo: train layer-24 probe on FOLIO, plot confidence curves and cosine similarity.

Reads traces from friend's data directory (no pre-computed activations), runs fresh
nnsight forward passes, trains a binary probe at layer 24, and generates plots for
milestone presentation.

Output:
  outputs/milestone_demo/
    probe_layer24.safetensors             -- trained probe weights
    auroc_results.json                    -- CV AUROC numbers
    plots/confidence_{id}_trace{N}.png    -- 10 per-trace confidence curve plots
    plots/cosine_{id}_trace{N}.png        -- 10 per-trace cosine similarity plots
    plots/all_confidence_overlay.png      -- all 10 curves overlaid
    plots/all_cosine_overlay.png          -- all 10 S_t curves overlaid

Usage:
    python scripts/milestone_demo.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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

from src.model import HIDDEN_DIM, load_model
from src.tokens import count_think_blocks, get_pre_handoff_window, get_think_content_mask
from src.utils import set_seed, setup_logging

logger = setup_logging("milestone_demo")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRACE_DIR = str(
    Path(__file__).resolve().parent.parent
    / "src" / "experimental" / "gorn" / "data" / "traces" / "valid" / "folio"
)
OUTPUT_DIR = str(
    Path(__file__).resolve().parent.parent / "outputs" / "milestone_demo"
)

LAYER = 24
WINDOW_SIZE = 3
SEED = 137
GPU_ID = 0
N_FOLDS = 5
N_DEMO_TRACES = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 100
PATIENCE = 10


# ---------------------------------------------------------------------------
# 1. Trace Discovery and Correctness Determination
# ---------------------------------------------------------------------------

def determine_correctness(metadata: dict, generated_text: str) -> bool | None:
    """Determine if the model answered correctly using Rho/Tau label mapping.

    If ground_truth is 'Yes', the correct response is valid_token.
    If ground_truth is 'No', the correct response is invalid_token.
    Checks whether the model's response after </think> matches.

    Returns True (correct), False (incorrect), or None (cannot determine).
    """
    ground_truth = metadata.get("ground_truth")
    valid_token = metadata.get("valid_token")
    invalid_token = metadata.get("invalid_token")

    if ground_truth is None or valid_token is None or invalid_token is None:
        return None

    idx = generated_text.find("</think>")
    if idx < 0:
        return None

    response = generated_text[idx + len("</think>"):].strip()

    if ground_truth == "Yes":
        expected = valid_token
    else:
        expected = invalid_token

    has_expected = expected in response
    other = invalid_token if ground_truth == "Yes" else valid_token
    has_other = other in response

    if has_expected and not has_other:
        return True
    if has_other and not has_expected:
        return False
    # Ambiguous (both or neither)
    return None


def discover_traces(traces_dir: str) -> list[dict]:
    """Discover FOLIO traces, determine correctness, filter to valid ones.

    Uses only trace_00.json per problem directory (primary trace) for
    training to avoid data leakage across seeds.

    Returns list of dicts with: trace_path, token_ids, generated_text,
    ground_truth, problem_id, config, trace_idx, n_tokens.
    """
    if not os.path.isdir(traces_dir):
        raise FileNotFoundError(f"Traces directory not found: {traces_dir}")

    problem_dirs = sorted(
        d for d in os.listdir(traces_dir)
        if os.path.isdir(os.path.join(traces_dir, d))
    )
    logger.info("Found %d problem directories in %s", len(problem_dirs), traces_dir)

    all_traces = []
    stats = {
        "total_dirs": len(problem_dirs),
        "loaded": 0,
        "multi_block": 0,
        "no_window": 0,
        "incorrect": 0,
        "undetermined": 0,
        "valid": 0,
    }

    for problem_dir_name in problem_dirs:
        problem_dir = os.path.join(traces_dir, problem_dir_name)

        # Primary trace only (trace_00.json)
        trace_path = os.path.join(problem_dir, "trace_00.json")
        if not os.path.isfile(trace_path):
            continue

        try:
            with open(trace_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Skipping unreadable trace %s: %s", trace_path, exc)
            continue

        stats["loaded"] += 1
        token_ids = data["token_ids"]
        metadata = data.get("metadata", {})
        generated_text = data.get("generated_text", "")

        # Filter: single think block
        if count_think_blocks(token_ids) != 1:
            stats["multi_block"] += 1
            continue

        # Filter: must have Pre-Handoff Window
        window = get_pre_handoff_window(token_ids, window_size=WINDOW_SIZE)
        if window is None:
            stats["no_window"] += 1
            continue

        # Determine correctness from generated text + label mapping
        is_correct = determine_correctness(metadata, generated_text)
        if is_correct is None:
            stats["undetermined"] += 1
            continue
        if not is_correct:
            stats["incorrect"] += 1
            continue

        # Extract config variant
        config = "unknown"
        if "_cfga" in problem_dir_name:
            config = "cfga"
        elif "_cfgb" in problem_dir_name:
            config = "cfgb"

        all_traces.append({
            "trace_path": trace_path,
            "token_ids": token_ids,
            "generated_text": generated_text,
            "ground_truth": metadata["ground_truth"],
            "problem_id": problem_dir_name,
            "config": config,
            "trace_idx": 0,
            "n_tokens": len(token_ids),
        })
        stats["valid"] += 1

    logger.info("Discovery complete: %s", json.dumps(stats))

    yes_count = sum(1 for t in all_traces if t["ground_truth"] == "Yes")
    no_count = len(all_traces) - yes_count
    logger.info("Ground truth balance: Yes=%d, No=%d (total=%d)", yes_count, no_count, len(all_traces))

    return all_traces


# ---------------------------------------------------------------------------
# 2. Activation Extraction (Pre-Handoff Window, Single Layer)
# ---------------------------------------------------------------------------

def extract_training_activations(
    model,
    traces: list[dict],
    layer: int,
    window_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Extract Pre-Handoff Window activations at a single layer for all traces.

    Returns:
        X: float32 tensor [n_samples, hidden_dim] on CPU.
        y: numpy array [n_samples] with 0/1 labels.
    """
    X_list = []
    y_list = []
    total = len(traces)
    start_time = time.time()

    for i, trace in enumerate(traces):
        token_ids = trace["token_ids"]
        ground_truth = trace["ground_truth"]

        window = get_pre_handoff_window(token_ids, window_size=window_size)
        if window is None:
            continue

        start_pos, end_pos = window
        label = 1 if ground_truth == "Yes" else 0

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        with model.trace(input_ids):
            saved = model.model.layers[layer].output.save()

        tensor = saved.detach().cpu()
        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)  # [seq_len, hidden_dim]

        window_vectors = tensor[start_pos:end_pos].float()  # [window_size, hidden_dim]
        X_list.append(window_vectors)
        y_list.extend([label] * window_vectors.shape[0])

        del saved, input_ids
        torch.cuda.empty_cache()

        if (i + 1) % 50 == 0 or i == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "Extraction: %d/%d traces (%.1f traces/s, %d tokens)",
                i + 1, total, rate, len(token_ids),
            )

    X = torch.cat(X_list, dim=0)
    y = np.array(y_list)

    elapsed = time.time() - start_time
    logger.info(
        "Extraction complete: %d traces -> %d samples in %.1fs. Balance: %d pos / %d neg",
        total, X.shape[0], elapsed, (y == 1).sum(), (y == 0).sum(),
    )
    return X, y


# ---------------------------------------------------------------------------
# 3. Probe Training
# ---------------------------------------------------------------------------

def train_probe_cv(
    X: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
) -> tuple[list[float], float]:
    """Train probe with 5-fold stratified CV.

    Returns (fold_aurocs, mean_auroc).
    """
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
        patience_counter = 0
        stopped_epoch = MAX_EPOCHS

        for epoch in range(MAX_EPOCHS):
            probe.train()
            logits = probe(X_train)
            loss = criterion(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probe.eval()
            with torch.no_grad():
                val_logits = probe(X_val)
                val_loss = criterion(val_logits, y_val).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in probe.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    stopped_epoch = epoch + 1
                    break

        if best_state is not None:
            probe.load_state_dict(best_state)
        probe.eval()

        with torch.no_grad():
            val_probs = torch.softmax(probe(X_val), dim=-1)[:, 1].cpu().numpy()

        y_val_np = y_val.cpu().numpy()
        if len(np.unique(y_val_np)) < 2:
            auroc = 0.5
            logger.warning("  Fold %d: only one class in val set, AUROC=0.5", fold_idx + 1)
        else:
            auroc = roc_auc_score(y_val_np, val_probs)

        fold_aurocs.append(auroc)
        logger.info("  Fold %d: AUROC = %.4f (stopped epoch %d)", fold_idx + 1, auroc, stopped_epoch)

    del X_gpu
    torch.cuda.empty_cache()

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc = float(np.std(fold_aurocs))
    logger.info("CV result: mean AUROC = %.4f +/- %.4f", mean_auroc, std_auroc)

    return fold_aurocs, mean_auroc


def train_probe_final(
    X: torch.Tensor,
    y: np.ndarray,
    device: torch.device,
) -> nn.Linear:
    """Train final probe on all data (no val split) for inference and weight saving."""
    X_gpu = X.to(device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    probe = nn.Linear(HIDDEN_DIM, 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(MAX_EPOCHS):
        probe.train()
        logits = probe(X_gpu)
        loss = criterion(logits, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()

    # Log final training accuracy
    with torch.no_grad():
        preds = probe(X_gpu).argmax(dim=-1)
        acc = (preds == y_tensor).float().mean().item()
    logger.info("Final probe training accuracy: %.4f", acc)

    del X_gpu, y_tensor
    torch.cuda.empty_cache()

    return probe


# ---------------------------------------------------------------------------
# 4. Per-Trace Inference (Confidence + Cosine in Single Forward Pass)
# ---------------------------------------------------------------------------

def process_demo_trace(
    model,
    probe: nn.Linear,
    token_ids: list[int],
    layer: int,
    device: torch.device,
) -> tuple[dict | None, dict | None]:
    """Single forward pass to compute both confidence curve and cosine similarity.

    Returns:
        (confidence_data, cosine_data) -- each is {positions, values} or None.
    """
    mask = get_think_content_mask(token_ids)
    eligible = [i for i, m in enumerate(mask) if m]

    if len(eligible) < 5:
        return None, None

    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    with model.trace(input_ids):
        saved = model.model.layers[layer].output.save()

    residual = saved.detach()
    if isinstance(residual, tuple):
        residual = residual[0]
    if residual.dim() == 2:
        residual = residual.unsqueeze(0)

    # --- Confidence curve: probe at every eligible position ---
    conf_positions = []
    conf_values = []

    with torch.no_grad():
        for pos in eligible:
            h = residual[0, pos, :].float().unsqueeze(0)  # [1, hidden_dim]
            logits = probe(h)
            prob = F.softmax(logits, dim=-1)
            conf_positions.append(pos)
            conf_values.append(prob[0, 1].item())

    # --- Cosine similarity: S_t = cos(h_t, h_{t-1}) ---
    cos_positions = []
    cos_values = []

    with torch.no_grad():
        for i in range(1, len(eligible)):
            pos_curr = eligible[i]
            pos_prev = eligible[i - 1]
            h_curr = residual[0, pos_curr, :].float().unsqueeze(0)
            h_prev = residual[0, pos_prev, :].float().unsqueeze(0)
            sim = F.cosine_similarity(h_curr, h_prev, dim=-1).item()
            cos_positions.append(pos_curr)
            cos_values.append(sim)

    del saved, input_ids, residual
    torch.cuda.empty_cache()

    confidence_data = {"positions": conf_positions, "values": conf_values}
    cosine_data = {"positions": cos_positions, "values": cos_values}

    return confidence_data, cosine_data


# ---------------------------------------------------------------------------
# 5. Plotting
# ---------------------------------------------------------------------------

def plot_confidence(
    positions: list[int],
    values: list[float],
    trace_info: dict,
    output_dir: str,
) -> str:
    """Plot probe confidence vs token position for one trace."""
    min_pos = min(positions)
    rel = [p - min_pos for p in positions]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rel, values, linewidth=0.8, color="#2c3e50", alpha=0.9)
    ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.5, label="Chance (0.5)")
    ax.axhline(y=0.9, color="#e74c3c", linestyle=":", alpha=0.4, label="High confidence (0.9)")

    gt = trace_info["ground_truth"]
    pid = trace_info["problem_id"]
    tidx = trace_info["trace_idx"]

    ax.set_xlabel("Token position (relative to think block start)", fontsize=11)
    ax.set_ylabel("Probe confidence P(Yes | h_t)", fontsize=11)
    ax.set_title(
        f"Probe Confidence -- {pid} trace_{tidx:02d} | GT={gt} | {trace_info['n_tokens']} tokens | Layer {LAYER}",
        fontsize=11,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"confidence_{pid}_trace{tidx:02d}.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cosine(
    positions: list[int],
    values: list[float],
    trace_info: dict,
    output_dir: str,
) -> str:
    """Plot cosine similarity vs token position for one trace."""
    min_pos = min(positions)
    rel = [p - min_pos for p in positions]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rel, values, linewidth=0.8, color="#8e44ad", alpha=0.9)

    mean_val = float(np.mean(values))
    ax.axhline(y=mean_val, color="grey", linestyle="--", alpha=0.5,
               label=f"Mean = {mean_val:.4f}")

    pid = trace_info["problem_id"]
    tidx = trace_info["trace_idx"]
    gt = trace_info["ground_truth"]

    ax.set_xlabel("Token position (relative to think block start)", fontsize=11)
    ax.set_ylabel("S_t = cos(h_t, h_{t-1})", fontsize=11)
    ax.set_title(
        f"Cosine Similarity -- {pid} trace_{tidx:02d} | GT={gt} | {trace_info['n_tokens']} tokens | Layer {LAYER}",
        fontsize=11,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = f"cosine_{pid}_trace{tidx:02d}.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overlay_confidence(
    all_curves: list[tuple[dict, dict]],
    output_dir: str,
) -> str:
    """Overlay all demo confidence curves on a single plot."""
    fig, ax = plt.subplots(figsize=(16, 6))
    cmap = plt.cm.tab10

    for i, (trace_info, conf_data) in enumerate(all_curves):
        positions = conf_data["positions"]
        values = conf_data["values"]
        # Normalize to [0, 1] range for overlay
        min_p, max_p = min(positions), max(positions)
        span = max_p - min_p if max_p > min_p else 1
        norm_pos = [(p - min_p) / span for p in positions]

        gt = trace_info["ground_truth"]
        pid = trace_info["problem_id"].split("_cfg")[0]  # shorter label
        color = cmap(i % 10)
        ax.plot(norm_pos, values, linewidth=0.8, color=color, alpha=0.7,
                label=f"{pid} (GT={gt})")

    ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.5)
    ax.set_xlabel("Normalized position within think block [0, 1]", fontsize=11)
    ax.set_ylabel("Probe confidence P(Yes | h_t)", fontsize=11)
    ax.set_title(
        f"Probe Confidence Curves -- {len(all_curves)} FOLIO Traces | Layer {LAYER}",
        fontsize=12,
    )
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "all_confidence_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_overlay_cosine(
    all_curves: list[tuple[dict, dict]],
    output_dir: str,
) -> str:
    """Overlay all demo cosine similarity curves on a single plot."""
    fig, ax = plt.subplots(figsize=(16, 6))
    cmap = plt.cm.tab10

    for i, (trace_info, cos_data) in enumerate(all_curves):
        positions = cos_data["positions"]
        values = cos_data["values"]
        min_p, max_p = min(positions), max(positions)
        span = max_p - min_p if max_p > min_p else 1
        norm_pos = [(p - min_p) / span for p in positions]

        gt = trace_info["ground_truth"]
        pid = trace_info["problem_id"].split("_cfg")[0]
        color = cmap(i % 10)
        ax.plot(norm_pos, values, linewidth=0.8, color=color, alpha=0.7,
                label=f"{pid} (GT={gt})")

    ax.set_xlabel("Normalized position within think block [0, 1]", fontsize=11)
    ax.set_ylabel("S_t = cos(h_t, h_{t-1})", fontsize=11)
    ax.set_title(
        f"Step-wise Cosine Similarity -- {len(all_curves)} FOLIO Traces | Layer {LAYER}",
        fontsize=12,
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "all_cosine_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t0 = time.time()
    logger.info("=" * 70)
    logger.info("MILESTONE DEMO: Layer %d Probe on FOLIO", LAYER)
    logger.info("  Traces: %s", TRACE_DIR)
    logger.info("  Output: %s", OUTPUT_DIR)
    logger.info("  GPU: %d", GPU_ID)
    logger.info("=" * 70)

    # ---- Step 1: Discover and filter traces ----
    logger.info("--- Step 1: Discovering FOLIO traces ---")
    traces = discover_traces(TRACE_DIR)
    if len(traces) < 20:
        raise RuntimeError(
            f"Only {len(traces)} valid traces found -- need at least 20 "
            f"for meaningful CV. Check {TRACE_DIR}"
        )

    # ---- Step 2: Load model ----
    logger.info("--- Step 2: Loading Qwen3-8B on cuda:%d ---", GPU_ID)
    device = torch.device(f"cuda:{GPU_ID}")
    model, tokenizer = load_model(gpu_id=GPU_ID)

    # ---- Step 3: Extract Pre-Handoff Window activations ----
    logger.info("--- Step 3: Extracting layer %d Pre-Handoff Window activations ---", LAYER)
    X, y = extract_training_activations(model, traces, LAYER, WINDOW_SIZE, device)

    # ---- Step 4: 5-fold stratified CV ----
    logger.info("--- Step 4: 5-fold stratified CV ---")
    fold_aurocs, mean_auroc = train_probe_cv(X, y, device)

    # ---- Step 5: Train final probe on all data ----
    logger.info("--- Step 5: Training final probe on all data ---")
    probe = train_probe_final(X, y, device)

    # Save probe weights as safetensors
    probe_tensors = {
        f"layer_{LAYER}_weight": probe.weight.data.cpu(),
        f"layer_{LAYER}_bias": probe.bias.data.cpu(),
    }
    probe_path = os.path.join(OUTPUT_DIR, "probe_layer24.safetensors")
    save_file(probe_tensors, probe_path)
    logger.info("Probe weights saved: %s", probe_path)

    # Save AUROC results as JSON
    auroc_results = {
        "layer": LAYER,
        "dataset": "folio",
        "n_traces": len(traces),
        "n_training_samples": int(X.shape[0]),
        "ground_truth_balance": {
            "Yes": int((y == 1).sum()),
            "No": int((y == 0).sum()),
        },
        "fold_aurocs": [round(a, 4) for a in fold_aurocs],
        "mean_auroc": round(mean_auroc, 4),
        "std_auroc": round(float(np.std(fold_aurocs)), 4),
        "n_folds": N_FOLDS,
        "hyperparams": {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "window_size": WINDOW_SIZE,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    auroc_path = os.path.join(OUTPUT_DIR, "auroc_results.json")
    with open(auroc_path, "w") as f:
        json.dump(auroc_results, f, indent=2)
    logger.info("AUROC results saved: %s", auroc_path)

    # ---- Step 6: Select demo traces ----
    # Pick 5 Yes + 5 No for balanced demo (from different problems)
    logger.info("--- Step 6: Selecting %d demo traces ---", N_DEMO_TRACES)
    yes_traces = [t for t in traces if t["ground_truth"] == "Yes"]
    no_traces = [t for t in traces if t["ground_truth"] == "No"]

    n_yes = min(N_DEMO_TRACES // 2, len(yes_traces))
    n_no = min(N_DEMO_TRACES - n_yes, len(no_traces))
    # If one class is short, take more from the other
    if n_yes + n_no < N_DEMO_TRACES:
        n_yes = min(N_DEMO_TRACES - n_no, len(yes_traces))

    demo_traces = yes_traces[:n_yes] + no_traces[:n_no]
    logger.info("Selected %d demo traces (%d Yes, %d No)", len(demo_traces), n_yes, n_no)

    # ---- Step 7: Generate per-trace plots ----
    logger.info("--- Step 7: Generating confidence and cosine plots ---")
    plots_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    overlay_confidence = []  # (trace_info, conf_data)
    overlay_cosine = []      # (trace_info, cos_data)

    for i, trace in enumerate(demo_traces):
        pid = trace["problem_id"]
        tidx = trace["trace_idx"]
        logger.info(
            "Processing demo trace %d/%d: %s trace_%02d (%d tokens, GT=%s)",
            i + 1, len(demo_traces), pid, tidx, trace["n_tokens"], trace["ground_truth"],
        )

        conf_data, cos_data = process_demo_trace(
            model, probe, trace["token_ids"], LAYER, device,
        )

        if conf_data is not None:
            path = plot_confidence(conf_data["positions"], conf_data["values"], trace, plots_dir)
            logger.info("  -> %s", path)
            overlay_confidence.append((trace, conf_data))

        if cos_data is not None:
            path = plot_cosine(cos_data["positions"], cos_data["values"], trace, plots_dir)
            logger.info("  -> %s", path)
            overlay_cosine.append((trace, cos_data))

    # Overlay plots
    if overlay_confidence:
        path = plot_overlay_confidence(overlay_confidence, plots_dir)
        logger.info("Confidence overlay: %s", path)
    if overlay_cosine:
        path = plot_overlay_cosine(overlay_cosine, plots_dir)
        logger.info("Cosine overlay: %s", path)

    # ---- Summary ----
    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("MILESTONE DEMO COMPLETE (%.1fs total)", elapsed)
    logger.info("  Layer: %d", LAYER)
    logger.info("  Dataset: folio (%d valid traces)", len(traces))
    logger.info("  Training samples: %d (window=%d)", X.shape[0], WINDOW_SIZE)
    logger.info("  MEAN AUROC (5-fold CV): %.4f +/- %.4f", mean_auroc, float(np.std(fold_aurocs)))
    logger.info("  Fold AUROCs: %s", [f"{a:.4f}" for a in fold_aurocs])
    logger.info("  Demo traces plotted: %d", len(demo_traces))
    logger.info("  Probe weights: %s", probe_path)
    logger.info("  AUROC JSON: %s", auroc_path)
    logger.info("  Plots dir: %s", plots_dir)
    logger.info("=" * 70)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Mean AUROC: {mean_auroc:.4f} +/- {float(np.std(fold_aurocs)):.4f}")
    print(f"  Folds: {[f'{a:.4f}' for a in fold_aurocs]}")
    print(f"  Training: {len(traces)} traces, {X.shape[0]} samples")
    print(f"  Probe: {probe_path}")
    print(f"  Plots: {plots_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
