"""Full probe training: per-dataset and pooled probes at top-N layers with CV and selectivity.

Builds on the layer selection pilot (scripts/layer_selection_pilot.py) by:
1. Discovering ALL valid traces (not just pilot subset).
2. Extracting Pre-Handoff Window activations at the top-N layers identified by pilot.
3. Training per-dataset and pooled probes with 5-fold stratified CV.
4. Running selectivity controls (10 random-label permutations).
5. Retraining final probes on all data and saving weights as safetensors (all 36 layers).

Output:
  - $DATA_DIR/results/phase3/probes/{dataset}_probes.safetensors (all 36 layers)
  - $DATA_DIR/results/phase3/probes/pooled_probes.safetensors (all 36 layers)
  - $DATA_DIR/results/phase3/probe_aurocs.json (per-layer per-dataset AUROC results)
  - $DATA_DIR/results/phase3/probe_auroc_comparison.png (AUROC comparison plot)

Usage:
    python scripts/train_probes.py --config configs/phase3.yaml
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from safetensors.torch import save_file
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import load_config
from src.model import HIDDEN_DIM, NUM_LAYERS, load_model
from src.tokens import count_think_blocks, get_pre_handoff_window
from src.utils import set_seed, setup_logging

logger = setup_logging("train_probes")


# ---------------------------------------------------------------------------
# 1. Trace Discovery
# ---------------------------------------------------------------------------

def discover_all_traces(traces_dir: str) -> list[dict]:
    """Discover ALL valid traces across all datasets.

    Walks $DATA_DIR/traces/valid/{dataset}/{problem_id}/ directories.
    For each problem, loads trace_00.json (primary trace per problem).
    Filters to single-think-block traces with valid Pre-Handoff Window.
    Training uses CORRECT traces only (probe learns from correct answers).

    Args:
        traces_dir: Root traces directory (contains valid/ subdirectory).

    Returns:
        List of dicts with keys: trace_path, token_ids, ground_truth, dataset,
        problem_id, is_correct.
    """
    valid_dir = os.path.join(traces_dir, "valid")
    if not os.path.isdir(valid_dir):
        raise FileNotFoundError(f"Valid traces directory not found: {valid_dir}")

    datasets = sorted(
        d for d in os.listdir(valid_dir)
        if os.path.isdir(os.path.join(valid_dir, d))
    )
    logger.info("Found datasets: %s", datasets)

    all_traces = []
    skipped_multi_block = 0
    skipped_no_window = 0
    skipped_incorrect = 0
    skipped_read_error = 0

    for dataset in datasets:
        ds_dir = os.path.join(valid_dir, dataset)
        problem_dirs = sorted(
            d for d in os.listdir(ds_dir)
            if os.path.isdir(os.path.join(ds_dir, d))
        )

        ds_traces = []

        for problem_id in problem_dirs:
            problem_dir = os.path.join(ds_dir, problem_id)

            # Load trace_00.json (primary trace per problem)
            trace_file = "trace_00.json"
            trace_path = os.path.join(problem_dir, trace_file)

            if not os.path.isfile(trace_path):
                continue

            try:
                with open(trace_path, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Skipping unreadable trace %s: %s", trace_path, e)
                skipped_read_error += 1
                continue

            token_ids = data["token_ids"]
            metadata = data["metadata"]

            # Filter: single think block only
            n_blocks = count_think_blocks(token_ids)
            if n_blocks != 1:
                skipped_multi_block += 1
                continue

            # Filter: must have Pre-Handoff Window
            window = get_pre_handoff_window(token_ids, window_size=3)
            if window is None:
                skipped_no_window += 1
                continue

            # Filter: correct traces only for training
            is_correct = metadata.get("is_correct", True)
            if not is_correct:
                skipped_incorrect += 1
                continue

            ground_truth = metadata["ground_truth"]
            entry = {
                "trace_path": trace_path,
                "token_ids": token_ids,
                "ground_truth": ground_truth,
                "dataset": dataset,
                "problem_id": problem_id,
            }
            ds_traces.append(entry)

        logger.info("Dataset %s: %d valid traces from %d problems",
                     dataset, len(ds_traces), len(problem_dirs))
        all_traces.extend(ds_traces)

    logger.info("Total valid traces: %d", len(all_traces))
    logger.info("Skipped: %d multi-block, %d no-window, %d incorrect, %d read-error",
                 skipped_multi_block, skipped_no_window, skipped_incorrect, skipped_read_error)

    # Report counts per dataset
    ds_counts = {}
    gt_counts = {"Yes": 0, "No": 0}
    for t in all_traces:
        ds = t["dataset"]
        ds_counts[ds] = ds_counts.get(ds, 0) + 1
        gt_counts[t["ground_truth"]] = gt_counts.get(t["ground_truth"], 0) + 1

    for ds, count in sorted(ds_counts.items()):
        logger.info("  %s: %d traces", ds, count)
    logger.info("  Ground truth balance: Yes=%d, No=%d", gt_counts["Yes"], gt_counts["No"])

    return all_traces


# ---------------------------------------------------------------------------
# 2. Activation Extraction at Top-N Layers
# ---------------------------------------------------------------------------

def extract_training_activations(
    model,
    traces: list[dict],
    top_layers: list[int],
    window_size: int = 3,
    num_layers: int = NUM_LAYERS,
) -> dict:
    """Extract Pre-Handoff Window activations at top-N layers for all traces.

    Runs a fresh nnsight forward pass per trace. Since we only need top-N layers
    (e.g. 6), a single pass suffices -- no chunking needed.

    IMPORTANT: Access ALL layers sequentially (for idx in range(36)) but only
    .save() the ones in top_layers. This avoids nnsight OutOfOrderError.

    Args:
        model: nnsight LanguageModel.
        traces: List of trace dicts from discover_all_traces.
        top_layers: Layer indices to extract (from pilot results).
        window_size: Number of tokens in Pre-Handoff Window.
        num_layers: Total number of transformer layers (36).

    Returns:
        Dict with structure:
          training_data[layer][dataset] = {"X": list of vectors, "y": list of labels}
          training_data[layer]["pooled"] = {"X": list of vectors, "y": list of labels}
    """
    top_layers_set = set(top_layers)

    # Initialize storage: layer -> scope -> {X, y}
    training_data = {}
    for layer in top_layers:
        training_data[layer] = {}
        # Per-dataset scopes
        datasets_seen = set(t["dataset"] for t in traces)
        for ds in datasets_seen:
            training_data[layer][ds] = {"X": [], "y": []}
        training_data[layer]["pooled"] = {"X": [], "y": []}

    device = model.device
    total = len(traces)
    extraction_start = time.time()

    for trace_idx, trace in enumerate(traces):
        token_ids = trace["token_ids"]
        ground_truth = trace["ground_truth"]
        dataset = trace["dataset"]

        # Compute Pre-Handoff Window
        window = get_pre_handoff_window(token_ids, window_size=window_size)
        if window is None:
            logger.warning("Trace %d has no Pre-Handoff Window, skipping", trace_idx)
            continue

        start_pos, end_pos = window
        label = 1 if ground_truth == "Yes" else 0

        # Prepare input tensor
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        if (trace_idx + 1) % 10 == 0 or trace_idx == 0:
            logger.info(
                "Extracting trace %d/%d (%d tokens, window [%d:%d], label=%d, dataset=%s)",
                trace_idx + 1, total, len(token_ids),
                start_pos, end_pos, label, dataset,
            )

        # Single nnsight pass -- save only top-N layers
        saved = {}
        with model.trace(input_ids) as tracer:
            # MUST iterate all layers sequentially to avoid OutOfOrderError
            for idx in range(num_layers):
                if idx in top_layers_set:
                    saved[idx] = model.model.layers[idx].output.save()

        # Extract Pre-Handoff Window vectors
        for layer in top_layers:
            tensor = saved[layer].detach().cpu()
            # Shape: [1, seq_len, hidden_dim] or [seq_len, hidden_dim]
            if tensor.dim() == 3 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)  # -> [seq_len, hidden_dim]
            # Extract window positions and cast to float32
            window_vectors = tensor[start_pos:end_pos].float()  # [window_size, hidden_dim]

            # Per-dataset
            if dataset not in training_data[layer]:
                training_data[layer][dataset] = {"X": [], "y": []}
            training_data[layer][dataset]["X"].append(window_vectors)
            training_data[layer][dataset]["y"].extend([label] * window_vectors.shape[0])

            # Pooled
            training_data[layer]["pooled"]["X"].append(window_vectors)
            training_data[layer]["pooled"]["y"].extend([label] * window_vectors.shape[0])

        # Free GPU memory
        del saved, input_ids
        torch.cuda.empty_cache()

    elapsed = time.time() - extraction_start
    logger.info("Extraction complete: %d traces in %.1fs (%.2f traces/s)",
                 total, elapsed, total / elapsed if elapsed > 0 else 0)

    # Log data sizes
    for layer in top_layers:
        for scope, data in training_data[layer].items():
            n = len(data["y"])
            if n > 0:
                balance = sum(data["y"]) / n
                logger.info("  Layer %d, %s: %d samples (%.1f%% positive)",
                             layer, scope, n, balance * 100)

    return training_data


# ---------------------------------------------------------------------------
# 3. Probe Training with CV
# ---------------------------------------------------------------------------

def _train_single_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    hidden_dim: int = HIDDEN_DIM,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[nn.Linear, float]:
    """Train a linear probe with early stopping and return AUROC.

    Args:
        X_train: Float32 tensor [n_train, hidden_dim] on device.
        y_train: Long tensor [n_train] on device.
        X_val: Float32 tensor [n_val, hidden_dim] on device.
        y_val: Long tensor [n_val] on device.
        device: torch.device.
        hidden_dim: Input feature dimension (4096).
        lr: Learning rate.
        weight_decay: L2 regularization.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.

    Returns:
        Tuple of (trained probe, best validation AUROC).
    """
    probe = nn.Linear(hidden_dim, 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
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
            if patience_counter >= patience:
                break

    if best_state is None:
        best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    probe.eval()

    # Compute AUROC
    with torch.no_grad():
        val_probs = torch.softmax(probe(X_val), dim=-1)[:, 1].cpu().numpy()

    y_val_np = y_val.cpu().numpy()

    # Handle degenerate case: only one class in validation set
    unique_labels = np.unique(y_val_np)
    if len(unique_labels) < 2:
        logger.warning("Only one class in validation fold, returning AUROC=0.5")
        return probe, 0.5

    auroc = roc_auc_score(y_val_np, val_probs)
    return probe, auroc


def _train_probe_full_data(
    X: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    hidden_dim: int = HIDDEN_DIM,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
) -> nn.Linear:
    """Train a linear probe on ALL data (no validation split) for final weights.

    Uses fixed epoch count (no early stopping since no val set).

    Args:
        X: Float32 tensor [n_samples, hidden_dim] on device.
        y: Long tensor [n_samples] on device.
        device: torch.device.
        hidden_dim: Input feature dimension (4096).
        lr: Learning rate.
        weight_decay: L2 regularization.
        max_epochs: Training epochs.

    Returns:
        Trained nn.Linear probe.
    """
    probe = nn.Linear(hidden_dim, 2).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(max_epochs):
        probe.train()
        logits = probe(X)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    return probe


def train_probes_cv(
    training_data: dict,
    top_layers: list[int],
    cfg,
    device: torch.device,
) -> dict:
    """Train per-dataset and pooled probes with 5-fold stratified CV.

    Args:
        training_data: Dict from extract_training_activations.
        top_layers: Layer indices.
        cfg: Phase 3 config.
        device: torch.device for training.

    Returns:
        Dict with structure:
          results[scope][layer] = {
            "real_auroc_mean": float,
            "real_auroc_folds": list[float],
          }
    """
    probe_cfg = cfg.phase3.probe
    lr = probe_cfg.lr
    weight_decay = probe_cfg.weight_decay
    max_epochs = probe_cfg.max_epochs
    patience = probe_cfg.patience
    n_folds = probe_cfg.n_folds

    results = {}

    for layer in top_layers:
        layer_data = training_data[layer]

        for scope, data in sorted(layer_data.items()):
            if scope not in results:
                results[scope] = {}

            X_list = data["X"]
            y_list = data["y"]

            if not X_list:
                logger.warning("No data for layer %d, scope %s", layer, scope)
                results[scope][layer] = {
                    "real_auroc_mean": 0.0,
                    "real_auroc_folds": [],
                }
                continue

            # Concatenate vectors: list of [ws, dim] -> [n_samples, dim]
            X_cat = torch.cat(X_list, dim=0)
            y_array = np.array(y_list)

            X_gpu = X_cat.to(device)
            y_tensor = torch.tensor(y_array, dtype=torch.long, device=device)

            n_samples = X_gpu.shape[0]
            logger.info(
                "=== Layer %d, scope %s: %d samples (%.1f%% positive) ===",
                layer, scope, n_samples,
                (y_array == 1).sum() / n_samples * 100,
            )

            # 5-fold stratified CV
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=137)
            fold_aurocs = []

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(
                X_gpu.cpu().numpy(), y_array
            )):
                train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
                val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

                X_train = X_gpu[train_idx_t]
                y_train = y_tensor[train_idx_t]
                X_val = X_gpu[val_idx_t]
                y_val = y_tensor[val_idx_t]

                _, auroc = _train_single_probe(
                    X_train, y_train, X_val, y_val, device,
                    lr=lr, weight_decay=weight_decay,
                    max_epochs=max_epochs, patience=patience,
                )
                fold_aurocs.append(auroc)
                logger.info("  Fold %d: AUROC = %.4f", fold_idx + 1, auroc)

            mean_auroc = float(np.mean(fold_aurocs))
            logger.info("  Layer %d %s: mean AUROC = %.4f", layer, scope, mean_auroc)

            results[scope][layer] = {
                "real_auroc_mean": mean_auroc,
                "real_auroc_folds": [float(a) for a in fold_aurocs],
            }

            del X_gpu, y_tensor
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# 4. Selectivity Controls
# ---------------------------------------------------------------------------

def train_selectivity_controls(
    training_data: dict,
    top_layers: list[int],
    cfg,
    device: torch.device,
) -> dict:
    """Run random-label permutation controls for each scope x layer.

    Args:
        training_data: Dict from extract_training_activations.
        top_layers: Layer indices.
        cfg: Phase 3 config.
        device: torch.device.

    Returns:
        Dict with structure:
          controls[scope][layer] = {
            "random_auroc_mean": float,
            "random_auroc_std": float,
            "random_aurocs": list[float],
          }
    """
    probe_cfg = cfg.phase3.probe
    select_cfg = cfg.phase3.selectivity
    lr = probe_cfg.lr
    weight_decay = probe_cfg.weight_decay
    max_epochs = probe_cfg.max_epochs
    patience = probe_cfg.patience
    n_folds = probe_cfg.n_folds
    n_permutations = select_cfg.n_permutations
    warning_threshold = select_cfg.warning_threshold

    controls = {}

    for layer in top_layers:
        layer_data = training_data[layer]

        for scope, data in sorted(layer_data.items()):
            if scope not in controls:
                controls[scope] = {}

            X_list = data["X"]
            y_list = data["y"]

            if not X_list:
                controls[scope][layer] = {
                    "random_auroc_mean": 0.0,
                    "random_auroc_std": 0.0,
                    "random_aurocs": [],
                }
                continue

            X_cat = torch.cat(X_list, dim=0)
            y_array = np.array(y_list)

            X_gpu = X_cat.to(device)

            logger.info("Selectivity controls: Layer %d, scope %s (%d permutations)",
                         layer, scope, n_permutations)

            rng = np.random.RandomState(137 + layer)
            random_aurocs = []

            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=137)

            for perm_idx in range(n_permutations):
                y_shuffled = y_array.copy()
                rng.shuffle(y_shuffled)
                y_shuffled_tensor = torch.tensor(y_shuffled, dtype=torch.long, device=device)

                perm_fold_aurocs = []
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(
                    X_gpu.cpu().numpy(), y_shuffled
                )):
                    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
                    val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

                    X_train = X_gpu[train_idx_t]
                    y_train = y_shuffled_tensor[train_idx_t]
                    X_val = X_gpu[val_idx_t]
                    y_val = y_shuffled_tensor[val_idx_t]

                    _, auroc = _train_single_probe(
                        X_train, y_train, X_val, y_val, device,
                        lr=lr, weight_decay=weight_decay,
                        max_epochs=max_epochs, patience=patience,
                    )
                    perm_fold_aurocs.append(auroc)

                perm_mean = float(np.mean(perm_fold_aurocs))
                random_aurocs.append(perm_mean)

            random_mean = float(np.mean(random_aurocs))
            random_std = float(np.std(random_aurocs))

            if random_mean > warning_threshold:
                logger.warning(
                    "  WARNING: Layer %d %s random AUROC mean = %.4f > %.2f threshold",
                    layer, scope, random_mean, warning_threshold,
                )

            logger.info(
                "  Layer %d %s: random AUROC mean=%.4f, std=%.4f",
                layer, scope, random_mean, random_std,
            )

            controls[scope][layer] = {
                "random_auroc_mean": random_mean,
                "random_auroc_std": random_std,
                "random_aurocs": [float(a) for a in random_aurocs],
            }

            del X_gpu
            torch.cuda.empty_cache()

    return controls


# ---------------------------------------------------------------------------
# 5. Final Probe Training (all 36 layers)
# ---------------------------------------------------------------------------

def train_final_probes(
    model,
    traces: list[dict],
    top_layers: list[int],
    cfg,
    device: torch.device,
    probes_dir: str,
) -> None:
    """Retrain final probes on ALL data and save weights for all 36 layers.

    For top-N layers: uses the full training data (no train/val split).
    For remaining 30 layers: trains quick probes (same procedure, single run).
    All 36 layers saved as safetensors per dataset and pooled.

    Args:
        model: nnsight LanguageModel (for extracting non-top-N layers).
        traces: All valid traces.
        top_layers: Top-N layer indices from pilot.
        cfg: Phase 3 config.
        device: torch.device.
        probes_dir: Directory to save probe safetensors.
    """
    probe_cfg = cfg.phase3.probe
    lr = probe_cfg.lr
    weight_decay = probe_cfg.weight_decay
    max_epochs = probe_cfg.max_epochs
    window_size = probe_cfg.pre_handoff_window
    num_layers = cfg.phase3.layers.num_total
    top_layers_set = set(top_layers)
    hidden_dim = HIDDEN_DIM

    os.makedirs(probes_dir, exist_ok=True)

    # Determine all scopes (datasets + pooled)
    datasets_seen = sorted(set(t["dataset"] for t in traces))
    scopes = datasets_seen + ["pooled"]

    logger.info("Training final probes for all 36 layers across %d scopes", len(scopes))

    # We need to process ALL 36 layers. For top-N layers, we already have
    # extracted data. For the remaining layers, we extract on-the-fly.
    # Strategy: extract all 36 layers in one pass per trace (chunked to avoid OOM).

    # Initialize storage: layer -> scope -> {X, y}
    all_layer_data = {}
    for layer_idx in range(num_layers):
        all_layer_data[layer_idx] = {}
        for ds in datasets_seen:
            all_layer_data[layer_idx][ds] = {"X": [], "y": []}
        all_layer_data[layer_idx]["pooled"] = {"X": [], "y": []}

    # Extract all 36 layers for all traces (chunked: 12 layers per pass)
    chunk_size = 12
    total = len(traces)
    extraction_start = time.time()

    for trace_idx, trace in enumerate(traces):
        token_ids = trace["token_ids"]
        ground_truth = trace["ground_truth"]
        dataset = trace["dataset"]

        window = get_pre_handoff_window(token_ids, window_size=window_size)
        if window is None:
            continue

        start_pos, end_pos = window
        label = 1 if ground_truth == "Yes" else 0

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=model.device)

        if (trace_idx + 1) % 10 == 0 or trace_idx == 0:
            logger.info(
                "Final extraction: trace %d/%d (%d tokens, dataset=%s)",
                trace_idx + 1, total, len(token_ids), dataset,
            )

        # Chunked capture: 12 layers per pass
        for chunk_start in range(0, num_layers, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_layers)
            chunk_layers = set(range(chunk_start, chunk_end))

            saved = {}
            with model.trace(input_ids) as tracer:
                for idx in range(num_layers):
                    if idx in chunk_layers:
                        saved[idx] = model.model.layers[idx].output.save()

            for idx in range(chunk_start, chunk_end):
                tensor = saved[idx].detach().cpu()
                if tensor.dim() == 3 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                window_vectors = tensor[start_pos:end_pos].float()

                all_layer_data[idx][dataset]["X"].append(window_vectors)
                all_layer_data[idx][dataset]["y"].extend([label] * window_vectors.shape[0])

                all_layer_data[idx]["pooled"]["X"].append(window_vectors)
                all_layer_data[idx]["pooled"]["y"].extend([label] * window_vectors.shape[0])

            del saved
            torch.cuda.empty_cache()

        del input_ids
        torch.cuda.empty_cache()

    elapsed = time.time() - extraction_start
    logger.info("Final extraction complete: %d traces in %.1fs", total, elapsed)

    # Free model from GPU before training
    del model
    torch.cuda.empty_cache()

    # Train final probes for each scope
    for scope in scopes:
        logger.info("Training final probes for scope: %s", scope)
        probe_tensors = {}

        for layer_idx in range(num_layers):
            data = all_layer_data[layer_idx][scope]
            X_list = data["X"]
            y_list = data["y"]

            if not X_list:
                logger.warning("No data for layer %d, scope %s -- saving zero weights", layer_idx, scope)
                zero_probe = nn.Linear(hidden_dim, 2)
                probe_tensors[f"layer_{layer_idx}_weight"] = zero_probe.weight.data.cpu()
                probe_tensors[f"layer_{layer_idx}_bias"] = zero_probe.bias.data.cpu()
                continue

            X_cat = torch.cat(X_list, dim=0).to(device)
            y_tensor = torch.tensor(y_list, dtype=torch.long, device=device)

            probe = _train_probe_full_data(
                X_cat, y_tensor, device,
                hidden_dim=hidden_dim,
                lr=lr,
                weight_decay=weight_decay,
                max_epochs=max_epochs,
            )

            # Store weights
            probe_tensors[f"layer_{layer_idx}_weight"] = probe.weight.data.cpu()
            probe_tensors[f"layer_{layer_idx}_bias"] = probe.bias.data.cpu()

            if layer_idx in top_layers_set:
                logger.info("  Layer %d (top-N): trained on %d samples", layer_idx, X_cat.shape[0])
            else:
                if (layer_idx + 1) % 6 == 0:
                    logger.info("  Layer %d (non-top): trained on %d samples", layer_idx, X_cat.shape[0])

            del X_cat, y_tensor, probe
            torch.cuda.empty_cache()

        # Save safetensors
        output_path = os.path.join(probes_dir, f"{scope}_probes.safetensors")
        save_file(probe_tensors, output_path)
        logger.info("Saved %s (%d tensors)", output_path, len(probe_tensors))

    logger.info("All final probe weights saved to %s", probes_dir)


# ---------------------------------------------------------------------------
# 6. Results Output
# ---------------------------------------------------------------------------

def save_results(
    cv_results: dict,
    controls: dict,
    top_layers: list[int],
    best_layer: int,
    traces: list[dict],
    results_dir: str,
) -> str:
    """Save probe AUROC results to JSON.

    Args:
        cv_results: Dict from train_probes_cv.
        controls: Dict from train_selectivity_controls.
        top_layers: Selected layer indices.
        best_layer: Best single layer by AUROC.
        traces: All traces used.
        results_dir: Output directory.

    Returns:
        Path to saved JSON file.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Count per-dataset and training samples
    ds_counts = {}
    for t in traces:
        ds_counts[t["dataset"]] = ds_counts.get(t["dataset"], 0) + 1

    window_size = 3
    n_training_samples = {ds: count * window_size for ds, count in ds_counts.items()}

    # Build output structure
    output = {
        "per_dataset": {},
        "pooled": {},
        "top_layers": top_layers,
        "best_layer": best_layer,
        "n_traces": ds_counts,
        "n_training_samples": n_training_samples,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Merge CV and selectivity results
    for scope in cv_results:
        if scope == "pooled":
            target = output["pooled"]
        else:
            output["per_dataset"][scope] = {}
            target = output["per_dataset"][scope]

        for layer in top_layers:
            cv = cv_results[scope].get(layer, {})
            ctrl = controls.get(scope, {}).get(layer, {})

            target[str(layer)] = {
                "real_auroc_mean": round(cv.get("real_auroc_mean", 0.0), 4),
                "real_auroc_folds": [round(a, 4) for a in cv.get("real_auroc_folds", [])],
                "random_auroc_mean": round(ctrl.get("random_auroc_mean", 0.0), 4),
                "random_auroc_std": round(ctrl.get("random_auroc_std", 0.0), 4),
            }

    output_path = os.path.join(results_dir, "probe_aurocs.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", output_path)
    return output_path


def plot_auroc_comparison(
    cv_results: dict,
    controls: dict,
    top_layers: list[int],
    n_traces: int,
    results_dir: str,
) -> str:
    """Generate AUROC comparison plot: per-dataset vs pooled at top-N layers.

    Args:
        cv_results: Dict from train_probes_cv.
        controls: Dict from train_selectivity_controls.
        top_layers: Top-N layer indices.
        n_traces: Total number of traces.
        results_dir: Output directory.

    Returns:
        Path to saved PNG file.
    """
    os.makedirs(results_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort layers for x-axis
    layers_sorted = sorted(top_layers)
    x_positions = list(range(len(layers_sorted)))

    # Collect scopes (datasets + pooled)
    scopes = sorted(s for s in cv_results if s != "pooled") + ["pooled"]

    colors = {"gsm_symbolic": "#1f77b4", "folio": "#ff7f0e", "strategyqa": "#2ca02c", "pooled": "#d62728"}
    markers = {"gsm_symbolic": "o", "folio": "s", "strategyqa": "^", "pooled": "D"}

    for scope in scopes:
        aurocs = []
        for layer in layers_sorted:
            cv = cv_results.get(scope, {}).get(layer, {})
            aurocs.append(cv.get("real_auroc_mean", 0.0))

        color = colors.get(scope, "#999999")
        marker = markers.get(scope, "o")
        linewidth = 2.5 if scope == "pooled" else 1.5
        label = scope.replace("_", " ").title() if scope != "pooled" else "Pooled"

        ax.plot(
            x_positions, aurocs, marker=marker, color=color,
            linewidth=linewidth, markersize=8, label=label,
        )

    # Random-label band from pooled controls
    pooled_ctrl = controls.get("pooled", {})
    if pooled_ctrl:
        random_means = []
        random_stds = []
        for layer in layers_sorted:
            ctrl = pooled_ctrl.get(layer, {})
            random_means.append(ctrl.get("random_auroc_mean", 0.5))
            random_stds.append(ctrl.get("random_auroc_std", 0.0))

        random_means_arr = np.array(random_means)
        random_stds_arr = np.array(random_stds)
        ax.fill_between(
            x_positions,
            random_means_arr - 2 * random_stds_arr,
            random_means_arr + 2 * random_stds_arr,
            alpha=0.2, color="grey", label="Random-label (mean +/- 2 std)",
        )
        ax.plot(x_positions, random_means, color="grey", linewidth=1, linestyle="--", alpha=0.7)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(l) for l in layers_sorted])
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title(
        f"Probe AUROC: Per-Dataset vs Pooled (top-{len(top_layers)} layers, N={n_traces} traces)",
        fontsize=14,
    )
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(results_dir, "probe_auroc_comparison.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("AUROC comparison plot saved to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    cfg = load_config()
    seed = cfg.get("seed", 137)
    set_seed(seed)

    gpu_id = cfg.gpus[0] if hasattr(cfg, "gpus") and cfg.gpus else 0
    device = torch.device(f"cuda:{gpu_id}")

    data_dir = os.environ.get("DATA_DIR")
    if not data_dir:
        raise RuntimeError("DATA_DIR environment variable is not set. Check .env file.")

    traces_dir = os.path.join(data_dir, "traces")
    results_dir = os.path.join(data_dir, "results", "phase3")
    probes_dir = os.path.join(data_dir, "results", "phase3", "probes")

    # Load pilot results to get top-N layers
    pilot_path = os.path.join(results_dir, "pilot_layer_aurocs.json")
    if not os.path.isfile(pilot_path):
        raise FileNotFoundError(
            f"Pilot results not found at {pilot_path}. "
            "Run scripts/layer_selection_pilot.py first."
        )

    with open(pilot_path, "r") as f:
        pilot_results = json.load(f)

    top_layers = pilot_results["top_layers"]
    best_layer = pilot_results["best_layer"]
    logger.info("Loaded pilot results: top_layers=%s, best_layer=%d", top_layers, best_layer)

    logger.info("Full Probe Training")
    logger.info("  Data dir: %s", data_dir)
    logger.info("  Traces dir: %s", traces_dir)
    logger.info("  Results dir: %s", results_dir)
    logger.info("  Probes dir: %s", probes_dir)
    logger.info("  Top layers: %s", top_layers)
    logger.info("  GPU: %d", gpu_id)

    # Step 1: Discover all valid traces
    logger.info("--- Step 1: Discovering all valid traces ---")
    traces = discover_all_traces(traces_dir)
    if not traces:
        raise RuntimeError("No valid traces found. Check trace data in %s" % traces_dir)

    # Step 2: Load model and extract activations at top-N layers
    logger.info("--- Step 2: Extracting activations at top-%d layers ---", len(top_layers))
    model, tokenizer = load_model(gpu_id=gpu_id)

    window_size = cfg.phase3.probe.pre_handoff_window
    training_data = extract_training_activations(
        model, traces, top_layers, window_size=window_size,
    )

    # Free model for probe training
    del model, tokenizer
    torch.cuda.empty_cache()

    # Step 3: Train probes with CV
    logger.info("--- Step 3: Training probes with 5-fold stratified CV ---")
    cv_results = train_probes_cv(training_data, top_layers, cfg, device)

    # Step 4: Run selectivity controls
    logger.info("--- Step 4: Running selectivity controls ---")
    controls = train_selectivity_controls(training_data, top_layers, cfg, device)

    # Free training data to make room for final probe training
    del training_data
    torch.cuda.empty_cache()

    # Step 5: Train final probes on all data (all 36 layers)
    logger.info("--- Step 5: Training final probes for all 36 layers ---")
    model2, _ = load_model(gpu_id=gpu_id)
    train_final_probes(model2, traces, top_layers, cfg, device, probes_dir)
    # model2 is freed inside train_final_probes

    # Step 6: Save results and plot
    logger.info("--- Step 6: Saving results ---")
    json_path = save_results(cv_results, controls, top_layers, best_layer, traces, results_dir)
    plot_path = plot_auroc_comparison(
        cv_results, controls, top_layers, len(traces), results_dir,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("FULL PROBE TRAINING COMPLETE")
    logger.info("  Top layers: %s", top_layers)
    logger.info("  Best layer (from pilot): %d", best_layer)

    # Best per-dataset
    datasets = sorted(set(t["dataset"] for t in traces))
    for ds in datasets:
        ds_results = cv_results.get(ds, {})
        if ds_results:
            best_ds_layer = max(ds_results, key=lambda l: ds_results[l]["real_auroc_mean"])
            logger.info(
                "  %s best: layer %d (AUROC=%.4f)",
                ds, best_ds_layer, ds_results[best_ds_layer]["real_auroc_mean"],
            )

    # Pooled
    pooled_results = cv_results.get("pooled", {})
    if pooled_results:
        best_pooled_layer = max(pooled_results, key=lambda l: pooled_results[l]["real_auroc_mean"])
        logger.info(
            "  Pooled best: layer %d (AUROC=%.4f)",
            best_pooled_layer, pooled_results[best_pooled_layer]["real_auroc_mean"],
        )

    # Selectivity warnings
    any_warning = False
    warning_threshold = cfg.phase3.selectivity.warning_threshold
    for scope in controls:
        for layer in controls[scope]:
            ctrl = controls[scope][layer]
            if ctrl["random_auroc_mean"] > warning_threshold:
                logger.warning(
                    "  SELECTIVITY WARNING: %s layer %d random AUROC = %.4f",
                    scope, layer, ctrl["random_auroc_mean"],
                )
                any_warning = True
    if not any_warning:
        logger.info("  No selectivity warnings (all random AUROCs below %.2f)", warning_threshold)

    logger.info("  Results: %s", json_path)
    logger.info("  Plot: %s", plot_path)
    logger.info("  Probes: %s", probes_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
