"""Layer selection pilot: identify top-N layers by probe AUROC across all 36 layers.

Runs a two-stage pilot process:
1. Select ~100 traces stratified across datasets and correctness labels.
2. Extract residual stream at ALL 36 layers via chunked nnsight forward passes.
3. Train linear probes per layer with 5-fold stratified CV.
4. Compute selectivity controls (10 random-label permutations per layer).
5. Select top-N layers for downstream full-scale probe training.

Output:
  - $DATA_DIR/results/phase3/pilot_layer_aurocs.json (per-layer AUROC results)
  - $DATA_DIR/results/phase3/pilot_auroc_curve.png (AUROC vs layer plot)

Usage:
    python scripts/layer_selection_pilot.py --config configs/phase3.yaml
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import load_config
from src.model import NUM_LAYERS, load_model
from src.tokens import count_think_blocks, get_pre_handoff_window
from src.utils import set_seed, setup_logging

logger = setup_logging("layer_selection_pilot")


# ---------------------------------------------------------------------------
# 1. Trace Selection
# ---------------------------------------------------------------------------

def select_pilot_traces(traces_dir: str, n_traces: int, traces_per_dataset: int) -> list[dict]:
    """Select traces for the pilot, stratified across datasets and correctness.

    Walks $DATA_DIR/traces/valid/{dataset}/{problem_id}/ directories, loads
    trace JSON files, filters to single-think-block traces, and stratifies.

    Args:
        traces_dir: Root traces directory (contains valid/ subdirectory).
        n_traces: Target number of traces total.
        traces_per_dataset: Max traces per dataset.

    Returns:
        List of dicts: {trace_path, token_ids, metadata, ground_truth, dataset}.
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

    for dataset in datasets:
        ds_dir = os.path.join(valid_dir, dataset)
        problem_dirs = sorted(
            d for d in os.listdir(ds_dir)
            if os.path.isdir(os.path.join(ds_dir, d))
        )

        ds_traces_yes = []
        ds_traces_no = []

        for problem_id in problem_dirs:
            problem_dir = os.path.join(ds_dir, problem_id)
            trace_files = sorted(
                f for f in os.listdir(problem_dir) if f.endswith(".json")
            )

            for trace_file in trace_files:
                trace_path = os.path.join(problem_dir, trace_file)
                try:
                    with open(trace_path, "r") as f:
                        data = json.load(f)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning("Skipping unreadable trace %s: %s", trace_path, e)
                    continue

                token_ids = data["token_ids"]
                metadata = data["metadata"]

                # Filter: single think block only
                n_blocks = count_think_blocks(token_ids)
                if n_blocks != 1:
                    logger.debug(
                        "Skipping %s: %d think blocks (want 1)", trace_path, n_blocks
                    )
                    continue

                # Filter: must have Pre-Handoff Window
                window = get_pre_handoff_window(token_ids, window_size=3)
                if window is None:
                    logger.debug("Skipping %s: no Pre-Handoff Window", trace_path)
                    continue

                ground_truth = metadata["ground_truth"]
                entry = {
                    "trace_path": trace_path,
                    "token_ids": token_ids,
                    "metadata": metadata,
                    "ground_truth": ground_truth,
                    "dataset": dataset,
                }

                if ground_truth == "Yes":
                    ds_traces_yes.append(entry)
                else:
                    ds_traces_no.append(entry)

        # Stratify: balanced correctness within each dataset
        half = traces_per_dataset // 2
        selected_yes = ds_traces_yes[:half]
        selected_no = ds_traces_no[:half]
        # If one side has fewer, take more from the other
        remaining_yes = half - len(selected_yes)
        remaining_no = half - len(selected_no)
        if remaining_yes > 0:
            selected_no = ds_traces_no[: half + remaining_yes]
        if remaining_no > 0:
            selected_yes = ds_traces_yes[: half + remaining_no]

        ds_selected = selected_yes + selected_no
        logger.info(
            "Dataset %s: %d Yes + %d No = %d traces selected (from %d Yes, %d No available)",
            dataset,
            len(selected_yes),
            len(selected_no),
            len(ds_selected),
            len(ds_traces_yes),
            len(ds_traces_no),
        )
        all_traces.extend(ds_selected)

    # Trim to n_traces if we have more
    if len(all_traces) > n_traces:
        all_traces = all_traces[:n_traces]

    logger.info("Total pilot traces selected: %d", len(all_traces))

    # Log dataset distribution
    ds_counts = {}
    for t in all_traces:
        ds_counts[t["dataset"]] = ds_counts.get(t["dataset"], 0) + 1
    for ds, count in sorted(ds_counts.items()):
        logger.info("  %s: %d traces", ds, count)

    return all_traces


# ---------------------------------------------------------------------------
# 2. Activation Extraction (chunked, all 36 layers)
# ---------------------------------------------------------------------------

def extract_pilot_activations(
    model,
    traces: list[dict],
    window_size: int = 3,
    num_layers: int = NUM_LAYERS,
    chunk_size: int = 12,
) -> tuple[dict[int, list[torch.Tensor]], list[int]]:
    """Extract Pre-Handoff Window activations at all 36 layers.

    Uses chunked capture (12 layers per nnsight pass, 3 passes) to avoid OOM.
    Only keeps Pre-Handoff Window vectors (3 positions per trace).

    Args:
        model: nnsight LanguageModel.
        traces: List of trace dicts from select_pilot_traces.
        window_size: Number of tokens in Pre-Handoff Window.
        num_layers: Total number of transformer layers (36).
        chunk_size: Number of layers per nnsight pass.

    Returns:
        Tuple of:
          - Dict mapping layer_idx -> list of [window_size, 4096] float32 tensors.
          - List of labels (0 or 1) for each trace.
    """
    X_per_layer = {i: [] for i in range(num_layers)}
    y_labels = []

    device = model.device
    total = len(traces)

    for trace_idx, trace in enumerate(traces):
        token_ids = trace["token_ids"]
        ground_truth = trace["ground_truth"]

        # Compute Pre-Handoff Window
        window = get_pre_handoff_window(token_ids, window_size=window_size)
        if window is None:
            logger.warning(
                "Trace %d has no Pre-Handoff Window, skipping", trace_idx
            )
            continue

        start_pos, end_pos = window
        label = 1 if ground_truth == "Yes" else 0

        # Prepare input tensor
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        logger.info(
            "Extracting trace %d/%d (%d tokens, window [%d:%d], label=%d)",
            trace_idx + 1,
            total,
            len(token_ids),
            start_pos,
            end_pos,
            label,
        )

        # Chunked capture: 12 layers per pass
        for chunk_start in range(0, num_layers, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_layers)
            chunk_layers = set(range(chunk_start, chunk_end))

            saved = {}
            with model.trace(input_ids) as tracer:
                # MUST iterate all layers sequentially to avoid OutOfOrderError
                for idx in range(num_layers):
                    if idx in chunk_layers:
                        saved[idx] = model.model.layers[idx].output.save()

            # Extract Pre-Handoff Window vectors and move to CPU
            for idx in range(chunk_start, chunk_end):
                tensor = saved[idx].detach().cpu()
                # Shape: [1, seq_len, hidden_dim] or [seq_len, hidden_dim]
                if tensor.dim() == 3 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)  # -> [seq_len, hidden_dim]
                # Extract window positions
                window_vectors = tensor[start_pos:end_pos].float()  # [3, 4096]
                X_per_layer[idx].append(window_vectors)

            # Free GPU memory
            del saved
            torch.cuda.empty_cache()

        y_labels.append(label)

    logger.info(
        "Extraction complete: %d traces, %d labels", len(y_labels), sum(y_labels)
    )
    return X_per_layer, y_labels


# ---------------------------------------------------------------------------
# 3. Probe Training per Layer
# ---------------------------------------------------------------------------

def train_probe(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 10,
) -> tuple[nn.Linear, float]:
    """Train a linear probe on GPU with early stopping.

    Args:
        X_train: Float32 tensor [n_train, 4096] on device.
        y_train: Long tensor [n_train] (0 or 1) on device.
        X_val: Float32 tensor [n_val, 4096] on device.
        y_val: Long tensor [n_val] on device.
        device: torch.device.
        lr: Learning rate.
        weight_decay: L2 regularization.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.

    Returns:
        Tuple of (trained probe, best validation AUROC).
    """
    hidden_dim = X_train.shape[1]
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
        # Should not happen, but guard against it
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


def train_layer_probes(
    X_per_layer: dict[int, list[torch.Tensor]],
    y_labels: list[int],
    cfg,
    device: torch.device,
) -> dict[int, dict]:
    """Train probes for all layers with CV and selectivity controls.

    Args:
        X_per_layer: Dict mapping layer_idx -> list of [window_size, 4096] tensors.
        y_labels: List of labels (0 or 1) per trace.
        cfg: Phase 3 config.
        device: torch.device for training.

    Returns:
        Dict mapping layer_idx -> {real_auroc, random_auroc_mean, random_auroc_std, fold_aurocs}.
    """
    probe_cfg = cfg.phase3.probe
    select_cfg = cfg.phase3.selectivity
    num_layers = cfg.phase3.layers.num_total

    lr = probe_cfg.lr
    weight_decay = probe_cfg.weight_decay
    max_epochs = probe_cfg.max_epochs
    patience = probe_cfg.patience
    n_folds = probe_cfg.n_folds
    n_permutations = select_cfg.n_permutations
    warning_threshold = select_cfg.warning_threshold

    results = {}

    for layer_idx in range(num_layers):
        logger.info("=== Layer %d/%d ===", layer_idx, num_layers - 1)

        # Concatenate all Pre-Handoff Window vectors for this layer
        # Each entry is [window_size, 4096], we stack and flatten across traces
        layer_tensors = X_per_layer[layer_idx]
        if not layer_tensors:
            logger.warning("No data for layer %d, skipping", layer_idx)
            results[layer_idx] = {
                "real_auroc": 0.0,
                "random_auroc_mean": 0.0,
                "random_auroc_std": 0.0,
                "fold_aurocs": [],
            }
            continue

        # Stack: [n_traces, window_size, 4096] -> [n_traces * window_size, 4096]
        X_stacked = torch.cat(layer_tensors, dim=0)  # [n_traces * ws, 4096]
        # Labels: repeat each label window_size times
        window_size = layer_tensors[0].shape[0]
        y_expanded = []
        for label in y_labels:
            y_expanded.extend([label] * window_size)
        y_array = np.array(y_expanded)

        X_gpu = X_stacked.to(device)
        y_tensor = torch.tensor(y_array, dtype=torch.long, device=device)

        n_samples = X_gpu.shape[0]
        logger.info(
            "  Training data: %d samples (%d features), class balance: %d/%d",
            n_samples,
            X_gpu.shape[1],
            (y_array == 1).sum(),
            (y_array == 0).sum(),
        )

        # -- Real probes: 5-fold stratified CV --
        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=137
        )
        fold_aurocs = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_gpu.cpu().numpy(), y_array)):
            train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
            val_idx_t = torch.tensor(val_idx, dtype=torch.long, device=device)

            X_train = X_gpu[train_idx_t]
            y_train = y_tensor[train_idx_t]
            X_val = X_gpu[val_idx_t]
            y_val = y_tensor[val_idx_t]

            _, auroc = train_probe(
                X_train, y_train, X_val, y_val, device,
                lr=lr, weight_decay=weight_decay,
                max_epochs=max_epochs, patience=patience,
            )
            fold_aurocs.append(auroc)
            logger.info("  Fold %d: AUROC = %.4f", fold_idx + 1, auroc)

        mean_auroc = float(np.mean(fold_aurocs))
        logger.info("  Layer %d real AUROC: %.4f (folds: %s)", layer_idx, mean_auroc,
                     [f"{a:.4f}" for a in fold_aurocs])

        # -- Selectivity controls: random-label permutations --
        rng = np.random.RandomState(137 + layer_idx)
        random_aurocs = []

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

                _, auroc = train_probe(
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
                "  WARNING: Layer %d random AUROC mean = %.4f > %.2f threshold",
                layer_idx,
                random_mean,
                warning_threshold,
            )

        logger.info(
            "  Layer %d random AUROC: mean=%.4f, std=%.4f",
            layer_idx, random_mean, random_std,
        )

        results[layer_idx] = {
            "real_auroc": mean_auroc,
            "random_auroc_mean": random_mean,
            "random_auroc_std": random_std,
            "fold_aurocs": fold_aurocs,
        }

        # Free GPU memory for this layer before moving to next
        del X_gpu, y_tensor
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# 4. Layer Selection
# ---------------------------------------------------------------------------

def select_top_layers(
    layer_results: dict[int, dict],
    n_top: int = 6,
    num_layers: int = NUM_LAYERS,
) -> list[int]:
    """Select top-N layers based on AUROC.

    Strategy: best layer + 2 above + 2 below + 1 early layer (layer < 8 with
    best AUROC among early layers). Adjusts if best layer is near edges.

    Args:
        layer_results: Dict from train_layer_probes.
        n_top: Number of top layers to select.
        num_layers: Total number of layers.

    Returns:
        Sorted list of selected layer indices.
    """
    # Sort layers by real AUROC descending
    sorted_layers = sorted(
        layer_results.keys(),
        key=lambda idx: layer_results[idx]["real_auroc"],
        reverse=True,
    )

    best_layer = sorted_layers[0]
    logger.info(
        "Best layer: %d (AUROC=%.4f)",
        best_layer,
        layer_results[best_layer]["real_auroc"],
    )

    selected = set()
    selected.add(best_layer)

    # Add 2 layers above and 2 below the best layer
    neighbors_above = [best_layer + i for i in range(1, 3) if best_layer + i < num_layers]
    neighbors_below = [best_layer - i for i in range(1, 3) if best_layer - i >= 0]

    for layer in neighbors_above + neighbors_below:
        selected.add(layer)

    # Add 1 early layer (layer < 8 with best AUROC among early layers)
    early_layers = [
        idx for idx in range(min(8, num_layers)) if idx not in selected
    ]
    if early_layers:
        best_early = max(
            early_layers, key=lambda idx: layer_results[idx]["real_auroc"]
        )
        selected.add(best_early)

    # If we still need more layers, fill from remaining sorted by AUROC
    remaining = [idx for idx in sorted_layers if idx not in selected]
    while len(selected) < n_top and remaining:
        selected.add(remaining.pop(0))

    # If we have more than n_top (unlikely), trim least informative
    if len(selected) > n_top:
        selected_list = sorted(
            selected,
            key=lambda idx: layer_results[idx]["real_auroc"],
            reverse=True,
        )
        selected = set(selected_list[:n_top])

    result = sorted(selected)
    logger.info("Selected top-%d layers: %s", n_top, result)
    return result


# ---------------------------------------------------------------------------
# 5. Output: Results JSON and AUROC Plot
# ---------------------------------------------------------------------------

def save_results(
    layer_results: dict[int, dict],
    top_layers: list[int],
    best_layer: int,
    traces: list[dict],
    results_dir: str,
) -> str:
    """Save pilot results to JSON.

    Args:
        layer_results: Dict from train_layer_probes.
        top_layers: Selected layer indices.
        best_layer: Best single layer by AUROC.
        traces: Pilot traces used.
        results_dir: Output directory.

    Returns:
        Path to saved JSON file.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Count dataset distribution
    ds_counts = {}
    for t in traces:
        ds_counts[t["dataset"]] = ds_counts.get(t["dataset"], 0) + 1

    # Count training samples (traces * window_size)
    window_size = 3  # Pre-Handoff Window
    n_training_samples = len(traces) * window_size

    # Build per-layer results
    per_layer = {}
    for idx in sorted(layer_results.keys()):
        res = layer_results[idx]
        per_layer[str(idx)] = {
            "real_auroc": round(res["real_auroc"], 4),
            "random_auroc_mean": round(res["random_auroc_mean"], 4),
            "random_auroc_std": round(res["random_auroc_std"], 4),
        }

    output = {
        "per_layer": per_layer,
        "top_layers": top_layers,
        "best_layer": best_layer,
        "n_traces_used": len(traces),
        "n_training_samples": n_training_samples,
        "datasets_represented": ds_counts,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    output_path = os.path.join(results_dir, "pilot_layer_aurocs.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info("Results saved to %s", output_path)
    return output_path


def plot_auroc_curve(
    layer_results: dict[int, dict],
    top_layers: list[int],
    n_traces: int,
    results_dir: str,
    warning_threshold: float = 0.60,
) -> str:
    """Generate AUROC vs layer plot.

    Args:
        layer_results: Dict from train_layer_probes.
        top_layers: Selected layer indices.
        n_traces: Number of traces used in pilot.
        results_dir: Output directory.
        warning_threshold: Red dashed line threshold.

    Returns:
        Path to saved PNG file.
    """
    os.makedirs(results_dir, exist_ok=True)

    layers = sorted(layer_results.keys())
    real_aurocs = [layer_results[idx]["real_auroc"] for idx in layers]
    random_means = [layer_results[idx]["random_auroc_mean"] for idx in layers]
    random_stds = [layer_results[idx]["random_auroc_std"] for idx in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Real AUROC line
    ax.plot(layers, real_aurocs, "b-o", linewidth=2, markersize=4, label="Real AUROC")

    # Random-label band (mean +/- 2*std)
    random_means_arr = np.array(random_means)
    random_stds_arr = np.array(random_stds)
    ax.fill_between(
        layers,
        random_means_arr - 2 * random_stds_arr,
        random_means_arr + 2 * random_stds_arr,
        alpha=0.3,
        color="grey",
        label="Random-label (mean +/- 2 std)",
    )
    ax.plot(layers, random_means, "grey", linewidth=1, linestyle="--", alpha=0.7)

    # Warning threshold
    ax.axhline(
        y=warning_threshold,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Warning threshold ({warning_threshold})",
    )

    # Selected top-N layers as vertical lines
    for i, layer in enumerate(top_layers):
        label = "Selected layers" if i == 0 else None
        ax.axvline(
            x=layer,
            color="green",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=label,
        )

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title(
        f"Layer Selection Pilot: AUROC per Layer (N={n_traces} traces)",
        fontsize=14,
    )
    ax.set_xlim(-0.5, max(layers) + 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(results_dir, "pilot_auroc_curve.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("AUROC curve plot saved to %s", output_path)
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

    pilot_cfg = cfg.phase3.pilot
    n_traces = pilot_cfg.n_traces
    traces_per_dataset = pilot_cfg.traces_per_dataset

    logger.info("Layer Selection Pilot")
    logger.info("  Data dir: %s", data_dir)
    logger.info("  Traces dir: %s", traces_dir)
    logger.info("  Results dir: %s", results_dir)
    logger.info("  Target traces: %d", n_traces)
    logger.info("  GPU: %d", gpu_id)

    # 1. Select pilot traces
    logger.info("--- Step 1: Selecting pilot traces ---")
    traces = select_pilot_traces(traces_dir, n_traces, traces_per_dataset)
    if not traces:
        raise RuntimeError("No traces selected. Check trace data in %s" % traces_dir)

    # 2. Load model and extract activations
    logger.info("--- Step 2: Extracting activations at all 36 layers ---")
    model, tokenizer = load_model(gpu_id=gpu_id)

    window_size = cfg.phase3.probe.pre_handoff_window
    X_per_layer, y_labels = extract_pilot_activations(
        model, traces, window_size=window_size
    )

    # Free model from GPU to make room for probe training
    del model, tokenizer
    torch.cuda.empty_cache()

    logger.info(
        "Extracted activations: %d layers, %d traces, %d labels",
        len(X_per_layer),
        len(traces),
        len(y_labels),
    )

    # 3. Train probes per layer
    logger.info("--- Step 3: Training probes per layer ---")
    layer_results = train_layer_probes(X_per_layer, y_labels, cfg, device)

    # 4. Select top-N layers
    logger.info("--- Step 4: Selecting top-N layers ---")
    n_top = cfg.phase3.layers.n_top
    top_layers = select_top_layers(layer_results, n_top=n_top)

    # Find best layer
    best_layer = max(
        layer_results.keys(),
        key=lambda idx: layer_results[idx]["real_auroc"],
    )

    # 5. Save results and plot
    logger.info("--- Step 5: Saving results ---")
    json_path = save_results(layer_results, top_layers, best_layer, traces, results_dir)
    warning_threshold = cfg.phase3.selectivity.warning_threshold
    plot_path = plot_auroc_curve(
        layer_results, top_layers, len(traces), results_dir,
        warning_threshold=warning_threshold,
    )

    # Print summary
    logger.info("=" * 60)
    logger.info("PILOT COMPLETE")
    logger.info("  Best layer: %d (AUROC=%.4f)", best_layer, layer_results[best_layer]["real_auroc"])
    logger.info("  Random AUROC at best layer: mean=%.4f, std=%.4f",
                layer_results[best_layer]["random_auroc_mean"],
                layer_results[best_layer]["random_auroc_std"])
    logger.info("  Top-%d layers: %s", n_top, top_layers)
    logger.info("  Top layer AUROCs:")
    for idx in top_layers:
        logger.info(
            "    Layer %d: AUROC=%.4f (random=%.4f +/- %.4f)",
            idx,
            layer_results[idx]["real_auroc"],
            layer_results[idx]["random_auroc_mean"],
            layer_results[idx]["random_auroc_std"],
        )
    logger.info("  Results: %s", json_path)
    logger.info("  Plot: %s", plot_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
