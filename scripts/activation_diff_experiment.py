"""Activation diffing: derive a per-layer 'knowing direction'.

For each trace, force-answers at the first N_EARLY and last N_LATE thinking
token positions while capturing residual-stream activations via nnsight.
Traces where the model gets all early probes wrong and all late probes right
are kept. The mean late activation minus mean early activation, averaged
across kept traces, yields one direction vector per layer.

The direction can then be used to project any token's residual stream onto
it, producing a scalar "knowledge score" that tracks how much the model
knows the answer at each point in its reasoning.

Usage:
    python scripts/activation_diff_experiment.py \
        --config configs/activation_diff.yaml --gpus 0,1,2,3
"""

import json
import logging
import os
import time

import pandas as pd
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from src.config import load_config
from src.model import HIDDEN_DIM, NUM_LAYERS, load_model
from src.multiprocess import run_parallel
from src.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token constants (same as force_answer_experiment.py)
# ---------------------------------------------------------------------------
MU_TOKEN_ID = 39120              # "Mu" (bare, single token -- after ** prefix)
NU_TOKEN_ID = 60335              # "Nu" (bare, single token -- after ** prefix)
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_all_traces(traces_dir, trace_index=0, min_think_length=20):
    """Load all valid single-think-block traces across datasets.

    Unlike load_traces_for_experiment (force_answer), this loads ALL valid
    traces without balanced selection -- we want maximum data for computing
    a robust direction.

    Args:
        traces_dir: Root traces directory (contains valid/ subdirectory).
        trace_index: Which trace file to use per problem (default 0).
        min_think_length: Minimum thinking tokens to include a trace.

    Returns:
        List of trace dicts with keys: trace_path, token_ids, metadata,
        dataset, expected_token, think_start, think_length.
    """
    valid_dir = os.path.join(traces_dir, "valid")
    datasets = ["gsm_symbolic", "folio", "strategyqa"]
    traces = []
    skipped = 0

    for ds in datasets:
        ds_dir = os.path.join(valid_dir, ds)
        if not os.path.isdir(ds_dir):
            logger.warning("Dataset directory not found: %s", ds_dir)
            continue

        for problem_id in sorted(os.listdir(ds_dir)):
            problem_dir = os.path.join(ds_dir, problem_id)
            if not os.path.isdir(problem_dir):
                continue

            trace_path = os.path.join(
                problem_dir, f"trace_{trace_index:02d}.json"
            )
            if not os.path.isfile(trace_path):
                continue

            with open(trace_path) as f:
                trace = json.load(f)

            meta = trace["metadata"]

            if not meta.get("has_think", False):
                skipped += 1
                continue
            if meta.get("num_think_blocks", 0) != 1:
                skipped += 1
                continue

            think_start, think_end = meta["think_boundaries"][0]
            think_length = think_end - think_start - 1

            if think_length < min_think_length:
                skipped += 1
                continue

            ground_truth = meta["ground_truth"]
            valid_token = meta["valid_token"]
            invalid_token = meta["invalid_token"]
            expected_token = valid_token if ground_truth == "Yes" else invalid_token

            traces.append({
                "trace_path": trace_path,
                "token_ids": trace["token_ids"],
                "metadata": meta,
                "dataset": ds,
                "expected_token": expected_token,
                "think_start": think_start,
                "think_length": think_length,
            })

    if skipped:
        logger.info(
            "Skipped %d traces (no think, multiple blocks, or too short)",
            skipped,
        )

    return traces


# ---------------------------------------------------------------------------
# Per-trace probing with activation capture
# ---------------------------------------------------------------------------

def probe_trace(model, trace_data, n_early, n_late, device):
    """Probe one trace at first N and last N thinking positions.

    For each probe position, runs a single nnsight forward pass on
    prefix + thinking[:pos] + close_suffix. Thanks to causal masking,
    the residual stream at the last thinking token is identical to what
    it would be in the original trace. The logits at the final position
    (after the Answer: suffix) give P(Mu) and P(Nu).

    Args:
        model: nnsight LanguageModel.
        trace_data: Trace dict from load_all_traces.
        n_early: Number of early probe positions (1, 2, ..., n_early).
        n_late: Number of late probe positions.
        device: Torch device string.

    Returns:
        Dict with correctness info and residual tensors, or None on error.
    """
    token_ids = trace_data["token_ids"]
    think_start = trace_data["think_start"]
    think_length = trace_data["think_length"]
    expected_token = trace_data["expected_token"]
    meta = trace_data["metadata"]

    prefix_ids = token_ids[:think_start + 1]
    think_content_ids = token_ids[think_start + 1 : think_start + 1 + think_length]
    full_input_ids = prefix_ids + think_content_ids

    # Probe positions: pos=k means k thinking tokens seen before suffix
    early_positions = list(range(1, n_early + 1))
    late_positions = list(range(
        think_length - n_late + 1, think_length + 1
    ))
    all_positions = early_positions + late_positions

    all_residuals = []
    all_correct = []
    all_mu_probs = []
    all_nu_probs = []

    for pos in all_positions:
        cache_len = len(prefix_ids) + pos
        input_ids_for_probe = full_input_ids[:cache_len] + CLOSE_TOKEN_IDS
        input_tensor = torch.tensor(
            [input_ids_for_probe], dtype=torch.long, device=device,
        )

        # Last thinking token position (causal: same as in original trace)
        residual_pos = cache_len - 1

        saved = {}
        with model.trace(input_tensor):
            for idx in range(NUM_LAYERS):
                saved[idx] = (
                    model.model.layers[idx].output[:, residual_pos, :].save()
                )
            saved_logits = model.lm_head.output[:, -1, :].save()

        # Stack layer residuals -> [NUM_LAYERS, HIDDEN_DIM]
        layers = []
        for idx in range(NUM_LAYERS):
            r = saved[idx].detach().cpu().to(torch.float32).squeeze(0)
            layers.append(r)
        all_residuals.append(torch.stack(layers))

        # Correctness from logits
        logits = saved_logits.detach().cpu().float().squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        mu_prob = probs[MU_TOKEN_ID].item()
        nu_prob = probs[NU_TOKEN_ID].item()
        answer = "Mu" if mu_prob >= nu_prob else "Nu"

        all_correct.append(answer == expected_token)
        all_mu_probs.append(mu_prob)
        all_nu_probs.append(nu_prob)

        del saved, saved_logits

    residuals_tensor = torch.stack(all_residuals)

    return {
        "problem_id": meta["problem_id"],
        "dataset": trace_data["dataset"],
        "expected_token": expected_token,
        "think_length": think_length,
        "early_correct": all_correct[:n_early],
        "late_correct": all_correct[n_early:],
        "early_mu_probs": all_mu_probs[:n_early],
        "early_nu_probs": all_nu_probs[:n_early],
        "late_mu_probs": all_mu_probs[n_early:],
        "late_nu_probs": all_nu_probs[n_early:],
        "early_residuals": residuals_tensor[:n_early],
        "late_residuals": residuals_tensor[n_early:],
    }


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------

def _worker(rank, gpu_id, items, cfg):
    """Worker: load model, probe traces, save shard."""
    worker_logger = setup_logging(f"act_diff.worker{rank}")
    worker_logger.info(
        "Worker %d starting on GPU %d with %d traces",
        rank, gpu_id, len(items),
    )

    if not items:
        worker_logger.info("Worker %d has no items, exiting", rank)
        return

    ad = cfg.activation_diff
    n_early = ad.n_early
    n_late = ad.n_late

    model, _tokenizer = load_model(gpu_id=gpu_id)
    device = f"cuda:{gpu_id}"
    worker_logger.info("Worker %d model loaded", rank)

    metadata_rows = []
    early_residuals_list = []
    late_residuals_list = []

    for trace_data in tqdm(items, desc=f"GPU {gpu_id}", position=rank):
        try:
            result = probe_trace(model, trace_data, n_early, n_late, device)
        except Exception:
            worker_logger.exception(
                "Error probing %s", trace_data["metadata"]["problem_id"],
            )
            continue

        if result is None:
            continue

        row = {
            "problem_id": result["problem_id"],
            "dataset": result["dataset"],
            "expected_token": result["expected_token"],
            "think_length": result["think_length"],
        }
        for j in range(n_early):
            row[f"early_correct_{j}"] = result["early_correct"][j]
            row[f"early_mu_prob_{j}"] = result["early_mu_probs"][j]
            row[f"early_nu_prob_{j}"] = result["early_nu_probs"][j]
        for j in range(n_late):
            row[f"late_correct_{j}"] = result["late_correct"][j]
            row[f"late_mu_prob_{j}"] = result["late_mu_probs"][j]
            row[f"late_nu_prob_{j}"] = result["late_nu_probs"][j]

        metadata_rows.append(row)
        early_residuals_list.append(result["early_residuals"])
        late_residuals_list.append(result["late_residuals"])

        torch.cuda.empty_cache()

    if not metadata_rows:
        worker_logger.warning("Worker %d: no results", rank)
        return

    output_dir = cfg.paths.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save metadata parquet
    df = pd.DataFrame(metadata_rows)
    df.to_parquet(
        os.path.join(output_dir, f"shard_{rank}_meta.parquet"), index=False,
    )

    # Save activations as safetensors
    early_tensor = torch.stack(early_residuals_list)
    late_tensor = torch.stack(late_residuals_list)
    save_file(
        {
            "early": early_tensor.to(torch.bfloat16),
            "late": late_tensor.to(torch.bfloat16),
        },
        os.path.join(output_dir, f"shard_{rank}_activations.safetensors"),
    )

    worker_logger.info(
        "Worker %d: saved %d traces. Early: %s, Late: %s",
        rank, len(metadata_rows),
        list(early_tensor.shape), list(late_tensor.shape),
    )


# ---------------------------------------------------------------------------
# Main: merge, filter, compute direction
# ---------------------------------------------------------------------------

def main():
    """Run the activation diffing experiment."""
    logger_main = setup_logging("activation_diff")
    cfg = load_config()
    set_seed(cfg.seed)

    ad = cfg.activation_diff
    gpu_ids = list(cfg.gpus)
    output_dir = cfg.paths.output_dir
    traces_dir = cfg.paths.traces_dir
    n_early = ad.n_early
    n_late = ad.n_late
    early_threshold = ad.early_threshold
    late_threshold = ad.late_threshold

    logger_main.info("Output directory: %s", output_dir)
    logger_main.info("GPUs: %s", gpu_ids)
    logger_main.info(
        "Probing first %d and last %d thinking positions", n_early, n_late,
    )

    # Load all valid traces
    logger_main.info("Loading traces from %s", traces_dir)
    traces = load_all_traces(
        traces_dir,
        trace_index=ad.trace_index,
        min_think_length=ad.min_think_length,
    )
    logger_main.info("Loaded %d traces", len(traces))

    ds_counts = {}
    for t in traces:
        ds_counts[t["dataset"]] = ds_counts.get(t["dataset"], 0) + 1
    for ds, count in sorted(ds_counts.items()):
        logger_main.info("  %s: %d traces", ds, count)

    if not traces:
        logger_main.error("No traces found. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1: probe all traces across GPUs
    # -----------------------------------------------------------------------
    start_time = time.time()
    run_parallel(_worker, traces, gpu_ids, cfg)
    elapsed = time.time() - start_time
    logger_main.info(
        "Probing completed in %.1fs across %d GPU(s)", elapsed, len(gpu_ids),
    )

    # -----------------------------------------------------------------------
    # Phase 2: merge shards
    # -----------------------------------------------------------------------
    meta_dfs = []
    early_tensors = []
    late_tensors = []

    for rank in range(len(gpu_ids)):
        meta_path = os.path.join(output_dir, f"shard_{rank}_meta.parquet")
        act_path = os.path.join(
            output_dir, f"shard_{rank}_activations.safetensors",
        )

        if not os.path.isfile(meta_path):
            continue

        meta_dfs.append(pd.read_parquet(meta_path))
        acts = load_file(act_path)
        early_tensors.append(acts["early"].float())
        late_tensors.append(acts["late"].float())

        os.remove(meta_path)
        os.remove(act_path)

    if not meta_dfs:
        logger_main.error("No result shards found")
        return

    df = pd.concat(meta_dfs, ignore_index=True)
    all_early = torch.cat(early_tensors, dim=0)
    all_late = torch.cat(late_tensors, dim=0)

    logger_main.info("Total traces probed: %d", len(df))
    logger_main.info("Early activations shape: %s", list(all_early.shape))
    logger_main.info("Late activations shape: %s", list(all_late.shape))

    # Save full metadata before filtering
    df.to_parquet(
        os.path.join(output_dir, "all_traces_meta.parquet"), index=False,
    )

    # -----------------------------------------------------------------------
    # Phase 3: filter traces (early wrong, late right)
    # -----------------------------------------------------------------------
    n_early_correct = pd.Series(0, index=df.index)
    for j in range(n_early):
        n_early_correct += df[f"early_correct_{j}"].astype(int)

    n_late_correct = pd.Series(0, index=df.index)
    for j in range(n_late):
        n_late_correct += df[f"late_correct_{j}"].astype(int)

    mask = (n_early_correct <= early_threshold) & (
        n_late_correct >= late_threshold
    )
    n_filtered = mask.sum()

    logger_main.info("--- Filtering ---")
    logger_main.info(
        "Early: <= %d correct out of %d", early_threshold, n_early,
    )
    logger_main.info(
        "Late: >= %d correct out of %d", late_threshold, n_late,
    )
    logger_main.info(
        "Traces passing filter: %d / %d (%.1f%%)",
        n_filtered, len(df),
        100.0 * n_filtered / len(df) if len(df) > 0 else 0,
    )

    # Also log how many traces are correct at each position
    for j in range(n_early):
        pct = 100.0 * df[f"early_correct_{j}"].mean()
        logger_main.info("  early pos %d: %.1f%% correct", j + 1, pct)
    for j in range(n_late):
        pct = 100.0 * df[f"late_correct_{j}"].mean()
        logger_main.info(
            "  late pos %d: %.1f%% correct",
            df["think_length"].iloc[0] - n_late + j + 1, pct,
        )

    if n_filtered == 0:
        logger_main.error(
            "No traces pass the filter. Cannot compute direction. "
            "Try relaxing early_threshold or late_threshold.",
        )
        return

    # -----------------------------------------------------------------------
    # Phase 3.5: balance Mu/Nu expected tokens in filtered set
    # -----------------------------------------------------------------------
    filtered_df = df[mask].reset_index(drop=True)
    filtered_indices = df.index[mask].tolist()

    mu_count = (filtered_df["expected_token"] == "Mu").sum()
    nu_count = (filtered_df["expected_token"] == "Nu").sum()
    logger_main.info("--- Mu/Nu Balance (pre-subsample) ---")
    logger_main.info("  Mu-expected: %d, Nu-expected: %d", mu_count, nu_count)

    min_count = min(mu_count, nu_count)
    if mu_count != nu_count:
        import numpy as np
        rng = np.random.RandomState(cfg.seed)

        mu_idx = filtered_df.index[filtered_df["expected_token"] == "Mu"].tolist()
        nu_idx = filtered_df.index[filtered_df["expected_token"] == "Nu"].tolist()

        mu_keep = sorted(rng.choice(mu_idx, min_count, replace=False).tolist())
        nu_keep = sorted(rng.choice(nu_idx, min_count, replace=False).tolist())
        balanced_idx = sorted(mu_keep + nu_keep)

        # Map back to original df indices for activation tensors
        filtered_indices = [filtered_indices[i] for i in balanced_idx]
        filtered_df = filtered_df.iloc[balanced_idx].reset_index(drop=True)

        logger_main.info(
            "  Subsampled to %d per label (%d total)",
            min_count, len(filtered_indices),
        )
    else:
        logger_main.info("  Already balanced, no subsampling needed")

    # -----------------------------------------------------------------------
    # Phase 4: compute knowing direction
    # -----------------------------------------------------------------------
    mask_bool = torch.zeros(len(df), dtype=torch.bool)
    for idx in filtered_indices:
        mask_bool[idx] = True
    filtered_early = all_early[mask_bool]
    filtered_late = all_late[mask_bool]

    # Average over traces and probe positions -> [NUM_LAYERS, HIDDEN_DIM]
    mean_early = filtered_early.mean(dim=(0, 1))
    mean_late = filtered_late.mean(dim=(0, 1))
    direction = mean_late - mean_early

    # Normalize
    norms = direction.norm(dim=-1, keepdim=True)
    direction_normalized = direction / norms.clamp(min=1e-8)

    logger_main.info("--- Direction Norms (unnormalized) ---")
    for layer_idx in range(NUM_LAYERS):
        logger_main.info(
            "  Layer %2d: norm = %.4f", layer_idx, norms[layer_idx, 0].item(),
        )

    # Validate: project filtered early/late activations onto direction
    logger_main.info("--- Separation Validation ---")
    for layer_idx in [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]:
        d = direction_normalized[layer_idx]
        early_proj = (filtered_early[:, :, layer_idx, :] @ d).mean().item()
        late_proj = (filtered_late[:, :, layer_idx, :] @ d).mean().item()
        early_std = (filtered_early[:, :, layer_idx, :] @ d).std().item()
        late_std = (filtered_late[:, :, layer_idx, :] @ d).std().item()
        gap = late_proj - early_proj

        logger_main.info(
            "  Layer %2d: early=%.3f (std=%.3f), late=%.3f (std=%.3f), gap=%.3f",
            layer_idx, early_proj, early_std, late_proj, late_std, gap,
        )

    # Per-trace cosine similarity of individual diff vs mean direction
    logger_main.info("--- Per-Trace Direction Consistency ---")
    for layer_idx in [0, 8, 16, 24, 32, 35]:
        d = direction_normalized[layer_idx]
        per_trace_early = filtered_early[:, :, layer_idx, :].mean(dim=1)
        per_trace_late = filtered_late[:, :, layer_idx, :].mean(dim=1)
        per_trace_diff = per_trace_late - per_trace_early
        per_trace_diff_norm = per_trace_diff / per_trace_diff.norm(
            dim=-1, keepdim=True,
        ).clamp(min=1e-8)
        cos_sims = (per_trace_diff_norm @ d).tolist()
        mean_cos = sum(cos_sims) / len(cos_sims)
        logger_main.info(
            "  Layer %2d: mean cosine similarity = %.4f (n=%d)",
            layer_idx, mean_cos, len(cos_sims),
        )

    # -----------------------------------------------------------------------
    # Phase 5: save results
    # -----------------------------------------------------------------------
    save_file(
        {
            "direction": direction.to(torch.float32),
            "direction_normalized": direction_normalized.to(torch.float32),
            "mean_early": mean_early.to(torch.float32),
            "mean_late": mean_late.to(torch.float32),
        },
        os.path.join(output_dir, "knowing_direction.safetensors"),
    )

    save_file(
        {
            "early": filtered_early.to(torch.bfloat16),
            "late": filtered_late.to(torch.bfloat16),
        },
        os.path.join(output_dir, "filtered_activations.safetensors"),
    )

    # Per-dataset breakdown of filtered traces
    logger_main.info("--- Per-Dataset Breakdown (filtered + balanced) ---")
    for ds in sorted(filtered_df["dataset"].unique()):
        count = (filtered_df["dataset"] == ds).sum()
        logger_main.info("  %s: %d traces", ds, count)

    logger_main.info(
        "Saved direction to %s",
        os.path.join(output_dir, "knowing_direction.safetensors"),
    )
    logger_main.info(
        "Saved metadata to %s",
        os.path.join(output_dir, "all_traces_meta.parquet"),
    )
    logger_main.info("Done.")


if __name__ == "__main__":
    main()
