"""Cross-seed force-answer probing experiment.

Loads all 20 traces (indices 0-19) per problem for the ~41 originally-correct
problems from the Phase 2.2 selection, runs force-answer probing at ~20
positions per trace using KV-cache-efficient truncation, and outputs a
seed-aware results.parquet with seed_index and trace_index columns.

This extends Phase 2.2's single-trace probing to 20-trace-per-problem probing,
producing the data needed for cross-seed transition analysis and visualization.

Usage:
    python scripts/cross_seed_experiment.py --config configs/cross_seed.yaml
"""

import copy
import json
import logging
import os
import random
import sys
import time

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import DynamicCache

from src.config import load_config
from src.model import load_model
from src.multiprocess import run_parallel_isolated
from src.storage import get_trace_json_path
from src.utils import set_seed, setup_logging

# Import probing functions from Phase 2.2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.force_answer_experiment import (
    CLOSE_TOKEN_IDS,
    RHO_FIRST_TOKEN_ID,
    RHO_TOKEN_IDS,
    TAU_TOKEN_ID,
    compute_probe_positions,
    extract_answer,
    extract_rho_tau_probs,
    probe_trace,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Problem selection (reproduces Phase 2.2 logic inline)
# ---------------------------------------------------------------------------


def select_problems(traces_dir, cfg):
    """Select problems using the same logic as Phase 2.2's load_traces_for_experiment.

    Walks valid/{dataset}/{problem_id}/ directories, loads trace_00.json from
    each, filters for single-think-block traces, determines correctness, and
    selects a balanced set using proportional allocation with seed 137.

    Args:
        traces_dir: Root traces directory (contains valid/ subdirectory).
        cfg: OmegaConf config with cross_seed section.

    Returns:
        Tuple of (all_selected, correct_only):
        - all_selected: List of dicts (~51 problems) with keys: trace_path,
          token_ids, metadata, is_correct, dataset.
        - correct_only: Sublist filtered to is_correct=True (~41 problems).
    """
    cs = cfg.cross_seed
    trace_index = cs.trace_index
    trace_filename = f"trace_{trace_index:02d}.json"

    valid_dir = os.path.join(traces_dir, "valid")
    datasets = ["gsm_symbolic", "folio", "strategyqa"]

    # Collect all eligible traces per dataset, split by correctness
    correct_by_ds = {ds: [] for ds in datasets}
    incorrect_by_ds = {ds: [] for ds in datasets}
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

            trace_path = os.path.join(problem_dir, trace_filename)
            if not os.path.isfile(trace_path):
                continue

            with open(trace_path) as f:
                trace = json.load(f)

            meta = trace["metadata"]

            # Filter: must have think block and exactly one
            if not meta.get("has_think", False):
                skipped += 1
                continue
            if meta.get("num_think_blocks", 0) != 1:
                skipped += 1
                continue

            # Determine correctness
            ground_truth = meta["ground_truth"]
            valid_token = meta["valid_token"]
            invalid_token = meta["invalid_token"]

            if ground_truth == "Yes":
                expected_token = valid_token
            else:
                expected_token = invalid_token

            # Check if expected token appears in generated text after last </think>
            generated_text = trace.get("generated_text", "")
            think_end_idx = generated_text.rfind("</think>")
            if think_end_idx >= 0:
                answer_text = generated_text[think_end_idx:]
            else:
                answer_text = generated_text

            is_correct = expected_token.lower() in answer_text.lower()

            entry = {
                "trace_path": trace_path,
                "token_ids": trace["token_ids"],
                "metadata": meta,
                "is_correct": is_correct,
                "dataset": ds,
            }

            if is_correct:
                correct_by_ds[ds].append(entry)
            else:
                incorrect_by_ds[ds].append(entry)

    # Log available counts
    for ds in datasets:
        logger.info(
            "Dataset %s: %d correct, %d incorrect available",
            ds,
            len(correct_by_ds[ds]),
            len(incorrect_by_ds[ds]),
        )
    if skipped > 0:
        logger.info("Skipped %d traces (no think or multiple think blocks)", skipped)

    # Select traces proportionally across datasets (same as Phase 2.2)
    rng = random.Random(cs.selection_seed)
    n_correct_target = cs.n_correct
    n_incorrect_target = cs.n_incorrect

    total_correct = sum(len(v) for v in correct_by_ds.values())
    total_incorrect = sum(len(v) for v in incorrect_by_ds.values())

    selected = []

    for ds in datasets:
        # Correct allocation
        if total_correct > 0:
            ds_n_correct = round(
                n_correct_target * len(correct_by_ds[ds]) / total_correct
            )
        else:
            ds_n_correct = 0
        ds_n_correct = min(ds_n_correct, len(correct_by_ds[ds]))

        # Incorrect allocation
        if total_incorrect > 0:
            ds_n_incorrect = round(
                n_incorrect_target * len(incorrect_by_ds[ds]) / total_incorrect
            )
        else:
            ds_n_incorrect = 0
        ds_n_incorrect = min(ds_n_incorrect, len(incorrect_by_ds[ds]))

        # Sample
        correct_sample = (
            rng.sample(correct_by_ds[ds], ds_n_correct) if ds_n_correct > 0 else []
        )
        incorrect_sample = (
            rng.sample(incorrect_by_ds[ds], ds_n_incorrect)
            if ds_n_incorrect > 0
            else []
        )

        selected.extend(correct_sample)
        selected.extend(incorrect_sample)

        logger.info(
            "Selected from %s: %d correct, %d incorrect",
            ds,
            len(correct_sample),
            len(incorrect_sample),
        )

    n_correct_selected = sum(1 for t in selected if t["is_correct"])
    n_incorrect_selected = len(selected) - n_correct_selected
    logger.info(
        "Total selected: %d traces (%d correct, %d incorrect)",
        len(selected),
        n_correct_selected,
        n_incorrect_selected,
    )

    correct_only = [t for t in selected if t["is_correct"]]
    return selected, correct_only


# ---------------------------------------------------------------------------
# Trace loading (all 20 seeds per problem)
# ---------------------------------------------------------------------------


def load_all_seed_traces(traces_dir, correct_problems, cfg):
    """Load all 20 traces (indices 0-19) for each correct problem.

    For each problem, reads trace JSON files at indices 0 through
    total_seeds-1. Traces that pass the single-think-block filter are
    included; others are skipped with a warning.

    Args:
        traces_dir: Root traces directory (contains valid/ subdirectory).
        correct_problems: List of trace entry dicts from select_problems,
            each with keys: trace_path, token_ids, metadata, is_correct, dataset.
        cfg: OmegaConf config with cross_seed section.

    Returns:
        Flat list of trace_data dicts, each with keys: trace_path, token_ids,
        metadata, is_correct, dataset, seed_index, trace_index.
    """
    total_seeds = cfg.cross_seed.total_seeds
    all_traces = []
    total_skipped = 0

    for problem_entry in correct_problems:
        meta = problem_entry["metadata"]
        problem_id = meta["problem_id"]
        dataset = problem_entry["dataset"]
        valid_count = 0

        for trace_index in range(total_seeds):
            trace_path = get_trace_json_path(
                traces_dir, dataset, problem_id, trace_index
            )

            if not os.path.isfile(trace_path):
                logger.warning(
                    "Trace file not found: %s (problem %s, seed %d)",
                    trace_path,
                    problem_id,
                    trace_index,
                )
                total_skipped += 1
                continue

            with open(trace_path) as f:
                trace = json.load(f)

            trace_meta = trace["metadata"]

            # Filter: must have think block and exactly one
            if not trace_meta.get("has_think", False):
                logger.debug(
                    "Skipping %s seed %d: no think block", problem_id, trace_index
                )
                total_skipped += 1
                continue
            if trace_meta.get("num_think_blocks", 0) != 1:
                logger.debug(
                    "Skipping %s seed %d: %d think blocks",
                    problem_id,
                    trace_index,
                    trace_meta.get("num_think_blocks", 0),
                )
                total_skipped += 1
                continue

            trace_data = {
                "trace_path": trace_path,
                "token_ids": trace["token_ids"],
                "metadata": trace_meta,
                "is_correct": True,  # all are originally correct
                "dataset": dataset,
                "seed_index": trace_index,
                "trace_index": trace_index,
            }

            all_traces.append(trace_data)
            valid_count += 1

        logger.info(
            "Problem %s: %d/%d valid traces", problem_id, valid_count, total_seeds
        )

    logger.info(
        "Total traces loaded: %d (skipped %d)", len(all_traces), total_skipped
    )
    return all_traces


# ---------------------------------------------------------------------------
# Probing wrapper (adds seed_index and trace_index to results)
# ---------------------------------------------------------------------------


def probe_trace_with_seed(raw_model, tokenizer, trace_data, cfg_shim, device):
    """Probe a trace and add seed_index and trace_index to each result row.

    Wraps probe_trace() from force_answer_experiment.py, passing cfg_shim
    (which has a force_answer key) instead of the full cross_seed config.

    Args:
        raw_model: Raw HuggingFace model (model._model).
        tokenizer: Model tokenizer.
        trace_data: Dict with token_ids, metadata, is_correct, dataset,
            seed_index, and trace_index.
        cfg_shim: OmegaConf dict with force_answer key containing
            num_probes, min_probe_spacing, max_new_tokens.
        device: Torch device for model inputs.

    Returns:
        List of result row dicts with seed_index and trace_index added.
    """
    results = probe_trace(raw_model, tokenizer, trace_data, cfg_shim, device)

    for row in results:
        row["seed_index"] = trace_data["seed_index"]
        row["trace_index"] = trace_data["trace_index"]
        # trace_id should be the problem_id (same across seeds)
        row["trace_id"] = trace_data["metadata"]["problem_id"]

    return results


# ---------------------------------------------------------------------------
# Worker function for multi-GPU execution
# ---------------------------------------------------------------------------


def _worker(rank, gpu_id, items, cfg):
    """Worker function: load model, probe assigned traces, write shard.

    Args:
        rank: Worker rank (0-indexed).
        gpu_id: CUDA device index.
        items: List of trace_data dicts assigned to this worker.
        cfg: OmegaConf config (full cross_seed config).
    """
    worker_logger = setup_logging(f"cross_seed.worker{rank}")
    worker_logger.info(
        "Worker %d starting on GPU %d with %d traces", rank, gpu_id, len(items)
    )

    if not items:
        worker_logger.info("Worker %d has no items, exiting", rank)
        return

    model, tokenizer = load_model(gpu_id=gpu_id)
    raw_model = model._model
    device = next(raw_model.parameters()).device
    worker_logger.info("Worker %d model loaded on %s", rank, device)

    # Build cfg_shim with force_answer key that probe_trace() expects
    cfg_shim = OmegaConf.create(
        {
            "force_answer": {
                "num_probes": cfg.cross_seed.num_probes,
                "min_probe_spacing": cfg.cross_seed.min_probe_spacing,
                "max_new_tokens": cfg.cross_seed.max_new_tokens,
            }
        }
    )

    all_results = []
    for trace_data in tqdm(items, desc=f"GPU {gpu_id}", position=rank):
        rows = probe_trace_with_seed(
            raw_model, tokenizer, trace_data, cfg_shim, device
        )
        all_results.extend(rows)
        torch.cuda.empty_cache()

    # Write shard to temporary parquet
    if all_results:
        output_dir = cfg.paths.output_dir
        os.makedirs(output_dir, exist_ok=True)
        shard_path = os.path.join(output_dir, f"results_shard_{rank}.parquet")
        df = pd.DataFrame(all_results)
        df.to_parquet(shard_path, index=False)
        worker_logger.info(
            "Worker %d wrote %d rows to %s", rank, len(df), shard_path
        )


# ---------------------------------------------------------------------------
# Required columns (Phase 2.2 columns + seed_index, trace_index)
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "trace_id",
    "dataset",
    "problem_id",
    "probe_position",
    "normalized_position",
    "correct",
    "no_answer",
    "rho_prob",
    "tau_prob",
    "correct_token_prob",
    "correct_token_rank",
    "ground_truth",
    "is_correct_trace",
    "is_fresh",
    "prompt_config",
    "valid_token",
    "invalid_token",
    "token_answer",
    "string_answer",
    "agreement",
    "think_length",
    # NEW for Phase 2.3
    "seed_index",
    "trace_index",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run the cross-seed force-answer probing experiment."""
    logger_main = setup_logging("cross_seed")
    cfg = load_config()
    set_seed(cfg.seed)

    gpu_ids = list(cfg.gpus)
    output_dir = cfg.paths.output_dir
    traces_dir = cfg.paths.traces_dir

    logger_main.info("Traces directory: %s", traces_dir)
    logger_main.info("Output directory: %s", output_dir)
    logger_main.info("GPUs: %s", gpu_ids)

    # Select problems using Phase 2.2 selection logic
    all_selected, correct_only = select_problems(traces_dir, cfg)
    logger_main.info(
        "Problem selection: %d total, %d correct (probing correct only)",
        len(all_selected),
        len(correct_only),
    )

    # Load all 20 seed traces for each correct problem
    all_trace_datas = load_all_seed_traces(traces_dir, correct_only, cfg)
    logger_main.info("Total traces to probe: %d", len(all_trace_datas))

    # Per-problem seed counts
    problem_seed_counts = {}
    for td in all_trace_datas:
        pid = td["metadata"]["problem_id"]
        problem_seed_counts[pid] = problem_seed_counts.get(pid, 0) + 1

    min_valid = cfg.cross_seed.min_valid_seeds
    low_seed_problems = [
        pid for pid, count in problem_seed_counts.items() if count < min_valid
    ]
    if low_seed_problems:
        logger_main.warning(
            "%d problems have fewer than %d valid seeds: %s",
            len(low_seed_problems),
            min_valid,
            low_seed_problems,
        )

    seed_counts = list(problem_seed_counts.values())
    logger_main.info(
        "Seed counts: mean=%.1f, min=%d, max=%d",
        np.mean(seed_counts),
        np.min(seed_counts),
        np.max(seed_counts),
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run probing across GPUs
    start_time = time.time()
    failures = run_parallel_isolated(_worker, all_trace_datas, gpu_ids, cfg)
    elapsed = time.time() - start_time
    logger_main.info(
        "Probing completed in %.1fs (%.1f min) across %d GPU(s)",
        elapsed,
        elapsed / 60,
        len(gpu_ids),
    )

    if failures:
        logger_main.warning(
            "%d/%d workers failed: %s", len(failures), len(gpu_ids), failures
        )

    # Merge shards into single results.parquet
    shard_dfs = []
    for rank in range(len(gpu_ids)):
        shard_path = os.path.join(output_dir, f"results_shard_{rank}.parquet")
        if os.path.isfile(shard_path):
            shard_dfs.append(pd.read_parquet(shard_path))
            os.remove(shard_path)

    if not shard_dfs:
        logger_main.error("No result shards found -- all workers may have failed")
        return

    df = pd.concat(shard_dfs, ignore_index=True)

    # Ensure all required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            logger_main.warning("Missing column %s, adding with None", col)
            df[col] = None

    # Sort by trace_id, seed_index, probe_position for deterministic output
    df = df.sort_values(
        ["trace_id", "seed_index", "probe_position"]
    ).reset_index(drop=True)

    # Save final parquet
    output_path = os.path.join(output_dir, "results.parquet")
    df.to_parquet(output_path, index=False)
    logger_main.info("Saved %d rows to %s", len(df), output_path)

    # Summary statistics
    logger_main.info("--- Summary ---")
    logger_main.info("Total probe measurements: %d", len(df))
    logger_main.info("Unique problems: %d", df["trace_id"].nunique())
    logger_main.info(
        "Unique seeds per problem: mean=%.1f, min=%d, max=%d",
        df.groupby("trace_id")["seed_index"].nunique().mean(),
        df.groupby("trace_id")["seed_index"].nunique().min(),
        df.groupby("trace_id")["seed_index"].nunique().max(),
    )
    logger_main.info("Datasets: %s", sorted(df["dataset"].unique().tolist()))
    logger_main.info(
        "Correct answers: %d / %d (%.1f%%)",
        df["correct"].sum(),
        len(df),
        100.0 * df["correct"].sum() / len(df) if len(df) > 0 else 0,
    )
    logger_main.info(
        "No-answer cases: %d / %d (%.1f%%)",
        df["no_answer"].sum(),
        len(df),
        100.0 * df["no_answer"].sum() / len(df) if len(df) > 0 else 0,
    )
    if not df.empty:
        logger_main.info(
            "Mean correct_token_rank: %.1f", df["correct_token_rank"].mean()
        )
        disagreements = (~df["agreement"]).sum()
        if disagreements > 0:
            logger_main.info(
                "Token/string answer disagreements: %d", disagreements
            )


if __name__ == "__main__":
    main()
