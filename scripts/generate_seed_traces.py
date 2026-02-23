"""Cross-seed trace generation for transition consistency analysis.

Generates 17 new traces (indices 3-19) for each of the ~41 originally-correct
problems from the Phase 2.2 selection. Uses the same prompts (extracted from
existing trace_00.json metadata) and generation parameters (temperature=0.6,
top_p=0.95, top_k=20) as the original traces.

Each trace gets a deterministic seed via derive_trace_seed(problem_id, trace_index).
File-based resume: existing trace files (valid or failed) are skipped.

Usage:
    python scripts/generate_seed_traces.py --config configs/cross_seed.yaml
"""

import gc
import json
import logging
import os
import random
import time

from dotenv import load_dotenv

load_dotenv()

import torch

from src.config import load_config
from src.model import generate_trace, load_model
from src.multiprocess import run_parallel_isolated
from src.storage import (
    derive_trace_seed,
    get_failed_trace_json_path,
    get_trace_json_path,
    write_trace_json,
)
from src.tokens import parse_think_boundaries
from src.utils import set_seed, setup_logging

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
# Problem info extraction
# ---------------------------------------------------------------------------


def extract_problem_info(trace_data):
    """Extract generation-ready problem info from a selected trace entry.

    Reads the verification_prompt and all metadata fields from the trace's
    metadata, returning a dict suitable for the generation worker.

    Args:
        trace_data: Dict from select_problems with keys: trace_path,
            token_ids, metadata, is_correct, dataset.

    Returns:
        Dict with keys: problem_id, dataset_name, verification_prompt,
        ground_truth, valid_token, invalid_token, prompt_config,
        label_mapping, is_fresh, question, original_answer,
        presented_answer.
    """
    meta = trace_data["metadata"]
    return {
        "problem_id": meta["problem_id"],
        "dataset_name": meta["dataset_name"],
        "verification_prompt": meta["verification_prompt"],
        "ground_truth": meta["ground_truth"],
        "valid_token": meta["valid_token"],
        "invalid_token": meta["invalid_token"],
        "prompt_config": meta.get("prompt_config"),
        "label_mapping": meta.get("label_mapping"),
        "is_fresh": meta.get("is_fresh"),
        "question": meta.get("question"),
        "original_answer": meta.get("original_answer"),
        "presented_answer": meta.get("presented_answer"),
    }


# ---------------------------------------------------------------------------
# Generation worker
# ---------------------------------------------------------------------------


def seed_generation_worker(rank, gpu_id, items, cfg):
    """Worker function: load model, generate new seed traces, write JSON files.

    For each problem in items, generates traces at indices
    [new_trace_start, new_trace_end) using deterministic seeds from
    derive_trace_seed(). Existing files are skipped (resume support).

    Args:
        rank: Worker rank (0-indexed).
        gpu_id: CUDA device index for this worker.
        items: List of problem_info dicts assigned to this worker.
        cfg: OmegaConf DictConfig with model, cross_seed, and paths settings.
    """
    worker_logger = setup_logging(f"seed_gen.worker{rank}")
    torch.cuda.set_device(gpu_id)

    worker_logger.info(
        "Worker %d starting on GPU %d with %d problems", rank, gpu_id, len(items)
    )

    # Load model on assigned GPU
    model, tokenizer = load_model(gpu_id=gpu_id)
    worker_logger.info("Worker %d: model loaded on GPU %d", rank, gpu_id)

    traces_dir = cfg.paths.traces_dir
    max_new_tokens = cfg.model.max_new_tokens
    new_trace_start = cfg.cross_seed.new_trace_start
    new_trace_end = cfg.cross_seed.new_trace_end

    generated_count = 0
    skipped_count = 0
    valid_count = 0
    routed_failed_count = 0
    error_count = 0

    for prob_idx, problem_info in enumerate(items):
        problem_id = problem_info["problem_id"]
        dataset_name = problem_info["dataset_name"]
        prompt = problem_info["verification_prompt"]

        worker_logger.info(
            "Worker %d: problem %d/%d -- %s",
            rank,
            prob_idx + 1,
            len(items),
            problem_id,
        )

        for trace_index in range(new_trace_start, new_trace_end):
            # Compute output paths for both valid and failed destinations
            valid_path = get_trace_json_path(
                traces_dir, dataset_name, problem_id, trace_index
            )
            failed_path = get_failed_trace_json_path(
                traces_dir, dataset_name, problem_id, trace_index
            )

            # Resume check: skip if either valid or failed file exists
            if os.path.exists(valid_path) or os.path.exists(failed_path):
                skipped_count += 1
                continue

            # Derive deterministic seed for this specific trace
            seed = derive_trace_seed(problem_id, trace_index)
            set_seed(seed)

            try:
                # Generate thinking-mode trace
                output_ids = generate_trace(
                    model, tokenizer, prompt, max_new_tokens=max_new_tokens
                )

                # Convert to list of ints for JSON serialization
                token_list = output_ids[0].tolist()

                # Decode generated text
                generated_text = tokenizer.decode(
                    token_list, skip_special_tokens=False
                )

                # Parse think boundaries to determine quality
                blocks = parse_think_boundaries(token_list)
                has_think = len(blocks) > 0

                # Build trace data dict with full metadata (same schema as
                # generate_traces.py)
                trace_data = {
                    "token_ids": token_list,
                    "generated_text": generated_text,
                    "metadata": {
                        "problem_id": problem_id,
                        "dataset_name": dataset_name,
                        "ground_truth": problem_info["ground_truth"],
                        "question": problem_info.get("question"),
                        "original_answer": problem_info.get("original_answer"),
                        "verification_prompt": prompt,
                        "presented_answer": problem_info.get("presented_answer"),
                        "trace_index": trace_index,
                        "seed": seed,
                        "model_name": cfg.model.name,
                        "max_new_tokens": max_new_tokens,
                        "temperature": cfg.generation.temperature,
                        "top_p": cfg.generation.top_p,
                        "top_k": cfg.generation.top_k,
                        "num_tokens": len(token_list),
                        "num_think_blocks": len(blocks),
                        "has_think": has_think,
                        "think_boundaries": blocks,
                        "timestamp": time.strftime(
                            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                        ),
                    },
                }

                # Include optional metadata fields from problem_info
                if problem_info.get("is_fresh") is not None:
                    trace_data["metadata"]["is_fresh"] = problem_info["is_fresh"]
                if problem_info.get("prompt_config") is not None:
                    trace_data["metadata"]["prompt_config"] = problem_info[
                        "prompt_config"
                    ]
                if problem_info.get("label_mapping") is not None:
                    trace_data["metadata"]["label_mapping"] = problem_info[
                        "label_mapping"
                    ]
                    trace_data["metadata"]["valid_token"] = problem_info["valid_token"]
                    trace_data["metadata"]["invalid_token"] = problem_info[
                        "invalid_token"
                    ]

                # Route to valid/ or failed/ based on think block presence
                if has_think:
                    dest_path = valid_path
                    valid_count += 1
                else:
                    dest_path = failed_path
                    routed_failed_count += 1
                    worker_logger.warning(
                        "Worker %d: no think block in %s trace %d, routing to failed/",
                        rank,
                        problem_id,
                        trace_index,
                    )

                # Atomic write to destination
                write_trace_json(trace_data, dest_path)
                generated_count += 1

            except Exception:
                error_count += 1
                worker_logger.exception(
                    "Worker %d: error generating %s trace %d",
                    rank,
                    problem_id,
                    trace_index,
                )
                continue

        # Free GPU cache between problems
        torch.cuda.empty_cache()

    # Cleanup: free model memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    worker_logger.info(
        "Worker %d done. Generated: %d, Skipped (resume): %d, "
        "Valid: %d, Routed to failed: %d, Errors: %d",
        rank,
        generated_count,
        skipped_count,
        valid_count,
        routed_failed_count,
        error_count,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Load config, select correct problems, dispatch workers for seed trace generation."""
    main_logger = setup_logging("generate_seed_traces")
    main_logger.info("Starting cross-seed trace generation")

    # Load configuration (base.yaml + cross_seed.yaml + CLI overrides)
    cfg = load_config()
    set_seed(cfg.seed)

    traces_dir = cfg.paths.traces_dir
    main_logger.info("Traces directory: %s", traces_dir)

    # Select problems using Phase 2.2 selection logic
    all_selected, correct_only = select_problems(traces_dir, cfg)
    main_logger.info(
        "Problem selection: %d total, %d correct (will generate new traces for correct only)",
        len(all_selected),
        len(correct_only),
    )

    # Extract generation-ready problem info
    problem_list = [extract_problem_info(t) for t in correct_only]

    # Log per-dataset counts
    ds_counts = {}
    for p in problem_list:
        ds = p["dataset_name"]
        ds_counts[ds] = ds_counts.get(ds, 0) + 1
    for ds_name, count in sorted(ds_counts.items()):
        main_logger.info("  %s: %d problems", ds_name, count)

    n_new_traces = cfg.cross_seed.new_trace_end - cfg.cross_seed.new_trace_start
    total_expected = len(problem_list) * n_new_traces
    main_logger.info(
        "Expected new traces: %d (%d problems x %d trace indices [%d-%d))",
        total_expected,
        len(problem_list),
        n_new_traces,
        cfg.cross_seed.new_trace_start,
        cfg.cross_seed.new_trace_end,
    )

    # Dispatch workers across GPUs
    gpu_ids = list(cfg.gpus)
    main_logger.info("Using GPUs: %s", gpu_ids)

    start_time = time.time()
    failures = run_parallel_isolated(seed_generation_worker, problem_list, gpu_ids, cfg)
    elapsed = time.time() - start_time

    main_logger.info(
        "All workers finished in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60
    )

    # Report results
    num_workers = len(gpu_ids)
    num_failed_workers = len(failures)
    num_succeeded = num_workers - num_failed_workers

    main_logger.info(
        "Workers: %d/%d succeeded, %d/%d failed",
        num_succeeded,
        num_workers,
        num_failed_workers,
        num_workers,
    )

    if failures:
        for rank_id, gid, exitcode in failures:
            main_logger.error(
                "Worker rank %d on GPU %d exited with code %d",
                rank_id,
                gid,
                exitcode,
            )

    # Count how many new trace files were actually created
    new_trace_count = 0
    for p in problem_list:
        pid = p["problem_id"]
        ds = p["dataset_name"]
        for ti in range(cfg.cross_seed.new_trace_start, cfg.cross_seed.new_trace_end):
            valid_path = get_trace_json_path(traces_dir, ds, pid, ti)
            failed_path = get_failed_trace_json_path(traces_dir, ds, pid, ti)
            if os.path.exists(valid_path) or os.path.exists(failed_path):
                new_trace_count += 1

    main_logger.info(
        "New trace files on disk: %d / %d expected", new_trace_count, total_expected
    )
    main_logger.info("Output directory: %s", traces_dir)
    main_logger.info("Cross-seed trace generation complete.")


if __name__ == "__main__":
    main()
