"""Multi-GPU trace generation for Qwen3-8B thinking-mode traces.

Stage 1 of the decoupled pipeline: generates raw traces (token IDs + text +
metadata) across GSM-Symbolic, FOLIO, and StrategyQA. Does NOT capture
activations -- that is handled separately by the activation capture script.

Each trace is saved as a JSON file with atomic writes. Traces with valid
<think> blocks go to data/traces/valid/, those without go to data/traces/failed/.
File-based resume: existing output files (valid or failed) are skipped.

Uses run_parallel_isolated for fault tolerance: worker crashes do not kill
other workers.

Usage:
    uv run python scripts/generate_traces.py \
        --config configs/generate.yaml --gpus 0,1,2,3
"""

import gc
import logging
import os
import time

import torch

# -- project imports ----------------------------------------------------------
from src.config import load_config
from src.datasets import (
    load_deepmath,
    load_folio,
    load_gsm_symbolic_contamination,
    load_strategyqa,
)
from src.model import generate_trace, load_model
from src.multiprocess import run_parallel_isolated
from src.prompts import build_verification_prompt
from src.storage import (
    derive_trace_seed,
    get_failed_trace_json_path,
    get_trace_json_path,
    write_trace_json,
)
from src.tokens import parse_think_boundaries
from src.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


def generation_worker(rank, gpu_id, problems, cfg):
    """Worker function: load model, generate traces, write JSON files.

    Each worker loads its own model copy on the assigned GPU and processes
    its chunk of problems sequentially. For each problem and trace index,
    it generates a thinking-mode trace, checks for think block presence,
    and routes the output to valid/ or failed/.

    Args:
        rank: Worker rank (0-indexed).
        gpu_id: CUDA device index for this worker.
        problems: List of problem dicts assigned to this worker.
        cfg: OmegaConf DictConfig with model, data, and paths settings.
    """
    worker_logger = setup_logging(f"gen_worker_{rank}")
    torch.cuda.set_device(gpu_id)

    worker_logger.info(
        "Worker %d starting on GPU %d with %d problems", rank, gpu_id, len(problems)
    )

    # Load model on assigned GPU.
    model, tokenizer = load_model(gpu_id=gpu_id, model_name=cfg.model.name)
    worker_logger.info("Worker %d: model loaded on GPU %d", rank, gpu_id)

    traces_dir = cfg.paths.traces_dir
    traces_per_problem = cfg.data.traces_per_problem
    max_new_tokens = cfg.model.max_new_tokens

    generated_count = 0
    skipped_count = 0
    failed_count = 0
    valid_count = 0
    routed_failed_count = 0

    for prob_idx, problem in enumerate(problems):
        problem_id = problem["problem_id"]
        dataset_name = problem["dataset_name"]
        prompt = problem["verification_prompt"]

        worker_logger.info(
            "Worker %d: problem %d/%d -- %s",
            rank,
            prob_idx + 1,
            len(problems),
            problem_id,
        )

        for trace_index in range(traces_per_problem):
            # Compute output paths for both valid and failed destinations.
            valid_path = get_trace_json_path(
                traces_dir, dataset_name, problem_id, trace_index
            )
            failed_path = get_failed_trace_json_path(
                traces_dir, dataset_name, problem_id, trace_index
            )

            # Resume check: skip if either valid or failed file exists.
            if os.path.exists(valid_path) or os.path.exists(failed_path):
                skipped_count += 1
                continue

            # Derive deterministic seed for this specific trace.
            seed = derive_trace_seed(problem_id, trace_index)
            set_seed(seed)

            try:
                # Generate thinking-mode trace.
                output_ids = generate_trace(
                    model, tokenizer, prompt, max_new_tokens=max_new_tokens
                )

                # Convert to list of ints for JSON serialization.
                token_list = output_ids[0].tolist()

                # Decode generated text.
                generated_text = tokenizer.decode(token_list, skip_special_tokens=False)

                # Parse think boundaries to determine quality.
                blocks = parse_think_boundaries(token_list)
                has_think = len(blocks) > 0

                # Build trace data dict with full metadata.
                trace_data = {
                    "token_ids": token_list,
                    "generated_text": generated_text,
                    "metadata": {
                        "problem_id": problem_id,
                        "dataset_name": dataset_name,
                        "ground_truth": problem["ground_truth"],
                        "question": problem["question"],
                        "original_answer": problem["original_answer"],
                        "verification_prompt": prompt,
                        "presented_answer": problem["presented_answer"],
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
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                }

                # Include is_fresh metadata if present (GSM-Symbolic contamination).
                if "is_fresh" in problem:
                    trace_data["metadata"]["is_fresh"] = problem["is_fresh"]

                # Include swapped-label protocol metadata if present.
                if "prompt_config" in problem:
                    trace_data["metadata"]["prompt_config"] = problem["prompt_config"]
                if "label_mapping" in problem:
                    trace_data["metadata"]["label_mapping"] = problem["label_mapping"]
                    # Store explicit valid/invalid token mapping for downstream clarity.
                    trace_data["metadata"]["valid_token"] = problem["label_mapping"]["valid"]
                    trace_data["metadata"]["invalid_token"] = problem["label_mapping"]["invalid"]

                # Route to valid/ or failed/ based on think block presence.
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

                # Atomic write to destination.
                write_trace_json(trace_data, dest_path)
                generated_count += 1

            except Exception:
                failed_count += 1
                worker_logger.exception(
                    "Worker %d: error generating %s trace %d",
                    rank,
                    problem_id,
                    trace_index,
                )
                continue

    # Cleanup: free model memory.
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
        failed_count,
    )


def main():
    """Load config, build problem list, dispatch workers for trace generation."""
    main_logger = setup_logging("generate_traces")
    main_logger.info("Starting trace generation")

    # Load configuration (base.yaml + generate.yaml + CLI overrides).
    cfg = load_config()

    main_logger.info("Configuration loaded:")
    main_logger.info("  Model: %s", cfg.model.name)
    main_logger.info("  Max new tokens: %d", cfg.model.max_new_tokens)
    main_logger.info("  Traces per problem: %d", cfg.data.traces_per_problem)
    main_logger.info("  GPUs: %s", list(cfg.gpus))
    main_logger.info("  Seed: %d", cfg.seed)
    main_logger.info("  Traces dir: %s", cfg.paths.traces_dir)

    # Build the full problem list from all datasets.
    main_logger.info("Loading datasets...")

    gsm_problems = load_gsm_symbolic_contamination(
        n_per_type=cfg.data.gsm_problems_per_type, seed=cfg.seed
    )
    main_logger.info(
        "GSM-Symbolic contamination: %d problems loaded", len(gsm_problems)
    )

    folio_problems = load_folio(n=cfg.data.folio_problems, seed=cfg.seed)
    main_logger.info("FOLIO: %d problems loaded", len(folio_problems))

    strategyqa_problems = load_strategyqa(
        n=cfg.data.strategyqa_problems, seed=cfg.seed
    )
    main_logger.info("StrategyQA: %d problems loaded", len(strategyqa_problems))

    deepmath_problems = load_deepmath(
        n=cfg.data.deepmath_problems, seed=cfg.seed
    )
    main_logger.info("DeepMath: %d problems loaded", len(deepmath_problems))

    all_problems = gsm_problems + folio_problems + strategyqa_problems + deepmath_problems
    main_logger.info("Total base problems: %d", len(all_problems))

    # Apply swapped-label protocol: expand each problem into config A and B variants.
    label_protocol = cfg.data.get("label_protocol", "swapped")
    main_logger.info("Label protocol: %s", label_protocol)

    expanded_problems = []
    for problem in all_problems:
        if label_protocol == "swapped":
            for config_name in ["A", "B"]:
                result = build_verification_prompt(problem, "swapped", config_name)
                variant = dict(problem)
                variant["verification_prompt"] = result["verification_prompt"]
                variant["prompt_config"] = result["prompt_config"]
                variant["label_mapping"] = result["label_mapping"]
                variant["problem_id"] = f"{problem['problem_id']}_cfg{config_name.lower()}"
                expanded_problems.append(variant)
        else:
            # yesno protocol: single variant per problem
            result = build_verification_prompt(problem, "yesno")
            variant = dict(problem)
            variant["verification_prompt"] = result["verification_prompt"]
            variant["prompt_config"] = result["prompt_config"]
            variant["label_mapping"] = result["label_mapping"]
            expanded_problems.append(variant)

    all_problems = expanded_problems
    main_logger.info(
        "After %s expansion: %d problems",
        label_protocol,
        len(all_problems),
    )

    # Count per-dataset for logging.
    dataset_counts = {}
    for p in all_problems:
        ds = p["dataset_name"]
        dataset_counts[ds] = dataset_counts.get(ds, 0) + 1
    for ds_name, count in sorted(dataset_counts.items()):
        main_logger.info("  %s: %d problems", ds_name, count)

    total_traces = len(all_problems) * cfg.data.traces_per_problem
    main_logger.info(
        "Expected total traces: %d (%d problems x %d traces each)",
        total_traces,
        len(all_problems),
        cfg.data.traces_per_problem,
    )

    # Resolve GPU IDs.
    gpu_ids = list(cfg.gpus)
    main_logger.info("Using GPUs: %s", gpu_ids)

    # Dispatch workers via fault-tolerant isolated processes.
    main_logger.info("Dispatching %d workers...", len(gpu_ids))
    start_time = time.time()

    failures = run_parallel_isolated(generation_worker, all_problems, gpu_ids, cfg)

    elapsed = time.time() - start_time
    main_logger.info("All workers finished in %.1f seconds", elapsed)

    # Report results.
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
        for rank, gid, exitcode in failures:
            main_logger.error(
                "Worker rank %d on GPU %d exited with code %d", rank, gid, exitcode
            )

    main_logger.info("Output directory: %s", cfg.paths.traces_dir)
    main_logger.info("Trace generation complete.")


if __name__ == "__main__":
    main()
