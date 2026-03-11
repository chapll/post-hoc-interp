"""Generate DeepMath traces across all 5 non-overlapping pools.

Pools (zero leakage):
  A (direction extraction): 300 binary problems, 5 traces each
  B (sleeper SFT):          300 binary problems, 3 traces each
  C (evaluation/scanning):  200 binary problems, 3 traces each
  D (binary transfer test): 100 binary problems, 3 traces each
  E (non-binary transfer):  100 freeform problems, 3 traces each

Usage:
    python scripts/generate_deepmath_traces.py --pool A --gpu 0
    python scripts/generate_deepmath_traces.py --pool A --gpu 0,1,2,3
    python scripts/generate_deepmath_traces.py --pool E --gpu 0  # non-binary
"""

import argparse
import gc
import json
import logging
import os
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.datasets import load_deepmath_pool
from src.model import generate_trace, load_model
from src.multiprocess import run_parallel_isolated
from src.prompts import build_freeform_prompt, build_verification_prompt
from src.storage import derive_trace_seed, get_trace_json_path, write_trace_json
from src.tokens import parse_think_boundaries
from src.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)

POOL_CONFIG = {
    "A": {"n": 300, "traces_per_problem": 5, "binary": True},
    "B": {"n": 300, "traces_per_problem": 3, "binary": True},
    "C": {"n": 200, "traces_per_problem": 3, "binary": True},
    "D": {"n": 100, "traces_per_problem": 3, "binary": True},
    "E": {"n": 100, "traces_per_problem": 3, "binary": False},
}

MAX_NEW_TOKENS = 4096
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
BASE_SEED = 137


def generation_worker(rank, gpu_id, problems, pool, traces_per_problem, binary):
    """Worker: generate traces for assigned problems on a single GPU."""
    worker_logger = setup_logging(f"pool{pool}_worker_{rank}")
    torch.cuda.set_device(gpu_id)

    worker_logger.info(
        "Worker %d starting on GPU %d with %d problems (pool=%s, binary=%s)",
        rank, gpu_id, len(problems), pool, binary,
    )

    model, tokenizer = load_model(gpu_id=gpu_id, model_name="Qwen/Qwen3-8B")
    worker_logger.info("Worker %d: model loaded on GPU %d", rank, gpu_id)

    traces_dir = f"data/traces_v2/valid"
    failed_dir = f"data/traces_v2/failed"
    generated_count = 0
    skipped_count = 0
    failed_count = 0

    configs = ["A", "B"] if binary else [None]

    for prob_idx, problem in enumerate(problems):
        for config in configs:
            if binary:
                result = build_verification_prompt(problem, "swapped", config)
                problem_id = f"{problem['problem_id']}_cfg{config.lower()}"
            else:
                result = build_freeform_prompt(problem)
                problem_id = problem["problem_id"]

            prompt = result["verification_prompt"]
            label_mapping = result["label_mapping"]
            prompt_config = result["prompt_config"]

            for trace_idx in range(traces_per_problem):
                # Check if already generated
                out_path = os.path.join(
                    traces_dir, problem["dataset_name"], problem_id,
                    f"trace_{trace_idx:02d}.json",
                )
                if os.path.isfile(out_path):
                    skipped_count += 1
                    continue

                seed = derive_trace_seed(problem_id, trace_idx)
                set_seed(seed)

                try:
                    output_ids = generate_trace(
                        model, tokenizer, prompt,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                    # Re-raise CUDA OOM and critical runtime errors
                    raise
                except (ValueError, KeyError) as e:
                    # Only catch data-level issues (malformed prompt, missing key)
                    worker_logger.error(
                        "Worker %d: generation failed for %s trace %d: %s",
                        rank, problem_id, trace_idx, e,
                    )
                    failed_count += 1
                    continue

                token_ids = output_ids[0].tolist()
                generated_text = tokenizer.decode(token_ids, skip_special_tokens=False)
                boundaries = parse_think_boundaries(token_ids)
                has_think = len(boundaries) > 0

                metadata = {
                    "problem_id": problem_id,
                    "dataset_name": problem["dataset_name"],
                    "ground_truth": problem["ground_truth"],
                    "question": problem["question"],
                    "original_answer": problem["original_answer"],
                    "verification_prompt": prompt,
                    "presented_answer": problem.get("presented_answer", ""),
                    "trace_index": trace_idx,
                    "seed": seed,
                    "model_name": "Qwen/Qwen3-8B",
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                    "top_k": TOP_K,
                    "num_tokens": len(token_ids),
                    "has_think": has_think,
                    "think_boundaries": boundaries,
                    "num_think_blocks": len(boundaries),
                    "pool": pool,
                    "binary": binary,
                }
                if label_mapping:
                    metadata["prompt_config"] = prompt_config
                    metadata["label_mapping"] = label_mapping
                    gt = problem["ground_truth"]
                    metadata["valid_token"] = label_mapping["valid"]
                    metadata["invalid_token"] = label_mapping["invalid"]
                if problem.get("difficulty"):
                    metadata["difficulty"] = problem["difficulty"]
                if problem.get("topic"):
                    metadata["topic"] = problem["topic"]

                trace_data = {
                    "token_ids": token_ids,
                    "generated_text": generated_text,
                    "metadata": metadata,
                }

                if has_think:
                    target_dir = traces_dir
                else:
                    target_dir = failed_dir

                save_path = os.path.join(
                    target_dir, problem["dataset_name"], problem_id,
                    f"trace_{trace_idx:02d}.json",
                )
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "w") as f:
                    json.dump(trace_data, f)
                generated_count += 1

        if (prob_idx + 1) % 10 == 0:
            worker_logger.info(
                "Worker %d: %d/%d problems done (%d gen, %d skip, %d fail)",
                rank, prob_idx + 1, len(problems),
                generated_count, skipped_count, failed_count,
            )
            gc.collect()
            torch.cuda.empty_cache()

    worker_logger.info(
        "Worker %d done: %d generated, %d skipped, %d failed",
        rank, generated_count, skipped_count, failed_count,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True, choices=list(POOL_CONFIG))
    parser.add_argument("--gpu", default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    args = parser.parse_args()

    setup_logging("generate_deepmath")
    pool = args.pool
    cfg = POOL_CONFIG[pool]
    gpu_ids = [int(g) for g in args.gpu.split(",")]

    print(f"Pool {pool}: n={cfg['n']}, traces_per={cfg['traces_per_problem']}, "
          f"binary={cfg['binary']}, GPUs={gpu_ids}")

    # Load problems for this pool
    problems = load_deepmath_pool(
        pool=pool, n=cfg["n"], seed=args.seed, binary=cfg["binary"],
    )
    print(f"Loaded {len(problems)} problems for pool {pool}")

    # Create output dirs
    os.makedirs("data/traces_v2/valid", exist_ok=True)
    os.makedirs("data/traces_v2/failed", exist_ok=True)

    if len(gpu_ids) == 1:
        # Single GPU: run directly
        generation_worker(
            rank=0, gpu_id=gpu_ids[0], problems=problems,
            pool=pool, traces_per_problem=cfg["traces_per_problem"],
            binary=cfg["binary"],
        )
    else:
        # Multi-GPU: distribute problems
        chunks = [[] for _ in gpu_ids]
        for i, p in enumerate(problems):
            chunks[i % len(gpu_ids)].append(p)

        run_parallel_isolated(
            worker_fn=generation_worker,
            gpu_ids=gpu_ids,
            worker_args=[(chunk, pool, cfg["traces_per_problem"], cfg["binary"])
                         for chunk in chunks],
        )

    print(f"Pool {pool} generation complete.")


if __name__ == "__main__":
    main()
