"""Generate DeepMath traces using vLLM for fast batched inference.

Uses vLLM's offline LLM class with tensor parallelism across multiple GPUs
for massively faster generation than sequential HF generate().

Pools (zero leakage):
  A (direction extraction): 300 binary problems, 5 traces each
  B (sleeper SFT):          300 binary problems, 3 traces each
  C (evaluation/scanning):  200 binary problems, 3 traces each
  D (binary transfer test): 100 binary problems, 3 traces each
  E (non-binary transfer):  100 freeform problems, 3 traces each

Usage:
    python scripts/generate_deepmath_vllm.py --pool A --tp 4
    python scripts/generate_deepmath_vllm.py --pool A --tp 8
    python scripts/generate_deepmath_vllm.py --all --tp 8
"""

import argparse
import hashlib
import json
import os
import random
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_deepmath_pool
from src.prompts import build_freeform_prompt, build_verification_prompt
from src.tokens import parse_think_boundaries

POOL_CONFIG = {
    "A": {"n": 6000, "traces_per_problem": 1, "binary": True},
    "B": {"n": 3000, "traces_per_problem": 1, "binary": True},
    "C": {"n": 3000, "traces_per_problem": 1, "binary": True},
    "D": {"n": 1500, "traces_per_problem": 1, "binary": True},
    "E": {"n": 1500, "traces_per_problem": 1, "binary": False},
}

MAX_NEW_TOKENS = 16384
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = 20
BASE_SEED = 137
MODEL_NAME = "Qwen/Qwen3-8B"


def derive_seed(problem_id, trace_idx):
    """Deterministic seed from problem_id and trace_idx."""
    raw = f"{problem_id}::{trace_idx}"
    h = hashlib.md5(raw.encode()).hexdigest()
    return int(h[:8], 16)


def build_all_requests(problems, pool, cfg):
    """Build all (prompt_text, metadata) pairs for a pool."""
    binary = cfg["binary"]
    traces_per = cfg["traces_per_problem"]
    configs = ["A", "B"] if binary else [None]

    requests = []
    for problem in problems:
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

            for trace_idx in range(traces_per):
                out_path = os.path.join(
                    "data/traces_v2/valid", problem["dataset_name"],
                    problem_id, f"trace_{trace_idx:02d}.json",
                )
                if os.path.isfile(out_path):
                    continue

                seed = derive_seed(problem_id, trace_idx)
                requests.append({
                    "prompt": prompt,
                    "problem_id": problem_id,
                    "problem": problem,
                    "trace_idx": trace_idx,
                    "seed": seed,
                    "label_mapping": label_mapping,
                    "prompt_config": prompt_config,
                    "pool": pool,
                    "binary": binary,
                })
    return requests


def format_chat_prompt(tokenizer, prompt_text):
    """Format prompt using Qwen3 chat template with thinking enabled."""
    messages = [{"role": "user", "content": prompt_text}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )


def save_trace(request, token_ids, generated_text):
    """Save a single trace to disk."""
    problem = request["problem"]
    problem_id = request["problem_id"]
    trace_idx = request["trace_idx"]

    boundaries = parse_think_boundaries(token_ids)
    has_think = len(boundaries) > 0

    metadata = {
        "problem_id": problem_id,
        "dataset_name": problem["dataset_name"],
        "ground_truth": problem["ground_truth"],
        "question": problem["question"],
        "original_answer": problem["original_answer"],
        "verification_prompt": request["prompt"],
        "presented_answer": problem.get("presented_answer", ""),
        "trace_index": trace_idx,
        "seed": request["seed"],
        "model_name": MODEL_NAME,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "num_tokens": len(token_ids),
        "has_think": has_think,
        "think_boundaries": boundaries,
        "num_think_blocks": len(boundaries),
        "pool": request["pool"],
        "binary": request["binary"],
    }
    if request["label_mapping"]:
        metadata["prompt_config"] = request["prompt_config"]
        metadata["label_mapping"] = request["label_mapping"]
        gt = problem["ground_truth"]
        metadata["valid_token"] = request["label_mapping"]["valid"]
        metadata["invalid_token"] = request["label_mapping"]["invalid"]
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
        target_dir = "data/traces_v2/valid"
    else:
        target_dir = "data/traces_v2/failed"

    save_path = os.path.join(
        target_dir, problem["dataset_name"], problem_id,
        f"trace_{trace_idx:02d}.json",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(trace_data, f)
    return has_think


def generate_pool(pool, tp_size):
    """Generate all traces for a single pool using vLLM."""
    from vllm import LLM, SamplingParams

    cfg = POOL_CONFIG[pool]
    print(f"Pool {pool}: n={cfg['n']}, traces_per={cfg['traces_per_problem']}, "
          f"binary={cfg['binary']}, tp={tp_size}")

    # Load problems
    problems = load_deepmath_pool(
        pool=pool, n=cfg["n"], seed=BASE_SEED, binary=cfg["binary"],
    )
    print(f"Loaded {len(problems)} problems")

    # Build all requests
    requests = build_all_requests(problems, pool, cfg)
    print(f"{len(requests)} traces to generate (skipped existing)")

    if not requests:
        print("Nothing to generate.")
        return

    # Load vLLM model
    print(f"Loading {MODEL_NAME} with tensor_parallel_size={tp_size}...")
    llm = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=16384 + 512,  # input + output
        seed=BASE_SEED,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # Format all prompts
    print("Formatting prompts...")
    formatted_prompts = []
    for req in requests:
        formatted = format_chat_prompt(tokenizer, req["prompt"])
        formatted_prompts.append(formatted)

    # Generate in batches with different seeds per trace
    # vLLM doesn't support per-request seeds in the same batch easily,
    # so we use a single sampling config and rely on sampling randomness
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        max_tokens=MAX_NEW_TOKENS,
        seed=BASE_SEED,
    )

    print(f"Generating {len(formatted_prompts)} traces...")
    start_time = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params)
    elapsed = time.time() - start_time
    print(f"Generation done in {elapsed:.1f}s "
          f"({len(outputs)} outputs, {elapsed/len(outputs):.1f}s/trace avg)")

    # Save results
    os.makedirs("data/traces_v2/valid", exist_ok=True)
    os.makedirs("data/traces_v2/failed", exist_ok=True)

    valid_count = 0
    failed_count = 0
    for req, output in zip(requests, outputs):
        # Get full token IDs (prompt + generated)
        prompt_token_ids = list(output.prompt_token_ids)
        generated_ids = list(output.outputs[0].token_ids)
        full_token_ids = prompt_token_ids + generated_ids
        full_text = tokenizer.decode(full_token_ids, skip_special_tokens=False)

        has_think = save_trace(req, full_token_ids, full_text)
        if has_think:
            valid_count += 1
        else:
            failed_count += 1

    print(f"Pool {pool} complete: {valid_count} valid, {failed_count} failed "
          f"(no think block)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=list(POOL_CONFIG), default=None,
                        help="Single pool to generate")
    parser.add_argument("--all", action="store_true",
                        help="Generate all pools sequentially")
    parser.add_argument("--tp", type=int, default=4,
                        help="Tensor parallel size (number of GPUs)")
    args = parser.parse_args()

    if not args.all and args.pool is None:
        parser.error("Specify --pool or --all")

    pools = list(POOL_CONFIG) if args.all else [args.pool]

    for pool in pools:
        print(f"\n{'='*40}")
        print(f"Generating Pool {pool}")
        print(f"{'='*40}")
        generate_pool(pool, args.tp)

    print("\nAll pools complete.")


if __name__ == "__main__":
    main()
