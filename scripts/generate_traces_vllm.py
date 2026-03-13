"""Fast trace generation using vLLM continuous batching.

Replaces generate_traces.py for ~10x faster generation. Produces identical
output format (same JSON structure, same directory layout).

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_traces_vllm.py
    CUDA_VISIBLE_DEVICES=1,2 python scripts/generate_traces_vllm.py --tp 2
"""

import argparse
import json
import logging
import os
import time

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.config import load_config
from src.datasets import (
    load_deepmath,
    load_folio,
    load_gsm_symbolic_contamination,
    load_strategyqa,
)
from src.prompts import build_verification_prompt
from src.storage import (
    derive_trace_seed,
    get_failed_trace_json_path,
    get_trace_json_path,
    write_trace_json,
)
from src.tokens import parse_think_boundaries
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--max-model-len", type=int, default=12288,
                        help="Max sequence length (input + output)")
    parser.add_argument("--gpu-mem", type=float, default=0.90,
                        help="GPU memory utilization fraction")
    parser.add_argument("--shard-id", type=int, default=None,
                        help="Shard index (0-based) for multi-GPU parallelism")
    parser.add_argument("--num-shards", type=int, default=None,
                        help="Total number of shards")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graph capture (avoids nvcc requirement)")
    args = parser.parse_args()

    main_logger = setup_logging("generate_traces_vllm")
    main_logger.info("Starting vLLM trace generation")

    # Load config (programmatic mode: base.yaml + generate.yaml)
    cfg = load_config("configs/generate.yaml")

    main_logger.info("Config: model=%s, max_new_tokens=%d, traces_per_problem=%d",
                     cfg.model.name, cfg.model.max_new_tokens, cfg.data.traces_per_problem)

    # Build problem list
    main_logger.info("Loading datasets...")
    gsm_problems = load_gsm_symbolic_contamination(
        n_per_type=cfg.data.gsm_problems_per_type, seed=cfg.seed
    )
    main_logger.info("GSM-Symbolic: %d problems", len(gsm_problems))

    folio_problems = load_folio(n=cfg.data.folio_problems, seed=cfg.seed)
    main_logger.info("FOLIO: %d problems", len(folio_problems))

    strategyqa_problems = load_strategyqa(n=cfg.data.strategyqa_problems, seed=cfg.seed)
    main_logger.info("StrategyQA: %d problems", len(strategyqa_problems))

    deepmath_problems = load_deepmath(n=cfg.data.deepmath_problems, seed=cfg.seed)
    main_logger.info("DeepMath: %d problems", len(deepmath_problems))

    all_problems = gsm_problems + folio_problems + strategyqa_problems + deepmath_problems
    main_logger.info("Total base problems: %d", len(all_problems))

    # Expand with config A/B
    label_protocol = cfg.data.get("label_protocol", "swapped")
    expanded = []
    for problem in all_problems:
        if label_protocol == "swapped":
            for config_name in ["A", "B"]:
                result = build_verification_prompt(problem, "swapped", config_name)
                variant = dict(problem)
                variant["verification_prompt"] = result["verification_prompt"]
                variant["prompt_config"] = result["prompt_config"]
                variant["label_mapping"] = result["label_mapping"]
                variant["problem_id"] = f"{problem['problem_id']}_cfg{config_name.lower()}"
                expanded.append(variant)
        else:
            result = build_verification_prompt(problem, "yesno")
            variant = dict(problem)
            variant["verification_prompt"] = result["verification_prompt"]
            variant["prompt_config"] = result["prompt_config"]
            variant["label_mapping"] = result["label_mapping"]
            expanded.append(variant)

    all_problems = expanded
    main_logger.info("After %s expansion: %d problems", label_protocol, len(all_problems))

    traces_dir = cfg.paths.traces_dir
    traces_per_problem = cfg.data.traces_per_problem
    max_new_tokens = cfg.model.max_new_tokens

    # Filter to problems needing generation (resume support)
    to_generate = []
    skipped = 0
    for problem in all_problems:
        for trace_index in range(traces_per_problem):
            valid_path = get_trace_json_path(
                traces_dir, problem["dataset_name"], problem["problem_id"], trace_index,
            )
            failed_path = get_failed_trace_json_path(
                traces_dir, problem["dataset_name"], problem["problem_id"], trace_index,
            )
            if os.path.exists(valid_path) or os.path.exists(failed_path):
                skipped += 1
                continue
            to_generate.append((problem, trace_index))

    # Apply sharding if requested
    if args.shard_id is not None and args.num_shards is not None:
        total = len(to_generate)
        shard_size = (total + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, total)
        to_generate = to_generate[start:end]
        main_logger.info("Shard %d/%d: traces [%d:%d] of %d",
                         args.shard_id, args.num_shards, start, end, total)

    main_logger.info("To generate: %d traces (skipped %d existing)", len(to_generate), skipped)
    if not to_generate:
        main_logger.info("Nothing to generate.")
        return

    # Format prompts with chat template
    main_logger.info("Formatting %d prompts...", len(to_generate))
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    formatted_prompts = []
    sampling_params_list = []
    for problem, trace_index in to_generate:
        messages = [{"role": "user", "content": problem["verification_prompt"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        formatted_prompts.append(text)

        seed = derive_trace_seed(problem["problem_id"], trace_index)
        sampling_params_list.append(SamplingParams(
            temperature=cfg.generation.temperature,
            top_p=cfg.generation.top_p,
            top_k=cfg.generation.top_k,
            max_tokens=max_new_tokens,
            seed=seed,
        ))

    # Initialize vLLM
    main_logger.info("Loading vLLM model (tp=%d, max_model_len=%d)...",
                     args.tp, args.max_model_len)
    llm = LLM(
        model=cfg.model.name,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
    )

    # Generate all traces
    main_logger.info("Generating %d traces...", len(formatted_prompts))
    start_time = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params_list)
    elapsed = time.time() - start_time

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    main_logger.info("Generated %d traces in %.1fs (%.1f traces/min, %.0f tok/s)",
                     len(outputs), elapsed,
                     len(outputs) / (elapsed / 60),
                     total_tokens / elapsed)

    # Save results
    valid_count = 0
    failed_count = 0
    for i, output in enumerate(outputs):
        problem, trace_index = to_generate[i]

        prompt_token_ids = list(output.prompt_token_ids)
        generated_token_ids = list(output.outputs[0].token_ids)
        token_list = prompt_token_ids + generated_token_ids

        generated_text = tokenizer.decode(token_list, skip_special_tokens=False)

        blocks = parse_think_boundaries(token_list)
        has_think = len(blocks) > 0

        seed = derive_trace_seed(problem["problem_id"], trace_index)

        trace_data = {
            "token_ids": token_list,
            "generated_text": generated_text,
            "metadata": {
                "problem_id": problem["problem_id"],
                "dataset_name": problem["dataset_name"],
                "ground_truth": problem["ground_truth"],
                "question": problem["question"],
                "original_answer": problem["original_answer"],
                "verification_prompt": problem["verification_prompt"],
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

        if "is_fresh" in problem:
            trace_data["metadata"]["is_fresh"] = problem["is_fresh"]
        if "prompt_config" in problem:
            trace_data["metadata"]["prompt_config"] = problem["prompt_config"]
        if "label_mapping" in problem:
            trace_data["metadata"]["label_mapping"] = problem["label_mapping"]
            trace_data["metadata"]["valid_token"] = problem["label_mapping"]["valid"]
            trace_data["metadata"]["invalid_token"] = problem["label_mapping"]["invalid"]

        valid_path = get_trace_json_path(
            traces_dir, problem["dataset_name"], problem["problem_id"], trace_index,
        )
        failed_path = get_failed_trace_json_path(
            traces_dir, problem["dataset_name"], problem["problem_id"], trace_index,
        )

        if has_think:
            write_trace_json(trace_data, valid_path)
            valid_count += 1
        else:
            write_trace_json(trace_data, failed_path)
            failed_count += 1

    main_logger.info("Done. Valid: %d, Failed (no think): %d, Total: %d",
                     valid_count, failed_count, valid_count + failed_count)


if __name__ == "__main__":
    main()
