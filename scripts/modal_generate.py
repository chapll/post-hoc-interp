"""Modal-based trace generation for Qwen3-8B thinking-mode traces.

Runs trace generation on Modal's serverless A100 GPUs. Equivalent to
generate_traces.py but executes remotely on Modal infrastructure.

Usage:
    # One-time model download to cache volume
    modal run scripts/modal_generate.py --download-only

    # Run trace generation (4 containers by default)
    modal run scripts/modal_generate.py

    # Run with custom settings
    modal run scripts/modal_generate.py --n-containers 8 --gsm-problems 50

    # Download results
    modal volume get traces-output /valid ./data/traces/valid
    modal volume get traces-output /failed ./data/traces/failed
"""

import hashlib
import json
import os
import time

import modal

# ---------------------------------------------------------------------------
# Modal App and Resources
# ---------------------------------------------------------------------------

app = modal.App("post-hoc-interp-traces")

# Volume for caching model weights (persistent, read-heavy)
model_cache_vol = modal.Volume.from_name("qwen3-8b-cache", create_if_missing=True)

# Volume for storing trace outputs (persistent, write-heavy)
traces_vol = modal.Volume.from_name("traces-output", create_if_missing=True)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install(
        # Core ML
        "torch==2.9.0",
        "transformers>=4.51.0",
        "nnsight>=0.5.15",
        # Data handling
        "datasets>=4.5.0",
        "safetensors>=0.7.0",
        # Utilities
        "omegaconf>=2.3.0",
        "numpy>=2.4.1",
        "einops>=0.8.1",
        "tqdm>=4.67.1",
    )
    .add_local_python_source("src")
)


# ---------------------------------------------------------------------------
# Helper Functions (ported from src/storage.py and src/utils.py)
# ---------------------------------------------------------------------------


def derive_trace_seed(problem_id: str, trace_index: int) -> int:
    """Derive a deterministic seed from problem_id and trace_index.

    Uses MD5 hash for deterministic mixing. Per project rules, seed 42
    is never returned (returns 43 instead).
    """
    key = f"{problem_id}::{trace_index}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    seed = int(h[:8], 16) % (2**31)
    if seed == 42:
        seed = 43
    return seed


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Model Download Function (run once to populate cache)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={"/model-cache": model_cache_vol},
    timeout=1800,  # 30 minutes for initial download
)
def download_model(model_name: str = "Qwen/Qwen3-8B"):
    """Download model weights to the cache volume."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = "/model-cache"
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    print(f"Downloading {model_name} to {cache_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    print("Tokenizer downloaded.")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype="auto",
    )
    print(f"Model downloaded. Size: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Commit changes to volume
    model_cache_vol.commit()
    print("Model cached to volume.")


# ---------------------------------------------------------------------------
# Trace Generation Worker (runs on A100)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="a100-80gb",
    volumes={
        "/model-cache": model_cache_vol,
        "/traces": traces_vol,
    },
    timeout=7200,  # 2 hours per batch
    retries=2,
)
def generate_traces_batch(
    problems: list[dict],
    config: dict,
    batch_id: int,
) -> dict:
    """Generate traces for a batch of problems on a single A100.

    This is the core worker function. Each invocation handles one batch
    of problems. Multiple containers can run in parallel via .map().

    Args:
        problems: List of problem dicts with verification_prompt, etc.
        config: Dict with model_name, traces_per_problem, generation params.
        batch_id: Identifier for this batch (for logging).

    Returns:
        Dict with counts: generated, skipped, failed, valid.
    """
    import torch
    from nnsight import LanguageModel

    from src.tokens import parse_think_boundaries

    # Set up HuggingFace cache
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["TRANSFORMERS_CACHE"] = "/model-cache"

    print(f"Batch {batch_id}: Starting with {len(problems)} problems")

    # Load model from cache
    print(f"Batch {batch_id}: Loading model...")
    model = LanguageModel(
        config["model_name"],
        device_map="cuda:0",
        dtype=torch.bfloat16,
        dispatch=True,
    )
    tokenizer = model.tokenizer
    print(f"Batch {batch_id}: Model loaded")

    traces_dir = "/traces"
    traces_per_problem = config["traces_per_problem"]
    max_new_tokens = config["max_new_tokens"]

    results = {"generated": 0, "skipped": 0, "failed": 0, "valid": 0}

    for prob_idx, problem in enumerate(problems):
        problem_id = problem["problem_id"]
        dataset_name = problem["dataset_name"]
        prompt = problem["verification_prompt"]

        if prob_idx % 10 == 0:
            print(f"Batch {batch_id}: Processing problem {prob_idx + 1}/{len(problems)}")

        for trace_index in range(traces_per_problem):
            # Compute output paths
            valid_path = f"{traces_dir}/valid/{dataset_name}/{problem_id}/trace_{trace_index:02d}.json"
            failed_path = f"{traces_dir}/failed/{dataset_name}/{problem_id}/trace_{trace_index:02d}.json"

            # Resume check: skip if already exists
            if os.path.exists(valid_path) or os.path.exists(failed_path):
                results["skipped"] += 1
                continue

            # Deterministic seed
            seed = derive_trace_seed(problem_id, trace_index)
            set_seed(seed)

            try:
                # Generate trace using Qwen3's thinking mode
                messages = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )

                inputs = tokenizer(text, return_tensors="pt")
                input_ids = inputs["input_ids"].cuda()
                attention_mask = inputs["attention_mask"].cuda()

                with torch.no_grad():
                    output_ids = model._model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=config["temperature"],
                        top_p=config["top_p"],
                        top_k=config["top_k"],
                        do_sample=True,
                    )

                token_list = output_ids[0].tolist()
                generated_text = tokenizer.decode(token_list, skip_special_tokens=False)

                # Parse think boundaries
                blocks = parse_think_boundaries(token_list)
                has_think = len(blocks) > 0

                # Build trace data
                trace_data = {
                    "token_ids": token_list,
                    "generated_text": generated_text,
                    "metadata": {
                        "problem_id": problem_id,
                        "dataset_name": dataset_name,
                        "ground_truth": problem["ground_truth"],
                        "question": problem["question"],
                        "original_answer": problem.get("original_answer", ""),
                        "verification_prompt": prompt,
                        "presented_answer": problem.get("presented_answer", ""),
                        "trace_index": trace_index,
                        "seed": seed,
                        "model_name": config["model_name"],
                        "max_new_tokens": max_new_tokens,
                        "temperature": config["temperature"],
                        "top_p": config["top_p"],
                        "top_k": config["top_k"],
                        "num_tokens": len(token_list),
                        "num_think_blocks": len(blocks),
                        "has_think": has_think,
                        "think_boundaries": blocks,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                }

                # Include is_fresh metadata if present (GSM-Symbolic)
                if "is_fresh" in problem:
                    trace_data["metadata"]["is_fresh"] = problem["is_fresh"]

                # Include swapped-label protocol metadata
                if "prompt_config" in problem:
                    trace_data["metadata"]["prompt_config"] = problem["prompt_config"]
                if "label_mapping" in problem:
                    trace_data["metadata"]["label_mapping"] = problem["label_mapping"]
                    trace_data["metadata"]["valid_token"] = problem["label_mapping"]["valid"]
                    trace_data["metadata"]["invalid_token"] = problem["label_mapping"]["invalid"]

                # Route to valid/ or failed/
                dest_path = valid_path if has_think else failed_path

                # Write atomically
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(dest_path, "w") as f:
                    json.dump(trace_data, f, indent=2)

                results["generated"] += 1
                if has_think:
                    results["valid"] += 1

            except Exception as e:
                results["failed"] += 1
                print(f"Batch {batch_id}: Error generating {problem_id} trace {trace_index}: {e}")

    # Commit trace writes to volume
    traces_vol.commit()

    print(
        f"Batch {batch_id}: Done. Generated: {results['generated']}, "
        f"Valid: {results['valid']}, Skipped: {results['skipped']}, "
        f"Failed: {results['failed']}"
    )

    return {"batch_id": batch_id, **results}


# ---------------------------------------------------------------------------
# Orchestrator Function (distributes work across containers)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    timeout=3600,
)
def generate_all_traces(
    n_containers: int = 4,
    gsm_problems_per_type: int = 100,
    folio_problems: int = 100,
    strategyqa_problems: int = 100,
    traces_per_problem: int = 3,
    max_new_tokens: int = 4096,
    seed: int = 137,
) -> dict:
    """Orchestrate trace generation across multiple containers.

    Loads datasets, builds problems with swapped-label protocol,
    splits work, and dispatches to parallel workers.

    Args:
        n_containers: Number of parallel A100 containers to use.
        gsm_problems_per_type: Problems per type for GSM-Symbolic.
        folio_problems: Number of FOLIO problems.
        strategyqa_problems: Number of StrategyQA problems.
        traces_per_problem: Traces to generate per problem.
        max_new_tokens: Max tokens for generation.
        seed: Random seed for dataset loading.

    Returns:
        Aggregated results dict with total counts.
    """
    from src.datasets import (
        load_folio,
        load_gsm_symbolic_contamination,
        load_strategyqa,
    )
    from src.prompts import build_verification_prompt

    print("Loading datasets...")

    # Load datasets
    gsm_probs = load_gsm_symbolic_contamination(n_per_type=gsm_problems_per_type, seed=seed)
    print(f"GSM-Symbolic: {len(gsm_probs)} problems")

    folio_probs = load_folio(n=folio_problems, seed=seed)
    print(f"FOLIO: {len(folio_probs)} problems")

    strategyqa_probs = load_strategyqa(n=strategyqa_problems, seed=seed)
    print(f"StrategyQA: {len(strategyqa_probs)} problems")

    all_problems = gsm_probs + folio_probs + strategyqa_probs
    print(f"Total base problems: {len(all_problems)}")

    # Apply swapped-label protocol (A/B configs)
    print("Applying swapped-label protocol...")
    expanded_problems = []
    for problem in all_problems:
        for config_name in ["A", "B"]:
            result = build_verification_prompt(problem, "swapped", config_name)
            variant = dict(problem)
            variant["verification_prompt"] = result["verification_prompt"]
            variant["prompt_config"] = result["prompt_config"]
            variant["label_mapping"] = result["label_mapping"]
            variant["problem_id"] = f"{problem['problem_id']}_cfg{config_name.lower()}"
            expanded_problems.append(variant)

    print(f"After expansion: {len(expanded_problems)} problems")
    print(f"Expected traces: {len(expanded_problems) * traces_per_problem}")

    # Split into batches for parallel processing
    batch_size = len(expanded_problems) // n_containers
    batches = []
    for i in range(n_containers):
        start = i * batch_size
        end = start + batch_size if i < n_containers - 1 else len(expanded_problems)
        batches.append(expanded_problems[start:end])

    print(f"Split into {len(batches)} batches")
    for i, batch in enumerate(batches):
        print(f"  Batch {i}: {len(batch)} problems")

    # Config for workers
    config = {
        "model_name": "Qwen/Qwen3-8B",
        "traces_per_problem": traces_per_problem,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
    }

    # Dispatch to workers via .map() for parallel execution
    print(f"Dispatching to {n_containers} containers...")
    start_time = time.time()

    results = list(
        generate_traces_batch.map(
            batches,
            kwargs={"config": config},
            order_outputs=False,
        )
    )

    elapsed = time.time() - start_time
    print(f"All containers finished in {elapsed:.1f}s")

    # Aggregate results
    total = {"generated": 0, "skipped": 0, "failed": 0, "valid": 0}
    for r in results:
        for key in total:
            total[key] += r[key]

    print(f"Total results: {total}")
    return total


# ---------------------------------------------------------------------------
# Local Entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    n_containers: int = 4,
    download_only: bool = False,
    gsm_problems: int = 100,
    folio_problems: int = 100,
    strategyqa_problems: int = 100,
    traces_per_problem: int = 3,
):
    """Entry point for `modal run scripts/modal_generate.py`.

    Args:
        n_containers: Number of parallel A100 containers.
        download_only: If True, only download and cache the model.
        gsm_problems: GSM-Symbolic problems per type (fresh/original).
        folio_problems: Number of FOLIO problems.
        strategyqa_problems: Number of StrategyQA problems.
        traces_per_problem: Traces to generate per problem.
    """
    if download_only:
        print("Downloading model to cache volume...")
        download_model.remote()
        print("Model download complete.")
        return

    print(f"Starting trace generation with {n_containers} containers...")
    print(f"  GSM problems per type: {gsm_problems}")
    print(f"  FOLIO problems: {folio_problems}")
    print(f"  StrategyQA problems: {strategyqa_problems}")
    print(f"  Traces per problem: {traces_per_problem}")

    results = generate_all_traces.remote(
        n_containers=n_containers,
        gsm_problems_per_type=gsm_problems,
        folio_problems=folio_problems,
        strategyqa_problems=strategyqa_problems,
        traces_per_problem=traces_per_problem,
    )

    print("\n=== Generation Complete ===")
    print(f"Generated: {results['generated']}")
    print(f"Valid (has <think>): {results['valid']}")
    print(f"Skipped (resume): {results['skipped']}")
    print(f"Failed: {results['failed']}")
    print("\nTo download results:")
    print("  modal volume get traces-output /valid ./data/traces/valid")
    print("  modal volume get traces-output /failed ./data/traces/failed")
