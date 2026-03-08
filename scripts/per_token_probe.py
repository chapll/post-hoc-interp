"""Per-token force-answer probing experiment (multi-trace).

Probes target traces at every thinking token position using KV-cache-efficient
descending-order probing. Supports multiple traces in a single run with resume
(traces with existing results.parquet are skipped).

Usage:
    python scripts/per_token_probe.py --config configs/per_token_probe.yaml
"""

import copy
import json
import logging
import os
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
from src.utils import set_seed, setup_logging

# Import probing helpers from Phase 2.2 (single source of truth)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.force_answer_experiment import (
    CLOSE_TOKEN_IDS,
    MU_TOKEN_ID,
    NU_TOKEN_ID,
    compute_probe_positions,
    extract_answer,
    extract_mu_nu_probs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace loading (single trace by path)
# ---------------------------------------------------------------------------


def load_single_trace(traces_dir, problem_id, dataset, trace_index=0):
    """Load a single specific trace directly by path.

    Does NOT use load_traces_for_experiment() -- that function does
    dataset-balanced random selection. For one specific trace, load
    it directly by path.

    Args:
        traces_dir: Root traces directory (contains valid/ subdirectory).
        problem_id: Problem identifier (e.g. "gsm_symbolic_68_fresh_false_cfga").
        dataset: Dataset subdirectory name (e.g. "gsm_symbolic").
        trace_index: Trace file index (default 0 for trace_00.json).

    Returns:
        Dict with keys: trace_path, token_ids, metadata, is_correct, dataset.
    """
    trace_path = os.path.join(
        traces_dir, "valid", dataset, problem_id,
        f"trace_{trace_index:02d}.json",
    )

    with open(trace_path) as f:
        trace = json.load(f)

    meta = trace["metadata"]

    # Determine correctness for this trace
    ground_truth = meta["ground_truth"]
    valid_token = meta["valid_token"]
    invalid_token = meta["invalid_token"]
    expected_token = valid_token if ground_truth == "Yes" else invalid_token

    generated_text = trace.get("generated_text", "")
    think_end_idx = generated_text.rfind("</think>")
    answer_text = generated_text[think_end_idx:] if think_end_idx >= 0 else generated_text
    is_correct = expected_token.lower() in answer_text.lower()

    return {
        "trace_path": trace_path,
        "token_ids": trace["token_ids"],
        "metadata": meta,
        "is_correct": is_correct,
        "dataset": dataset,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run per-token dense probing on target traces (single or multi-trace)."""
    logger_main = setup_logging("per_token_probe")
    cfg = load_config()
    set_seed(cfg.seed)

    ptp = cfg.per_token_probe
    traces_dir = cfg.paths.traces_dir
    output_dir = cfg.paths.output_dir

    # Parse traces list with backward compat for single-trace configs
    if hasattr(ptp, "traces") and ptp.traces:
        traces = list(ptp.traces)
    else:
        # Backward compat: single-trace config
        traces = [OmegaConf.create({
            "problem_id": ptp.problem_id,
            "dataset": ptp.dataset,
            "trace_index": ptp.trace_index,
        })]

    logger_main.info("Traces to probe: %d", len(traces))
    logger_main.info("Traces directory: %s", traces_dir)
    logger_main.info("Output directory: %s", output_dir)

    # -------------------------------------------------------------------
    # Load model ONCE (shared across all traces)
    # -------------------------------------------------------------------
    gpu_id = cfg.gpus[0]
    logger_main.info("Loading model on GPU %d...", gpu_id)
    model, tokenizer = load_model(gpu_id=gpu_id)
    raw_model = model._model
    device = next(raw_model.parameters()).device
    logger_main.info("Model loaded on %s", device)

    # -------------------------------------------------------------------
    # Loop over traces
    # -------------------------------------------------------------------
    processed = 0
    skipped = 0
    total_start = time.time()

    for trace_idx, trace_cfg in enumerate(traces):
        problem_id = trace_cfg.problem_id
        dataset = trace_cfg.dataset
        trace_index = trace_cfg.trace_index if hasattr(trace_cfg, "trace_index") else 0

        trace_output_dir = os.path.join(output_dir, problem_id)

        # Resume check: skip if results already exist
        existing_parquet = os.path.join(trace_output_dir, "results.parquet")
        if os.path.isfile(existing_parquet):
            logger_main.info(
                "[%d/%d] Skipping %s -- results.parquet already exists",
                trace_idx + 1, len(traces), problem_id,
            )
            skipped += 1
            continue

        logger_main.info(
            "[%d/%d] Processing %s (trace_%02d)...",
            trace_idx + 1, len(traces), problem_id, trace_index,
        )

        # ---------------------------------------------------------------
        # 1. Load trace
        # ---------------------------------------------------------------
        trace_data = load_single_trace(traces_dir, problem_id, dataset, trace_index)
        meta = trace_data["metadata"]
        logger_main.info(
            "Loaded trace: %s (correct=%s, ground_truth=%s)",
            meta["problem_id"], trace_data["is_correct"], meta["ground_truth"],
        )

        # ---------------------------------------------------------------
        # 2. Compute think_length dynamically
        # ---------------------------------------------------------------
        think_start, think_end = meta["think_boundaries"][0]
        think_length = think_end - think_start - 1
        logger_main.info("Think length: %d tokens", think_length)

        # ---------------------------------------------------------------
        # 3. Create cfg_shim and compute probe positions
        # ---------------------------------------------------------------
        num_probes = think_length  # probe every position (min_probe_spacing=1)
        cfg_shim = OmegaConf.create({
            "force_answer": {
                "num_probes": num_probes,
                "min_probe_spacing": ptp.min_probe_spacing,
                "max_new_tokens": ptp.max_new_tokens,
            }
        })

        positions = compute_probe_positions(
            think_length,
            cfg_shim.force_answer.num_probes,
            cfg_shim.force_answer.min_probe_spacing,
        )
        logger_main.info(
            "Probing %d positions (think_length=%d)", len(positions), think_length,
        )

        # ---------------------------------------------------------------
        # 4. Prepare token sequences
        # ---------------------------------------------------------------
        token_ids = trace_data["token_ids"]
        prefix_ids = token_ids[:think_start + 1]
        think_content_ids = token_ids[think_start + 1 : think_start + 1 + think_length]
        full_input_ids = prefix_ids + think_content_ids

        # Determine expected answer
        ground_truth = meta["ground_truth"]
        valid_token = meta["valid_token"]
        invalid_token = meta["invalid_token"]
        expected_token = valid_token if ground_truth == "Yes" else invalid_token
        correct_token_id = (
            MU_TOKEN_ID if expected_token == "Mu" else NU_TOKEN_ID
        )

        logger_main.info(
            "Expected answer: %s (ground_truth=%s, valid=%s, invalid=%s)",
            expected_token, ground_truth, valid_token, invalid_token,
        )
        logger_main.info(
            "Prefix: %d tokens, think content: %d tokens, full input: %d tokens",
            len(prefix_ids), len(think_content_ids), len(full_input_ids),
        )

        # ---------------------------------------------------------------
        # 5. Prefill KV cache and probe in descending order
        # ---------------------------------------------------------------
        logger_main.info("Prefilling KV cache with %d tokens...", len(full_input_ids))
        full_tensor = torch.tensor([full_input_ids], dtype=torch.long, device=device)
        cache = DynamicCache()
        with torch.no_grad():
            raw_model(input_ids=full_tensor, past_key_values=cache, use_cache=True)
        logger_main.info("KV cache prefilled")

        results = []
        start_time = time.time()
        descending_positions = sorted(positions, reverse=True)

        for i, pos in enumerate(tqdm(descending_positions, desc=f"Probing {problem_id}")):
            cache_len = len(prefix_ids) + pos
            cache.crop(cache_len)

            # Build input: prefix + think_content[:pos] + close_tokens
            input_for_generate = full_input_ids[:cache_len] + CLOSE_TOKEN_IDS
            input_tensor = torch.tensor(
                [input_for_generate], dtype=torch.long, device=device,
            )
            attention_mask = torch.ones_like(input_tensor)

            with torch.no_grad():
                gen_output = raw_model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    past_key_values=copy.deepcopy(cache),
                    max_new_tokens=ptp.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_logits=True,
                )

            generated_ids = gen_output.sequences[0, len(input_for_generate):].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

            # Extract first token logits
            if len(gen_output.logits) > 0:
                first_logits = gen_output.logits[0][0]  # shape [vocab_size]
            else:
                logger_main.warning("No logits returned for probe pos %d", pos)
                first_logits = torch.zeros(
                    raw_model.config.vocab_size, device=device,
                )

            answer_info = extract_answer(generated_ids, generated_text, expected_token)
            prob_info = extract_mu_nu_probs(first_logits, correct_token_id)

            normalized_position = pos / (think_length - 1) if think_length > 1 else 0.5
            correct_token_prob = (
                prob_info["mu_prob"] if expected_token == "Mu"
                else prob_info["nu_prob"]
            )

            row = {
                "trace_id": meta["problem_id"],
                "dataset": trace_data["dataset"],
                "problem_id": meta["problem_id"],
                "probe_position": pos,
                "normalized_position": normalized_position,
                "correct": answer_info["correct"],
                "no_answer": answer_info["no_answer"],
                "mu_prob": prob_info["mu_prob"],
                "nu_prob": prob_info["nu_prob"],
                "correct_token_prob": correct_token_prob,
                "correct_token_rank": prob_info["correct_token_rank"],
                "ground_truth": ground_truth,
                "is_correct_trace": trace_data["is_correct"],
                "is_fresh": meta.get("is_fresh", None),
                "prompt_config": meta.get("prompt_config", None),
                "valid_token": valid_token,
                "invalid_token": invalid_token,
                "token_answer": answer_info["token_answer"],
                "string_answer": answer_info["string_answer"],
                "agreement": answer_info["agreement"],
                "think_length": think_length,
            }
            results.append(row)

            # Periodic memory cleanup
            if (i + 1) % ptp.empty_cache_interval == 0:
                torch.cuda.empty_cache()
                logger_main.info(
                    "Cleared CUDA cache after %d probes (%.1fs elapsed)",
                    i + 1, time.time() - start_time,
                )

        elapsed = time.time() - start_time
        logger_main.info(
            "Probing complete for %s: %d positions in %.1fs (%.2fs/probe)",
            problem_id, len(results), elapsed, elapsed / len(results) if results else 0,
        )

        # ---------------------------------------------------------------
        # 6. Save results as parquet (sorted by probe_position ascending)
        # ---------------------------------------------------------------
        os.makedirs(trace_output_dir, exist_ok=True)

        df = pd.DataFrame(results)
        df = df.sort_values("probe_position").reset_index(drop=True)
        parquet_path = os.path.join(trace_output_dir, "results.parquet")
        df.to_parquet(parquet_path, index=False)
        logger_main.info("Saved %d rows to %s", len(df), parquet_path)

        # ---------------------------------------------------------------
        # 7. Compute transition characterization and save JSON summary
        # ---------------------------------------------------------------
        correct_array = df["correct"].values.astype(int)
        prob_array = df["correct_token_prob"].values

        # Transition counts
        transitions = np.diff(correct_array)
        n_correct_to_incorrect = int(np.sum(transitions == -1))
        n_incorrect_to_correct = int(np.sum(transitions == 1))

        # Longest runs
        def longest_run(arr, value):
            max_run = 0
            current = 0
            for v in arr:
                if v == value:
                    current += 1
                    max_run = max(max_run, current)
                else:
                    current = 0
            return max_run

        longest_correct_run = longest_run(correct_array, 1)
        longest_incorrect_run = longest_run(correct_array, 0)

        # First sustained correctness (5 consecutive correct)
        sustained_k = 5
        first_sustained = None
        for idx in range(len(correct_array) - sustained_k + 1):
            if all(correct_array[idx : idx + sustained_k]):
                first_sustained = int(df.iloc[idx]["probe_position"])
                break

        # Probability variance in middle third
        n = len(prob_array)
        mid_start = n // 3
        mid_end = 2 * n // 3
        mid_prob_variance = float(np.var(prob_array[mid_start:mid_end]))

        # Overall accuracy
        overall_accuracy = float(correct_array.mean())

        # Accuracy in first/middle/last thirds
        third = n // 3
        first_third_acc = float(correct_array[:third].mean()) if third > 0 else 0.0
        middle_third_acc = float(correct_array[third : 2 * third].mean()) if third > 0 else 0.0
        last_third_acc = float(correct_array[2 * third :].mean()) if n - 2 * third > 0 else 0.0

        summary = {
            "problem_id": problem_id,
            "dataset": dataset,
            "trace_index": trace_index,
            "think_length": think_length,
            "num_probes": len(results),
            "elapsed_seconds": round(elapsed, 1),
            "seconds_per_probe": round(elapsed / len(results), 3) if results else 0,
            "overall_accuracy": round(overall_accuracy, 4),
            "first_third_accuracy": round(first_third_acc, 4),
            "middle_third_accuracy": round(middle_third_acc, 4),
            "last_third_accuracy": round(last_third_acc, 4),
            "transition_characterization": {
                "correct_to_incorrect_transitions": n_correct_to_incorrect,
                "incorrect_to_correct_transitions": n_incorrect_to_correct,
                "total_transitions": n_correct_to_incorrect + n_incorrect_to_correct,
                "longest_correct_run": longest_correct_run,
                "longest_incorrect_run": longest_incorrect_run,
                "first_sustained_correct_position": first_sustained,
                "sustained_k": sustained_k,
                "middle_third_prob_variance": round(mid_prob_variance, 6),
            },
        }

        summary_path = os.path.join(trace_output_dir, "results_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger_main.info("Saved summary to %s", summary_path)

        # ---------------------------------------------------------------
        # 8. Log per-trace summary
        # ---------------------------------------------------------------
        logger_main.info("--- Trace Summary: %s ---", problem_id)
        logger_main.info("Think length: %d tokens", think_length)
        logger_main.info("Probes: %d", len(results))
        logger_main.info("Duration: %.1fs (%.2fs/probe)", elapsed, elapsed / len(results) if results else 0)
        logger_main.info("Overall accuracy: %.1f%%", 100.0 * overall_accuracy)
        logger_main.info(
            "Accuracy by thirds: first=%.1f%%, middle=%.1f%%, last=%.1f%%",
            100.0 * first_third_acc,
            100.0 * middle_third_acc,
            100.0 * last_third_acc,
        )

        disagreements = (~df["agreement"]).sum()
        if disagreements > 0:
            logger_main.info("Token/string answer disagreements: %d", disagreements)

        no_answers = df["no_answer"].sum()
        if no_answers > 0:
            logger_main.info("No-answer cases: %d", no_answers)

        processed += 1

        # ---------------------------------------------------------------
        # 9. Cleanup between traces (KV cache + CUDA memory)
        # ---------------------------------------------------------------
        del cache
        torch.cuda.empty_cache()
        logger_main.info("Cleaned up CUDA memory between traces")

    # -------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------
    total_elapsed = time.time() - total_start
    logger_main.info("=== Multi-Trace Summary ===")
    logger_main.info("Total traces: %d", len(traces))
    logger_main.info("Processed: %d", processed)
    logger_main.info("Skipped (resume): %d", skipped)
    logger_main.info("Total time: %.1fs", total_elapsed)


if __name__ == "__main__":
    main()
