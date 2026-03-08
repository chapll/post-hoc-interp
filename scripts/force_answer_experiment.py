"""Force-answer transition experiment.

Loads validated traces, truncates thinking at evenly-spaced probe positions,
force-closes with </think> newline, generates a short continuation, and records
whether the model produces the correct answer (Mu or Nu) along with logit
probabilities and correct-token rank. Results are saved as a parquet file.

Usage:
    python scripts/force_answer_experiment.py --config configs/force_answer.yaml
"""

import copy
import json
import logging
import os
import random
import time

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import DynamicCache

from src.config import load_config
from src.model import load_model
from src.multiprocess import run_parallel, split_work
from src.utils import set_seed, setup_logging

# ---------------------------------------------------------------------------
# Token constants (verified: Mu and Nu are each single tokens)
# ---------------------------------------------------------------------------
MU_TOKEN_ID = 39120              # "Mu" (single token)
NU_TOKEN_ID = 60335              # "Nu" (single token)
CLOSE_TOKEN_IDS = [151668, 198]  # </think> + \n

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------

def load_traces_for_experiment(traces_dir, cfg):
    """Load and select traces for the force-answer experiment.

    Walks valid/{dataset}/{problem_id}/ directories, loads trace_00.json from
    each, filters for single-think-block traces, determines correctness, and
    selects a balanced set of correct/incorrect traces across datasets.

    Args:
        traces_dir: Root traces directory (contains valid/ subdirectory).
        cfg: OmegaConf config with force_answer section.

    Returns:
        List of dicts with keys: trace_path, token_ids, metadata,
        is_correct, dataset.
    """
    fa = cfg.force_answer
    trace_index = fa.trace_index
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
            ds, len(correct_by_ds[ds]), len(incorrect_by_ds[ds]),
        )
    if skipped > 0:
        logger.info("Skipped %d traces (no think or multiple think blocks)", skipped)

    # Select traces proportionally across datasets
    rng = random.Random(fa.selection_seed)
    n_correct_target = fa.n_correct
    n_incorrect_target = fa.n_incorrect

    total_correct = sum(len(v) for v in correct_by_ds.values())
    total_incorrect = sum(len(v) for v in incorrect_by_ds.values())

    selected = []

    for ds in datasets:
        # Correct allocation
        if total_correct > 0:
            ds_n_correct = round(n_correct_target * len(correct_by_ds[ds]) / total_correct)
        else:
            ds_n_correct = 0
        ds_n_correct = min(ds_n_correct, len(correct_by_ds[ds]))

        # Incorrect allocation
        if total_incorrect > 0:
            ds_n_incorrect = round(n_incorrect_target * len(incorrect_by_ds[ds]) / total_incorrect)
        else:
            ds_n_incorrect = 0
        ds_n_incorrect = min(ds_n_incorrect, len(incorrect_by_ds[ds]))

        # Sample
        correct_sample = rng.sample(correct_by_ds[ds], ds_n_correct) if ds_n_correct > 0 else []
        incorrect_sample = rng.sample(incorrect_by_ds[ds], ds_n_incorrect) if ds_n_incorrect > 0 else []

        selected.extend(correct_sample)
        selected.extend(incorrect_sample)

        logger.info(
            "Selected from %s: %d correct, %d incorrect",
            ds, len(correct_sample), len(incorrect_sample),
        )

    logger.info(
        "Total selected: %d traces (%d correct, %d incorrect)",
        len(selected),
        sum(1 for t in selected if t["is_correct"]),
        sum(1 for t in selected if not t["is_correct"]),
    )

    return selected


# ---------------------------------------------------------------------------
# Probe position computation
# ---------------------------------------------------------------------------

def compute_probe_positions(think_length, num_probes=20, min_spacing=5):
    """Compute evenly-spaced probe positions within a thinking block.

    Positions are 0-indexed relative to the start of thinking content
    (the token after <think>).

    Args:
        think_length: Number of content tokens in the thinking block.
        num_probes: Desired number of probe positions.
        min_spacing: Minimum token spacing between probes.

    Returns:
        Sorted list of unique integer probe positions.
    """
    if think_length <= 0:
        return []

    max_probes = max(1, think_length // min_spacing)
    actual = min(num_probes, max_probes)

    if actual <= 1:
        return [think_length // 2]

    positions = np.linspace(0, think_length - 1, actual, dtype=int)
    # Remove duplicates and sort
    positions = sorted(set(positions.tolist()))
    return positions


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(generated_ids, generated_text, expected_token):
    """Extract the answer from generated tokens using dual matching.

    Token-level matching scans for the first Mu or Nu token.
    String-level matching checks for "mu" or "nu" in decoded text.
    Disagreements between methods are flagged.

    Args:
        generated_ids: List of generated token IDs.
        generated_text: Decoded text of generated tokens.
        expected_token: "Mu" or "Nu" -- the correct answer.

    Returns:
        Dict with keys: answer, correct, no_answer, token_answer,
        string_answer, agreement.
    """
    # Token-level matching (both Mu and Nu are single tokens)
    token_answer = None
    for tid in generated_ids:
        if tid == MU_TOKEN_ID:
            token_answer = "Mu"
            break
        if tid == NU_TOKEN_ID:
            token_answer = "Nu"
            break

    # String-level matching
    string_answer = None
    text_lower = generated_text.lower()
    mu_pos = text_lower.find("mu")
    nu_pos = text_lower.find("nu")

    if mu_pos >= 0 and nu_pos >= 0:
        # Both found -- take whichever appears first
        string_answer = "Mu" if mu_pos < nu_pos else "Nu"
    elif mu_pos >= 0:
        string_answer = "Mu"
    elif nu_pos >= 0:
        string_answer = "Nu"

    # Determine final answer and flags
    if token_answer is None and string_answer is None:
        return {
            "answer": None,
            "correct": False,
            "no_answer": True,
            "token_answer": None,
            "string_answer": None,
            "agreement": True,
        }

    # Use token_answer as primary if available, else string_answer
    answer = token_answer if token_answer is not None else string_answer
    agreement = (token_answer == string_answer) if (token_answer is not None and string_answer is not None) else True

    correct = (answer.lower() == expected_token.lower()) if answer else False

    return {
        "answer": answer,
        "correct": correct,
        "no_answer": False,
        "token_answer": token_answer,
        "string_answer": string_answer,
        "agreement": agreement,
    }


# ---------------------------------------------------------------------------
# Logit probability extraction
# ---------------------------------------------------------------------------

def extract_mu_nu_probs(first_logits, correct_token_id):
    """Extract Mu/Nu probabilities and correct-token rank from logits.

    Applies float32 softmax to the first generated token's logits, then
    extracts probabilities for the Mu and Nu tokens.

    Args:
        first_logits: Logits tensor of shape [vocab_size] for the first
            generated token.
        correct_token_id: Token ID of the correct answer
            (MU_TOKEN_ID or NU_TOKEN_ID).

    Returns:
        Dict with keys: mu_prob (float), nu_prob (float),
        correct_token_rank (int, 0-indexed).
    """
    probs = torch.softmax(first_logits.float(), dim=-1)
    mu_prob = probs[MU_TOKEN_ID].item()
    nu_prob = probs[NU_TOKEN_ID].item()

    # Compute rank of correct token (0-indexed, 0 = highest probability)
    sorted_indices = torch.argsort(probs, descending=True)
    rank_positions = torch.argsort(sorted_indices)
    correct_token_rank = rank_positions[correct_token_id].item()

    return {
        "mu_prob": mu_prob,
        "nu_prob": nu_prob,
        "correct_token_rank": correct_token_rank,
    }


# ---------------------------------------------------------------------------
# Per-trace probing
# ---------------------------------------------------------------------------

def probe_trace(raw_model, tokenizer, trace_data, cfg, device):
    """Probe a single trace at multiple positions using KV-cache truncation.

    For each probe position within the thinking block, truncates the
    thinking content, appends the force-close sequence (</think> newline),
    and generates a short continuation. Records the answer, logit
    probabilities, and correct-token rank.

    Args:
        raw_model: Raw HuggingFace model (model._model).
        tokenizer: Model tokenizer.
        trace_data: Dict from load_traces_for_experiment with token_ids,
            metadata, is_correct, dataset.
        cfg: OmegaConf config.
        device: Torch device for model inputs.

    Returns:
        List of result row dicts, one per probe position.
    """
    meta = trace_data["metadata"]
    token_ids = trace_data["token_ids"]
    fa = cfg.force_answer

    # Extract think boundaries
    think_start, think_end = meta["think_boundaries"][0]
    # Content tokens are between think_start+1 and think_end-1 (inclusive)
    think_length = think_end - think_start - 1

    if think_length <= 0:
        logger.warning(
            "Trace %s has zero-length thinking block, skipping",
            meta["problem_id"],
        )
        return []

    # Probe positions (0-indexed relative to think content start)
    probe_positions = compute_probe_positions(
        think_length, fa.num_probes, fa.min_probe_spacing,
    )

    # Determine expected answer
    ground_truth = meta["ground_truth"]
    valid_token = meta["valid_token"]
    invalid_token = meta["invalid_token"]
    expected_token = valid_token if ground_truth == "Yes" else invalid_token
    correct_token_id = MU_TOKEN_ID if expected_token == "Mu" else NU_TOKEN_ID

    # Prefix: everything up to and including <think>
    prefix_ids = token_ids[:think_start + 1]
    # Think content tokens
    think_content_ids = token_ids[think_start + 1 : think_start + 1 + think_length]
    # Full input for cache prefill
    full_input_ids = prefix_ids + think_content_ids

    results = []
    use_cache_approach = True

    try:
        # Prefill KV cache with full input
        full_tensor = torch.tensor([full_input_ids], dtype=torch.long, device=device)
        cache = DynamicCache()
        with torch.no_grad():
            raw_model(input_ids=full_tensor, past_key_values=cache, use_cache=True)

        # Process probes in DESCENDING order (rightmost first for efficient cropping)
        for pos in sorted(probe_positions, reverse=True):
            cache_len = len(prefix_ids) + pos
            cache.crop(cache_len)

            # Build input: prefix + think_content[:pos] + close_tokens
            input_for_generate = full_input_ids[:cache_len] + CLOSE_TOKEN_IDS
            input_tensor = torch.tensor([input_for_generate], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_tensor)

            with torch.no_grad():
                gen_output = raw_model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    past_key_values=copy.deepcopy(cache),
                    max_new_tokens=fa.max_new_tokens,
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
                # No tokens generated -- create zero logits
                logger.warning("No logits returned for probe pos %d", pos)
                first_logits = torch.zeros(raw_model.config.vocab_size, device=device)

            answer_info = extract_answer(generated_ids, generated_text, expected_token)
            prob_info = extract_mu_nu_probs(first_logits, correct_token_id)

            normalized_position = pos / (think_length - 1) if think_length > 1 else 0.5
            correct_token_prob = prob_info["mu_prob"] if expected_token == "Mu" else prob_info["nu_prob"]

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

    except Exception as e:
        if use_cache_approach:
            logger.warning(
                "KV cache approach failed for %s: %s. Falling back to simple approach.",
                meta["problem_id"], e,
            )
            use_cache_approach = False
        else:
            raise

    # Fallback: simple approach without cache reuse
    if not use_cache_approach:
        results = []
        for pos in probe_positions:
            cache_len = len(prefix_ids) + pos
            input_for_generate = full_input_ids[:cache_len] + CLOSE_TOKEN_IDS
            input_tensor = torch.tensor([input_for_generate], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_tensor)

            with torch.no_grad():
                gen_output = raw_model.generate(
                    input_ids=input_tensor,
                    attention_mask=attention_mask,
                    max_new_tokens=fa.max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_logits=True,
                )

            generated_ids = gen_output.sequences[0, len(input_for_generate):].tolist()
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

            if len(gen_output.logits) > 0:
                first_logits = gen_output.logits[0][0]
            else:
                logger.warning("No logits returned for probe pos %d (fallback)", pos)
                first_logits = torch.zeros(raw_model.config.vocab_size, device=device)

            answer_info = extract_answer(generated_ids, generated_text, expected_token)
            prob_info = extract_mu_nu_probs(first_logits, correct_token_id)

            normalized_position = pos / (think_length - 1) if think_length > 1 else 0.5
            correct_token_prob = prob_info["mu_prob"] if expected_token == "Mu" else prob_info["nu_prob"]

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

    return results


# ---------------------------------------------------------------------------
# Worker function for multi-GPU execution
# ---------------------------------------------------------------------------

def _worker(rank, gpu_id, items, cfg):
    """Worker function: load model, probe assigned traces, write shard.

    Args:
        rank: Worker rank (0-indexed).
        gpu_id: CUDA device index.
        items: List of trace dicts assigned to this worker.
        cfg: OmegaConf config.
    """
    worker_logger = setup_logging(f"force_answer.worker{rank}")
    worker_logger.info("Worker %d starting on GPU %d with %d traces", rank, gpu_id, len(items))

    if not items:
        worker_logger.info("Worker %d has no items, exiting", rank)
        return

    model, tokenizer = load_model(gpu_id=gpu_id)
    raw_model = model._model
    device = next(raw_model.parameters()).device
    worker_logger.info("Worker %d model loaded on %s", rank, device)

    all_results = []
    for trace_data in tqdm(items, desc=f"GPU {gpu_id}", position=rank):
        rows = probe_trace(raw_model, tokenizer, trace_data, cfg, device)
        all_results.extend(rows)
        torch.cuda.empty_cache()

    # Write shard to temporary parquet
    if all_results:
        output_dir = cfg.paths.output_dir
        os.makedirs(output_dir, exist_ok=True)
        shard_path = os.path.join(output_dir, f"results_shard_{rank}.parquet")
        df = pd.DataFrame(all_results)
        df.to_parquet(shard_path, index=False)
        worker_logger.info("Worker %d wrote %d rows to %s", rank, len(df), shard_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "trace_id", "dataset", "problem_id", "probe_position",
    "normalized_position", "correct", "no_answer", "mu_prob",
    "nu_prob", "correct_token_prob", "correct_token_rank",
    "ground_truth", "is_correct_trace", "is_fresh", "prompt_config",
    "valid_token", "invalid_token", "token_answer", "string_answer",
    "agreement", "think_length",
]


def main():
    """Run the force-answer transition experiment."""
    logger_main = setup_logging("force_answer")
    cfg = load_config()
    set_seed(cfg.seed)

    gpu_ids = list(cfg.gpus)
    output_dir = cfg.paths.output_dir
    traces_dir = cfg.paths.traces_dir

    logger_main.info("Traces directory: %s", traces_dir)
    logger_main.info("Output directory: %s", output_dir)
    logger_main.info("GPUs: %s", gpu_ids)

    # Load and select traces (done once in main process)
    traces = load_traces_for_experiment(traces_dir, cfg)
    n_correct = sum(1 for t in traces if t["is_correct"])
    n_incorrect = len(traces) - n_correct
    logger_main.info(
        "Loaded %d traces: %d correct, %d incorrect",
        len(traces), n_correct, n_incorrect,
    )

    # Per-dataset summary
    ds_counts = {}
    for t in traces:
        ds = t["dataset"]
        if ds not in ds_counts:
            ds_counts[ds] = {"correct": 0, "incorrect": 0}
        if t["is_correct"]:
            ds_counts[ds]["correct"] += 1
        else:
            ds_counts[ds]["incorrect"] += 1
    for ds, counts in sorted(ds_counts.items()):
        logger_main.info(
            "  %s: %d correct, %d incorrect",
            ds, counts["correct"], counts["incorrect"],
        )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run probing across GPUs
    start_time = time.time()
    run_parallel(_worker, traces, gpu_ids, cfg)
    elapsed = time.time() - start_time
    logger_main.info("Probing completed in %.1fs across %d GPU(s)", elapsed, len(gpu_ids))

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

    # Sort by trace_id and probe_position for deterministic output
    df = df.sort_values(["trace_id", "probe_position"]).reset_index(drop=True)

    # Save final parquet
    output_path = os.path.join(output_dir, "results.parquet")
    df.to_parquet(output_path, index=False)
    logger_main.info("Saved %d rows to %s", len(df), output_path)

    # Summary statistics
    logger_main.info("--- Summary ---")
    logger_main.info("Total probe measurements: %d", len(df))
    logger_main.info("Unique traces: %d", df["trace_id"].nunique())
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
            "Mean correct_token_rank: %.1f",
            df["correct_token_rank"].mean(),
        )
        disagreements = (~df["agreement"]).sum()
        if disagreements > 0:
            logger_main.info(
                "Token/string answer disagreements: %d", disagreements,
            )


if __name__ == "__main__":
    main()
