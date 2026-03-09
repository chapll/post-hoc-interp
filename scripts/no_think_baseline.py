"""No-think baseline evaluation.

Evaluates Qwen3-8B on all benchmark problems with thinking disabled: the
think block is force-closed at position 0 (immediately after <think>), so
the model must answer without any chain-of-thought reasoning.

Problems answered correctly without thinking are flagged as trivial or
memorized and saved to a discard list (trivial_problems.json) for exclusion
from downstream probing experiments.

Usage:
    python scripts/no_think_baseline.py --config configs/no_think_baseline.yaml
"""

import json
import logging
import os
import time

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import torch
from tqdm import tqdm

from src.config import load_config
from src.datasets import load_dataset_by_name
from src.model import load_model
from src.multiprocess import run_parallel
from src.prompts import build_verification_prompt
from src.utils import set_seed, setup_logging

# ---------------------------------------------------------------------------
# Token constants (verified: Mu and Nu are each single tokens)
# ---------------------------------------------------------------------------
MU_TOKEN_ID = 39120              # "Mu" (bare, single token -- after ** prefix)
NU_TOKEN_ID = 60335              # "Nu" (bare, single token -- after ** prefix)
CLOSE_TOKEN_IDS = [151668, 271, 16141, 25, 3070]  # </think>\n\nAnswer: **

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Answer extraction (inlined from force_answer_experiment.py for simplicity)
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


def extract_mu_nu_probs(first_logits, correct_token_id):
    """Extract Mu/Nu probabilities and correct-token rank from logits.

    Args:
        first_logits: Logits tensor of shape [vocab_size] for the first
            generated token.
        correct_token_id: Token ID of the correct answer
            (MU_TOKEN_ID or NU_TOKEN_ID).

    Returns:
        Dict with keys: mu_prob, nu_prob, correct_token_prob,
        correct_token_rank.
    """
    probs = torch.softmax(first_logits.float(), dim=-1)
    mu_prob = probs[MU_TOKEN_ID].item()
    nu_prob = probs[NU_TOKEN_ID].item()

    # Compute rank of correct token (0-indexed, 0 = highest probability)
    sorted_indices = torch.argsort(probs, descending=True)
    rank_positions = torch.argsort(sorted_indices)
    correct_token_rank = rank_positions[correct_token_id].item()

    correct_token_prob = probs[correct_token_id].item()

    return {
        "mu_prob": mu_prob,
        "nu_prob": nu_prob,
        "correct_token_prob": correct_token_prob,
        "correct_token_rank": correct_token_rank,
    }


# ---------------------------------------------------------------------------
# Worker function for multi-GPU execution
# ---------------------------------------------------------------------------

def _worker(rank, gpu_id, items, cfg):
    """Worker function: load model, evaluate assigned problems, write shard.

    Args:
        rank: Worker rank (0-indexed).
        gpu_id: CUDA device index.
        items: List of problem dicts assigned to this worker.
        cfg: OmegaConf config.
    """
    worker_logger = setup_logging(f"no_think_baseline.worker{rank}")
    worker_logger.info("Worker %d starting on GPU %d with %d problems", rank, gpu_id, len(items))

    if not items:
        worker_logger.info("Worker %d has no items, exiting", rank)
        return

    model, tokenizer = load_model(gpu_id=gpu_id)
    raw_model = model._model
    device = next(raw_model.parameters()).device
    worker_logger.info("Worker %d model loaded on %s", rank, device)

    all_results = []
    close_ids = torch.tensor([CLOSE_TOKEN_IDS], dtype=torch.long, device=device)

    for problem in tqdm(items, desc=f"GPU {gpu_id}", position=rank):
        # Build verification prompt with swapped-label protocol, config A only
        prompt_info = build_verification_prompt(problem, label_protocol="swapped", config="A")
        label_mapping = prompt_info["label_mapping"]

        # Determine expected answer
        ground_truth = problem["ground_truth"]
        if ground_truth == "Yes":
            expected_token = label_mapping["valid"]   # "Mu" for config A
        else:
            expected_token = label_mapping["invalid"]  # "Nu" for config A

        correct_token_id = MU_TOKEN_ID if expected_token == "Mu" else NU_TOKEN_ID

        # Tokenize with chat template + enable_thinking=True
        messages = [{"role": "user", "content": prompt_info["verification_prompt"]}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

        # Force-close thinking at position 0: append </think>\n right after input
        # The generation prompt ends with <think>\n, so appending close_ids
        # immediately closes the think block before any thinking content.
        input_with_close = torch.cat([input_ids, close_ids], dim=1)

        # Generate up to max_new_tokens with greedy decoding
        with torch.no_grad():
            gen_output = raw_model.generate(
                input_ids=input_with_close,
                attention_mask=torch.ones_like(input_with_close),
                max_new_tokens=cfg.no_think_baseline.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_logits=True,
            )

        generated_ids = gen_output.sequences[0, input_with_close.shape[1]:].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)

        # Extract answer using dual matching
        answer_info = extract_answer(generated_ids, generated_text, expected_token)

        # Extract probabilities from first token logits
        if len(gen_output.logits) > 0:
            first_logits = gen_output.logits[0][0]  # shape [vocab_size]
        else:
            worker_logger.warning("No logits for problem %s", problem["problem_id"])
            first_logits = torch.zeros(raw_model.config.vocab_size, device=device)

        prob_info = extract_mu_nu_probs(first_logits, correct_token_id)

        row = {
            "problem_id": problem["problem_id"],
            "dataset": problem["dataset_name"],
            "ground_truth": ground_truth,
            "expected_token": expected_token,
            "correct": answer_info["correct"],
            "answer": answer_info["answer"],
            "no_answer": answer_info["no_answer"],
            "mu_prob": prob_info["mu_prob"],
            "nu_prob": prob_info["nu_prob"],
            "correct_token_prob": prob_info["correct_token_prob"],
            "correct_token_rank": prob_info["correct_token_rank"],
            "token_answer": answer_info["token_answer"],
            "string_answer": answer_info["string_answer"],
            "agreement": answer_info["agreement"],
        }
        all_results.append(row)

        torch.cuda.empty_cache()

    # Write shard to parquet
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

def main():
    """Run the no-think baseline evaluation."""
    logger_main = setup_logging("no_think_baseline")
    cfg = load_config()
    set_seed(cfg.seed)

    gpu_ids = list(cfg.gpus)
    output_dir = cfg.paths.output_dir
    n_per_dataset = cfg.no_think_baseline.problems_per_dataset

    logger_main.info("Output directory: %s", output_dir)
    logger_main.info("GPUs: %s", gpu_ids)
    logger_main.info("Problems per dataset: %d", n_per_dataset)

    # Load all 3 datasets
    dataset_names = ["gsm_symbolic", "folio", "strategyqa"]
    all_items = []
    for ds_name in dataset_names:
        problems = load_dataset_by_name(ds_name, n=n_per_dataset, seed=cfg.seed)
        logger_main.info("Loaded %d problems from %s", len(problems), ds_name)
        all_items.extend(problems)

    logger_main.info("Total problems: %d", len(all_items))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run evaluation across GPUs
    start_time = time.time()
    run_parallel(_worker, all_items, gpu_ids, cfg)
    elapsed = time.time() - start_time
    logger_main.info("Evaluation completed in %.1fs across %d GPU(s)", elapsed, len(gpu_ids))

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

    # Sort by problem_id for deterministic output
    df = df.sort_values("problem_id").reset_index(drop=True)

    # Save final parquet
    output_path = os.path.join(output_dir, "results.parquet")
    df.to_parquet(output_path, index=False)
    logger_main.info("Saved %d rows to %s", len(df), output_path)

    # --- Post-merge analysis ---
    logger_main.info("--- Per-Dataset Summary ---")

    trivial_by_dataset = {}
    for ds_name in dataset_names:
        ds_df = df[df["dataset"] == ds_name]
        total = len(ds_df)
        correct_count = ds_df["correct"].sum()
        pct = 100.0 * correct_count / total if total > 0 else 0.0

        logger_main.info(
            "  %s: total=%d, correct_without_thinking=%d, pct_trivial=%.1f%%",
            ds_name, total, correct_count, pct,
        )

        trivial_ids = ds_df[ds_df["correct"]]["problem_id"].tolist()
        trivial_by_dataset[ds_name] = trivial_ids

    # Overall summary
    total = len(df)
    total_correct = df["correct"].sum()
    total_pct = 100.0 * total_correct / total if total > 0 else 0.0
    logger_main.info(
        "Overall: total=%d, correct_without_thinking=%d, pct_trivial=%.1f%%",
        total, total_correct, total_pct,
    )

    no_answer_count = df["no_answer"].sum()
    if no_answer_count > 0:
        logger_main.info("No-answer cases: %d (%.1f%%)", no_answer_count, 100.0 * no_answer_count / total)

    disagreements = (~df["agreement"]).sum()
    if disagreements > 0:
        logger_main.info("Token/string answer disagreements: %d", disagreements)

    # Build and save discard list
    all_trivial_ids = df[df["correct"]]["problem_id"].tolist()
    discard_data = {
        "description": "Problems correctly answered without thinking (trivial/memorized)",
        "total_problems": int(total),
        "trivial_count": int(total_correct),
        "trivial_pct": round(float(total_pct), 1),
        "by_dataset": trivial_by_dataset,
        "all_trivial_ids": all_trivial_ids,
    }

    discard_path = os.path.join(output_dir, "trivial_problems.json")
    with open(discard_path, "w") as f:
        json.dump(discard_data, f, indent=2)
    logger_main.info("Saved discard list to %s (%d trivial problems)", discard_path, len(all_trivial_ids))


if __name__ == "__main__":
    main()
