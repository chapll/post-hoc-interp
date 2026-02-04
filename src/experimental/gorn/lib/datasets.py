"""Dataset loaders for binary verification format.

Loads GSM-Symbolic, FOLIO, and StrategyQA from HuggingFace, converts each
problem into a verification prompt with balanced Yes/No ground truth labels.

Each loader returns a list of dicts with a uniform schema:
    problem_id: str           -- unique ID: "{dataset}_{index}"
    question: str             -- original question text
    ground_truth: str         -- "Yes" or "No" (probe label, matches prompt framing)
    dataset_name: str         -- "gsm_symbolic", "folio", "strategyqa"
    original_answer: str      -- original answer from dataset
    verification_prompt: str  -- full prompt for model input
    presented_answer: str     -- what was presented for verification
"""

import logging
import random

from datasets import concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


def _balance_and_sample(
    items: list[dict],
    n: int,
    rng: random.Random,
    dataset_name: str,
) -> list[dict]:
    """Balance items to minority class count, cap at n/2 per class, shuffle."""
    yes_items = [x for x in items if x["ground_truth"] == "Yes"]
    no_items = [x for x in items if x["ground_truth"] == "No"]

    minority_count = min(len(yes_items), len(no_items))
    per_class = min(minority_count, n // 2)

    if per_class < n // 2:
        logger.warning(
            "%s: requested %d per class but only %d available (minority class). Returning %d total items.",
            dataset_name,
            n // 2,
            per_class,
            per_class * 2,
        )

    rng.shuffle(yes_items)
    rng.shuffle(no_items)

    balanced = yes_items[:per_class] + no_items[:per_class]
    rng.shuffle(balanced)
    return balanced


def load_gsm_symbolic(n: int = 300, seed: int = 137) -> list[dict]:
    """Load GSM-Symbolic in binary verification format with plausible distractors.

    For each problem, creates two items:
    - Yes item: "Is the answer {correct}?" with ground_truth="Yes"
    - No item: "Is the answer {distractor}?" with ground_truth="No"

    Distractors are small perturbations of the correct answer to ensure
    plausible wrong answers that require actual computation to reject.

    Args:
        n: Target number of problems (balanced n/2 Yes + n/2 No).
        seed: Random seed for reproducibility.

    Returns:
        List of problem dicts in uniform schema.
    """
    rng = random.Random(seed)
    ds = load_dataset("apple/GSM-Symbolic", "main", split="test")
    logger.info("GSM-Symbolic: loaded %d rows", len(ds))

    items = []
    for idx, row in enumerate(ds):
        question = row["question"]
        answer_text = row["answer"]

        # Parse correct answer from after ####
        parts = answer_text.split("####")
        if len(parts) < 2:
            logger.warning("GSM-Symbolic row %d: no #### separator in answer, skipping", idx)
            continue
        raw_answer = parts[-1].strip()

        try:
            correct_value = int(raw_answer)
        except ValueError:
            try:
                correct_value = float(raw_answer)
            except ValueError:
                logger.warning(
                    "GSM-Symbolic row %d: cannot parse answer '%s', skipping",
                    idx,
                    raw_answer,
                )
                continue

        # Generate distractor: small perturbation of correct answer
        distractor = _generate_distractor(correct_value, rng)
        correct_str = str(correct_value)
        distractor_str = str(distractor)

        # Yes item: correct answer presented
        yes_prompt = f"{question}\n\nIs the answer {correct_str} (Yes or No)? Please think step-by-step to verify whether the answer is correct."
        items.append(
            {
                "problem_id": f"gsm_symbolic_{idx}_true",
                "question": question,
                "ground_truth": "Yes",
                "dataset_name": "gsm_symbolic",
                "original_answer": correct_str,
                "verification_prompt": yes_prompt,
                "presented_answer": correct_str,
            }
        )

        # No item: distractor presented
        no_prompt = f"{question}\n\nIs the answer {distractor_str} (Yes or No)? Please think step-by-step to verify whether the answer is correct."
        items.append(
            {
                "problem_id": f"gsm_symbolic_{idx}_false",
                "question": question,
                "ground_truth": "No",
                "dataset_name": "gsm_symbolic",
                "original_answer": correct_str,
                "verification_prompt": no_prompt,
                "presented_answer": distractor_str,
            }
        )

    logger.info(
        "GSM-Symbolic: generated %d verification items from %d rows",
        len(items),
        len(ds),
    )

    return _balance_and_sample(items, n, rng, "gsm_symbolic")


def _generate_distractor(correct: int | float, rng: random.Random) -> int | float:
    """Generate a plausible numerical distractor close to the correct answer.

    For integers: offset by 1-10% of value (minimum offset 1-2).
    For zero: use +/-1 or +/-2.
    Always ensures distractor != correct.
    """
    if correct == 0:
        return rng.choice([-2, -1, 1, 2])

    abs_val = abs(correct)

    # Offset range: 1-10% of the value, minimum of 1
    min_offset = max(1, int(abs_val * 0.01))
    max_offset = max(2, int(abs_val * 0.10))

    offset = rng.randint(min_offset, max_offset)
    direction = rng.choice([-1, 1])
    distractor = correct + direction * offset

    # Safety: ensure distractor is different from correct
    if distractor == correct:
        distractor = correct + offset

    # Preserve type
    if isinstance(correct, int):
        distractor = int(distractor)

    return distractor


def load_folio(n: int = 300, seed: int = 137) -> list[dict]:
    """Load FOLIO in binary verification format, filtering Uncertain labels.

    Combines train and validation splits. Filters out items with label
    "Uncertain". Maps "True" -> ground_truth="Yes" (conclusion is valid)
    and "False" -> ground_truth="No" (conclusion is not valid).

    Balances to minority class count, capped at n/2 per class.

    Args:
        n: Target number of problems (balanced).
        seed: Random seed for reproducibility.

    Returns:
        List of problem dicts in uniform schema.
    """
    rng = random.Random(seed)

    ds_train = load_dataset("tasksource/folio", split="train")
    ds_val = load_dataset("tasksource/folio", split="validation")
    ds = concatenate_datasets([ds_train, ds_val])
    total_rows = len(ds)
    logger.info("FOLIO: loaded %d rows (train=%d, val=%d)", total_rows, len(ds_train), len(ds_val))

    # Filter out Uncertain
    ds_filtered = [row for row in ds if row["label"] != "Uncertain"]
    after_filter = len(ds_filtered)
    logger.info(
        "FOLIO: after filtering Uncertain: %d rows (removed %d)",
        after_filter,
        total_rows - after_filter,
    )

    items = []
    for idx, row in enumerate(ds_filtered):
        premises = row["premises"]
        conclusion = row["conclusion"]
        label = row["label"]

        if label == "True":
            ground_truth = "Yes"
        elif label == "False":
            ground_truth = "No"
        else:
            # Should not happen after filtering, but be safe
            continue

        prompt = (
            f"Given the following premises:\n"
            f"{premises}\n\n"
            f'Is the following conclusion valid: "{conclusion}" (Yes or No)?\n'
            f"Please think step-by-step to verify whether this conclusion follows from the premises."
        )

        items.append(
            {
                "problem_id": f"folio_{idx}",
                "question": conclusion,
                "ground_truth": ground_truth,
                "dataset_name": "folio",
                "original_answer": label,
                "verification_prompt": prompt,
                "presented_answer": conclusion,
            }
        )

    balanced = _balance_and_sample(items, n, rng, "folio")
    logger.info(
        "FOLIO: total=%d, after_filter=%d, after_balance=%d",
        total_rows,
        after_filter,
        len(balanced),
    )
    return balanced


def load_strategyqa(n: int = 300, seed: int = 137) -> list[dict]:
    """Load StrategyQA in binary verification format with balanced labels.

    Uses the train split (1603 rows). Maps bool answer to Yes/No ground truth.
    Balances to minority class count, capped at n/2 per class.

    Args:
        n: Target number of problems (balanced).
        seed: Random seed for reproducibility.

    Returns:
        List of problem dicts in uniform schema.
    """
    rng = random.Random(seed)

    ds = load_dataset("ChilleD/StrategyQA", split="train")
    logger.info("StrategyQA: loaded %d rows", len(ds))

    items = []
    for idx, row in enumerate(ds):
        question = row["question"]
        answer = row["answer"]  # bool: True=yes, False=no

        ground_truth = "Yes" if answer else "No"
        presented = "yes" if answer else "no"

        prompt = f"Answer the following question (Yes or No): {question}\nPlease think step-by-step to verify your answer."

        items.append(
            {
                "problem_id": f"strategyqa_{idx}",
                "question": question,
                "ground_truth": ground_truth,
                "dataset_name": "strategyqa",
                "original_answer": str(answer),
                "verification_prompt": prompt,
                "presented_answer": presented,
            }
        )

    return _balance_and_sample(items, n, rng, "strategyqa")


def load_dataset_by_name(name: str, n: int = 300, seed: int = 137) -> list[dict]:
    """Dispatch to the appropriate dataset loader by name.

    Args:
        name: One of "gsm_symbolic", "folio", "strategyqa".
        n: Target number of problems.
        seed: Random seed for reproducibility.

    Returns:
        List of problem dicts in uniform schema.

    Raises:
        ValueError: If name is not a recognized dataset.
    """
    loaders = {
        "gsm_symbolic": load_gsm_symbolic,
        "folio": load_folio,
        "strategyqa": load_strategyqa,
    }

    if name not in loaders:
        raise ValueError(f"Unknown dataset: '{name}'. Available: {sorted(loaders.keys())}")

    return loaders[name](n=n, seed=seed)
