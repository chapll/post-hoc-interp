"""Dataset loaders for binary verification format.

Loads GSM-Symbolic, FOLIO, StrategyQA, and DeepMath-103K from HuggingFace,
converts each problem into base problem dicts with balanced Yes/No ground
truth labels.

Prompts are NOT built here. Use lib.prompts.build_verification_prompt() to
construct prompts with configurable label tokens (Mu/Nu for swapped-label
protocol, or Yes/No for backward compatibility).

Each loader returns a list of dicts with a uniform schema:
    problem_id: str           -- unique ID: "{dataset}_{index}"
    question: str             -- original question text
    ground_truth: str         -- "Yes" or "No" (abstract validity label)
    dataset_name: str         -- "gsm_symbolic", "folio", "strategyqa", "deepmath"
    original_answer: str      -- original answer from dataset
    presented_answer: str     -- what was presented for verification
    premises: str             -- (FOLIO only) formatted premises string
    difficulty: str           -- (DeepMath only) problem difficulty level
    topic: str                -- (DeepMath only) problem topic/category
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
        items.append(
            {
                "problem_id": f"gsm_symbolic_{idx}_true",
                "question": question,
                "ground_truth": "Yes",
                "dataset_name": "gsm_symbolic",
                "original_answer": correct_str,
                "presented_answer": correct_str,
            }
        )

        # No item: distractor presented
        items.append(
            {
                "problem_id": f"gsm_symbolic_{idx}_false",
                "question": question,
                "ground_truth": "No",
                "dataset_name": "gsm_symbolic",
                "original_answer": correct_str,
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

        items.append(
            {
                "problem_id": f"folio_{idx}",
                "question": conclusion,
                "ground_truth": ground_truth,
                "dataset_name": "folio",
                "original_answer": label,
                "presented_answer": conclusion,
                "premises": premises,
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

        items.append(
            {
                "problem_id": f"strategyqa_{idx}",
                "question": question,
                "ground_truth": ground_truth,
                "dataset_name": "strategyqa",
                "original_answer": str(answer),
                "presented_answer": presented,
            }
        )

    return _balance_and_sample(items, n, rng, "strategyqa")


def load_gsm_symbolic_contamination(
    n_per_type: int = 100,
    seed: int = 137,
    fresh_instance_id: int = 0,
) -> list[dict]:
    """Load GSM-Symbolic with both original and fresh instances for contamination analysis.

    For each of the 100 GSM-Symbolic templates, creates two problems:
    - A "fresh" problem using a generated variant (instance fresh_instance_id),
      which has different numerical values from the GSM8K training data.
    - An "original" problem using the original_question and original_answer fields,
      which contain the exact GSM8K text the model may have been trained on.

    Each problem is converted to binary verification format (Yes item with correct
    answer, No item with distractor), then balanced and sampled separately for
    fresh and original items.

    Args:
        n_per_type: Target number of problems per type (fresh/original).
            Each type will have n_per_type/2 Yes + n_per_type/2 No items.
        seed: Random seed for reproducibility.
        fresh_instance_id: Which generated instance to use as the "fresh" variant
            (0-49 available per template).

    Returns:
        Combined list of problem dicts with all standard fields plus:
        - is_fresh: bool indicating whether this is a fresh (True) or
          original/potentially-contaminated (False) instance.
    """
    rng = random.Random(seed)
    ds = load_dataset("apple/GSM-Symbolic", "main", split="test")
    logger.info("GSM-Symbolic contamination: loaded %d rows", len(ds))

    # Group rows by template id
    templates: dict[int, list] = {}
    for row in ds:
        templates.setdefault(row["id"], []).append(row)

    logger.info(
        "GSM-Symbolic contamination: %d templates, %d instances per template (avg)",
        len(templates),
        len(ds) // max(len(templates), 1),
    )

    fresh_items = []
    original_items = []

    for template_id in sorted(templates.keys()):
        rows = templates[template_id]

        # Find the fresh instance row
        fresh_rows = [r for r in rows if r["instance"] == fresh_instance_id]
        if not fresh_rows:
            logger.warning(
                "GSM-Symbolic contamination: template %d has no instance %d, skipping",
                template_id,
                fresh_instance_id,
            )
            continue
        fresh_row = fresh_rows[0]

        # --- Fresh problem (generated variant with different numbers) ---
        fresh_answer_text = fresh_row["answer"]
        fresh_parsed = _parse_gsm_answer(fresh_answer_text, f"fresh template {template_id}")
        if fresh_parsed is None:
            continue

        fresh_correct_str = str(fresh_parsed)
        fresh_distractor = _generate_distractor(fresh_parsed, rng)
        fresh_distractor_str = str(fresh_distractor)
        fresh_question = fresh_row["question"]
        fresh_pid = f"gsm_symbolic_{template_id}_fresh"

        # Yes item: correct answer presented
        fresh_items.append(
            {
                "problem_id": f"{fresh_pid}_true",
                "question": fresh_question,
                "ground_truth": "Yes",
                "dataset_name": "gsm_symbolic",
                "original_answer": fresh_correct_str,
                "presented_answer": fresh_correct_str,
                "is_fresh": True,
            }
        )

        # No item: distractor presented
        fresh_items.append(
            {
                "problem_id": f"{fresh_pid}_false",
                "question": fresh_question,
                "ground_truth": "No",
                "dataset_name": "gsm_symbolic",
                "original_answer": fresh_correct_str,
                "presented_answer": fresh_distractor_str,
                "is_fresh": True,
            }
        )

        # --- Original problem (GSM8K text, potentially contaminated) ---
        orig_question = fresh_row["original_question"]
        orig_answer_text = fresh_row["original_answer"]
        orig_parsed = _parse_gsm_answer(orig_answer_text, f"original template {template_id}")
        if orig_parsed is None:
            continue

        orig_correct_str = str(orig_parsed)
        orig_distractor = _generate_distractor(orig_parsed, rng)
        orig_distractor_str = str(orig_distractor)
        orig_pid = f"gsm_symbolic_{template_id}_original"

        # Yes item: correct answer presented
        original_items.append(
            {
                "problem_id": f"{orig_pid}_true",
                "question": orig_question,
                "ground_truth": "Yes",
                "dataset_name": "gsm_symbolic",
                "original_answer": orig_correct_str,
                "presented_answer": orig_correct_str,
                "is_fresh": False,
            }
        )

        # No item: distractor presented
        original_items.append(
            {
                "problem_id": f"{orig_pid}_false",
                "question": orig_question,
                "ground_truth": "No",
                "dataset_name": "gsm_symbolic",
                "original_answer": orig_correct_str,
                "presented_answer": orig_distractor_str,
                "is_fresh": False,
            }
        )

    logger.info(
        "GSM-Symbolic contamination: %d fresh items, %d original items (before balancing)",
        len(fresh_items),
        len(original_items),
    )

    # Balance and sample separately for fresh and original
    balanced_fresh = _balance_and_sample(fresh_items, n_per_type, rng, "gsm_symbolic_fresh")
    balanced_original = _balance_and_sample(
        original_items, n_per_type, rng, "gsm_symbolic_original"
    )

    combined = balanced_fresh + balanced_original
    rng.shuffle(combined)

    logger.info(
        "GSM-Symbolic contamination: returning %d items (%d fresh + %d original)",
        len(combined),
        len(balanced_fresh),
        len(balanced_original),
    )

    return combined


def _parse_gsm_answer(answer_text: str, context: str) -> int | float | None:
    """Parse the numerical answer from GSM-style '#### <number>' format.

    Args:
        answer_text: Full answer text containing #### separator.
        context: Description of the source for warning messages.

    Returns:
        Parsed int or float value, or None if parsing fails.
    """
    parts = answer_text.split("####")
    if len(parts) < 2:
        logger.warning("GSM-Symbolic %s: no #### separator in answer, skipping", context)
        return None

    raw_answer = parts[-1].strip()

    try:
        return int(raw_answer)
    except ValueError:
        try:
            return float(raw_answer)
        except ValueError:
            logger.warning(
                "GSM-Symbolic %s: cannot parse answer '%s', skipping",
                context,
                raw_answer,
            )
            return None


def load_deepmath(n: int = 300, seed: int = 137) -> list[dict]:
    """Load DeepMath-103K in binary verification format, filtering to numeric answers.

    For each problem with a numeric final_answer, creates two items:
    - Yes item: "Is the answer {correct}?" with ground_truth="Yes"
    - No item: "Is the answer {distractor}?" with ground_truth="No"

    Non-numeric answers (symbolic, LaTeX, etc.) are skipped.
    Distractors are generated via _generate_distractor().

    Args:
        n: Target number of problems (balanced n/2 Yes + n/2 No).
        seed: Random seed for reproducibility.

    Returns:
        List of problem dicts in uniform schema with extra difficulty/topic fields.
    """
    rng = random.Random(seed)
    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    logger.info("DeepMath: loaded %d rows", len(ds))

    items = []
    skipped = 0
    for idx, row in enumerate(ds):
        question = row["question"]
        raw_answer = row["final_answer"].strip()

        # Parse numeric answer: try int then float, skip non-numeric
        try:
            correct_value = int(raw_answer)
        except ValueError:
            try:
                correct_value = float(raw_answer)
            except ValueError:
                skipped += 1
                continue

        # Generate distractor: small perturbation of correct answer
        distractor = _generate_distractor(correct_value, rng)
        correct_str = str(correct_value)
        distractor_str = str(distractor)

        base = {
            "question": question,
            "dataset_name": "deepmath",
            "original_answer": correct_str,
            "difficulty": row.get("difficulty", ""),
            "topic": row.get("topic", ""),
        }

        # Yes item: correct answer presented
        items.append(
            {
                **base,
                "problem_id": f"deepmath_{idx}_true",
                "ground_truth": "Yes",
                "presented_answer": correct_str,
            }
        )

        # No item: distractor presented
        items.append(
            {
                **base,
                "problem_id": f"deepmath_{idx}_false",
                "ground_truth": "No",
                "presented_answer": distractor_str,
            }
        )

    logger.info(
        "DeepMath: generated %d verification items from %d rows (%d non-numeric skipped)",
        len(items),
        len(ds),
        skipped,
    )

    return _balance_and_sample(items, n, rng, "deepmath")


def load_deepmath_pool(
    pool: str,
    n: int = 300,
    seed: int = 137,
    binary: bool = True,
) -> list[dict]:
    """Load DeepMath problems from a specific non-overlapping pool.

    Pools use disjoint index ranges from the full DeepMath-103K dataset to
    guarantee zero leakage between direction extraction, sleeper training,
    evaluation, and transferability testing.

    Pool allocations (by HF dataset row index, after filtering to numeric):
        A (direction):    rows 0-2999      (seed_offset=1000)
        B (sleeper SFT):  rows 3000-5999   (seed_offset=2000)
        C (evaluation):   rows 6000-7999   (seed_offset=3000)
        D (binary xfer):  rows 8000-8999   (seed_offset=4000)
        E (nonbinary):    rows 9000-9999   (seed_offset=5000)

    Args:
        pool: One of "A", "B", "C", "D", "E".
        n: Target number of problems (balanced Yes/No for binary).
        seed: Base seed (combined with pool offset for reproducibility).
        binary: If True, create Mu/Nu verification items. If False, create
            freeform items (for non-binary transferability testing).

    Returns:
        List of problem dicts. Binary mode returns Yes/No verification pairs.
        Non-binary mode returns raw problems with ground_truth answer.
    """
    pool_ranges = {
        "A": (0, 3000),
        "B": (3000, 6000),
        "C": (6000, 8000),
        "D": (8000, 9000),
        "E": (9000, 10000),
    }
    pool_seed_offsets = {"A": 1000, "B": 2000, "C": 3000, "D": 4000, "E": 5000}

    if pool not in pool_ranges:
        raise ValueError(f"Unknown pool: '{pool}'. Available: {sorted(pool_ranges)}")

    start_idx, end_idx = pool_ranges[pool]
    rng = random.Random(seed + pool_seed_offsets[pool])

    ds = load_dataset("zwhe99/DeepMath-103K", split="train")
    logger.info("DeepMath pool %s: loaded %d rows, filtering indices %d-%d",
                pool, len(ds), start_idx, end_idx)

    # Filter to numeric answers within pool's index range
    numeric_rows = []
    for idx, row in enumerate(ds):
        raw_answer = row["final_answer"].strip()
        try:
            correct_value = int(raw_answer)
        except ValueError:
            try:
                correct_value = float(raw_answer)
            except ValueError:
                continue
        numeric_rows.append((idx, row, correct_value))

    # Select rows within this pool's range
    pool_rows = numeric_rows[start_idx:end_idx]
    logger.info("DeepMath pool %s: %d numeric rows in range", pool, len(pool_rows))

    if not binary:
        # Non-binary: return raw problems for freeform solving
        items = []
        for idx, row, correct_value in pool_rows:
            items.append({
                "problem_id": f"deepmath_{idx}_freeform",
                "question": row["question"],
                "ground_truth": str(correct_value),
                "dataset_name": "deepmath_freeform",
                "original_answer": str(correct_value),
                "presented_answer": str(correct_value),
                "difficulty": row.get("difficulty", ""),
                "topic": row.get("topic", ""),
            })
        rng.shuffle(items)
        return items[:n]

    # Binary: create Yes/No verification pairs (same as load_deepmath)
    items = []
    for idx, row, correct_value in pool_rows:
        distractor = _generate_distractor(correct_value, rng)
        correct_str = str(correct_value)
        distractor_str = str(distractor)

        base = {
            "question": row["question"],
            "dataset_name": "deepmath",
            "original_answer": correct_str,
            "difficulty": row.get("difficulty", ""),
            "topic": row.get("topic", ""),
        }

        items.append({
            **base,
            "problem_id": f"deepmath_{idx}_true",
            "ground_truth": "Yes",
            "presented_answer": correct_str,
        })
        items.append({
            **base,
            "problem_id": f"deepmath_{idx}_false",
            "ground_truth": "No",
            "presented_answer": distractor_str,
        })

    return _balance_and_sample(items, n, rng, f"deepmath_pool_{pool}")


def load_dataset_by_name(name: str, n: int = 300, seed: int = 137) -> list[dict]:
    """Dispatch to the appropriate dataset loader by name.

    Args:
        name: One of "gsm_symbolic", "gsm_symbolic_contamination", "folio", "strategyqa", "deepmath".
        n: Target number of problems.
        seed: Random seed for reproducibility.

    Returns:
        List of problem dicts in uniform schema.

    Raises:
        ValueError: If name is not a recognized dataset.
    """
    if name == "gsm_symbolic_contamination":
        return load_gsm_symbolic_contamination(n_per_type=n, seed=seed)

    loaders = {
        "gsm_symbolic": load_gsm_symbolic,
        "folio": load_folio,
        "strategyqa": load_strategyqa,
        "deepmath": load_deepmath,
    }

    if name not in loaders:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            f"Available: {sorted(list(loaders.keys()) + ['gsm_symbolic_contamination'])}"
        )

    return loaders[name](n=n, seed=seed)
