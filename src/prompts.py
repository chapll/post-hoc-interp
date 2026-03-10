"""Prompt construction for swapped-label probing protocol.

Builds verification prompts with configurable label tokens (Mu/Nu instead of
Yes/No) to eliminate the confound where probes detect hypothetical reasoning
about "Yes"/"No" tokens rather than abstract validity.

The swapped-label protocol generates each problem under two complementary label
mappings:
  - Config A: Valid=Mu, Invalid=Nu
  - Config B: Valid=Nu, Invalid=Mu

With both configs in training data, Mu/Nu have zero correlation with the
validity label, forcing the probe to find the abstract logical signal.
"""

LABEL_MAPPINGS = {
    "A": {"valid": "Mu", "invalid": "Nu"},
    "B": {"valid": "Nu", "invalid": "Mu"},
}

YESNO_MAPPING = {"valid": "Yes", "invalid": "No"}


def build_verification_prompt(
    problem: dict,
    label_protocol: str = "swapped",
    config: str | None = None,
) -> dict:
    """Build a verification prompt with configurable label tokens.

    Args:
        problem: Problem dict with keys: question, dataset_name, and
            dataset-specific fields (presented_answer, premises, etc.).
        label_protocol: "swapped" for Mu/Nu protocol, "yesno" for Yes/No.
        config: For swapped protocol, "A" or "B". Ignored for yesno.

    Returns:
        Dict with:
        - verification_prompt: The complete prompt string.
        - prompt_config: "A", "B", or None (for yesno).
        - label_mapping: Dict with "valid" and "invalid" keys mapping to tokens.

    Raises:
        ValueError: If config is invalid or protocol is unknown.
    """
    if label_protocol == "swapped":
        if config not in ("A", "B"):
            raise ValueError(f"config must be 'A' or 'B' for swapped protocol, got: {config}")
        mapping = LABEL_MAPPINGS[config]
        prompt_config = config
    elif label_protocol == "yesno":
        mapping = YESNO_MAPPING
        prompt_config = None
    else:
        raise ValueError(f"Unknown label_protocol: {label_protocol}")

    valid_word = mapping["valid"]
    invalid_word = mapping["invalid"]
    dataset_name = problem["dataset_name"]

    if dataset_name == "gsm_symbolic":
        prompt = _build_gsm_prompt(problem, valid_word, invalid_word)
    elif dataset_name == "folio":
        prompt = _build_folio_prompt(problem, valid_word, invalid_word)
    elif dataset_name == "strategyqa":
        prompt = _build_strategyqa_prompt(problem, valid_word, invalid_word)
    elif dataset_name == "deepmath":
        prompt = _build_deepmath_prompt(problem, valid_word, invalid_word)
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return {
        "verification_prompt": prompt,
        "prompt_config": prompt_config,
        "label_mapping": mapping,
    }


def _build_gsm_prompt(problem: dict, valid_word: str, invalid_word: str) -> str:
    """Build GSM-Symbolic verification prompt."""
    question = problem["question"]
    presented_answer = problem["presented_answer"]

    return (
        f"{question}\n\n"
        f"Is the answer {presented_answer}? "
        f"If it is correct, output exactly the word: {valid_word}. "
        f"If it is not correct, output exactly the word: {invalid_word}.\n"
        f"Please think step-by-step to verify whether the answer is correct."
    )


def _build_folio_prompt(problem: dict, valid_word: str, invalid_word: str) -> str:
    """Build FOLIO verification prompt."""
    premises = problem["premises"]
    conclusion = problem["question"]  # conclusion stored in question field

    return (
        f"Given the following premises:\n"
        f"{premises}\n\n"
        f'Does the following conclusion logically follow from the premises: "{conclusion}"?\n'
        f"If it logically follows, output exactly the word: {valid_word}. "
        f"If it does not follow, output exactly the word: {invalid_word}.\n"
        f"Please think step-by-step to verify."
    )


def _build_strategyqa_prompt(problem: dict, valid_word: str, invalid_word: str) -> str:
    """Build StrategyQA verification prompt."""
    question = problem["question"]

    return (
        f"Answer the following question: {question}\n"
        f"If the answer is yes, output exactly the word: {valid_word}. "
        f"If the answer is no, output exactly the word: {invalid_word}.\n"
        f"Please think step-by-step to verify your answer."
    )


def _build_deepmath_prompt(problem: dict, valid_word: str, invalid_word: str) -> str:
    """Build DeepMath verification prompt."""
    question = problem["question"]
    presented_answer = problem["presented_answer"]

    return (
        f"{question}\n\n"
        f"Is the answer {presented_answer}? "
        f"If it is correct, output exactly the word: {valid_word}. "
        f"If it is not correct, output exactly the word: {invalid_word}.\n"
        f"Please think step-by-step to verify whether the answer is correct."
    )


def build_freeform_prompt(problem: dict) -> dict:
    """Build a freeform (non-binary) prompt for transferability testing.

    Instead of Mu/Nu verification, the model solves the problem and gives
    its final answer directly. Used for testing whether directions extracted
    from binary-format traces transfer to freeform reasoning.

    Args:
        problem: Problem dict with question and dataset_name fields.

    Returns:
        Dict with verification_prompt (freeform), prompt_config=None,
        label_mapping=None.
    """
    dataset_name = problem["dataset_name"]

    if dataset_name in ("deepmath", "deepmath_freeform"):
        prompt = (
            f"{problem['question']}\n\n"
            f"Please think step-by-step and give your final numerical answer."
        )
    elif dataset_name == "gsm_symbolic":
        prompt = (
            f"{problem['question']}\n\n"
            f"Please think step-by-step and give your final numerical answer."
        )
    elif dataset_name == "folio":
        prompt = (
            f"Given the following premises:\n"
            f"{problem.get('premises', '')}\n\n"
            f'Does the following conclusion logically follow: "{problem["question"]}"?\n'
            f"Please think step-by-step and give your final answer (True or False)."
        )
    elif dataset_name == "strategyqa":
        prompt = (
            f"Answer the following question: {problem['question']}\n"
            f"Please think step-by-step and give your final answer (Yes or No)."
        )
    else:
        prompt = (
            f"{problem['question']}\n\n"
            f"Please think step-by-step and give your final answer."
        )

    return {
        "verification_prompt": prompt,
        "prompt_config": None,
        "label_mapping": None,
    }
