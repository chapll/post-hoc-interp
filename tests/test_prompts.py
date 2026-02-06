"""Tests for src.infra.prompts.

Tested function:
    build_verification_prompt(problem: dict, label_protocol: str = "swapped", config: str | None = None) -> dict

Returns: {"verification_prompt": str, "prompt_config": str|None, "label_mapping": dict}

Datasets: gsm_symbolic (question, presented_answer), folio (premises, question), strategyqa (question)
"""

import pytest

from src.infra.prompts import LABEL_MAPPINGS, build_verification_prompt

GSM_PROBLEM = {
    "dataset_name": "gsm_symbolic",
    "question": "If Alice has 3 apples and buys 2 more, how many does she have?",
    "presented_answer": "5",
}

FOLIO_PROBLEM = {
    "dataset_name": "folio",
    "premises": "All cats are animals. Whiskers is a cat.",
    "question": "Whiskers is an animal.",
}

STRATEGYQA_PROBLEM = {
    "dataset_name": "strategyqa",
    "question": "Is the sky blue?",
}


class TestSwappedProtocol:
    def test_config_a(self):
        result = build_verification_prompt(GSM_PROBLEM, label_protocol="swapped", config="A")
        assert result["prompt_config"] == "A"
        assert result["label_mapping"] == {"valid": "Rho", "invalid": "Tau"}
        assert "Rho" in result["verification_prompt"]
        assert "Tau" in result["verification_prompt"]

    def test_config_b(self):
        result = build_verification_prompt(GSM_PROBLEM, label_protocol="swapped", config="B")
        assert result["prompt_config"] == "B"
        assert result["label_mapping"] == {"valid": "Tau", "invalid": "Rho"}
        assert "Tau" in result["verification_prompt"]
        assert "Rho" in result["verification_prompt"]

    def test_invalid_config_raises(self):
        with pytest.raises(ValueError):
            build_verification_prompt(GSM_PROBLEM, label_protocol="swapped", config="C")

    def test_none_config_raises(self):
        """swapped protocol requires config A or B."""
        with pytest.raises(ValueError):
            build_verification_prompt(GSM_PROBLEM, label_protocol="swapped", config=None)


class TestYesNoProtocol:
    def test_yesno(self):
        result = build_verification_prompt(GSM_PROBLEM, label_protocol="yesno")
        assert result["prompt_config"] is None
        assert "Yes" in result["verification_prompt"]
        assert "No" in result["verification_prompt"]

    def test_yesno_mapping(self):
        result = build_verification_prompt(GSM_PROBLEM, label_protocol="yesno")
        assert result["label_mapping"] == {"valid": "Yes", "invalid": "No"}


class TestUnknownProtocol:
    def test_raises(self):
        with pytest.raises(ValueError):
            build_verification_prompt(GSM_PROBLEM, label_protocol="unknown")


class TestDatasetSpecificPrompts:
    def test_gsm_contains_answer(self):
        result = build_verification_prompt(GSM_PROBLEM, label_protocol="swapped", config="A")
        assert "5" in result["verification_prompt"]
        assert "Alice" in result["verification_prompt"]

    def test_folio_contains_premises(self):
        result = build_verification_prompt(FOLIO_PROBLEM, label_protocol="swapped", config="A")
        assert "All cats are animals" in result["verification_prompt"]
        assert "Whiskers is an animal" in result["verification_prompt"]

    def test_strategyqa_contains_question(self):
        result = build_verification_prompt(STRATEGYQA_PROBLEM, label_protocol="swapped", config="A")
        assert "Is the sky blue?" in result["verification_prompt"]

    def test_unknown_dataset_raises(self):
        bad_problem = {"dataset_name": "nonexistent", "question": "x"}
        with pytest.raises(ValueError, match="nonexistent"):
            build_verification_prompt(bad_problem, label_protocol="swapped", config="A")


class TestLabelMappingsConstant:
    def test_label_mappings_structure(self):
        assert "A" in LABEL_MAPPINGS
        assert "B" in LABEL_MAPPINGS
        assert LABEL_MAPPINGS["A"]["valid"] == "Rho"
        assert LABEL_MAPPINGS["A"]["invalid"] == "Tau"
        assert LABEL_MAPPINGS["B"]["valid"] == "Tau"
        assert LABEL_MAPPINGS["B"]["invalid"] == "Rho"
