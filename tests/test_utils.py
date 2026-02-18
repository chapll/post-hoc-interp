"""Tests for src.utils.

Tested functions:
    set_seed(seed: int) -> None  -- raises ValueError if seed == 42
    setup_logging(name: str, level: str = "INFO") -> logging.Logger

Note: get_device(gpu_id) is not tested because it requires CUDA.
"""

import logging

import pytest

from src.utils import set_seed, setup_logging


class TestSetSeed:
    def test_valid_seed(self):
        """set_seed with a non-42 value should not raise."""
        set_seed(123)  # Should complete without error

    def test_seed_42_raises(self):
        """set_seed(42) must raise ValueError per project rules."""
        with pytest.raises(ValueError, match="42"):
            set_seed(42)

    def test_seed_zero(self):
        """set_seed(0) should work fine."""
        set_seed(0)

    def test_seed_large(self):
        """Large seeds should work."""
        set_seed(999999999)


class TestSetupLogging:
    def test_returns_logger(self):
        logger = setup_logging("test_logger_returns")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger_returns"

    def test_default_level_info(self):
        logger = setup_logging("test_default_level")
        assert logger.level == logging.INFO

    def test_custom_level_debug(self):
        logger = setup_logging("test_debug_level", level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_no_duplicate_handlers(self):
        """Calling setup_logging twice should not add duplicate handlers."""
        logger1 = setup_logging("dedup_test_unique")
        n_handlers = len(logger1.handlers)
        logger2 = setup_logging("dedup_test_unique")
        assert len(logger2.handlers) == n_handlers
        assert logger1 is logger2  # same logger instance

    def test_propagate_disabled(self):
        """Logger should have propagate=False to avoid duplicate output."""
        logger = setup_logging("test_propagate")
        assert logger.propagate is False
