"""Utility functions for experimental pipeline: seeding, device, logging."""

import logging
import random
import sys

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set deterministic random seeds across all relevant libraries.

    Args:
        seed: Integer seed value. Must not be 42.

    Raises:
        ValueError: If seed is 42.
    """
    if seed == 42:
        raise ValueError("Seed 42 is not allowed. Use any other seed value.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(gpu_id: int) -> torch.device:
    """Return a CUDA device for the given GPU ID.

    Args:
        gpu_id: Integer GPU index.

    Returns:
        torch.device for the specified CUDA GPU.

    Raises:
        RuntimeError: If CUDA is not available. No silent CPU fallback.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This pipeline requires GPU. "
            "No CPU fallback is provided."
        )
    if gpu_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPU {gpu_id} but only {torch.cuda.device_count()} "
            f"GPU(s) are available."
        )
    return torch.device(f"cuda:{gpu_id}")


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    """Create a plain text logger with the given name.

    Format: [LEVEL] name: message
    Output goes to stdout. No emoji, no decorative formatting.

    Args:
        name: Logger name.
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Avoid adding duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

    return logger
