# Coding Conventions

**Analysis Date:** 2026-02-06

## Naming Patterns

**Files:**
- Python modules: `snake_case.py` (`train.py`, `config.py`, `logging.py`)
- Scripts: Descriptive names in `snake_case` (`generate_traces.py`, `capture_activations.py`, `verify_phase2.py`)
- Config files: Lowercase with underscores (`base.yaml`, `generate.yaml`, `submitit_example.yaml`)

**Functions:**
- Standard: `snake_case` (`set_seed`, `load_model`, `get_device`, `build_verification_prompt`)
- Private helpers: Prefix with underscore (`_balance_and_sample`, `_generate_distractor`, `_build_gsm_prompt`)
- Context managers: Use descriptive names (`TimedContext`)

**Variables:**
- Local variables: `snake_case` (`token_ids`, `problem_id`, `dataset_name`, `max_new_tokens`)
- Constants: `UPPER_SNAKE_CASE` (`THINK_TOKEN_ID`, `THINK_END_TOKEN_ID`, `DEFAULT_TIMEZONE`, `LABEL_MAPPINGS`)
- Class attributes: `UPPER_SNAKE_CASE` for constants (`NO_WEIGHT_DECAY_PATTERNS`, `TTT_PARAMETER_PATTERNS`, `WEIGHT_DECAY_VALUE`)

**Classes:**
- `PascalCase` (`TwoLayerMLP`, `MultiLogger`, `WandBLogger`, `GarbageCollection`, `ParameterGroupManager`)
- Enums: `PascalCase` with members in `UPPER_CASE` (`ScheduleType.COSINE`, `ScheduleType.LINEAR`)
- Dataclasses: Config classes use `PascalCase` with `Config` suffix (`JobConfig`, `ModelConfig`, `TrainingConfig`)

**Modules:**
- Package structure: `src/` root with subpackages `infra/`, `models/`, `experimental/gorn/`
- Experimental work: Isolated in `src/experimental/{username}/` directories (e.g., `src/experimental/gorn/`)

## Code Style

**Formatting:**
- Tool: Ruff (configured in `pyproject.toml`)
- Line length: 160 characters (see `pyproject.toml`: `line-length = 160`)
- Python version: Requires Python 3.13+

**Linting:**
- No explicit linter config files (`.flake8`, `.pylintrc`) detected
- Ruff handles both formatting and linting

**Import Sorting:**
Not enforced via config. Observed pattern varies between modules.

## Import Organization

**Order:**
1. Standard library imports (grouped, typically unordered)
2. Third-party imports (PyTorch, transformers, datasets, etc.)
3. Local project imports (using `from src.` or relative imports)

**Examples:**

From `src/train.py`:
```python
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# ... OmegaConf setup ...

import hydra
import torch
import torch.nn as nn

from src.infra.logging import get_logger
from src.models.mlp import TwoLayerMLP
```

From `src/experimental/gorn/scripts/generate_traces.py`:
```python
import gc
import logging
import os
import sys
import time

import torch

# -- project imports ----------------------------------------------------------
sys.path.insert(0, "src/experimental/gorn")

from lib.config import load_config
from lib.datasets import (...)
```

**Path Aliases:**
- Standard project imports use absolute: `from src.infra.logging import ...`
- Experimental code uses `sys.path.insert` hack for relative imports within experiment: `sys.path.insert(0, "src/experimental/gorn")` then `from lib.config import ...`

**Note on Team Conventions:**
The `sys.path.insert` pattern in experimental code is user-specific (Gorn's experimental directory). Main codebase uses clean absolute imports from `src.`.

## Error Handling

**Patterns:**
- Use exceptions for control flow: `raise ValueError(...)`, `raise RuntimeError(...)`, `raise FileNotFoundError(...)`
- Assertions for invariant checks: `assert dist.get_world_size() % tp_sharding == 0, "..."`
- Try-except for graceful degradation in worker processes (see `src/experimental/gorn/scripts/generate_traces.py:191`)

**Common patterns:**

Validation with descriptive errors:
```python
if seed == 42:
    raise ValueError("Seed 42 is not allowed. Use any other seed value.")
```

GPU availability checks (no silent CPU fallback):
```python
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This pipeline requires GPU. "
        "No CPU fallback is provided."
    )
```

Worker-level exception handling (log and continue):
```python
try:
    # Generate trace...
except Exception:
    failed_count += 1
    worker_logger.exception(
        "Worker %d: error generating %s trace %d",
        rank,
        problem_id,
        trace_index,
    )
    continue
```

## Logging

**Framework:** Python `logging` module (standard library)

**Patterns:**
- Create module-level loggers: `logger = logging.getLogger(__name__)`
- Use `setup_logging(name)` helper in experimental code (see `src/experimental/gorn/lib/utils.py:56`)
- Log levels: INFO for progress, WARNING for issues, ERROR/EXCEPTION for failures
- Format: `[LEVEL] name: message` (see `src/experimental/gorn/lib/utils.py:76`)

**Multi-logger pattern:**
Main training infrastructure uses custom `MultiLogger` (`src/infra/logging.py:90`) that:
- Writes to both terminal and log file
- Integrates with WandB for metric tracking
- Supports rank-aware logging (only rank 0 writes to file in distributed mode)

**Examples:**

Module-level logger:
```python
logger = logging.getLogger(__name__)
# ...
logger.info("GSM-Symbolic: loaded %d rows", len(ds))
logger.warning("No think block found, routing to failed/")
```

Custom MultiLogger for training:
```python
logger = get_logger(cfg)
logger.write(f"Step {step}/{total} | Loss: {loss.item():.6f}")
logger.write("Config details", log_terminal=False)  # File only
```

## Comments

**When to Comment:**
- Module docstrings: Every module has a docstring explaining purpose (see `src/experimental/gorn/lib/datasets.py:1`)
- Function docstrings: Public functions have Google-style docstrings with Args/Returns/Raises
- Inline comments: Used sparingly for non-obvious logic or important warnings
- Copyright notices: Files adapted from external sources include attribution (see `src/infra/logging.py:1-3`)

**Docstring Style:**
Google-style with typed parameters:

```python
def set_seed(seed: int) -> None:
    """Set deterministic random seeds across all relevant libraries.

    Args:
        seed: Integer seed value. Must not be 42.

    Raises:
        ValueError: If seed is 42.
    """
```

Multi-paragraph module docstrings for complex modules:
```python
"""Dataset loaders for binary verification format.

Loads GSM-Symbolic, FOLIO, and StrategyQA from HuggingFace, converts each
problem into base problem dicts with balanced Yes/No ground truth labels.

Prompts are NOT built here. Use lib.prompts.build_verification_prompt() to
construct prompts with configurable label tokens...
"""
```

## Function Design

**Size:** No strict limit, but worker functions tend to be 50-150 lines. Helpers are shorter (10-50 lines).

**Parameters:**
- Use type hints consistently: `def load_config(script_config: str = None) -> DictConfig:`
- Config objects passed as first parameter: `def train(cfg: DictConfig, logger) -> None:`
- Use keyword arguments for clarity in calls with many parameters

**Return Values:**
- Explicit return type hints: `-> None`, `-> dict[str, torch.Tensor]`, `-> list[dict]`
- Return tuples when multiple values needed: `-> tuple[int, int] | None`
- Use `| None` for optional returns (Python 3.10+ union syntax)

## Module Design

**Exports:**
- No explicit `__all__` declarations
- Import specific names: `from lib.config import load_config`
- Group related imports: `from lib.datasets import (load_folio, load_gsm_symbolic_contamination, load_strategyqa)`

**Barrel Files:**
- `__init__.py` files are empty (see `src/experimental/gorn/lib/__init__.py`, `src/experimental/gorn/scripts/__init__.py`)
- No re-exports; imports are direct from modules

## Configuration Management

**Framework:** Hydra + OmegaConf for main training, custom OmegaConf loader for experimental scripts

**Patterns:**
- Environment variable interpolation: `${oc.env:EXP_DIR,./exp}` (see `configs/config.yaml:13`)
- Custom resolvers for dynamic values: `${now_date:}`, `${now_time:}` (see `src/train.py:11-12`)
- Hierarchical configs with defaults: `defaults:` section loads base configs (see `configs/config.yaml:1-9`)
- CLI overrides: Hydra supports `+deploy=interactive` syntax; experimental code uses dotlist (`model.name=foo`)

**Per-user configuration:**
- `.env` file for user-specific paths (gitignored, with `.env.example` template)
- Deployment configs: Users create `configs/deploy/submitit_yourname.yaml` from example
- Data/experiment paths configurable via environment variables (`EXP_DIR`, `DATA_DIR`, `SLURM_PARTITION`)

**Team convention standardization needed:**
- Import style differs between main codebase (clean absolute imports) and experimental code (`sys.path.insert` hack)
- Configuration approach differs: Hydra for main training vs. custom argparse+OmegaConf for experimental scripts
- Consider standardizing on one config approach for consistency

## Type Annotations

**Usage:** Extensive type hints on function signatures, return values, and complex variables

**Patterns:**
- Modern union syntax: `int | float`, `str | None` (Python 3.10+)
- Generic types: `list[dict]`, `dict[str, torch.Tensor]`, `tuple[int, int]`
- Dataclasses use field type annotations: `exp_name: str = "default-job"`

## File Headers

**Copyright/Attribution:**
Files adapted from external sources include header:
```python
# Adapted from https://github.com/test-time-training/ttt-video-dit/...
# Copyright 2025 test-time-training. All rights reserved.
# Licensed under the MIT License
```

**Module docstrings:**
All modules start with triple-quoted docstring explaining purpose and usage.

---

*Convention analysis: 2026-02-06*
