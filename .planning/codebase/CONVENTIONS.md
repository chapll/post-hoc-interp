# Coding Conventions

**Analysis Date:** 2026-02-03

## Naming Patterns

**Files:**
- Python modules use snake_case: `train_iterator.py`, `parallelisms.py`, `checkpointer.py`
- Test files (none present): would follow `test_*.py` or `*_test.py` pattern
- Config files: YAML files in `configs/` with flat structure
- Classes use PascalCase: `TwoLayerMLP`, `TrainingIterator`, `Checkpointer`, `MultiLogger`

**Functions:**
- Functions use snake_case: `get_logger()`, `init_distributed()`, `apply_parallelisms()`, `add_metric()`
- Private methods prefixed with underscore: `_checkpoint()`, `_print_progress()`, `_should_checkpoint()`, `_time_step()`
- Static methods in utility classes: `ParameterGroupManager.is_ttt_parameter()`, `LRScheduleFunctions.cosine_decay_with_warmup()`

**Variables:**
- Local variables use snake_case: `model`, `optimizer`, `checkpoint_interval`, `total_steps`
- Constants use UPPER_SNAKE_CASE: `MODEL_STATE_DICT_KEY`, `DEFAULT_TIMEZONE`, `TORCH_DTYPE_MAP`, `NO_WEIGHT_DECAY_PATTERNS`
- Private globals use leading underscore: `_multi_logger`

**Types:**
- Type hints used throughout: `def __init__(self, start_step, total_steps, logger: MultiLogger, checkpointer: Checkpointer)`
- Return type hints: `-> None`, `-> int`, `-> torch.Tensor`
- Dataclasses for configuration: `@dataclass class JobConfig:`, `@dataclass(frozen=True) class ScheduleConfig:`
- Enum for schedule types: `class ScheduleType(str, Enum):` with values `COSINE`, `LINEAR`

## Code Style

**Formatting:**
- Line length: 160 characters (configured in `pyproject.toml` with `[tool.ruff] line-length = 160`)
- Import organization follows: stdlib → third-party (torch, omegaconf, hydra) → local imports
- No extraneous comments or decorative formatting
- Plain text output with simple print statements

**Linting:**
- Tool: ruff (configured in `pyproject.toml`)
- Commands: `uv run ruff check .` and `uv run ruff format .`
- No configuration for other linters (eslint, mypy, black) found in project

## Import Organization

**Order:**
1. Standard library imports: `import os`, `from datetime import datetime`, `from pathlib import Path`
2. Third-party imports: `import torch`, `from omegaconf import DictConfig`, `from hydra import main`
3. Local imports: `from src.infra.logging import get_logger`, `from src.models.mlp import TwoLayerMLP`

**Path Aliases:**
- No aliases configured in source files, uses absolute imports from project root: `from src.infra.checkpointer import Checkpointer`

## Error Handling

**Patterns:**
- Assertions for validation: `assert wandb.run is not None, "WandB run failed to initialize."`
- Try-except for fallback loading in `Checkpointer.load_pretrained()`: catches `RuntimeError` when loading model state
- Direct error propagation: `raise ValueError("No parameters found for the group")` in `ParameterGroupManager.create_param_group()`
- Type assertions with type ignore comments: `torch.distributed.all_reduce(..., op=torch.distributed.ReduceOp.MAX)  # type: ignore`

## Logging

**Framework:**
- `logging` standard library for terminal logging, configured by Hydra
- `MultiLogger` custom wrapper class in `src/infra/logging.py` handles both terminal and file logging
- `WandBLogger` wrapper for WandB integration

**Patterns:**
- Only rank 0 writes logs in distributed training: `if get_rank() == 0`
- Log to file with `logger.write(text)` method
- Log terminal vs file separately: `logger.write(text, log_terminal=False)` for file-only logs
- Metrics logged via `WandBLogger.log(stats, step=step)` in `save_multi()` method

## Comments

**When to Comment:**
- Licensing headers on adapted/borrowed code: all adapted infrastructure files include copyright notice
- Docstrings on public classes and methods: `"""Train a simple MLP on the identity task (0→0, 1→1)."""`
- Inline comments for non-obvious logic: `# Throw out first time since it usually is an outlier`, `# Only rank 0 writes logs`
- Parameter explanations in docstrings: `Args:` and `Returns:` sections

**JSDoc/TSDoc:**
- Not applicable (Python project, not JavaScript/TypeScript)
- Uses standard Python docstrings with Args/Returns/Raises sections in complex functions

## Function Design

**Size:**
- Utility functions are focused and short: `get_logger()` is 5 lines, `set_random_seed()` is 8 lines
- Iterator methods split into multiple small private methods: `__next__()` calls `_time_step()`, `_print_progress()`, `_checkpoint()`
- Larger functions document internal steps: `apply_parallelisms()` is 11 lines with clear progression

**Parameters:**
- Named parameters in `__init__` methods: `__init__(self, start_step, total_steps, logger: MultiLogger, checkpointer: Checkpointer, checkpoint_interval=0, ...)`
- Kwargs used for flexible stats: `add_multi(**kwargs)` in `TrainingIterator`
- Dataclass configs for grouped parameters: `config: DictConfig` contains all training settings

**Return Values:**
- Functions return single values or tuples: `create_specialized_optimizer()` returns `Tuple[torch.optim.AdamW, List[ScheduleConfig]]`
- None returned for side-effect functions: `def train(cfg: DictConfig, logger) -> None`
- Optional returns for failures: `get_latest_checkpoint_step()` returns `None` if no checkpoints found

## Module Design

**Exports:**
- Infrastructure modules export main classes: `TrainingIterator`, `Checkpointer`, `MultiLogger`
- Utility functions are module-level: `get_logger()`, `init_distributed()`, `set_random_seed()`
- No explicit `__all__` declarations in modules

**Barrel Files:**
- No barrel files (`__init__.py`) importing and re-exporting found in project structure
- Direct imports from specific modules: `from src.infra.train_iterator import TrainingIterator`

---

*Convention analysis: 2026-02-03*
