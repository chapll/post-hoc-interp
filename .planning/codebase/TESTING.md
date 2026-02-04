# Testing Patterns

**Analysis Date:** 2026-02-03

## Test Framework

**Status:** Not Currently Configured

This project has no test framework installed or configured. The codebase does not include:
- pytest, unittest, or other test runners
- Test configuration files (pytest.ini, setup.cfg, pyproject.toml test settings)
- Any test files in the repository
- Testing-related dependencies in `pyproject.toml`

**Linting Framework:**
- Tool: ruff
- Version: Configured in `pyproject.toml`
- Run Commands:
```bash
uv run ruff check .              # Check for linting issues
uv run ruff format .             # Format code
```

## Test File Organization

**Current State:** No test files exist

**Expected Pattern (if tests were added):**
- Location: `tests/` directory at project root (co-located approach is not yet established)
- Naming: `test_*.py` or `*_test.py` pattern
- Structure: One test file per source module (e.g., `tests/test_train_iterator.py` for `src/infra/train_iterator.py`)

**Recommended Directory Structure:**
```
tests/
├── __init__.py
├── conftest.py                    # Shared fixtures and configuration
├── test_infra/
│   ├── __init__.py
│   ├── test_train_iterator.py
│   ├── test_checkpointer.py
│   ├── test_logging.py
│   ├── test_parallelisms.py
│   └── test_optimizers.py
├── test_models/
│   ├── __init__.py
│   └── test_mlp.py
└── test_train.py
```

## Test Structure

**Components that would benefit from testing:**

**1. TrainingIterator (`src/infra/train_iterator.py`):**
- Critical class for training loop management
- Complex logic for timeout detection and checkpointing
- Would need tests for:
  - Step iteration and resumption
  - Metric tracking via `add_metric()` and `add_multi()`
  - Checkpoint interval logic
  - Timeout checkpoint detection
  - Progress message formatting

**2. Checkpointer (`src/infra/checkpointer.py`):**
- Handles distributed checkpoint save/load
- Uses PyTorch distributed checkpoint API
- Would need tests for:
  - State dict creation and loading
  - Metadata handling (WandB ID persistence)
  - Model, optimizer, scheduler state preservation
  - Fallback loading for different checkpoint formats

**3. ParameterGroupManager (`src/infra/optimizers.py`):**
- Pure function logic for parameter categorization
- Would benefit from unit tests:
  - `is_ttt_parameter()` pattern matching
  - `should_skip_weight_decay()` pattern matching
  - Parameter grouping correctness

**4. LRScheduleFunctions (`src/infra/optimizers.py`):**
- Schedule calculations that must be numerically correct
- Would need tests for:
  - Warmup phase behavior
  - Cosine decay calculations
  - Linear decay calculations
  - Edge cases (lr_peak == 0, step boundaries)

**5. Distributed Utilities (`src/infra/parallelisms.py`):**
- Device mesh initialization and configuration
- Would need tests for:
  - World mesh creation with various sharding configs
  - Assertion validation
  - Tensor parallelism plan generation

## Mocking Strategy (If Tests Were Written)

**Distributed Operations:**
- Mock `torch.distributed` calls in non-distributed tests
- Mock rank determination: `get_rank()` should return 0 for single-process tests
- Mock broadcast operations for cross-rank communication

**File I/O:**
- Mock `os.makedirs` and file operations in logging tests
- Use temporary directories for integration tests

**WandB:**
- Mock `wandb.init()` and `wandb.log()` in MultiLogger tests
- Mock `wandb.run` object for run resumption tests

**PyTorch Checkpoint:**
- Mock `torch.distributed.checkpoint` calls for unit tests
- Use real checkpoints for integration tests

## Fixtures and Test Data

**Potential Test Fixtures (if tests were added):**

```python
# conftest.py example structure (not present)
import pytest
from omegaconf import OmegaConf
from src.config import Config

@pytest.fixture
def base_config():
    """Fixture providing minimal valid config"""
    return OmegaConf.structured(Config)

@pytest.fixture
def simple_model():
    """Fixture for simple test model"""
    from src.models.mlp import TwoLayerMLP
    return TwoLayerMLP(input_dim=1, hidden_dim=8, output_dim=1)

@pytest.fixture
def simple_optimizer(simple_model):
    """Fixture for optimizer"""
    return torch.optim.Adam(simple_model.parameters(), lr=1e-3)
```

**Test Data Location (if tests were written):**
- `tests/fixtures/` directory for YAML config samples
- `tests/fixtures/checkpoints/` for test checkpoint files
- In-memory generation for simple tensor data in unit tests

## Coverage

**Requirements:** Not enforced (no coverage tools configured)

**If Coverage Were Added:**
- Recommended minimum: 80% for core infrastructure
- Priority areas: `TrainingIterator`, `Checkpointer`, `ParameterGroupManager`, `LRScheduleFunctions`
- Lower priority: model definitions (tested via integration tests), config loading (Hydra tested externally)

## Test Types (Not Yet Implemented)

**Unit Tests (Would Test):**
- Pure functions: `LRScheduleFunctions.cosine_decay_with_warmup()`, `set_random_seed()`
- Parameter categorization: `ParameterGroupManager` methods
- Config loading and validation
- Individual optimizer/scheduler creation

**Integration Tests (Would Test):**
- Full training loop with `TrainingIterator`
- Checkpoint save and load cycle
- Distributed training setup (with 2-4 processes)
- Multi-logger file writing and WandB integration

**End-to-End Tests (Would Test):**
- Full training run from config to checkpoint
- Resumption from checkpoint at various steps
- Timeout-based checkpointing
- Multi-node distributed training on Slurm

## Patterns for Future Tests

**Async/Distributed Testing Pattern:**
```python
# Example (not present in codebase)
@pytest.mark.parametrize("tp_sharding,dp_sharding", [(1, 4), (2, 2), (4, 1)])
def test_parallelism_combinations(tp_sharding, dp_sharding):
    # Mock distributed setup
    # Verify mesh creation
    pass
```

**Config-Parametrized Tests:**
```python
# Example (not present in codebase)
@pytest.mark.parametrize("schedule_type", [ScheduleType.COSINE, ScheduleType.LINEAR])
def test_lr_schedules(schedule_type):
    # Test different schedule implementations
    pass
```

**Fixture-Based State Testing:**
```python
# Example (not present in codebase)
def test_training_iterator_resume(training_iterator):
    # Simulate step progression
    # Save state
    # Load state
    # Verify resumption
    pass
```

## Known Testing Gaps

**Critical Gaps:**
- No testing infrastructure at all
- Complex distributed training code (`parallelisms.py`) untested
- Checkpoint loading/saving has no regression tests
- Training loop state management (`TrainingIterator`) has no unit tests

**Impact:**
- Refactoring carries high risk
- Bugs in distributed code may only appear under specific Slurm configurations
- Checkpoint compatibility changes could break resumption silently

**Recommendation:**
Start with unit tests for pure functions in `optimizers.py` and `utils.py`, then add integration tests for `TrainingIterator` with mocked distributed operations.

---

*Testing analysis: 2026-02-03*
