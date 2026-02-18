# Testing Patterns

**Analysis Date:** 2026-02-06

## Test Framework

**Status:** No formal test framework detected

**Observations:**
- No test files found (`test_*.py`, `*_test.py`, `*.spec.py` patterns returned no results)
- No test framework config files (`pytest.ini`, `jest.config.*`, etc.)
- No test dependencies in `pyproject.toml` (only production dependencies listed)

**Test Runner:**
Not applicable - no test suite present

**Run Commands:**
No test commands defined in `pyproject.toml` scripts section

## Test File Organization

**Location:**
No test files exist in the codebase

**Naming:**
Not established

**Structure:**
Not applicable

## Test Structure

**Suite Organization:**
Not applicable - no test suite exists

**Current Validation Approach:**
The codebase uses runtime validation scripts instead of unit tests:

Validation script in experimental pipeline:
- `src/experimental/gorn/scripts/validate.py` (690 lines)
- `src/experimental/gorn/scripts/check_traces.py` (371 lines)
- `src/experimental/gorn/scripts/verify_phase2.py` (902 lines)

These scripts validate data integrity and trace quality at runtime rather than using a test framework.

## Mocking

**Framework:** None

**Patterns:**
No mocking infrastructure. The codebase appears to be research-oriented with validation done on real data rather than via mocked unit tests.

## Fixtures and Factories

**Test Data:**
Not applicable for formal tests.

**Research Data:**
Dataset loaders handle real data:
- GSM-Symbolic: Loaded from HuggingFace `apple/GSM-Symbolic` dataset
- FOLIO: Loaded from HuggingFace `tasksource/folio` dataset
- StrategyQA: Loaded from HuggingFace `ChilleD/StrategyQA` dataset

Deterministic data generation via seeded RNG:
```python
rng = random.Random(seed)
# ... use rng for reproducible shuffling/sampling
```

**Location:**
- Dataset loaders: `src/experimental/gorn/lib/datasets.py`
- Prompt builders: `src/experimental/gorn/lib/prompts.py`

## Coverage

**Requirements:** None enforced

**View Coverage:**
Not applicable - no coverage tooling configured

## Test Types

**Unit Tests:**
Not present

**Integration Tests:**
Not present

**E2E Tests:**
Not present

**Validation Scripts:**
The codebase uses validation scripts that serve similar purposes to integration tests:

- **Trace validation** (`src/experimental/gorn/scripts/validate.py`): Validates generated traces for correctness, checks token boundaries, verifies think blocks
- **Trace checking** (`src/experimental/gorn/scripts/check_traces.py`): Summarizes trace statistics, detects anomalies
- **Phase 2 verification** (`src/experimental/gorn/scripts/verify_phase2.py`): End-to-end verification of multi-stage pipeline outputs

## Common Patterns

**Error Testing:**
Not applicable in traditional test sense.

Runtime error handling uses assertions and exceptions:
```python
assert gc_freq > 0, "gc_freq must be a positive integer"

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. This pipeline requires GPU. "
        "No CPU fallback is provided."
    )
```

**Async Testing:**
Not applicable

## Validation Patterns

Since the codebase lacks formal tests, validation is done through:

**1. Type Checking:**
Extensive type hints enable static analysis via mypy (though no mypy config detected)

**2. Runtime Assertions:**
Invariant checks throughout codebase:
```python
assert dist.get_world_size() % tp_sharding == 0, "world size must be divisible by tp_sharding"
assert dp_sharding <= global_bs, "dp_sharding must be greater than or equal to the global batch size"
```

**3. Configuration Validation:**
Configs are validated at runtime via OmegaConf schema and explicit checks

**4. Data Validation Scripts:**
Dedicated validation scripts check pipeline outputs:

From `src/experimental/gorn/scripts/validate.py`:
- Validates trace structure (token_ids, metadata, think blocks)
- Checks activation shapes match expected dimensions
- Verifies logit top-k values and indices
- Ensures entropy calculations are correct

**5. Logging and Monitoring:**
Extensive logging throughout pipeline serves diagnostic purpose:
```python
logger.warning(
    "GSM-Symbolic row %d: no #### separator in answer, skipping",
    idx
)
```

**6. Reproducibility via Seeding:**
Deterministic behavior enforced through seed management:

From `src/experimental/gorn/lib/utils.py:11`:
```python
def set_seed(seed: int) -> None:
    """Set deterministic random seeds across all relevant libraries."""
    if seed == 42:
        raise ValueError("Seed 42 is not allowed. Use any other seed value.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

## Testing Recommendations

**For Future Development:**

1. **Add pytest framework:**
```bash
# Add to pyproject.toml dependencies
pytest>=8.0.0
pytest-cov>=4.1.0
pytest-mock>=3.12.0
```

2. **Organize tests:**
```
tests/
├── unit/
│   ├── test_datasets.py
│   ├── test_prompts.py
│   └── test_utils.py
├── integration/
│   ├── test_pipeline.py
│   └── test_trace_generation.py
└── fixtures/
    └── sample_data.py
```

3. **Test small functions first:**
Start with pure functions in `src/experimental/gorn/lib/`:
- `tokens.py`: `parse_think_boundaries()`, `get_think_content_mask()`
- `prompts.py`: `build_verification_prompt()`, `_build_gsm_prompt()`
- `utils.py`: `set_seed()`, `setup_logging()`

4. **Mock external dependencies:**
- HuggingFace dataset loaders
- Model inference calls
- File I/O operations

5. **Consider property-based testing:**
For data validation logic, use `hypothesis` to test edge cases:
```python
from hypothesis import given
import hypothesis.strategies as st

@given(st.lists(st.integers(min_value=0, max_value=200000)))
def test_parse_think_boundaries_robust(token_ids):
    blocks = parse_think_boundaries(token_ids)
    # Verify blocks are well-formed, non-overlapping, etc.
```

## Current Testing Gap Analysis

**Untested Critical Areas:**

1. **Token boundary parsing** (`src/experimental/gorn/lib/tokens.py`):
   - Risk: Incorrect think block detection breaks entire pipeline
   - Files: `src/experimental/gorn/lib/tokens.py`
   - Priority: High

2. **Dataset loaders** (`src/experimental/gorn/lib/datasets.py`):
   - Risk: Incorrect label mapping or data corruption
   - Files: `src/experimental/gorn/lib/datasets.py`
   - Priority: High

3. **Prompt construction** (`src/experimental/gorn/lib/prompts.py`):
   - Risk: Wrong label tokens break swapped-label protocol
   - Files: `src/experimental/gorn/lib/prompts.py`
   - Priority: High

4. **Configuration loading** (`src/experimental/gorn/lib/config.py`, `src/config.py`):
   - Risk: Config merging bugs lead to wrong hyperparameters
   - Files: `src/experimental/gorn/lib/config.py`, `src/config.py`
   - Priority: Medium

5. **Training infrastructure** (`archive/src/infra/*`) [archived]:
   - Risk: Distributed training setup failures, checkpoint corruption
   - Files: `archive/src/infra/parallelisms.py`, `archive/src/infra/checkpointer.py`, `archive/src/infra/train_iterator.py`
   - Priority: Low (archived, not used by current research pipeline)

**Team Conventions Needed:**

Since this is a multi-user research codebase (Ali, Gorn, Eric), establish:
- **Who writes tests:** Should experimental code include tests, or only core infrastructure?
- **Test data management:** How to share test fixtures across team members with different compute environments
- **CI/CD:** No CI detected - consider adding GitHub Actions for basic smoke tests
- **Pre-commit hooks:** No pre-commit config - consider adding ruff formatting checks

---

*Testing analysis: 2026-02-06*
