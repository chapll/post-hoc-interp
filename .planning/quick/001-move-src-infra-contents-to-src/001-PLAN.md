---
phase: quick
plan: 001
type: execute
wave: 1
depends_on: []
files_modified:
  # Moved files (git mv src/infra/*.py -> src/*.py)
  - src/config.py
  - src/datasets.py
  - src/model.py
  - src/multiprocess.py
  - src/prompts.py
  - src/storage.py
  - src/tokens.py
  - src/utils.py
  - src/__init__.py
  # Scripts with imports updated
  - scripts/generate_traces.py
  - scripts/capture_activations.py
  - scripts/check_traces.py
  - scripts/validate.py
  - scripts/verify_phase2.py
  # Tests with imports updated
  - tests/test_imports.py
  - tests/test_config.py
  - tests/test_tokens.py
  - tests/test_storage.py
  - tests/test_utils.py
  - tests/test_prompts.py
  - tests/test_multiprocess.py
  # Planning docs updated
  - .planning/PROJECT.md
  - .planning/codebase/CONVENTIONS.md
  - .planning/codebase/CONCERNS.md
  - .planning/codebase/STRUCTURE.md
  - .planning/codebase/ARCHITECTURE.md
  - .planning/codebase/TESTING.md
autonomous: true

must_haves:
  truths:
    - "All Python imports use `from src.X import Y` (no more `from src.infra.X`)"
    - "All existing tests pass after the move"
    - "src/infra/ directory no longer exists"
    - "All 8 modules are directly under src/"
  artifacts:
    - path: "src/config.py"
      provides: "Configuration loader"
    - path: "src/datasets.py"
      provides: "Dataset loaders"
    - path: "src/model.py"
      provides: "Model loading and trace generation"
    - path: "src/multiprocess.py"
      provides: "Multi-GPU parallel execution"
    - path: "src/prompts.py"
      provides: "Prompt building"
    - path: "src/storage.py"
      provides: "Trace storage"
    - path: "src/tokens.py"
      provides: "Token parsing"
    - path: "src/utils.py"
      provides: "Utilities"
  key_links:
    - from: "scripts/*.py"
      to: "src/*.py"
      via: "from src.X import Y"
      pattern: "from src\\.(config|datasets|model|multiprocess|prompts|storage|tokens|utils) import"
    - from: "tests/*.py"
      to: "src/*.py"
      via: "from src.X import Y"
      pattern: "from src\\.(config|datasets|model|multiprocess|prompts|storage|tokens|utils) import"
---

<objective>
Flatten `src/infra/` by moving all 8 Python modules up one level into `src/` directly, then update every import reference across the entire codebase (scripts, tests, archive references) and planning documentation from `src.infra.X` to `src.X`.

Purpose: Simplify the module hierarchy -- the `infra/` subdirectory adds no organizational value now that the old training infra (logging, checkpointer, parallelisms, etc.) has been archived. Direct `from src.config import load_config` is cleaner than `from src.infra.config import load_config`.

Output: Flat `src/` package with all modules accessible via `from src.X import Y`. All tests passing.
</objective>

<execution_context>
@/home/aahmad/.claude/get-shit-done/workflows/execute-plan.md
@/home/aahmad/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Move files and update all Python imports</name>
  <files>
    src/config.py
    src/datasets.py
    src/model.py
    src/multiprocess.py
    src/prompts.py
    src/storage.py
    src/tokens.py
    src/utils.py
    scripts/generate_traces.py
    scripts/capture_activations.py
    scripts/check_traces.py
    scripts/validate.py
    scripts/verify_phase2.py
    tests/test_imports.py
    tests/test_config.py
    tests/test_tokens.py
    tests/test_storage.py
    tests/test_utils.py
    tests/test_prompts.py
    tests/test_multiprocess.py
  </files>
  <action>
    1. Move all 8 modules from src/infra/ to src/ using git mv:
       ```
       git mv src/infra/config.py src/config.py
       git mv src/infra/datasets.py src/datasets.py
       git mv src/infra/model.py src/model.py
       git mv src/infra/multiprocess.py src/multiprocess.py
       git mv src/infra/prompts.py src/prompts.py
       git mv src/infra/storage.py src/storage.py
       git mv src/infra/tokens.py src/tokens.py
       git mv src/infra/utils.py src/utils.py
       ```

    2. Remove the now-empty src/infra/ directory:
       ```
       rm src/infra/__init__.py
       rm -rf src/infra/__pycache__
       rmdir src/infra
       ```
       (Or `git rm src/infra/__init__.py` then `rm -rf src/infra/`)

    3. Find-and-replace ALL Python import references. The pattern is:
       - `from src.infra.X` -> `from src.X`
       - `import src.infra.X` -> `import src.X`
       - `src.infra.X` in docstrings/comments -> `src.X`

       Files to update (comprehensive list):

       **scripts/** (5 files):
       - scripts/generate_traces.py -- lines 27-43: all `from src.infra.X import` statements
       - scripts/capture_activations.py -- lines 27-31: all `from src.infra.X import` statements
       - scripts/check_traces.py -- lines 20-21: `from src.infra.config`, `from src.infra.tokens`
       - scripts/validate.py -- lines 28-53 (top-level imports) AND lines 476-481, 567 (lazy imports inside worker functions)
       - scripts/verify_phase2.py -- lines 31-33: `from src.infra.config`, `from src.infra.storage`, `from src.infra.tokens`

       **tests/** (7 files):
       - tests/test_imports.py -- lines 1, 9, 13, 17, 21, 25, 29, 33, 37, 41, 47 (imports AND docstring AND subprocess command string on line 47)
       - tests/test_config.py -- lines 1, 16 (docstring and import)
       - tests/test_tokens.py -- lines 1, 16 (docstring and import)
       - tests/test_storage.py -- lines 1, 19 (docstring and import)
       - tests/test_utils.py -- lines 1, 14 (docstring and import)
       - tests/test_prompts.py -- lines 1, 13 (docstring and import)
       - tests/test_multiprocess.py -- lines 1, 10 (docstring and import)

       IMPORTANT: In tests/test_imports.py line 47, there is a subprocess command STRING that also contains `src.infra`:
       ```python
       "from src.infra.config import load_config; from src.infra.tokens import parse_think_boundaries; print('OK')",
       ```
       This MUST also be updated to `from src.config ...` and `from src.tokens ...`.

       IMPORTANT: In scripts/validate.py, there are LAZY imports inside worker functions (lines 476-481 and 567). These are NOT at the top of the file. Search the ENTIRE file for `src.infra`.

    4. Do NOT touch files under archive/ -- those are historical and should keep their old import paths.

    5. Do NOT touch .planning/ docs in this task -- that is Task 2.
  </action>
  <verify>
    Run the full test suite:
    ```
    cd /home/aahmad/post-hoc-interp && uv run pytest tests/ -v
    ```
    All tests must pass. Also verify no stale references remain in Python files:
    ```
    grep -r "from src\.infra" scripts/ tests/ src/ --include="*.py"
    grep -r "import src\.infra" scripts/ tests/ src/ --include="*.py"
    ```
    Both grep commands should return zero results.

    Also verify src/infra/ directory is gone:
    ```
    ls src/infra/ 2>&1  # Should say "No such file or directory"
    ```

    Also do a quick import smoke test:
    ```
    uv run python -c "from src.config import load_config; from src.tokens import parse_think_boundaries; from src.model import load_model; print('OK')"
    ```
  </verify>
  <done>All 8 modules live at src/*.py. All scripts, tests, and src/ files use `from src.X import Y`. No `src.infra` references remain in any .py file outside archive/. All tests pass. src/infra/ directory no longer exists.</done>
</task>

<task type="auto">
  <name>Task 2: Update planning documentation references</name>
  <files>
    .planning/PROJECT.md
    .planning/codebase/CONVENTIONS.md
    .planning/codebase/CONCERNS.md
    .planning/codebase/STRUCTURE.md
    .planning/codebase/ARCHITECTURE.md
    .planning/codebase/TESTING.md
  </files>
  <action>
    Update all `.planning/` documentation files that reference `src/infra` or `src.infra` to reflect the new flat structure. Only update files that describe CURRENT codebase state -- do NOT update historical phase plan/summary/verification/research docs (those are historical records).

    Files to update and what to change:

    1. **.planning/PROJECT.md** (line 116):
       - `src/infra/` -> `src/` in workspace description
       - Change: "All new code in `src/infra/` (shared library)" -> "All new code in `src/` (shared library)"

    2. **.planning/codebase/CONVENTIONS.md**:
       - Line 28: `src/` root with subpackages `infra/`, ... -> remove `infra/` from subpackages list, note modules are directly in `src/`
       - Line 68: `from src.infra.logging import get_logger` -> This references the OLD training infra logging (now archived). Update the example to use a current module, e.g., `from src.config import load_config`
       - Line 90: `from src.infra.logging import ...` -> `from src.config import ...` (or similar current module)
       - Line 146: `src/infra/logging.py:90` -> `archive/src/infra/logging.py` (or note this is archived)
       - Line 174: `src/infra/logging.py:1-3` -> `archive/src/infra/logging.py`
       - Line 221-222: Update import examples from `from lib.config import load_config` to `from src.config import load_config` and similar

    3. **.planning/codebase/CONCERNS.md**:
       - Lines 20-21: `src/infra/logging.py` references -> `archive/src/infra/logging.py` (these concern the OLD archived logging module, not the moved modules)

    4. **.planning/codebase/STRUCTURE.md**:
       - Lines 93-102: The entire `src/infra/:` section needs restructuring. The research modules (config, datasets, model, multiprocess, prompts, storage, tokens, utils) are now directly under `src/`. The old training infra (logging, checkpointer, parallelisms, optimizers, train_iterator) is in `archive/src/infra/`.
       - Lines 165-166, 199, 225, 268, 278: Update all `src/infra/` path references to either `src/` (for research modules) or `archive/src/infra/` (for old training infra)

    5. **.planning/codebase/ARCHITECTURE.md**:
       - Lines 18-20: `src/infra/` -> `src/` for infrastructure layer description
       - Lines 56, 81, 87-99: Update all `src/infra/` references. For current research modules -> `src/`. For old training infra (logging, checkpointer, parallelisms) -> `archive/src/infra/`

    6. **.planning/codebase/TESTING.md**:
       - Lines 244-246: `src/infra/*` -> Note these are archived. Update file paths to `archive/src/infra/`

    Do NOT update these files (historical records):
    - .planning/phases/02.1-*/*.md (phase plans, summaries, verification, research, UAT)
    - .planning/ROADMAP.md (historical phase descriptions)
  </action>
  <verify>
    Verify no stale `src/infra` or `src.infra` references remain in the active planning docs:
    ```
    grep -r "src/infra\|src\.infra" .planning/PROJECT.md .planning/codebase/
    ```
    This should return zero results (or only results that explicitly say "archive/src/infra" for the old training modules).

    Manually spot-check that STRUCTURE.md accurately reflects the new directory layout (modules directly under src/).
  </verify>
  <done>All active planning documentation reflects the new `src/` flat structure. References to current modules use `src/X` paths. References to old training infra correctly point to `archive/src/infra/`. Historical phase docs are untouched.</done>
</task>

</tasks>

<verification>
1. `uv run pytest tests/ -v` -- all tests pass
2. `uv run python -c "from src.config import load_config; from src.model import load_model; from src.tokens import parse_think_boundaries; print('OK')"` -- prints OK
3. `grep -rn "from src\.infra\|import src\.infra" scripts/ tests/ src/ --include="*.py"` -- zero results
4. `ls src/infra/ 2>&1` -- "No such file or directory"
5. `ls src/config.py src/datasets.py src/model.py src/multiprocess.py src/prompts.py src/storage.py src/tokens.py src/utils.py` -- all exist
6. `grep -r "src/infra\|src\.infra" .planning/PROJECT.md .planning/codebase/` -- only "archive/src/infra" references (if any)
</verification>

<success_criteria>
All 8 modules accessible via `from src.X import Y`. All tests pass. No `src.infra` references remain in active code or active documentation. The `src/infra/` directory no longer exists. Git history preserved via `git mv`.
</success_criteria>

<output>
After completion, create `.planning/quick/001-move-src-infra-contents-to-src/001-SUMMARY.md`
</output>
