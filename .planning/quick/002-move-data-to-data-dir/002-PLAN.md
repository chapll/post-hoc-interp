---
phase: quick
plan: 002
type: execute
wave: 1
depends_on: []
files_modified:
  - configs/base.yaml
  - configs/generate.yaml
  - configs/capture.yaml
  - tests/test_config.py
  - src/storage.py
autonomous: true

must_haves:
  truths:
    - "OmegaConf resolves paths.data_dir from DATA_DIR env var with fallback to 'data'"
    - "paths.traces_dir and paths.activations_dir derive from paths.data_dir"
    - "Trace data physically exists at /mnt/data/aahmad/data/post-hoc-interp/traces/"
    - "All tests pass with both DATA_DIR set and unset"
  artifacts:
    - path: "configs/base.yaml"
      provides: "data_dir using oc.env interpolation"
      contains: "oc.env:DATA_DIR"
    - path: "configs/generate.yaml"
      provides: "traces_dir derived from paths.data_dir"
      contains: "paths.data_dir"
    - path: "configs/capture.yaml"
      provides: "traces_dir and activations_dir derived from paths.data_dir"
      contains: "paths.data_dir"
  key_links:
    - from: "configs/generate.yaml"
      to: "configs/base.yaml"
      via: "OmegaConf interpolation ${paths.data_dir}"
      pattern: "\\$\\{paths\\.data_dir\\}"
    - from: "configs/capture.yaml"
      to: "configs/base.yaml"
      via: "OmegaConf interpolation ${paths.data_dir}"
      pattern: "\\$\\{paths\\.data_dir\\}"
---

<objective>
Move trace data from ./data/ to the DATA_DIR location (/mnt/data/aahmad/data/post-hoc-interp/)
and update all YAML configs to resolve paths.data_dir from the DATA_DIR environment variable
using OmegaConf interpolation, with a fallback to "data" for local dev.

Purpose: Decouple data storage from the repo directory so large trace files live on
the shared data mount, not in the working tree.

Output: Updated configs, updated tests, physically relocated trace data.
</objective>

<execution_context>
@/home/aahmad/.claude/get-shit-done/workflows/execute-plan.md
@/home/aahmad/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@configs/base.yaml
@configs/generate.yaml
@configs/capture.yaml
@src/config.py
@src/storage.py
@tests/test_config.py
@.env
</context>

<tasks>

<task type="auto">
  <name>Task 1: Update config YAMLs to use DATA_DIR env var interpolation</name>
  <files>configs/base.yaml, configs/generate.yaml, configs/capture.yaml</files>
  <action>
Update configs/base.yaml line 30:
  BEFORE: `data_dir: "data"`
  AFTER:  `data_dir: "${oc.env:DATA_DIR,data}"`

This uses OmegaConf's built-in env resolver. When DATA_DIR is set in the environment,
it resolves to that value. When unset, falls back to "data" (preserving local dev behavior).

Update configs/generate.yaml line 22:
  BEFORE: `traces_dir: "data/traces"`
  AFTER:  `traces_dir: "${paths.data_dir}/traces"`

This derives traces_dir from the resolved data_dir, so it automatically points to
DATA_DIR/traces when DATA_DIR is set, or data/traces when unset.

Update configs/capture.yaml lines 13-14:
  BEFORE:
    `traces_dir: "data/traces"`
    `activations_dir: "data/activations"`
  AFTER:
    `traces_dir: "${paths.data_dir}/traces"`
    `activations_dir: "${paths.data_dir}/activations"`

Same derivation pattern as generate.yaml.

Do NOT touch paths.experiments_dir in base.yaml (leave as "experiments").
  </action>
  <verify>
Run: `cd /home/aahmad/post-hoc-interp && python -c "
from src.config import load_config
import os

# Test 1: Without DATA_DIR - should fall back to 'data'
os.environ.pop('DATA_DIR', None)
cfg = load_config(script_config='configs/generate.yaml')
assert cfg.paths.data_dir == 'data', f'Expected data, got {cfg.paths.data_dir}'
assert cfg.paths.traces_dir == 'data/traces', f'Expected data/traces, got {cfg.paths.traces_dir}'
print('Fallback OK: data_dir=data, traces_dir=data/traces')

# Test 2: With DATA_DIR set - should resolve to env value
os.environ['DATA_DIR'] = '/mnt/data/aahmad/data/post-hoc-interp'
cfg2 = load_config(script_config='configs/capture.yaml')
assert cfg2.paths.data_dir == '/mnt/data/aahmad/data/post-hoc-interp', f'Got {cfg2.paths.data_dir}'
assert cfg2.paths.traces_dir == '/mnt/data/aahmad/data/post-hoc-interp/traces', f'Got {cfg2.paths.traces_dir}'
assert cfg2.paths.activations_dir == '/mnt/data/aahmad/data/post-hoc-interp/activations', f'Got {cfg2.paths.activations_dir}'
print('Env override OK: all paths resolve from DATA_DIR')
os.environ.pop('DATA_DIR', None)
print('All config interpolation checks passed.')
"`
  </verify>
  <done>
All three config files use OmegaConf interpolation. data_dir resolves from DATA_DIR env var
with "data" fallback. traces_dir and activations_dir derive from data_dir. Verified by
loading configs with and without DATA_DIR set.
  </done>
</task>

<task type="auto">
  <name>Task 2: Move physical data, update tests and docstring</name>
  <files>tests/test_config.py, src/storage.py</files>
  <action>
**Step A: Move trace data to DATA_DIR location.**

Create the target directory and copy trace data:
```
mkdir -p /mnt/data/aahmad/data/post-hoc-interp/traces
cp -r ./data/traces/valid /mnt/data/aahmad/data/post-hoc-interp/traces/
```

Use `cp` first (not `mv`) so we can verify before deleting the local copy.
After verifying the copy succeeded (check a few files exist and sizes match),
remove the local ./data/ directory:
```
rm -rf ./data/
```

**Step B: Update tests/test_config.py.**

The test `test_load_base_config_has_paths` on line 28 asserts:
  `assert cfg.paths.data_dir == "data"`

This still works because the fallback is "data" when DATA_DIR is unset.
However, the test is fragile -- if CI sets DATA_DIR, it would break.
Make the test explicit: unset DATA_DIR before loading config.

Update the test to:
```python
def test_load_base_config_has_paths():
    """base.yaml resolves paths.data_dir from DATA_DIR env var, falling back to 'data'."""
    os.environ.pop("DATA_DIR", None)
    cfg = load_config(script_config="")
    assert cfg.paths.data_dir == "data"
    assert cfg.paths.experiments_dir == "experiments"
```

Similarly, update `test_load_config_nonexistent_script_config` (line 60) which also
asserts `cfg.paths.data_dir == "data"`:
```python
def test_load_config_nonexistent_script_config():
    """Non-existent script_config path should still return base config."""
    os.environ.pop("DATA_DIR", None)
    cfg = load_config(script_config="/nonexistent/path.yaml")
    assert isinstance(cfg, DictConfig)
    assert cfg.paths.data_dir == "data"
```

And update `test_load_with_script_config_merges` (line 46) which asserts
`cfg.paths.traces_dir == "data/traces"`:
```python
def test_load_with_script_config_merges():
    """Merging generate.yaml over base.yaml should add generation-specific keys."""
    os.environ.pop("DATA_DIR", None)
    cfg = load_config(script_config="configs/generate.yaml")
    assert isinstance(cfg, DictConfig)
    assert cfg.paths.traces_dir == "data/traces"
    assert cfg.model.max_new_tokens == 4096
```

Add a NEW test that verifies DATA_DIR override works:
```python
def test_data_dir_env_override():
    """Setting DATA_DIR env var overrides paths.data_dir and derived paths."""
    os.environ["DATA_DIR"] = "/tmp/test-data"
    try:
        cfg = load_config(script_config="configs/generate.yaml")
        assert cfg.paths.data_dir == "/tmp/test-data"
        assert cfg.paths.traces_dir == "/tmp/test-data/traces"
    finally:
        os.environ.pop("DATA_DIR", None)
```

**Step C: Update src/storage.py module docstring.**

Update lines 8-9 of the docstring from:
```
- Traces (JSON): data/traces/{valid,failed}/{dataset}/{problem_id}/trace_NN.json
- Activations (safetensors): data/activations/{dataset}/{problem_id}/trace_NN.safetensors
```
to:
```
- Traces (JSON): {traces_dir}/{valid,failed}/{dataset}/{problem_id}/trace_NN.json
- Activations (safetensors): {activations_dir}/{dataset}/{problem_id}/trace_NN.safetensors
```

This removes the hardcoded "data/" prefix since the actual paths now come from config.
  </action>
  <verify>
1. Verify data was copied:
   `ls /mnt/data/aahmad/data/post-hoc-interp/traces/valid/gsm_symbolic/ | head -5`
   (should show problem directories)

2. Verify local data is removed:
   `ls ./data/ 2>&1` should show "No such file or directory"

3. Run the test suite:
   `cd /home/aahmad/post-hoc-interp && python -m pytest tests/test_config.py -v`
   All tests must pass, including the new test_data_dir_env_override.
  </verify>
  <done>
Trace data physically lives at /mnt/data/aahmad/data/post-hoc-interp/traces/valid/
with folio, gsm_symbolic, strategyqa subdirs intact. Local ./data/ removed.
All config tests pass with explicit env var handling. storage.py docstring updated.
  </done>
</task>

</tasks>

<verification>
1. Config interpolation: `python -c "from src.config import load_config; import os; os.environ.pop('DATA_DIR', None); cfg = load_config(script_config='configs/capture.yaml'); print(cfg.paths)"`
   -- Should show data_dir=data, traces_dir=data/traces, activations_dir=data/activations

2. Full test suite: `python -m pytest tests/ -v`
   -- All tests pass

3. Data location: `du -sh /mnt/data/aahmad/data/post-hoc-interp/traces/`
   -- Should show ~52M

4. No local data: `test -d ./data && echo "FAIL: local data still exists" || echo "OK: local data removed"`
</verification>

<success_criteria>
- configs/base.yaml uses `${oc.env:DATA_DIR,data}` for paths.data_dir
- configs/generate.yaml and capture.yaml derive paths from `${paths.data_dir}`
- Trace data exists at /mnt/data/aahmad/data/post-hoc-interp/traces/valid/
- Local ./data/ directory is removed
- All tests pass (including new env override test)
- No hardcoded "data/" prefix in config path values (except as OmegaConf fallback)
</success_criteria>

<output>
After completion, create `.planning/quick/002-move-data-to-data-dir/002-SUMMARY.md`
</output>
