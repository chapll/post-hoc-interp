---
phase: quick-006
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - configs/per_token_probe.yaml
  - scripts/per_token_probe.py
  - scripts/plot_per_token_probe.py
  - slurm/per-token-probe.sbatch
autonomous: true

must_haves:
  truths:
    - "Config supports a traces list with 4 entries"
    - "per_token_probe.py loops over all traces, creating per-trace output subdirs"
    - "Traces with existing results.parquet are skipped (resume support)"
    - "Plot script discovers all trace subdirs and generates per-trace + multi-trace comparison plots"
    - "SLURM sbatch runs multi-trace config end-to-end"
  artifacts:
    - path: "configs/per_token_probe.yaml"
      provides: "Multi-trace configuration with 4 gsm_symbolic traces"
      contains: "traces:"
    - path: "scripts/per_token_probe.py"
      provides: "Multi-trace probing loop with resume support"
      contains: "for trace_cfg in traces"
    - path: "scripts/plot_per_token_probe.py"
      provides: "Multi-trace plotting with comparison overlay"
      contains: "multi_trace_comparison"
    - path: "slurm/per-token-probe.sbatch"
      provides: "Updated SLURM script for multi-trace run"
  key_links:
    - from: "configs/per_token_probe.yaml"
      to: "scripts/per_token_probe.py"
      via: "cfg.per_token_probe.traces list"
      pattern: "cfg\\.per_token_probe\\.traces"
    - from: "scripts/per_token_probe.py"
      to: "output subdirs"
      via: "os.path.join(output_dir, problem_id)"
      pattern: "os\\.path\\.join.*output_dir.*problem_id"
    - from: "scripts/plot_per_token_probe.py"
      to: "trace subdirs"
      via: "glob/listdir discovery of results.parquet files"
      pattern: "results\\.parquet"
---

<objective>
Extend per-token probing from single-trace to multi-trace. Update config to hold a list of 4 traces,
update the probing script to loop over traces with resume support, update the plot script to discover
all trace subdirs and generate a multi-trace comparison plot, and update the SLURM sbatch.

Purpose: Enable probing multiple traces in a single run to compare transition dynamics across different
problem types (fresh vs contaminated, different configs).

Output: Updated config, probing script, plot script, and SLURM sbatch -- all backward-compatible.
</objective>

<execution_context>
@/home/aahmad/.claude/get-shit-done/workflows/execute-plan.md
@/home/aahmad/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@configs/per_token_probe.yaml
@scripts/per_token_probe.py
@scripts/plot_per_token_probe.py
@slurm/per-token-probe.sbatch
</context>

<tasks>

<task type="auto">
  <name>Task 1: Multi-trace config and probing script</name>
  <files>
    configs/per_token_probe.yaml
    scripts/per_token_probe.py
  </files>
  <action>
**Config (configs/per_token_probe.yaml):**

Replace the single-trace fields (`problem_id`, `dataset`, `trace_index`, `expected_think_length`) with a
`traces` list. Keep the probing parameters (`num_probes`, `min_probe_spacing`, etc.) at the top level
since they apply to all traces. Remove `expected_think_length` entirely -- think_length will be read
dynamically from trace metadata.

The new `traces` list:

```yaml
per_token_probe:
  # Target traces
  traces:
    - problem_id: "gsm_symbolic_68_fresh_false_cfga"
      dataset: "gsm_symbolic"
      trace_index: 0
    - problem_id: "gsm_symbolic_56_original_false_cfgb"
      dataset: "gsm_symbolic"
      trace_index: 0
    - problem_id: "gsm_symbolic_81_original_false_cfga"
      dataset: "gsm_symbolic"
      trace_index: 0
    - problem_id: "gsm_symbolic_2_original_false_cfgb"
      dataset: "gsm_symbolic"
      trace_index: 0

  # Probing parameters (unchanged)
  min_probe_spacing: 1
  max_new_tokens: 15
  empty_cache_interval: 100

  # Token IDs (unchanged)
  close_token_ids: [151668, 198]
  rho_token_ids: [49, 6161]
  tau_token_id: 92713
```

Remove `num_probes: 824` from config -- it will be derived from `think_length` per trace (since each
trace has a different think_length). Keep `min_probe_spacing: 1` which ensures every position is probed.

**Script (scripts/per_token_probe.py):**

Restructure `main()` to loop over traces. The key changes:

1. **Parse traces list with backward compat.** After loading config, check if `cfg.per_token_probe.traces`
   exists (list). If not, fall back to reading the old single-trace fields (`problem_id`, `dataset`,
   `trace_index`) and wrapping them into a one-element list. This keeps old configs working.

   ```python
   ptp = cfg.per_token_probe
   if hasattr(ptp, "traces") and ptp.traces:
       traces = list(ptp.traces)
   else:
       # Backward compat: single-trace config
       traces = [{"problem_id": ptp.problem_id, "dataset": ptp.dataset, "trace_index": ptp.trace_index}]
   ```

2. **Load model ONCE before the loop.** Move model loading (Section 5) above the trace loop. The model
   and tokenizer are shared across all traces.

3. **Loop over traces.** For each trace entry in `traces`:

   a. Derive `trace_output_dir = os.path.join(output_dir, trace_cfg.problem_id)` (or `trace_cfg["problem_id"]`
      depending on whether OmegaConf returns DictConfig or dict).

   b. **Resume check:** If `os.path.join(trace_output_dir, "results.parquet")` already exists, log
      "Skipping {problem_id} -- results.parquet already exists" and `continue`.

   c. Load the trace via `load_single_trace()`.

   d. Compute `think_length` dynamically from `meta["think_boundaries"]` (same as current lines 133-134).
      Remove the `expected_think_length` assertion entirely. Just log the think_length.

   e. Set `num_probes = think_length` (probe every position since min_probe_spacing=1).

   f. Create cfg_shim with `num_probes=think_length` (not from config).

   g. Run the full probing loop (Sections 4-8 from current code). The probing loop body stays mostly
      the same -- it just runs inside the trace loop now.

   h. After probing a trace, save `results.parquet` and `results_summary.json` to `trace_output_dir`.
      In the summary JSON, use `trace_cfg.problem_id` / `trace_cfg.dataset` / `trace_cfg.trace_index`
      (not `ptp.problem_id` which no longer exists in multi-trace config).

   i. Delete the DynamicCache and call `torch.cuda.empty_cache()` between traces to avoid OOM buildup.
      The KV cache must be recreated (prefilled) for each new trace anyway.

4. **Important detail on KV cache lifecycle:** Each trace needs its own prefill. Create DynamicCache
   fresh inside the loop, prefill it, run descending probes, then delete it before the next trace.
   The model itself stays loaded.

5. **Update the docstring** to reflect multi-trace support.

6. **Update the summary log at the end** to show how many traces were processed vs skipped.
  </action>
  <verify>
    python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/per_token_probe.yaml')
traces = cfg.per_token_probe.traces
assert len(traces) == 4, f'Expected 4 traces, got {len(traces)}'
assert traces[0].problem_id == 'gsm_symbolic_68_fresh_false_cfga'
assert traces[3].problem_id == 'gsm_symbolic_2_original_false_cfgb'
assert not hasattr(cfg.per_token_probe, 'expected_think_length'), 'expected_think_length should be removed'
print('Config OK')
"

    python -c "
import ast, sys
with open('scripts/per_token_probe.py') as f:
    src = f.read()
# Check key patterns exist
assert 'traces' in src, 'Missing traces loop'
assert 'results.parquet' in src, 'Missing results output'
assert 'continue' in src, 'Missing skip/resume logic'
assert 'expected_think_length' not in src, 'expected_think_length should be removed'
# Verify it parses
ast.parse(src)
print('Script OK')
"
  </verify>
  <done>
    - Config has 4 traces in a `traces` list, no `expected_think_length`
    - Script loops over traces, creates per-trace subdirs, has resume support
    - Model loaded once, KV cache created per trace
    - Old single-trace configs still work via backward-compat fallback
  </done>
</task>

<task type="auto">
  <name>Task 2: Multi-trace plot script and SLURM update</name>
  <files>
    scripts/plot_per_token_probe.py
    slurm/per-token-probe.sbatch
  </files>
  <action>
**Plot script (scripts/plot_per_token_probe.py):**

Update `main()` to handle multi-trace directory structure where each trace's results live in
`$EXP_DIR/per_token_probe/{problem_id}/results.parquet`.

1. **Discovery logic.** When `--results-dir` points to the parent `per_token_probe/` directory:
   - Check if `results.parquet` exists directly in the dir (old single-trace layout). If so,
     treat as single-trace mode and run current logic unchanged.
   - Otherwise, scan for subdirectories containing `results.parquet`. Collect list of
     `(subdir_name, subdir_path)` pairs.

   ```python
   direct_parquet = os.path.join(results_dir, "results.parquet")
   if os.path.isfile(direct_parquet):
       # Single-trace mode (backward compat)
       trace_dirs = [("", results_dir)]
   else:
       # Multi-trace mode: discover subdirs
       trace_dirs = []
       for entry in sorted(os.listdir(results_dir)):
           subdir = os.path.join(results_dir, entry)
           if os.path.isdir(subdir) and os.path.isfile(os.path.join(subdir, "results.parquet")):
               trace_dirs.append((entry, subdir))
       if not trace_dirs:
           print(f"ERROR: No results.parquet found in {results_dir} or subdirs", file=sys.stderr)
           sys.exit(1)
   ```

2. **Per-trace plots.** For each discovered trace dir, generate the existing 3 plots
   (per_token_curve.png, comparison_overlay.png, transition_characterization.png) in that subdir.
   Refactor the existing main-body logic into a helper function like `process_single_trace(trace_dir, phase22_dir)`
   that contains the current load + plot logic.

3. **Multi-trace comparison plot.** After processing all individual traces, if there are 2+ traces,
   generate `multi_trace_comparison.png` in the parent `results_dir`. This is a NEW plot function:

   ```python
   def plot_multi_trace_comparison(all_trace_data, output_dir):
       """Overlay rolling-average accuracy curves from all traces on one figure.

       Args:
           all_trace_data: List of (problem_id, df) tuples, each df sorted by probe_position.
           output_dir: Parent directory where multi_trace_comparison.png is saved.
       """
   ```

   The plot should be a single-panel figure (not two-panel) showing:
   - X-axis: **normalized_position** (0 to 1) so traces with different think_lengths align
   - Y-axis: rolling-average accuracy (window=20)
   - One line per trace, each with a different color from a qualitative colormap (e.g., tab10)
   - Legend showing problem_id for each trace
   - Horizontal line at y=0.5
   - Title: "Multi-Trace Per-Token Accuracy Comparison"
   - Grid on, tight_layout

   Also generate a second comparison plot `multi_trace_probability.png` with the same layout
   but showing rolling-average `correct_token_prob` instead of accuracy.

4. **Print a summary table** at the end showing per-trace stats:
   ```
   === Multi-Trace Summary ===
   problem_id                              think_len  accuracy  shape
   gsm_symbolic_68_fresh_false_cfga        824        72.3%     oscillatory
   gsm_symbolic_56_original_false_cfgb     612        85.1%     gradual ramp
   ...
   ```

**SLURM (slurm/per-token-probe.sbatch):**

Update for multi-trace:
- Increase `--time` from `00:30:00` to `02:00:00` (4 traces, ~15 min each = ~60 min, with margin).
- Update the comment at the top to say "Per-token force-answer probing: multiple traces."
- Update the echo to say "Starting per-token probe (multi-trace)..."
- The actual command stays the same (`uv run python scripts/per_token_probe.py --config configs/per_token_probe.yaml`)
  since the script now loops internally.
- The plot command stays the same (`uv run python scripts/plot_per_token_probe.py`) since it auto-discovers.
  </action>
  <verify>
    python -c "
import ast
with open('scripts/plot_per_token_probe.py') as f:
    src = f.read()
ast.parse(src)
assert 'multi_trace_comparison' in src, 'Missing multi-trace comparison plot'
assert 'trace_dirs' in src or 'subdirs' in src or 'listdir' in src, 'Missing subdir discovery'
assert 'normalized_position' in src, 'Must use normalized_position for comparison'
print('Plot script OK')
"

    grep -q '02:00:00' slurm/per-token-probe.sbatch && echo 'SLURM time OK' || echo 'SLURM time FAIL'
    grep -q 'multi-trace' slurm/per-token-probe.sbatch && echo 'SLURM comment OK' || echo 'SLURM comment FAIL'
  </verify>
  <done>
    - Plot script auto-discovers trace subdirs with results.parquet
    - Per-trace plots generated in each subdir (same 3 plots as before)
    - multi_trace_comparison.png and multi_trace_probability.png generated in parent dir
    - Summary table printed to console
    - SLURM sbatch updated with 2h time limit and multi-trace description
    - Backward compatible with old single-trace results layout
  </done>
</task>

</tasks>

<verification>
1. Config parses and has 4 traces: `python -c "from omegaconf import OmegaConf; c=OmegaConf.load('configs/per_token_probe.yaml'); assert len(c.per_token_probe.traces)==4"`
2. Probing script has no syntax errors: `python -c "import ast; ast.parse(open('scripts/per_token_probe.py').read())"`
3. Plot script has no syntax errors: `python -c "import ast; ast.parse(open('scripts/plot_per_token_probe.py').read())"`
4. No references to `expected_think_length` remain: `! grep -r expected_think_length configs/ scripts/`
5. SLURM has correct time: `grep '02:00:00' slurm/per-token-probe.sbatch`
</verification>

<success_criteria>
- per_token_probe.py loops over 4 traces from config, with resume support (skip existing results.parquet)
- Each trace's results go to `$EXP_DIR/per_token_probe/{problem_id}/`
- plot_per_token_probe.py discovers all trace subdirs and generates per-trace + multi-trace comparison plots
- SLURM sbatch has sufficient time (2h) for multi-trace run
- All scripts parse without errors
- Backward compatibility preserved for old single-trace configs/results
</success_criteria>

<output>
After completion, create `.planning/quick/006-multi-trace-per-token-probe/006-SUMMARY.md`
</output>
