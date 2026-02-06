---
phase: 02-trace-generation-at-scale
plan: 03
subsystem: infra
tags: [quality-gate, csv-report, trace-validation, quarantine, diagnostics]

# Dependency graph
requires:
  - phase: 02-trace-generation-at-scale
    provides: "lib/tokens.py parse_think_boundaries, lib/config.py load_config, lib/storage.py path helpers, configs/generate.yaml traces_dir"
provides:
  - "check_traces.py quality gate script producing CSV diagnostic report"
  - "Objective checks: has_think_block, hit_max_tokens, degenerate_repetition"
  - "Warning flags: length_outlier, multi_block"
  - "Quarantine mode for moving failed traces from valid/ to failed/"
affects:
  - 02-04 (capture_activations.py only processes traces that pass quality gate)

# Tech tracking
tech-stack:
  added: [ruff]
  patterns:
    - "Two-mode script: diagnostic scan (default) vs. quarantine (--quarantine)"
    - "CSV report as intermediate artifact for human-in-the-loop review"
    - "Path-based dataset/problem_id extraction with metadata fallback"

key-files:
  created:
    - "src/experimental/gorn/scripts/check_traces.py"
  modified: []

key-decisions:
  - "Use pre-parser for --quarantine flag before load_config's argparse to avoid conflicts"
  - "Load generate.yaml programmatically (not via CLI) to get traces_dir path"
  - "multi_block warning derived from num_think_blocks column (not separate CSV column) to match plan spec"

patterns-established:
  - "Quality gate between pipeline stages: scan, report, human review, then quarantine"
  - "Degenerate repetition check: >50 consecutive identical tokens"
  - "Length outlier detection: flag traces outside [p5, p95] percentile range"

# Metrics
duration: 2min
completed: 2026-02-05
---

# Phase 2 Plan 03: Check Traces Quality Gate Summary

**Trace quality gate script with CSV diagnostic report, objective pass/fail checks (think blocks, max tokens, degenerate repetition), warning flags (length outliers, multi-block), and quarantine mode**

## Performance

- **Duration:** 2 min
- **Started:** 2026-02-05T08:28:56Z
- **Completed:** 2026-02-05T08:31:19Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created check_traces.py that scans all trace_*.json files in valid/ directory and produces a CSV quality report at {traces_dir}/quality_report.csv
- Implemented three objective checks (has_think_block via parse_think_boundaries, hit_max_tokens from metadata, degenerate_repetition via consecutive token count) and two warning flags (length_outlier via p5/p95 percentiles, multi_block)
- Summary output prints per-dataset pass/fail breakdown with length statistics (mean, std, p5, p95)
- Quarantine mode (--quarantine) reads the CSV report and moves failed traces from valid/ to failed/ with mirrored directory structure

## Task Commits

Each task was committed atomically:

1. **Task 1: Create check_traces.py quality gate script** - `adc6db6` (feat)

## Files Created/Modified

- `src/experimental/gorn/scripts/check_traces.py` - Quality gate script with scan mode (CSV report + summary) and quarantine mode (move failed traces)

## Decisions Made

- **Pre-parser for --quarantine:** Used a separate argparse pre-parser to handle the --quarantine flag before load_config's own argparse, avoiding conflicts between script-specific and config CLI arguments.
- **Programmatic config loading:** Loads generate.yaml via script_config parameter (not CLI --config) since check_traces.py always needs the same config as the generation pipeline.
- **CSV column spec:** multi_block is a warning computed and reported in summary output but not a separate CSV column, since it can be derived from num_think_blocks > 1 which is already in the CSV.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing ruff dependency**
- **Found during:** Task 1 (verification step)
- **Issue:** ruff was not installed in the project's virtualenv, blocking lint verification
- **Fix:** Ran `uv pip install ruff`
- **Files modified:** .venv (package installation only)
- **Verification:** `uv run ruff check` passes
- **Committed in:** N/A (package installation, not source code)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Trivial dependency installation. No scope creep.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Quality gate script ready for use after trace generation (02-02) completes
- Produces CSV report that can be reviewed before activation capture (02-04)
- Quarantine mode available for researcher-initiated cleanup of failed traces

---
*Phase: 02-trace-generation-at-scale*
*Completed: 2026-02-05*
