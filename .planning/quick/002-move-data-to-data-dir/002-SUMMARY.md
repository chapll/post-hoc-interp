---
phase: quick
plan: 002
subsystem: infra
tags: [omegaconf, env-var, data-paths, config]

# Dependency graph
requires:
  - phase: 02-trace-generation-at-scale
    provides: "generated trace data in data/traces/"
  - phase: 02.1-repo-restructure
    provides: "stable src/ layout and test suite"
provides:
  - "DATA_DIR env var interpolation in all config YAMLs"
  - "Trace data at /mnt/data/aahmad/data/traces/"
  - "test_data_dir_env_override test for CI safety"
affects: [03-probing, activation-capture, any-phase-using-config-paths]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "OmegaConf oc.env resolver for env-var config paths"
    - "Derived paths via ${paths.data_dir}/subdir interpolation"

key-files:
  created: []
  modified:
    - configs/base.yaml
    - configs/generate.yaml
    - configs/capture.yaml
    - tests/test_config.py
    - src/storage.py

key-decisions:
  - "Use ${oc.env:DATA_DIR,data} with 'data' fallback for local dev"
  - "Derive traces_dir and activations_dir from data_dir via interpolation"
  - "DATA_DIR corrected to /mnt/data/aahmad/data (not post-hoc-interp subdir)"

patterns-established:
  - "Env-var path resolution: all data paths resolve from DATA_DIR env var at runtime"
  - "Explicit env cleanup in tests: os.environ.pop('DATA_DIR', None) before path assertions"

# Metrics
duration: 4min
completed: 2026-02-18
---

# Quick Task 002: Move Data to DATA_DIR Summary

**OmegaConf env-var interpolation for data paths, trace data relocated to /mnt/data/aahmad/data/traces/**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-18T22:52:36Z
- **Completed:** 2026-02-18T22:56:42Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- All config YAMLs now resolve data_dir from DATA_DIR env var with "data" fallback
- 2400 trace files (52MB) physically relocated from ./data/traces/ to /mnt/data/aahmad/data/traces/
- Test suite hardened with explicit env-var cleanup and new DATA_DIR override test
- Full test suite passes: 81/81 tests green

## Task Commits

Each task was committed atomically:

1. **Task 1: Update config YAMLs to use DATA_DIR env var interpolation** - `ef9cda1` (feat)
2. **Task 2: Move physical data, update tests and docstring** - `4ac6096` (feat)

## Files Created/Modified
- `configs/base.yaml` - paths.data_dir now uses ${oc.env:DATA_DIR,data}
- `configs/generate.yaml` - traces_dir derives from ${paths.data_dir}/traces
- `configs/capture.yaml` - traces_dir and activations_dir derive from ${paths.data_dir}
- `tests/test_config.py` - 3 tests hardened with env cleanup, 1 new test added
- `src/storage.py` - Docstring updated to use {traces_dir}/{activations_dir} placeholders
- `.env` - DATA_DIR corrected to /mnt/data/aahmad/data (gitignored, not committed)

## Decisions Made
- Used `${oc.env:DATA_DIR,data}` OmegaConf resolver -- built-in, no custom resolver needed
- Corrected DATA_DIR from /mnt/data/aahmad/data/post-hoc-interp/ to /mnt/data/aahmad/data/ per user instruction
- Updated .env to match corrected DATA_DIR (gitignored, not in commits)

## Deviations from Plan

### Path Correction (User-Directed)

The plan specified `/mnt/data/aahmad/data/post-hoc-interp/` as DATA_DIR. Per user's critical correction, all paths were updated to use `/mnt/data/aahmad/data/` instead. This affected:
- Physical data move destination
- Test verification paths
- .env file value

### .env Update

Updated `.env` DATA_DIR value from `/mnt/data/aahmad/data/post-hoc-interp/` to `/mnt/data/aahmad/data` to match the corrected path. This file is gitignored so it does not appear in commits.

---

**Total deviations:** 1 user-directed path correction, 1 .env update
**Impact on plan:** Corrective -- ensures data paths match actual mount layout.

## Issues Encountered
None -- OmegaConf oc.env resolver worked out of the box, no custom registration needed.

## User Setup Required
None - DATA_DIR is already set in .env. SLURM scripts source .env automatically.

## Next Phase Readiness
- Config paths fully decoupled from repo directory
- Trace data at /mnt/data/aahmad/data/traces/valid/ with all 3 datasets (folio, gsm_symbolic, strategyqa)
- Ready for activation capture phase (will write to /mnt/data/aahmad/data/activations/)
- Ready for Phase 3 (probing and transition detection)

## Self-Check: PASSED

---
*Quick task: 002-move-data-to-data-dir*
*Completed: 2026-02-18*
