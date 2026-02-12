# Codebase Concerns

**Analysis Date:** 2026-02-06

## Tech Debt

**Configuration Portability:**
- Issue: Multiple configuration systems overlap (Hydra for training infra in `configs/`, OmegaConf for research pipelines in `src/experimental/gorn/configs/`). Environment variables via `.env` files are required but `.env` itself is gitignored, forcing each user to manually recreate from `.env.example`.
- Files: `configs/config.yaml`, `configs/deploy/submitit_example.yaml`, `src/experimental/gorn/configs/base.yaml`, `.env.example`
- Impact: New team members must manually configure paths before any code runs. Different users may have inconsistent configurations. Config drift between environments creates "works on my machine" bugs.
- Fix approach: Add setup validation script that checks for `.env` file presence and required variables. Document minimum required env vars more prominently in README. Consider defaulting more values to reduce manual setup friction.

**Hardcoded GPU Assumptions:**
- Issue: Code requires CUDA with no CPU fallback (intentional per `src/experimental/gorn/lib/utils.py:43-46`). GPU count and availability checks exist but no graceful degradation for limited hardware.
- Files: `src/experimental/gorn/lib/utils.py` (lines 31-53), `src/experimental/gorn/lib/model.py` (lines 40-49)
- Impact: Users without GPUs cannot run validation or unit tests. Different GPU availability across team members (Ali: B200, Gorn: A100, Eric: limited) means code that works for Ali may OOM for Eric.
- Fix approach: Add CPU-only unit tests for non-model logic (dataset loaders, storage, token parsing). Gate GPU-requiring tests with environment checks. Document minimum GPU requirements per operation (validation: 1x40GB, full trace gen: 4x80GB).

**Hardcoded Pacific Timezone:**
- Issue: Logging timestamps hardcoded to "America/Los_Angeles" timezone in `src/infra/logging.py:21` and `src/infra/logging.py:117`.
- Files: `src/infra/logging.py` (lines 21, 117)
- Impact: Log timestamps may be confusing for team members in other timezones. Not critical but reduces usability for distributed teams.
- Fix approach: Read timezone from environment variable with LA as default, or use UTC for all logs.

**User-Specific Workspaces Are Stubs:**
- Issue: `src/experimental/ali/` and `src/experimental/eric/` exist but contain only placeholder READMEs (ali/REAMDME.md is even misspelled). Only `src/experimental/gorn/` has actual implementation.
- Files: `src/experimental/ali/REAMDME.md`, `src/experimental/eric/README.md`
- Impact: Misleading directory structure suggests multi-user isolation that doesn't exist. If Ali or Eric add code to their directories, path assumptions in `sys.path.insert(0, "src/experimental/gorn")` may break.
- Fix approach: Either populate all user directories with proper READMEs explaining ownership, or remove unused directories. Document that current work lives exclusively in `src/experimental/gorn/`.

**Path Injection Pattern:**
- Issue: Scripts use `sys.path.insert(0, "src/experimental/gorn")` to enable imports, but this only works if launched from repo root. Breaks if scripts are run from other directories or if package structure changes.
- Files: `src/experimental/gorn/scripts/validate.py:28`, `src/experimental/gorn/scripts/generate_traces.py:28`, `src/experimental/gorn/scripts/capture_activations.py:28`
- Impact: Scripts fail with confusing import errors if not run from correct directory. Fragile to repo refactoring.
- Fix approach: Use proper Python package installation (`pip install -e .` or `uv pip install -e .`) and absolute imports. Remove path injection.

**Seed 42 Prohibition:**
- Issue: Code explicitly rejects seed value 42 with ValueError in `src/experimental/gorn/lib/utils.py:20-21`. No explanation in code or docs.
- Files: `src/experimental/gorn/lib/utils.py` (lines 20-21)
- Impact: Confusing error for users who habitually use seed=42. Appears to be an inside joke or arbitrary constraint.
- Fix approach: Either remove the restriction or document the rationale. If it's for reproducibility testing (ensuring users changed defaults), explain in error message.

## Known Bugs

**Misspelled File:**
- Symptoms: File named `REAMDME.md` instead of `README.md` in `src/experimental/ali/`
- Files: `src/experimental/ali/REAMDME.md`
- Trigger: Always present
- Workaround: None needed, file is just a placeholder stub

**Missing .env File Fails Silently:**
- Symptoms: Config system falls back to defaults (`./exp`, `./data`) if `.env` file is missing. Scripts proceed but write to wrong locations or fail with permission errors.
- Files: `configs/config.yaml` (lines 12-14)
- Trigger: Running any script without creating `.env` file first
- Workaround: Copy `.env.example` to `.env` as documented in README

## Security Considerations

**Environment File Contains Paths Only:**
- Risk: `.env.example` shows structure but actual `.env` is gitignored. Low security risk since it only contains paths and Slurm account names, not API keys or credentials.
- Files: `.env.example`, `.gitignore` (line 142)
- Current mitigation: `.env` is in `.gitignore`. No sensitive credentials detected in codebase.
- Recommendations: Document that `.env` should never contain API keys or tokens. If WandB API keys are added later, use separate secrets management.

**Submitit Config Files Are Gitignored:**
- Risk: Personal Slurm configs (`configs/deploy/submitit_*.yaml`) are gitignored except the example. Could contain Slurm account names or partition names that reveal cluster infrastructure.
- Files: `.gitignore` (lines 217-218), `configs/deploy/submitit_example.yaml`
- Current mitigation: Pattern-based gitignore prevents accidental commits of personal configs.
- Recommendations: Current approach is reasonable. Ensure documentation emphasizes not to commit personal Slurm configs with hardcoded paths.

## Performance Bottlenecks

**OOM on Long Traces:**
- Problem: Full 10-layer residual stream capture of 3500+ token sequences exhausts 80GB on A100. Documented in `.planning/STATE.md:84` and multiple planning docs.
- Files: `src/experimental/gorn/lib/model.py`, activation capture logic
- Cause: Saving all 36 layers of activations for long sequences keeps tensors in GPU memory until trace context exits.
- Improvement path: Current mitigation uses chunked capture (12 layers per pass, 3 passes). Alternative: immediately offload to CPU with `.cpu()` after each layer, or use selective layer saving (configurable layer subset).

**Sequential Trace Processing:**
- Problem: Each worker processes traces sequentially (batch_size=1) per `src/experimental/gorn/scripts/generate_traces.py` logic. Cannot batch multiple problems due to variable generation lengths.
- Files: `src/experimental/gorn/scripts/generate_traces.py` (lines 86-179)
- Cause: Thinking-mode generation produces variable-length outputs (100-3500+ tokens). Batching would require padding to max length, wasting compute.
- Improvement path: Current approach (multi-GPU parallelism with one model per GPU) is appropriate for variable-length generation. No obvious optimization without compromising memory or correctness.

**Verification Scripts Are 600-900 Lines:**
- Problem: `verify_phase2.py` (902 lines) and `validate.py` (690 lines) are large monolithic scripts that perform many checks sequentially. Long runtime for full verification.
- Files: `src/experimental/gorn/scripts/verify_phase2.py`, `src/experimental/gorn/scripts/validate.py`
- Cause: Comprehensive validation across multiple dimensions (completeness, metadata, think boundaries, contamination, seeds, quality, activations, disk usage). Each check reads many files from disk.
- Improvement path: Verification is inherently I/O-bound (reading thousands of trace files). Could parallelize by partitioning checks across multiple processes, but added complexity may not be worth speedup for occasional validation runs.

## Fragile Areas

**Multi-GPU Process Spawning:**
- Files: `src/experimental/gorn/lib/multiprocess.py` (lines 200-304)
- Why fragile: Depends on `mp.get_context("spawn")` for CUDA compatibility (line 263). Uses independent processes with manual error collection. If one worker crashes, others continue but failure detection relies on exit codes.
- Safe modification: Always test multi-GPU code paths with deliberate worker failures. Add integration tests that inject errors and verify proper error propagation. Never modify spawn context to "fork" on Linux with CUDA.
- Test coverage: Validated in Phase 1 per `01-VERIFICATION.md` but no automated regression tests.

**Think Token Boundary Parsing:**
- Files: `src/experimental/gorn/lib/tokens.py` (lines 1-125)
- Why fragile: Depends on hardcoded token IDs (151667 for `<think>`, 151668 for `</think>`). Assumes single contiguous think block per trace for primary analysis. Multi-block handling requires fallback logic.
- Safe modification: Always validate token IDs against tokenizer before deployment. Add tests for edge cases (empty think blocks, nested markers, truncated generation). Verify token IDs match Qwen3-8B tokenizer version.
- Test coverage: Token parsing validated extensively in Phase 1, but token ID changes in future Qwen models would break silently.

**nnsight Layer Access Pattern:**
- Files: `src/experimental/gorn/lib/model.py` (lines 130-180)
- Why fragile: Uses `.output` (not `.output[0]`) to get layer activations with shape `[batch, seq_len, hidden]`. Documented in `STATE.md:58` as critical decision. Alternative access patterns produce wrong shapes.
- Safe modification: Never change layer access pattern without validating tensor shapes. Add assertion checks for expected shapes immediately after capture. Document that nnsight API may differ across versions.
- Test coverage: Validated in Phase 1, but brittle to nnsight version updates.

**BF16 Numerical Stability:**
- Files: `src/experimental/gorn/lib/storage.py` (lines 100-150), entropy computation
- Why fragile: Softmax computed in FP32 then cast to BF16 for numerical stability per `STATE.md:52`. Direct BF16 softmax causes numerical errors in entropy calculation.
- Safe modification: Never remove FP32 cast in entropy/softmax computation. Add unit tests comparing FP32 vs BF16 entropy for known distributions. Document precision requirements.
- Test coverage: Validated during Phase 1 but no automated precision tests.

## Scaling Limits

**Disk Storage at Scale:**
- Current capacity: Estimated 1-2 TB for full dataset (2000 traces × 36 layers × safetensors format). Documented concern in `STATE.md:85`.
- Limit: Depends on cluster storage quota per user. Not documented.
- Scaling path: Use selective layer saving (only layers 0,4,8,12,16,20,24,28,32,35 per config). Compress old traces after analysis. Store only final results for paper. Add disk quota checks to scripts before starting large jobs.

**4-GPU Parallelism Ceiling:**
- Current capacity: Pipeline designed for 4 GPUs (Ali's config, Gorn's A100s). Validated at 4 GPU scale in Phase 1.
- Limit: Linear scaling up to GPU count, but no tested path beyond 8 GPUs. Multi-node not implemented.
- Scaling path: Current data-parallel approach (independent model copies) scales to arbitrary GPU count within single node. Multi-node would require job scheduling changes and shared storage validation.

**RAM for Dataset Loading:**
- Current capacity: Three datasets (GSM-Symbolic, FOLIO, StrategyQA) loaded into memory per `src/experimental/gorn/lib/datasets.py`. Small datasets (300 problems each × 2 configs = 1200 items total).
- Limit: Dataset loading is not a bottleneck at current scale. Could become issue if expanding to full GSM-Symbolic (11.5K problems) or adding more datasets.
- Scaling path: Use HuggingFace datasets streaming mode or lazy loading if memory becomes constraint.

## Dependencies at Risk

**nnsight Version Sensitivity:**
- Risk: nnsight 0.5.15 is specified in `pyproject.toml:13`. Library is under active development. API changes in future versions could break layer access patterns or trace generation.
- Impact: Activation capture would fail with shape mismatches or attribute errors.
- Migration plan: Pin nnsight version strictly. Test against new versions in isolated environment before upgrading. Have fallback to direct HuggingFace hooks if nnsight becomes unmaintained.

**PyTorch 2.9.0 Pinned:**
- Risk: `pyproject.toml:20` pins torch==2.9.0 exactly. Strict pin prevents security patches and may conflict with other dependencies that require newer PyTorch.
- Impact: Cannot use newer PyTorch features or bug fixes without testing full pipeline compatibility.
- Migration plan: Current pin is intentional for reproducibility. Document upgrade testing protocol: validate model loading, generation, activation capture, and BF16 behavior on new versions before updating pin.

**Python 3.13 Requirement:**
- Risk: `pyproject.toml:6` requires Python >=3.13. Very recent Python version (released Oct 2024). Some HPC clusters may not have 3.13 available yet.
- Impact: Eric or new team members on older clusters may not be able to run code without building Python from source.
- Migration plan: Test compatibility with Python 3.11+ and relax requirement if possible. If 3.13-specific features are used, document them and assess whether they're necessary.

## Missing Critical Features

**No Automated Tests:**
- Problem: No test files detected (no `test_*.py` or `*_test.py` files in experimental code). Only validation scripts that require full GPU setup.
- Blocks: Regression detection, refactoring safety, CI/CD integration
- Priority: High - Makes codebase fragile to changes and hard to validate incrementally.

**No Setup Validation:**
- Problem: No script to verify environment setup (GPU availability, required env vars, dependency versions, disk quota).
- Blocks: New user onboarding, detecting environment issues before long-running jobs fail
- Priority: Medium - Would significantly improve multi-user experience.

**No Compute Environment Documentation:**
- Problem: GPU requirements documented in planning docs but not in user-facing README. Different hardware available per user (Ali: B200, Gorn: A100, Eric: limited) not captured in setup docs.
- Blocks: Setting realistic expectations for new users, job sizing decisions
- Priority: Medium - Critical for onboarding but low urgency for current 3-person team.

**No Path Portability Between Users:**
- Problem: Data paths in `src/experimental/gorn/configs/base.yaml:30-31` are relative to repo root (`src/experimental/gorn/data`). Works if all users structure identically, but `.env` approach suggests per-user paths should be supported.
- Blocks: Users cannot easily point to different data locations without editing configs
- Priority: Low - Current team uses consistent structure, but limits flexibility.

## Test Coverage Gaps

**Dataset Loaders Not Unit Tested:**
- What's not tested: `src/experimental/gorn/lib/datasets.py` (504 lines) has no unit tests. Balance checking, distractor generation, and dataset-specific parsing only validated by full pipeline runs.
- Files: `src/experimental/gorn/lib/datasets.py`
- Risk: Bugs in distractor generation or label balance could silently corrupt training data for probes. Dataset loading errors only caught after expensive trace generation completes.
- Priority: High - Data correctness is critical for research validity.

**Multiprocessing Error Handling Not Tested:**
- What's not tested: Isolated worker failure modes in `run_parallel_isolated` (`src/experimental/gorn/lib/multiprocess.py:200-304`). Exit code propagation and partial success handling not validated with injected errors.
- Files: `src/experimental/gorn/lib/multiprocess.py`
- Risk: Worker crashes might not be detected correctly. Failed traces could be silently missed in large-scale runs.
- Priority: Medium - Manual validation in Phase 1 but no automated regression tests.

**Storage Format Edge Cases Not Tested:**
- What's not tested: Safetensors save/load with edge cases (empty traces, NaN values, mismatched dtypes). Only validated with successful generation runs.
- Files: `src/experimental/gorn/lib/storage.py`
- Risk: Corrupted traces might not be detected until analysis phase. Loading failures could be confusing.
- Priority: Low - Format is robust and validated at scale, but automated tests would catch regressions.

**Config Override System Not Tested:**
- What's not tested: OmegaConf config loading with CLI overrides (`src/experimental/gorn/lib/config.py`). Validation script tests successful cases but not malformed inputs or conflicting overrides.
- Files: `src/experimental/gorn/lib/config.py`
- Risk: Bad config overrides could cause jobs to run with wrong parameters.
- Priority: Low - Config errors are usually caught early, but better error messages would help.

## Multi-User Friction Points

**No .env File by Default:**
- Problem: Every new team member must manually create `.env` from `.env.example`. Easy to forget or misconfigure. Failure mode is confusing (scripts use wrong paths).
- Friction: High for first-time setup, zero after initial config
- Recommendation: Add check in main scripts that errors clearly if `.env` is missing. Consider setup wizard script.

**GPU Assignment Requires Manual Coordination:**
- Problem: Scripts take `--gpus` flag or read from config, but no mechanism to detect which GPUs are available or in-use by other users. Ali on B200, Gorn on A100, Eric on limited compute must manually coordinate to avoid collisions.
- Friction: Medium - Requires Slack/manual coordination before launching jobs
- Recommendation: Add GPU utilization check (via `nvidia-smi`) with warning if requested GPUs are >50% utilized. Document GPU ownership/reservation protocol.

**Data Directory Isolation Unclear:**
- Problem: `data_dir` in `.env` can be per-user or shared. No documentation on which approach is intended. Shared data saves space, per-user data prevents collisions but wastes disk.
- Friction: Medium - Team must decide convention and communicate it out-of-band
- Recommendation: Document recommended data organization strategy. Add example for shared data setup with user-specific output subdirectories.

**Planning Docs Tracked in Git:**
- Problem: `.planning/` directory is committed to git (line 224 in `.gitignore` is commented out). Contains user-specific execution state (`STATE.md`), session notes, and temporary planning artifacts. Multiple users pushing to same branch could create conflicts.
- Friction: Low currently (single-user workflow), High if team parallelizes work
- Recommendation: Decide if planning docs should be gitignored or treated as shared docs. If shared, establish protocol for STATE.md updates (e.g., per-user state files).

**No Shared Result Repository:**
- Problem: Each user's trace outputs go to their own `data_dir`. No established protocol for sharing generated traces across team. If Eric wants to analyze traces Ali generated, must coordinate manual file transfer.
- Friction: Medium - Limits collaboration and wastes compute duplicating trace generation
- Recommendation: Document shared data organization strategy. Consider rsync scripts or shared cluster storage with appropriate permissions.

**Config Templates Don't Cover All Hardware:**
- Problem: Example configs show 8-GPU setup (`configs/deploy/submitit_example.yaml:28`), but Eric has limited compute. No example config for small-scale runs (1-2 GPUs).
- Friction: Medium - Users without full resources must figure out correct settings by trial-and-error
- Recommendation: Add example configs for common hardware profiles (1-GPU laptop, 2-GPU workstation, 4-GPU cluster, 8-GPU HPC). Document memory requirements per setup.

---

*Concerns audit: 2026-02-06*
