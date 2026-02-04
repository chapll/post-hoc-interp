# Codebase Concerns

**Analysis Date:** 2026-02-03

## Tech Debt

**Unreachable Code in Mesh Initialization:**
- Issue: Lines 81-83 in `src/infra/parallelisms.py` are unconditionally overwritten by lines 85-89, making the initial mesh/names assignment dead code.
- Files: `src/infra/parallelisms.py` (lines 81-89)
- Impact: Confusing logic flow. The `dp_sharding * dp_replicate * tp_sharding == 1` branch is never used, suggesting incomplete refactoring or copy-paste error from original source.
- Fix approach: Remove the dead block (lines 81-83) or clarify intent if this is intentional fallback logic. Currently `get_world_mesh()` will crash if called with all factors = 1 because `mesh` and `names` are reset to empty lists.

**Silent Fallback in Checkpointing:**
- Issue: `load_pretrained()` in `src/infra/checkpointer.py` (lines 55-61) silently catches `RuntimeError` and retries with a different state dict format instead of propagating the error.
- Files: `src/infra/checkpointer.py` (lines 51-63)
- Impact: Masks checkpoint loading failures. If the actual problem is incompatible checkpoint format or corrupted file, the user gets no diagnostic information. The retry may fail silently or with a different error further down the stack.
- Fix approach: Log the original error before retry, or fail loudly if both attempts fail. Per CLAUDE.md strict rules, errors should fail loudly so they can be traced.

**Missing Validation in Config:**
- Issue: `config.py` has no defaults or validation for critical distributed training parameters (e.g., `parallelism.tp_sharding`, `parallelism.dp_sharding`, `parallelism.dp_replicate` default to 1, and there are no runtime checks that they compose correctly).
- Files: `src/config.py` (lines 48-52)
- Impact: Bad configurations can propagate to `get_world_mesh()` where assertions fail. Errors occur late in initialization rather than at config parse time.
- Fix approach: Add runtime validation in config post-hook or in `apply_parallelisms()` before mesh creation.

## Known Bugs

**Timeout Threshold Calculation Off by 6 Minutes:**
- Symptoms: When approaching timeout, the effective threshold calculation subtracts 6 extra minutes (`timedelta(minutes=6)`) in addition to average iteration time in `src/infra/train_iterator.py` line 134.
- Files: `src/infra/train_iterator.py` (line 134)
- Trigger: Any training run with `timeout_minutes` > 0 and sufficient iterations to calculate `avg_iter_time`.
- Workaround: Adjust `timeout_minutes` config value to compensate (add 6 minutes to desired threshold), though this is fragile.
- Note: Comment says "Timeout minutes - (average iteration time + 2 minutes)" but code subtracts 6 minutes, not 2. Off-by-four discrepancy.

**Distributed Barrier Always Uses Current Device:**
- Symptoms: `end_distributed()` in `src/infra/parallelisms.py` line 45 calls `torch.distributed.barrier(device_ids=[torch.cuda.current_device()])` which may fail or hang if device is not properly set on all ranks.
- Files: `src/infra/parallelisms.py` (line 45)
- Trigger: Multi-node training where `LOCAL_RANK` environment variable is not properly set before `init_distributed()` or if CUDA devices are not synchronized across ranks.
- Workaround: Ensure `init_distributed()` is always called before `end_distributed()`.

**Checkpoint Step Parsing Assumes Single Dash:**
- Symptoms: `get_latest_checkpoint_step()` in `src/infra/logging.py` line 174 splits on "-" and takes index [1], but checkpoint dir name format is strictly enforced as "step-{step}".
- Files: `src/infra/logging.py` (lines 169-175)
- Trigger: If checkpoint directories are manually created with different naming scheme (e.g., "step-0-backup"), parsing will fail with `IndexError` or return wrong step number.
- Workaround: Follow strict naming convention "step-{number}" when creating checkpoints manually.

## Security Considerations

**Insecure Pickle Loading:**
- Risk: `torch.load()` in `src/infra/logging.py` line 161 loads untrusted checkpoint files without validation. Pickle protocol can execute arbitrary code.
- Files: `src/infra/logging.py` (line 161)
- Current mitigation: Relies on filesystem permissions and trusted checkpoint sources.
- Recommendations: Use `torch.load(..., weights_only=True)` when loading distributed checkpoints to restrict to tensor data only. For metadata, use `torch.load(..., map_location=...)` with explicit device handling. Document that checkpoints should only be loaded from trusted sources.

**WandB API Key Exposure:**
- Risk: WandB run ID and metadata stored in checkpoint metadata state dict may expose project/entity information in logs or shared checkpoints.
- Files: `src/infra/checkpointer.py` (line 41), `src/infra/logging.py` (lines 49-83)
- Current mitigation: Metadata stored locally, WandB API communication uses environment credentials.
- Recommendations: Document that checkpoint directories should not be shared without considering privacy implications. Consider encrypting metadata in checkpoints if sharing across teams.

## Performance Bottlenecks

**No Async Checkpoint Saving:**
- Problem: Checkpoint save in `src/infra/train_iterator.py` line 191 is blocking. On distributed training, all ranks pause while rank 0 writes to disk, causing synchronization overhead.
- Files: `src/infra/train_iterator.py` (lines 185-192)
- Cause: Uses synchronous `torch.distributed.checkpoint.save()` with no background threading.
- Improvement path: Use async checkpointing with background thread, or use distributed checkpoint's built-in async support if available. Coordinate with timeout checkpoint logic to avoid early termination during async save.

**GarbageCollection Disabled by Default:**
- Problem: `GarbageCollection` class in `src/infra/utils.py` disables Python GC entirely (line 53) and only re-enables it at `gc_freq` intervals.
- Files: `src/infra/utils.py` (lines 49-58)
- Cause: Comment says "avoid stragglers in garbage collection" but implementation is fragile. If step count never triggers `gc_freq` boundary (e.g., training crashes at step 51 when `gc_freq=100`), GC never runs again.
- Improvement path: Use `gc.set_threshold()` instead of `gc.disable()` to reduce collection frequency without disabling entirely. Add cleanup in `__del__` or exception handlers to ensure GC is re-enabled on exit.

**MultiLogger Disk I/O on Every Step:**
- Problem: `logger.write()` calls `flush()` immediately in `src/infra/logging.py` line 112 for every progress message. On distributed training with NFS, this causes synchronization overhead.
- Files: `src/infra/logging.py` (lines 106-112)
- Cause: Unbuffered write to maintain up-to-date log file, but causes I/O contention.
- Improvement path: Use buffered writes with periodic flush (e.g., every 10 steps) for progress messages. Keep unbuffered writes only for critical events (checkpoints, errors). Use `fcntl.lockf()` to avoid file corruption in multi-process scenarios.

**WandB Logs Every Reduced Stat:**
- Problem: In `src/infra/logging.py` line 146, `get_last_values(stats)` is called for every step in `save_multi()`, which logs to WandB even if stats haven't changed significantly.
- Files: `src/infra/logging.py` (lines 140-151)
- Cause: No batching or filtering of metrics before logging. On experiments with many custom metrics, this causes API overhead.
- Improvement path: Only log on configurable interval (e.g., every 10 steps) or implement delta-based logging to reduce API calls.

## Fragile Areas

**TrainingIterator __del__ Cleanup:**
- Files: `src/infra/train_iterator.py` (lines 181-183)
- Why fragile: `__del__()` calls `self.logger.close()` but if logger is None or already closed, will crash. In Python, `__del__` order is unpredictable in distributed settings.
- Safe modification: Move cleanup to explicit context manager (`__enter__`/`__exit__`) instead of `__del__`. Use try-finally blocks to ensure cleanup on exit.
- Test coverage: No unit tests for cleanup logic. Missing tests for exception handling during checkpoint/resume.

**Timeout Logic with Distributed Synchronization:**
- Files: `src/infra/train_iterator.py` (lines 120-148)
- Why fragile: Timeout detection uses `torch.distributed.all_reduce()` which blocks. If one rank detects timeout but others don't, can cause hangs or divergence in training state. The `timeout_checkpoint` flag is process-local and not synchronized.
- Safe modification: Use collective timeout detection with coordination. Document that timeout should be set conservatively to avoid stragglers.
- Test coverage: No tests for timeout edge cases (e.g., timeout exactly at step boundary, timeout during checkpoint).

**Distributed Checkpoint State Dict Structure:**
- Files: `src/infra/checkpointer.py` (lines 65-95)
- Why fragile: Load expects specific keys (`MODEL_STATE_DICT_KEY`, `OPTIMIZER_STATE_DICT_KEY`, etc.) but if checkpoint is from different version of code with renamed/removed keys, loading silently skips them or crashes on `KeyError`.
- Safe modification: Add version stamps to checkpoint metadata. Implement migration logic for checkpoint format changes. Validate all expected keys exist before loading.
- Test coverage: No regression tests for checkpoint compatibility across code versions.

**Config Resolution Order with Overrides:**
- Files: `src/config.py`, `src/train.py` (line 75)
- Why fragile: Hydra config resolution happens before any validation. If user provides conflicting overrides (e.g., `tp_sharding=4 dp_sharding=2 dp_replicate=3` but only 16 GPUs available), error occurs late in `get_world_mesh()` with obscure assertion message.
- Safe modification: Add pre-flight validation in `main()` before calling distributed initialization. Check world size divisibility early.
- Test coverage: No integration tests with Hydra config overrides.

## Scaling Limits

**Single Rank Logging Bottleneck:**
- Current capacity: Supports arbitrary number of ranks, but all non-rank-0 processes are silent.
- Limit: On very large runs (100+ nodes), rank 0 becomes bottleneck for logging and checkpointing coordination. No parallel I/O.
- Scaling path: Implement hierarchical logging where subset of ranks (e.g., one per node) write locally, then aggregate. Use `torch.distributed.all_gather()` for stats collection instead of centralizing on rank 0.

**Checkpoint Storage Proliferation:**
- Current capacity: Each checkpoint interval creates full state dict copy (model + optimizer + scheduler + data module state).
- Limit: On large models (billions of parameters) with frequent checkpoints, disk usage grows unbounded. No automatic cleanup of old checkpoints.
- Scaling path: Implement checkpoint cleanup policy (keep last N checkpoints, or delete older than X days). Use incremental/delta checkpointing. Add storage quota enforcement.

**Distributed Communication Overhead:**
- Current capacity: Tensor parallelism and FSDP introduce all-reduce/all-gather collectives. With poor bandwidth (e.g., multi-node over Ethernet), scaling is limited.
- Limit: Tensor parallelism is only validated for `tp_sharding` but no mechanism to detect/warn about network bottlenecks.
- Scaling path: Add communication profile logging (track collective latencies). Provide diagnostics for slow collectives. Implement communication-avoiding algorithms where possible.

## Dependencies at Risk

**PyTorch 2.9.0 with Torch Distributed Checkpoint:**
- Risk: Project pins `torch==2.9.0` exactly in `pyproject.toml` line 16. Distributed checkpoint API is relatively new and may have stability issues. Comments in `src/infra/parallelisms.py` lines 120-131 explicitly note "all-reduce has caused training divergences on torch==2.6.0", suggesting prior version issues.
- Impact: Upgrades to PyTorch 2.10+ may break distributed checkpointing API or introduce new numerical instability. Downgrades lock out security fixes.
- Migration plan: Monitor PyTorch releases for stable distributed checkpoint support. Create integration tests for each PyTorch version to validate checkpoint round-trip and numerical stability. Pin to version range (e.g., `torch>=2.9,<3.0`) instead of exact version once confidence is established.

**Hydra Framework Stability:**
- Risk: Uses `hydra-core>=1.3.2` and `hydra-submitit-launcher>=1.2.0` with version constraints. Hydra has had breaking changes between minor versions.
- Impact: Config format or launcher API changes can break deployment pipeline. No version lock for launcher.
- Migration plan: Pin launcher version when known to be stable with core. Add integration test that validates Hydra config resolution and Submitit job submission.

**WandB API Coupling:**
- Risk: Tight coupling to WandB in `src/infra/logging.py` with hard-coded project/entity in `src/config.py` line 41-42. If WandB becomes unavailable or changes API, training cannot proceed (unless disabled via config).
- Impact: Network outage or service degradation can block training. No graceful degradation.
- Migration plan: Make WandB optional with better fallback logging. Decouple project/entity from config schema to environment variables only. Add retry logic with exponential backoff for WandB operations.

## Test Coverage Gaps

**No Unit Tests:**
- What's not tested: All infrastructure modules (`train_iterator.py`, `checkpointer.py`, `logging.py`, `parallelisms.py`, `optimizers.py`) lack any unit tests.
- Files: `src/infra/*.py` (all files)
- Risk: Regressions in checkpoint loading, timeout detection, or distributed initialization are caught only in end-to-end training, which is slow and expensive.
- Priority: High - Distributed training is error-prone and should have isolated unit tests.

**No Distributed Training Integration Tests:**
- What's not tested: Multi-GPU and multi-node training scenarios are never validated. Only single-GPU runs can be tested locally.
- Files: All files that use `torch.distributed`
- Risk: Deadlocks, collective synchronization bugs, and rank-mismatched state appear only in production multi-GPU runs. Cannot validate scaling or detect bottlenecks.
- Priority: High - Distributed mode is critical path and has different code paths than single-GPU.

**No Checkpoint Compatibility Tests:**
- What's not tested: Checkpoint save/load round-trip, resumption from checkpoints, compatibility across code versions.
- Files: `src/infra/checkpointer.py`, `src/infra/train_iterator.py` (resume logic)
- Risk: Silent state corruption during checkpoint load. Resume from checkpoint may train from wrong state without error.
- Priority: High - Checkpoint correctness is critical for long training runs.

**No Config Override Tests:**
- What's not tested: Hydra config composition, CLI overrides, experiment presets.
- Files: `configs/` directory, `src/train.py` (Hydra main)
- Risk: User config mistakes (incompatible parallelism settings, missing required fields) are caught only at runtime with cryptic Hydra errors.
- Priority: Medium - Can be caught earlier with proper validation.

**No Edge Case Tests:**
- What's not tested: Timeout edge cases, very small batch sizes, 0 warmup steps, extreme learning rates, checkpoint interval = 0, resume from step -1 (auto-detect).
- Files: Multiple files
- Risk: Edge cases cause silent failures or numerical issues that only appear in specific configurations.
- Priority: Medium - Edge cases are common in research experiments.

---

*Concerns audit: 2026-02-03*
