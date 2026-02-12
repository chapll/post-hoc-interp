# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-04)

**Core value:** Identify the exact internal transition point where Qwen3-8B shifts from "searching" to "knowing" the answer, and provide mechanistic causal evidence that post-transition tokens are performative.
**Current focus:** Phase 2.1 complete (repo restructure and shared infra stabilization). All 3 plans done. Ready for Phase 3.

## Current Position

Phase: 2.1 of 6 (Repo Restructure and Shared Infra Stabilization)
Plan: 3 of 3 (smoke test and Tier 1 unit tests)
Status: Phase complete
Last activity: 2026-02-06 -- Completed 02.1-03-PLAN.md (smoke test and Tier 1 unit tests)

Progress: [#########.] ~45%

## Performance Metrics

**Velocity:**
- Total plans completed: 15
- Average duration: 5min
- Total execution time: ~1.55 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan | Status |
|-------|-------|-------|----------|--------|
| 01-foundation | 6/6 | 48min | 8min | Complete |
| 02-trace-generation-at-scale | 6/6 | 31min | 5min | Complete |
| 02.1-repo-restructure | 3/3 | 12min | 4min | Complete |

**Recent Trend:**
- Last 5 plans: 02-05 (5min), 02-06 (15min), 02.1-01 (3min), 02.1-02 (3min), 02.1-03 (6min)
- Trend: Steady, test writing slightly slower than mechanical restructure tasks

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- nnsight over TransformerLens (Qwen3 compatibility)
- OmegaConf over Hydra (multi-script pipeline, fixed data paths)
- Two-pass trace generation (generate text, then collect activations in single forward pass)
- Process spawning over FSDP/DDP (independent model copies for inference)
- BF16 activation storage via safetensors (not FP32, not torch.save)
- Squeeze batch dim in stored traces: [seq_len, hidden] not [1, seq_len, hidden]
- Float32 softmax for entropy computation, cast to BF16 after (numerical stability)
- Use model._model.generate() for generation (raw HF model, not nnsight wrapper)
- Use dtype= instead of torch_dtype= for transformers 5.0 compatibility
- Token-ID-based think boundary parsing (not string matching; tokens have special=false)
- Single-GPU path runs in-process (no mp.spawn) for debuggability
- Worker signature standardized: (rank, gpu_id, items, cfg) -- worker owns model loading
- **nnsight layer access: use .output (not .output[0]) for [batch, seq_len, hidden] shape**
- **Long traces (3000+ tokens) need chunked capture or reduced layer subset to avoid OOM**
- Validation requires dedicated GPU access (no sharing with other processes)
- GSM-Symbolic contamination: use HF dataset instance 0 as fresh, original_question as contaminated baseline
- run_parallel_isolated always uses mp.Process even for single GPU (run_parallel for in-process debugging)
- Decoupled trace/activation storage: separate roots, shared relative paths
- Full metadata per trace JSON: seed, model params, sampling config, timestamps, think boundaries
- Include is_fresh in trace metadata for GSM-Symbolic contamination analysis
- Decode generated text with skip_special_tokens=False to preserve think markers
- Pre-parser for --quarantine flag before load_config's argparse (avoids conflicts)
- Chunked capture: 12 layers per pass, 3 passes for 36 layers, sequential access in each
- Logits verification: argmax match primary, allclose secondary (BF16 tolerance)
- Store logits_last_token in activation safetensors for post-hoc verification
- Verify logits on first 3 traces per worker only (avoid overhead)
- Directory scanning over manifest for trace discovery
- **mp.get_context("spawn") required for CUDA-compatible multiprocessing on Linux**
- **Swapped Rho/Tau labels to eliminate Yes/No probe confound** -- each problem expands to A/B config variants
- Removed dead .gitignore entries for old Hydra configs/deploy/ patterns during restructure
- **_find_project_root() walks up to pyproject.toml for config path resolution** (robust to future moves)
- **pytest added as dev optional dependency (not main) to keep production env lean**

### Roadmap Evolution

- Phase 2.1 inserted after Phase 2: Repo restructure and shared infra stabilization (INSERTED) -- promote experimental/gorn/ to main repo, eliminate experimental dirs, validate stability

### Pending Todos

- Delete old test data in data/traces/ before full-scale run (old Yes/No prompts)
- Run full-scale trace generation on 4 GPUs with swapped-label protocol (~400 problems x 2 configs x 3 traces = ~2400 traces)
- Run full-scale activation capture after generation completes
- GPU integration smoke test (manual): generate traces + capture activations on single GPU to validate pipeline end-to-end

### Blockers/Concerns

- OOM on long traces: Full 10-layer capture of 3500+ token sequences exhausts 80GB on A100. Phase 2 uses chunked capture strategy (12 layers per pass).
- Disk storage at scale: 2000 traces at 36 layers may approach 1-2 TB -- validate disk quota before capture phase
- GPU sharing causes OOM: validation requires dedicated GPU access

## Session Continuity

Last session: 2026-02-06
Stopped at: Completed 02.1-03-PLAN.md (smoke test and Tier 1 unit tests). Phase 2.1 complete. Next: Phase 3 (Activation Capture).
Resume file: None
