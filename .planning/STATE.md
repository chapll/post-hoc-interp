# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-04)

**Core value:** Identify the exact internal transition point where Qwen3-8B shifts from "searching" to "knowing" the answer, and provide mechanistic causal evidence that post-transition tokens are performative.
**Current focus:** Phase 2.1 complete and UAT passed (10/10). Full-scale trace generation running on 8 GPUs. Quick task 001 completed (flatten src/infra/ to src/). Ready for Phase 3.

## Current Position

Phase: 2.1 of 6 (Repo Restructure and Shared Infra Stabilization)
Plan: 3 of 3 (smoke test and Tier 1 unit tests)
Status: Phase complete
Last activity: 2026-02-18 -- Quick task 001: flattened src/infra/ into src/. All 80 tests pass.

Progress: [#########.] ~45%

## Performance Metrics

**Velocity:**
- Total plans completed: 16 (15 phase plans + 1 quick task)
- Average duration: 5min
- Total execution time: ~1.73 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan | Status |
|-------|-------|-------|----------|--------|
| 01-foundation | 6/6 | 48min | 8min | Complete |
| 02-trace-generation-at-scale | 6/6 | 31min | 5min | Complete |
| 02.1-repo-restructure | 3/3 | 12min | 4min | Complete |

**Recent Trend:**
- Last 5 plans: 02-06 (15min), 02.1-01 (3min), 02.1-02 (3min), 02.1-03 (6min), quick-001 (11min)
- Trend: Steady, mechanical restructure tasks run efficiently

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
- **SLURM scripts in slurm/**: generate.sbatch (full run, 8 GPU), smoke-test.sbatch (quick sanity check, 1 GPU)
- **slurm/submit wrapper**: sources .env and passes --partition to sbatch automatically
- **User .env required**: EXP_DIR, DATA_DIR, SLURM_PARTITION (see PROJECT.md Environment Setup)
- **Flat src/ package**: all modules directly under src/ (no infra/ subpackage) -- imports use `from src.X import Y`

### Roadmap Evolution

- Phase 2.1 inserted after Phase 2: Repo restructure and shared infra stabilization (INSERTED) -- promote experimental/gorn/ to main repo, eliminate experimental dirs, validate stability

### Pending Todos

- ~~Delete old test data in data/traces/ before full-scale run (old Yes/No prompts)~~ -- checked 2026-02-12, no old data found
- ~~Run full-scale trace generation with swapped-label protocol~~ -- submitted on 8 GPUs (2026-02-12)
- ~~GPU integration smoke test~~ -- passed during UAT (job 281033, 49 traces generated)
- Run full-scale activation capture after generation completes

### Blockers/Concerns

- OOM on long traces: Full 10-layer capture of 3500+ token sequences exhausts 80GB on A100. Phase 2 uses chunked capture strategy (12 layers per pass).
- Disk storage at scale: 2000 traces at 36 layers may approach 1-2 TB -- validate disk quota before capture phase
- GPU sharing causes OOM: validation requires dedicated GPU access

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Flatten src/infra/ into src/ | 2026-02-18 | 62e26e1 | [001-move-src-infra-contents-to-src](./quick/001-move-src-infra-contents-to-src/) |

## Session Continuity

Last session: 2026-02-18
Stopped at: Quick task 001 complete (flatten src/infra/ to src/). Next: Phase 3 (Probing and Transition Detection).
Resume file: None
