# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-04)

**Core value:** Identify the exact internal transition point where Qwen3-8B shifts from "searching" to "knowing" the answer, and provide mechanistic causal evidence that post-transition tokens are performative.
**Current focus:** Phase 2.4 complete (per-token dense probing script + visualization). Next: GPU execution of per-token probing, then Phase 3.

## Current Position

Phase: 2.4 of 6 (Per-Token Force-Answer Probing)
Plan: 2 of 2
Status: Phase complete
Last activity: 2026-02-26 -- Completed 02.4-02-PLAN.md (per-token probe visualization)

Progress: [#############] ~60%

## Performance Metrics

**Velocity:**
- Total plans completed: 27 (21 phase plans + 6 quick tasks)
- Average duration: 5min
- Total execution time: ~2.4 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan | Status |
|-------|-------|-------|----------|--------|
| 01-foundation | 6/6 | 48min | 8min | Complete |
| 02-trace-generation-at-scale | 6/6 | 31min | 5min | Complete |
| 02.1-repo-restructure | 3/3 | 12min | 4min | Complete |
| 02.2-force-answer-experiment | 2/2 | 16min | 8min | Complete |
| 02.3-cross-seed-consistency | 3/3 | 14min | 4.7min | Complete |
| 02.4-per-token-force-answer | 2/2 | 5min | 2.5min | Complete |

**Recent Trend:**
- Last 5 plans: 02.3-02 (4min), 02.3-03 (5min), 02.4-01 (3min), 02.4-02 (2min)
- Trend: Steady execution, visualization scripts fastest at ~2 min

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
- **DATA_DIR env-var interpolation**: configs use `${oc.env:DATA_DIR,data}` with "data" fallback; traces_dir and activations_dir derive from data_dir
- **DATA_DIR is /mnt/data/aahmad/data** (not post-hoc-interp subdir) -- trace data at /mnt/data/aahmad/data/traces/
- **DynamicCache crop() for KV-cache-efficient probing** -- process probes in descending order, deepcopy before generate()
- **Dual answer matching** -- token-level (Rho/Tau IDs) + string-level (decoded text), disagreements flagged
- **correct_token_rank via double argsort** on float32 softmax probabilities (0-indexed, 0 = highest)
- **Re-implement Phase 2.2 selection logic inline** in select_problems() rather than importing -- avoids config shim coupling
- **Extract verification_prompt from existing trace metadata** (not reconstructed) -- guarantees byte-identical prompts across seeds
- **Trace indices 3-19 for new seeds; 0-2 existing traces left untouched**
- **cfg_shim pattern**: when reusing probe_trace() across experiments with different config schemas, create minimal OmegaConf wrapper with expected `force_answer` key
- **Import probing functions from force_answer_experiment.py** via sys.path -- single source of truth for KV-cache probing logic
- **run_parallel_isolated for cross-seed probing** -- longer job (~20-30 min) where one OOM should not abort all workers
- **21-point interpolation grid** (np.linspace(0, 1, 21)) for cross-seed curve comparison
- **Sharpness = max absolute slope of mean accuracy curve** (max of |np.diff / np.diff(grid)|)
- **No formal transition point detection** -- raw data visualization per locked decision
- **Inlined extract_answer/extract_rho_tau_probs in no_think_baseline.py** -- avoids cross-script import complexity for ~60 lines of self-contained logic
- **Config A only for no-think baseline** -- baseline filter only needs correctness, not both label mappings
- **Inline probing loop for per-token experiment** -- enables periodic torch.cuda.empty_cache() every 100 probes over 824 iterations
- **Direct trace loading by path** for single known trace -- no need for load_traces_for_experiment() selection logic
- **COARSE_COLOR=#BB5566** for Phase 2.2 overlay in per-token visualization -- distinct from ACC_COLOR and PROB_COLOR
- **Transition region detection** via rolling-20 accuracy crossing 0.5 threshold, 10% margin expansion, fallback to 25%-85%
- **Transition shape heuristic** -- <=2 transitions=step function, <=6=gradual ramp, >6=oscillatory (rough label, plots are primary)
- **Multi-trace config pattern** -- traces list in config replaces single-trace fields; num_probes derived dynamically per trace
- **Resume-by-output-existence** -- skip traces with existing results.parquet before processing
- **Normalized position for multi-trace comparison** -- traces with different think_lengths align on 0-1 scale

### Roadmap Evolution

- Phase 2.1 inserted after Phase 2: Repo restructure and shared infra stabilization (INSERTED) -- promote experimental/gorn/ to main repo, eliminate experimental dirs, validate stability
- Phase 2.2 inserted after Phase 2.1: Force-answer transition experiment (INSERTED) -- truncate thinking at probe points, force model to answer, plot accuracy-vs-position to detect phase transition. Exploratory experiment to validate core hypothesis before full probing infrastructure.
- Phase 2.3 inserted after Phase 2.2: Cross-seed transition consistency (INSERTED) -- for each problem, probe all 3 traces (different seeds) to check if the transition point is consistent across seeds. Per-problem visualization + cross-seed agreement quantification. Aggregate analysis from 2.2 dilutes signal; per-problem cross-seed analysis is the right lens.
- Phase 2.4 inserted after Phase 2.3: Per-token force-answer probing (INSERTED) -- probe at every thinking token for a single trace (gsm_symbolic_68_fresh_false_cfga) to get high-resolution transition dynamics. Reveals exact shape of transition that ~20-point grid cannot resolve.

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
| 002 | Move data to DATA_DIR, env-var config paths | 2026-02-18 | 4ac6096 | [002-move-data-to-data-dir](./quick/002-move-data-to-data-dir/) |
| 003 | Write CS224N milestone report draft | 2026-02-26 | bf181ac | [003-write-milestone-report-drafts](./quick/003-write-milestone-report-drafts/) |
| 004 | Condense milestone to ~2 pages | 2026-02-26 | 66ab174 | [004-condense-milestone-to-2-pages](./quick/004-condense-milestone-to-2-pages/) |
| 005 | No-think baseline filter script | 2026-02-26 | e156946 | [005-no-think-baseline-filter](./quick/005-no-think-baseline-filter/) |
| 006 | Multi-trace per-token probing | 2026-02-26 | dc8668d | [006-multi-trace-per-token-probe](./quick/006-multi-trace-per-token-probe/) |

## Session Continuity

Last session: 2026-02-26
Stopped at: Completed quick-006 (multi-trace per-token probing). Ready to submit multi-trace run via slurm/submit slurm/per-token-probe.sbatch.
Resume file: None
