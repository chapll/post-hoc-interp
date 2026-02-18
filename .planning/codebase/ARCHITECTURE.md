# Architecture

**Analysis Date:** 2026-02-06

## Pattern Overview

**Overall:** Dual-mode research architecture with shared training infrastructure and per-researcher experimental sandboxes

**Key Characteristics:**
- Hydra-based configuration system with hierarchical composition
- Shared infrastructure modules for distributed training, checkpointing, and logging
- Per-researcher experimental directories with isolated environments and configs
- Two-pass activation capture pipeline: trace generation decoupled from activation storage
- Multi-GPU parallelism via both torch.distributed (training) and torch.multiprocessing (inference)

## Layers

**Infrastructure Layer (archived -- archive/src/infra/):**
- Purpose: Provided reusable distributed training primitives and experiment management (archived, not used by current research pipeline)
- Location: `archive/src/infra/`
- Contains: Parallelism setup (FSDP, TP), checkpointing, logging (WandB + file), optimizers, training iterators
- Depends on: PyTorch distributed, Hydra config system, WandB
- Used by: Previously used by main training entry point; now archived

**Model Layer (src/models/):**
- Purpose: Model implementations and architecture definitions
- Location: `src/models/`
- Contains: Currently only `src/models/mlp.py` (simple 2-layer MLP for infrastructure validation)
- Depends on: PyTorch nn modules
- Used by: Training scripts
- Note: Minimal currently; main research uses external Qwen3-8B via HuggingFace/nnsight

**Experimental Sandboxes (src/experimental/{user}/):**
- Purpose: Per-researcher isolated workspaces with user-specific hardware configs and experiments
- Location: `src/experimental/ali/`, `src/experimental/eric/`, `src/experimental/gorn/`
- Contains: User-specific scripts, configs, and library modules
- Depends on: Researcher's GPU availability (Ali: B200, Gorn: A100, Eric: limited compute)
- Used by: Individual researchers running experiments outside shared training loop
- Note: Most active is `src/experimental/gorn/` with trace generation pipeline

**Configuration Layer (configs/):**
- Purpose: Hydra configuration composition system with user-customizable overlays
- Location: `configs/` (project-wide templates), `src/experimental/gorn/configs/` (user-specific)
- Contains: Base configs, deploy profiles (interactive, submitit), experiment presets, parallelism settings
- Depends on: OmegaConf resolvers, environment variables from `.env`
- Used by: Hydra decorator in entry points
- Note: Configs are PLACEHOLDERS—each user customizes via `.env` and personal deploy configs

## Data Flow

**Shared Training Flow:**

1. Entry point: `src/train.py` main() decorated with `@hydra.main`
2. Hydra composes config from `configs/config.yaml` + defaults + CLI overrides
3. Environment variables loaded from `.env` (EXP_DIR, DATA_DIR, SLURM settings)
4. Logger initialized: `archive/src/infra/logging.py` creates MultiLogger with WandB and file output
5. Training function executed with simple MLP identity task (0→0, 1→1)
6. Metrics logged to both terminal and `{run_dir}/log.txt`

**Gorn's Trace Generation Pipeline (Decoupled Two-Pass):**

1. Stage 1 - Trace Generation (`src/experimental/gorn/scripts/generate_traces.py`):
   - Load datasets via `src/experimental/gorn/lib/datasets.py` (GSM-Symbolic, FOLIO, StrategyQA)
   - Apply swapped-label protocol expansion (A/B configs with Rho/Tau tokens)
   - Split problems across GPUs via `src/experimental/gorn/lib/multiprocess.py`
   - Each worker loads Qwen3-8B via `src/experimental/gorn/lib/model.py` (nnsight wrapper)
   - Generate thinking-mode traces with `enable_thinking=True` chat template
   - Parse think block boundaries via `src/experimental/gorn/lib/tokens.py`
   - Route to valid/ or failed/ directories based on think block presence
   - Save JSON with token_ids, text, metadata atomically via `src/experimental/gorn/lib/storage.py`

2. Stage 2 - Activation Capture (`src/experimental/gorn/scripts/capture_activations.py`):
   - Load valid trace JSONs from `data/traces/valid/`
   - For each trace, run forward pass via nnsight with trace context
   - Capture residual stream at layer subset [0, 4, 8, 12, 16, 20, 24, 28, 32, 35]
   - Capture last-token activations at all 36 layers
   - Compute top-k logits and entropy per position
   - Save as safetensors to `data/activations/` with BF16 dtype

**State Management:**
- Training checkpoints: Distributed checkpoint format via `archive/src/infra/checkpointer.py` (DCP)
- Experiment logs: Per-run directories with timestamped names `{date}/{exp_name}_{time}/`
- Activation storage: Decoupled safetensors files (activations) + JSON metadata (traces)

## Key Abstractions

**MultiLogger (archive/src/infra/logging.py) [archived]:**
- Purpose: Unified logging interface combining terminal, file, and WandB outputs
- Examples: `archive/src/infra/logging.py:MultiLogger`
- Pattern: Singleton pattern via get_logger(); rank-0-only file writes in distributed mode

**Checkpointer (archive/src/infra/checkpointer.py) [archived]:**
- Purpose: Save/load model, optimizer, scheduler, and data sampler state for distributed training
- Examples: `archive/src/infra/checkpointer.py:Checkpointer`
- Pattern: Wrapper around torch.distributed.checkpoint with metadata preservation (WandB ID tracking)

**Parallelism Mesh (archive/src/infra/parallelisms.py) [archived]:**
- Purpose: Configure tensor parallel, FSDP, and data parallel dimensions for distributed training
- Examples: `archive/src/infra/parallelisms.py:get_world_mesh()`, `apply_parallelisms()`
- Pattern: Device mesh creation with named dimensions (dp_replicate, dp_shard, tp)

**Multi-GPU Work Splitting (src/experimental/gorn/lib/multiprocess.py):**
- Purpose: Distribute data-parallel inference across GPUs with fault isolation
- Examples: `run_parallel_isolated()` for production trace generation, `run_parallel()` for debugging
- Pattern: torch.multiprocessing with 'spawn' context; round-robin work distribution; per-worker model loading

**nnsight Activation Capture (src/experimental/gorn/lib/model.py):**
- Purpose: Capture intermediate activations from transformer layers during forward pass
- Examples: `capture_activations()` in `src/experimental/gorn/lib/model.py`
- Pattern: Context manager wrapping forward pass; .save() proxies for deferred tensor access

**Storage Layer (src/experimental/gorn/lib/storage.py):**
- Purpose: Atomic file writes, deterministic seed derivation, dtype enforcement for trace/activation persistence
- Examples: `write_trace_json()`, `prepare_trace_tensors()`, `derive_trace_seed()`
- Pattern: Tempfile + os.rename for atomicity; MD5-based deterministic seeding; dtype normalization (BF16, int32)

## Entry Points

**Main Training Entry Point:**
- Location: `src/train.py`
- Triggers: CLI via `uv run train` (configured in `pyproject.toml:project.scripts`)
- Responsibilities: Hydra config composition, .env loading, logger initialization, simple MLP training loop

**Trace Generation Entry Point:**
- Location: `src/experimental/gorn/scripts/generate_traces.py`
- Triggers: Manual CLI execution with `uv run python src/experimental/gorn/scripts/generate_traces.py --config ... --gpus ...`
- Responsibilities: Dataset loading, swapped-label expansion, multi-GPU generation dispatch, think block validation, JSON output routing

**Activation Capture Entry Point:**
- Location: `src/experimental/gorn/scripts/capture_activations.py`
- Triggers: Manual CLI execution after trace generation completes
- Responsibilities: Load valid traces, nnsight forward pass, residual/logit capture, safetensors persistence

**Trace Validation Entry Point:**
- Location: `src/experimental/gorn/scripts/validate.py`
- Triggers: Manual CLI execution to verify pipeline integrity
- Responsibilities: Check trace/activation pair completeness, validate tensor shapes and dtypes

## Error Handling

**Strategy:** Layer-specific with graceful degradation and explicit logging

**Patterns:**
- Training infrastructure: Exceptions propagate to Hydra/submitit, logged via MultiLogger
- Multi-GPU generation: Worker crashes isolated via mp.Process; main reports failures without killing other workers
- File I/O: Atomic writes via tempfile + rename; existence checks for resume capability
- Model loading: RuntimeError on CUDA unavailability or invalid GPU ID
- Data parsing: Skip malformed dataset rows with warning logs; continue processing remainder

## Cross-Cutting Concerns

**Logging:**
- MultiLogger for training (terminal + file + WandB)
- Python logging module for experimental scripts (per-worker loggers in multi-GPU)
- Pacific time timezone (pytz America/Los_Angeles)

**Validation:**
- Config validation via OmegaConf with required fields marked `???`
- Activation shape/dtype validation in storage layer
- Think block presence validation for trace routing

**Authentication:**
- WandB API key via environment (implicit from wandb.login())
- HuggingFace token via environment (for Qwen3 model access)
- Slurm account/partition via .env file (SLURM_PARTITION, SLURM_ACCOUNT)

**Hardware Awareness:**
- Per-user GPU configs: Ali (B200), Gorn (A100), Eric (limited)
- Config files contain PLACEHOLDERS—users customize via .env
- Experimental scripts accept --gpus CLI arg for explicit device selection
- Infrastructure layer uses LOCAL_RANK environment variable for distributed training
- torch.cuda.set_device() called explicitly in worker processes

**Path Configuration:**
- User-specific paths via .env: EXP_DIR, DATA_DIR
- Hydra resolvers for dynamic paths: ${now_date:}, ${now_time:}
- Run directories: `{EXP_DIR}/{date}/{exp_name}_{time}/`
- Experimental data: Hardcoded relative paths in `src/experimental/gorn/configs/base.yaml`
- Note: README instructs users to customize configs for their machine

---

*Architecture analysis: 2026-02-06*
