# Architecture

**Analysis Date:** 2026-02-03

## Pattern Overview

**Overall:** Layered distributed training framework with pluggable infrastructure components.

**Key Characteristics:**
- Hydra-based configuration management with composable config groups
- Distributed training support (multi-node NCCL, FSDP, tensor parallelism)
- Infrastructure abstraction layer separating training logic from implementation details
- Checkpoint/resume capability with distributed state management
- Experiment-based configuration composition

## Layers

**Configuration Layer:**
- Purpose: Define and compose all training parameters using Hydra
- Location: `configs/`
- Contains: YAML configuration files organized by domain (job, model, training, checkpoint, parallelism, comm, wandb, deploy)
- Depends on: OmegaConf for structure and resolvers
- Used by: All other layers via DictConfig instances

**Entry Point & Orchestration:**
- Purpose: Bootstrap training, coordinate all components
- Location: `src/train.py`
- Contains: `@hydra.main` decorator, config initialization, logger setup, train function dispatch
- Depends on: Config, Logger, Model
- Used by: CLI invocation

**Infrastructure Layer (`src/infra/`):**
- Purpose: Provide reusable training utilities and distributed coordination
- Location: `src/infra/`
- Contains: Checkpointing, logging, distributed setup, optimization, utilities
- Depends on: PyTorch distributed, WandB, OmegaConf
- Used by: Training scripts and models

**Model Layer:**
- Purpose: Define trainable neural network architectures
- Location: `src/models/`
- Contains: TwoLayerMLP (test model), extensible for real models
- Depends on: PyTorch nn module
- Used by: Training layer

**Experimental Layer:**
- Purpose: Isolated per-researcher code and data
- Location: `src/experimental/{researcher}/`
- Contains: Personal experiments, data artifacts, research code
- Depends on: Core infrastructure
- Used by: Individual researchers

## Data Flow

**Training Initialization Flow:**

1. Hydra resolves `configs/config.yaml` with defaults and CLI overrides
2. `src/train.py::main()` receives resolved `DictConfig`
3. Logger initialized via `get_logger(cfg)` (singleton pattern)
4. Config saved to `run_dir/config.yaml` for reproducibility
5. Distributed setup via `init_distributed()` if multi-GPU
6. Model instantiated (e.g., `TwoLayerMLP` from `src/models/mlp.py`)
7. Optimizer created via `create_optimizer()` with parameter groups
8. Optional: Load checkpoint via `checkpointer.load()`
9. Training loop begins

**Training Step Flow:**

1. `TrainingIterator` wraps the main loop (iterator protocol)
2. Forward pass: model(batch) → outputs
3. Loss computation: criterion(outputs, targets)
4. Backward pass: loss.backward()
5. Optimizer step: optimizer.step()
6. Metrics tracked via `add_metric()` and `add_multi()`
7. After each step, check if checkpoint/save needed
8. Logger writes progress with EMA iteration timing
9. WandB log on rank 0 only

**Checkpoint Flow (on interval or timeout):**

1. `TrainingIterator._checkpoint()` triggered
2. `Checkpointer.save()` called with step path
3. Distributed checkpoint via `torch.distributed.checkpoint`
4. State saved: model, optimizer, lr_scheduler, datamodule sampler, metadata (wandb_id)
5. Statistics saved to `step-{N}/all_stat_dict.pth`

**Resume Flow:**

1. `TrainingIterator.resume(step)` called
2. If step=-1, auto-detect latest via `get_latest_checkpoint_step()`
3. Load distributed state via `Checkpointer.load()`
4. Multi-stats loaded, WandB job_id restored
5. Training continues from checkpoint step

**State Management:**

- Training state stored in distributed checkpoint format (rank-sharded)
- Statistics accumulated per-step in `multi_stats` dict (lists)
- Logging state: file handle (rank 0 only) + WandB run
- Optimizer state: per-parameter-group learning rate via scheduler
- Model state: FSDP shards + optional tensor parallel shards

## Key Abstractions

**TrainingIterator:**
- Purpose: Provide progress tracking, checkpointing, timeout handling as iterator
- Examples: `src/infra/train_iterator.py`
- Pattern: Python iterator protocol with side effects (timing, logging, checkpoint logic)
- Key methods: `__iter__`, `__next__`, `add_metric()`, `add_multi()`, `resume()`

**MultiLogger:**
- Purpose: Unified logging to file + WandB across ranks
- Examples: `src/infra/logging.py`
- Pattern: Singleton accessed via `get_logger(cfg)`, rank-gated writes
- Key methods: `write()`, `init_log()`, `save_multi()`, `load_multi()`

**Checkpointer:**
- Purpose: Save/load distributed training state atomically
- Examples: `src/infra/checkpointer.py`
- Pattern: Wrapper around torch.distributed.checkpoint with metadata handling
- Key methods: `save()`, `load()`, `load_pretrained()`

**ParameterGroupManager:**
- Purpose: Categorize model parameters for differential optimizer treatment
- Examples: `src/infra/optimizers.py`
- Pattern: Static methods for TTT/SSM vs. base parameter separation, weight decay rules

**Parallelism (FSDP + TP):**
- Purpose: Distributed model sharding across devices/nodes
- Examples: `src/infra/parallelisms.py`
- Pattern: `init_distributed()` sets up process group, `apply_parallelisms()` shards model
- Configurable: tp_sharding, dp_sharding, dp_replicate mesh dimensions

## Entry Points

**Main Training:**
- Location: `src/train.py::main()`
- Triggers: `uv run train` with Hydra config overrides
- Responsibilities: Config loading, logger init, distributed setup, train execution, cleanup

**Training Loop:**
- Location: `src/train.py::train(cfg, logger)`
- Triggers: Called from `main()` after setup
- Responsibilities: Model/optimizer creation, data loop, loss computation, checkpointing via TrainingIterator

## Error Handling

**Strategy:** Fail loudly with clear tracebacks. No fallbacks or mocks.

**Patterns:**

- Config validation: Hydra/OmegaConf type checking on dataclass fields
- Distributed errors: NCCL backend defaults to timeout on init failure
- Checkpoint errors: RuntimeError caught in `Checkpointer.load_pretrained()` to handle format mismatch
- Training errors: Exceptions propagate up (no try-catch in main loop)

## Cross-Cutting Concerns

**Logging:**
- MultiLogger writes all text to both terminal (via logging module) and file (rank 0 only)
- WandB initialized on rank 0, job_id broadcast to other ranks for resumption
- All print statements go through logger.write() for consistency

**Validation:**
- Config schema defined as dataclasses in `src/config.py` with OmegaConf integration
- Parallelism dimensions validated: world_size must equal tp_sharding * dp_sharding * dp_replicate
- Parameter group validation: no trainable params raises ValueError

**Authentication:**
- WandB entity/project from config, credentials from `wandb login` (local)
- Distributed init uses environment variables (LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)

---

*Architecture analysis: 2026-02-03*
