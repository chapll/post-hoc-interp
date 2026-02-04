# Codebase Structure

**Analysis Date:** 2026-02-03

## Directory Layout

```
post-hoc-interp/
├── src/                           # Main source code
│   ├── train.py                   # Entry point with @hydra.main
│   ├── config.py                  # Structured config dataclasses
│   ├── infra/                     # Infrastructure/utilities
│   │   ├── train_iterator.py      # Training loop wrapper with checkpointing
│   │   ├── checkpointer.py        # Distributed state save/load
│   │   ├── logging.py             # MultiLogger + WandBLogger
│   │   ├── parallelisms.py        # FSDP + tensor parallelism setup
│   │   ├── optimizers.py          # AdamW with schedules and parameter groups
│   │   └── utils.py               # Random seed, timing, GarbageCollection
│   ├── models/                    # Model implementations
│   │   └── mlp.py                 # TwoLayerMLP (reference implementation)
│   └── experimental/              # Per-researcher isolated workspaces
│       ├── gorn/                  # Researcher: gorn
│       │   ├── experiments/       # Experiment configs/code
│       │   └── data/              # Local data artifacts
│       ├── eric/                  # Researcher: eric
│       │   └── README.md
│       └── ali/                   # Researcher: ali
│           └── README.md
├── configs/                       # Hydra configuration files
│   ├── config.yaml                # Root config with defaults
│   ├── job/                       # Job config group
│   │   └── default.yaml           # exp_name, seed
│   ├── model/                     # Model config group
│   │   └── default.yaml           # name, size, lr
│   ├── training/                  # Training config group
│   │   └── default.yaml           # steps, batch_size, warmup_steps, gc_freq
│   ├── checkpoint/                # Checkpoint config group
│   │   └── default.yaml           # interval, resume, timeout_minutes
│   ├── wandb/                     # WandB config group
│   │   └── default.yaml           # project, entity, disable, log_interval
│   ├── parallelism/               # Parallelism config group
│   │   └── default.yaml           # tp_sharding, dp_sharding, fsdp_unsharded_dtype
│   ├── comm/                      # Communication config group
│   │   └── default.yaml           # init_timeout_seconds
│   ├── deploy/                    # Deployment config group
│   │   ├── interactive.yaml       # Single-node/local execution
│   │   ├── submitit.yaml          # Multi-node Slurm submission
│   │   └── submitit_example.yaml  # Template for personal configs
│   └── experiment/                # Experiment presets
│       ├── identity.yaml          # Identity function learning task
│       ├── debug.yaml             # Debug settings
│       └── test.yaml              # Test run settings
├── .planning/                     # GSD planning documents
│   └── codebase/                  # Analysis outputs
├── .env.example                   # Environment variable template
├── .venv/                         # Python virtual environment (on dfs)
├── pyproject.toml                 # Project metadata and dependencies
├── uv.lock                        # Dependency lock file
└── CLAUDE.md                      # Project instructions
```

## Directory Purposes

**src/ - Source Code:**
- Purpose: All trainable and utility code
- Contains: Entry points, models, infrastructure modules
- Key files: `train.py` (entry), `config.py` (schema)

**src/infra/ - Infrastructure:**
- Purpose: Reusable training utilities for any model
- Contains: Distributed coordination, checkpointing, logging, optimization
- Key files: `train_iterator.py`, `checkpointer.py`, `logging.py`, `parallelisms.py`
- No model-specific code; pure utilities

**src/models/ - Models:**
- Purpose: Neural network architectures
- Contains: PyTorch nn.Module implementations
- Key files: `mlp.py` (reference/test model)

**src/experimental/ - Researcher Workspaces:**
- Purpose: Isolated development space per researcher
- Contains: Personal experiments, custom data, ablations
- Organization: One subdirectory per researcher (gorn/, eric/, ali/)
- Gitignore note: Use for uncommitted research; commit results to main src/

**configs/ - Configuration Files:**
- Purpose: All training parameters via Hydra
- Contains: YAML files organized by config groups
- Key structure: `config.yaml` (root) + group subdirectories
- Composition: `defaults` in root config compose groups; CLI overrides supported

**configs/experiment/ - Experiment Presets:**
- Purpose: Pre-built config combinations for common setups
- Contains: YAML files composing multiple group overrides
- Examples: `identity.yaml` (identity learning), `debug.yaml` (small/fast)
- Usage: `uv run train +experiment=debug` applies debug preset

**configs/deploy/ - Deployment Modes:**
- Purpose: Hardware-specific execution settings
- Contains: `interactive.yaml` (single-GPU), `submitit.yaml` (multi-node), `submitit_example.yaml` (template)
- Key params: launcher, nodes, partition, account

**docs/ - Documentation:**
- Purpose: Project-level documentation
- Contains: Architecture notes, setup guides (not currently populated)

## Key File Locations

**Entry Points:**
- `src/train.py`: Main entry point with Hydra decorator; calls train() function
- `configs/config.yaml`: Root Hydra config; specifies defaults and paths

**Configuration:**
- `src/config.py`: Dataclass schema for all config groups (JobConfig, TrainingConfig, etc.)
- `pyproject.toml`: Project metadata, uv dependencies, scripts

**Core Logic:**
- `src/train.py::train()`: Simple MLP training loop (reference implementation)
- `src/infra/train_iterator.py`: Progress, metrics, checkpointing wrapper
- `src/infra/checkpointer.py`: Distributed checkpoint save/load logic

**Infrastructure:**
- `src/infra/logging.py`: MultiLogger (file + WandB), rank-gated writes
- `src/infra/parallelisms.py`: FSDP + tensor parallelism mesh setup
- `src/infra/optimizers.py`: AdamW with parameter groups and schedules
- `src/infra/utils.py`: Random seed, GarbageCollection, timing utilities

**Models:**
- `src/models/mlp.py`: TwoLayerMLP (test model)

## Naming Conventions

**Files:**
- `{module_name}.py`: Snake case for module names (e.g., `train_iterator.py`, `parallelisms.py`)
- `config.yaml`: Lowercase, no special chars
- Checkpoint dirs: `step-{N}` format (e.g., `step-1000`, `step-5000`)

**Directories:**
- `src/{module}/`: Snake case for subdirectories
- `configs/{group}/`: Snake case for config groups
- `src/experimental/{researcher}/`: Researcher name as subdir
- `src/experimental/{researcher}/{namespace}/`: Namespace under researcher

**Classes:**
- PascalCase: `TrainingIterator`, `MultiLogger`, `Checkpointer`, `TwoLayerMLP`

**Functions:**
- snake_case: `create_optimizer()`, `apply_parallelisms()`, `get_logger()`

**Config Keys:**
- snake_case in YAML: `exp_name`, `global_batch_size`, `warmup_steps`
- Dot notation in Python: `cfg.job.exp_name`, `cfg.training.steps`

## Where to Add New Code

**New Model:**
1. Create `src/models/{model_name}.py` with nn.Module class
2. Implement `__init__()` and `forward()`
3. Add model config class to `src/config.py`
4. Create config YAML: `configs/model/{model_name}.yaml`
5. Use: `uv run train +model={model_name}`

**New Feature (training, optimizer, etc.):**
1. Add to appropriate `src/infra/{module}.py` if general utility
2. Or add to `src/train.py::train()` if experiment-specific
3. For infrastructure: Add config to appropriate group in `configs/`
4. For experiment-specific: Use `src/experimental/{your_name}/`

**New Experiment Config:**
1. Create `configs/experiment/{exp_name}.yaml` with preset overrides
2. Reference: `configs/experiment/debug.yaml` for pattern
3. Use: `uv run train +experiment={exp_name}`

**Personal Work (uncommitted):**
1. Use `src/experimental/{your_name}/experiments/` for code
2. Use `src/experimental/{your_name}/data/` for data artifacts
3. Directory is .gitignored for personal branches
4. Commit final results to main codebase when ready

## Special Directories

**src/experimental/:**
- Purpose: Isolated per-researcher development
- Generated: No, user-created
- Committed: Only structure/README; actual work typically gitignored
- Note: Each researcher should have own subdir to avoid conflicts

**.venv/**
- Purpose: Python virtual environment
- Generated: Yes (via uv venv)
- Committed: No (.gitignored)
- Location: On dfs for space and cache isolation

**outputs/ (referenced in config but not committed):**
- Purpose: Training run outputs (logs, checkpoints, configs)
- Generated: Yes (at runtime via deploy_paths)
- Committed: No
- Default location: `${EXP_DIR}/{YYYYMMDD}/{exp_name}_{HHMMSS}/`

**checkpoint/ (under run_dir):**
- Purpose: Distributed checkpoints per step
- Generated: Yes (on interval or timeout)
- Structure: `checkpoint/step-{N}/` containing distributed state

---

*Structure analysis: 2026-02-03*
