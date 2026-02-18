# Codebase Structure

**Analysis Date:** 2026-02-06

## Directory Layout

```
post-hoc-interp/
├── .env.example           # Template for user-specific environment variables
├── .python-version        # Python version spec (3.13)
├── pyproject.toml         # Project metadata, dependencies, uv config
├── uv.lock                # Locked dependency versions
├── README.md              # Setup instructions
├── configs/               # Hydra configuration hierarchy (PLACEHOLDERS)
│   ├── config.yaml        # Root config with defaults composition
│   ├── checkpoint/        # Checkpoint interval configs
│   ├── comm/              # Communication timeout configs
│   ├── deploy/            # Deployment profiles (interactive, submitit)
│   ├── experiment/        # Experiment presets (debug, test, identity)
│   ├── job/               # Job configs (exp_name, seed)
│   ├── model/             # Model configs (name, size, lr)
│   ├── parallelism/       # Distributed training configs (TP, DP, FSDP)
│   ├── training/          # Training hyperparameters
│   └── wandb/             # WandB project/entity settings
├── docs/                  # Documentation
│   └── configuration.md   # Config system docs
├── src/                   # Source code
│   ├── train.py           # Main training entry point
│   ├── config.py          # Dataclass config definitions
│   ├── models/            # Model implementations
│   │   └── mlp.py         # Simple 2-layer MLP
│   ├── infra/             # Shared training infrastructure
│   │   ├── checkpointer.py      # Distributed checkpoint save/load
│   │   ├── logging.py           # MultiLogger with WandB integration
│   │   ├── optimizers.py        # Custom optimizer implementations
│   │   ├── parallelisms.py      # FSDP, TP, device mesh setup
│   │   ├── train_iterator.py   # Training loop utilities
│   │   └── utils.py             # Misc utilities (seed, timing)
│   └── experimental/      # Per-researcher sandboxes
│       ├── ali/           # Ali's experiments (B200 GPUs)
│       ├── eric/          # Eric's experiments (limited compute)
│       └── gorn/          # Gorn's experiments (A100 GPUs)
│           ├── configs/   # Gorn-specific config files
│           │   ├── base.yaml        # Base config for all experiments
│           │   ├── generate.yaml    # Trace generation config
│           │   ├── capture.yaml     # Activation capture config
│           │   └── validate.yaml    # Validation config
│           ├── scripts/   # Executable experiment scripts
│           │   ├── generate_traces.py      # Stage 1: Generate thinking traces
│           │   ├── capture_activations.py  # Stage 2: Capture activations
│           │   ├── check_traces.py         # Diagnostic script
│           │   ├── validate.py             # Pipeline validation
│           │   └── verify_phase2.py        # Phase 2 verification
│           └── lib/       # Reusable library modules
│               ├── config.py          # Config loading with CLI overrides
│               ├── datasets.py        # Dataset loaders (GSM, FOLIO, StrategyQA)
│               ├── model.py           # nnsight model wrapper and generation
│               ├── multiprocess.py    # Multi-GPU work distribution
│               ├── prompts.py         # Prompt templates and label protocols
│               ├── storage.py         # Trace/activation file I/O
│               ├── tokens.py          # Token parsing and think boundaries
│               └── utils.py           # Utility functions (seed, logging)
└── .planning/             # GSD planning artifacts
    ├── codebase/          # Codebase documentation (this file)
    ├── phases/            # Phase plans
    └── research/          # Research context docs
```

## Directory Purposes

**Root:**
- Purpose: Project configuration and metadata
- Contains: Package config (pyproject.toml), dependency locks (uv.lock), environment template (.env.example)
- Key files: `pyproject.toml` defines train script entry point; `.env.example` shows required user customization

**configs/:**
- Purpose: Hierarchical Hydra configuration templates (PLACEHOLDERS for user customization)
- Contains: YAML configs organized by concern (deploy, model, training, wandb, etc.)
- Key files: `configs/config.yaml` is the root with defaults composition; `configs/deploy/submitit.yaml` and `submitit_example.yaml` show Slurm integration
- Note: Users customize by creating personal deploy configs (e.g., `submitit_yourname.yaml`) or setting .env variables

**src/:**
- Purpose: All Python source code
- Contains: Entry points, models, infrastructure, and experimental sandboxes
- Key files: `src/train.py` is the main training entry point with Hydra decorator

**src/models/:**
- Purpose: Neural network model definitions
- Contains: Currently only `mlp.py` with simple 2-layer MLP for infrastructure validation
- Key files: `src/models/mlp.py`
- Note: Research uses external Qwen3-8B, not defined here

**archive/src/infra/:**
- Purpose: Archived distributed training infrastructure adapted from test-time-training/ttt-video-dit (not used by current research pipeline)
- Contains: Logging, checkpointing, parallelism setup, optimizers, training utilities
- Key files:
  - `archive/src/infra/logging.py`: MultiLogger class for terminal + file + WandB
  - `archive/src/infra/checkpointer.py`: Distributed checkpoint save/load wrapper
  - `archive/src/infra/parallelisms.py`: FSDP, TP, and device mesh configuration
  - `archive/src/infra/optimizers.py`: Custom optimizer implementations
  - `archive/src/infra/train_iterator.py`: Training loop helpers

**src/experimental/:**
- Purpose: Per-researcher isolated workspaces with user-specific code and configs
- Contains: Three subdirectories for Ali, Eric, and Gorn
- Key files: Most active is `src/experimental/gorn/` with complete trace generation pipeline
- Note: Sandboxed to avoid merge conflicts; each user has different GPU hardware (Ali: B200, Gorn: A100, Eric: limited)

**src/experimental/gorn/configs/:**
- Purpose: Gorn's experiment-specific configuration files
- Contains: Base config plus stage-specific overlays (generate, capture, validate)
- Key files:
  - `src/experimental/gorn/configs/base.yaml`: Model, data, storage, paths config
  - `src/experimental/gorn/configs/generate.yaml`: Trace generation settings (GSM problems per type, FOLIO/StrategyQA counts)
  - `src/experimental/gorn/configs/capture.yaml`: Activation capture layer subset
  - `src/experimental/gorn/configs/validate.yaml`: Validation thresholds

**src/experimental/gorn/scripts/:**
- Purpose: Executable Python scripts for trace generation pipeline stages
- Contains: Multi-GPU scripts for generation, capture, validation
- Key files:
  - `src/experimental/gorn/scripts/generate_traces.py`: Stage 1 trace generation with multi-GPU
  - `src/experimental/gorn/scripts/capture_activations.py`: Stage 2 activation capture
  - `src/experimental/gorn/scripts/validate.py`: Pipeline integrity validation
  - `src/experimental/gorn/scripts/check_traces.py`: Diagnostic trace inspection
  - `src/experimental/gorn/scripts/verify_phase2.py`: Phase 2 verification script

**src/experimental/gorn/lib/:**
- Purpose: Reusable library modules for trace generation pipeline
- Contains: Model wrappers, dataset loaders, multiprocessing utilities, storage layer
- Key files:
  - `src/experimental/gorn/lib/model.py`: nnsight model loading and activation capture
  - `src/experimental/gorn/lib/datasets.py`: GSM-Symbolic, FOLIO, StrategyQA loaders with binary verification format
  - `src/experimental/gorn/lib/multiprocess.py`: Multi-GPU work splitting with fault isolation
  - `src/experimental/gorn/lib/storage.py`: Safetensors save/load, atomic JSON writes, deterministic seeding
  - `src/experimental/gorn/lib/prompts.py`: Swapped-label prompt construction
  - `src/experimental/gorn/lib/tokens.py`: Think block boundary parsing
  - `src/experimental/gorn/lib/config.py`: Config file loading with argparse CLI overrides
  - `src/experimental/gorn/lib/utils.py`: Seed setting, logger setup

**docs/:**
- Purpose: Human-readable documentation
- Contains: Currently only `configuration.md`
- Key files: `docs/configuration.md` explains Hydra config system usage

**.planning/:**
- Purpose: GSD (Get Stuff Done) planning and tracking artifacts
- Contains: Codebase docs, phase plans, research context
- Key files: This file is in `.planning/codebase/STRUCTURE.md`

## Key File Locations

**Entry Points:**
- `src/train.py`: Main training script with Hydra decorator
- `src/experimental/gorn/scripts/generate_traces.py`: Trace generation entry point
- `src/experimental/gorn/scripts/capture_activations.py`: Activation capture entry point

**Configuration:**
- `configs/config.yaml`: Root Hydra config
- `.env.example`: Template for user-specific paths and Slurm settings (copy to `.env`)
- `src/experimental/gorn/configs/base.yaml`: Gorn's base experiment config

**Core Logic:**
- `src/model.py`: Model loading and activation capture
- `src/multiprocess.py`: Multi-GPU orchestration
- `src/datasets.py`: Dataset loading and preprocessing
- `archive/src/infra/logging.py`: MultiLogger implementation (archived)
- `archive/src/infra/parallelisms.py`: Distributed training setup (archived)

**Testing:**
- No dedicated test directory currently
- Validation scripts: `src/experimental/gorn/scripts/validate.py`, `verify_phase2.py`

## Naming Conventions

**Files:**
- Python modules: `lowercase_with_underscores.py`
- Config files: `lowercase.yaml`
- Entry point scripts: Descriptive verb phrases (e.g., `generate_traces.py`, `capture_activations.py`)

**Directories:**
- All lowercase, no underscores for major divisions (`configs`, `models`, `infra`, `experimental`)
- User-specific directories: First name lowercase (`ali`, `eric`, `gorn`)

**Python identifiers:**
- Classes: PascalCase (e.g., `MultiLogger`, `Checkpointer`, `TwoLayerMLP`)
- Functions: snake_case (e.g., `get_logger`, `load_model`, `capture_activations`)
- Constants: UPPER_SNAKE_CASE (e.g., `DEFAULT_LAYER_SUBSET`, `NUM_LAYERS`, `HIDDEN_DIM`)
- Config dataclasses: PascalCase with "Config" suffix (e.g., `JobConfig`, `ModelConfig`)

**Config keys:**
- YAML keys: snake_case (e.g., `exp_dir`, `max_new_tokens`, `traces_per_problem`)
- Hydra special: Dotted paths (e.g., `job.exp_name`, `deploy_paths.run_dir`)

## Where to Add New Code

**New Shared Infrastructure Module:**
- Primary code: `src/{module_name}.py`
- Config: `configs/{module_name}/default.yaml`
- Import in entry points as needed

**New Model Architecture:**
- Implementation: `src/models/{model_name}.py`
- Config: `configs/model/{model_name}.yaml`
- Register in training entry point

**New Experimental Script (for specific researcher):**
- Script: `src/experimental/{username}/scripts/{script_name}.py`
- Config: `src/experimental/{username}/configs/{script_name}.yaml`
- Library support: `src/experimental/{username}/lib/{module_name}.py`
- Note: Use existing researcher directory (ali, eric, gorn) or create new one for new team member

**New Dataset Loader:**
- Implementation: `src/experimental/gorn/lib/datasets.py` (add new function)
- Follow pattern: Return list of dicts with uniform schema (problem_id, question, ground_truth, dataset_name, etc.)
- Register in `load_dataset_by_name()` dispatcher

**New Deployment Profile:**
- Config: `configs/deploy/{profile_name}.yaml`
- Override Hydra launcher settings
- Example: Copy `configs/deploy/submitit_example.yaml` and customize for specific cluster

**Utilities:**
- Shared helpers: `src/utils.py`

## Special Directories

**configs/:**
- Purpose: Hydra configuration templates
- Generated: No
- Committed: Yes
- Note: Contains PLACEHOLDERS—users must customize via .env or personal deploy configs

**src/experimental/{user}/data/:**
- Purpose: User-specific data outputs (traces, activations)
- Generated: Yes (by generation/capture scripts)
- Committed: No (in .gitignore)
- Structure: `data/traces/{valid,failed}/{dataset}/{problem_id}/trace_NN.json` and `data/activations/{dataset}/{problem_id}/trace_NN.safetensors`

**Experiment output directories (from .env EXP_DIR):**
- Purpose: Training run outputs (logs, checkpoints)
- Generated: Yes (by training runs)
- Committed: No (user-specific paths outside repo)
- Structure: `{EXP_DIR}/{date}/{exp_name}_{time}/`

**.planning/:**
- Purpose: GSD planning artifacts
- Generated: Yes (by GSD commands)
- Committed: Yes
- Note: Used by `/gsd:*` commands for context and planning

**__pycache__/:**
- Purpose: Python bytecode cache
- Generated: Yes
- Committed: No (in .gitignore)

## User-Specific vs Shared Code

**User-Specific Directories:**
- `src/experimental/ali/`
- `src/experimental/eric/`
- `src/experimental/gorn/`
- Each researcher's experiments, configs, and hardware-specific settings

**Shared Code:**
- `src/`: Research infrastructure modules (config, datasets, model, multiprocess, prompts, storage, tokens, utils)
- `configs/`: Template configurations (users customize via .env and personal deploy configs)

**Machine-Specific Configuration:**
- `.env` file (gitignored): EXP_DIR, DATA_DIR, SLURM_PARTITION, SLURM_ACCOUNT
- `configs/deploy/submitit_{username}.yaml`: Per-user Slurm cluster settings
- `src/experimental/{user}/configs/`: Hardcoded relative paths in base.yaml are user-customizable

**Cross-User Collaboration Pattern:**
- Shared infrastructure changes: Commit to `src/` and `configs/` templates
- User experiments: Keep in `src/experimental/{user}/`; coordinate via .planning/ docs
- Hardware differences: Document in config comments; use CLI args (--gpus) for flexibility

---

*Structure analysis: 2026-02-06*
