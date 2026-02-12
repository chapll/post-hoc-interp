# Technology Stack

**Analysis Date:** 2026-02-06

## Languages

**Primary:**
- Python >=3.13 - All source code and training scripts
  - Project requires Python 3.13+ (specified in `pyproject.toml`)
  - System Python detected: 3.12.3 (needs upgrade to meet requirements)

**Secondary:**
- YAML - Configuration files throughout `configs/` and experimental subdirs

## Runtime

**Environment:**
- Python 3.13+ (required)
- CUDA-enabled GPU required (no CPU fallback)
- Multi-GPU support via PyTorch distributed

**Package Manager:**
- uv 0.9.14 (modern Python package manager)
- Lockfile: `uv.lock` (present, 432KB)
- Installation via: `curl -LsSf https://astral.sh/uv/install.sh | sh`

**Environment Variables:**
- `.env` file (gitignored, per-user configuration)
- `.env.example` provides template
- Required vars: `EXP_DIR`, `DATA_DIR`, `SLURM_PARTITION` (for cluster), optional `SLURM_ACCOUNT`

## Frameworks

**Core:**
- torch 2.9.0 - PyTorch for training and inference (pinned version)
- nnsight >=0.5.15 - Language model interpretation and activation capture
- hydra-core >=1.3.2 - Hierarchical configuration management
- omegaconf >=2.3.0 - Configuration merging and CLI overrides

**Data & ML:**
- datasets >=4.5.0 - HuggingFace datasets library (GSM-Symbolic, FOLIO, StrategyQA)
- einops >=0.8.1 - Tensor operations with readable notation
- scikit-learn >=1.8.0 - ML utilities and metrics
- numpy >=2.4.1 - Numerical computing (note: v2.x series)

**Monitoring:**
- wandb >=0.23.1 - Experiment tracking and logging (project: "cs224n", entity: "chapll")

**Utilities:**
- python-dotenv >=1.2.1 - `.env` file loading
- tqdm >=4.67.1 - Progress bars
- matplotlib >=3.10.8 - Plotting and visualization
- safetensors >=0.7.0 - Safe tensor serialization format
- pytz >=2025.2 - Timezone handling (default: America/Los_Angeles)

**Compute Orchestration:**
- hydra-submitit-launcher >=1.2.0 - Slurm job submission via Hydra

## Key Dependencies

**Critical:**
- torch 2.9.0 - Core deep learning framework with FSDP, tensor parallelism, distributed checkpointing
- nnsight >=0.5.15 - Enables activation capture from `Qwen/Qwen3-8B` via traced forward passes
- hydra-core >=1.3.2 - Powers entire config system with composition and overrides
- datasets >=4.5.0 - Loads reasoning datasets from HuggingFace Hub

**Infrastructure:**
- wandb >=0.23.1 - Team uses shared project for experiment tracking across compute environments
- safetensors >=0.7.0 - Storage format for captured activations and checkpoints
- hydra-submitit-launcher >=1.2.0 - Enables Slurm cluster execution for team members with cluster access

## Configuration

**Environment:**
- Managed via `.env` file (per-user, gitignored)
- Team members maintain separate `.env` files for their compute environments (Ali: B200 GPUs, Gorn: A100 GPUs, Eric: limited compute)
- Template provided in `.env.example`

**Build:**
- `pyproject.toml` - Project metadata, dependencies, build config (Hatch)
- `uv.lock` - Locked dependency versions
- Entry point script: `uv run train` (alias to `src.train:main`)

**Hydra Configuration System:**
- Main config: `configs/config.yaml`
- Modular configs in `configs/`: `job/`, `model/`, `training/`, `checkpoint/`, `wandb/`, `parallelism/`, `comm/`, `deploy/`, `experiment/`
- Experimental configs: `src/experimental/gorn/configs/` (base.yaml, generate.yaml, capture.yaml, validate.yaml)
- Note: YAML configs are **templates/placeholders** customized per team member - do not treat as source of truth for actual runtime settings
- Compute-dependent settings (batch sizes, GPU configs, model sizes) vary per team member

**Code Quality:**
- Ruff (line length: 160) - Linting and formatting configured in `pyproject.toml`

## Platform Requirements

**Development:**
- Python 3.13+
- CUDA-capable GPU (model loading fails without GPU)
- uv package manager
- Per-user paths for `EXP_DIR` and `DATA_DIR`

**Production:**
- Interactive mode: Local GPU machine
- Cluster mode: Slurm cluster with Submitit
  - Configurable resources: nodes, tasks_per_node, gpus_per_node, cpus_per_task, timeout_min, mem_gb
  - Example from `configs/deploy/submitit.yaml`: 1 node, 8 GPUs, 32 CPUs/task, 256GB RAM, 60min timeout
  - Per-user: SLURM_PARTITION and optional SLURM_ACCOUNT set in `.env`

**Multi-User Research Environment:**
- Team uses shared repository with per-user `.env` configurations
- GPU resources vary: B200 (Ali), A100 (Gorn), limited compute (Eric)
- Config files are templates - actual runtime settings customized per user
- Batch sizes, memory limits, and parallelism settings adjusted per available hardware

---

*Stack analysis: 2026-02-06*
