# Technology Stack

**Analysis Date:** 2026-02-03

## Languages

**Primary:**
- Python 3.13 - Core application language, training and infrastructure code

## Runtime

**Environment:**
- Python 3.13+ (specified in `pyproject.toml` and `.python-version`)

**Package Manager:**
- uv (Astral's universal Python package manager)
- Lockfile: `uv.lock` (present, pinned dependencies)

## Frameworks

**Core Training:**
- PyTorch 2.9.0 - Deep learning framework for model training and distributed computation
- Hydra 1.3.2+ - Configuration management for experiment parameters

**Distributed Training:**
- hydra-submitit-launcher 1.2.0 - Slurm job submission integration via Submitit
- PyTorch Distributed (torch.distributed) - Multi-GPU/multi-node training via NCCL backend
  - FSDP (Fully Sharded Data Parallel) for model parallelism
  - Tensor Parallelism for distributed computation
  - Device mesh for multi-dimensional parallelism

**Experiment Management:**
- WandB (wandb 0.23.1+) - Experiment tracking, metrics logging, alerts

**Utilities:**
- OmegaConf 2.3.0+ - Structured configuration objects compatible with Hydra
- python-dotenv 1.2.1+ - Environment variable loading from .env files
- tqdm 4.67.1+ - Progress bar for training loops
- einops 0.8.1+ - Flexible tensor operations for neural networks
- numpy 2.4.1+ - Numerical computing
- matplotlib 3.10.8+ - Plotting and visualization
- pytz 2025.2+ - Timezone handling for logging timestamps

## Key Dependencies

**Critical:**
- torch==2.9.0 - Pinned version for deterministic distributed training behavior
- hydra-core>=1.3.2 - Configuration framework with multi-run capabilities
- hydra-submitit-launcher>=1.2.0 - Cluster job submission (Slurm integration)
- wandb>=0.23.1 - Experiment tracking and metrics logging

**Infrastructure:**
- omegaconf>=2.3.0 - OmegaConf data structures for Hydra configs
- python-dotenv>=1.2.1 - Per-user environment configuration
- pytz>=2025.2 - Timezone handling for run timestamps
- einops>=0.8.1 - Tensor reshaping and pattern-based operations

## Configuration

**Environment:**
- `.env` file (gitignored) for per-user settings:
  - `EXP_DIR`: experiment output directory
  - `DATA_DIR`: data directory
  - `SLURM_PARTITION`: Slurm partition name (for cluster submission)
  - `SLURM_ACCOUNT`: optional Slurm account (for billing)

**Hydra Configuration:**
- Base config: `configs/config.yaml`
- Config groups in `configs/`:
  - `job/`: experiment name, seed
  - `model/`: model name, size, learning rate
  - `training/`: batch size, steps, warmup, gradient accumulation
  - `checkpoint/`: interval, resume, timeout
  - `wandb/`: project, entity, logging interval, alerts
  - `parallelism/`: tensor parallelism, data parallelism, FSDP dtype
  - `comm/`: distributed communication timeout
  - `deploy/`: interactive (local) or submitit (Slurm cluster)

**Build:**
- Hatchling backend (specified in `pyproject.toml`)
- Entry point: `train = "src.train:main"` (Python package script)

## Platform Requirements

**Development:**
- Python 3.13+
- CUDA-capable GPU for PyTorch (training assumes NCCL backend on CUDA)
- uv package manager installed

**Production/Cluster:**
- NVIDIA GPU cluster with NCCL support (required for torch.distributed)
- Slurm job scheduler for multi-node training via submitit
- Network connectivity for WandB cloud integration (if enabled)
- Local filesystem for experiments (`EXP_DIR`) and data (`DATA_DIR`)

---

*Stack analysis: 2026-02-03*
