# External Integrations

**Analysis Date:** 2026-02-03

## APIs & External Services

**Experiment Tracking:**
- WandB (Weights & Biases)
  - Purpose: Cloud-based experiment tracking, metrics logging, run resumption, alerts
  - SDK/Client: `wandb` 0.23.1+
  - Auth: WandB API key (via local `~/.config/wandb` or `~/.cache/wandb`)
  - Configuration: `configs/wandb/default.yaml`
    - Project: `cs224n`
    - Entity: `chapll` (WandB team/account)
    - Log interval: 50 steps
    - Optional alerts enabled via `cfg.wandb.alert`
  - Usage: `src/infra/logging.py` WandBLogger class handles init and logging
  - Features:
    - Run resumption via run ID broadcast across distributed ranks
    - Automatic config serialization to WandB metadata
    - Per-rank coordination (only rank 0 writes to WandB)

## Data Storage

**Databases:**
- Not applicable - no database integrations

**File Storage:**
- Local filesystem only
  - Experiment outputs: `EXP_DIR` (from `.env`)
  - Training data: `DATA_DIR` (from `.env`)
  - Checkpoints: `${deploy_paths.run_dir}/checkpoint/` (timestamped run directory)
  - Logs: `${deploy_paths.run_dir}/log.txt`
  - Config snapshots: `${deploy_paths.run_dir}/config.yaml` (resolved YAML)
  - Statistics: `${deploy_paths.run_dir}/all_stat_dict.pth` (PyTorch tensor dict)

**Caching:**
- None (model runs are designed for deterministic training)
- Directory structure auto-generated per run with timestamp

## Authentication & Identity

**Auth Provider:**
- Custom (per-user environment variables)
  - Implementation: `.env` file-based configuration
  - Load mechanism: `python-dotenv` in `src/train.py`
  - WandB authentication: Uses local WandB config from previous `wandb login` (not env var based)

## Monitoring & Observability

**Error Tracking:**
- None detected

**Logs:**
- Dual file/terminal logging via `MultiLogger` in `src/infra/logging.py`
  - Terminal: Python logging (rank 0 only in distributed mode)
  - File: `log.txt` in run directory
  - Both: Metrics via WandB (when enabled)
- Timestamp format: America/Los_Angeles timezone (hardcoded in logging)
- Progress: tqdm progress bars (if used in model training)

**Distributed Communication Debugging:**
- NCCL trace environment variables (set in `src/infra/parallelisms.py`):
  - `TORCH_NCCL_AVOID_RECORD_STREAMS=1`: Mitigates memory issues with async collectives
  - NCCL_TRACE_BUFFER_SIZE, TORCH_NCCL_DEBUG_INFO_TEMP_FILE, etc. available but not pre-enabled

## CI/CD & Deployment

**Hosting:**
- Slurm HPC clusters (primary deployment target)
- Interactive/local single-node (development mode)

**CI Pipeline:**
- None detected

**Deployment Methods:**
- Interactive mode: `uv run train +deploy=interactive job.exp_name=experiment`
  - Runs locally on a single machine
  - Config: `configs/deploy/interactive.yaml`
  - Hydra mode: RUN (single run, not multirun)

- Cluster submission: `uv run train +deploy=submitit` or `+deploy=submitit_yourname`
  - Uses submitit to launch Slurm jobs
  - Config: `configs/deploy/submitit.yaml` or `configs/deploy/submitit_yourname.yaml` (per-user)
  - Slurm parameters (from config):
    - nodes: 1 (default, override with `hydra.launcher.nodes=4`)
    - tasks_per_node: 1
    - gpus_per_node: 8
    - cpus_per_task: 32
    - timeout_min: 60
    - mem_gb: 256
  - Partition/account: From `SLURM_PARTITION` and `SLURM_ACCOUNT` env vars
  - Hydra mode: MULTIRUN (supports sweep directories)

## Environment Configuration

**Required env vars:**
- `EXP_DIR`: Path where experiment outputs are saved (defaults to `./exp`)
- `DATA_DIR`: Path where training data is located (defaults to `./data`)
- `SLURM_PARTITION`: Slurm partition name (required for cluster submission, no default)
- `SLURM_ACCOUNT`: Optional Slurm account for billing

**Optional env vars for NCCL debugging:**
- `KINETO_LOG_LEVEL=5`: Suppress PyTorch Profiler noise (set in logging.py)
- `TORCH_NCCL_*`: NCCL trace/debug flags (defined but not activated in parallelisms.py)

**Secrets location:**
- WandB: `~/.config/wandb/settings-` (login token stored locally after `wandb login`)
- `.env` file: Gitignored, contains user-specific paths only (no credentials)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- WandB alerts (optional):
  - Title: Job launch or resume notifications
  - Sent via `wandb.run.alert()` when `cfg.wandb.alert=true`
  - Messages include job name and wandb_id for resumption

## Distributed Training Integration

**NCCL Backend:**
- Process group init: `torch.distributed.init_process_group("nccl")`
- Timeout: `cfg.comm.init_timeout_seconds` (default: 300 seconds)
- Device assignment: Per-rank CUDA device via `LOCAL_RANK` env var
- Synchronization: Barrier before cleanup via `torch.distributed.barrier()`

**Checkpoint Storage & Resumption:**
- Uses `torch.distributed.checkpoint` (DCP)
- State dicts saved:
  - Model weights/buffers
  - Optimizer state
  - LR scheduler state
  - Data module sampler state (if applicable)
  - Metadata (wandb_id for resumption)
- Checkpoint path: `${deploy_paths.run_dir}/checkpoint/step-{step}/`
- Auto-resume: Via `cfg.checkpoint.resume=true` (loads latest step by default)

---

*Integration audit: 2026-02-03*
