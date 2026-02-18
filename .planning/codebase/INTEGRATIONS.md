# External Integrations

**Analysis Date:** 2026-02-06

## APIs & External Services

**HuggingFace Hub:**
- Model loading: `Qwen/Qwen3-8B` (8B parameter language model)
  - SDK/Client: `nnsight` (wraps HuggingFace transformers)
  - Direct model access: via `nnsight.LanguageModel` with `device_map` and `dispatch=True`
  - Architecture: 36 layers, 4096 hidden dim, bfloat16 dtype
  - Implementation: `src/experimental/gorn/lib/model.py`
- Dataset loading: HuggingFace `datasets` library
  - GSM-Symbolic: `apple/GSM-Symbolic` (main split)
  - FOLIO: `tasksource/folio` (train + validation splits)
  - StrategyQA: `ChilleD/StrategyQA` (train split)
  - Implementation: `src/experimental/gorn/lib/datasets.py`
- Auth: Not required (public datasets and models)

**Experiment Tracking:**
- Weights & Biases (wandb)
  - Project: `cs224n`
  - Entity: `chapll`
  - SDK: `wandb` Python library
  - Configuration: `configs/wandb/default.yaml`
  - Features: Metrics logging, alerts on job launch/resume
  - Implementation: `archive/src/infra/logging.py` (WandBLogger class)

## Data Storage

**Databases:**
- None (research codebase uses local file storage)

**File Storage:**
- Local filesystem
  - Experiment outputs: `${EXP_DIR}` (per-user, set in `.env`)
  - Datasets: `${DATA_DIR}` (per-user, set in `.env`)
  - Activations: Stored as safetensors files (`.safetensors` format)
  - Checkpoints: PyTorch distributed checkpoints via `torch.distributed.checkpoint`
  - Logs: Text logs (`log.txt`) and PyTorch state dicts (`all_stat_dict.pth`)

**Caching:**
- HuggingFace datasets cache (default `~/.cache/huggingface/`)
- Model cache (managed by nnsight/transformers, default `~/.cache/huggingface/hub/`)

## Authentication & Identity

**Auth Provider:**
- None (no user authentication system)

**Team Access:**
- Git-based collaboration (shared repository)
- Per-user environment configuration via `.env` files (gitignored)
- Shared wandb entity/project for experiment tracking

## Monitoring & Observability

**Error Tracking:**
- None (standard Python logging only)

**Logs:**
- Python `logging` module (INFO level)
- Dual logging: terminal + file (`log.txt` in experiment dir)
- Timezone: America/Los_Angeles
- Per-GPU logs in distributed mode (rank-based filtering)
- Implementation: `archive/src/infra/logging.py` (MultiLogger class)

**Metrics:**
- Wandb metrics logged at configurable intervals (default: every 50 steps)
- Local metrics saved to `all_stat_dict.pth`
- Checkpoint metrics saved to `checkpoint/step-{N}/all_stat_dict.pth`

## CI/CD & Deployment

**Hosting:**
- Not applicable (research code, local/cluster execution only)

**CI Pipeline:**
- None detected

**Execution Modes:**
1. Interactive (local): `uv run train +deploy=interactive`
2. Slurm cluster: `uv run train +deploy=submitit`
   - Launcher: `hydra-submitit-launcher` with `submitit_slurm`
   - Per-user cluster config via `.env`: `SLURM_PARTITION`, `SLURM_ACCOUNT`

## Environment Configuration

**Required env vars:**
- `EXP_DIR` - Experiment output directory (per-user)
- `DATA_DIR` - Dataset storage directory (per-user)
- `SLURM_PARTITION` - Slurm partition name (for cluster execution)
- `SLURM_ACCOUNT` - Slurm account (optional, for cluster execution)

**Secrets location:**
- `.env` file (gitignored, per-user)
- `.env.example` provides template (committed to git)
- Note: Current setup does not require API keys or secrets for HuggingFace (uses public resources only)

**Config resolution:**
- OmegaConf resolvers for environment variables: `${oc.env:VAR_NAME,default}`
- Custom resolvers for timestamps: `${now_date:}`, `${now_time:}`
- Registered in `src/train.py` before Hydra initialization

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- Wandb alerts (optional, configured via `cfg.wandb.alert`)
  - Job launch notifications
  - Job resume notifications

## Model Inference

**Qwen3-8B Inference:**
- Generation via HuggingFace `model.generate()`
  - Temperature: 0.6, top_p: 0.95, top_k: 20
  - Sampling: enabled (`do_sample=True`)
  - Max tokens: 8192 (configurable)
  - Special mode: `enable_thinking=True` in chat template
- Activation capture via nnsight trace
  - Full-sequence residual streams at subset of layers (default: [0, 4, 8, 12, 16, 20, 24, 28, 32, 35])
  - Last-token activations at all 36 layers
  - Full logits from lm_head
  - Storage: bfloat16 CPU tensors in dict format

## Compute Resources

**GPU Management:**
- CUDA device assignment per process/worker
- Multi-GPU support:
  - PyTorch distributed (NCCL backend)
  - FSDP (Fully Sharded Data Parallel)
  - Tensor Parallelism (configurable via `configs/parallelism/`)
  - Data Parallelism
- Multiprocessing: Custom isolated worker pools in `src/experimental/gorn/lib/multiprocess.py`

**Team-Specific Compute:**
- Ali: B200 GPUs (highest compute)
- Gorn: A100 GPUs (high compute)
- Eric: Limited compute resources
- Config files provide templates; actual batch sizes and parallelism settings customized per member

---

*Integration audit: 2026-02-06*
