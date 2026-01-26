# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project for Stanford CS224N. The infrastructure is adapted from [ttt-video-dit](https://github.com/test-time-training/ttt-video-dit) and [e2e](https://github.com/test-time-training/e2e) for distributed training with PyTorch.

## Commands

**Interactive training (local/single-node):**
```bash
uv run train +deploy=interactive job.exp_name=my-experiment
```

**Slurm submission (multi-node):**
```bash
uv run train +deploy=submitit hydra.launcher.nodes=4 +experiment=test
```

**Using personal Slurm config:**
```bash
uv run train +deploy=submitit_yourname +experiment=test
```

**Lint:**
```bash
uv run ruff check .
uv run ruff format .
```

## Configuration (Hydra)

Uses [Hydra](https://hydra.cc/) for configuration. Configs are YAML files in `configs/`.

**Config groups:**
- `job`: exp_name, seed
- `model`: name, size, lr
- `training`: steps, batch_size, warmup_steps, gc_freq
- `checkpoint`: interval, resume, timeout_minutes
- `wandb`: project, entity, disable
- `parallelism`: tp_sharding, dp_sharding, fsdp_unsharded_dtype
- `deploy`: interactive or submitit

**Override on CLI:**
```bash
uv run train +deploy=interactive job.exp_name=test training.steps=1000
```

**Experiment presets** in `configs/experiment/` compose multiple overrides:
```yaml
# configs/experiment/my_exp.yaml
# @package _global_
job:
  exp_name: my-experiment
training:
  steps: 50000
```

## Per-User Setup

1. Copy `.env.example` to `.env` and set your paths:
```
EXP_DIR="/your/exp/folder"
DATA_DIR="/your/data/folder"
SLURM_PARTITION="your-partition"
SLURM_ACCOUNT="your-account"
```

2. For custom Slurm settings, copy `configs/deploy/submitit_example.yaml` to `configs/deploy/submitit_yourname.yaml` (gitignored).

## Architecture

### Source Code (`src/`)

**Entry Points:**
- **train.py**: Main entry point with `@hydra.main` decorator. All training scripts use this.
- **config.py**: Structured config dataclasses for type hints.

**Infrastructure (`src/infra/`):**
- **train_iterator.py**: `TrainingIterator` wraps the training loop. Handles progress tracking, metric logging, automatic checkpointing at intervals, and timeout-based checkpointing for preemptible jobs. Use `add_metric()` for per-step metrics, `add_multi()` for cumulative stats.

- **checkpointer.py**: Uses PyTorch distributed checkpointing (`torch.distributed.checkpoint`). Saves/loads: model, optimizer, lr_scheduler, data_module sampler state, and wandb run ID for resumption.

- **logging.py**: `MultiLogger` handles file logging and WandB integration. Only rank 0 writes logs. Use `get_logger(cfg)` to get the singleton.

- **parallelisms.py**: Distributed training setup with FSDP and optional tensor parallelism. `init_distributed()` before training, `end_distributed()` after. `apply_parallelisms()` shards the model.

- **optimizers.py**: AdamW with cosine/linear LR schedules. Supports separate learning rates for TTT/SSM parameters vs base parameters.

**Models (`src/models/`):**
- **mlp.py**: Simple MLP for testing the training infrastructure.

## Key Patterns

- All distributed ops assume NCCL backend and CUDA devices
- Checkpoints auto-resume via `checkpoint.resume=true` (uses latest step by default)
- Garbage collection is manually controlled via `GarbageCollection` class to avoid stragglers
- Config accessible via dot notation: `cfg.job.exp_name`, `cfg.training.steps`, etc.
