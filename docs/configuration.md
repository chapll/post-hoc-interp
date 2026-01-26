# Configuration Guide

This project uses [Hydra](https://hydra.cc/) for configuration management. All configs are YAML files in `configs/`.

## Quick Start

```bash
# Interactive training (local)
uv run train +deploy=interactive job.exp_name=my-experiment

# Slurm submission
uv run train +deploy=submitit +experiment=test
```

## Directory Structure

```
configs/
├── config.yaml              # Main config with defaults
├── deploy/
│   ├── interactive.yaml     # Local/single-node mode
│   ├── submitit.yaml        # Slurm submission
│   └── submitit_example.yaml # Template for personal configs
├── job/
│   └── default.yaml         # exp_name, seed
├── model/
│   └── default.yaml         # model name, size, lr
├── training/
│   └── default.yaml         # steps, batch_size, warmup
├── checkpoint/
│   └── default.yaml         # interval, resume settings
├── wandb/
│   └── default.yaml         # project, entity, disable
├── parallelism/
│   └── default.yaml         # tp/dp sharding, fsdp dtype
├── comm/
│   └── default.yaml         # distributed init timeout
└── experiment/
    └── test.yaml            # Example experiment preset
```

## Config Groups

### job
| Field | Default | Description |
|-------|---------|-------------|
| `exp_name` | `"default-job"` | Experiment name (used for output dirs) |
| `seed` | `13` | Random seed |

### model
| Field | Default | Description |
|-------|---------|-------------|
| `name` | `"default"` | Model architecture name |
| `size` | `"5B"` | Model size variant |
| `lr` | `1e-4` | Learning rate |

### training
| Field | Default | Description |
|-------|---------|-------------|
| `dataset_path` | `null` | Path to dataset |
| `global_batch_size` | `8` | Global batch size across all GPUs |
| `grad_accum_steps` | `1` | Gradient accumulation steps |
| `warmup_steps` | `50` | LR warmup steps |
| `steps` | `5000` | Total training steps |
| `gc_freq` | `50` | Garbage collection frequency |

### checkpoint
| Field | Default | Description |
|-------|---------|-------------|
| `init_state_dir` | `null` | Path to initial model weights |
| `interval` | `0` | Checkpoint every N steps (0 = disabled) |
| `resume` | `false` | Resume from checkpoint |
| `resume_step` | `-1` | Step to resume from (-1 = latest) |
| `timeout_minutes` | `0` | Auto-checkpoint before timeout |

### wandb
| Field | Default | Description |
|-------|---------|-------------|
| `disable` | `false` | Disable WandB logging |
| `project` | `"cs224n"` | WandB project name |
| `entity` | `"chapll"` | WandB entity/team |
| `log_interval` | `50` | Log every N steps |
| `alert` | `false` | Send WandB alerts on start/resume |

### parallelism
| Field | Default | Description |
|-------|---------|-------------|
| `tp_sharding` | `1` | Tensor parallel sharding |
| `dp_sharding` | `1` | Data parallel sharding (FSDP) |
| `dp_replicate` | `1` | Data parallel replication |
| `fsdp_unsharded_dtype` | `"bfloat16"` | FSDP compute dtype |

## Per-User Setup

### 1. Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit with your paths:

```bash
EXP_DIR="/your/exp/folder"
DATA_DIR="/your/data/folder"
SLURM_PARTITION="your-partition"
SLURM_ACCOUNT="your-account"  # optional
```

### 2. Personal Slurm Config (Optional)

For custom Slurm settings, create your own deploy config:

```bash
cp configs/deploy/submitit_example.yaml configs/deploy/submitit_yourname.yaml
```

Edit the file with your settings (partition, account, node resources). Personal configs are gitignored.

Use it with:

```bash
uv run train +deploy=submitit_yourname +experiment=test
```

## Experiment Presets

Create experiment presets in `configs/experiment/` to bundle common overrides:

```yaml
# configs/experiment/my_exp.yaml
# @package _global_
job:
  exp_name: my-experiment

training:
  steps: 50000
  global_batch_size: 32

checkpoint:
  interval: 1000
```

Use with:

```bash
uv run train +deploy=interactive +experiment=my_exp
```

## CLI Overrides

Override any config value on the command line:

```bash
# Single values
uv run train +deploy=interactive job.exp_name=test training.steps=1000

# Nested values
uv run train +deploy=submitit hydra.launcher.nodes=4

# Boolean flags
uv run train +deploy=interactive wandb.disable=true checkpoint.resume=true
```

## Hydra Features

### Print Config Without Running

```bash
uv run train +deploy=interactive --cfg job
```

### Multirun (Hyperparameter Sweeps)

```bash
uv run train +deploy=interactive -m training.lr=1e-3,1e-4,1e-5
```

### Tab Completion

Enable shell completion:

```bash
eval "$(uv run train --shell-completion bash)"
```

## Troubleshooting

### Missing Required Value

If you see `??? is not a valid value`:

```
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: hydra.launcher.partition
```

Set the required value via CLI or environment variable:

```bash
# Via CLI
uv run train +deploy=submitit hydra.launcher.partition=gpu

# Via .env
SLURM_PARTITION=gpu  # or whatever partition you use
```

### Config Not Found

If a config group isn't found, check:
1. The YAML file exists in the correct `configs/<group>/` directory
2. The filename matches (without `.yaml` extension)
3. No syntax errors in the YAML file
