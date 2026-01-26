# cs224n-final-proj

final project repository for stanford cs224n

## getting started

install [uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## setup

1. copy `.env.example` to `.env` and configure your paths:

```bash
cp .env.example .env
```

```txt
EXP_DIR="/path/to/your/exp/folder"
DATA_DIR="/path/to/your/data/folder"
SLURM_PARTITION="your-partition"
SLURM_ACCOUNT="your-account"  # optional
```

2. (optional) for custom slurm settings, copy the example deploy config:

```bash
cp configs/deploy/submitit_example.yaml configs/deploy/submitit_yourname.yaml
```

## usage

**interactive training:**
```bash
uv run train +deploy=interactive job.exp_name=my-experiment
```

**slurm submission:**
```bash
uv run train +deploy=submitit +experiment=test
```

**with custom slurm config:**
```bash
uv run train +deploy=submitit_yourname +experiment=test
```

see `CLAUDE.md` for detailed architecture and configuration docs.
