#!/bin/bash
#SBATCH --job-name=per-token-probe
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=general
#SBATCH --partition=high

# Per-token force-answer probing: multiple traces.
# Estimated runtime: 4 traces x ~15 min each = ~60 min, with margin.
#
# Usage: slurm/submit slurm/per-token-probe.sbatch

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

set -a
source "${PROJECT_DIR}/.env"
set +a

JOB_DIR="${EXP_DIR}/${SLURM_JOB_ID:-local}"
mkdir -p "${JOB_DIR}"
exec > "${JOB_DIR}/${SLURM_JOB_NAME:-per-token-probe}_${SLURM_JOB_ID:-local}.out" \
    2> "${JOB_DIR}/${SLURM_JOB_NAME:-per-token-probe}_${SLURM_JOB_ID:-local}.err"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Project dir: $PROJECT_DIR"
echo "Log dir: $JOB_DIR"

cd "${PROJECT_DIR}"

uv sync

echo "Starting per-token probe (multi-trace)..."
uv run python scripts/per_token_probe.py \
    --config configs/per_token_probe.yaml

echo "Probing complete. Running visualization..."
uv run python scripts/plot_per_token_probe.py

echo "Done."
