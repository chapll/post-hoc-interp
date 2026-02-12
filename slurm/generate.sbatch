#!/bin/bash
#SBATCH --job-name=gen-traces
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=256G
#SBATCH --gres=gpu:8

# Usage: sbatch --partition=$SLURM_PARTITION slurm/generate.sbatch [overrides...]
# Example: sbatch --partition=high slurm/generate.sbatch

# SLURM_SUBMIT_DIR is the directory where sbatch was invoked.
# $(dirname "$0") does NOT work -- SLURM copies scripts to a spool directory.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Load .env for user-specific paths (EXP_DIR, DATA_DIR, SLURM_PARTITION)
set -a
source "${PROJECT_DIR}/.env"
set +a

# Create job log directory and redirect all output there
JOB_DIR="${EXP_DIR}/${SLURM_JOB_ID:-local}"
mkdir -p "${JOB_DIR}"
exec > "${JOB_DIR}/${SLURM_JOB_NAME:-gen-traces}_${SLURM_JOB_ID:-local}.out" \
    2> "${JOB_DIR}/${SLURM_JOB_NAME:-gen-traces}_${SLURM_JOB_ID:-local}.err"

# Forward any CLI args as OmegaConf overrides
EXTRA_ARGS=("$@")

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Project dir: $PROJECT_DIR"
echo "Log dir: $JOB_DIR"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "Overrides: ${EXTRA_ARGS[*]}"
fi

cd "${PROJECT_DIR}"

echo "Syncing dependencies..."
uv sync

echo "Starting trace generation..."
uv run python scripts/generate_traces.py \
    --config configs/generate.yaml \
    "${EXTRA_ARGS[@]}"

echo "Trace generation completed."
echo "Results in: $(pwd)/data/traces/"
