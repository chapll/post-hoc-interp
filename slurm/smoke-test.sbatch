#!/bin/bash
#SBATCH --job-name=smoke-test
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# Quick smoke test: 2 problems per dataset, 50 tokens max.
# Validates that the full pipeline runs without import or runtime errors.
#
# Usage: sbatch --partition=$SLURM_PARTITION slurm/smoke-test.sbatch

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

set -a
source "${PROJECT_DIR}/.env"
set +a

JOB_DIR="${EXP_DIR}/${SLURM_JOB_ID:-local}"
mkdir -p "${JOB_DIR}"
exec > "${JOB_DIR}/${SLURM_JOB_NAME:-smoke-test}_${SLURM_JOB_ID:-local}.out" \
    2> "${JOB_DIR}/${SLURM_JOB_NAME:-smoke-test}_${SLURM_JOB_ID:-local}.err"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Project dir: $PROJECT_DIR"
echo "Log dir: $JOB_DIR"

cd "${PROJECT_DIR}"

uv sync

echo "Starting smoke test..."
uv run python scripts/generate_traces.py \
    --config configs/generate.yaml \
    model.max_new_tokens=50 \
    data.gsm_problems_per_type=1 \
    data.folio_problems=1 \
    data.strategyqa_problems=1

echo "Smoke test completed."
