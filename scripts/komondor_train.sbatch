#!/bin/bash
#SBATCH --account=nr_haml2025
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:1
# GPU: use partition gpu or ai, add e.g. #SBATCH --gres=gpu:1 , and Singularity adds --nv below when SLURM_GPUS_ON_NODE is set.

# Path to the built image (build on Komondor: singularity build --fakeroot ...).
# Default: moderate-data-science.sif next to where you ran sbatch (repo root if you cd there first).
# Override from the submit shell: CONTAINER=/path/to/image.sif sbatch ...
# If your site strips the environment, use: sbatch --export=ALL,CONTAINER=/path/to/image.sif ...
CONTAINER="${CONTAINER:-${SLURM_SUBMIT_DIR:-$HOME}/moderate-data-science.sif}"

# Komondor project name for scratch/project bind mounts (see DKF Singularity docs).
PROJECT="${PROJECT:-nr_haml2025}"

# Host path to a dotenv file passed into the container (override with ENV_FILE=...).
ENV_FILE="${ENV_FILE:-${SLURM_SUBMIT_DIR:-.}/.env}"

# Host path for checkpoints; bind-mounted to /app/checkpoints inside container.
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-/home/${USER}/caries-segmentation/checkpoints}"

set -euo pipefail

# Avoid leaking host venv into the container (uv warns and it can confuse env resolution).
unset VIRTUAL_ENV || true

module load singularity

SING_EXTRA=(--nv)

mkdir -p "$CHECKPOINTS_DIR"

BINDS=( "-B" "/scratch/${PROJECT}" )
if [ -d "/project/${PROJECT}" ]; then
    BINDS+=( "-B" "/project/${PROJECT}" )
fi
BINDS+=( "-B" "${CHECKPOINTS_DIR}:/app/checkpoints" )

EXTRA=( "${SING_EXTRA[@]}" --env-file "$ENV_FILE" )

singularity exec "${EXTRA[@]}" "${BINDS[@]}" \
    --pwd /app \
    "${CONTAINER}" \
    uv run --no-dev python -m src.train
