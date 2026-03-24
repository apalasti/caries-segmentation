#!/bin/bash
#SBATCH --account=nr_haml2025
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpu:1
# GPU: use partition gpu or ai, add e.g. #SBATCH --gres=gpu:1 , and Singularity adds --nv below when SLURM_GPUS_ON_NODE is set.

# Path to the built image (build on Komondor: singularity build --fakeroot ...).
CONTAINER="${CONTAINER:-$HOME/moderate-data-science.sif}"

# Komondor project name for scratch/project bind mounts (see DKF Singularity docs).
PROJECT="${PROJECT:-YOUR_PROJECT_NAME}"

# Host path to a dotenv file passed into the container (override with ENV_FILE=...).
ENV_FILE="${ENV_FILE:-${SLURM_SUBMIT_DIR:-.}/.env}"

set -euo pipefail

module load singularity

SING_EXTRA=()
if [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [ "${SLURM_GPUS_ON_NODE}" != "0" ]; then
    SING_EXTRA+=(--nv)
fi

BINDS=( "-B" "/scratch/${PROJECT}" )
if [ -d "/project/${PROJECT}" ]; then
    BINDS+=( "-B" "/project/${PROJECT}" )
fi

EXTRA=( "${SING_EXTRA[@]}" --env-file "$ENV_FILE" )

singularity exec "${EXTRA[@]}" "${BINDS[@]}" \
    --pwd /app \
    "${CONTAINER}" \
    uv run python -m src.train
