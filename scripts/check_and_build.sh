#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

git remote update

LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u})

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "Up to date. Nothing to do."
    exit 0
fi

git pull

if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
elif [ -f "$REPO_ROOT/venv/bin/activate" ]; then
    source "$REPO_ROOT/venv/bin/activate"
else
    echo "Neither $REPO_ROOT/.venv nor $REPO_ROOT/venv found; create a virtualenv first." >&2
    exit 1
fi
spython recipe "$REPO_ROOT/Dockerfile" > "$REPO_ROOT/Singularity.def"

if command -v sbatch >/dev/null 2>&1; then

    module load singularity
    singularity build --fakeroot --fix-perms "$REPO_ROOT/Singularity.sif" "$REPO_ROOT/Singularity.def"

    JOB_NAME="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
    if ! SBATCH_OUT="$(sbatch --parsable --job-name="$JOB_NAME" "$REPO_ROOT/scripts/komondor_train.sbatch" 2>&1)"; then
        echo "sbatch failed: $SBATCH_OUT" >&2
        exit 1
    fi
    JOB_ID="${SBATCH_OUT%%;*}"
    echo "Submitted batch job $JOB_ID (job-name=$JOB_NAME)"
    echo "Check status: squeue -j $JOB_ID"
    echo "After completion: sacct -j $JOB_ID"
else
    echo "sbatch not in PATH; skipped cluster submit. Generate image and run sbatch on Komondor."
fi
