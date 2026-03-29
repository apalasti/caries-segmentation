#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
elif [ -f "$REPO_ROOT/venv/bin/activate" ]; then
    source "$REPO_ROOT/venv/bin/activate"
else
    echo "Neither $REPO_ROOT/.venv nor $REPO_ROOT/venv found; create a virtualenv first." >&2
    exit 1
fi
RECIPE_FILE="$REPO_ROOT/moderate-data-science.recipe"
IMAGE="$REPO_ROOT/moderate-data-science.sif"

if command -v spython >/dev/null 2>&1; then
    spython recipe "$REPO_ROOT/Dockerfile" > "$RECIPE_FILE"
elif python -c "import spython" >/dev/null 2>&1; then
    python -m spython.main recipe "$REPO_ROOT/Dockerfile" > "$RECIPE_FILE"
else
    echo "spython missing; installing into active virtualenv..." >&2
    python -m pip install --upgrade pip
    python -m pip install spython
    python -m spython.main recipe "$REPO_ROOT/Dockerfile" > "$RECIPE_FILE"
fi

if command -v sbatch >/dev/null 2>&1; then
    JOB_NAME="$(git -C "$REPO_ROOT" rev-parse --short HEAD)"

    module load singularity
    # Must match CONTAINER in komondor_train.sbatch (default: $SLURM_SUBMIT_DIR/moderate-data-science.sif).
    singularity build --fakeroot --fix-perms --force "$IMAGE" "$RECIPE_FILE"

    # SLURM_SUBMIT_DIR is the cwd at sbatch time; build lives in REPO_ROOT.
    if ! SBATCH_OUT="$(cd "$REPO_ROOT" && sbatch --parsable --job-name="$JOB_NAME" "$REPO_ROOT/scripts/komondor_train.sbatch" 2>&1)"; then
        echo "sbatch failed: $SBATCH_OUT" >&2
        exit 1
    fi
    JOB_ID="${SBATCH_OUT%%;*}"
    echo "Submitted batch job $JOB_ID (job-name=$JOB_NAME)"
    echo "Check status: squeue -j $JOB_ID"
    echo "After completion: sacct -j $JOB_ID"

    SLURM_OUT="$REPO_ROOT/slurm-${JOB_ID}.out"
    echo "Streaming logs: $SLURM_OUT (Ctrl-C to stop streaming)"
    for _ in {1..60}; do
        [ -f "$SLURM_OUT" ] && break
        sleep 1
    done
    if [ -f "$SLURM_OUT" ]; then
        tail -n +1 -f "$SLURM_OUT"
    else
        echo "Could not find $SLURM_OUT yet; check later with: tail -f \"$SLURM_OUT\"" >&2
    fi
else
    echo "sbatch not in PATH; skipped cluster submit. Generate image and run sbatch on Komondor."
fi
