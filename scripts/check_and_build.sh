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

source "$REPO_ROOT/.venv/bin/activate"
spython recipe "$REPO_ROOT/Dockerfile" > "$REPO_ROOT/Singularity.def"
