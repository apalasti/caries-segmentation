#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE_USER_HOST="${REMOTE_USER_HOST:-nr_hapa2@komondor.hpc.dkf.hu}"
if [[ "$REMOTE_USER_HOST" == *"@"* ]]; then
  REMOTE_USER="${REMOTE_USER_HOST%%@*}"
else
  REMOTE_USER="$USER"
fi
REMOTE_REPO="${REMOTE_REPO:-/home/${REMOTE_USER}/caries-segmentation}"
REPO_URL="${REPO_URL:-https://github.com/apalasti/caries-segmentation}"
LOCAL_DATASET="${LOCAL_DATASET:-$REPO_ROOT/dataset.tar.gz}"
LOCAL_PREPROCESSED="${LOCAL_PREPROCESSED:-$REPO_ROOT/data/preprocessed}"
LOCAL_ENV="${LOCAL_ENV:-$REPO_ROOT/.env}"

SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
)

usage() {
  cat <<'EOF'
Usage:
  scripts/run_on_komondor.sh [--preflight-only] <branch>

Options:
  --preflight-only   Run checks/fixes only, do not execute build_and_run.sh
EOF
}

log() {
  printf '[run_on_komondor] %s\n' "$*"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

RUN_BUILD=1
BRANCH=""
for arg in "$@"; do
  case "$arg" in
    --preflight-only)
      RUN_BUILD=0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $arg" >&2
      usage
      exit 1
      ;;
    *)
      if [[ -n "$BRANCH" ]]; then
        echo "Only one branch argument is allowed." >&2
        usage
        exit 1
      fi
      BRANCH="$arg"
      ;;
  esac
done

if [[ -z "$BRANCH" ]]; then
  echo "Branch argument is required." >&2
  usage
  exit 1
fi

if [[ ! -f "$LOCAL_DATASET" ]]; then
  if [[ -d "$LOCAL_PREPROCESSED" ]]; then
    log "Creating $LOCAL_DATASET from $LOCAL_PREPROCESSED"
    prep_parent="$(cd "$(dirname "$LOCAL_PREPROCESSED")" && pwd)"
    prep_base="$(basename "$LOCAL_PREPROCESSED")"
    tar -czf "$LOCAL_DATASET" -C "$prep_parent" "$prep_base"
  else
    echo "Local dataset archive missing: $LOCAL_DATASET" >&2
    echo "and preprocessed directory not found: $LOCAL_PREPROCESSED" >&2
    exit 1
  fi
fi

run_remote() {
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER_HOST" \
    "bash -s -- '$REMOTE_REPO' '$REPO_URL' '$BRANCH' '$RUN_BUILD'" <<'REMOTE_SCRIPT'
set -euo pipefail

remote_repo="$1"
repo_url="$2"
branch="$3"
run_build="$4"

expand_home_path() {
  case "$1" in
    "~") printf '%s\n' "$HOME" ;;
    "~/"*) printf '%s/%s\n' "$HOME" "${1#~/}" ;;
    */"~/"*) printf '%s/%s\n' "${1%%/~/}" "${1#*/~/}" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

remote_repo="$(expand_home_path "$remote_repo")"
meta_dir="$remote_repo/.run_on_komondor"

printf '[remote_preflight] repo path: %s\n' "$remote_repo"
if [[ ! -d "$remote_repo/.git" ]]; then
  printf '[remote_preflight] cloning repo from %s\n' "$repo_url"
  mkdir -p "$(dirname "$remote_repo")"
  git clone "$repo_url" "$remote_repo"
fi

cd "$remote_repo"

printf '[remote_preflight] fetching remotes\n'
git fetch --all --prune

if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
  printf '[remote_preflight] checking out origin/%s\n' "$branch"
  git checkout -B "$branch" "origin/$branch"
  printf '[remote_preflight] pulling latest changes for %s\n' "$branch"
  git pull --ff-only origin "$branch"
else
  printf '[remote_preflight] ERROR: branch not found on origin: %s\n' "$branch" >&2
  exit 10
fi

venv_dir=""
if [[ -f ".venv/bin/activate" ]]; then
  venv_dir=".venv"
elif [[ -f "venv/bin/activate" ]]; then
  venv_dir="venv"
else
  printf '[remote_preflight] creating virtual environment .venv\n'
  python3 -m venv .venv
  venv_dir=".venv"
fi

# shellcheck disable=SC1090
source "$venv_dir/bin/activate"

if python -c "import spython" >/dev/null 2>&1; then
  printf '[remote_preflight] spython already installed in %s\n' "$venv_dir"
else
  printf '[remote_preflight] installing spython in %s\n' "$venv_dir"
  python -m pip install --upgrade pip
  python -m pip install spython
fi

if [[ ! -f "dataset.tar.gz" ]]; then
  printf '[remote_preflight] dataset.tar.gz missing in %s\n' "$remote_repo" >&2
  exit 42
fi

if [[ ! -f ".env" ]]; then
  printf '[remote_preflight] .env missing in %s\n' "$remote_repo" >&2
  exit 43
fi

if [[ "$run_build" == "1" ]]; then
  mkdir -p "$meta_dir"
  run_log="$meta_dir/run_$(date +%Y%m%d_%H%M%S).log"
  pid_file="$meta_dir/current.pid"
  log_file="$meta_dir/current.log"

  printf '[remote_preflight] launching detached scripts/build_and_run.sh\n'
  nohup bash -lc "cd '$remote_repo' && bash scripts/build_and_run.sh" >"$run_log" 2>&1 < /dev/null &
  run_pid="$!"

  printf '%s\n' "$run_pid" > "$pid_file"
  printf '%s\n' "$run_log" > "$log_file"

  printf '__KOMONDOR_PID__=%s\n' "$run_pid"
  printf '__KOMONDOR_LOG__=%s\n' "$run_log"
  printf '__KOMONDOR_META_DIR__=%s\n' "$meta_dir"
else
  printf '[remote_preflight] preflight completed; skipping build execution\n'
fi
REMOTE_SCRIPT
}

resolve_remote_repo_for_copy() {
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER_HOST" \
    "bash -s -- '$REMOTE_REPO'" <<'REMOTE_RESOLVE'
set -euo pipefail
remote_repo="$1"
case "$remote_repo" in
  "~") remote_repo="$HOME" ;;
  "~/"*) remote_repo="$HOME/${remote_repo#~/}" ;;
  */"~/"*) remote_repo="${remote_repo%%/~/}/${remote_repo#*/~/}" ;;
esac
printf '%s\n' "$remote_repo"
REMOTE_RESOLVE
}

stream_remote_logs() {
  local run_pid="$1"
  local run_log="$2"

  log "Streaming remote logs (Ctrl-C stops local stream only)"
  set +e
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER_HOST" \
    "bash -s -- '$run_pid' '$run_log'" <<'REMOTE_TAIL'
set -euo pipefail
run_pid="$1"
run_log="$2"

if [[ ! -f "$run_log" ]]; then
  echo "[remote_stream] log file not found: $run_log" >&2
  exit 2
fi

if kill -0 "$run_pid" >/dev/null 2>&1; then
  tail -n +1 -F --pid="$run_pid" "$run_log"
else
  echo "[remote_stream] process $run_pid is not running; showing tail"
  tail -n 200 "$run_log"
fi
REMOTE_TAIL
  local stream_status=$?
  set -e

  if [[ $stream_status -ne 0 ]]; then
    log "Log streaming stopped (connection may have dropped). Remote process should continue."
  else
    log "Remote process finished and log streaming ended."
  fi
}

log "Running remote preflight on branch '$BRANCH'"
set +e
remote_output="$(run_remote 2>&1)"
remote_status=$?
set -e

printf '%s\n' "$remote_output"

if [[ $remote_status -eq 42 ]]; then
  log "Remote dataset.tar.gz missing, copying archive to server"
  remote_repo_for_copy="$(resolve_remote_repo_for_copy)"
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER_HOST" "mkdir -p '$remote_repo_for_copy'"
  scp "${SSH_OPTS[@]}" "$LOCAL_DATASET" "$REMOTE_USER_HOST:$remote_repo_for_copy/dataset.tar.gz"
  log "Retrying remote preflight/run after dataset copy"
  remote_output="$(run_remote 2>&1)"
  remote_status=$?
  printf '%s\n' "$remote_output"
fi

if [[ $remote_status -eq 43 ]]; then
  if [[ ! -f "$LOCAL_ENV" ]]; then
    echo "Local .env file missing: $LOCAL_ENV" >&2
    exit 1
  fi
  log "Remote .env missing, copying .env to server"
  remote_repo_for_copy="$(resolve_remote_repo_for_copy)"
  ssh "${SSH_OPTS[@]}" "$REMOTE_USER_HOST" "mkdir -p '$remote_repo_for_copy'"
  scp "${SSH_OPTS[@]}" "$LOCAL_ENV" "$REMOTE_USER_HOST:$remote_repo_for_copy/.env"
  log "Retrying remote preflight/run after .env copy"
  remote_output="$(run_remote 2>&1)"
  remote_status=$?
  printf '%s\n' "$remote_output"
fi

if [[ $remote_status -ne 0 ]]; then
  echo "Remote preflight failed with exit code: $remote_status" >&2
  exit "$remote_status"
fi

if [[ "$RUN_BUILD" == "1" ]]; then
  run_pid="$(printf '%s\n' "$remote_output" | awk -F= '/^__KOMONDOR_PID__=/{print $2}' | tail -n1)"
  run_log="$(printf '%s\n' "$remote_output" | awk -F= '/^__KOMONDOR_LOG__=/{print $2}' | tail -n1)"
  meta_dir="$(printf '%s\n' "$remote_output" | awk -F= '/^__KOMONDOR_META_DIR__=/{print $2}' | tail -n1)"

  if [[ -z "$run_pid" || -z "$run_log" ]]; then
    echo "Detached launch succeeded but run metadata could not be parsed." >&2
    exit 3
  fi

  log "Detached run started on remote host"
  log "PID: $run_pid"
  log "Log: $run_log"
  log "Metadata dir: ${meta_dir:-unknown}"
  log "Reattach log: ssh $REMOTE_USER_HOST 'tail -F \"$run_log\"'"
  log "Check status: ssh $REMOTE_USER_HOST 'kill -0 $run_pid && echo running || echo stopped'"

  stream_remote_logs "$run_pid" "$run_log"
fi

log "Completed successfully"
