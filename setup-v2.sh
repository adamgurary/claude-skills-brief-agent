#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
source ./source-env.sh
setup_arguments=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-open) shift ;;
    --assignment)
      if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "Usage: bash setup-v2.sh [--assignment /path/to/assignment.json] [--no-open]" >&2
        exit 2
      fi
      shift 2
      ;;
    *)
      echo "Usage: bash setup-v2.sh [--assignment /path/to/assignment.json] [--no-open]" >&2
      exit 2
      ;;
  esac
done
set -- "${setup_arguments[@]}"
proxy="https://pypi-proxy.cloud.databricks.com/simple/"
if ! command -v python3.12 >/dev/null 2>&1; then
  echo "Python 3.12 is required for this locked V2 release." >&2
  exit 1
fi
mkdir -p .study
lock_hash="$(python3.12 - <<'PY'
import hashlib
from pathlib import Path
print(hashlib.sha256(Path('requirements.lock').read_bytes()).hexdigest())
PY
)"
if [[ ! -x .venv/bin/python || ! -f .study/installed-lock.sha256 || "$(cat .study/installed-lock.sha256)" != "$lock_hash" ]]; then
  echo "Installing the locked Python environment..."
  if [[ ! -x .venv/bin/python ]]; then
    python3.12 -m venv .venv
  fi
  if command -v uv >/dev/null 2>&1; then
    UV_INDEX_URL= UV_EXTRA_INDEX_URL= \
      uv --no-config --quiet pip sync --require-hashes --default-index "$proxy" \
      --python .venv/bin/python requirements.lock
  else
    PIP_CONFIG_FILE=/dev/null PIP_INDEX_URL="$proxy" PIP_EXTRA_INDEX_URL= PIP_FIND_LINKS= \
      .venv/bin/python -m pip install --quiet --require-hashes --no-input \
      -r requirements.lock
  fi
  printf '%s\n' "$lock_hash" > .study/installed-lock.sha256
fi
MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/python participant_setup.py "$@"
