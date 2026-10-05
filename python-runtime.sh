#!/usr/bin/env bash
# Source from setup-v2.sh after entering the release directory.

study_python_compatible() {
  "$1" -c 'import sys, venv; sys.exit(sys.version_info[:2] != (3, 12))' >/dev/null 2>&1
}

study_find_python() {
  local candidate
  # An existing environment must never be silently replaced or downgraded.
  if [[ -e .venv || -L .venv ]]; then
    if study_python_compatible "$PWD/.venv/bin/python"; then
      printf '%s\n' "$PWD/.venv/bin/python"
      return 0
    fi
    echo 'The existing .venv is not a working Python 3.12 environment. Keep this folder intact and use a new checkout.' >&2
    return 1
  fi
  if [[ -n "${BRIEF_AGENT_PYTHON:-}" ]]; then
    if study_python_compatible "$BRIEF_AGENT_PYTHON"; then
      printf '%s\n' "$BRIEF_AGENT_PYTHON"
      return 0
    fi
    echo 'BRIEF_AGENT_PYTHON must point to a working Python 3.12 interpreter with venv support.' >&2
    return 1
  fi
  for candidate in python3.12 python3 python \
    /opt/homebrew/opt/python@3.12/bin/python3.12 \
    /usr/local/opt/python@3.12/bin/python3.12 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
    "$HOME"/Library/Caches/dbexec/_runtimes/python312/*/python/bin/python3.12 \
    "$HOME"/.cache/dbexec/_runtimes/python312/*/python/bin/python3.12; do
    if study_python_compatible "$candidate"; then
      command -v "$candidate"
      return 0
    fi
  done
  if command -v uv >/dev/null 2>&1; then
    candidate="$(uv --no-config python find --offline --no-python-downloads 3.12 2>/dev/null)" || candidate=''
    if [[ -n "$candidate" ]] && study_python_compatible "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  fi
  if command -v dbexec >/dev/null 2>&1; then
    echo 'Installing Python 3.12.11 using the Databricks runtime manager...' >&2
    if candidate="$(dbexec "$PWD/python-runtime.dbexec" -c 'import sys; print(sys.executable)')" &&
      study_python_compatible "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
    echo 'The Databricks Python download failed. Check your company network connection and rerun the same setup command. Your study files are preserved.' >&2
    return 1
  fi
  echo 'No usable Python 3.12 runtime was found. Setup has not provisioned study resources.' >&2
  echo 'This machine has neither Python 3.12 nor the company runtime manager (dbexec). Ask IT to enable dbexec, then rerun this command. If Python is already installed elsewhere, use BRIEF_AGENT_PYTHON=/absolute/path/to/python3.12 bash setup-v2.sh.' >&2
  return 1
}
