#!/usr/bin/env bash
# One-command setup for the Claude Code brief generator.
#
#   bash setup.sh
#
# Self-bootstrapping: installs uv and the Databricks CLI if they are missing (no
# Homebrew, no sudo), builds the Python env, logs you into the dogfood workspace,
# then runs the agent and opens the brief in your browser. The only thing you need
# beforehand is a Mac (or Linux) with an internet connection.
#
# Re-running is safe. It reuses completed setup and refreshes MLflow.
set -euo pipefail

cd "$(dirname "$0")"
HOST="https://e2-dogfood.staging.cloud.databricks.com"
PROFILE="${BRIEF_AGENT_PROFILE:-dogfood}"
if [ -z "${BRIEF_AGENT_PROFILE:-}" ] && [ -f .brief-agent-profile ]; then
  saved_profile=""
  IFS= read -r saved_profile < .brief-agent-profile || true
  PROFILE="${saved_profile:-dogfood}"
fi
PY_VERSION="3.12"
DATABRICKS_CLI_VERSION="1.10.0"

# Tools we install ourselves land in ~/.local/bin (no sudo needed). Put it first on
# PATH for this script so a just-installed tool is found immediately.
LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"
export PATH="$LOCAL_BIN:$PATH"

say() { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }

# --- 1. uv (Python package + version manager) --------------------------------
# uv also provides Python 3.12 itself, so we do not need a system Python or brew.
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv (no sudo, into $LOCAL_BIN)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer may place uv in ~/.local/bin or a cargo bin dir. Make sure it is
  # reachable for the rest of this script regardless of which.
  [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env" || true
  hash -r 2>/dev/null || true
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv install did not put uv on PATH. Open a new terminal and re-run bash setup.sh." >&2
  exit 1
fi

# --- 2. Databricks CLI -------------------------------------------------------
# Download the release binary straight into ~/.local/bin (no brew, no sudo).
if ! command -v databricks >/dev/null 2>&1; then
  say "Installing the Databricks CLI (no sudo, into $LOCAL_BIN)"
  os="$(uname -s)"; arch="$(uname -m)"
  case "$os" in
    Darwin) os_tag="darwin" ;;
    Linux)  os_tag="linux" ;;
    *) echo "Unsupported OS: $os. Install the Databricks CLI manually: https://docs.databricks.com/en/dev-tools/cli/install.html" >&2; exit 1 ;;
  esac
  case "$arch" in
    x86_64|amd64) arch_tag="amd64" ;;
    arm64|aarch64) arch_tag="arm64" ;;
    *) echo "Unsupported architecture: $arch. Install the Databricks CLI manually: https://docs.databricks.com/en/dev-tools/cli/install.html" >&2; exit 1 ;;
  esac
  file="databricks_cli_${DATABRICKS_CLI_VERSION}_${os_tag}_${arch_tag}"
  url="https://github.com/databricks/cli/releases/download/v${DATABRICKS_CLI_VERSION}/${file}.zip"
  tmp="$(mktemp -d)"
  curl -fsSL -o "$tmp/cli.zip" "$url"
  unzip -q "$tmp/cli.zip" -d "$tmp"
  chmod +x "$tmp/databricks"
  mv "$tmp/databricks" "$LOCAL_BIN/databricks"
  rm -rf "$tmp"
  hash -r 2>/dev/null || true
fi
say "Databricks CLI: $(databricks -v)"

# --- 3. Package index --------------------------------------------------------
# On the Databricks network, public PyPI is DNS-blocked for supply-chain security,
# so a plain `uv pip install` fails. Installs must go through the internal proxy.
# We probe reachability and only switch uv to the proxy when public PyPI is blocked,
# so this also works off-network. Setting UV_DEFAULT_INDEX makes every uv command
# below use it, no per-command flag. Respect a proxy the caller already configured.
PROXY_INDEX="https://pypi-proxy.cloud.databricks.com/simple/"
if [ -z "${UV_DEFAULT_INDEX:-}" ] && [ -z "${UV_INDEX_URL:-}" ]; then
  if curl -fsS --max-time 6 -o /dev/null "https://pypi.org/simple/pip/" 2>/dev/null; then
    say "Public PyPI reachable, using it"
  else
    say "Public PyPI blocked (Databricks network), using the internal proxy"
    export UV_DEFAULT_INDEX="$PROXY_INDEX"
  fi
fi

# --- 4. Python env -----------------------------------------------------------
if [ ! -x ".venv/bin/python" ]; then
  say "Creating .venv with uv (Python $PY_VERSION, downloaded by uv if needed)"
  uv venv --python "$PY_VERSION" .venv
else
  say ".venv already exists, refreshing dependencies"
fi
say "Installing dependencies and the latest MLflow"
if [ -n "${BRIEF_AGENT_MLFLOW_VERSION:-}" ]; then
  MLFLOW_VERSION="$BRIEF_AGENT_MLFLOW_VERSION"
else
  PACKAGE_INDEX="${UV_DEFAULT_INDEX:-${UV_INDEX_URL:-https://pypi.org/simple/}}"
  MLFLOW_VERSION="$(
    curl -fsSL "${PACKAGE_INDEX%/}/mlflow/" |
      .venv/bin/python latest_mlflow.py
  )"
fi
case "$MLFLOW_VERSION" in
  ''|*[!0-9.]*)
    echo "Could not resolve a valid stable MLflow version: '$MLFLOW_VERSION'" >&2
    exit 1
    ;;
esac
say "Resolved latest stable MLflow: $MLFLOW_VERSION"
uv pip install -r requirements.txt "mlflow==$MLFLOW_VERSION" \
  --upgrade-package mlflow --python .venv/bin/python
INSTALLED_MLFLOW_VERSION="$(.venv/bin/python -c 'import mlflow; print(mlflow.__version__)')"
if [ "$INSTALLED_MLFLOW_VERSION" != "$MLFLOW_VERSION" ]; then
  echo "MLflow version mismatch: resolved $MLFLOW_VERSION but installed $INSTALLED_MLFLOW_VERSION" >&2
  exit 1
fi
printf '%s\n' "$INSTALLED_MLFLOW_VERSION" > .mlflow-version
say "MLflow: $INSTALLED_MLFLOW_VERSION"

# --- 5. Dogfood auth ---------------------------------------------------------
# The agent authenticates via the CLI profile named "$PROFILE". If that profile has
# no valid token, log in (opens a browser for the OAuth click-through).
if databricks auth token -p "$PROFILE" >/dev/null 2>&1; then
  say "Dogfood auth OK (profile '$PROFILE')"
else
  say "Logging into dogfood (profile '$PROFILE') -- finish the OAuth in your browser"
  databricks auth login --host "$HOST" --profile "$PROFILE"
fi

# Keep the selected profile for subsequent plain/run.py and setup.sh commands.
# This file contains a profile name, never an access or refresh token.
printf '%s\n' "$PROFILE" > .brief-agent-profile.tmp
mv .brief-agent-profile.tmp .brief-agent-profile

# --- 6. Run the agent --------------------------------------------------------
say "Running the agent"
BRIEF_AGENT_PROFILE="$PROFILE" .venv/bin/python plain/run.py

say "Done. The brief opened in your browser and is saved in output/."
