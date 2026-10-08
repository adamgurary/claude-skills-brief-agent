#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./source-env.sh
.venv/bin/python study_auth.py --manifest .study/manifest.json
export MLFLOW_ASYNC_TRACE_LOGGING_MAX_SPAN_BATCH_SIZE=100
export MLFLOW_ASYNC_TRACE_LOGGING_MAX_INTERVAL_MILLIS=60000
exec .venv/bin/python plain/run.py --no-open "$@"
