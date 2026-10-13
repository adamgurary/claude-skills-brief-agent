from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import os
import shutil


ROOT = Path(__file__).resolve().parents[1]


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _setup_fixture(tmp_path: Path) -> tuple[Path, dict[str, str], Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copy(ROOT / "setup.sh", workspace / "setup.sh")
    shutil.copy(ROOT / "latest_mlflow.py", workspace / "latest_mlflow.py")
    shutil.copy(ROOT / "requirements.txt", workspace / "requirements.txt")
    (workspace / "plain").mkdir()
    (workspace / "plain/run.py").write_text("")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    uv_log = tmp_path / "uv.log"
    run_marker = tmp_path / "agent-ran"
    _write_executable(
        fake_bin / "curl",
        """#!/bin/bash
case "$*" in
  *pypi.org/simple/pip/*) exit 0 ;;
  *pypi.org/simple/mlflow/*)
    printf '%s\n' '<a href="mlflow-3.15.2-py3-none-any.whl">old</a>'
    printf '%s\n' '<a href="mlflow-3.16.0-py3-none-any.whl">current</a>'
    exit 0 ;;
esac
exit 1
""",
    )
    _write_executable(
        fake_bin / "databricks",
        """#!/bin/bash
if [ "${1:-}" = "-v" ]; then echo 'Databricks CLI v1.10.0'; fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "uv",
        """#!/bin/bash
printf '%s\n' "$*" >> "$UV_LOG"
for arg in "$@"; do
  case "$arg" in
    mlflow==*) printf '%s\n' "${arg#mlflow==}" > .installed-mlflow-version ;;
  esac
done
exit 0
""",
    )
    python = workspace / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    _write_executable(
        python,
        """#!/bin/bash
if [ "${1:-}" = "latest_mlflow.py" ]; then exec "$REAL_PYTHON" "$@"; fi
if [ "${1:-}" = "-c" ]; then cat .installed-mlflow-version; exit 0; fi
if [ "${1:-}" = "plain/run.py" ]; then touch "$RUN_MARKER"; exit 0; fi
exit 2
""",
    )
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "REAL_PYTHON": sys.executable,
        "RUN_MARKER": str(run_marker),
        "UV_LOG": str(uv_log),
    }
    return workspace, env, uv_log, run_marker


def test_selects_newest_stable_release_from_simple_index() -> None:
    index = """
    <a href="mlflow-3.15.2-py3-none-any.whl">mlflow-3.15.2</a>
    <a href="mlflow-3.16.0rc0-py3-none-any.whl">mlflow-3.16.0rc0</a>
    <a href="mlflow-3.16.0-py3-none-any.whl">mlflow-3.16.0</a>
    <a href="mlflow-3.16.0.tar.gz">mlflow-3.16.0 source</a>
    """

    result = subprocess.run(
        [sys.executable, ROOT / "latest_mlflow.py"],
        input=index,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "3.16.0"


def test_setup_installs_exact_newest_release_and_runs_agent(tmp_path: Path) -> None:
    workspace, env, uv_log, run_marker = _setup_fixture(tmp_path)

    result = subprocess.run(
        ["bash", "setup.sh"],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "mlflow==3.16.0" in uv_log.read_text().split()
    assert "MLflow: 3.16.0" in result.stdout
    assert run_marker.exists()


def test_setup_remembers_profile_across_commands(tmp_path: Path) -> None:
    workspace, env, _, _ = _setup_fixture(tmp_path)
    first = subprocess.run(
        ["bash", "setup.sh"], cwd=workspace,
        env={**env, "BRIEF_AGENT_PROFILE": "brief-agent"},
        capture_output=True, text=True, check=False,
    )
    assert first.returncode == 0, first.stderr
    assert (workspace / ".brief-agent-profile").read_text() == "brief-agent\n"
    env.pop("BRIEF_AGENT_PROFILE", None)
    second = subprocess.run(
        ["bash", "setup.sh"], cwd=workspace, env=env,
        capture_output=True, text=True, check=False,
    )
    assert second.returncode == 0, second.stderr
    assert "profile 'brief-agent'" in second.stdout
    assert (workspace / ".brief-agent-profile").read_text() == "brief-agent\n"
