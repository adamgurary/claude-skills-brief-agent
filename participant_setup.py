"""Verify a prepared V2 assignment without running the agent or a model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shlex
import subprocess
import sys
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from study_auth import browser_workspace_client

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / ".study"
MANIFEST = STATE_DIR / "manifest.json"
EXPECTED_MODEL = "databricks-gpt-5-6-terra"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hash() -> str:
    files = sorted((ROOT / "plain").rglob("*.py"))
    files.append(ROOT / "requirements.lock")
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _authenticate(assignment: dict) -> tuple[WorkspaceClient, str]:
    profile = assignment.get("profile") or "dogfood"
    try:
        client = WorkspaceClient(profile=profile)
        if client.config.host.rstrip("/") == assignment["workspace_host"].rstrip("/"):
            identity = client.current_user.me().user_name
            if identity == assignment["principal"]:
                return client, "profile"
    except Exception:
        pass
    client = browser_workspace_client(assignment["workspace_host"])
    identity = client.current_user.me().user_name
    if identity != assignment["principal"]:
        raise RuntimeError(f"assignment belongs to {assignment['principal']}, authenticated as {identity}")
    return client, "external-browser"


def _configure_mlflow(assignment: dict, auth_mode: str, profile: str) -> None:
    if auth_mode == "profile":
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
        mlflow.set_tracking_uri(f"databricks://{profile}")
    else:
        os.environ["DATABRICKS_HOST"] = assignment["workspace_host"]
        os.environ["DATABRICKS_AUTH_TYPE"] = "external-browser"
        mlflow.set_tracking_uri("databricks")
    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = assignment["warehouse_id"]


def _check_sources() -> dict:
    sys.path.insert(0, str(ROOT / "plain"))
    from brief_agent import research

    with mlflow.tracing.context(enabled=False):
        hn = research.hn_search.invoke({"query": "Claude Code skills", "max_results": 1})
        github = research.github_search.invoke({"query": "Claude Code skills", "max_results": 1})
        readme = research.github_readme.invoke({
            "full_name": "adamgurary/claude-skills-brief-agent", "max_chars": 200,
        })
    if not readme:
        raise RuntimeError("GitHub README source health returned empty content")
    return {"hacker_news_results": len(hn), "github_results": len(github),
            "github_readme_chars": len(readme)}


def _verify_baselines(assignment: dict, traces: list) -> None:
    captures = {item["trace_id"]: item for item in assignment["baseline_captures"]}
    if len(traces) != 5:
        raise RuntimeError("five complete baseline traces are required")
    for trace, trace_id in zip(traces, assignment["baseline_trace_ids"]):
        if trace is None or trace.info.trace_id != trace_id:
            raise RuntimeError(f"baseline trace is missing: {trace_id}")
        capture = captures[trace_id]
        expected = capture.get("span_graph")
        actual = {span.span_id: span.parent_id for span in trace.data.spans}
        if (not expected or len(actual) != len(trace.data.spans) or actual != expected):
            raise RuntimeError(f"baseline trace graph is incomplete or changed: {trace_id}")


def _verify_model_access(endpoint) -> dict:
    metadata = endpoint.as_dict()
    if metadata.get("name") != EXPECTED_MODEL:
        raise RuntimeError("Terra endpoint is unavailable")
    permission = metadata.get("permission_level")
    if permission not in {"CAN_QUERY", "CAN_MANAGE"}:
        raise RuntimeError(f"Terra query access is not verified for this login: {permission!r}")
    return {"endpoint": EXPECTED_MODEL, "permission_level": permission,
            "verification": "authenticated endpoint metadata", "model_calls": 0}


def setup(assignment_path: Path) -> dict:
    started = time.monotonic()
    assignment_path = assignment_path.resolve()
    assignment = json.loads(assignment_path.read_text())
    release = json.loads((ROOT / "release.json").read_text())
    required = (
        "study_id", "attempt_id", "principal", "workspace_host", "experiment_id",
        "experiment_name", "experiment_url", "catalog", "schema", "table_prefix",
        "spans_table", "warehouse_id", "baseline_trace_ids", "briefs",
        "baseline_captures", "fixture_version", "fixture_sha256",
        "source_sha256", "release_ref", "release_commit", "model_endpoint",
    )
    missing = [key for key in required if key not in assignment]
    if missing:
        raise RuntimeError(f"assignment is missing {missing}")
    if assignment["model_endpoint"] != EXPECTED_MODEL:
        raise RuntimeError("assignment model does not match the Terra release")
    if len(assignment["baseline_trace_ids"]) != 5 or len(set(assignment["baseline_trace_ids"])) != 5:
        raise RuntimeError("assignment does not contain five distinct baseline traces")
    if len(assignment["briefs"]) != 5:
        raise RuntimeError("assignment does not contain five briefs")
    if (len(assignment["baseline_captures"]) != 5 or
            {item["trace_id"] for item in assignment["baseline_captures"]} !=
            set(assignment["baseline_trace_ids"])):
        raise RuntimeError("assignment capture times do not match five baseline traces")
    if release["release_ref"] != assignment["release_ref"]:
        raise RuntimeError("assignment selects a different release ref")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          check=True, capture_output=True, text=True).stdout.strip()
    if head != assignment["release_commit"]:
        raise RuntimeError("checkout commit does not match the assigned release")
    source_digest = _source_hash()
    if release["source_sha256"] != assignment["source_sha256"]:
        raise RuntimeError("source or fixture version mismatch")
    if not MANIFEST.exists():
        if source_digest != release["source_sha256"]:
            raise RuntimeError("source was edited before first setup; preserve edits and use a clean release")
        for relative, expected in release["files"].items():
            path = ROOT / relative
            if not path.is_file() or _sha256(path) != expected:
                raise RuntimeError(f"staged release file changed before first setup: {relative}")
    if _sha256(ROOT / "requirements.lock") != release["lock_sha256"]:
        raise RuntimeError("dependency lockfile changed")
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text())
        for key in ("study_id", "attempt_id", "principal", "fixture_sha256",
                    "source_sha256", "release_ref", "release_commit",
                    "experiment_id", "table_prefix"):
            if existing.get(key) != assignment[key]:
                raise RuntimeError(f"existing setup conflicts on {key}; use a fresh checkout")
    client, auth_mode = _authenticate(assignment)
    profile = assignment.get("profile") or "dogfood"
    _configure_mlflow(assignment, auth_mode, profile)
    warehouse = client.warehouses.get(assignment["warehouse_id"])
    if str(warehouse.state).split(".")[-1] != "RUNNING":
        raise RuntimeError(f"study warehouse is not warm: {warehouse.state}")
    endpoint = client.serving_endpoints.get(assignment["model_endpoint"])
    model_access = _verify_model_access(endpoint)
    experiment = mlflow.get_experiment(assignment["experiment_id"])
    if experiment is None or experiment.name != assignment["experiment_name"]:
        raise RuntimeError("assigned MLflow experiment is unavailable")
    with ThreadPoolExecutor(max_workers=5) as pool:
        traces = list(pool.map(mlflow.get_trace, assignment["baseline_trace_ids"]))
    _verify_baselines(assignment, traces)
    source_health = _check_sources()
    STATE_DIR.mkdir(exist_ok=True)
    brief_dir = STATE_DIR / "briefs"
    brief_dir.mkdir(exist_ok=True)
    local_briefs = []
    for brief in assignment["briefs"]:
        source = assignment_path.parent / brief["file"]
        if _sha256(source) != brief["sha256"]:
            raise RuntimeError(f"assignment brief checksum mismatch: {source}")
        destination = brief_dir / f"{brief['label']}.md"
        if destination.exists() and _sha256(destination) != brief["sha256"]:
            raise RuntimeError(f"local brief was edited; preserving it: {destination}")
        if not destination.exists():
            destination.write_bytes(source.read_bytes())
        local_briefs.append({"label": brief["label"], "path": str(destination),
                             "sha256": brief["sha256"]})
    manifest = {key: assignment[key] for key in required}
    manifest.update({
        "auth_mode": auth_mode,
        "profile": profile if auth_mode == "profile" else None,
        "model_endpoint": EXPECTED_MODEL,
        "mlflow_version": mlflow.__version__,
        "sdk_version": importlib.metadata.version("databricks-sdk"),
        "assignment_path": str(assignment_path),
        "local_briefs": local_briefs,
        "source_health": source_health,
        "model_access": model_access,
        "release_manifest_sha256": _sha256(ROOT / "release.json"),
        "lock_sha256": release["lock_sha256"],
        "current_source_sha256": source_digest,
        "output_dir": str(ROOT / "output"),
        "rerun_command": "bash run-v2.sh",
        "setup_machine_seconds": round(time.monotonic() - started, 3),
        "ready_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text())
        existing.update({"auth_mode": auth_mode, "profile": manifest["profile"],
                         "source_health": source_health,
                         "model_access": model_access,
                         "current_source_sha256": source_digest,
                         "ready_at_utc": manifest["ready_at_utc"],
                         "setup_machine_seconds": manifest["setup_machine_seconds"]})
        _write_json(MANIFEST, existing)
        _write_json(STATE_DIR / "latest-readiness.json", {
            "attempt_id": assignment["attempt_id"],
            "ready_at_utc": manifest["ready_at_utc"],
            "setup_machine_seconds": manifest["setup_machine_seconds"],
            "source_health": source_health,
            "model_access": model_access,
        })
    else:
        _write_json(MANIFEST, manifest)
    return manifest


def _show_ready(manifest: dict, *, open_browser: bool = True) -> None:
    command = f"cd {shlex.quote(str(ROOT))} && bash run-v2.sh"
    instructions = (
        f"# Your agent is ready\n\n"
        f"Open your MLflow experiment:\n{manifest['experiment_url']}\n\n"
        "Edit the Python code under `plain/` with your coding agent or editor.\n"
        "To generate another brief, run this in your terminal:\n\n"
        f"```bash\n{command}\n```\n\n"
        "You can run that command again after each change. Briefs are saved in\n"
        "`output/`, and each run adds a trace to the same experiment.\n\n"
        "The five existing briefs:\n"
        + "".join(f"- {brief['path']}\n" for brief in manifest["local_briefs"])
    )
    ready_path = STATE_DIR / "ready.md"
    ready_path.write_text(instructions)
    example = manifest["local_briefs"][0]
    print("\nAGENT OUTPUT EXAMPLE:")
    print(f"An existing weekly brief: {example['path']}\n")
    print(Path(example["path"]).read_text().rstrip())
    print("\nSETUP COMPLETE")
    print(f"\nOpen your MLflow experiment:\n  {manifest['experiment_url']}")
    if open_browser:
        try:
            opened = webbrowser.open(manifest["experiment_url"], new=2)
        except (OSError, webbrowser.Error):
            opened = False
        if opened:
            print("Opened your experiment in your browser.")
        else:
            print("Could not open a browser. Open the experiment link above.")
    print("\nTo generate another brief, run:")
    print(f"  {command}")
    print("\nEdit plain/ and rerun that command after each change.")
    print("New briefs: output/ | New traces: the same experiment")
    print("\nFive existing briefs:")
    for brief in manifest["local_briefs"]:
        print(f"  {brief['label']}: {brief['path']}")
    print(f"\nThese instructions are saved in: {ready_path}")
    print(f"Manifest: {MANIFEST}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", type=Path)
    parser.add_argument("--no-open", action="store_true",
                        help="Print the experiment link without opening a browser.")
    args = parser.parse_args()
    try:
        assignment_path = args.assignment
        if assignment_path is None:
            from self_setup import provision
            assignment_path = provision(ROOT)
        manifest = setup(assignment_path)
    except Exception as error:
        parser.exit(1, f"V2 setup failed: {error}\n")
    _show_ready(manifest, open_browser=not args.no_open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
