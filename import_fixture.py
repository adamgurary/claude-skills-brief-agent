"""Import an immutable five-run fixture into one assigned UC trace destination."""

from __future__ import annotations

import argparse
import base64
import datetime
import email.utils
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import mlflow
import requests
from databricks.sdk import WorkspaceClient
from google.protobuf.json_format import MessageToDict
from mlflow.entities import Trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _replace_identifier(value: str, source: str, destination: str) -> str:
    if not source:
        return value
    if value == source:
        return destination
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return value.replace(source, destination)
    if isinstance(parsed, str) and parsed == source:
        return json.dumps(destination)
    return value.replace(source, destination)


def _replace_entire_value(value: str, destination: str) -> str:
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return destination
    return json.dumps(destination) if isinstance(parsed, str) else destination


def _remap_attributes(attributes, old_trace: str, new_trace: str,
                      old_experiment: str, new_experiment: str) -> None:
    for attribute in attributes:
        # UC stores an absent AnyValue as an empty string; encode null explicitly.
        if attribute.value.WhichOneof("value") is None:
            attribute.value.string_value = "null"
        if not attribute.value.HasField("string_value"):
            continue
        value = attribute.value.string_value
        if attribute.key == "mlflow.traceRequestId":
            value = _replace_entire_value(value, new_trace)
        elif attribute.key == "mlflow.experimentId":
            value = _replace_entire_value(value, new_experiment)
        else:
            value = _replace_identifier(value, old_trace, new_trace)
        attribute.value.string_value = value


def build_payload(source: Trace, destination_trace_id: str,
                  experiment_id: str) -> tuple[bytes, int]:
    trace_bytes = bytes.fromhex(destination_trace_id.rsplit("/", 1)[1])
    source_experiment = str(source.info.trace_metadata.get("mlflow.experimentId", ""))
    source_spans = [span.to_otel_proto() for span in source.data.spans]
    if not source_spans:
        raise RuntimeError("source trace has no spans")
    source_trace_bytes = source_spans[0].trace_id
    span_ids = {span.span_id: os.urandom(8) for span in source_spans}
    if len(span_ids) != len(source_spans):
        raise RuntimeError("source trace contains duplicate span IDs")
    request = ExportTraceServiceRequest()
    scope = request.resource_spans.add().scope_spans.add()
    for span in source_spans:
        old_span_id = span.span_id
        old_parent_id = span.parent_span_id
        span.trace_id = trace_bytes
        span.span_id = span_ids[old_span_id]
        if old_parent_id:
            if old_parent_id not in span_ids:
                raise RuntimeError("source trace has a missing parent")
            span.parent_span_id = span_ids[old_parent_id]
        _remap_attributes(span.attributes, source.info.trace_id,
                          destination_trace_id, source_experiment, experiment_id)
        for event in span.events:
            _remap_attributes(event.attributes, source.info.trace_id,
                              destination_trace_id, source_experiment, experiment_id)
        for link in span.links:
            if link.trace_id == source_trace_bytes:
                link.trace_id = trace_bytes
            if link.span_id in span_ids:
                link.span_id = span_ids[link.span_id]
            _remap_attributes(link.attributes, source.info.trace_id,
                              destination_trace_id, source_experiment, experiment_id)
        scope.spans.add().CopyFrom(span)
    return request.SerializeToString(), len(source_spans)


def prepare_plan(fixture_dir: Path, attempt_dir: Path, experiment_id: str,
                 spans_table: str, workspace_host: str,
                 warehouse_id: str) -> dict:
    fixture_dir = fixture_dir.resolve()
    attempt_dir = attempt_dir.resolve()
    fixture_path = fixture_dir / "manifest.json"
    fixture = json.loads(fixture_path.read_text())
    if fixture.get("status") != "captured" or len(fixture.get("runs", [])) != 5:
        raise RuntimeError("only a complete five-run fixture can be imported")
    if len({run["trace_id"] for run in fixture["runs"]}) != 5:
        raise RuntimeError("source fixture trace IDs are not distinct")
    if not spans_table.endswith("_otel_spans") or spans_table.count(".") != 2:
        raise RuntimeError("destination must be a fully qualified UC spans table")
    attempt_dir.mkdir(parents=True, exist_ok=True)
    plan_path = attempt_dir / "import-plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        expected = (plan["fixture_sha256"], plan["experiment_id"],
                    plan["spans_table"], plan["workspace_host"], plan["warehouse_id"])
        actual = (_sha256(fixture_path), experiment_id, spans_table, workspace_host,
                  warehouse_id)
        if expected != actual:
            raise RuntimeError("existing import plan has a version or destination conflict")
        for record in plan["imports"]:
            payload_path = attempt_dir / record["payload_file"]
            if _sha256(payload_path) != record["payload_sha256"]:
                raise RuntimeError(f"persisted payload changed: {record['label']}")
            payload = ExportTraceServiceRequest.FromString(payload_path.read_bytes())
            graph = {span.span_id.hex(): span.parent_span_id.hex() or None
                     for resource in payload.resource_spans
                     for scope in resource.scope_spans for span in scope.spans}
            if record.get("span_graph", graph) != graph:
                raise RuntimeError(f"persisted span graph changed: {record['label']}")
            record["span_graph"] = graph
        _write_json(plan_path, plan)
        return plan
    plan = {
        "fixture_sha256": _sha256(fixture_path),
        "fixture_version": fixture["fixture_version"],
        "source_sha256": fixture["source_sha256"],
        "experiment_id": experiment_id,
        "spans_table": spans_table,
        "workspace_host": workspace_host,
        "warehouse_id": warehouse_id,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "imports": [],
    }
    payload_dir = attempt_dir / "otlp"
    payload_dir.mkdir(exist_ok=True)
    briefs_dir = attempt_dir / "briefs"
    briefs_dir.mkdir(exist_ok=True)
    for run in fixture["runs"]:
        trace_path = fixture_dir / run["trace_file"]
        brief_path = fixture_dir / run["brief_file"]
        if _sha256(trace_path) != run["trace_file_sha256"]:
            raise RuntimeError(f"source trace checksum mismatch: {run['label']}")
        if _sha256(brief_path) != run["brief_file_sha256"]:
            raise RuntimeError(f"source brief checksum mismatch: {run['label']}")
        source = Trace.from_json(trace_path.read_text())
        if source.info.trace_id != run["trace_id"]:
            raise RuntimeError(f"source trace ID mismatch: {run['label']}")
        destination_id = f"trace:/{spans_table.removesuffix('_otel_spans')}/{uuid.uuid4().hex}"
        payload, span_count = build_payload(source, destination_id, experiment_id)
        payload_path = payload_dir / f"{run['label']}.pb"
        payload_path.write_bytes(payload)
        output_brief = briefs_dir / f"{run['label']}.md"
        output_brief.write_text(
            brief_path.read_text().replace(run["trace_id"], destination_id)
        )
        plan["imports"].append({
            "label": run["label"],
            "simulated_week": [run["research_start"], run["research_end"]],
            "captured_at_utc": run["captured_at_utc"],
            "source_trace_id": run["trace_id"],
            "destination_trace_id": destination_id,
            "source_trace_file": run["trace_file"],
            "payload_file": str(payload_path.relative_to(attempt_dir)),
            "payload_sha256": _sha256(payload_path),
            "span_count": span_count,
            "span_graph": {span.span_id.hex(): span.parent_span_id.hex() or None
                           for resource in ExportTraceServiceRequest.FromString(payload).resource_spans
                           for scope in resource.scope_spans for span in scope.spans},
            "brief_file": str(output_brief.relative_to(attempt_dir)),
            "brief_sha256": _sha256(output_brief),
            "state": "planned",
            "upload_attempts": 0,
        })
    _write_json(plan_path, plan)
    return plan


def _readback(trace_id: str, expected_spans: int, attempts: int = 6) -> Trace | None:
    # This loop owns retries; nested SDK HTTP retries can otherwise multiply waits by minutes.
    transport = {"MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0", "MLFLOW_HTTP_REQUEST_TIMEOUT": "30"}
    previous = {key: os.environ.get(key) for key in transport}
    os.environ.update(transport)
    try:
        for index in range(attempts):
            try:
                trace = mlflow.get_trace(trace_id)
            except Exception:
                trace = None
            if trace is not None:
                if len(trace.data.spans) > expected_spans:
                    raise RuntimeError(f"imported graph has unexpected extra spans: {trace_id}")
                if len(trace.data.spans) == expected_spans:
                    return trace
            if index + 1 < attempts:
                time.sleep(min(2 ** index, 8))
        return None
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _rate_limit_rejected(record: dict) -> bool:
    # Dogfood has returned an explicit rejection inside an HTTP 500 OTLP response.
    return record.get("http_status") == 429 or (
        record.get("http_status") == 500
        and "REQUEST_LIMIT_EXCEEDED: Your request was rejected" in
        record.get("response_body", "")
    )


def _retry_delay(record: dict) -> float:
    delay = 30.0
    started = record.get("upload_started_utc")
    started_at = (datetime.datetime.fromisoformat(started).timestamp()
                  if started else time.time())
    value = record.get("response_retry_after")
    if value:
        try:
            delay = max(delay, float(value))
        except ValueError:
            try:
                deadline = email.utils.parsedate_to_datetime(value).timestamp()
                delay = max(delay, deadline - started_at)
            except (ValueError, TypeError, OverflowError):
                pass
    elapsed = time.time() - started_at
    return max(0.0, delay - elapsed)


def _begin_upload(record: dict) -> None:
    if record["upload_attempts"]:
        record.setdefault("upload_history", []).append({
            key: value for key, value in record.items()
            if key.startswith(("response_", "upload_", "transport_"))
            and key != "upload_history"
        } | {"state": record["state"], "http_status": record.get("http_status")})
        for key in list(record):
            if key.startswith(("response_", "transport_")) or key == "http_status":
                del record[key]
    record["state"] = "uploading"
    record["upload_attempts"] += 1
    record["upload_started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _attribute_value(value: dict | None, *, decode_json: bool = True):
    """Compare OTLP values with the UC service's JSON attribute representation."""
    if not value:
        return None
    if "string_value" in value:
        text = value["string_value"]
        if decode_json:
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                pass
        return text
    if "kvlist_value" in value:
        return {item["key"]: _attribute_value(item.get("value"), decode_json=False)
                for item in value["kvlist_value"].get("values", [])}
    if "array_value" in value:
        return [_attribute_value(item, decode_json=False)
                for item in value["array_value"].get("values", [])]
    if "int_value" in value:
        return int(value["int_value"])
    if "double_value" in value:
        return value["double_value"]
    if "bool_value" in value:
        return value["bool_value"]
    return value


def _span_signature(span) -> dict:
    value = MessageToDict(span, preserving_proto_field_name=True,
                          always_print_fields_with_no_presence=True)

    def normalize(item):
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if not isinstance(item, dict):
            return item
        return {key: ({attribute["key"]: _attribute_value(attribute.get("value"))
                       for attribute in child} if key == "attributes" else normalize(child))
                for key, child in item.items()}

    return normalize(value)


def verify_import(record: dict, source: Trace, destination: Trace, payload: bytes) -> None:
    if destination.info.trace_id != record["destination_trace_id"]:
        raise RuntimeError(f"destination trace ID mismatch: {record['label']}")
    expected = ExportTraceServiceRequest.FromString(payload)
    planned = {span.span_id: span for resource in expected.resource_spans
               for scope in resource.scope_spans for span in scope.spans}
    actual = {span.to_otel_proto().span_id: span.to_otel_proto()
              for span in destination.data.spans}
    if len(actual) != len(destination.data.spans) or actual.keys() != planned.keys():
        raise RuntimeError(f"imported span IDs differ: {record['label']}")
    for span_id, planned_span in planned.items():
        left, right = _span_signature(planned_span), _span_signature(actual[span_id])
        # The service may add attributes; every uploaded attribute must survive.
        right["attributes"] = {key: right["attributes"].get(key)
                               for key in left["attributes"]}
        if left != right:
            fields = [key for key in left if left[key] != right.get(key)]
            raise RuntimeError(
                f"imported span differs: {record['label']} {planned_span.name} {fields}"
            )
    if len(source.data.spans) != len(planned):
        raise RuntimeError(f"source span count differs: {record['label']}")


class RateLimitRejected(RuntimeError):
    """A server-confirmed rejection that permits bounded retry with stable IDs."""


def upload_plan(fixture_dir: Path, attempt_dir: Path, profile: str | None, *,
                readback_attempts: int = 6, workspace: WorkspaceClient | None = None) -> dict:
    records = json.loads((attempt_dir / "import-plan.json").read_text())["imports"]
    attempt_limits = {r["destination_trace_id"]: r["upload_attempts"] + 3 for r in records}
    # Budgets belong to individual traces and this invocation; saved rejections stay resumable.
    rounds = 1 + 3 * len(records)
    for retry in range(rounds):
        try:
            return _upload_plan_once(fixture_dir, attempt_dir, profile,
                                     readback_attempts=readback_attempts, workspace=workspace,
                                     attempt_limits=attempt_limits)
        except RateLimitRejected:
            if retry == rounds - 1:
                raise
    raise AssertionError("retry loop exhausted")


def _upload_plan_once(fixture_dir: Path, attempt_dir: Path, profile: str | None, *,
                      readback_attempts: int = 6, workspace: WorkspaceClient | None = None,
                      attempt_limits: dict[str, int]) -> dict:
    fixture_dir = fixture_dir.resolve()
    attempt_dir = attempt_dir.resolve()
    plan_path = attempt_dir / "import-plan.json"
    plan = json.loads(plan_path.read_text())
    workspace = workspace or WorkspaceClient(profile=profile)
    if workspace.config.host.rstrip("/") != plan["workspace_host"].rstrip("/"):
        raise RuntimeError("authenticated workspace does not match import plan")
    if profile:
        os.environ["DATABRICKS_CONFIG_PROFILE"] = profile
    os.environ["MLFLOW_TRACING_SQL_WAREHOUSE_ID"] = plan["warehouse_id"]
    mlflow.set_tracking_uri(f"databricks://{profile}" if profile else "databricks")
    for record in plan["imports"]:
        if _sha256(attempt_dir / record["payload_file"]) != record["payload_sha256"]:
            raise RuntimeError(f"payload changed: {record['label']}")
        payload = (attempt_dir / record["payload_file"]).read_bytes()
        source = Trace.from_json((fixture_dir / record["source_trace_file"]).read_text())
        if record["state"] == "visible":
            existing = _readback(record["destination_trace_id"], record["span_count"],
                                 attempts=readback_attempts)
            if existing is None:
                raise RuntimeError(f"previously visible trace is missing: {record['label']}")
            verify_import(record, source, existing, payload)
            (attempt_dir / f"{record['label']}-readback.json").write_text(existing.to_json())
            continue
        # A durable zero-attempt record cannot have reached the server. Avoid a
        # warehouse query for that guaranteed miss; resumed delivery still reads back.
        never_uploaded = record["state"] == "planned" and record["upload_attempts"] == 0
        confirmed_rejection = record["state"] == "rejected" and _rate_limit_rejected(record)
        existing = (None if never_uploaded or confirmed_rejection else
                    _readback(record["destination_trace_id"], record["span_count"],
                              attempts=readback_attempts))
        if existing is not None:
            verify_import(record, source, existing, payload)
            (attempt_dir / f"{record['label']}-readback.json").write_text(existing.to_json())
            record["state"] = "visible"
            record["visible_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _write_json(plan_path, plan)
            continue
        retry_rejected = (record["state"] in {"uncertain", "rejected"}
                          and _rate_limit_rejected(record))
        if retry_rejected:
            record["state"] = "rejected"
            _write_json(plan_path, plan)
            delay = _retry_delay(record)
            if record["upload_attempts"] >= attempt_limits[record["destination_trace_id"]] or delay > 60:
                raise RuntimeError(
                    f"rate-limit rejection for {record['label']}; "
                    f"attempts={record['upload_attempts']}, retry delay={delay:.0f}s"
                )
            time.sleep(delay)
        elif record["state"] != "planned" or record["upload_attempts"] != 0:
            raise RuntimeError(f"delivery uncertain for {record['label']}; readback did not resolve it")
        _begin_upload(record)
        _write_json(plan_path, plan)
        try:
            response = requests.post(
                workspace.config.host.rstrip("/") + "/api/2.0/otel/v1/traces",
                headers={**workspace.config.authenticate(),
                         "Content-Type": "application/x-protobuf",
                         "X-Databricks-UC-Table-Name": plan["spans_table"]},
                data=(attempt_dir / record["payload_file"]).read_bytes(),
                timeout=45,
            )
            record["http_status"] = response.status_code
            record["response_content_type"] = response.headers.get("Content-Type")
            record["response_retry_after"] = response.headers.get("Retry-After")
            record["response_sha256"] = hashlib.sha256(response.content).hexdigest()
            record["response_base64"] = base64.b64encode(response.content).decode("ascii")
            record["response_body"] = response.text[:4000] if response.status_code != 200 else ""
            record["response_request_id"] = response.headers.get("x-request-id")
            record["state"] = ("accepted" if response.status_code == 200 else
                               "rejected" if _rate_limit_rejected(record) else "uncertain")
        except requests.RequestException as error:
            record["transport_error"] = type(error).__name__
            record["state"] = "uncertain"
        _write_json(plan_path, plan)
        if record["state"] == "rejected":
            # A confirmed rejection cannot become visible; honor backoff before retrying.
            raise RateLimitRejected(
                f"rate-limit rejection for {record['label']}; attempts={record['upload_attempts']}"
            )
        existing = _readback(record["destination_trace_id"], record["span_count"],
                             attempts=readback_attempts)
        if existing is None:
            if record["state"] == "rejected":
                raise RateLimitRejected(
                    f"rate-limit rejection for {record['label']}; "
                    f"attempts={record['upload_attempts']}"
                )
            raise RuntimeError(
                f"upload {record['state']} (HTTP {record.get('http_status')}), but "
                f"{record['label']} is not visible yet; resume readback"
            )
        verify_import(record, source, existing, payload)
        (attempt_dir / f"{record['label']}-readback.json").write_text(existing.to_json())
        record["state"] = "visible"
        record["visible_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json(plan_path, plan)
    if len(plan["imports"]) != 5 or any(item["state"] != "visible" for item in plan["imports"]):
        raise RuntimeError("fewer than five imported traces are visible")
    if len({item["destination_trace_id"] for item in plan["imports"]}) != 5:
        raise RuntimeError("destination trace IDs are not distinct")
    plan["status"] = "verified"
    _write_json(plan_path, plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--spans-table", required=True)
    parser.add_argument("--workspace-host", required=True)
    parser.add_argument("--warehouse-id", required=True)
    parser.add_argument("--profile", default="dogfood")
    args = parser.parse_args()
    prepare_plan(args.fixture_dir, args.attempt_dir, args.experiment_id,
                 args.spans_table, args.workspace_host, args.warehouse_id)
    plan = upload_plan(args.fixture_dir, args.attempt_dir, args.profile)
    print(json.dumps({"status": plan["status"], "trace_ids":
                      [item["destination_trace_id"] for item in plan["imports"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
