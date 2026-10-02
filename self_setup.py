"""Create and resume one study attempt for the currently authenticated participant."""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import tarfile
import time
import uuid
from pathlib import Path

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from mlflow.entities.trace_location import UnityCatalog

from import_fixture import prepare_plan, upload_plan
from participant_setup import _configure_mlflow, _sha256, _verify_model_access, _write_json
from study_auth import browser_workspace_client


def _authenticate_self(config: dict) -> tuple[WorkspaceClient, str]:
    try:
        client = WorkspaceClient(profile=config.get('profile', 'dogfood'))
        if client.config.host.rstrip('/') == config['workspace_host'].rstrip('/'):
            client.current_user.me()
            return client, 'profile'
    except Exception:
        pass
    client = browser_workspace_client(config['workspace_host'])
    client.current_user.me()
    return client, 'external-browser'


def _check_identity(previous: dict, expected: dict) -> None:
    for key in ('principal', 'workspace_host', 'release_commit', 'fixture_sha256'):
        if previous.get(key) != expected[key]:
            raise RuntimeError(f'existing attempt conflicts on {key}; preserve it and use a fresh clone')


def _study_schema(client: WorkspaceClient, config: dict) -> None:
    name = f"{config['catalog']}.{config['schema']}"
    try:
        client.schemas.get(name)
    except NotFound as exc:
        raise RuntimeError(f'Shared study schema {name} is unavailable. '
                           'Ask the organizer to verify study access; '
                           'you do not need to create or own a catalog.') from exc


def _sql_name(*parts: str) -> str:
    return '.'.join('`' + part.replace('`', '``') + '`' for part in parts)


def _prepare_trace_tables(client: WorkspaceClient, config: dict, preparation: dict,
                          state_dir: Path) -> None:
    template = config.get('trace_table_template_prefix')
    if not template:
        return
    receipt_path = state_dir / 'trace-table-preparation.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    for suffix in ('otel_spans', 'otel_logs', 'otel_metrics', 'otel_annotations'):
        name = preparation['table_prefix'] + '_' + suffix
        full_name = f"{config['catalog']}.{config['schema']}.{name}"
        try:
            client.tables.get(full_name)
            continue
        except NotFound:
            pass
        entry = receipt.setdefault(name, {})
        if entry.get('statement_id'):
            result = client.statement_execution.get_statement(entry['statement_id'])
        else:
            destination = _sql_name(config['catalog'], config['schema'], name)
            source = _sql_name(config.get('trace_table_template_catalog', config['catalog']),
                               config.get('trace_table_template_schema', config['schema']),
                               template + '_' + suffix)
            # Warehouse SQL avoids the trace service's failing managed-storage create path.
            statement = f'CREATE TABLE IF NOT EXISTS {destination} LIKE {source}'
            entry.update(statement=statement, status='submitting')
            _write_json(receipt_path, receipt)
            result = client.statement_execution.execute_statement(
                statement=statement, warehouse_id=config['warehouse_id'], wait_timeout='50s')
        entry['statement_id'] = result.statement_id
        _write_json(receipt_path, receipt)
        deadline = time.monotonic() + 120
        while str(result.status.state).split('.')[-1] in {'PENDING', 'RUNNING'}:
            if time.monotonic() >= deadline:
                raise RuntimeError('Study table preparation is still running. Rerun setup to resume.')
            time.sleep(2)
            result = client.statement_execution.get_statement(result.statement_id)
        entry.update(status=str(result.status.state).split('.')[-1], result=result.as_dict())
        _write_json(receipt_path, receipt)
        if entry['status'] != 'SUCCEEDED':
            # A known failed statement can be retried; unknown delivery keeps its saved ID.
            entry.pop('statement_id', None)
            _write_json(receipt_path, receipt)
            raise RuntimeError(f"Study table preparation failed: {result.status.error}")
        client.tables.get(full_name)


def _usable_warehouse(client: WorkspaceClient, identity, preferred: str) -> str:
    groups = {group.display for group in identity.groups or [] if group.display}
    # Workspace's built-in users group includes every authenticated workspace user,
    # but current_user.me() may list only their account-level SCIM groups.
    groups.add('users')
    # Other accessible warehouses may not support the trace tables' Delta features.
    acl = client.warehouses.get_permissions(preferred).as_dict()
    if any((entry.get('user_name') == identity.user_name or entry.get('group_name') in groups)
           and any(p['permission_level'] in {'CAN_USE', 'CAN_MANAGE', 'IS_OWNER'}
                   for p in entry.get('all_permissions', []))
           for entry in acl.get('access_control_list', [])):
        return preferred
    raise RuntimeError(f'Shared study warehouse {preferred} is not available to this login. '
                       'Ask Adam to verify its CAN USE permission. '
                       'Setup will not select a different warehouse or change permissions.')


def _extract_fixture(root: Path, config: dict) -> Path:
    archive = root / 'baseline-fixture.tar.gz'
    if _sha256(archive) != config['fixture_archive_sha256']:
        raise RuntimeError('baseline fixture archive checksum mismatch')
    fixture = root / '.study' / 'fixture'
    if not fixture.exists():
        temporary = fixture.with_name('fixture-extract-' + uuid.uuid4().hex)
        temporary.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(temporary, filter='data')
        temporary.rename(fixture)
    if _sha256(fixture / 'manifest.json') != config['fixture_sha256']:
        raise RuntimeError('baseline fixture manifest checksum mismatch')
    return fixture


def provision(root: Path) -> Path:
    started = time.monotonic()
    state_dir = root / '.study'
    state_dir.mkdir(exist_ok=True)
    config = json.loads((root / 'study-config.json').read_text())
    release = json.loads((root / 'release.json').read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    tag_head = subprocess.check_output(['git', 'rev-parse', release['release_ref']],
                                       cwd=root, text=True).strip()
    if head != tag_head:
        raise RuntimeError('checkout does not match the selected immutable study release')
    if not (state_dir / 'manifest.json').exists():
        for name, digest in release['files'].items():
            if _sha256(root / name) != digest:
                raise RuntimeError(f'release file changed before first setup: {name}')
    client, auth_mode = _authenticate_self(config)
    identity = client.current_user.me()
    principal = identity.user_name
    if not principal:
        raise RuntimeError('authenticated participant has no workspace user name')
    expected = {'principal': principal, 'workspace_host': client.config.host,
                'release_commit': head, 'fixture_sha256': config['fixture_sha256']}
    assignment_path = state_dir / 'assignment.json'
    if assignment_path.exists():
        existing = json.loads(assignment_path.read_text())
        _check_identity(existing, expected)
        config.update({key: existing[key] for key in ('catalog', 'schema', 'warehouse_id')})
    else:
        preparation_path = state_dir / 'preparation.json'
        if preparation_path.exists():
            previous = json.loads(preparation_path.read_text())
            _check_identity(previous, expected)
            config.update({key: previous[key] for key in ('catalog', 'schema', 'warehouse_id')})
        else:
            config['warehouse_id'] = _usable_warehouse(client, identity, config['warehouse_id'])
    _study_schema(client, config)
    profile = config.get('profile', 'dogfood')
    _configure_mlflow(config, auth_mode, profile)
    _verify_model_access(client.serving_endpoints.get(release['model_endpoint']))
    warehouse = client.warehouses.get(config['warehouse_id'])
    warm_started = time.monotonic()
    if str(warehouse.state).split('.')[-1] != 'RUNNING':
        print('Starting the study warehouse. This preparation time is recorded separately.', flush=True)
        client.warehouses.start(config['warehouse_id']).result(timeout=datetime.timedelta(minutes=4))
    warm_seconds = round(time.monotonic() - warm_started, 3)
    if assignment_path.exists():
        _write_json(state_dir / 'latest-preparation.json', {
            'principal': principal, 'warehouse_warm_seconds': warm_seconds,
            'elapsed_seconds': round(time.monotonic() - started, 3), 'resumed_ready_attempt': True,
            'model_calls': 0,
        })
        return assignment_path
    fixture_dir = _extract_fixture(root, config)
    fixture = json.loads((fixture_dir / 'manifest.json').read_text())
    if fixture['source_sha256'] != release['source_sha256']:
        raise RuntimeError('fixture and agent source hashes differ')
    preparation_path = state_dir / 'preparation.json'
    if preparation_path.exists():
        preparation = json.loads(preparation_path.read_text())
        _check_identity(preparation, expected)
    else:
        attempt_id = uuid.uuid4().hex
        preparation = expected | {
            'attempt_id': attempt_id,
            'table_prefix': 'v2_' + hashlib.sha256(f'{principal}:{attempt_id}'.encode()).hexdigest()[:18],
            'experiment_name': f'/Users/{principal}/brief-agent-v2-{attempt_id}',
            'started_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'status': 'preparing', 'model_calls': 0,
            'catalog': config['catalog'], 'schema': config['schema'],
            'warehouse_id': config['warehouse_id'],
        }
        _write_json(preparation_path, preparation)
    print(f"Preparing your five baseline runs for {principal}. No model calls are made.", flush=True)
    _prepare_trace_tables(client, config, preparation, state_dir)
    experiment = mlflow.set_experiment(
        experiment_name=preparation['experiment_name'],
        trace_location=UnityCatalog(catalog_name=config['catalog'], schema_name=config['schema'],
                                    table_prefix=preparation['table_prefix']),
    )
    attempt_dir = state_dir / 'attempt'
    spans_table = experiment.trace_location.full_otel_spans_table_name
    prepare_plan(fixture_dir, attempt_dir, experiment.experiment_id, spans_table,
                 client.config.host, config['warehouse_id'])
    plan = upload_plan(fixture_dir, attempt_dir, profile if auth_mode == 'profile' else None,
                       workspace=client)
    assignment = expected | {
        'study_id': fixture['fixture_version'], 'attempt_id': preparation['attempt_id'],
        'organizer_dry_run': False, 'provisioning_mode': 'participant_self_service',
        'read_visibility': 'workspace', 'private_read_isolation': 'not_required_shared_reads',
        'write_visibility': 'shared_schema',
        'profile': profile, 'model_endpoint': release['model_endpoint'],
        'catalog': config['catalog'], 'schema': config['schema'],
        'table_prefix': preparation['table_prefix'], 'spans_table': spans_table,
        'warehouse_id': config['warehouse_id'], 'experiment_id': experiment.experiment_id,
        'experiment_name': experiment.name,
        'experiment_url': f'{client.config.host}/ml/experiments/{experiment.experiment_id}/traces',
        'release_ref': release['release_ref'], 'source_sha256': release['source_sha256'],
        'fixture_version': fixture['fixture_version'],
        'baseline_trace_ids': [item['destination_trace_id'] for item in plan['imports']],
        'baseline_captures': [
            {'label': item['label'], 'captured_at_utc': item['captured_at_utc'],
             'simulated_week': item['simulated_week'], 'trace_id': item['destination_trace_id'],
             'span_graph': item['span_graph']} for item in plan['imports']],
        'briefs': [{'label': item['label'], 'file': 'attempt/' + item['brief_file'],
                    'sha256': item['brief_sha256']} for item in plan['imports']],
    }
    _write_json(assignment_path, assignment)
    preparation.update(status='prepared', experiment_id=experiment.experiment_id,
                       elapsed_seconds=round(time.monotonic() - started, 3),
                       warehouse_warm_seconds=warm_seconds,
                       finished_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    _write_json(preparation_path, preparation)
    return assignment_path
