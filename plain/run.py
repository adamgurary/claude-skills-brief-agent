"""Command-line entry point for the Claude Code brief generator.

    generate_brief(topic=None, date=None)

runs the pipeline end-to-end, writes the brief to
output/{date}-{topic-slug}.md, and returns the final state.

CLI:
    python run.py
    python run.py --topic "Claude Code MCP servers"
    python run.py --date 2026-07-22
"""
import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import mlflow
from mlflow.entities.trace_location import UnityCatalog

_HERE = Path(__file__).resolve().parent
_STUDY_MANIFEST = _HERE.parent / ".study" / "manifest.json"
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from brief_agent import prompts  # noqa: E402
from brief_agent.graph import build_graph  # noqa: E402
from render import render_and_open  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("brief_agent.run")


def generate_brief(topic: str | None = None, date: str | None = None,
                   research_start: str | None = None,
                   research_end: str | None = None) -> dict:
    """Run the brief pipeline. Returns the final graph state (includes `brief`
    and `output_path`)."""
    # Resolve defaults before entering the traced function. Otherwise MLflow
    # records the caller's None arguments as the root inputs even though the
    # graph runs with a real topic and date.
    return _run_brief(
        topic=topic or prompts.DEFAULT_TOPIC,
        date=date or datetime.date.today().strftime("%Y-%m-%d"),
        research_start=research_start,
        research_end=research_end,
    )


@mlflow.trace(name="brief_agent", span_type="CHAIN")
def _run_brief(topic: str, date: str, research_start: str | None,
               research_end: str | None) -> dict:
    initial = {"topic": topic, "date": date,
               "research_start": research_start, "research_end": research_end}
    app = build_graph()
    final = app.invoke(initial)
    return final


def configure_tracing() -> None:
    if _STUDY_MANIFEST.is_file() and os.environ.get("BRIEF_AGENT_CAPTURE_LOCAL") != "1":
        assignment = json.loads(_STUDY_MANIFEST.read_text())
        common = {
            "MLFLOW_TRACING_SQL_WAREHOUSE_ID": assignment["warehouse_id"],
            "BRIEF_AGENT_EXPERIMENT": assignment["experiment_name"],
            "BRIEF_AGENT_TRACE_CATALOG": assignment["catalog"],
            "BRIEF_AGENT_TRACE_SCHEMA": assignment["schema"],
            "BRIEF_AGENT_TRACE_PREFIX": assignment["table_prefix"],
        }
        if assignment["auth_mode"] == "profile":
            profile = assignment["profile"]
            common.update({
                "DATABRICKS_CONFIG_PROFILE": profile,
                "BRIEF_AGENT_PROFILE": profile,
                "MLFLOW_TRACKING_URI": f"databricks://{profile}",
            })
        else:
            common.update({
                "DATABRICKS_HOST": assignment["workspace_host"],
                "DATABRICKS_AUTH_TYPE": "external-browser",
                "MLFLOW_TRACKING_URI": "databricks",
                "BRIEF_AGENT_WORKSPACE_HOST": assignment["workspace_host"],
            })
        os.environ.update(common)
    experiment_name = os.environ.get("BRIEF_AGENT_EXPERIMENT")
    if os.environ.get("BRIEF_AGENT_CAPTURE_LOCAL") == "1":
        if not experiment_name:
            raise RuntimeError("fixture capture requires BRIEF_AGENT_EXPERIMENT")
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(experiment_name=experiment_name)
        mlflow.langchain.autolog()
        return
    catalog = os.environ.get("BRIEF_AGENT_TRACE_CATALOG")
    schema = os.environ.get("BRIEF_AGENT_TRACE_SCHEMA")
    prefix = os.environ.get("BRIEF_AGENT_TRACE_PREFIX")
    if not all((experiment_name, catalog, schema, prefix)):
        raise RuntimeError("V2 requires an assigned experiment and UC trace location")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "databricks"))
    mlflow.set_experiment(
        experiment_name=experiment_name,
        trace_location=UnityCatalog(catalog_name=catalog, schema_name=schema,
                                    table_prefix=prefix),
    )
    mlflow.langchain.autolog()


def _main() -> None:
    ap = argparse.ArgumentParser(description="Generate a Claude Code skills/plugins brief.")
    ap.add_argument("--topic", help="Topic to brief on (default: cool new Claude Code skills and plugins)")
    ap.add_argument("--date", help="Brief date YYYY-MM-DD (default today)")
    ap.add_argument("--no-open", action="store_true", help="Don't render/open the brief in the browser")
    ap.add_argument("--research-start", help=argparse.SUPPRESS)
    ap.add_argument("--research-end", help=argparse.SUPPRESS)
    a = ap.parse_args()

    if bool(a.research_start) != bool(a.research_end):
        ap.error("research-start and research-end must be supplied together")
    configure_tracing()
    final = generate_brief(topic=a.topic, date=a.date,
                           research_start=a.research_start,
                           research_end=a.research_end)
    mlflow.flush_trace_async_logging()
    topic = final.get("topic") or (a.topic or "")
    html_path = render_and_open(
        final.get("brief", ""), final.get("output_path"), topic, open_browser=not a.no_open,
    )
    print("\n" + "=" * 70)
    print("BRIEF (markdown):", final.get("output_path"))
    if html_path:
        print("BRIEF (opened in browser):", html_path)
    print("Results gathered:", final.get("result_count"))
    print("Brief chars:", len(final.get("brief") or ""))
    print("=" * 70)


if __name__ == "__main__":
    _main()
