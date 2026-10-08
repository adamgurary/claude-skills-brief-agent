"""Pipeline nodes for the brief generator."""
import datetime
import logging
import os
import re
from pathlib import Path

import mlflow

from . import llm, prompts, research

logger = logging.getLogger(__name__)

_AGENT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = Path(os.environ.get("BRIEF_AGENT_OUTPUT_DIR", str(_AGENT_ROOT / "output")))

_PER_QUERY = 5
_MAX_QUERIES = 4
_ENRICH_TOP = 8


def _today() -> str:
    return datetime.date.today().strftime("%Y-%m-%d")


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s or "brief"


@mlflow.trace(name="plan_queries", span_type="CHAIN")
def plan_queries(state: dict) -> dict:
    topic = state.get("topic") or prompts.DEFAULT_TOPIC
    date = state.get("date") or _today()

    system = prompts.PLAN_SYSTEM
    user = prompts.PLAN_USER_TEMPLATE.format(topic=topic, date=date)
    raw = llm.complete(system, user, max_tokens=12_000)

    queries = []
    for line in (raw or "").splitlines():
        q = line.strip().lstrip("-*0123456789. ").strip().strip('"')
        if q and 3 < len(q) < 200 and not q.lower().startswith(("thinking", "[{'", '[{"')):
            queries.append(q)
    if not queries:
        queries = [topic, "Claude Code plugins", "Claude Code skills"]
    queries = queries[:_MAX_QUERIES]

    logger.info("planned %d queries: %s", len(queries), queries)
    return {"topic": topic, "date": date, "search_queries": queries}


def _hn_line(r: dict) -> str:
    return (f"- [{r['title']}]({r['url']}) | {r['source']} | {r['date']} | "
            f"{r['points']} points, {r['comments']} comments")


def _gh_line(r: dict) -> str:
    desc = f" | {r['description']}" if r.get("description") else ""
    return f"- [{r['title']}]({r['url']}) | {r['source']} | updated {r['updated']} | {r['stars']} stars{desc}"


@mlflow.trace(name="web_research", span_type="CHAIN")
def web_research(state: dict) -> dict:
    queries = state.get("search_queries") or [prompts.DEFAULT_TOPIC]

    results = []
    seen_urls = set()

    for q in queries:
        for r in research.hn_search.invoke({"query": q, "max_results": _PER_QUERY,
                                           "start": state.get("research_start"),
                                           "end": state.get("research_end")}):
            url = r.get("url")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            results.append(r)
        for r in research.github_search.invoke(
            {"query": q, "max_results": _PER_QUERY, "sort": "stars",
             "start": state.get("research_start"), "end": state.get("research_end")}
        ):
            url = r.get("url")
            if url and url in seen_urls:
                continue
            if url:
                seen_urls.add(url)
            results.append(r)

    hn = [r for r in results if r.get("source") == "Hacker News"]
    gh = [r for r in results if r.get("source") == "GitHub"]
    findings = _format_findings(hn, gh)
    logger.info("research gathered %d results (%d HN, %d GH)", len(results), len(hn), len(gh))
    return {"results": results, "findings": findings, "result_count": len(results)}


def _format_findings(hn: list, gh: list) -> str:
    sections = []
    if hn:
        sections.append("## Hacker News (recent discussion / launches)\n" + "\n".join(_hn_line(r) for r in hn))
    if gh:
        sections.append("## GitHub (popular / active projects)\n" + "\n".join(_gh_line(r) for r in gh))
    return "\n\n".join(sections) if sections else "No results found."


def _rank_key(r: dict):
    return r.get("stars") or r.get("points") or 0


@mlflow.trace(name="enrich_findings", span_type="CHAIN")
def enrich_findings(state: dict) -> dict:
    results = state.get("results") or []
    top = sorted(results, key=_rank_key, reverse=True)[:_ENRICH_TOP]

    enriched = []
    for r in top:
        content = ""
        if r.get("source") == "GitHub":
            content = research.github_readme.invoke(
                {"full_name": r.get("title", ""), "max_chars": 4000}
            )
        if not content and r.get("url"):
            content = research.web_fetch.invoke({"url": r["url"], "max_chars": 3000})
        if content:
            enriched.append({**r, "content": content})

    logger.info("enriched %d of %d top items with fetched content", len(enriched), len(top))
    return {"enriched": enriched, "enriched_count": len(enriched)}


def _strip_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n", "", t)
        if t.endswith("```"):
            t = t[:-3]
    return t.strip()


def _ensure_source_link(md: str, results: list) -> str:
    if re.search(r"\]\(https?://[^)]+\)", md):
        return md
    for result in results:
        url = result.get("url") or ""
        if url.startswith(("http://", "https://")):
            title = result.get("title") or "Source"
            return f"{md.rstrip()}\n\n## Source\n- [{title}]({url})"
    return md


def _fallback_brief(date: str, results: list) -> str:
    lines = [
        "# This week in Claude Code skills & plugins",
        "",
        f"_Brief generated {date}_",
        "",
        "## Notable projects and releases",
    ]
    for result in results[:8]:
        title = result.get("title") or "Untitled"
        url = result.get("url") or ""
        detail = result.get("description") or "Worth a look."
        lines.append(f"- **[{title}]({url})** {detail}")
    if not results:
        lines.append("- No results were available.")
    return "\n".join(lines)


def _format_enriched(enriched: list) -> str:
    if not enriched:
        return "(no deep content could be fetched for the top items)"
    blocks = []
    for r in enriched:
        if r.get("source") == "GitHub":
            head = f"### {r['title']} ({r.get('stars', 0)} stars, updated {r.get('updated', '?')})\nLink: {r['url']}"
        else:
            head = f"### {r['title']} ({r.get('points', 0)} points on Hacker News, {r.get('date', '?')})\nLink: {r['url']}"
        blocks.append(f"{head}\nContent:\n{r.get('content', '').strip()}")
    return "\n\n".join(blocks)


@mlflow.trace(name="write_brief", span_type="CHAIN")
def write_brief(state: dict) -> dict:
    topic = state.get("topic") or prompts.DEFAULT_TOPIC
    date = state.get("date") or _today()
    findings = state.get("findings") or "No results found."
    enriched_block = _format_enriched(state.get("enriched") or [])

    system = prompts.WRITE_SYSTEM.format(
        title="This week in Claude Code skills & plugins", date=date,
    )
    user = prompts.WRITE_USER_TEMPLATE.format(
        topic=topic, date=date, findings=findings, enriched=enriched_block,
    )
    results = state.get("results") or []
    draft = _strip_fence(llm.complete(system, user, max_tokens=12_000))
    if not draft:
        draft = _fallback_brief(date, results)
    brief = _ensure_source_link(draft, results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{date}-{_slug(topic)}.md"
    out_path.write_text(brief + "\n", encoding="utf-8")
    logger.info("wrote brief -> %s (%d chars)", out_path, len(brief))
    return {"brief": brief, "output_path": str(out_path)}
