"""Hacker News and GitHub research helpers."""
from __future__ import annotations

import base64
import datetime
import logging
import os

import mlflow
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"}


class SourceRequestError(RuntimeError):
    """A failed source request cannot be treated as an empty search."""


def _get_json(source: str, url: str, *, params: dict | None = None,
              headers: dict | None = None) -> dict:
    try:
        response = requests.get(url, params=params, headers=headers or _HEADERS,
                                timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("expected a JSON object")
        return payload
    except (requests.RequestException, ValueError) as error:
        response = getattr(error, "response", None)
        status = getattr(response, "status_code", None)
        remaining = response.headers.get("X-RateLimit-Remaining") if response is not None else None
        retry_after = response.headers.get("Retry-After") if response is not None else None
        raise SourceRequestError(f"{source} request failed (HTTP {status or 'unknown'}): "
                                 f"{type(error).__name__}; rate-limit remaining={remaining}; "
                                 f"retry-after={retry_after}") from error


def _window(start: str | None, end: str | None) -> tuple[datetime.date, datetime.date] | None:
    if start is None and end is None:
        return None
    if not start or not end:
        raise ValueError("a research window requires both dates")
    first, last = datetime.date.fromisoformat(start), datetime.date.fromisoformat(end)
    if first > last:
        raise ValueError("research window ends before it starts")
    return first, last


def _github_headers() -> dict:
    """Build GitHub API headers, with optional token authentication."""
    headers = {**_HEADERS, "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@tool
@mlflow.trace(name="hn_search", span_type="TOOL")
def hn_search(query: str, max_results: int = 6,
              start: str | None = None, end: str | None = None) -> list:
    """Search Hacker News stories (most recent first) via the Algolia API.

    Returns search hits. Raises SourceRequestError on a failed request."""
    params = {"query": query, "tags": "story", "hitsPerPage": max_results}
    window = _window(start, end)
    if window:
        first, last = window
        after = int(datetime.datetime.combine(first, datetime.time.min,
                                              tzinfo=datetime.timezone.utc).timestamp())
        before = int(datetime.datetime.combine(last + datetime.timedelta(days=1),
                                               datetime.time.min,
                                               tzinfo=datetime.timezone.utc).timestamp())
        params["numericFilters"] = f"created_at_i>={after},created_at_i<{before}"
    hits = _get_json("Hacker News", "https://hn.algolia.com/api/v1/search_by_date",
                     params=params).get("hits")
    if not isinstance(hits, list):
        raise SourceRequestError("Hacker News response omitted search hits")

    out = []
    for h in hits:
        try:
            object_id = h.get("objectID")
            out.append({
                "title": h.get("title") or h.get("story_title") or "(untitled)",
                "url": h.get("url") or (f"https://news.ycombinator.com/item?id={object_id}" if object_id else ""),
                "points": h.get("points") or 0,
                "comments": h.get("num_comments") or 0,
                "date": (h.get("created_at") or "")[:10] or "unknown-date",
                "source": "Hacker News",
            })
        except Exception:
            continue
    return out


@tool
@mlflow.trace(name="github_search", span_type="TOOL")
def github_search(query: str, max_results: int = 6, sort: str = "stars",
                  start: str | None = None, end: str | None = None) -> list:
    """Search GitHub repositories via the public search API.

    sort: "stars" (most popular) or "updated" (most recently active). Returns a
    list of {title, url, description, stars, updated, source} dicts."""
    window = _window(start, end)
    if window:
        query = f"{query} pushed:{window[0].isoformat()}..{window[1].isoformat()}"
    items = _get_json(
        "GitHub search", "https://api.github.com/search/repositories",
        params={"q": query, "sort": sort, "order": "desc", "per_page": max_results},
        headers=_github_headers(),
    ).get("items")
    if not isinstance(items, list):
        raise SourceRequestError("GitHub search response omitted items")

    out = []
    for it in items:
        try:
            out.append({
                "title": it.get("full_name") or "(unnamed)",
                "url": it.get("html_url") or "",
                "description": (it.get("description") or "").strip(),
                "stars": it.get("stargazers_count") or 0,
                "updated": (it.get("pushed_at") or "")[:10] or "unknown-date",
                "source": "GitHub",
            })
        except Exception:
            continue
    return out


@tool
@mlflow.trace(name="github_readme", span_type="TOOL")
def github_readme(full_name: str, max_chars: int = 4000) -> str:
    """Fetch a repo's README text via the GitHub API (`repos/{full_name}/readme`).

    `full_name` is "owner/repo". Returns the decoded README (light markdown
    stripped to plain-ish text), truncated to max_chars. A missing README is empty."""
    if not full_name or "/" not in full_name:
        return ""
    try:
        payload = _get_json("GitHub README", f"https://api.github.com/repos/{full_name}/readme",
                            headers=_github_headers())
    except SourceRequestError as error:
        if "HTTP 404" in str(error):
            return ""
        raise

    try:
        content = payload.get("content") or ""
        if payload.get("encoding") == "base64" and content:
            text = base64.b64decode(content).decode("utf-8", errors="replace")
        else:
            text = content
        cleaned = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("<!--") or s.startswith("![") or "img.shields.io" in s:
                continue
            cleaned.append(line)
        out = "\n".join(cleaned).strip()
        return out[:max_chars]
    except Exception:
        logger.exception("Failed to decode README for %s", full_name)
        return ""


@tool
@mlflow.trace(name="web_fetch", span_type="TOOL")
def web_fetch(url: str, max_chars: int = 3500) -> str:
    """Fetch a URL and return its extracted text, truncated. Returns "" on any failure.

    Best-effort enrichment for a single top link. Not every page allows scraping,
    so callers must tolerate an empty string."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("web_fetch failed for %s: %s", url, e)
        return ""

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars]
    except Exception:
        logger.exception("Failed to parse web_fetch content for %s", url)
        return ""
