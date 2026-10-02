"""State shared by the brief pipeline."""
from typing import List, TypedDict


class BriefState(TypedDict, total=False):
    topic: str
    date: str
    research_start: str | None
    research_end: str | None
    search_queries: List[str]
    results: List[dict]
    findings: str
    result_count: int
    enriched: List[dict]
    enriched_count: int
    brief: str
    output_path: str
    errors: List[str]
