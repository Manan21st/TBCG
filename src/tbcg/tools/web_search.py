"""Web search tool.

Default backend is DuckDuckGo via ``ddgs`` (keyless, free). If
``SEARCH_BACKEND=tavily`` and a key is present, Tavily is used instead.
All failures degrade to an empty list so the workflow never crashes on search.
"""

from __future__ import annotations

from typing import Any

from langsmith import traceable

from ..config import get_settings


def _search_ddgs(query: str, max_results: int) -> list[dict[str, Any]]:
    try:
        from ddgs import DDGS
    except ImportError:  # pragma: no cover - older package name
        from duckduckgo_search import DDGS  # type: ignore

    results: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "snippet": r.get("body", ""),
                    "url": r.get("href", ""),
                }
            )
    return results


def _search_tavily(query: str, max_results: int) -> list[dict[str, Any]]:
    from tavily import TavilyClient

    s = get_settings()
    client = TavilyClient(api_key=s.tavily_api_key)
    resp = client.search(query=query, max_results=max_results)
    return [
        {"title": r.get("title", ""), "snippet": r.get("content", ""), "url": r.get("url", "")}
        for r in resp.get("results", [])
    ]


@traceable(run_type="tool", name="web_search.query")
def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """Run a web search with the configured backend. Never raises."""
    s = get_settings()
    try:
        if s.search_backend == "tavily" and s.tavily_api_key:
            return _search_tavily(query, max_results)
        return _search_ddgs(query, max_results)
    except Exception as exc:  # noqa: BLE001 - search is best-effort
        return [{"title": "search-error", "snippet": str(exc), "url": ""}]


@traceable(run_type="tool", name="web_search")
def gather(queries: list[str], total: int = 8) -> list[dict[str, Any]]:
    """Run several queries and merge results, de-duplicated by URL/title."""
    if not queries:
        return []
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    per_query = max(2, total // max(1, len(queries)))
    for q in queries:
        for r in web_search(q, max_results=per_query):
            key = r.get("url") or r.get("title", "")
            if key and key not in seen:
                seen.add(key)
                merged.append(r)
    return merged[:total]
