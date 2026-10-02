"""tools.py - DuckDuckGo search tool for the research agent."""
from ddgs import DDGS
from crewai.tools import tool

MAX_RESULTS = 6          # keep small: fewer tokens = fewer Groq 429 errors
MAX_SNIPPET_CHARS = 400


@tool("DuckDuckGo Web Search")
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo. Input: a short, specific search query.
    Returns numbered results with title, URL and a text snippet."""
    try:
        results = DDGS().text(query, max_results=MAX_RESULTS)
    except Exception as exc:  # network / rate-limit errors from DDG
        return f"SEARCH_ERROR: {exc}. Try a different or shorter query."

    if not results:
        return "NO_RESULTS: nothing found. Try rephrasing the query."

    lines = []
    for i, r in enumerate(results, start=1):
        title = r.get("title", "Untitled")
        url = r.get("href", "")
        body = (r.get("body") or "")[:MAX_SNIPPET_CHARS]
        lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {body}")
    return "\n\n".join(lines)
