"""tools.py - DuckDuckGo search helpers."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from ddgs import DDGS
from crewai.tools import tool

MAX_RESULTS = 5          # keep small: fewer tokens = fewer Groq 429 errors
MAX_SNIPPET_CHARS = 350


def search_web(query: str) -> str:
    """Plain DuckDuckGo search. Returns formatted text, or an ERROR/NO_RESULTS marker."""
    try:
        results = DDGS().text(query, max_results=MAX_RESULTS)
    except Exception as exc:
        return f"SEARCH_ERROR: {exc}"
    if not results:
        return "NO_RESULTS"
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[{i}] {r.get('title', 'Untitled')}\n"
            f"URL: {r.get('href', '')}\n"
            f"Snippet: {(r.get('body') or '')[:MAX_SNIPPET_CHARS]}"
        )
    return "\n\n".join(lines)


@tool("DuckDuckGo Web Search")
def web_search(query: str) -> str:
    """Search the web with DuckDuckGo. Input: a short search query string."""
    return search_web(query)


def gather_trend_data(industry: str) -> str:
    """Run several trend searches in parallel (deterministic, no LLM tool-calling)."""
    year = date.today().year
    queries = [
        f"viral TikTok trends {industry} this week",
        f"trending Instagram Reels formats {industry} {year}",
        f"Gen Z slang meme trends {industry} {year}",
    ]
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(search_web, queries))

    blocks = []
    for q, out in zip(queries, outputs):
        if out.startswith(("SEARCH_ERROR", "NO_RESULTS")):
            continue
        blocks.append(f"### Search: {q}\n{out}")
    if not blocks:
        return "NO_LIVE_DATA: all web searches failed or returned nothing."
    return "\n\n".join(blocks)
