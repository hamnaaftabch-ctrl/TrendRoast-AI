"""trendjacks_crew.py - the 3-agent TrendJacks crew (Critic -> Scout -> Copywriter)."""
import os
import re

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent, Crew, Process, Task  # noqa: E402

from llm_utils import build_llm  # noqa: E402
from tools import gather_trend_data  # noqa: E402


def build_crew(api_key: str) -> Crew:
    llm = build_llm(api_key)

    # ---------- Agent 1: Brutal Critic ----------
    critic = Agent(
        role="Brutal Gen Z Brand Auditor",
        goal="Expose corporate jargon, fake authenticity and cringe in marketing copy.",
        backstory=(
            "You are a 22-year-old chronically-online creative director. You hate "
            "buzzwords, fake relatability and 'how do you do, fellow kids' energy, "
            "but you are FAIR: you reward honest, plain-spoken writing and only "
            "go savage when the copy earns it. Your criticism is always specific."
        ),
        llm=llm, max_iter=2, allow_delegation=False, verbose=False,
    )

    # ---------- Agent 2: Trend Scout ----------
    scout = Agent(
        role="Trend Scout (Live Radar)",
        goal="Find what is trending RIGHT NOW on TikTok/Instagram Reels in a given niche.",
        backstory=(
            "You live on the For You Page. You track viral formats, sounds, memes "
            "and slang, and only report trends backed by what you find on the web."
        ),
        llm=llm, max_iter=2, allow_delegation=False, verbose=False,
    )

    # ---------- Agent 3: Viral Copywriter & Director ----------
    director = Agent(
        role="Viral Copywriter & Reel Director",
        goal="Turn a roast and live trends into a ready-to-shoot short-form video package.",
        backstory=(
            "You have written dozens of 1M+ view Reels. You write hooks that stop "
            "thumbs in 2 seconds and captions that sound like a real human, not a brand."
        ),
        llm=llm, max_iter=2, allow_delegation=False, verbose=False,
    )

    roast_task = Task(
        description=(
            "Audit this copy for the brand '{brand}' (industry: {industry}).\n\n"
            "COPY:\n\"\"\"{copy}\"\"\"\n\n"
            "SCORING RUBRIC (decide the score FIRST, from the copy itself):\n"
            "- 9-10: sounds like a real person; specific, honest, zero buzzwords.\n"
            "- 7-8: honest and straightforward, no corporate buzzwords. Plain or a "
            "bit safe is FINE - that is not cringe.\n"
            "- 5-6: mixed; some generic filler or mild jargon, but mostly clear.\n"
            "- 3-4: noticeable jargon, forced slang, or mild fake humility.\n"
            "- 1-2: ONLY for heavy jargon, fake humblebrags, or forced corporate slang.\n"
            "Do NOT give 1-3 just because copy is simple, short, or not trendy. "
            "Honest, direct copy with no buzzwords MUST score 7 or higher.\n\n"
            "Then write the feedback to match the score: funny and specific, savage "
            "only when the score is low, and lighter nitpicks plus real praise when "
            "the score is 7+."
        ),
        expected_output=(
            "Exactly this format:\n"
            "SCORE: <integer 1-10>/10\n"
            "VERDICT: <one witty sentence that matches the score>\n"
            "ROAST:\n- <3 to 5 bullets, each quoting or referencing a specific phrase; "
            "for high scores, include what works and one or two small nitpicks>"
        ),
        agent=critic,
    )

    trend_task = Task(
        description=(
            "Identify 3-4 CURRENT viral trends, formats, sounds or slang on TikTok and "
            "Instagram Reels relevant to the '{industry}' niche, using ONLY the live "
            "DuckDuckGo results below. Cite the source URL for each. If the data says "
            "NO_LIVE_DATA, state that clearly, then give 3 well-known evergreen formats "
            "labelled '(evergreen, not live)'.\n\n"
            "LIVE SEARCH RESULTS:\n{search_results}"
        ),
        expected_output=(
            "3-4 bullets in the format:\n"
            "- **<Trend name>**: <what it is and how it works, 1-2 sentences> "
            "(Source: <url>)"
        ),
        agent=scout,
    )

    blueprint_task = Task(
        description=(
            "Brand: {brand} | Industry: {industry}\n"
            "Using the critic's roast and the scout's trends (in context), write ONE "
            "Reel/TikTok package that fixes the cringe and hijacks the best-fitting "
            "trend. Keep it authentic, a bit unhinged, and true to the brand's product."
        ),
        expected_output=(
            "Exactly this format, nothing else:\n"
            "TREND USED: <trend name>\n"
            "HOOK: <thumb-stopping first line, max 12 words>\n"
            "CAPTION: <authentic, unhinged caption with 2-3 hashtags>\n"
            "SCENE 1 (0-3s): <visual + on-screen text>\n"
            "SCENE 2 (3-10s): <visual + action>\n"
            "SCENE 3 (10-15s): <payoff + call to action>"
        ),
        agent=director,
        context=[roast_task, trend_task],
    )

    return Crew(
        agents=[critic, scout, director],
        tasks=[roast_task, trend_task, blueprint_task],
        process=Process.sequential,
        verbose=False,
    )


def run_trendjacks(brand: str, industry: str, copy: str, api_key: str) -> dict:
    """Run the crew and return the three raw section outputs."""
    crew = build_crew(api_key)
    search_results = gather_trend_data(industry)  # live web data, fetched in Python
    result = crew.kickoff(inputs={"brand": brand, "industry": industry,
                                  "copy": copy, "search_results": search_results})
    outs = [(t.raw or "").strip() for t in result.tasks_output]
    if len(outs) < 3 or not outs[2]:
        raise RuntimeError("The agents returned incomplete results. Please try again.")
    return {"roast": outs[0], "trends": outs[1], "blueprint": outs[2]}


# ---------- Output parsing helpers (used by the UI) ----------
def parse_score(roast: str):
    m = re.search(r"SCORE:\s*(\d{1,2})\s*/\s*10", roast, re.I)
    return max(0, min(10, int(m.group(1)))) if m else None


def parse_blueprint(text: str) -> dict:
    keys = ["TREND USED", "HOOK", "CAPTION", "SCENE 1", "SCENE 2", "SCENE 3"]
    out = {}
    for k in keys:
        m = re.search(rf"{k}[^:\n]*:\s*(.+?)(?=\n[A-Z][A-Z ]+(?:\d)?[^:\n]*:|\Z)", text, re.S)
        if m:
            out[k] = m.group(1).strip()
    return out
