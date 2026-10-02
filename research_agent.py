"""research_agent.py - CrewAI agent, task and crew."""
import os

# Disable telemetry BEFORE importing crewai (quieter logs, no outbound calls).
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent, Crew, LLM, Process, Task  # noqa: E402

from tools import web_search  # noqa: E402

DEFAULT_MODEL = "groq/openai/gpt-oss-120b"


class EmptyResearchError(Exception):
    """Raised when the crew returns no usable report."""


def build_llm(api_key: str, model: str = DEFAULT_MODEL) -> LLM:
    return LLM(
        model=model,
        api_key=api_key,
        temperature=0.2,
        max_tokens=4096,
    )


def build_crew(api_key: str, model: str = DEFAULT_MODEL) -> Crew:
    researcher = Agent(
        role="Senior Research Analyst",
        goal="Research {topic} on the web and write an accurate, well-sourced report.",
        backstory=(
            "You are a meticulous analyst. You search the web, cross-check "
            "sources, never invent facts or URLs, and cite everything."
        ),
        tools=[web_search],
        llm=build_llm(api_key, model),
        max_iter=4,             # hard cap on reasoning/tool loops (protects rate limits)
        max_rpm=20,             # throttle LLM requests per minute
        allow_delegation=False,
        verbose=False,
    )

    task = Task(
        description=(
            "Research the topic: '{topic}'.\n"
            "1. Run 2-3 DuckDuckGo searches with different, specific queries.\n"
            "2. Use ONLY information found in the search results.\n"
            "3. If searches return NO_RESULTS or SEARCH_ERROR repeatedly, stop and "
            "say plainly that no reliable sources were found. Do not make things up."
        ),
        expected_output=(
            "A Markdown report with these sections:\n"
            "# <Title>\n"
            "## Summary (3-4 sentences)\n"
            "## Key Findings (5-7 bullets, each ending with a citation like [1])\n"
            "## Details (2-3 short paragraphs)\n"
            "## Limitations (what is uncertain or missing)\n"
            "## References (numbered list of title - URL, only URLs seen in search results)"
        ),
        agent=researcher,
    )

    return Crew(
        agents=[researcher],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )


def run_research(topic: str, api_key: str, model: str = DEFAULT_MODEL) -> str:
    """Run the crew and return the Markdown report."""
    crew = build_crew(api_key, model)
    result = crew.kickoff(inputs={"topic": topic})
    report = (getattr(result, "raw", None) or str(result) or "").strip()
    if not report:
        raise EmptyResearchError("The agent returned an empty report.")
    return report
