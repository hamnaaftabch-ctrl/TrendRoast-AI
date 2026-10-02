"""app.py - Streamlit UI for the AI Research Agent."""
import os

import streamlit as st

st.set_page_config(page_title="AI Research Agent", page_icon="🔎", layout="centered")


def get_secret(name: str, default: str = "") -> str:
    """Read from Streamlit secrets; return default if secrets file/key is missing."""
    try:
        return str(st.secrets[name]).strip()
    except (KeyError, FileNotFoundError):
        return default


GROQ_API_KEY = get_secret("GROQ_API_KEY")
MODEL = get_secret("GROQ_MODEL", "groq/openai/gpt-oss-120b")

st.title("🔎 AI Research Agent")
st.caption("CrewAI + Groq + DuckDuckGo: enter a topic, get a sourced report.")

if not GROQ_API_KEY:
    st.error(
        "**GROQ_API_KEY is missing.** Add it in Streamlit Cloud → App settings → "
        "Secrets, or locally in `.streamlit/secrets.toml`:\n\n"
        '`GROQ_API_KEY = "gsk_..."`'
    )
    st.stop()

# Import AFTER the key check so a missing key never crashes on import.
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
from research_agent import EmptyResearchError, run_research  # noqa: E402

topic = st.text_input("Research topic", placeholder="e.g. Latest trends in solid-state batteries")
go = st.button("Run research", type="primary", disabled=not topic.strip())

if go:
    try:
        with st.spinner("Agent is searching the web and writing your report..."):
            report = run_research(topic.strip(), GROQ_API_KEY, MODEL)
        st.session_state["report"] = report
        st.session_state["topic"] = topic.strip()
    except EmptyResearchError:
        st.warning("The agent finished but produced no report. Try rephrasing your topic.")
    except Exception as exc:
        msg = str(exc)
        if "429" in msg or "rate" in msg.lower():
            st.error("Groq rate limit hit. Wait ~30 seconds and try again.")
        elif "401" in msg or "authentication" in msg.lower() or "invalid api key" in msg.lower():
            st.error("Groq rejected your API key. Check `GROQ_API_KEY` in your secrets.")
        else:
            st.error(f"Something went wrong: {msg}")

if "report" in st.session_state:
    st.divider()
    st.markdown(st.session_state["report"])
    st.download_button(
        "Download report (.md)",
        data=st.session_state["report"],
        file_name="research_report.md",
        mime="text/markdown",
    )
