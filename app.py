"""app.py - TrendJacks AI: Streamlit UI."""
import os

import streamlit as st

st.set_page_config(page_title="TrendJacks AI", page_icon="🔥", layout="wide")

from pathlib import Path  # noqa: E402

st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text()}</style>",
            unsafe_allow_html=True)


def get_secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets[name]).strip()
    except (KeyError, FileNotFoundError):
        return default


API_KEY = get_secret("GROQ_API_KEY")
INDUSTRIES = ["Fashion", "Tech", "Food & Beverage", "Fitness", "Beauty", "Gaming",
              "Travel", "Education", "Finance", "Other"]

st.markdown(
    """
    <div class="tj-hero">
      <h1>🔥 TrendJacks AI</h1>
      <p>Get roasted by a Gen Z critic, then hijack what's trending right now.</p>
      <div class="tj-pills"><span class="tj-pill">01 · ROAST</span>
      <span class="tj-pill">02 · LIVE TRENDS</span><span class="tj-pill">03 · REEL BLUEPRINT</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not API_KEY:
    st.error("**GROQ_API_KEY is missing.** Add it under Streamlit Cloud → Settings → Secrets, "
             'or in `.streamlit/secrets.toml`: `GROQ_API_KEY = "gsk_..."`')
    st.stop()

os.environ["GROQ_API_KEY"] = API_KEY
from trendjacks_crew import parse_blueprint, parse_score, run_trendjacks  # noqa: E402

# ---------------- Feature 1: Input ----------------
with st.container(border=True):
    c1, c2 = st.columns(2)
    brand = c1.text_input("Brand Name / Handle", placeholder="@sunnyside_coffee")
    industry = c2.selectbox("Industry / Niche", INDUSTRIES)
    copy = st.text_area(
        "Current Campaign Copy / Ad Text / Website Description", height=150,
        placeholder="At Sunnyside, we leverage synergy to deliver a best-in-class coffee experience...",
    )
    go = st.button("🔥 Roast & TrendJack", type="primary", use_container_width=True)

# ---------------- Feature 2: Run agents ----------------
if go:
    if not brand.strip() or not copy.strip():
        st.warning("Please fill in the brand name and your campaign copy.")
    else:
        try:
            with st.spinner("Critic is sharpening claws → Scout is scanning the web → Director is cooking..."):
                st.session_state["result"] = run_trendjacks(
                    brand.strip(), industry, copy.strip(), API_KEY)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "rate" in msg.lower():
                st.error("Groq rate limit hit. Wait ~30 seconds and try again.")
            elif "401" in msg or "invalid api key" in msg.lower():
                st.error("Groq rejected your API key. Check GROQ_API_KEY in your secrets.")
            else:
                st.error(f"Something went wrong: {msg}")

# ---------------- Feature 3: Dashboard ----------------
res = st.session_state.get("result")
if res:
    st.divider()
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("🔪 The Roast")
        score = parse_score(res["roast"])
        if score is not None:
            tone, tag = (("tj-high", "AUTHENTIC ✅") if score >= 7
                         else ("tj-low", "CRINGE ALERT 🚨") if score <= 4 else ("tj-mid", "MID — NEEDS WORK"))
            st.markdown(
                f'<div class="tj-score {tone}" style="--tj-pct:{score * 10}%">'
                f'<div class="label">Gen Z Authenticity Score</div>'
                f'<div class="num">{score}<span>/10</span></div>'
                f'<div class="tag">{tag}</div>'
                f'<div class="tj-bar"><div></div></div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(res["roast"])

    with right:
        st.subheader("📡 Active Trend Radar")
        st.markdown(res["trends"])

    st.divider()
    st.subheader("🎬 The Gen Z Content Blueprint")
    bp = parse_blueprint(res["blueprint"])
    if "HOOK" in bp and "CAPTION" in bp:
        if "TREND USED" in bp:
            st.caption(f"Trend hijacked: **{bp['TREND USED']}**")
        st.markdown("#### 🪝 Hook")
        st.success(bp["HOOK"])
        st.markdown("#### ✍️ Caption")
        st.info(bp["CAPTION"])
        st.markdown("#### 🎞️ 3-Part Storyboard")
        cols = st.columns(3)
        for col, (i, label) in zip(cols, enumerate(["0-3s", "3-10s", "10-15s"], start=1)):
            with col, st.container(border=True):
                st.markdown(f"**Scene {i}** · {label}")
                st.write(bp.get(f"SCENE {i}", "-"))
    else:
        st.markdown(res["blueprint"])  # fallback if the model ignored the format

    st.download_button(
        "Download full report (.md)",
        data=f"# The Roast\n{res['roast']}\n\n# Trend Radar\n{res['trends']}\n\n"
             f"# Content Blueprint\n{res['blueprint']}",
        file_name="trendjacks_report.md", mime="text/markdown",
    )
