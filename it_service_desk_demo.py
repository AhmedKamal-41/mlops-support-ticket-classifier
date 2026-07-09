"""
Streamlit demo for the IT Service Desk Ticket Classification & Routing System.

A single-page app: paste a support ticket, and the ML model predicts the IT
category while the service desk layer shows the priority, routing team, suggested
knowledge-base article, first troubleshooting steps, and escalation guidance.

Run with:
    streamlit run it_service_desk_demo.py

(The heavier operational/monitoring dashboard lives in dashboard/app.py; this
file is the focused, recruiter-friendly demo of the classifier + triage.)
"""

import streamlit as st

from src.model import classify_and_triage

PRIORITY_COLORS = {
    "Low": "#2e7d32",
    "Medium": "#f9a825",
    "High": "#ef6c00",
    "Critical": "#c62828",
}

EXAMPLES = [
    "My Outlook is not syncing and I cannot receive emails.",
    "I can't connect to the VPN from home.",
    "I received a suspicious email asking me to confirm my password.",
    "The entire office has lost internet, nobody can work.",
    "A new hire starts Monday and needs a laptop and accounts set up.",
]

st.set_page_config(page_title="IT Service Desk Triage", page_icon="🛠️", layout="centered")

# Hide Streamlit's default toolbar/footer for a cleaner, product-like view.
st.markdown(
    """
    <style>
      [data-testid="stToolbar"] {visibility: hidden;}
      footer {visibility: hidden;}
      #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛠️ IT Service Desk Ticket Triage")
st.caption(
    "ML-powered (NLP) ticket classification + service desk routing, priority, "
    "and knowledge-base logic."
)

with st.sidebar:
    st.header("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["ticket"] = ex
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. TF-IDF + LogisticRegression predicts the IT category.\n"
        "2. A rules layer maps the category to priority, routing team, KB "
        "article, first steps, and escalation.\n"
        "3. Low model confidence flags the ticket for manual review."
    )

ticket = st.text_area(
    "Support ticket",
    value=st.session_state.get("ticket", ""),
    placeholder="e.g. My Outlook won't sync and I can't receive emails",
    height=110,
)

if st.button("Classify & route", type="primary") and ticket.strip():
    triage = classify_and_triage(ticket.strip())

    conf = triage["confidence"]
    priority = triage["priority"]
    color = PRIORITY_COLORS.get(priority, "#555")

    # Result banner: full category name (no truncation), priority pill, confidence.
    st.markdown(
        f"""
        <div style="border:1px solid #e6e8eb;border-radius:14px;padding:22px 26px;
                    background:#ffffff;box-shadow:0 1px 3px rgba(0,0,0,0.04);
                    display:flex;justify-content:space-between;align-items:center;
                    gap:18px;flex-wrap:wrap;margin-top:8px;">
          <div>
            <div style="font-size:0.72rem;text-transform:uppercase;
                        letter-spacing:0.06em;color:#8a94a6;font-weight:600;">
              ML Predicted Category
            </div>
            <div style="font-size:1.9rem;font-weight:700;color:#1a2233;
                        line-height:1.15;margin-top:2px;">
              {triage['predicted_category']}
            </div>
          </div>
          <div style="text-align:right;">
            <span style="display:inline-block;background:{color};color:#fff;
                         padding:6px 18px;border-radius:999px;font-weight:700;
                         font-size:0.95rem;letter-spacing:0.02em;">
              {priority} priority
            </span>
            <div style="margin-top:10px;color:#5a6472;font-size:0.95rem;">
              Model confidence &nbsp;<b style="color:#1a2233;">{conf:.0%}</b>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(conf, 0.0), 1.0))

    if triage["needs_review"]:
        st.warning(
            "Low model confidence - have a service desk analyst confirm the "
            "category before routing."
        )

    st.subheader("Routing")
    st.write(f"**Team:** {triage['routing_team']}")
    st.write(f"**Suggested KB:** `{triage['suggested_kb']}`")

    st.subheader("First troubleshooting steps")
    for i, step in enumerate(triage["first_steps"], start=1):
        st.write(f"{i}. {step}")

    st.subheader("Escalation")
    st.info(triage["escalation"])
