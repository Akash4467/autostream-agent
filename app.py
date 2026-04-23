"""
app.py

Streamlit UI for the AutoStream AI Agent.
"""

from __future__ import annotations
import streamlit as st
import html

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoStream AI Assistant",
    page_icon="🎬",
    layout="wide",
)

# ── Styling ─────────────────────────────────────────────────
st.markdown("""
<style>
body { background-color: #0e1117; color: white; }

/* Chat bubbles */
.chat-bubble {
    padding: 12px 14px;
    border-radius: 12px;
    margin-bottom: 10px;
    max-width: 75%;
    font-size: 14px;
}
.user-bubble {
    background-color: #1f77ff;
    color: white;
    margin-left: auto;
}
.agent-bubble {
    background-color: #2b2f3a;
    color: #e6e6e6;
}

/* Avatar */
.avatar {
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 4px;
}

/* Sidebar */
.sidebar-box {
    background: #1a1d24;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ── Agent imports ───────────────────────────────────────────
from agent.graph import chat, new_state
from tools.lead_capture import get_all_leads

# ── Session state ───────────────────────────────────────────
if "agent_state" not in st.session_state:
    st.session_state.agent_state = new_state()

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "processing" not in st.session_state:
    st.session_state.processing = False


# ── Normalize LLM output (ROBUST) ───────────────────────────
def normalize_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if "text" in content:
            return normalize_text(content["text"])
        return " ".join(normalize_text(v) for v in content.values())

    if isinstance(content, list):
        return " ".join(normalize_text(item) for item in content)

    if content is None:
        return ""

    return str(content)


# ── Escape HTML safely ──────────────────────────────────────
def safe_text(text):
    text = normalize_text(text)
    return html.escape(text).replace("\n", "<br>")


# ── Chat bubble UI ──────────────────────────────────────────
def render_bubble(role, text, intent=None):
    text = safe_text(text)

    if role == "user":
        st.markdown(
            f"""
            <div class="chat-bubble user-bubble">
                <div class="avatar">You</div>
                {text}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        # 🎯 Intent badge box
        intent_html = ""
        if intent:
            if intent == "greeting":
                label, color = "Greeting", "#4caf50"
            elif intent == "product_inquiry":
                label, color = "Product Inquiry", "#2196f3"
            elif intent == "high_intent":
                label, color = "High Intent", "#ff5252"
            else:
                label, color = intent, "#999"

            intent_html = f"""
            <div style="
                display:inline-block;
                margin-bottom:6px;
                padding:4px 8px;
                border-radius:6px;
                font-size:11px;
                font-weight:600;
                background:{color}20;
                color:{color};
                border:1px solid {color}40;
            ">
                INTENT: {label}
            </div>
            """

        st.markdown(
            f"""
            {intent_html}
            <div class="chat-bubble agent-bubble">
                <div class="avatar">AutoStream AI</div>
                {text}
            </div>
            """,
            unsafe_allow_html=True
        )


# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 AutoStream")
    st.markdown("### AI Sales Assistant")

    s = st.session_state.agent_state

    st.markdown("### 📊 Lead Progress")

    def field(label, value):
        st.markdown(
            f'<div class="sidebar-box"><b>{label}</b><br>{value or "—"}</div>',
            unsafe_allow_html=True
        )

    field("Name", s.get("lead_name"))
    field("Email", s.get("lead_email"))
    field("Platform", s.get("lead_platform"))

    if s.get("lead_captured"):
        st.success(f"✅ Lead Captured!\nID: {s.get('lead_id')}")

    st.markdown("---")

    leads = get_all_leads()
    st.markdown(f"### 📥 Captured Leads ({len(leads)})")

    for lead in reversed(leads[-5:]):
        st.markdown(
            f'<div class="sidebar-box">{lead["name"]}<br><small>{lead["email"]}</small></div>',
            unsafe_allow_html=True
        )

    if st.button("🔄 New Conversation"):
        st.session_state.agent_state = new_state()
        st.session_state.conversation = []
        st.session_state.processing = False
        st.rerun()


# ── Header ─────────────────────────────────────────────────
st.markdown("""
# 🎬 AutoStream AI Assistant
### Convert conversations into qualified leads 🚀
""")


# ── Render chat history ────────────────────────────────────
for msg in st.session_state.conversation:
    render_bubble(
        role=msg["role"],
        text=msg["text"],
        intent=msg.get("intent"),
    )


# ── Input ─────────────────────────────────────────────────
user_input = st.chat_input("Ask about pricing, features, or plans...")

if user_input and not st.session_state.processing:
    st.session_state.conversation.append({
        "role": "user",
        "text": user_input.strip()
    })
    st.session_state.processing = True
    st.rerun()


# ── Processing ─────────────────────────────────────────────
if st.session_state.processing:
    last_user = next(
        (m["text"] for m in reversed(st.session_state.conversation)
         if m["role"] == "user"),
        None,
    )

    if last_user:
        try:
            with st.spinner("AI is thinking..."):
                reply, updated_state = chat(
                    last_user,
                    st.session_state.agent_state
                )

            reply = normalize_text(reply)

            st.session_state.agent_state = updated_state
            st.session_state.conversation.append({
                "role": "agent",
                "text": reply,
                "intent": updated_state.get("intent"),
            })

        except Exception as e:
            st.session_state.conversation.append({
                "role": "agent",
                "text": f"⚠️ Error: {str(e)}",
                "intent": None,
            })

    st.session_state.processing = False
    st.rerun()