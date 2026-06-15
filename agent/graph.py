from __future__ import annotations

import os
from typing import Annotated, Optional

from dotenv import load_dotenv
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.rag import build_kb_context
from tools.lead_capture import capture_lead, get_all_leads

load_dotenv()


# ── ✅ NEW: normalize helper ───────────────────────────────────────────────────
def normalize_content(content):
    """Normalize LLM output (Gemini/OpenAI/Anthropic) into a clean string."""

    # Gemini dict response
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return str(content)

    # List response
    if isinstance(content, list):
        return " ".join(normalize_content(c) for c in content)

    # None
    if content is None:
        return ""

    return str(content)


# ── LLM factory ───────────────────────────────────────────────────────────────
def _build_llm():
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-4-5", temperature=0.3)

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        # ✅ FIX: use valid model from your list
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

    raise EnvironmentError("❌ No LLM API key found.")


# ── Agent state ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Optional[str]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    lead_id: Optional[str]


# ── System prompt ─────────────────────────────────────────────────────────────
_KB_CONTEXT = build_kb_context()

_SYSTEM_PROMPT = f"""You are Alex, the AI sales assistant for AutoStream.

Use only the knowledge base.

{_KB_CONTEXT}
"""


# ── Intent classifier ─────────────────────────────────────────────────────────
_HIGH_INTENT_KW = ["buy", "subscribe", "start", "upgrade"]
_GREETING_KW = ["hi", "hello", "hey"]


def classify_intent(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in _HIGH_INTENT_KW):
        return "high_intent"
    if any(k in lower for k in _GREETING_KW):
        return "greeting"
    return "product_inquiry"


# ── Node ──────────────────────────────────────────────────────────────────────
def respond(state: AgentState) -> AgentState:
    llm = _build_llm()
    llm_with_tools = llm.bind_tools([capture_lead])

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    intent = classify_intent(last_human.content) if last_human else "greeting"

    full_messages = [SystemMessage(content=_SYSTEM_PROMPT)] + list(state["messages"])

    response: AIMessage = llm_with_tools.invoke(full_messages)

    new_messages: list[BaseMessage] = [response]

    lead_name = state.get("lead_name")
    lead_email = state.get("lead_email")
    lead_platform = state.get("lead_platform")
    lead_captured = state.get("lead_captured", False)
    lead_id = state.get("lead_id")

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == "capture_lead":
                args = tc["args"]
                tool_result = capture_lead.invoke(args)

                lead_name = args.get("name", lead_name)
                lead_email = args.get("email", lead_email)
                lead_platform = args.get("platform", lead_platform)
                lead_captured = True

                leads = get_all_leads()
                if leads:
                    lead_id = leads[-1].get("id")

                new_messages.append(
                    ToolMessage(content=str(tool_result), tool_call_id=tc["id"])
                )

        follow_up = (
            [SystemMessage(content=_SYSTEM_PROMPT)]
            + list(state["messages"])
            + new_messages
        )

        final_response = llm.invoke(follow_up)
        new_messages.append(final_response)

    return {
        "messages": new_messages,
        "intent": intent,
        "lead_name": lead_name,
        "lead_email": lead_email,
        "lead_platform": lead_platform,
        "lead_captured": lead_captured,
        "lead_id": lead_id,
    }


# ── Graph ─────────────────────────────────────────────────────────────────────
def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────
def new_state() -> AgentState:
    return AgentState(
        messages=[],
        intent=None,
        lead_name=None,
        lead_email=None,
        lead_platform=None,
        lead_captured=False,
        lead_id=None,
    )


def chat(user_input: str, state: Optional[AgentState] = None):
    if state is None:
        state = new_state()

    state = dict(state)
    state["messages"] = list(state["messages"]) + [
        HumanMessage(content=user_input)
    ]

    updated_state: AgentState = _graph.invoke(state)

    reply = ""

    for msg in reversed(updated_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            # ✅ FIX: normalize content
            reply = normalize_content(msg.content)
            break

    if not reply:
        raise RuntimeError("LLM failed to produce a valid response.")

    return reply, updated_state