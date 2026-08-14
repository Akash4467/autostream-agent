"""
agent/graph.py

Single Responsibility: graph wiring and public API only.

Graph topology (multi-node):
    START → classify_intent_node → respond_node → END

All implementation details live in their own focused modules:
  agent/state.py       — AgentState schema
  agent/intent.py      — keyword classifier
  agent/prompts.py     — system prompt
  agent/llm_factory.py — LLM construction
  agent/nodes.py       — node functions

Backward-compatible re-exports are provided so that existing callers
(app.py, main.py, tests/) continue to work without any changes:
    from agent.graph import chat, new_state, AgentState, classify_intent
"""

from __future__ import annotations

from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from agent.nodes import classify_intent_node, normalize_content, respond_node
from agent.state import AgentState

# ── Re-exports for backward compatibility ─────────────────────────────────────
from agent.intent import classify_intent  # noqa: F401 — used by tests & __init__

load_dotenv()


# ── Graph ─────────────────────────────────────────────────────────────────────

def _build_graph():
    """
    Wire the two-node graph:
        classify_intent_node  — sets state["intent"] (no LLM cost)
        respond_node          — LLM call, tool execution, follow-up
    """
    builder = StateGraph(AgentState)

    builder.add_node("classify_intent", classify_intent_node)
    builder.add_node("respond", respond_node)

    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "respond")
    builder.add_edge("respond", END)

    return builder.compile()


_graph = _build_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def new_state() -> AgentState:
    """Return a blank AgentState for a new conversation."""
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
    """
    Process one user turn and return (reply_text, updated_state).

    Args:
        user_input: The raw message from the user.
        state:      The current AgentState; a fresh state is created if None.

    Returns:
        Tuple of (str reply, AgentState updated_state).
    """
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
            reply = normalize_content(msg.content)
            break

    if not reply:
        raise RuntimeError("LLM failed to produce a valid response.")

    return reply, updated_state
