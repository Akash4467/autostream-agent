"""
agent/nodes.py

Single Responsibility: owns the two LangGraph node functions only.

Multi-node graph topology:
    START → classify_intent_node → respond_node → END

  classify_intent_node  — pure keyword classification, zero LLM cost.
                          Writes state["intent"]; no messages produced.
  respond_node          — LLM invocation, tool execution, follow-up call.
                          Writes state["messages"] and lead fields.
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent.intent import classify_intent
from agent.llm_factory import get_llm, get_llm_with_tools
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from tools.lead_capture import capture_lead, get_all_leads


# ── Node 1: intent classification ─────────────────────────────────────────────

def classify_intent_node(state: AgentState) -> AgentState:
    """
    Pure keyword classification node — no LLM call.
    Reads the last human message and writes the intent back into state.
    """
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    intent = classify_intent(last_human.content) if last_human else "greeting"
    return {"intent": intent}


# ── Normalize helper ───────────────────────────────────────────────────────────

def normalize_content(content) -> str:
    """Normalize LLM output (Gemini/OpenAI/Anthropic) into a clean string."""
    if isinstance(content, dict):
        return str(content.get("text", content))
    if isinstance(content, list):
        return " ".join(normalize_content(c) for c in content)
    if content is None:
        return ""
    return str(content)


# ── Node 2: LLM response + tool execution ─────────────────────────────────────

def respond_node(state: AgentState) -> AgentState:
    """
    LLM invocation node.
    Reads state["intent"] (set by classify_intent_node) and the message history,
    calls the LLM, executes any tool calls, and returns updated state.
    """
    llm_with_tools = get_llm_with_tools([capture_lead])

    full_messages: list[BaseMessage] = (
        [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    )

    response: AIMessage = llm_with_tools.invoke(full_messages)
    new_messages: list[BaseMessage] = [response]

    # Carry forward whatever lead fields were collected in previous turns
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

        # Follow-up call so the LLM can render a natural response after tool use
        follow_up: list[BaseMessage] = (
            [SystemMessage(content=SYSTEM_PROMPT)]
            + list(state["messages"])
            + new_messages
        )
        final_response = get_llm().invoke(follow_up)
        new_messages.append(final_response)

    return {
        "messages": new_messages,
        "lead_name": lead_name,
        "lead_email": lead_email,
        "lead_platform": lead_platform,
        "lead_captured": lead_captured,
        "lead_id": lead_id,
    }
