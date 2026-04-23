"""
agent/graph.py

LangGraph conversational agent for AutoStream (RAG enabled)
"""

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

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings

from tools.lead_capture import capture_lead, get_all_leads

# ── Load ENV ───────────────────────────────────────────────
load_dotenv()


# ── LLM FACTORY ────────────────────────────────────────────
def _build_llm():
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if anthropic_key:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.3)

    if openai_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    if google_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3
        )

    raise EnvironmentError("❌ No LLM API key found.")


# ── STATE ──────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    intent: Optional[str]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    lead_captured: bool
    lead_id: Optional[str]


# ── INTENT CLASSIFIER ──────────────────────────────────────
_HIGH_INTENT_KW = [
    "buy", "purchase", "subscribe", "start",
    "get started", "sign up", "upgrade"
]

_GREETING_KW = ["hi", "hello", "hey"]


def classify_intent(text: str) -> str:
    text = text.lower()
    if any(k in text for k in _HIGH_INTENT_KW):
        return "high_intent"
    if any(k in text for k in _GREETING_KW):
        return "greeting"
    return "product_inquiry"


# ── RETRIEVER (FIXED + SAFE) ───────────────────────────────
def get_retriever():
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    return db.as_retriever(search_kwargs={"k": 3})


# ── NORMALIZE OUTPUT ───────────────────────────────────────
def normalize_output(content):
    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        if "text" in content:
            return normalize_output(content["text"])
        return " ".join(normalize_output(v) for v in content.values())

    if isinstance(content, list):
        return " ".join(normalize_output(item) for item in content)

    return str(content)


# ── MAIN NODE ──────────────────────────────────────────────
def respond(state: AgentState) -> AgentState:
    llm = _build_llm()
    llm_tools = llm.bind_tools([capture_lead])

    retriever = get_retriever()

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )

    user_input = last_human.content if last_human else ""

    intent = classify_intent(user_input)

    # 🔥 RAG retrieval
    docs = retriever.invoke(user_input)
    context = "\n\n".join([d.page_content for d in docs])

    # 🔥 Dynamic prompt (NO _KB_CONTEXT)
    system_prompt = f"""
You are Alex, AI sales assistant for AutoStream.

Rules:
- Answer ONLY using the provided context
- Be concise and helpful
- Detect intent
- Collect lead info (name → email → platform)
- Call capture_lead ONLY when ready

CONTEXT:
{context}
"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response: AIMessage = llm_tools.invoke(messages)
    response.content = normalize_output(response.content)

    new_messages: list[BaseMessage] = [response]

    # ── Lead state ─────────────────
    lead_name = state.get("lead_name")
    lead_email = state.get("lead_email")
    lead_platform = state.get("lead_platform")
    lead_captured = state.get("lead_captured", False)
    lead_id = state.get("lead_id")

    # ── Tool handling ──────────────
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] == "capture_lead":
                args = tc["args"]

                result = capture_lead.invoke(args)

                lead_name = args.get("name", lead_name)
                lead_email = args.get("email", lead_email)
                lead_platform = args.get("platform", lead_platform)
                lead_captured = True

                leads = get_all_leads()
                if leads:
                    lead_id = leads[-1].get("id")

                new_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"]
                    )
                )

        followup_msgs = (
            [SystemMessage(content=system_prompt)]
            + state["messages"]
            + new_messages
        )

        final = llm.invoke(followup_msgs)
        final.content = normalize_output(final.content)

        new_messages.append(final)

    return {
        "messages": new_messages,
        "intent": intent,
        "lead_name": lead_name,
        "lead_email": lead_email,
        "lead_platform": lead_platform,
        "lead_captured": lead_captured,
        "lead_id": lead_id,
    }


# ── GRAPH ─────────────────────────────────────────────────
def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("respond", respond)
    builder.add_edge(START, "respond")
    builder.add_edge("respond", END)
    return builder.compile()


_graph = _build_graph()


# ── API ───────────────────────────────────────────────────
def new_state() -> AgentState:
    return {
        "messages": [],
        "intent": None,
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "lead_captured": False,
        "lead_id": None,
    }


def chat(user_input: str, state: Optional[AgentState] = None):
    if state is None:
        state = new_state()

    state = dict(state)
    state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

    updated = _graph.invoke(state)

    reply = ""
    for msg in reversed(updated["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            reply = normalize_output(msg.content)
            break

    if not reply:
        raise RuntimeError("No valid response from LLM")

    return reply, updated