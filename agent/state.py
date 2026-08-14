"""
agent/state.py

Single Responsibility: owns the AgentState schema only.
All other modules import from here — nothing else defines the state shape.
"""

from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Shared state flowing through every node in the LangGraph graph."""

    messages: Annotated[list[BaseMessage], add_messages]

    # Intent classification result (set by classify_intent_node)
    intent: Optional[str]

    # Lead fields collected over multiple turns
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]

    # Set to True after capture_lead tool succeeds
    lead_captured: bool
    lead_id: Optional[str]
