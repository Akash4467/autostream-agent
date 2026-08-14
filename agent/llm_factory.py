"""
agent/llm_factory.py

Single Responsibility: owns LLM construction only.
Dependency Inversion: the rest of the codebase depends on the abstract
  BaseChatModel interface returned here, not on any concrete provider class.
Open/Closed: add a new provider by adding an env-var check here — no other
  module needs to change.

The singletons are created lazily on first access so tests that mock
environment variables before import still get the right client.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool


def _build_llm() -> BaseChatModel:
    """
    Instantiate the appropriate LLM based on whichever API key is set.
    Priority: Anthropic → OpenAI → Google.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-haiku-4-5", temperature=0.3)

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    raise EnvironmentError("❌ No LLM API key found.")


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return the cached LLM singleton."""
    return _build_llm()


def get_llm_with_tools(tools: list["BaseTool"]) -> BaseChatModel:
    """Return the LLM bound to the given tools (cached per tool-list identity)."""
    return get_llm().bind_tools(tools)
