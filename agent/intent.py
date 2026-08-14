"""
agent/intent.py

Single Responsibility: owns intent classification only.
Open/Closed: add a new intent by adding a new keyword list and a check in
  classify_intent() — no other module needs to change.
"""

from __future__ import annotations

import re

# ── Keyword lists ─────────────────────────────────────────────────────────────

# High-intent: clear purchase/signup signals
_HIGH_INTENT_KW = [
    "buy", "purchase", "subscribe", "sign up", "signup", "sign-up",
    "upgrade", "get started", "start my", "start a plan", "start the",
    "ready to", "i want to join", "i'd like to join", "let's go",
    "take the", "i'm in", "count me in",
]

# Pricing questions
_PRICING_KW = [
    "price", "pricing", "cost", "how much", "fee", "fees",
    "plan", "plans", "tier", "tiers", "affordable", "cheap",
    "expensive", "discount", "offer", "deal",
]

# Support / problem signals
_SUPPORT_KW = [
    "help", "issue", "problem", "trouble", "bug", "broken",
    "not working", "error", "cancel", "cancellation", "refund",
    "complaint", "wrong",
]

# User is in the process of providing their lead info
_LEAD_COLLECTION_KW = [
    "my name is", "i am", "i'm", "call me",
    "my email", "email is", "reach me at", "contact me",
    "@gmail", "@yahoo", "@hotmail", "@outlook", ".com", ".net", ".org",
    "youtube", "instagram", "tiktok", "facebook", "twitter", "linkedin", "twitch",
]

# Greetings (checked last — lowest priority)
_GREETING_KW = [
    "hi", "hello", "hey", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "yo", "sup",
]


#Classifier 

def classify_intent(text: str) -> str:
    """
    Keyword-based intent classifier — no LLM call, runs on every turn.

    Priority order (highest → lowest):
      high_intent → pricing_inquiry → support → lead_collection → greeting → product_inquiry
    """
    lower = text.lower()

    if any(k in lower for k in _HIGH_INTENT_KW):
        return "high_intent"
    if any(k in lower for k in _PRICING_KW):
        return "pricing_inquiry"
    if any(k in lower for k in _SUPPORT_KW):
        return "support"
    if any(k in lower for k in _LEAD_COLLECTION_KW):
        return "lead_collection"
    # Use word-boundary matching to avoid false positives like "yo" inside "you"
    if any(re.search(r"\b" + re.escape(k) + r"\b", lower) for k in _GREETING_KW):
        return "greeting"
    return "product_inquiry"
