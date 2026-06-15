"""
agent/rag.py

Loads the AutoStream knowledge base from a local JSON file and converts
it into a compact, LLM-readable context string injected into the system prompt.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"


def load_knowledge_base() -> dict:
    """Load and return the knowledge base dict from disk."""
    with open(_KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_kb_context(kb: Optional[dict] = None) -> str:
    """
    Serialise the KB dict into a flat, LLM-readable context block.
    Called once at agent startup; the result is embedded in the system prompt.
    """
    if kb is None:
        kb = load_knowledge_base()

    lines: list[str] = [
        f"=== {kb['company']} — {kb['tagline']} ===",
        "",
        "## Pricing Plans",
    ]

    for plan in kb["plans"].values():
        lines.append(f"\n### {plan['name']} — ${plan['price_monthly']}/month")
        for feat in plan["features"]:
            lines.append(f"  - {feat}")

    lines += ["", "## Policies"]
    for policy_name, policy_text in kb["policies"].items():
        label = policy_name.replace("_", " ").title()
        lines.append(f"  - {label}: {policy_text}")

    lines += [
        "",
        "## Supported Creator Platforms",
        "  " + ", ".join(kb["supported_platforms"]),
        "",
        "## FAQ",
    ]
    for item in kb["faq"]:
        lines.append(f"  Q: {item['question']}")
        lines.append(f"  A: {item['answer']}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    print(build_kb_context())