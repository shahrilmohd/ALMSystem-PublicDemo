"""
decisions_loader — load relevant sections of DECISIONS.md for agent system prompts.

Reads document/DECISIONS.md from the project root and extracts the sections
most relevant to the requesting agent.  This ensures agents are grounded in
the project's actual financial and actuarial modelling decisions.

Usage
-----
    from ai_layer.knowledge_base.decisions_loader import get_decisions_text
    text = get_decisions_text(sections=["financial", "modelling"])
"""
from __future__ import annotations

import os
import re
from typing import Optional

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_DECISIONS_PATH = os.path.join(_PROJECT_ROOT, "document", "DECISIONS.md")

# Section keyword map — maps logical category names to heading keywords that
# identify matching sections in DECISIONS.md.
_SECTION_KEYWORDS: dict[str, list[str]] = {
    "financial":    ["BEL", "TVOG", "discount", "liability", "cash flow",
                     "bonus", "risk-free", "best estimate"],
    "modelling":    ["assumption", "decrement", "mortality", "lapse", "surrender",
                     "model point", "projection", "seriatim", "group"],
    "architecture": ["storage", "repository", "schema", "batch", "worker",
                     "API", "database", "result_record"],
    "ai":           ["AI Layer", "agent", "knowledge base", "RAG", "injection",
                     "privacy", "confidentiality", "PNC", "deployment"],
    "ifrs17":       ["IFRS 17", "CSM", "RA", "LRC", "LIC", "GMM"],
    "solvency2":    ["Solvency II", "SCR", "spread stress", "matching adjustment",
                     "fundamental spread"],
    "bpa":          ["BPA", "bulk purchase annuity", "matching adjustment",
                     "in-payment", "deferred"],
}

# Sections that are always included regardless of requested categories.
_ALWAYS_INCLUDE_HEADINGS = [
    "Introduction",
    "§1",
]


def get_decisions_text(
    sections: Optional[list[str]] = None,
    full: bool = False,
) -> str:
    """
    Return relevant sections of DECISIONS.md as a text block.

    Parameters
    ----------
    sections : list[str] | None
        Categories to include.  Valid values: "financial", "modelling",
        "architecture", "ai", "ifrs17", "solvency2", "bpa".
        If None, returns financial + modelling (suitable for RunAnalystAgent).
    full : bool
        If True, return the complete DECISIONS.md regardless of sections.
        Use for ReviewerAgent which needs full context.

    Returns
    -------
    str
        Formatted text ready for injection into a system prompt.
    """
    try:
        with open(_DECISIONS_PATH, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return "[DECISIONS.md not found — financial rationale unavailable]"

    if full:
        return f"## DECISIONS.md (full)\n\n{content}"

    if sections is None:
        sections = ["financial", "modelling"]

    # Build set of keywords to match against.
    keywords: set[str] = set()
    for cat in sections:
        keywords.update(_SECTION_KEYWORDS.get(cat, []))

    selected = _extract_sections(content, keywords)
    if not selected:
        return "[No matching sections found in DECISIONS.md for the requested categories]"

    header = f"## Relevant sections from DECISIONS.md (categories: {', '.join(sections)})\n\n"
    return header + "\n\n---\n\n".join(selected)


def _extract_sections(content: str, keywords: set[str]) -> list[str]:
    """
    Split DECISIONS.md by ## headings and return those whose heading or body
    contains at least one of the given keywords (case-insensitive).
    """
    # Split on level-2 headings (##) while keeping the heading line.
    parts = re.split(r"(?=^## )", content, flags=re.MULTILINE)

    selected: list[str] = []
    for part in parts:
        if not part.strip():
            continue
        # Always include sections matching _ALWAYS_INCLUDE_HEADINGS.
        first_line = part.splitlines()[0] if part.splitlines() else ""
        always = any(h in first_line for h in _ALWAYS_INCLUDE_HEADINGS)
        if always or any(kw.lower() in part.lower() for kw in keywords):
            selected.append(part.strip())

    return selected
