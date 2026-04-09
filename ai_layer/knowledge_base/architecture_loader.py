"""
architecture_loader — load relevant sections of ALM_Architecture.md for agent prompts.

Reads document/ALM_Architecture.md from the project root and extracts sections
most relevant to the requesting agent — module map, result field definitions,
run type descriptions.

Usage
-----
    from ai_layer.knowledge_base.architecture_loader import get_architecture_text
    text = get_architecture_text(sections=["results", "run_types"])
"""
from __future__ import annotations

import os
import re
from typing import Optional

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_ARCH_PATH = os.path.join(_PROJECT_ROOT, "document", "ALM_Architecture.md")

_SECTION_KEYWORDS: dict[str, list[str]] = {
    "run_types":    ["LiabilityOnly", "Deterministic", "Stochastic", "run_type",
                     "LIABILITY_ONLY", "DETERMINISTIC", "STOCHASTIC", "run mode"],
    "results":      ["ResultStore", "result_record", "BEL", "TVOG", "net_outgo",
                     "scenario_id", "timestep", "cohort_id", "result field"],
    "config":       ["RunConfig", "InputSources", "AssumptionTables", "FundConfig",
                     "run_config", "config_json", "SubmitRunRequest"],
    "api":          ["router", "endpoint", "GET /runs", "POST /runs", "GET /results",
                     "GET /workers", "/ai/chat", "FastAPI"],
    "ai_layer":     ["ai_layer", "ALMOrchestrator", "RunAnalystAgent", "ConfigAdvisorAgent",
                     "ReviewerAgent", "knowledge_base", "tool"],
    "storage":      ["RunRecord", "ResultRecord", "RunRepository", "ResultRepository",
                     "SQLAlchemy", "database"],
}


def get_architecture_text(
    sections: Optional[list[str]] = None,
    full: bool = False,
) -> str:
    """
    Return relevant sections of ALM_Architecture.md as a text block.

    Parameters
    ----------
    sections : list[str] | None
        Categories to include.  Valid values: "run_types", "results", "config",
        "api", "ai_layer", "storage".
        If None, returns run_types + results (suitable for RunAnalystAgent).
    full : bool
        If True, return the complete document.

    Returns
    -------
    str
        Formatted text ready for injection into a system prompt.
    """
    try:
        with open(_ARCH_PATH, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return "[ALM_Architecture.md not found — architecture context unavailable]"

    if full:
        return f"## ALM_Architecture.md (full)\n\n{content}"

    if sections is None:
        sections = ["run_types", "results"]

    keywords: set[str] = set()
    for cat in sections:
        keywords.update(_SECTION_KEYWORDS.get(cat, []))

    selected = _extract_sections(content, keywords)
    if not selected:
        return "[No matching sections found in ALM_Architecture.md for the requested categories]"

    header = f"## Relevant sections from ALM_Architecture.md (categories: {', '.join(sections)})\n\n"
    return header + "\n\n---\n\n".join(selected)


def _extract_sections(content: str, keywords: set[str]) -> list[str]:
    """
    Split on level-2 headings (##) and return those whose content contains
    at least one keyword (case-insensitive).
    """
    parts = re.split(r"(?=^## )", content, flags=re.MULTILINE)
    selected: list[str] = []
    for part in parts:
        if not part.strip():
            continue
        if any(kw.lower() in part.lower() for kw in keywords):
            selected.append(part.strip())
    return selected
