"""
code_loader — inject selected engine source files into agent system prompts.

Why source code rather than hand-written summaries
---------------------------------------------------
The engine source is the single source of truth for how calculations work.
Hand-written summaries drift as the code evolves; source code never lies.
Each file is small enough (< 300 lines) that the full context fits easily in
a 200K-token window even with all files combined.

Usage
-----
    from ai_layer.knowledge_base.code_loader import get_engine_code_text
    text = get_engine_code_text()           # all default files
    text = get_engine_code_text(["engine/core/fund.py"])  # specific subset
"""
from __future__ import annotations

import os
from typing import Optional

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# ---------------------------------------------------------------------------
# Default file list — the modules that together cover the full calculation path
# from RunConfig through results.
# ---------------------------------------------------------------------------

_DEFAULT_ENGINE_FILES: list[str] = [
    # Orchestration — the time loop and run mode implementations
    "engine/core/fund.py",
    "engine/run_modes/deterministic_run.py",
    "engine/run_modes/stochastic_run.py",
    "engine/run_modes/liability_only_run.py",
    # Liability — cashflow and decrement logic (core calculation files)
    "engine/liability/conventional.py",
    "engine/liability/multi_decrement.py",
    # Results — accumulation and TVOG
    "engine/results/result_store.py",
    "engine/results/tvog_calculator.py",
    # Scenarios
    "engine/scenarios/scenario_engine.py",
    # Assets — bond accounting (AC / FVTPL / FVOCI branches)
    "engine/asset/bond.py",
    "engine/strategy/investment_strategy.py",
    # Rates
    "engine/curves/rate_curve.py",
]
# Files intentionally excluded from the default list:
#   engine/config/run_config.py  — covered by schema_export.py (no duplication)
#   engine/run_modes/base_run.py — abstract base class; no calculation logic
#   engine/liability/base_liability.py — abstract interface only
#   engine/asset/base_asset.py   — abstract interface only


def get_engine_code_text(
    files: Optional[list[str]] = None,
    max_lines_per_file: int = 500,
) -> str:
    """
    Return selected engine source files formatted for system prompt injection.

    Parameters
    ----------
    files : list[str] | None
        Relative paths from project root.  Defaults to _DEFAULT_ENGINE_FILES.
    max_lines_per_file : int
        Truncate files longer than this to prevent context overflow.
        Most engine files are well under 300 lines; 500 is a safety ceiling.

    Returns
    -------
    str
        Formatted block ready to embed in a system prompt.
    """
    if files is None:
        files = _DEFAULT_ENGINE_FILES

    sections: list[str] = ["## Engine source code (read-only reference)\n"]

    for rel_path in files:
        abs_path = os.path.join(_PROJECT_ROOT, rel_path.replace("/", os.sep))
        try:
            with open(abs_path, encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            sections.append(f"### {rel_path}\n[File not found]\n")
            continue

        if len(lines) > max_lines_per_file:
            body = "".join(lines[:max_lines_per_file])
            body += f"\n... [{len(lines) - max_lines_per_file} lines truncated]\n"
        else:
            body = "".join(lines)

        sections.append(f"### {rel_path}\n```python\n{body}```\n")

    return "\n".join(sections)


def list_engine_files() -> list[str]:
    """Return the default list of engine files (useful for inspection / tests)."""
    return list(_DEFAULT_ENGINE_FILES)
