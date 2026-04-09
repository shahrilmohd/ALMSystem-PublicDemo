"""
ModellingAgent — explains how calculations work inside the ALM engine.

Responsibilities
----------------
- Answer "how does the model calculate X?" questions using actual engine source.
- Walk through the BEL two-pass calculation, TVOG derivation, liability
  decrement logic, bond accounting branches, and the Fund coordinator pattern.
- Reference specific file paths and line-level logic when helpful.
- Read-only: this agent never proposes configuration changes or submits runs.

Knowledge sources injected at instantiation (not RAG — full injection):
  1. Engine source code    — every engine .py file (the authoritative truth)
  2. model_mechanics.md   — conceptual primer: two-pass BEL, TVOG, coordinator
  3. DECISIONS.md (full)  — the financial/actuarial rationale behind every choice
  4. RunConfig schema      — the config structure the engine reads

Tools available: get_run_config (to ground answers in a specific run)
"""
from __future__ import annotations

import os
from typing import Any

from ai_layer.agents.base_agent import BaseAgent
from ai_layer.config import AILayerConfig
from ai_layer.knowledge_base.code_loader import get_engine_code_text
from ai_layer.knowledge_base.decisions_loader import get_decisions_text
from ai_layer.knowledge_base.schema_export import get_schema_text
from ai_layer.tools.get_run_config import (
    TOOL_SCHEMA as GET_RUN_CONFIG_SCHEMA,
    get_run_config,
)

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_MODEL_MECHANICS_PATH = os.path.join(
    _PROJECT_ROOT, "ai_layer", "knowledge_base", "model_mechanics.md"
)

_SYSTEM_PROMPT_TEMPLATE = """\
You are the Modelling Agent for an actuarial Asset and Liability Model (ALM) system.

Your purpose is to explain exactly how calculations work inside the engine —
not just what the numbers mean, but the mechanics of how they are produced.

## Your role

- Answer "how does the model calculate X?" questions with precision.
- Reference specific Python files, class names, and method names when useful.
- Walk through computation steps sequentially when asked (e.g. "how is BEL computed?").
- Explain design choices using the DECISIONS.md rationale.
- You may call get_run_config to inspect the parameters of a specific run.

## What you must NOT do

- Propose changes to assumptions or submit runs — that is the Advisor's role.
- Explain results in isolation — that is the Analyst's role.
- Make up behaviour that is not in the source code you have been given.

## Knowledge base

### Model mechanics primer (conceptual overview)
{model_mechanics_text}

---

### RunConfig JSON schema (structure of inputs)
{schema_text}

---

### Modelling decisions (financial and actuarial rationale)
{decisions_text}

---

{engine_code_text}
"""


def _load_model_mechanics() -> str:
    try:
        with open(_MODEL_MECHANICS_PATH, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "[model_mechanics.md not found]"


class ModellingAgent(BaseAgent):
    """
    Explains how the ALM engine performs its calculations.

    Instantiation injects the full engine source, model_mechanics.md,
    full DECISIONS.md, and the RunConfig schema into the system prompt.
    The LLM sees the actual code — there are no hand-written summaries
    that can drift out of sync.
    """

    _AGENT_TYPE = "modelling"

    def __init__(self, config: AILayerConfig, api_client: Any = None) -> None:
        super().__init__(config, api_client)
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            model_mechanics_text=_load_model_mechanics(),
            schema_text=get_schema_text(),
            decisions_text=get_decisions_text(sections=["financial", "modelling", "architecture"]),
            engine_code_text=get_engine_code_text(),
        )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [GET_RUN_CONFIG_SCHEMA]

    def _execute_tool(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        if name == "get_run_config":
            return get_run_config(self._api_client, inputs["run_id"])
        return {"error": f"Unknown tool: {name!r}"}
