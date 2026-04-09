"""
ConfigAdvisorAgent — proposes revised RunConfig assumptions.

Responsibilities
----------------
- Fetch the existing config for a run via get_run_config.
- Understand what the actuary wants to change (mortality, lapses, expenses, term, etc.).
- Produce a modified RunConfig JSON string with the proposed changes.
- Always hand off to ReviewerAgent before any config reaches submit_run.

Tools available: get_run_config
  (submit_run is listed in the schema so the model knows it exists, but BaseAgent
   intercepts it and stores it in AgentResponse.pending_submit — it is NEVER
   executed automatically.)

Safety note
-----------
This agent may produce a submit_run tool call.  BaseAgent will intercept it and
return it as AgentResponse.pending_submit.  The orchestrator must then pass it
to ReviewerAgent for cross-checking before the desktop UI shows the confirmation
dialog.  See DECISIONS.md §30.
"""
from __future__ import annotations

from typing import Any

from ai_layer.agents.base_agent import BaseAgent
from ai_layer.config import AILayerConfig
from ai_layer.knowledge_base.architecture_loader import get_architecture_text
from ai_layer.knowledge_base.decisions_loader import get_decisions_text
from ai_layer.knowledge_base.schema_export import get_schema_text
from ai_layer.tools.get_run_config import (
    TOOL_SCHEMA as GET_RUN_CONFIG_SCHEMA,
    get_run_config,
)
from ai_layer.tools.submit_run import TOOL_SCHEMA as SUBMIT_RUN_SCHEMA

_SYSTEM_PROMPT_TEMPLATE = """\
You are the Config Advisor agent for an actuarial Asset and Liability Model (ALM) system.

Your role is to propose revised projection assumptions in response to actuary instructions.
You produce a modified RunConfig JSON and hand it off for review — you never submit directly.

## Workflow
1. Call get_run_config to read the current assumptions for the run being modified.
2. Apply the actuary's requested changes carefully.
3. Produce the complete updated RunConfig as a JSON string.
4. Call submit_run with the revised config_json — this will be intercepted for human approval
   and will NOT be executed without the actuary's explicit confirmation in the UI.

## What you must NOT do
- Invent or fabricate assumption values not grounded in the current config or explicit instructions.
- Change fields the actuary did not ask to change.
- Submit a run without calling submit_run (the interception mechanism requires the tool call).

## RunConfig schema — use this to understand every field before modifying
{schema_text}

## Actuarial modelling decisions — use these to justify proposed changes
{decisions_text}

## System context
{architecture_text}
"""


class ConfigAdvisorAgent(BaseAgent):
    """Proposes revised RunConfig assumptions based on actuary instructions."""

    _AGENT_TYPE = "advisor"

    def __init__(self, config: AILayerConfig, api_client: Any = None) -> None:
        super().__init__(config, api_client)
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            schema_text=get_schema_text(),
            decisions_text=get_decisions_text(sections=["financial", "modelling", "architecture"]),
            architecture_text=get_architecture_text(sections=["config", "run_types"]),
        )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        # submit_run is included so the model can call it; BaseAgent intercepts it.
        return [GET_RUN_CONFIG_SCHEMA, SUBMIT_RUN_SCHEMA]

    def _execute_tool(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        # submit_run is intercepted by BaseAgent before reaching here.
        if name == "get_run_config":
            return get_run_config(self._api_client, inputs["run_id"])
        return {"error": f"Unknown tool: {name!r}"}
