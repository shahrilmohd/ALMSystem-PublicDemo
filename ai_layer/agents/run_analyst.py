"""
RunAnalystAgent — explains BEL, TVOG, and cash flows for a given ALM run.

Responsibilities
----------------
- Fetch run results and configuration via tools.
- Explain what the numbers mean in actuarial terms.
- Highlight unusual values (high TVOG relative to BEL, unexpected scenario spread, etc.).
- Read-only: this agent never proposes configuration changes or submits runs.

Tools available: get_run_results, get_run_config
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
from ai_layer.tools.get_run_results import (
    TOOL_SCHEMA as GET_RUN_RESULTS_SCHEMA,
    get_run_results,
)

_SYSTEM_PROMPT_TEMPLATE = """\
You are the Run Analyst agent for an actuarial Asset and Liability Model (ALM) system.

Your sole purpose is to explain projection results to the actuary clearly and accurately.
You do not propose configuration changes and you never submit runs.

## What you can do
- Call get_run_results to retrieve BEL, TVOG, scenario counts, and result row previews.
- Call get_run_config to understand the assumptions that drove the run.
- Explain results in plain actuarial language.
- Flag values that appear unusual relative to the configuration.

## What you must NOT do
- Propose changes to assumptions or configuration — that is the Config Advisor's role.
- Submit or re-run projections.
- Make up numbers — always fetch before narrating.

## Actuarial knowledge base

### RunConfig schema
{schema_text}

### Relevant modelling decisions
{decisions_text}

### System architecture (results and config context)
{architecture_text}
"""


class RunAnalystAgent(BaseAgent):
    """Explains BEL, TVOG, and cash flows for a given ALM run."""

    _AGENT_TYPE = "analyst"

    def __init__(self, config: AILayerConfig, api_client: Any = None) -> None:
        super().__init__(config, api_client)
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            schema_text=get_schema_text(),
            decisions_text=get_decisions_text(sections=["financial", "modelling"]),
            architecture_text=get_architecture_text(sections=["results", "run_types"]),
        )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return [GET_RUN_RESULTS_SCHEMA, GET_RUN_CONFIG_SCHEMA]

    def _execute_tool(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        if name == "get_run_results":
            return get_run_results(self._api_client, inputs["run_id"])
        if name == "get_run_config":
            return get_run_config(self._api_client, inputs["run_id"])
        return {"error": f"Unknown tool: {name!r}"}
