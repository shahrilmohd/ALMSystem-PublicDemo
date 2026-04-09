"""
ReviewerAgent — cross-checks proposed RunConfig for actuarial consistency.

Responsibilities
----------------
- Receive a proposed RunConfig JSON string (from ConfigAdvisorAgent).
- Check it for internal consistency: projection term, assumption tables, run type, etc.
- Return a structured verdict: approved / needs_revision / rejected.
- Provide a concise commentary explaining any concerns.

This agent is ALWAYS invoked between ConfigAdvisorAgent and the desktop confirmation
dialog.  It is read-only: it never submits runs.

Tools available: get_run_config (to compare against a reference run if needed)
"""
from __future__ import annotations

import json
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

_SYSTEM_PROMPT_TEMPLATE = """\
You are the Reviewer agent for an actuarial Asset and Liability Model (ALM) system.

Your role is to independently cross-check a proposed RunConfig for actuarial consistency
BEFORE it is presented to the actuary for approval.  You are the last line of defence
against obviously inconsistent or erroneous assumptions reaching a production run.

## How to respond

Always return a JSON object with this structure:
{{
  "verdict":     "approved" | "needs_revision" | "rejected",
  "summary":     "<one-sentence summary of your finding>",
  "issues":      ["<issue 1>", "<issue 2>", ...],   // empty list if approved
  "suggestions": ["<suggestion 1>", ...]             // optional improvements
}}

## What to check
- Projection term is positive and consistent with the liability type.
- Mortality, lapse, and expense assumption table names exist and are plausible.
- Discount rate / scenario count is appropriate for the run type (stochastic needs ≥ 100 scenarios).
- No fields have clearly erroneous values (negative rates, zero term, etc.).
- The run type is valid (LIABILITY_ONLY, DETERMINISTIC, or STOCHASTIC).
- Changes requested by the actuary are reflected correctly.

## What you must NOT do
- Approve a config that has obvious errors.
- Reject a config for stylistic reasons or minor preference differences.
- Submit runs or modify configs — you are read-only.

## RunConfig schema
{schema_text}

## Actuarial modelling decisions (for consistency checks)
{decisions_text}

## System context
{architecture_text}
"""

_REVIEW_USER_TEMPLATE = """\
Please review the following proposed RunConfig for actuarial consistency.

Proposed config:
```json
{config_json}
```

{context}
Return only the JSON verdict object described in your instructions.
"""


class ReviewerAgent(BaseAgent):
    """Cross-checks a proposed RunConfig and returns a structured verdict."""

    _AGENT_TYPE = "reviewer"

    def __init__(self, config: AILayerConfig, api_client: Any = None) -> None:
        super().__init__(config, api_client)
        self._system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
            schema_text=get_schema_text(),
            decisions_text=get_decisions_text(sections=["financial", "modelling"]),
            architecture_text=get_architecture_text(sections=["config"]),
        )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        # get_run_config is available so the reviewer can compare against a reference run.
        return [GET_RUN_CONFIG_SCHEMA]

    def _execute_tool(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        if name == "get_run_config":
            return get_run_config(self._api_client, inputs["run_id"])
        return {"error": f"Unknown tool: {name!r}"}

    # ------------------------------------------------------------------
    # Convenience method — wraps chat() for the reviewer pattern
    # ------------------------------------------------------------------

    def review(
        self,
        config_json: str,
        reference_run_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Review a proposed RunConfig JSON string.

        Parameters
        ----------
        config_json : str
            The proposed RunConfig serialised as a JSON string.
        reference_run_id : str | None
            If provided, the reviewer will fetch this run for comparison context.

        Returns
        -------
        dict with keys: verdict, summary, issues, suggestions
        Verdict is one of: "approved", "needs_revision", "rejected".
        """
        context = ""
        if reference_run_id:
            context = (
                f"The proposed config is a modification of run {reference_run_id}. "
                f"You may call get_run_config with that run_id to compare the original.\n\n"
            )

        user_message = _REVIEW_USER_TEMPLATE.format(
            config_json=config_json,
            context=context,
        )
        response = self.chat([{"role": "user", "content": user_message}])

        # Parse the JSON verdict from the reply.
        try:
            # Strip any markdown code fences the model may have added.
            raw = response.reply.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw.strip())
        except (json.JSONDecodeError, IndexError):
            # If parsing fails, return a safe rejection with the raw text.
            return {
                "verdict":     "needs_revision",
                "summary":     "Reviewer response could not be parsed as JSON.",
                "issues":      [response.reply],
                "suggestions": [],
            }
