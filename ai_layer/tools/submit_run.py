"""
submit_run — submit a new ALM projection run via the AI agent.

Calls POST /runs with a RunConfig JSON string.  This is the ONLY mutating tool
in the Phase 2 AI layer.

IMPORTANT — Human approval required
-------------------------------------
The desktop UI must intercept this tool call and present a confirmation dialog
showing the proposed RunConfig and the ReviewerAgent's commentary BEFORE this
function is called.  This function must never be called automatically without
explicit actuary approval.  See DECISIONS.md §30 (Safety Constraints).

Tool schema (registered in BaseAgent)
--------------------------------------
name:        "submit_run"
description: "Submit a new ALM projection run with the given RunConfig JSON.
              Only call this after the actuary has reviewed and approved the config."
parameters:
  config_json (string, required): Full RunConfig serialised as a JSON string.
"""
from __future__ import annotations

from typing import Any

from frontend.desktop.api_client import ALMApiClient, APIError, APIResponseError


def submit_run(client: ALMApiClient, config_json: str) -> dict[str, Any]:
    """
    Submit a new projection run.

    Parameters
    ----------
    client : ALMApiClient
        Configured HTTP client pointing at the ALM API.
    config_json : str
        Full RunConfig serialised as a JSON string.

    Returns
    -------
    dict with keys:
        run_id   : str   — UUID of the newly created run (if successful)
        status   : str   — initial status (always "PENDING" on success)
        run_name : str | None
        error    : str | None — set if submission failed
    """
    result: dict[str, Any] = {"error": None}

    try:
        run_status = client.submit_run(config_json)
        result.update({
            "run_id":   run_status.run_id,
            "status":   run_status.status,
            "run_name": run_status.run_name,
        })
    except APIResponseError as exc:
        result["error"] = f"Submission failed (HTTP {exc.status_code}): {exc.detail}"
    except APIError as exc:
        result["error"] = f"API unreachable: {exc}"

    return result


# Tool JSON schema — registered in BaseAgent's tool list.
TOOL_SCHEMA: dict[str, Any] = {
    "name": "submit_run",
    "description": (
        "Submit a new ALM projection run using the provided RunConfig JSON string. "
        "IMPORTANT: Only call this tool after the actuary has explicitly approved the "
        "proposed configuration. The desktop UI will show a confirmation dialog before "
        "this tool executes — do not attempt to bypass it. "
        "Returns the new run_id on success."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "config_json": {
                "type": "string",
                "description": (
                    "Full RunConfig serialised as a JSON string. "
                    "Must be valid JSON matching the RunConfig schema. "
                    "Use get_run_config to read an existing config before modifying it."
                ),
            }
        },
        "required": ["config_json"],
    },
}
