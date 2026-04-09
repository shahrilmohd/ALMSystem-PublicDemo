"""
get_run_config — fetch the RunConfig for an AI agent.

Calls GET /runs/{run_id} and returns the config_json as a parsed dict
alongside the run status metadata.  Read-only — no data is written.

Tool schema (registered in BaseAgent)
--------------------------------------
name:        "get_run_config"
description: "Fetch the run configuration (assumptions, projection settings) for a
              given run_id.  Use this to understand what drove the results before
              proposing any changes."
parameters:
  run_id (string, required): The UUID of the ALM projection run.
"""
from __future__ import annotations

import json
from typing import Any

from frontend.desktop.api_client import ALMApiClient, APIError, APIResponseError


def get_run_config(client: ALMApiClient, run_id: str) -> dict[str, Any]:
    """
    Fetch the run status and parsed RunConfig for a given run.

    Parameters
    ----------
    client : ALMApiClient
        Configured HTTP client pointing at the ALM API.
    run_id : str
        UUID of the run whose config to fetch.

    Returns
    -------
    dict with keys:
        run_id        : str
        run_name      : str | None
        run_type      : str
        status        : str
        created_at    : str  (ISO datetime)
        config        : dict — parsed RunConfig (from config_json field)
        error         : str | None — set if the fetch failed
    """
    result: dict[str, Any] = {"run_id": run_id, "error": None}

    try:
        run_status = client.get_run(run_id)
    except APIResponseError as exc:
        result["error"] = f"Run fetch failed (HTTP {exc.status_code}): {exc.detail}"
        return result
    except APIError as exc:
        result["error"] = f"API unreachable: {exc}"
        return result

    result.update({
        "run_name":   run_status.run_name,
        "run_type":   run_status.run_type,
        "status":     run_status.status,
        "created_at": run_status.created_at.isoformat() if run_status.created_at else None,
    })

    # The API client doesn't expose config_json directly — we need the raw dict.
    # We re-fetch via a direct GET to extract it.
    try:
        raw = client._get(f"/runs/{run_id}")
        config_json_str = raw.get("config_json", "{}")
        result["config"] = json.loads(config_json_str)
    except Exception as exc:
        result["config"] = {}
        result["config_parse_error"] = str(exc)

    return result


# Tool JSON schema — registered in BaseAgent's tool list.
TOOL_SCHEMA: dict[str, Any] = {
    "name": "get_run_config",
    "description": (
        "Fetch the run configuration (assumptions, projection settings, input file paths) "
        "for a given ALM run. "
        "Use this to understand the mortality rates, lapse rates, expense loadings, "
        "projection term, and all other settings that drove the run before explaining "
        "results or proposing changes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "UUID of the ALM projection run.",
            }
        },
        "required": ["run_id"],
    },
}
