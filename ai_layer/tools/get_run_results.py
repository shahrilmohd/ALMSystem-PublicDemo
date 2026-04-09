"""
get_run_results — fetch run results for an AI agent.

Calls:
  GET /results/{run_id}/summary   — lightweight result summary
  GET /results/{run_id}?format=csv — full result rows (first _MAX_ROWS only)

Returns a dict the agent can read and narrate.  Read-only — no data is written.

Tool schema (registered in BaseAgent)
--------------------------------------
name:        "get_run_results"
description: "Fetch the projection results and summary for a given run_id.
              Returns BEL, TVOG, scenario count, and the first rows of result data."
parameters:
  run_id (string, required): The UUID of the completed ALM projection run.
"""
from __future__ import annotations

import csv
import io
from typing import Any

from frontend.desktop.api_client import ALMApiClient, APIError, APIResponseError

_MAX_ROWS = 200  # cap rows sent to the model to avoid excessive token usage


def get_run_results(client: ALMApiClient, run_id: str) -> dict[str, Any]:
    """
    Fetch the result summary and first rows of result data for a run.

    Parameters
    ----------
    client : ALMApiClient
        Configured HTTP client pointing at the ALM API.
    run_id : str
        UUID of the run whose results to fetch.

    Returns
    -------
    dict with keys:
        run_id        : str
        n_result_rows : int
        n_scenarios   : int
        n_timesteps   : int
        final_bel     : float | None
        final_total_market_value : float | None
        result_preview : list[dict]  — first _MAX_ROWS rows as list of dicts
        error         : str | None   — set if the fetch failed

    Raises
    ------
    Never raises — errors are captured in the "error" key so the agent can
    report them gracefully.
    """
    result: dict[str, Any] = {"run_id": run_id, "error": None}

    # Fetch summary.
    try:
        summary = client.get_results_summary(run_id)
        result.update({
            "n_result_rows":            summary.n_result_rows,
            "n_scenarios":              summary.n_scenarios,
            "n_timesteps":              summary.n_timesteps,
            "final_bel":                summary.final_bel,
            "final_total_market_value": summary.final_total_market_value,
        })
    except APIResponseError as exc:
        result["error"] = f"Summary fetch failed: {exc.detail}"
        return result
    except APIError as exc:
        result["error"] = f"API unreachable: {exc}"
        return result

    # Fetch CSV and parse first _MAX_ROWS rows.
    try:
        csv_bytes = client.get_results_csv(run_id)
        result["result_preview"] = _parse_csv_preview(csv_bytes)
    except (APIError, APIResponseError) as exc:
        # Non-fatal — summary is still available.
        result["result_preview"] = []
        result["csv_error"] = str(exc)

    return result


def _parse_csv_preview(csv_bytes: bytes) -> list[dict[str, str]]:
    """Parse CSV bytes and return first _MAX_ROWS rows as a list of dicts."""
    text = csv_bytes.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows: list[dict[str, str]] = []
    for i, row in enumerate(reader):
        if i >= _MAX_ROWS:
            break
        rows.append(dict(row))
    return rows


# Tool JSON schema — registered in BaseAgent's tool list.
TOOL_SCHEMA: dict[str, Any] = {
    "name": "get_run_results",
    "description": (
        "Fetch the projection results and summary for a given ALM run. "
        "Returns key metrics (BEL, TVOG, scenario count, timestep count) and "
        f"a preview of the first {_MAX_ROWS} result rows. "
        "Use this before explaining or interpreting any run output."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "run_id": {
                "type": "string",
                "description": "UUID of the completed ALM projection run.",
            }
        },
        "required": ["run_id"],
    },
}
