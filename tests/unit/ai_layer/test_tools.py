"""
Unit tests for ai_layer tools: get_run_results, get_run_config, submit_run.

All tests mock ALMApiClient — no real HTTP calls are made.

Coverage
--------
get_run_results
    - returns summary fields and result_preview on success
    - sets error key when summary fetch fails (APIResponseError)
    - sets error key when API unreachable (APIError)
    - result_preview is empty list when CSV fetch fails (non-fatal)
    - result_preview is capped at _MAX_ROWS rows

get_run_config
    - returns run metadata and parsed config dict on success
    - sets error key when run fetch fails (APIResponseError)
    - sets error key when API unreachable (APIError)
    - config_parse_error set when config_json is not valid JSON

submit_run
    - returns run_id, status, run_name on success
    - sets error key on APIResponseError
    - sets error key on APIError
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from ai_layer.tools.get_run_config import get_run_config
from ai_layer.tools.get_run_results import _MAX_ROWS, get_run_results
from ai_layer.tools.submit_run import submit_run
from frontend.desktop.api_client import APIError, APIResponseError


# ---------------------------------------------------------------------------
# Shared test doubles
# ---------------------------------------------------------------------------

@dataclass
class _FakeRunStatus:
    run_id:      str
    run_type:    str
    status:      str
    run_name:    Optional[str]
    created_at:  Optional[datetime]


@dataclass
class _FakeResultsSummary:
    run_id:                   str
    n_result_rows:            int
    n_scenarios:              int
    n_timesteps:              int
    final_bel:                Optional[float]
    final_total_market_value: Optional[float]


def _make_csv_bytes(n_rows: int) -> bytes:
    header = "scenario_id,timestep,bel\n"
    rows   = "".join(f"1,{t},100.0\n" for t in range(n_rows))
    return (header + rows).encode()


# ---------------------------------------------------------------------------
# get_run_results
# ---------------------------------------------------------------------------

class TestGetRunResults:
    def _make_client(self, summary=None, csv_bytes=None, summary_exc=None, csv_exc=None):
        client = MagicMock()
        if summary_exc:
            client.get_results_summary.side_effect = summary_exc
        else:
            client.get_results_summary.return_value = summary or _FakeResultsSummary(
                run_id="r1", n_result_rows=10, n_scenarios=5,
                n_timesteps=12, final_bel=1000.0, final_total_market_value=1100.0,
            )
        if csv_exc:
            client.get_results_csv.side_effect = csv_exc
        else:
            client.get_results_csv.return_value = csv_bytes or _make_csv_bytes(5)
        return client

    def test_returns_summary_fields(self):
        client = self._make_client()
        result = get_run_results(client, "r1")
        assert result["n_scenarios"]  == 5
        assert result["n_timesteps"]  == 12
        assert result["final_bel"]    == 1000.0
        assert result["error"] is None

    def test_result_preview_populated(self):
        client = self._make_client(csv_bytes=_make_csv_bytes(3))
        result = get_run_results(client, "r1")
        assert len(result["result_preview"]) == 3
        assert result["result_preview"][0]["bel"] == "100.0"

    def test_result_preview_capped_at_max_rows(self):
        client = self._make_client(csv_bytes=_make_csv_bytes(_MAX_ROWS + 50))
        result = get_run_results(client, "r1")
        assert len(result["result_preview"]) == _MAX_ROWS

    def test_summary_api_response_error_sets_error_key(self):
        exc    = APIResponseError(404, "Run not found")
        client = self._make_client(summary_exc=exc)
        result = get_run_results(client, "r1")
        assert result["error"] is not None
        assert "404" in result["error"] or "Run not found" in result["error"]

    def test_summary_api_error_sets_error_key(self):
        client = self._make_client(summary_exc=APIError("connection refused"))
        result = get_run_results(client, "r1")
        assert "API unreachable" in result["error"]

    def test_csv_failure_is_non_fatal(self):
        client = self._make_client(csv_exc=APIError("timeout"))
        result = get_run_results(client, "r1")
        assert result["error"] is None          # summary succeeded
        assert result["result_preview"] == []   # CSV failed gracefully
        assert "csv_error" in result


# ---------------------------------------------------------------------------
# get_run_config
# ---------------------------------------------------------------------------

class TestGetRunConfig:
    _CONFIG = {"run_type": "STOCHASTIC", "n_scenarios": 200}

    def _make_client(self, run_exc=None, raw_config=None, raw_exc=None):
        client = MagicMock()
        if run_exc:
            client.get_run.side_effect = run_exc
        else:
            client.get_run.return_value = _FakeRunStatus(
                run_id="r1", run_type="STOCHASTIC", status="COMPLETED",
                run_name="test run", created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            )
        if raw_exc:
            client._get.side_effect = raw_exc
        else:
            config_json = json.dumps(raw_config or self._CONFIG)
            client._get.return_value = {"config_json": config_json}
        return client

    def test_returns_run_metadata(self):
        client = self._make_client()
        result = get_run_config(client, "r1")
        assert result["run_type"]  == "STOCHASTIC"
        assert result["status"]    == "COMPLETED"
        assert result["run_name"]  == "test run"
        assert result["error"] is None

    def test_returns_parsed_config(self):
        client = self._make_client()
        result = get_run_config(client, "r1")
        assert result["config"]["n_scenarios"] == 200

    def test_run_fetch_error_sets_error_key(self):
        client = self._make_client(run_exc=APIResponseError(404, "not found"))
        result = get_run_config(client, "r1")
        assert result["error"] is not None

    def test_api_unreachable_sets_error_key(self):
        client = self._make_client(run_exc=APIError("timeout"))
        result = get_run_config(client, "r1")
        assert "API unreachable" in result["error"]

    def test_invalid_config_json_sets_parse_error(self):
        client = self._make_client()
        client._get.return_value = {"config_json": "not valid json {"}
        result = get_run_config(client, "r1")
        assert "config_parse_error" in result
        assert result["config"] == {}


# ---------------------------------------------------------------------------
# submit_run
# ---------------------------------------------------------------------------

class TestSubmitRun:
    _CONFIG_JSON = json.dumps({"run_type": "DETERMINISTIC"})

    def _make_client(self, exc=None):
        client = MagicMock()
        if exc:
            client.submit_run.side_effect = exc
        else:
            client.submit_run.return_value = _FakeRunStatus(
                run_id="new-run-1", run_type="DETERMINISTIC",
                status="PENDING", run_name="stress test",
                created_at=datetime.now(timezone.utc),
            )
        return client

    def test_returns_run_id_and_status_on_success(self):
        client = self._make_client()
        result = submit_run(client, self._CONFIG_JSON)
        assert result["run_id"]  == "new-run-1"
        assert result["status"]  == "PENDING"
        assert result["run_name"] == "stress test"
        assert result["error"] is None

    def test_api_response_error_sets_error_key(self):
        client = self._make_client(exc=APIResponseError(422, "invalid config"))
        result = submit_run(client, self._CONFIG_JSON)
        assert result["error"] is not None
        assert "422" in result["error"] or "invalid config" in result["error"]

    def test_api_error_sets_error_key(self):
        client = self._make_client(exc=APIError("connection refused"))
        result = submit_run(client, self._CONFIG_JSON)
        assert "API unreachable" in result["error"]
