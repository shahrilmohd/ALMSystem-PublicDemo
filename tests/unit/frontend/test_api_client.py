"""
Unit tests for frontend/desktop/api_client.py.

Uses the `responses` library to mock HTTP calls so no running API server
is needed.  Every public method on ALMApiClient is tested.

Test groups
-----------
TestSubmitRun       — POST /runs
TestGetRun          — GET /runs/{run_id}
TestListRuns        — GET /runs
TestSubmitBatch     — POST /batches
TestGetBatch        — GET /batches/{batch_id}
TestListBatches     — GET /batches
TestListWorkers     — GET /workers
TestGetResultsSummary — GET /results/{run_id}/summary
TestGetResultsCsv   — GET /results/{run_id}?format=csv
TestValidateConfig  — POST /config/validate
TestIsReachable     — connectivity helper
TestErrors          — APIError and APIResponseError propagation
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import requests
import responses as resp_lib  # alias to avoid collision with pytest fixture name
from responses import RequestsMock

from frontend.desktop.api_client import (
    ALMApiClient,
    APIError,
    APIResponseError,
    BatchStatus,
    ResultsSummary,
    RunStatus,
    WorkerList,
)

BASE = "http://localhost:8000"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client() -> ALMApiClient:
    return ALMApiClient(base_url=BASE, timeout=5.0)


def _run_dict(**overrides) -> dict:
    """Minimal valid RunStatusResponse dict."""
    d = {
        "run_id":           "run-abc-123",
        "run_type":         "DETERMINISTIC",
        "status":           "PENDING",
        "created_at":       "2026-03-01T10:00:00+00:00",
        "started_at":       None,
        "completed_at":     None,
        "duration_seconds": None,
        "error_message":    None,
        "n_scenarios":      None,
        "n_timesteps":      None,
    }
    d.update(overrides)
    return d


def _batch_dict(**overrides) -> dict:
    d = {
        "batch_id":       "batch-xyz",
        "label":          "Test batch",
        "status":         "PENDING",
        "created_at":     "2026-03-01T10:00:00+00:00",
        "total_runs":     2,
        "completed_runs": 0,
        "failed_runs":    0,
        "pending_runs":   2,
        "runs":           [_run_dict(run_id=f"run-{i}") for i in range(2)],
    }
    d.update(overrides)
    return d


def _worker_list_dict(**overrides) -> dict:
    d = {
        "total_workers": 2,
        "idle_workers":  1,
        "busy_workers":  1,
        "workers": [
            {"name": "worker-1", "state": "idle",  "current_job_id": None,      "queues": ["alm"]},
            {"name": "worker-2", "state": "busy",  "current_job_id": "job-999", "queues": ["alm"]},
        ],
    }
    d.update(overrides)
    return d


# ---------------------------------------------------------------------------
# POST /runs
# ---------------------------------------------------------------------------

class TestSubmitRun:

    @resp_lib.activate
    def test_returns_run_status(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.POST, f"{BASE}/runs/",
            json=_run_dict(status="PENDING"),
            status=201,
        )
        run = client.submit_run('{"run_type": "deterministic"}')
        assert isinstance(run, RunStatus)
        assert run.run_id == "run-abc-123"
        assert run.status == "PENDING"

    @resp_lib.activate
    def test_raises_api_response_error_on_400(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.POST, f"{BASE}/runs/",
            json={"detail": "config_json is not valid JSON"},
            status=400,
        )
        with pytest.raises(APIResponseError) as exc_info:
            client.submit_run("not-json")
        assert exc_info.value.status_code == 400
        assert "config_json" in exc_info.value.detail

    @resp_lib.activate
    def test_raises_api_response_error_on_503(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.POST, f"{BASE}/runs/",
            json={"detail": "Redis not reachable"},
            status=503,
        )
        with pytest.raises(APIResponseError) as exc_info:
            client.submit_run('{}')
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# GET /runs/{run_id}
# ---------------------------------------------------------------------------

class TestGetRun:

    @resp_lib.activate
    def test_returns_run_status(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/runs/run-abc-123",
            json=_run_dict(status="COMPLETED", duration_seconds=42.5),
            status=200,
        )
        run = client.get_run("run-abc-123")
        assert run.status == "COMPLETED"
        assert run.duration_seconds == 42.5

    @resp_lib.activate
    def test_raises_404(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/runs/missing",
            json={"detail": "Run not found"},
            status=404,
        )
        with pytest.raises(APIResponseError) as exc_info:
            client.get_run("missing")
        assert exc_info.value.status_code == 404

    @resp_lib.activate
    def test_parses_optional_datetime_fields(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/runs/run-abc-123",
            json=_run_dict(
                status="COMPLETED",
                started_at="2026-03-01T10:01:00+00:00",
                completed_at="2026-03-01T10:05:00+00:00",
                duration_seconds=240.0,
            ),
            status=200,
        )
        run = client.get_run("run-abc-123")
        assert run.started_at is not None
        assert run.completed_at is not None
        assert run.duration_seconds == 240.0


# ---------------------------------------------------------------------------
# GET /runs
# ---------------------------------------------------------------------------

class TestListRuns:

    @resp_lib.activate
    def test_returns_list(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/runs/",
            json={"runs": [_run_dict(), _run_dict(run_id="run-2")], "total": 2},
            status=200,
        )
        runs = client.list_runs()
        assert len(runs) == 2
        assert all(isinstance(r, RunStatus) for r in runs)

    @resp_lib.activate
    def test_returns_empty_list(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/runs/",
            json={"runs": [], "total": 0},
            status=200,
        )
        runs = client.list_runs()
        assert runs == []


# ---------------------------------------------------------------------------
# POST /batches
# ---------------------------------------------------------------------------

class TestSubmitBatch:

    @resp_lib.activate
    def test_returns_batch_status(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.POST, f"{BASE}/batches/",
            json=_batch_dict(),
            status=201,
        )
        batch = client.submit_batch(['{"run_type":"deterministic"}'] * 2, label="Test batch")
        assert isinstance(batch, BatchStatus)
        assert batch.total_runs == 2
        assert batch.status == "PENDING"
        assert len(batch.runs) == 2

    @resp_lib.activate
    def test_raises_on_400(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.POST, f"{BASE}/batches/",
            json={"detail": "configs[0] is not valid JSON"},
            status=400,
        )
        with pytest.raises(APIResponseError) as exc_info:
            client.submit_batch(["bad-json"])
        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# GET /batches/{batch_id}
# ---------------------------------------------------------------------------

class TestGetBatch:

    @resp_lib.activate
    def test_returns_batch_status(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/batches/batch-xyz",
            json=_batch_dict(status="RUNNING", completed_runs=1),
            status=200,
        )
        batch = client.get_batch("batch-xyz")
        assert batch.batch_id == "batch-xyz"
        assert batch.status == "RUNNING"
        assert batch.completed_runs == 1

    @resp_lib.activate
    def test_raises_404(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/batches/missing",
            json={"detail": "Batch not found"},
            status=404,
        )
        with pytest.raises(APIResponseError) as exc_info:
            client.get_batch("missing")
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# GET /batches
# ---------------------------------------------------------------------------

class TestListBatches:

    @resp_lib.activate
    def test_returns_list(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/batches/",
            json={"batches": [_batch_dict()], "total": 1},
            status=200,
        )
        batches = client.list_batches()
        assert len(batches) == 1
        assert isinstance(batches[0], BatchStatus)


# ---------------------------------------------------------------------------
# GET /workers
# ---------------------------------------------------------------------------

class TestListWorkers:

    @resp_lib.activate
    def test_returns_worker_list(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/workers/",
            json=_worker_list_dict(),
            status=200,
        )
        result = client.list_workers()
        assert isinstance(result, WorkerList)
        assert result.total_workers == 2
        assert result.idle_workers  == 1
        assert result.busy_workers  == 1
        assert len(result.workers)  == 2
        assert result.workers[0].name  == "worker-1"
        assert result.workers[0].state == "idle"
        assert result.workers[1].current_job_id == "job-999"

    @resp_lib.activate
    def test_raises_503_when_redis_down(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/workers/",
            json={"detail": "Redis not reachable"},
            status=503,
        )
        with pytest.raises(APIResponseError) as exc_info:
            client.list_workers()
        assert exc_info.value.status_code == 503


# ---------------------------------------------------------------------------
# GET /results/{run_id}/summary
# ---------------------------------------------------------------------------

class TestGetResultsSummary:

    @resp_lib.activate
    def test_returns_summary(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/results/run-abc-123/summary",
            json={
                "run_id":                   "run-abc-123",
                "n_result_rows":            360,
                "n_scenarios":              1,
                "n_timesteps":              360,
                "final_bel":                1_234_567.89,
                "final_total_market_value": 2_000_000.00,
            },
            status=200,
        )
        summary = client.get_results_summary("run-abc-123")
        assert isinstance(summary, ResultsSummary)
        assert summary.n_result_rows == 360
        assert summary.final_bel == pytest.approx(1_234_567.89)

    @resp_lib.activate
    def test_returns_none_for_empty_run(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/results/run-abc-123/summary",
            json={
                "run_id": "run-abc-123",
                "n_result_rows": 0,
                "n_scenarios": 0,
                "n_timesteps": 0,
                "final_bel": None,
                "final_total_market_value": None,
            },
            status=200,
        )
        summary = client.get_results_summary("run-abc-123")
        assert summary.final_bel is None
        assert summary.final_total_market_value is None


# ---------------------------------------------------------------------------
# GET /results/{run_id}?format=csv
# ---------------------------------------------------------------------------

class TestGetResultsCsv:

    @resp_lib.activate
    def test_returns_bytes(self, client: ALMApiClient) -> None:
        csv_content = b"run_id,scenario_id,timestep,bel\nrun-1,0,1,100000\n"
        resp_lib.add(
            resp_lib.GET, f"{BASE}/results/run-abc-123",
            body=csv_content,
            status=200,
            content_type="text/csv",
        )
        result = client.get_results_csv("run-abc-123")
        assert isinstance(result, bytes)
        assert b"bel" in result

    @resp_lib.activate
    def test_raises_404_for_unknown_run(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/results/missing",
            json={"detail": "Run not found"},
            status=404,
        )
        with pytest.raises(APIResponseError) as exc_info:
            client.get_results_csv("missing")
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# POST /config/validate
# ---------------------------------------------------------------------------

class TestValidateConfig:

    @resp_lib.activate
    def test_returns_dict_on_success(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.POST, f"{BASE}/config/validate",
            json={"valid": True, "message": "Config is valid."},
            status=200,
        )
        result = client.validate_config('{"run_type": "deterministic"}')
        assert result["valid"] is True

    @resp_lib.activate
    def test_raises_on_invalid_config(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.POST, f"{BASE}/config/validate",
            json={"detail": "run_name is required"},
            status=422,
        )
        with pytest.raises(APIResponseError) as exc_info:
            client.validate_config("{}")
        assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# is_reachable()
# ---------------------------------------------------------------------------

class TestIsReachable:

    @resp_lib.activate
    def test_returns_true_when_api_responds(self, client: ALMApiClient) -> None:
        resp_lib.add(resp_lib.GET, f"{BASE}/runs/", json={"runs": [], "total": 0}, status=200)
        assert client.is_reachable() is True

    @resp_lib.activate
    def test_returns_false_on_connection_error(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/runs/",
            body=requests.exceptions.ConnectionError("refused"),
        )
        assert client.is_reachable() is False


# ---------------------------------------------------------------------------
# Connection errors (APIError)
# ---------------------------------------------------------------------------

class TestConnectionErrors:

    @resp_lib.activate
    def test_get_run_raises_api_error_on_connection_failure(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.GET, f"{BASE}/runs/run-abc-123",
            body=requests.exceptions.ConnectionError("connection refused"),
        )
        with pytest.raises(APIError):
            client.get_run("run-abc-123")

    @resp_lib.activate
    def test_submit_run_raises_api_error_on_connection_failure(self, client: ALMApiClient) -> None:
        resp_lib.add(
            resp_lib.POST, f"{BASE}/runs/",
            body=requests.exceptions.ConnectionError("connection refused"),
        )
        with pytest.raises(APIError):
            client.submit_run("{}")
