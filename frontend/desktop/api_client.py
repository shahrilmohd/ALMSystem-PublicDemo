"""
API client for the ALM desktop frontend.

All HTTP communication between the desktop app and the FastAPI backend goes
through this single class.  No other frontend module imports `requests` or
constructs URLs directly.

Design
------
- Every method maps 1-to-1 to one API endpoint.
- All responses are returned as plain dicts or dataclasses — no Pydantic
  engine imports.  The frontend never imports from engine/ or api/.
- Errors surface as APIError (connection failures) or APIResponseError
  (HTTP 4xx/5xx from the server).
- base_url is configurable so tests can point at a test server.

Usage
-----
    client = ALMApiClient(base_url="http://localhost:8000")
    status = client.submit_run(config_json)
    status = client.get_run(run_id)
    workers = client.list_workers()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import requests


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class APIError(Exception):
    """Raised when the API cannot be reached (network / connection error)."""


class APIResponseError(Exception):
    """Raised when the API returns a 4xx or 5xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


# ---------------------------------------------------------------------------
# Response dataclasses
# The frontend only needs a subset of fields — keep these lean.
# ---------------------------------------------------------------------------

@dataclass
class RunStatus:
    run_id:           str
    run_type:         str
    status:           str
    created_at:       Optional[datetime]   = None
    run_name:         Optional[str]        = None
    started_at:       Optional[datetime]   = None
    completed_at:     Optional[datetime]   = None
    duration_seconds: Optional[float]      = None
    error_message:    Optional[str]        = None
    n_scenarios:      Optional[int]        = None
    n_timesteps:      Optional[int]        = None


@dataclass
class BatchStatus:
    batch_id:       str
    status:         str
    total_runs:     int
    completed_runs: int
    failed_runs:    int
    pending_runs:   int
    created_at:     Optional[datetime]     = None
    runs:           list[RunStatus]        = field(default_factory=list)
    label:          Optional[str]          = None


@dataclass
class WorkerInfo:
    name:            str
    state:           str
    current_job_id:  Optional[str]
    queues:          list[str]             = field(default_factory=list)


@dataclass
class WorkerList:
    total_workers:  int
    idle_workers:   int
    busy_workers:   int
    workers:        list[WorkerInfo]       = field(default_factory=list)


@dataclass
class ResultsSummary:
    run_id:                   str
    n_result_rows:            int
    n_scenarios:              int
    n_timesteps:              int
    final_bel:                Optional[float]
    final_total_market_value: Optional[float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_dt(v: Optional[str]) -> Optional[datetime]:
    """Parse an ISO datetime string returned by the API, or return None."""
    if v is None:
        return None
    # Python 3.11+ handles 'Z' suffix; earlier versions need replacement.
    return datetime.fromisoformat(v.replace("Z", "+00:00"))


def _parse_run(d: dict) -> RunStatus:
    return RunStatus(
        run_id=d["run_id"],
        run_name=d.get("run_name"),
        run_type=d["run_type"],
        status=d["status"],
        created_at=_parse_dt(d["created_at"]),
        started_at=_parse_dt(d.get("started_at")),
        completed_at=_parse_dt(d.get("completed_at")),
        duration_seconds=d.get("duration_seconds"),
        error_message=d.get("error_message"),
        n_scenarios=d.get("n_scenarios"),
        n_timesteps=d.get("n_timesteps"),
    )


def _parse_batch(d: dict) -> BatchStatus:
    return BatchStatus(
        batch_id=d["batch_id"],
        label=d.get("label"),
        status=d["status"],
        created_at=_parse_dt(d["created_at"]),
        total_runs=d["total_runs"],
        completed_runs=d["completed_runs"],
        failed_runs=d["failed_runs"],
        pending_runs=d["pending_runs"],
        runs=[_parse_run(r) for r in d.get("runs", [])],
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class ALMApiClient:
    """
    HTTP client for the ALM FastAPI backend.

    Parameters
    ----------
    base_url : str
        Base URL of the running FastAPI server.
        Default: http://localhost:8000
    timeout : float
        Request timeout in seconds.  Default: 10.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_detail(resp: requests.Response) -> str:
        """Extract a human-readable error detail from a non-OK response."""
        if not resp.content:
            return resp.reason or str(resp.status_code)
        try:
            return resp.json().get("detail", resp.text)
        except ValueError:
            return resp.text or resp.reason or str(resp.status_code)

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self._base}{path}"
        try:
            resp = requests.get(url, params=params, timeout=self._timeout)
        except requests.exceptions.ConnectionError as exc:
            raise APIError(f"Cannot reach API at {url}: {exc}") from exc
        if not resp.ok:
            raise APIResponseError(resp.status_code, self._extract_detail(resp))
        return resp.json()

    def _post(self, path: str, body: dict, timeout: float | None = None) -> dict:
        url = f"{self._base}{path}"
        try:
            resp = requests.post(url, json=body, timeout=timeout or self._timeout)
        except requests.exceptions.ConnectionError as exc:
            raise APIError(f"Cannot reach API at {url}: {exc}") from exc
        if not resp.ok:
            raise APIResponseError(resp.status_code, self._extract_detail(resp))
        return resp.json()

    def _get_bytes(self, path: str, params: Optional[dict] = None) -> bytes:
        url = f"{self._base}{path}"
        try:
            resp = requests.get(url, params=params, timeout=self._timeout)
        except requests.exceptions.ConnectionError as exc:
            raise APIError(f"Cannot reach API at {url}: {exc}") from exc
        if not resp.ok:
            raise APIResponseError(resp.status_code, self._extract_detail(resp))
        return resp.content

    # -----------------------------------------------------------------------
    # Runs
    # -----------------------------------------------------------------------

    def submit_run(self, config_json: str) -> RunStatus:
        """POST /runs — submit a single projection run."""
        data = self._post("/runs/", {"config_json": config_json})
        return _parse_run(data)

    def get_run(self, run_id: str) -> RunStatus:
        """GET /runs/{run_id} — poll run status."""
        data = self._get(f"/runs/{run_id}")
        return _parse_run(data)

    def list_runs(self) -> list[RunStatus]:
        """GET /runs — list all runs, most recent first."""
        data = self._get("/runs/")
        return [_parse_run(r) for r in data.get("runs", [])]

    # -----------------------------------------------------------------------
    # Batches
    # -----------------------------------------------------------------------

    def submit_batch(self, configs: list[str], label: Optional[str] = None) -> BatchStatus:
        """POST /batches — submit multiple run configs as one batch."""
        body: dict = {"configs": configs}
        if label is not None:
            body["label"] = label
        data = self._post("/batches/", body)
        return _parse_batch(data)

    def get_batch(self, batch_id: str) -> BatchStatus:
        """GET /batches/{batch_id} — poll batch status."""
        data = self._get(f"/batches/{batch_id}")
        return _parse_batch(data)

    def list_batches(self) -> list[BatchStatus]:
        """GET /batches — list all batches, most recent first."""
        data = self._get("/batches/")
        return [_parse_batch(b) for b in data.get("batches", [])]

    # -----------------------------------------------------------------------
    # Workers
    # -----------------------------------------------------------------------

    def list_workers(self) -> WorkerList:
        """GET /workers — read-only worker status from Redis/RQ."""
        data = self._get("/workers/")
        workers = [
            WorkerInfo(
                name=w["name"],
                state=w["state"],
                current_job_id=w.get("current_job_id"),
                queues=w.get("queues", []),
            )
            for w in data.get("workers", [])
        ]
        return WorkerList(
            total_workers=data["total_workers"],
            idle_workers=data["idle_workers"],
            busy_workers=data["busy_workers"],
            workers=workers,
        )

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------

    def get_results_summary(self, run_id: str) -> ResultsSummary:
        """GET /results/{run_id}/summary — lightweight result summary."""
        data = self._get(f"/results/{run_id}/summary")
        return ResultsSummary(
            run_id=data["run_id"],
            n_result_rows=data["n_result_rows"],
            n_scenarios=data["n_scenarios"],
            n_timesteps=data["n_timesteps"],
            final_bel=data.get("final_bel"),
            final_total_market_value=data.get("final_total_market_value"),
        )

    def get_results_csv(self, run_id: str, scenario_id: Optional[int] = None) -> bytes:
        """GET /results/{run_id}?format=csv — download full results as CSV bytes."""
        params: dict = {"format": "csv"}
        if scenario_id is not None:
            params["scenario_id"] = scenario_id
        return self._get_bytes(f"/results/{run_id}", params=params)

    # -----------------------------------------------------------------------
    # Data — model points and assumption tables (AI tool use)
    # -----------------------------------------------------------------------

    def get_model_points(
        self,
        run_id: str,
        population_type: Optional[str] = None,
    ) -> dict:
        """
        GET /runs/{run_id}/model_points — fetch BPA model point rows for a run.

        Returns the raw response dict with keys:
            run_id, population_type, row_count, truncated, columns, data
        """
        params: dict = {}
        if population_type is not None:
            params["population_type"] = population_type
        return self._get(f"/runs/{run_id}/model_points", params=params or None)

    def get_assumption_table(
        self,
        table_name: str,
        valuation_date: str,
    ) -> dict:
        """
        GET /assumptions/{table_name}?valuation_date=... — read one assumption table.

        Returns the raw response dict with keys:
            table_name, valuation_date, row_count, truncated, columns, data
        """
        return self._get(f"/assumptions/{table_name}", params={"valuation_date": valuation_date})

    def list_assumption_tables(self, valuation_date: str) -> dict:
        """
        GET /assumptions?valuation_date=... — list available assumption tables.

        Returns the raw response dict with keys:
            valuation_date, tables
        """
        return self._get("/assumptions", params={"valuation_date": valuation_date})

    # -----------------------------------------------------------------------
    # Config validation
    # -----------------------------------------------------------------------

    def validate_config(self, config_json: str) -> dict:
        """POST /config/validate — validate a RunConfig JSON string server-side."""
        return self._post("/config/validate", {"config_json": config_json})

    # -----------------------------------------------------------------------
    # AI Assistant
    # -----------------------------------------------------------------------

    def ai_chat(
        self,
        message: str,
        *,
        session_id: Optional[str] = None,
        context_run_id: Optional[str] = None,
        provider: str = "anthropic",
        model: str = "claude-opus-4-6",
        base_url: Optional[str] = None,
        deployment_mode: str = "development",
        external_provider: Optional[str] = None,
        external_model: Optional[str] = None,
        external_base_url: Optional[str] = None,
    ) -> dict:
        """
        POST /ai/chat — send a message to the AI assistant.

        external_provider / external_model / external_base_url are optional.
        When set they configure the separate external LLM used exclusively by
        RegulatoryResearchAgent (public regulatory questions, no firm data).
        Omit them to disable regulatory research routing for this session.

        Returns the raw response dict with keys:
            reply, session_id, agent_used, pending_submit, tool_calls
        """
        body: dict = {
            "message":         message,
            "provider":        provider,
            "model":           model,
            "deployment_mode": deployment_mode,
        }
        if session_id:
            body["session_id"] = session_id
        if context_run_id:
            body["context_run_id"] = context_run_id
        if base_url:
            body["base_url"] = base_url
        if external_provider:
            body["external_provider"] = external_provider
        if external_model:
            body["external_model"] = external_model
        if external_base_url:
            body["external_base_url"] = external_base_url
        # LLM calls with large contexts can take 60–120 s — use a dedicated timeout.
        return self._post("/ai/chat", body, timeout=120.0)

    def ai_clear_session(self, session_id: str) -> None:
        """DELETE /ai/sessions/{session_id} — clear conversation history."""
        url = f"{self._base}/ai/sessions/{session_id}"
        try:
            import requests as _req
            _req.delete(url, timeout=self._timeout)
        except Exception:  # noqa: BLE001
            pass  # best-effort; session will expire naturally

    # -----------------------------------------------------------------------
    # Health
    # -----------------------------------------------------------------------

    def is_reachable(self) -> bool:
        """Return True if the API server responds, False otherwise."""
        try:
            requests.get(f"{self._base}/runs/", timeout=2.0)
            return True
        except requests.exceptions.RequestException:
            return False
