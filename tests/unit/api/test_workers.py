"""
Unit tests for the /workers router.

Coverage
--------
GET /workers
    - no workers registered → 200, total=0, empty list
    - one idle worker → correct fields returned
    - one busy worker → state="busy", current_job_id populated
    - mixed idle + busy → counts are correct
    - Redis unreachable → 503

Strategy
--------
The router now queries Redis directly (no rq import) to avoid the Windows
fork-context crash in rq.scheduler.  Tests mock the Redis connection object
returned by get_redis_connection() and configure smembers / hgetall responses.

Redis data layout used by RQ 2.x:
    smembers("rq:workers")            → set of worker-key bytes
                                        e.g. {b"rq:worker:hostname.12345"}
    hgetall("rq:worker:<name>")       → dict[bytes, bytes] with fields:
                                        name, state, current_job, queues
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import redis as redis_lib
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_redis_conn(workers: list[dict]) -> MagicMock:
    """
    Build a mock Redis connection whose smembers / hgetall return the given
    worker dicts.

    Each dict should have keys matching what our router reads:
        name, state, current_job (optional), queues (comma-separated string)
    """
    conn = MagicMock()

    # Build the set of worker key bytes
    worker_keys = {
        f"rq:worker:{w['name']}".encode() for w in workers
    }
    conn.smembers.return_value = worker_keys

    # Map each key to its hgetall response
    def hgetall_side_effect(key: str | bytes) -> dict[bytes, bytes]:
        key_str = key.decode() if isinstance(key, bytes) else key
        for w in workers:
            if key_str == f"rq:worker:{w['name']}":
                data: dict[bytes, bytes] = {
                    b"name":   w["name"].encode(),
                    b"state":  w["state"].encode(),
                    b"queues": w.get("queues", "alm_jobs").encode(),
                }
                if w.get("current_job"):
                    data[b"current_job"] = w["current_job"].encode()
                return data
        return {}

    conn.hgetall.side_effect = hgetall_side_effect
    return conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestListWorkers:
    def test_no_workers_returns_empty_list(self, client: TestClient):
        mock_conn = _make_redis_conn([])
        with patch("api.routers.workers.get_redis_connection", return_value=mock_conn):
            response = client.get("/workers/")

        assert response.status_code == 200
        body = response.json()
        assert body["total_workers"] == 0
        assert body["idle_workers"] == 0
        assert body["busy_workers"] == 0
        assert body["workers"] == []

    def test_single_idle_worker_fields(self, client: TestClient):
        mock_conn = _make_redis_conn([
            {"name": "worker-1", "state": "idle"},
        ])
        with patch("api.routers.workers.get_redis_connection", return_value=mock_conn):
            response = client.get("/workers/")

        assert response.status_code == 200
        body = response.json()
        assert body["total_workers"] == 1
        assert body["idle_workers"] == 1
        assert body["busy_workers"] == 0

        w = body["workers"][0]
        assert w["name"] == "worker-1"
        assert w["state"] == "idle"
        assert w["current_job_id"] is None
        assert w["queues"] == ["alm_jobs"]

    def test_busy_worker_includes_job_id(self, client: TestClient):
        mock_conn = _make_redis_conn([
            {"name": "worker-2", "state": "busy", "current_job": "run-abc-123"},
        ])
        with patch("api.routers.workers.get_redis_connection", return_value=mock_conn):
            response = client.get("/workers/")

        assert response.status_code == 200
        w = response.json()["workers"][0]
        assert w["state"] == "busy"
        assert w["current_job_id"] == "run-abc-123"

    def test_mixed_workers_counts_are_correct(self, client: TestClient):
        mock_conn = _make_redis_conn([
            {"name": "w1", "state": "idle"},
            {"name": "w2", "state": "busy", "current_job": "run-1"},
            {"name": "w3", "state": "busy", "current_job": "run-2"},
            {"name": "w4", "state": "idle"},
        ])
        with patch("api.routers.workers.get_redis_connection", return_value=mock_conn):
            response = client.get("/workers/")

        assert response.status_code == 200
        body = response.json()
        assert body["total_workers"] == 4
        assert body["idle_workers"] == 2
        assert body["busy_workers"] == 2
        assert len(body["workers"]) == 4

    def test_redis_unavailable_returns_503(self, client: TestClient):
        with patch(
            "api.routers.workers.get_redis_connection",
            side_effect=redis_lib.exceptions.ConnectionError("refused"),
        ):
            response = client.get("/workers/")

        assert response.status_code == 503
        assert "Redis" in response.json()["detail"]
