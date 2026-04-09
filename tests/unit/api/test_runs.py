"""
Unit tests for the /runs router.

Coverage
--------
POST /runs
    - valid JSON config → 201, PENDING status, run_id returned
    - invalid (non-JSON) config → 400
    - missing config_json field → 422 (FastAPI schema validation)

GET /runs
    - empty DB → 200 with empty list
    - one record → 200 with that record

GET /runs/{run_id}
    - existing run → 200 with correct fields
    - unknown run_id → 404
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from tests.unit.api.conftest import make_run_record


# ---------------------------------------------------------------------------
# POST /runs
# ---------------------------------------------------------------------------

class TestSubmitRun:
    def test_valid_config_returns_201_and_pending_status(self, client: TestClient):
        config = json.dumps({"run_type": "deterministic", "notes": "test run"})
        response = client.post("/runs/", json={"config_json": config})

        assert response.status_code == 201
        body = response.json()
        assert body["status"] == "PENDING"
        assert "run_id" in body
        assert len(body["run_id"]) > 0

    def test_valid_config_stores_run_type_from_config(self, client: TestClient):
        config = json.dumps({"run_type": "stochastic"})
        response = client.post("/runs/", json={"config_json": config})

        assert response.status_code == 201
        assert response.json()["run_type"] == "STOCHASTIC"

    def test_non_json_config_returns_400(self, client: TestClient):
        response = client.post("/runs/", json={"config_json": "this is not json {"})

        assert response.status_code == 400
        assert "not valid JSON" in response.json()["detail"]

    def test_missing_config_json_field_returns_422(self, client: TestClient):
        response = client.post("/runs/", json={})

        assert response.status_code == 422

    def test_run_is_persisted_in_db(self, client: TestClient, db_session: Session):
        config = json.dumps({"run_type": "liability_only"})
        response = client.post("/runs/", json={"config_json": config})
        run_id = response.json()["run_id"]

        # Verify the record is in the DB
        get_response = client.get(f"/runs/{run_id}")
        assert get_response.status_code == 200
        assert get_response.json()["run_id"] == run_id

    def test_each_submission_gets_unique_run_id(self, client: TestClient):
        config = json.dumps({"run_type": "deterministic"})
        r1 = client.post("/runs/", json={"config_json": config})
        r2 = client.post("/runs/", json={"config_json": config})

        assert r1.json()["run_id"] != r2.json()["run_id"]


# ---------------------------------------------------------------------------
# GET /runs
# ---------------------------------------------------------------------------

class TestListRuns:
    def test_empty_db_returns_empty_list(self, client: TestClient):
        response = client.get("/runs/")

        assert response.status_code == 200
        body = response.json()
        assert body["runs"] == []
        assert body["total"] == 0

    def test_returns_all_records(self, client: TestClient, db_session: Session):
        make_run_record(db_session, run_id="r1")
        make_run_record(db_session, run_id="r2", status="RUNNING")

        response = client.get("/runs/")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        returned_ids = {r["run_id"] for r in body["runs"]}
        assert returned_ids == {"r1", "r2"}

    def test_records_ordered_most_recent_first(self, client: TestClient, db_session: Session):
        # Insert two runs — list_all() orders by created_at descending
        make_run_record(db_session, run_id="older")
        make_run_record(db_session, run_id="newer")

        response = client.get("/runs/")
        run_ids = [r["run_id"] for r in response.json()["runs"]]
        # "newer" was inserted after "older" so should come first
        assert run_ids.index("newer") < run_ids.index("older")


# ---------------------------------------------------------------------------
# GET /runs/{run_id}
# ---------------------------------------------------------------------------

class TestGetRun:
    def test_returns_correct_run(self, client: TestClient, db_session: Session):
        make_run_record(db_session, run_id="run-abc", status="COMPLETED")

        response = client.get("/runs/run-abc")

        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == "run-abc"
        assert body["status"] == "COMPLETED"
        assert body["run_type"] == "DETERMINISTIC"

    def test_unknown_run_id_returns_404(self, client: TestClient):
        response = client.get("/runs/does-not-exist")

        assert response.status_code == 404

    def test_response_includes_all_schema_fields(self, client: TestClient, db_session: Session):
        make_run_record(db_session, run_id="run-fields")

        body = client.get("/runs/run-fields").json()

        expected_keys = {
            "run_id", "run_type", "status", "created_at",
            "started_at", "completed_at", "duration_seconds",
            "error_message", "n_scenarios", "n_timesteps",
        }
        assert expected_keys.issubset(body.keys())
