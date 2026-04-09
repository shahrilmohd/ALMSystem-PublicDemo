"""
Unit tests for the /batches router.

Coverage
--------
POST /batches
    - valid batch (2 configs) → 201, batch_id returned, status PENDING
    - total_runs count matches number of configs submitted
    - each member run gets a distinct run_id
    - label is stored and echoed back
    - invalid JSON in configs → 400 with index in detail
    - empty configs list → 422 (Pydantic min_length validation)
    - run_type is derived from config content

GET /batches/{batch_id}
    - unknown batch_id → 404
    - existing batch → 200 with correct fields
    - freshly submitted batch has status PENDING

GET /batches
    - empty DB → 200 with empty list
    - multiple batches → all returned
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session


# ---------------------------------------------------------------------------
# POST /batches
# ---------------------------------------------------------------------------

class TestSubmitBatch:
    def test_valid_batch_returns_201_with_batch_id(self, client: TestClient):
        configs = [
            json.dumps({"run_type": "deterministic"}),
            json.dumps({"run_type": "liability_only"}),
        ]
        response = client.post("/batches/", json={"configs": configs})

        assert response.status_code == 201
        body = response.json()
        assert "batch_id" in body
        assert len(body["batch_id"]) > 0
        assert body["status"] == "PENDING"

    def test_batch_creates_correct_number_of_runs(self, client: TestClient):
        configs = [
            json.dumps({"run_type": "deterministic"}),
            json.dumps({"run_type": "stochastic"}),
            json.dumps({"run_type": "liability_only"}),
        ]
        response = client.post("/batches/", json={"configs": configs})

        assert response.status_code == 201
        body = response.json()
        assert body["total_runs"] == 3
        assert len(body["runs"]) == 3

    def test_each_run_gets_unique_run_id(self, client: TestClient):
        configs = [
            json.dumps({"run_type": "deterministic"}),
            json.dumps({"run_type": "deterministic"}),
        ]
        response = client.post("/batches/", json={"configs": configs})

        assert response.status_code == 201
        runs = response.json()["runs"]
        run_ids = [r["run_id"] for r in runs]
        assert run_ids[0] != run_ids[1]

    def test_label_is_stored(self, client: TestClient):
        configs = [json.dumps({"run_type": "deterministic"})]
        response = client.post(
            "/batches/",
            json={"label": "my batch", "configs": configs},
        )

        assert response.status_code == 201
        assert response.json()["label"] == "my batch"

    def test_invalid_json_in_configs_returns_400(self, client: TestClient):
        configs = [
            json.dumps({"run_type": "deterministic"}),
            "this is { not json",
        ]
        response = client.post("/batches/", json={"configs": configs})

        assert response.status_code == 400
        detail = response.json()["detail"]
        # The error should identify which index failed.
        assert "configs[1]" in detail

    def test_empty_configs_returns_422(self, client: TestClient):
        response = client.post("/batches/", json={"configs": []})

        assert response.status_code == 422

    def test_run_type_derived_from_config(self, client: TestClient):
        configs = [json.dumps({"run_type": "stochastic", "n_scenarios": 1000})]
        response = client.post("/batches/", json={"configs": configs})

        assert response.status_code == 201
        runs = response.json()["runs"]
        assert runs[0]["run_type"] == "STOCHASTIC"


# ---------------------------------------------------------------------------
# GET /batches/{batch_id}
# ---------------------------------------------------------------------------

class TestGetBatch:
    def test_unknown_batch_returns_404(self, client: TestClient):
        response = client.get("/batches/does-not-exist")

        assert response.status_code == 404

    def test_returns_correct_batch(self, client: TestClient):
        configs = [
            json.dumps({"run_type": "deterministic"}),
            json.dumps({"run_type": "liability_only"}),
        ]
        post_response = client.post(
            "/batches/",
            json={"label": "regression run", "configs": configs},
        )
        batch_id = post_response.json()["batch_id"]

        get_response = client.get(f"/batches/{batch_id}")

        assert get_response.status_code == 200
        body = get_response.json()
        assert body["batch_id"] == batch_id
        assert body["label"] == "regression run"
        assert body["total_runs"] == 2
        assert len(body["runs"]) == 2

    def test_status_is_pending_for_new_batch(self, client: TestClient):
        configs = [json.dumps({"run_type": "deterministic"})]
        post_response = client.post("/batches/", json={"configs": configs})
        batch_id = post_response.json()["batch_id"]

        get_response = client.get(f"/batches/{batch_id}")

        assert get_response.status_code == 200
        assert get_response.json()["status"] == "PENDING"


# ---------------------------------------------------------------------------
# GET /batches
# ---------------------------------------------------------------------------

class TestListBatches:
    def test_empty_db_returns_empty_list(self, client: TestClient):
        response = client.get("/batches/")

        assert response.status_code == 200
        body = response.json()
        assert body["batches"] == []
        assert body["total"] == 0

    def test_returns_all_batches(self, client: TestClient):
        configs = [json.dumps({"run_type": "deterministic"})]

        client.post("/batches/", json={"label": "batch A", "configs": configs})
        client.post("/batches/", json={"label": "batch B", "configs": configs})

        response = client.get("/batches/")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        labels = {b["label"] for b in body["batches"]}
        assert labels == {"batch A", "batch B"}
