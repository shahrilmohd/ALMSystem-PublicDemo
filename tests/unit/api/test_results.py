"""
Unit tests for the /results router.

Coverage
--------
GET /results/{run_id}
    - run exists, no results yet → 200 with empty rows list
    - run exists, with results → 200 with correct rows
    - unknown run_id → 404
    - ?scenario_id filter → returns only that scenario's rows
    - ?format=csv → CSV download response

GET /results/{run_id}/summary
    - run exists, no results → 200 with zeroed counts
    - run exists, with results → correct n_scenarios, n_timesteps, final_bel
    - unknown run_id → 404
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from storage.models.result_record import ResultRecord
from tests.unit.api.conftest import make_run_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insert_result_rows(db: Session, run_id: str, rows: list[dict]) -> None:
    """Bulk-insert result rows directly into the test DB."""
    for row in rows:
        record = ResultRecord(
            run_id=run_id,
            scenario_id=row.get("scenario_id", 0),
            timestep=row.get("timestep", 0),
            cohort_id=row.get("cohort_id"),
            premiums=row.get("premiums", 0.0),
            death_claims=row.get("death_claims", 0.0),
            surrender_payments=row.get("surrender_payments", 0.0),
            maturity_payments=row.get("maturity_payments", 0.0),
            expenses=row.get("expenses", 0.0),
            net_outgo=row.get("net_outgo", 0.0),
            in_force_start=row.get("in_force_start", 100.0),
            deaths=row.get("deaths", 1.0),
            lapses=row.get("lapses", 2.0),
            maturities=row.get("maturities", 0.0),
            in_force_end=row.get("in_force_end", 97.0),
            bel=row.get("bel", 1000.0),
            reserve=row.get("reserve", 1050.0),
        )
        db.add(record)
    db.commit()


# ---------------------------------------------------------------------------
# GET /results/{run_id}
# ---------------------------------------------------------------------------

class TestGetResults:
    def test_unknown_run_returns_404(self, client: TestClient):
        response = client.get("/results/no-such-run")

        assert response.status_code == 404

    def test_run_with_no_results_returns_empty_list(
        self, client: TestClient, db_session: Session
    ):
        make_run_record(db_session, run_id="run-empty")

        response = client.get("/results/run-empty")

        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == "run-empty"
        assert body["total_rows"] == 0
        assert body["rows"] == []

    def test_returns_all_rows_for_run(
        self, client: TestClient, db_session: Session
    ):
        make_run_record(db_session, run_id="run-rows")
        _insert_result_rows(db_session, "run-rows", [
            {"scenario_id": 0, "timestep": 0, "bel": 1000.0},
            {"scenario_id": 0, "timestep": 1, "bel": 950.0},
            {"scenario_id": 0, "timestep": 2, "bel": 900.0},
        ])

        response = client.get("/results/run-rows")

        assert response.status_code == 200
        body = response.json()
        assert body["total_rows"] == 3
        assert len(body["rows"]) == 3

    def test_scenario_filter_returns_only_that_scenario(
        self, client: TestClient, db_session: Session
    ):
        make_run_record(db_session, run_id="run-multi")
        _insert_result_rows(db_session, "run-multi", [
            {"scenario_id": 0, "timestep": 0, "bel": 1000.0},
            {"scenario_id": 1, "timestep": 0, "bel": 1100.0},
            {"scenario_id": 2, "timestep": 0, "bel": 900.0},
        ])

        response = client.get("/results/run-multi?scenario_id=1")

        assert response.status_code == 200
        rows = response.json()["rows"]
        assert all(r["scenario_id"] == 1 for r in rows)
        assert len(rows) == 1

    def test_csv_format_returns_streaming_response(
        self, client: TestClient, db_session: Session
    ):
        make_run_record(db_session, run_id="run-csv")
        _insert_result_rows(db_session, "run-csv", [
            {"scenario_id": 0, "timestep": 0, "bel": 500.0},
        ])

        response = client.get("/results/run-csv?format=csv")

        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]
        # Response body should be CSV text with a header row
        text = response.text
        assert "run_id" in text
        assert "bel" in text

    def test_row_fields_match_result_schema(
        self, client: TestClient, db_session: Session
    ):
        make_run_record(db_session, run_id="run-schema")
        _insert_result_rows(db_session, "run-schema", [
            {"scenario_id": 0, "timestep": 0, "bel": 800.0},
        ])

        row = client.get("/results/run-schema").json()["rows"][0]

        required_keys = {
            "run_id", "scenario_id", "timestep", "cohort_id",
            "premiums", "death_claims", "bel", "reserve",
            "in_force_start", "in_force_end",
        }
        assert required_keys.issubset(row.keys())


# ---------------------------------------------------------------------------
# GET /results/{run_id}/summary
# ---------------------------------------------------------------------------

class TestGetResultsSummary:
    def test_unknown_run_returns_404(self, client: TestClient):
        response = client.get("/results/no-such-run/summary")

        assert response.status_code == 404

    def test_run_with_no_results_returns_zeroed_summary(
        self, client: TestClient, db_session: Session
    ):
        make_run_record(db_session, run_id="run-empty-sum")

        response = client.get("/results/run-empty-sum/summary")

        assert response.status_code == 200
        body = response.json()
        assert body["run_id"] == "run-empty-sum"
        assert body["n_result_rows"] == 0
        assert body["n_scenarios"] == 0
        assert body["n_timesteps"] == 0
        assert body["final_bel"] is None

    def test_correct_scenario_and_timestep_counts(
        self, client: TestClient, db_session: Session
    ):
        make_run_record(db_session, run_id="run-counts")
        _insert_result_rows(db_session, "run-counts", [
            {"scenario_id": 0, "timestep": 0, "bel": 1000.0},
            {"scenario_id": 0, "timestep": 1, "bel": 950.0},
            {"scenario_id": 1, "timestep": 0, "bel": 1050.0},
            {"scenario_id": 1, "timestep": 1, "bel": 990.0},
        ])

        body = client.get("/results/run-counts/summary").json()

        assert body["n_result_rows"] == 4
        assert body["n_scenarios"] == 2
        assert body["n_timesteps"] == 2

    def test_final_bel_is_mean_across_scenarios_at_last_timestep(
        self, client: TestClient, db_session: Session
    ):
        make_run_record(db_session, run_id="run-bel")
        _insert_result_rows(db_session, "run-bel", [
            {"scenario_id": 0, "timestep": 0, "bel": 1000.0},
            {"scenario_id": 0, "timestep": 1, "bel": 800.0},   # final ts
            {"scenario_id": 1, "timestep": 0, "bel": 1100.0},
            {"scenario_id": 1, "timestep": 1, "bel": 900.0},   # final ts
        ])

        body = client.get("/results/run-bel/summary").json()

        # Mean of 800 and 900 at timestep=1
        assert abs(body["final_bel"] - 850.0) < 0.01

    def test_summary_response_has_all_schema_fields(
        self, client: TestClient, db_session: Session
    ):
        make_run_record(db_session, run_id="run-fields")

        body = client.get("/results/run-fields/summary").json()

        expected_keys = {
            "run_id", "n_result_rows", "n_scenarios",
            "n_timesteps", "final_bel", "final_total_market_value",
        }
        assert expected_keys.issubset(body.keys())
