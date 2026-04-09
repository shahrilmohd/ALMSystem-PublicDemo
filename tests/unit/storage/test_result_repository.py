"""
Unit tests for ResultRepository (storage/result_repository.py).

Rules under test
----------------
ResultRepository — save_all():
  1.  All rows from a ResultStore are persisted.
  2.  result_count() matches store.result_count() after save_all().
  3.  Mismatched run_id raises ValueError.
  4.  Empty ResultStore saves without error; result_count = 0.
  5.  Asset fields are NULL for LIABILITY_ONLY results (None values preserved).
  6.  cohort_id=None is stored as NULL and round-trips correctly.
  7.  cohort_id="cohort_A" is stored as a string and round-trips correctly.

ResultRepository — get_dataframe():
  8.  Returns DataFrame with RESULT_COLUMNS columns.
  9.  Row count matches stored result count.
  10. Numerical values round-trip correctly (BEL, premiums, net_outgo).
  11. Rows sorted by (scenario_id, cohort_id, timestep).
  12. Returns empty DataFrame (correct schema) for unknown run_id.

ResultRepository — get_scenario():
  13. Returns only rows for the requested (scenario_id, cohort_id=None).
  14. Returns rows for a specific cohort_id.
  15. Returns empty DataFrame for unknown scenario_id.
  16. Rows sorted by timestep ascending.

ResultRepository — result_count():
  17. Returns 0 for unknown run_id.
  18. Returns correct count after save_all().

ResultRepository — multi-scenario:
  19. Results from multiple scenarios stored and retrieved independently.
  20. get_scenario() for one scenario does not return rows from another.
"""
from __future__ import annotations

import pytest

from engine.liability.base_liability import Decrements, LiabilityCashflows
from engine.results.result_store import RESULT_COLUMNS, ResultStore, TimestepResult
from storage.result_repository import ResultRepository
from storage.run_repository import RunRepository
from storage.models.run_record import RunRecord
from datetime import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_cashflows(timestep: int = 0, premiums: float = 1_000.0) -> LiabilityCashflows:
    return LiabilityCashflows(
        timestep=timestep,
        premiums=premiums,
        death_claims=500.0,
        surrender_payments=100.0,
        maturity_payments=0.0,
        expenses=50.0,
    )


def make_decrements(timestep: int = 0) -> Decrements:
    return Decrements(
        timestep=timestep,
        in_force_start=100.0,
        deaths=1.0,
        lapses=2.0,
        maturities=0.0,
        in_force_end=97.0,
    )


def make_result(
    run_id: str = "run_001",
    scenario_id: int = 0,
    timestep: int = 0,
    bel: float = 50_000.0,
    cohort_id: str | None = None,
    with_assets: bool = False,
) -> TimestepResult:
    return TimestepResult(
        run_id=run_id,
        scenario_id=scenario_id,
        timestep=timestep,
        cashflows=make_cashflows(timestep=timestep),
        decrements=make_decrements(timestep=timestep),
        bel=bel,
        reserve=bel,
        cohort_id=cohort_id,
        total_market_value=1_000_000.0 if with_assets else None,
        total_book_value=980_000.0    if with_assets else None,
        cash_balance=10_000.0         if with_assets else None,
        eir_income=500.0              if with_assets else None,
        coupon_income=400.0           if with_assets else None,
        dividend_income=100.0         if with_assets else None,
        unrealised_gl=20_000.0        if with_assets else None,
        realised_gl=0.0               if with_assets else None,
        oci_reserve=5_000.0           if with_assets else None,
        mv_ac=600_000.0               if with_assets else None,
        mv_fvtpl=400_000.0            if with_assets else None,
        mv_fvoci=0.0                  if with_assets else None,
    )


def make_store(run_id: str = "run_001", n_timesteps: int = 3,
               scenario_id: int = 0) -> ResultStore:
    store = ResultStore(run_id)
    for t in range(n_timesteps):
        store.store(make_result(run_id=run_id, scenario_id=scenario_id, timestep=t))
    return store


def _insert_run_record(session, run_id: str = "run_001") -> None:
    """Insert a minimal RunRecord so FK constraint is satisfied."""
    repo = RunRepository(session)
    repo.save(RunRecord(
        run_id=run_id,
        run_type="LIABILITY_ONLY",
        status="COMPLETED",
        created_at=datetime(2026, 1, 1),
        config_json="{}",
    ))
    session.commit()


# ---------------------------------------------------------------------------
# save_all()
# ---------------------------------------------------------------------------

class TestSaveAll:

    def test_all_rows_persisted(self, session):
        _insert_run_record(session)
        store = make_store(n_timesteps=5)
        repo = ResultRepository(session)
        repo.save_all("run_001", store)
        session.commit()
        assert repo.result_count("run_001") == 5

    def test_result_count_matches_store(self, session):
        _insert_run_record(session)
        store = make_store(n_timesteps=12)
        repo = ResultRepository(session)
        repo.save_all("run_001", store)
        session.commit()
        assert repo.result_count("run_001") == store.result_count()

    def test_mismatched_run_id_raises(self, session):
        store = ResultStore("run_A")
        store.store(make_result(run_id="run_A"))
        with pytest.raises(ValueError, match="run_id"):
            ResultRepository(session).save_all("run_B", store)

    def test_empty_store_saves_without_error(self, session):
        _insert_run_record(session)
        store = ResultStore("run_001")
        ResultRepository(session).save_all("run_001", store)
        session.commit()
        assert ResultRepository(session).result_count("run_001") == 0

    def test_asset_fields_null_for_liability_only(self, session):
        _insert_run_record(session)
        store = make_store(n_timesteps=1)   # make_result with_assets=False
        repo = ResultRepository(session)
        repo.save_all("run_001", store)
        session.commit()

        df = repo.get_dataframe("run_001")
        assert df.iloc[0]["total_market_value"] is None or \
               (df.iloc[0]["total_market_value"] != df.iloc[0]["total_market_value"])  # NaN check

    def test_asset_fields_populated_for_deterministic(self, session):
        _insert_run_record(session)
        store = ResultStore("run_001")
        store.store(make_result(with_assets=True))
        repo = ResultRepository(session)
        repo.save_all("run_001", store)
        session.commit()

        df = repo.get_dataframe("run_001")
        assert df.iloc[0]["total_market_value"] == pytest.approx(1_000_000.0)
        assert df.iloc[0]["mv_ac"] == pytest.approx(600_000.0)

    def test_cohort_id_none_stored_as_null(self, session):
        _insert_run_record(session)
        store = ResultStore("run_001")
        store.store(make_result(cohort_id=None))
        ResultRepository(session).save_all("run_001", store)
        session.commit()

        df = ResultRepository(session).get_dataframe("run_001")
        assert df.iloc[0]["cohort_id"] is None or \
               str(df.iloc[0]["cohort_id"]) in ("None", "nan")

    def test_cohort_id_string_round_trips(self, session):
        _insert_run_record(session)
        store = ResultStore("run_001")
        store.store(make_result(cohort_id="cohort_A"))
        ResultRepository(session).save_all("run_001", store)
        session.commit()

        df = ResultRepository(session).get_dataframe("run_001")
        assert df.iloc[0]["cohort_id"] == "cohort_A"


# ---------------------------------------------------------------------------
# get_dataframe()
# ---------------------------------------------------------------------------

class TestGetDataframe:

    def test_returns_result_columns_schema(self, session):
        _insert_run_record(session)
        store = make_store(n_timesteps=1)
        repo = ResultRepository(session)
        repo.save_all("run_001", store)
        session.commit()

        df = repo.get_dataframe("run_001")
        for col in RESULT_COLUMNS:
            assert col in df.columns

    def test_row_count_matches_stored_count(self, session):
        _insert_run_record(session)
        store = make_store(n_timesteps=6)
        repo = ResultRepository(session)
        repo.save_all("run_001", store)
        session.commit()

        assert len(repo.get_dataframe("run_001")) == 6

    def test_numerical_values_round_trip(self, session):
        _insert_run_record(session)
        store = ResultStore("run_001")
        store.store(make_result(bel=99_999.0, timestep=0))
        ResultRepository(session).save_all("run_001", store)
        session.commit()

        df = ResultRepository(session).get_dataframe("run_001")
        assert df.iloc[0]["bel"]      == pytest.approx(99_999.0)
        assert df.iloc[0]["premiums"] == pytest.approx(1_000.0)
        assert df.iloc[0]["net_outgo"]== pytest.approx(
            500.0 + 100.0 + 0.0 + 50.0 - 1_000.0   # death+surr+mat+exp-prem
        )

    def test_rows_sorted_by_scenario_then_timestep(self, session):
        _insert_run_record(session)
        store = ResultStore("run_001")
        # Store out of natural order
        store.store(make_result(scenario_id=2, timestep=1))
        store.store(make_result(scenario_id=1, timestep=0))
        store.store(make_result(scenario_id=2, timestep=0))
        store.store(make_result(scenario_id=1, timestep=1))
        ResultRepository(session).save_all("run_001", store)
        session.commit()

        df = ResultRepository(session).get_dataframe("run_001")
        assert list(df["scenario_id"]) == [1, 1, 2, 2]
        assert list(df["timestep"])    == [0, 1, 0, 1]

    def test_empty_dataframe_for_unknown_run(self, session):
        df = ResultRepository(session).get_dataframe("ghost_run")
        assert len(df) == 0
        assert list(df.columns) == list(RESULT_COLUMNS)


# ---------------------------------------------------------------------------
# get_scenario()
# ---------------------------------------------------------------------------

class TestGetScenario:

    def test_returns_only_requested_scenario(self, session):
        _insert_run_record(session)
        store = ResultStore("run_001")
        for t in range(3):
            store.store(make_result(scenario_id=1, timestep=t))
            store.store(make_result(scenario_id=2, timestep=t))
        repo = ResultRepository(session)
        repo.save_all("run_001", store)
        session.commit()

        df = repo.get_scenario("run_001", scenario_id=1)
        assert len(df) == 3
        assert (df["scenario_id"] == 1).all()

    def test_returns_specific_cohort(self, session):
        _insert_run_record(session)
        store = ResultStore("run_001")
        store.store(make_result(scenario_id=1, timestep=0, cohort_id="cohort_A"))
        store.store(make_result(scenario_id=1, timestep=0, cohort_id="cohort_B"))
        repo = ResultRepository(session)
        repo.save_all("run_001", store)
        session.commit()

        df = repo.get_scenario("run_001", scenario_id=1, cohort_id="cohort_A")
        assert len(df) == 1
        assert df.iloc[0]["cohort_id"] == "cohort_A"

    def test_empty_dataframe_for_unknown_scenario(self, session):
        _insert_run_record(session)
        store = make_store(n_timesteps=1, scenario_id=1)
        ResultRepository(session).save_all("run_001", store)
        session.commit()

        df = ResultRepository(session).get_scenario("run_001", scenario_id=99)
        assert len(df) == 0
        assert list(df.columns) == list(RESULT_COLUMNS)

    def test_rows_sorted_by_timestep(self, session):
        _insert_run_record(session)
        store = ResultStore("run_001")
        for t in [2, 0, 1]:
            store.store(make_result(scenario_id=1, timestep=t))
        ResultRepository(session).save_all("run_001", store)
        session.commit()

        df = ResultRepository(session).get_scenario("run_001", scenario_id=1)
        assert list(df["timestep"]) == [0, 1, 2]


# ---------------------------------------------------------------------------
# result_count()
# ---------------------------------------------------------------------------

class TestResultCount:

    def test_zero_for_unknown_run(self, session):
        assert ResultRepository(session).result_count("ghost") == 0

    def test_correct_count_after_save(self, session):
        _insert_run_record(session)
        store = make_store(n_timesteps=10)
        repo = ResultRepository(session)
        repo.save_all("run_001", store)
        session.commit()
        assert repo.result_count("run_001") == 10


# ---------------------------------------------------------------------------
# Multi-scenario isolation
# ---------------------------------------------------------------------------

class TestMultiScenario:

    def test_results_stored_for_all_scenarios(self, session):
        _insert_run_record(session)
        store = ResultStore("run_001")
        for scen in range(1, 4):
            for t in range(3):
                store.store(make_result(scenario_id=scen, timestep=t))
        repo = ResultRepository(session)
        repo.save_all("run_001", store)
        session.commit()

        assert repo.result_count("run_001") == 9

    def test_get_scenario_does_not_bleed_across_scenarios(self, session):
        _insert_run_record(session)
        store = ResultStore("run_001")
        for scen in range(1, 3):
            for t in range(3):
                store.store(make_result(scenario_id=scen, timestep=t,
                                        bel=float(scen * 10_000)))
        repo = ResultRepository(session)
        repo.save_all("run_001", store)
        session.commit()

        df1 = repo.get_scenario("run_001", scenario_id=1)
        df2 = repo.get_scenario("run_001", scenario_id=2)
        assert (df1["bel"] - 10_000.0).abs().max() < 1e-6
        assert (df2["bel"] - 20_000.0).abs().max() < 1e-6
