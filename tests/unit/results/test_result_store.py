"""
Unit tests for ResultStore and TimestepResult (engine/results/result_store.py).

Rules under test
----------------
TimestepResult:
  1.  All fields stored and accessible.
  2.  Dataclass equality: two instances with identical fields are equal.
  3.  Dataclass inequality: instances with any differing field are not equal.

ResultStore — construction:
  4.  run_id property returns the value passed to __init__.
  5.  New store has zero results (result_count = 0).
  6.  New store has zero scenarios (scenario_count = 0).

ResultStore — store():
  7.  Single result stored; result_count increments to 1.
  8.  Duplicate (scenario_id, timestep) raises ValueError.
  9.  Wrong run_id raises ValueError containing both IDs.
  10. Multiple results with different keys all accepted.
  11. Same scenario, different timestep: accepted.
  12. Same timestep, different scenario: accepted.

ResultStore — get():
  13. Returns the correct result for a known key.
  14. Unknown (scenario_id, timestep) raises KeyError.
  15. After storing multiple results, each is retrieved independently.

ResultStore — all_timesteps():
  16. Returns results sorted by timestep ascending.
  17. Returns only results belonging to the requested scenario.
  18. Returns empty list for an unknown scenario_id.

ResultStore — all_scenarios():
  19. Returns sorted list of unique scenario IDs.
  20. Returns empty list when the store is empty.

ResultStore — counts:
  21. result_count reflects total stored results.
  22. scenario_count reflects distinct scenarios.
  23. timestep_count reflects timesteps for a specific scenario.
  24. timestep_count returns 0 for an unknown scenario.

ResultStore — as_dataframe():
  25. Empty store returns empty DataFrame with RESULT_COLUMNS as columns.
  26. Single result: DataFrame has exactly 1 row.
  27. Column names match RESULT_COLUMNS exactly.
  28. Numerical values match the stored result.
  29. net_outgo column equals cashflows.net_outgo (derived property).
  30. Multi-scenario: one row per (scenario_id, timestep) combination.
  31. Rows are sorted by (scenario_id, timestep) ascending.

ResultStore — summary():
  32. Empty store returns zeroed summary with correct keys.
  33. Single result: totals and mean equal the single result's values.
  34. Multi-result: total_premiums is the sum across all results.
  35. Multi-result: mean_bel = total_bel / result_count.
  36. scenario_count and result_count are correct.

cohort_id (DECISIONS.md §17 — BPA contract group granularity):
  37. TimestepResult.cohort_id defaults to None.
  38. TimestepResult can be constructed with a non-None cohort_id.
  39. Same (scenario_id, timestep) with different cohort_ids accepted as separate entries.
  40. Duplicate (scenario_id, timestep, cohort_id=None) still raises ValueError.
  41. Duplicate (scenario_id, timestep, cohort_id="A") raises ValueError.
  42. get() with cohort_id retrieves the correct BPA result.
  43. get() with cohort_id=None still works for conventional runs.
  44. get() with wrong cohort_id raises KeyError.
  45. all_timesteps(scenario_id) returns only cohort_id=None results.
  46. all_timesteps(scenario_id, cohort_id) returns only that cohort's results.
  47. as_dataframe() includes cohort_id column; None for conventional, str for BPA.
"""
from __future__ import annotations

import pytest

from engine.liability.base_liability import Decrements, LiabilityCashflows
from engine.results.result_store import (
    RESULT_COLUMNS,
    ResultStore,
    TimestepResult,
)


# ---------------------------------------------------------------------------
# Helpers — build lightweight test objects
# ---------------------------------------------------------------------------

def make_cashflows(
    timestep: int = 0,
    premiums: float = 1_000.0,
    death_claims: float = 5_000.0,
    surrender_payments: float = 500.0,
    maturity_payments: float = 0.0,
    expenses: float = 100.0,
) -> LiabilityCashflows:
    return LiabilityCashflows(
        timestep=timestep,
        premiums=premiums,
        death_claims=death_claims,
        surrender_payments=surrender_payments,
        maturity_payments=maturity_payments,
        expenses=expenses,
    )


def make_decrements(
    timestep: int = 0,
    in_force_start: float = 100.0,
    deaths: float = 1.0,
    lapses: float = 2.0,
    maturities: float = 0.0,
    in_force_end: float = 97.0,
) -> Decrements:
    return Decrements(
        timestep=timestep,
        in_force_start=in_force_start,
        deaths=deaths,
        lapses=lapses,
        maturities=maturities,
        in_force_end=in_force_end,
    )


def make_result(
    run_id: str = "run_001",
    scenario_id: int = 0,
    timestep: int = 0,
    bel: float = 50_000.0,
    reserve: float = 50_000.0,
) -> TimestepResult:
    return TimestepResult(
        run_id=run_id,
        scenario_id=scenario_id,
        timestep=timestep,
        cashflows=make_cashflows(timestep=timestep),
        decrements=make_decrements(timestep=timestep),
        bel=bel,
        reserve=reserve,
    )


# ---------------------------------------------------------------------------
# TimestepResult
# ---------------------------------------------------------------------------

class TestTimestepResult:
    def test_all_fields_stored(self):
        cf = make_cashflows()
        d  = make_decrements()
        r  = TimestepResult(
            run_id="run_001", scenario_id=2, timestep=5,
            cashflows=cf, decrements=d, bel=12_000.0, reserve=12_500.0,
        )
        assert r.run_id      == "run_001"
        assert r.scenario_id == 2
        assert r.timestep    == 5
        assert r.cashflows   is cf
        assert r.decrements  is d
        assert r.bel         == pytest.approx(12_000.0)
        assert r.reserve     == pytest.approx(12_500.0)

    def test_dataclass_equality(self):
        r1 = make_result(run_id="r", scenario_id=0, timestep=0, bel=1.0)
        r2 = make_result(run_id="r", scenario_id=0, timestep=0, bel=1.0)
        assert r1 == r2

    def test_dataclass_inequality_different_bel(self):
        r1 = make_result(bel=1.0)
        r2 = make_result(bel=2.0)
        assert r1 != r2

    def test_dataclass_inequality_different_scenario(self):
        r1 = make_result(scenario_id=0)
        r2 = make_result(scenario_id=1)
        assert r1 != r2


# ---------------------------------------------------------------------------
# ResultStore — construction
# ---------------------------------------------------------------------------

class TestResultStoreConstruction:
    def test_run_id_property(self):
        store = ResultStore(run_id="run_abc")
        assert store.run_id == "run_abc"

    def test_new_store_has_zero_results(self):
        assert ResultStore("r").result_count() == 0

    def test_new_store_has_zero_scenarios(self):
        assert ResultStore("r").scenario_count() == 0


# ---------------------------------------------------------------------------
# ResultStore — store()
# ---------------------------------------------------------------------------

class TestResultStoreWrite:
    def test_single_result_stored(self):
        store = ResultStore("run_001")
        store.store(make_result())
        assert store.result_count() == 1

    def test_duplicate_key_raises_value_error(self):
        store = ResultStore("run_001")
        store.store(make_result(scenario_id=0, timestep=0))
        with pytest.raises(ValueError, match="already been stored"):
            store.store(make_result(scenario_id=0, timestep=0))

    def test_wrong_run_id_raises_value_error(self):
        store = ResultStore("run_001")
        with pytest.raises(ValueError) as exc_info:
            store.store(make_result(run_id="run_999"))
        assert "run_999" in str(exc_info.value)
        assert "run_001" in str(exc_info.value)

    def test_multiple_different_keys_accepted(self):
        store = ResultStore("run_001")
        store.store(make_result(scenario_id=0, timestep=0))
        store.store(make_result(scenario_id=0, timestep=1))
        store.store(make_result(scenario_id=1, timestep=0))
        assert store.result_count() == 3

    def test_same_scenario_different_timestep_accepted(self):
        store = ResultStore("run_001")
        store.store(make_result(scenario_id=0, timestep=0))
        store.store(make_result(scenario_id=0, timestep=1))
        assert store.result_count() == 2

    def test_same_timestep_different_scenario_accepted(self):
        store = ResultStore("run_001")
        store.store(make_result(scenario_id=0, timestep=0))
        store.store(make_result(scenario_id=1, timestep=0))
        assert store.result_count() == 2


# ---------------------------------------------------------------------------
# ResultStore — get()
# ---------------------------------------------------------------------------

class TestResultStoreGet:
    def test_returns_correct_result(self):
        store = ResultStore("run_001")
        r = make_result(scenario_id=0, timestep=3, bel=99_000.0)
        store.store(r)
        retrieved = store.get(scenario_id=0, timestep=3)
        assert retrieved == r

    def test_unknown_key_raises_key_error(self):
        store = ResultStore("run_001")
        with pytest.raises(KeyError):
            store.get(scenario_id=0, timestep=99)

    def test_multiple_results_retrieved_independently(self):
        store = ResultStore("run_001")
        r0 = make_result(scenario_id=0, timestep=0, bel=1.0)
        r1 = make_result(scenario_id=0, timestep=1, bel=2.0)
        r2 = make_result(scenario_id=1, timestep=0, bel=3.0)
        for r in [r0, r1, r2]:
            store.store(r)
        assert store.get(0, 0).bel == pytest.approx(1.0)
        assert store.get(0, 1).bel == pytest.approx(2.0)
        assert store.get(1, 0).bel == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# ResultStore — all_timesteps()
# ---------------------------------------------------------------------------

class TestAllTimesteps:
    def test_returns_timesteps_in_ascending_order(self):
        store = ResultStore("run_001")
        # Store out of order intentionally
        store.store(make_result(scenario_id=0, timestep=2))
        store.store(make_result(scenario_id=0, timestep=0))
        store.store(make_result(scenario_id=0, timestep=1))
        ts = [r.timestep for r in store.all_timesteps(scenario_id=0)]
        assert ts == [0, 1, 2]

    def test_returns_only_requested_scenario(self):
        store = ResultStore("run_001")
        store.store(make_result(scenario_id=0, timestep=0))
        store.store(make_result(scenario_id=0, timestep=1))
        store.store(make_result(scenario_id=1, timestep=0))
        results = store.all_timesteps(scenario_id=0)
        assert all(r.scenario_id == 0 for r in results)
        assert len(results) == 2

    def test_unknown_scenario_returns_empty_list(self):
        store = ResultStore("run_001")
        store.store(make_result(scenario_id=0, timestep=0))
        assert store.all_timesteps(scenario_id=99) == []


# ---------------------------------------------------------------------------
# ResultStore — all_scenarios()
# ---------------------------------------------------------------------------

class TestAllScenarios:
    def test_returns_sorted_unique_scenario_ids(self):
        store = ResultStore("run_001")
        store.store(make_result(scenario_id=3, timestep=0))
        store.store(make_result(scenario_id=1, timestep=0))
        store.store(make_result(scenario_id=3, timestep=1))
        store.store(make_result(scenario_id=2, timestep=0))
        assert store.all_scenarios() == [1, 2, 3]

    def test_empty_store_returns_empty_list(self):
        assert ResultStore("run_001").all_scenarios() == []


# ---------------------------------------------------------------------------
# ResultStore — counts
# ---------------------------------------------------------------------------

class TestCounts:
    def test_result_count(self):
        store = ResultStore("run_001")
        for t in range(5):
            store.store(make_result(scenario_id=0, timestep=t))
        assert store.result_count() == 5

    def test_scenario_count(self):
        store = ResultStore("run_001")
        for s in range(3):
            store.store(make_result(scenario_id=s, timestep=0))
        assert store.scenario_count() == 3

    def test_timestep_count_for_scenario(self):
        store = ResultStore("run_001")
        for t in range(4):
            store.store(make_result(scenario_id=0, timestep=t))
        store.store(make_result(scenario_id=1, timestep=0))
        assert store.timestep_count(scenario_id=0) == 4
        assert store.timestep_count(scenario_id=1) == 1

    def test_timestep_count_unknown_scenario_returns_zero(self):
        store = ResultStore("run_001")
        assert store.timestep_count(scenario_id=99) == 0


# ---------------------------------------------------------------------------
# ResultStore — as_dataframe()
# ---------------------------------------------------------------------------

class TestAsDataframe:
    def test_empty_store_returns_empty_dataframe_with_correct_columns(self):
        df = ResultStore("run_001").as_dataframe()
        assert len(df) == 0
        assert list(df.columns) == list(RESULT_COLUMNS)

    def test_single_result_one_row(self):
        store = ResultStore("run_001")
        store.store(make_result())
        df = store.as_dataframe()
        assert len(df) == 1

    def test_column_names_match_result_columns_constant(self):
        store = ResultStore("run_001")
        store.store(make_result())
        assert list(store.as_dataframe().columns) == list(RESULT_COLUMNS)

    def test_numerical_values_match_stored_result(self):
        cf = make_cashflows(premiums=2_000.0, death_claims=8_000.0, expenses=200.0)
        d  = make_decrements(in_force_start=150.0, deaths=3.0, lapses=5.0)
        r  = TimestepResult(
            run_id="run_001", scenario_id=0, timestep=0,
            cashflows=cf, decrements=d, bel=75_000.0, reserve=75_000.0,
        )
        store = ResultStore("run_001")
        store.store(r)
        row = store.as_dataframe().iloc[0]

        assert row["premiums"]       == pytest.approx(2_000.0)
        assert row["death_claims"]   == pytest.approx(8_000.0)
        assert row["expenses"]       == pytest.approx(200.0)
        assert row["in_force_start"] == pytest.approx(150.0)
        assert row["deaths"]         == pytest.approx(3.0)
        assert row["lapses"]         == pytest.approx(5.0)
        assert row["bel"]            == pytest.approx(75_000.0)
        assert row["reserve"]        == pytest.approx(75_000.0)

    def test_net_outgo_column_is_derived_from_cashflows(self):
        """
        net_outgo = death_claims + surrender_payments + maturity_payments
                    + expenses - premiums
        """
        cf = make_cashflows(
            premiums=1_000.0, death_claims=5_000.0,
            surrender_payments=500.0, maturity_payments=0.0, expenses=100.0,
        )
        r = TimestepResult(
            run_id="r", scenario_id=0, timestep=0,
            cashflows=cf, decrements=make_decrements(),
            bel=0.0, reserve=0.0,
        )
        store = ResultStore("r")
        store.store(r)
        row = store.as_dataframe().iloc[0]
        expected = cf.net_outgo   # 5_000 + 500 + 0 + 100 - 1_000 = 4_600
        assert row["net_outgo"] == pytest.approx(expected)

    def test_multi_scenario_correct_row_count(self):
        store = ResultStore("run_001")
        for s in range(3):
            for t in range(4):
                store.store(make_result(scenario_id=s, timestep=t))
        assert len(store.as_dataframe()) == 12

    def test_rows_sorted_by_scenario_then_timestep(self):
        store = ResultStore("run_001")
        # Store in reverse order
        store.store(make_result(scenario_id=1, timestep=2))
        store.store(make_result(scenario_id=0, timestep=1))
        store.store(make_result(scenario_id=1, timestep=0))
        store.store(make_result(scenario_id=0, timestep=0))
        df = store.as_dataframe()
        assert list(df["scenario_id"]) == [0, 0, 1, 1]
        assert list(df["timestep"])    == [0, 1, 0, 2]


# ---------------------------------------------------------------------------
# ResultStore — summary()
# ---------------------------------------------------------------------------

class TestSummary:
    def test_empty_store_summary_has_correct_keys(self):
        s = ResultStore("run_001").summary()
        assert s["run_id"]          == "run_001"
        assert s["scenario_count"]  == 0
        assert s["result_count"]    == 0
        assert s["total_premiums"]  == pytest.approx(0.0)
        assert s["total_net_outgo"] == pytest.approx(0.0)
        assert s["total_bel"]       == pytest.approx(0.0)
        assert s["mean_bel"]        == pytest.approx(0.0)

    def test_single_result_summary(self):
        store = ResultStore("run_001")
        store.store(make_result(bel=50_000.0))
        s = store.summary()
        assert s["scenario_count"] == 1
        assert s["result_count"]   == 1
        assert s["total_bel"]      == pytest.approx(50_000.0)
        assert s["mean_bel"]       == pytest.approx(50_000.0)

    def test_multi_result_total_premiums(self):
        """Total premiums sum across all (scenario, timestep) results."""
        store = ResultStore("run_001")
        # Each result has premiums = 1,000 (from make_cashflows default)
        for t in range(3):
            store.store(make_result(scenario_id=0, timestep=t))
        s = store.summary()
        assert s["total_premiums"] == pytest.approx(3_000.0)

    def test_multi_result_mean_bel(self):
        """mean_bel = total_bel / result_count."""
        store = ResultStore("run_001")
        store.store(make_result(scenario_id=0, timestep=0, bel=10_000.0))
        store.store(make_result(scenario_id=0, timestep=1, bel=20_000.0))
        store.store(make_result(scenario_id=1, timestep=0, bel=30_000.0))
        s = store.summary()
        assert s["total_bel"]  == pytest.approx(60_000.0)
        assert s["mean_bel"]   == pytest.approx(20_000.0)
        assert s["result_count"]  == 3
        assert s["scenario_count"] == 2


# ---------------------------------------------------------------------------
# cohort_id — BPA contract group granularity (DECISIONS.md §17)
# ---------------------------------------------------------------------------

def make_bpa_result(
    run_id: str = "run_bpa",
    scenario_id: int = 1,
    timestep: int = 0,
    cohort_id: str | None = "cohort_A",
    bel: float = 100_000.0,
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
    )


class TestCohortId:

    def test_cohort_id_defaults_to_none(self):
        r = make_result()
        assert r.cohort_id is None

    def test_cohort_id_stored_when_set(self):
        r = make_bpa_result(cohort_id="cohort_B")
        assert r.cohort_id == "cohort_B"

    def test_same_scenario_timestep_different_cohorts_accepted(self):
        """(scenario=1, t=0, cohort_A) and (scenario=1, t=0, cohort_B) are distinct keys."""
        store = ResultStore("run_bpa")
        store.store(make_bpa_result(cohort_id="cohort_A"))
        store.store(make_bpa_result(cohort_id="cohort_B"))
        assert store.result_count() == 2

    def test_duplicate_conventional_key_still_raises(self):
        """Duplicate (scenario_id, timestep, cohort_id=None) still raises ValueError."""
        store = ResultStore("run_001")
        store.store(make_result(scenario_id=0, timestep=0))
        with pytest.raises(ValueError, match="already been stored"):
            store.store(make_result(scenario_id=0, timestep=0))

    def test_duplicate_bpa_key_raises(self):
        """Duplicate (scenario_id, timestep, cohort_id='A') raises ValueError."""
        store = ResultStore("run_bpa")
        store.store(make_bpa_result(cohort_id="cohort_A"))
        with pytest.raises(ValueError, match="already been stored"):
            store.store(make_bpa_result(cohort_id="cohort_A"))

    def test_get_with_cohort_id(self):
        store = ResultStore("run_bpa")
        ra = make_bpa_result(cohort_id="cohort_A", bel=100_000.0)
        rb = make_bpa_result(cohort_id="cohort_B", bel=200_000.0)
        store.store(ra)
        store.store(rb)
        assert store.get(1, 0, "cohort_A").bel == pytest.approx(100_000.0)
        assert store.get(1, 0, "cohort_B").bel == pytest.approx(200_000.0)

    def test_get_without_cohort_id_retrieves_none_cohort(self):
        """get(scenario, timestep) finds result where cohort_id=None."""
        store = ResultStore("run_001")
        r = make_result(scenario_id=1, timestep=0, bel=55_000.0)
        store.store(r)
        assert store.get(1, 0).bel == pytest.approx(55_000.0)

    def test_get_wrong_cohort_raises_key_error(self):
        store = ResultStore("run_bpa")
        store.store(make_bpa_result(cohort_id="cohort_A"))
        with pytest.raises(KeyError):
            store.get(1, 0, "cohort_MISSING")

    def test_all_timesteps_no_cohort_returns_only_none_cohort(self):
        """all_timesteps(scenario_id) without cohort_id returns only cohort_id=None results."""
        store = ResultStore("run_bpa")
        store.store(make_bpa_result(cohort_id="cohort_A", timestep=0))
        store.store(make_bpa_result(cohort_id="cohort_A", timestep=1))
        # Conventional result with cohort_id=None
        store.store(make_result(scenario_id=1, timestep=0, run_id="run_bpa"))
        results = store.all_timesteps(scenario_id=1)
        assert len(results) == 1
        assert results[0].cohort_id is None

    def test_all_timesteps_with_cohort_id(self):
        store = ResultStore("run_bpa")
        store.store(make_bpa_result(cohort_id="cohort_A", timestep=1, bel=10.0))
        store.store(make_bpa_result(cohort_id="cohort_A", timestep=0, bel=20.0))
        store.store(make_bpa_result(cohort_id="cohort_B", timestep=0, bel=30.0))
        results = store.all_timesteps(scenario_id=1, cohort_id="cohort_A")
        assert len(results) == 2
        assert [r.timestep for r in results] == [0, 1]
        assert all(r.cohort_id == "cohort_A" for r in results)

    def test_as_dataframe_includes_cohort_id_column(self):
        """cohort_id column present; NaN for conventional (None→NaN in pandas), str for BPA."""
        import pandas as _pd

        store = ResultStore("run_mix")
        store.store(make_result(scenario_id=0, timestep=0, run_id="run_mix"))
        store.store(make_bpa_result(run_id="run_mix", scenario_id=1, timestep=0,
                                     cohort_id="cohort_A"))
        df = store.as_dataframe()
        assert "cohort_id" in df.columns
        conventional_row = df[df["scenario_id"] == 0].iloc[0]
        bpa_row          = df[df["scenario_id"] == 1].iloc[0]
        # pandas converts None → NaN in mixed-type object columns
        assert _pd.isna(conventional_row["cohort_id"])
        assert bpa_row["cohort_id"] == "cohort_A"
