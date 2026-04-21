"""
Unit tests for TvogCalculator.

Numerical anchors
-----------------
All expected values are computed by hand or by simple arithmetic and recorded
here before the tests are written, per project testing convention.

Anchor 1 — zero TVOG (flat ESG, no option value):
  3 scenarios, BEL₀ all = 1_000.0, deterministic BEL₀ = 1_000.0
  mean_stochastic_bel = 1_000.0
  tvog                = 0.0

Anchor 2 — positive TVOG:
  4 scenarios, BEL₀ = [1_000, 1_100, 1_200, 1_300], deterministic = 1_000.0
  mean_stochastic_bel = 1_150.0
  tvog                = 150.0

Anchor 3 — negative TVOG (stochastic mean < deterministic):
  4 scenarios, BEL₀ = [700, 800, 900, 1_000], deterministic = 1_000.0
  mean_stochastic_bel = 850.0
  tvog                = −150.0

Anchor 4 — single scenario:
  1 scenario, BEL₀ = 500.0, deterministic = 400.0
  tvog = 100.0

Anchor 5 — percentile correctness:
  5 scenarios, BEL₀ = [100, 200, 300, 400, 500]
  50th pct (median) = 300.0  (numpy linear interpolation: value at index 2)
  0th  pct          = 100.0
  100th pct         = 500.0
"""
from __future__ import annotations

import pytest

from engine.liability.base_liability import Decrements, LiabilityCashflows
from engine.results.result_store import ResultStore, TimestepResult
from engine.results.tvog_calculator import TvogCalculator, TvogResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_cashflows(t: int = 0) -> LiabilityCashflows:
    """Minimal cashflows — all zero, only timestep varies."""
    return LiabilityCashflows(
        timestep=t,
        premiums=0.0,
        death_claims=0.0,
        surrender_payments=0.0,
        maturity_payments=0.0,
        expenses=0.0,
    )


def _null_decrements(t: int = 0) -> Decrements:
    """Minimal decrements — all zero."""
    return Decrements(
        timestep=t,
        in_force_start=0.0,
        deaths=0.0,
        lapses=0.0,
        maturities=0.0,
        in_force_end=0.0,
    )


def _make_store(scenario_bels: dict[int, float], run_id: str = "run_test") -> ResultStore:
    """
    Build a ResultStore with one result per scenario at timestep 0 only.

    Parameters
    ----------
    scenario_bels : dict[int, float]
        Mapping of scenario_id → BEL value at t=0.
    """
    store = ResultStore(run_id=run_id)
    for sid, bel in scenario_bels.items():
        store.store(TimestepResult(
            run_id=run_id,
            scenario_id=sid,
            timestep=0,
            cashflows=_null_cashflows(0),
            decrements=_null_decrements(0),
            bel=bel,
            reserve=bel,
        ))
    return store


# ---------------------------------------------------------------------------
# Anchor 1 — zero TVOG
# ---------------------------------------------------------------------------

class TestZeroTvog:
    """Flat ESG paths produce zero TVOG."""

    def test_tvog_is_zero(self):
        store = _make_store({1: 1_000.0, 2: 1_000.0, 3: 1_000.0})
        result = TvogCalculator(store, deterministic_bel=1_000.0).calculate()
        assert result.tvog == pytest.approx(0.0)

    def test_mean_equals_deterministic(self):
        store = _make_store({1: 1_000.0, 2: 1_000.0, 3: 1_000.0})
        result = TvogCalculator(store, deterministic_bel=1_000.0).calculate()
        assert result.mean_stochastic_bel == pytest.approx(1_000.0)
        assert result.deterministic_bel == pytest.approx(1_000.0)

    def test_n_scenarios(self):
        store = _make_store({1: 1_000.0, 2: 1_000.0, 3: 1_000.0})
        result = TvogCalculator(store, deterministic_bel=1_000.0).calculate()
        assert result.n_scenarios == 3


# ---------------------------------------------------------------------------
# Anchor 2 — positive TVOG
# ---------------------------------------------------------------------------

class TestPositiveTvog:
    """Stochastic mean above deterministic BEL."""

    def test_tvog_value(self):
        # BELs: 1000, 1100, 1200, 1300  → mean = 1150, deterministic = 1000
        store = _make_store({1: 1_000.0, 2: 1_100.0, 3: 1_200.0, 4: 1_300.0})
        result = TvogCalculator(store, deterministic_bel=1_000.0).calculate()
        assert result.tvog == pytest.approx(150.0)

    def test_mean_stochastic_bel(self):
        store = _make_store({1: 1_000.0, 2: 1_100.0, 3: 1_200.0, 4: 1_300.0})
        result = TvogCalculator(store, deterministic_bel=1_000.0).calculate()
        assert result.mean_stochastic_bel == pytest.approx(1_150.0)

    def test_scenario_bels_ordered_by_id(self):
        # IDs inserted out of order — should be returned sorted ascending
        store = _make_store({3: 1_200.0, 1: 1_000.0, 4: 1_300.0, 2: 1_100.0})
        result = TvogCalculator(store, deterministic_bel=1_000.0).calculate()
        assert result.scenario_bels == pytest.approx([1_000.0, 1_100.0, 1_200.0, 1_300.0])


# ---------------------------------------------------------------------------
# Anchor 3 — negative TVOG
# ---------------------------------------------------------------------------

class TestNegativeTvog:
    """Stochastic mean below deterministic BEL — guarantees out of the money."""

    def test_tvog_is_negative(self):
        # BELs: 700, 800, 900, 1000 → mean = 850, deterministic = 1000
        store = _make_store({1: 700.0, 2: 800.0, 3: 900.0, 4: 1_000.0})
        result = TvogCalculator(store, deterministic_bel=1_000.0).calculate()
        assert result.tvog == pytest.approx(-150.0)

    def test_mean_stochastic_bel(self):
        store = _make_store({1: 700.0, 2: 800.0, 3: 900.0, 4: 1_000.0})
        result = TvogCalculator(store, deterministic_bel=1_000.0).calculate()
        assert result.mean_stochastic_bel == pytest.approx(850.0)


# ---------------------------------------------------------------------------
# Anchor 4 — single scenario
# ---------------------------------------------------------------------------

class TestSingleScenario:
    def test_single_scenario_tvog(self):
        store = _make_store({1: 500.0})
        result = TvogCalculator(store, deterministic_bel=400.0).calculate()
        assert result.tvog == pytest.approx(100.0)
        assert result.n_scenarios == 1
        assert result.scenario_bels == pytest.approx([500.0])

    def test_single_scenario_zero_tvog(self):
        store = _make_store({1: 1_234.56})
        result = TvogCalculator(store, deterministic_bel=1_234.56).calculate()
        assert result.tvog == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Anchor 5 — percentile distribution
# ---------------------------------------------------------------------------

class TestPercentiles:
    """Percentile values computed over scenario BEL distribution."""

    def test_median(self):
        # 5 scenarios: 100, 200, 300, 400, 500 → median = 300
        store = _make_store({1: 100.0, 2: 200.0, 3: 300.0, 4: 400.0, 5: 500.0})
        result = TvogCalculator(
            store, deterministic_bel=300.0, percentiles=[50]
        ).calculate()
        assert result.percentile_bels[50] == pytest.approx(300.0)

    def test_min_max_percentiles(self):
        store = _make_store({1: 100.0, 2: 200.0, 3: 300.0, 4: 400.0, 5: 500.0})
        result = TvogCalculator(
            store, deterministic_bel=300.0, percentiles=[0, 100]
        ).calculate()
        assert result.percentile_bels[0]   == pytest.approx(100.0)
        assert result.percentile_bels[100] == pytest.approx(500.0)

    def test_default_percentile_keys(self):
        store = _make_store({1: 1_000.0, 2: 2_000.0})
        result = TvogCalculator(store, deterministic_bel=1_500.0).calculate()
        assert set(result.percentile_bels.keys()) == {5, 10, 25, 50, 75, 90, 95}

    def test_custom_percentiles(self):
        store = _make_store({1: 1_000.0, 2: 2_000.0, 3: 3_000.0})
        result = TvogCalculator(
            store, deterministic_bel=2_000.0, percentiles=[25, 75]
        ).calculate()
        assert set(result.percentile_bels.keys()) == {25, 75}
        # 25th pct of [1000, 2000, 3000] = 1500 (linear interpolation)
        assert result.percentile_bels[25] == pytest.approx(1_500.0)
        assert result.percentile_bels[75] == pytest.approx(2_500.0)

    def test_no_percentiles(self):
        store = _make_store({1: 1_000.0})
        result = TvogCalculator(
            store, deterministic_bel=1_000.0, percentiles=[]
        ).calculate()
        assert result.percentile_bels == {}


# ---------------------------------------------------------------------------
# TvogResult attributes
# ---------------------------------------------------------------------------

class TestTvogResultAttributes:
    """All fields on TvogResult are populated correctly."""

    def test_all_fields_populated(self):
        store = _make_store({1: 800.0, 2: 1_200.0})
        result = TvogCalculator(store, deterministic_bel=1_000.0).calculate()
        assert isinstance(result, TvogResult)
        assert result.n_scenarios == 2
        assert result.deterministic_bel == pytest.approx(1_000.0)
        assert result.mean_stochastic_bel == pytest.approx(1_000.0)
        assert result.tvog == pytest.approx(0.0)
        assert len(result.scenario_bels) == 2

    def test_tvog_equals_mean_minus_deterministic(self):
        """Invariant: tvog == mean_stochastic_bel − deterministic_bel."""
        store = _make_store({1: 1_100.0, 2: 900.0, 3: 1_300.0, 4: 700.0})
        result = TvogCalculator(store, deterministic_bel=850.0).calculate()
        assert result.tvog == pytest.approx(
            result.mean_stochastic_bel - result.deterministic_bel
        )


# ---------------------------------------------------------------------------
# Error conditions
# ---------------------------------------------------------------------------

class TestErrorConditions:
    def test_empty_store_raises(self):
        store = ResultStore(run_id="empty")
        with pytest.raises(ValueError, match="empty"):
            TvogCalculator(store, deterministic_bel=0.0).calculate()

    def test_missing_timestep_zero_raises(self):
        """Store has scenario 1 at t=1 but not t=0 — KeyError from ResultStore."""
        store = ResultStore(run_id="bad")
        store.store(TimestepResult(
            run_id="bad",
            scenario_id=1,
            timestep=1,          # t=1, not t=0
            cashflows=_null_cashflows(1),
            decrements=_null_decrements(1),
            bel=1_000.0,
            reserve=1_000.0,
        ))
        with pytest.raises(KeyError):
            TvogCalculator(store, deterministic_bel=1_000.0).calculate()


# ---------------------------------------------------------------------------
# Large scenario set — numerical stability
# ---------------------------------------------------------------------------

class TestLargeScenarioSet:
    """100 scenarios — confirms numpy mean and percentile are correct."""

    def test_100_scenarios_mean(self):
        # BELs = 1, 2, ..., 100  →  mean = 50.5
        bels = {i: float(i) for i in range(1, 101)}
        store = _make_store(bels)
        result = TvogCalculator(store, deterministic_bel=50.5).calculate()
        assert result.mean_stochastic_bel == pytest.approx(50.5)
        assert result.tvog == pytest.approx(0.0)
        assert result.n_scenarios == 100

    def test_100_scenarios_percentiles(self):
        bels = {i: float(i) for i in range(1, 101)}
        store = _make_store(bels)
        result = TvogCalculator(
            store, deterministic_bel=50.5, percentiles=[0, 50, 100]
        ).calculate()
        assert result.percentile_bels[0]   == pytest.approx(1.0)
        assert result.percentile_bels[50]  == pytest.approx(50.5)
        assert result.percentile_bels[100] == pytest.approx(100.0)
