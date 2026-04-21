"""
tests/unit/scr/test_scr_calculator.py

Integration-level tests for SCRCalculator.

Strategy
--------
We stub the three stress engines with simple doubles that return
pre-configured results.  This isolates SCRCalculator from the
internals of each engine and focuses the tests on:
  - Correct wiring (each engine's compute() is called once).
  - SCRResult assembly (scr_spread, scr_interest, scr_longevity formulas).
  - Governance fields (base_asset_mv, shock bps, etc.).
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.asset.bond import Bond
from engine.core.projection_calendar import ProjectionCalendar
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.matching_adjustment.ma_calculator import MAResult
from engine.scr.interest_stress import InterestStressResult
from engine.scr.longevity_stress import LongevityStressResult
from engine.scr.scr_calculator import SCRCalculator
from engine.scr.scr_result import SCRResult
from engine.scr.spread_stress import SpreadStressResult


# ---------------------------------------------------------------------------
# Stub engines
# ---------------------------------------------------------------------------

class _StubSpreadEngine:
    """Returns fixed SpreadStressResult pair without running any calculation."""
    spread_up_bps   = 75.0
    spread_down_bps = 25.0

    def __init__(self, up: SpreadStressResult, down: SpreadStressResult):
        self._up   = up
        self._down = down
        self.call_count = 0

    def compute(self, **kwargs):
        self.call_count += 1
        return self._up, self._down


class _StubInterestEngine:
    """Returns fixed InterestStressResult pair without running any calculation."""
    rate_up_bps   = 100.0
    rate_down_bps = 100.0

    def __init__(self, up: InterestStressResult, down: InterestStressResult):
        self._up   = up
        self._down = down
        self.call_count = 0

    def compute(self, **kwargs):
        self.call_count += 1
        return self._up, self._down


class _StubLongevityEngine:
    """Returns a fixed LongevityStressResult without running any calculation."""
    mortality_stress_factor = 0.20

    def __init__(self, result: LongevityStressResult):
        self._result    = result
        self.call_count = 0

    def compute(self, **kwargs):
        self.call_count += 1
        return self._result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rfr_curve() -> RiskFreeRateCurve:
    return RiskFreeRateCurve.flat(0.03)


@pytest.fixture
def scenario(rfr_curve) -> AssetScenarioPoint:
    return AssetScenarioPoint(
        timestep=0,
        rate_curve=rfr_curve,
        equity_total_return_yr=0.0,
        dt=1 / 12,
    )


@pytest.fixture
def asset_model() -> AssetModel:
    bond = Bond(
        asset_id="B1",
        face_value=100_000.0,
        annual_coupon_rate=0.05,
        maturity_month=60,
        accounting_basis="AC",
        initial_book_value=100_000.0,
        eir=0.05,
        calibration_spread=0.01,
    )
    return AssetModel([bond])


@pytest.fixture
def calendar() -> ProjectionCalendar:
    return ProjectionCalendar(projection_years=30, monthly_years=5)


@pytest.fixture
def ma_result() -> MAResult:
    return MAResult(
        ma_benefit_bps=50.0,
        eligible_asset_ids=[],
        cashflow_test_passes=True,
        failing_periods=[],
        per_asset_contributions=pd.DataFrame(),
        fs_table_effective_date=pd.Timestamp("2024-07-01").date(),
        fs_table_source_ref="PRA PS10/24",
    )


# Canonical stub results — own_funds_change negative for spread up (loss)
_SP_UP   = SpreadStressResult(own_funds_change=-8_000.0, asset_mv_change=-9_000.0,
                               bel_change=-1_000.0, stressed_ma_benefit_bps=60.0)
_SP_DOWN = SpreadStressResult(own_funds_change=+3_000.0, asset_mv_change=+3_500.0,
                               bel_change=+500.0, stressed_ma_benefit_bps=40.0)

_IR_UP   = InterestStressResult(own_funds_change=-5_000.0, asset_mv_change=-7_000.0,
                                 bel_change=-2_000.0)
_IR_DOWN = InterestStressResult(own_funds_change=-4_500.0, asset_mv_change=+6_000.0,
                                 bel_change=+10_500.0)

_LONG    = LongevityStressResult(scr_longevity=3_000.0,
                                  stressed_bel_series=(105_000.0, 95_000.0),
                                  stressed_bel_t0=105_000.0)


@pytest.fixture
def calculator() -> SCRCalculator:
    return SCRCalculator(
        spread_engine    = _StubSpreadEngine(_SP_UP, _SP_DOWN),
        interest_engine  = _StubInterestEngine(_IR_UP, _IR_DOWN),
        longevity_engine = _StubLongevityEngine(_LONG),
    )


# ---------------------------------------------------------------------------
# Test 1 — each engine is called exactly once
# ---------------------------------------------------------------------------

def test_each_engine_called_once(
    calculator, asset_model, scenario, calendar, ma_result, rfr_curve
):
    calculator.compute(
        asset_model=asset_model,
        assets_df=pd.DataFrame(),
        asset_cfs=pd.DataFrame(columns=["t", "asset_id", "cf"]),
        liability_cashflows={0: 3_000.0},
        rfr_curve=rfr_curve,
        ma_result=ma_result,
        base_bel_post_ma=100_000.0,
        base_bel_series=[100_000.0, 90_000.0],
        scenario=scenario,
        calendar=calendar,
        base_assumptions=None,
        per_cohort_mps={},
        per_cohort_models={},
    )
    assert calculator._spread.call_count    == 1
    assert calculator._interest.call_count  == 1
    assert calculator._longevity.call_count == 1


# ---------------------------------------------------------------------------
# Test 2 — scr_spread = max(−own_funds_change_up, 0)
# ---------------------------------------------------------------------------

def test_scr_spread_formula(
    calculator, asset_model, scenario, calendar, ma_result, rfr_curve
):
    result = calculator.compute(
        asset_model=asset_model,
        assets_df=pd.DataFrame(),
        asset_cfs=pd.DataFrame(columns=["t", "asset_id", "cf"]),
        liability_cashflows={0: 3_000.0},
        rfr_curve=rfr_curve,
        ma_result=ma_result,
        base_bel_post_ma=100_000.0,
        base_bel_series=[100_000.0],
        scenario=scenario,
        calendar=calendar,
        base_assumptions=None,
        per_cohort_mps={},
        per_cohort_models={},
    )
    assert result.scr_spread == pytest.approx(max(-_SP_UP.own_funds_change, 0.0))


# ---------------------------------------------------------------------------
# Test 3 — scr_interest = max(−up, −down, 0)
# ---------------------------------------------------------------------------

def test_scr_interest_formula(
    calculator, asset_model, scenario, calendar, ma_result, rfr_curve
):
    result = calculator.compute(
        asset_model=asset_model,
        assets_df=pd.DataFrame(),
        asset_cfs=pd.DataFrame(columns=["t", "asset_id", "cf"]),
        liability_cashflows={0: 3_000.0},
        rfr_curve=rfr_curve,
        ma_result=ma_result,
        base_bel_post_ma=100_000.0,
        base_bel_series=[100_000.0],
        scenario=scenario,
        calendar=calendar,
        base_assumptions=None,
        per_cohort_mps={},
        per_cohort_models={},
    )
    expected = max(-_IR_UP.own_funds_change, -_IR_DOWN.own_funds_change, 0.0)
    assert result.scr_interest == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Test 4 — scr_longevity passed through unchanged
# ---------------------------------------------------------------------------

def test_scr_longevity_passthrough(
    calculator, asset_model, scenario, calendar, ma_result, rfr_curve
):
    result = calculator.compute(
        asset_model=asset_model,
        assets_df=pd.DataFrame(),
        asset_cfs=pd.DataFrame(columns=["t", "asset_id", "cf"]),
        liability_cashflows={0: 3_000.0},
        rfr_curve=rfr_curve,
        ma_result=ma_result,
        base_bel_post_ma=100_000.0,
        base_bel_series=[100_000.0],
        scenario=scenario,
        calendar=calendar,
        base_assumptions=None,
        per_cohort_mps={},
        per_cohort_models={},
    )
    assert result.scr_longevity == pytest.approx(_LONG.scr_longevity)
    assert result.longevity_stressed_bel_series == _LONG.stressed_bel_series


# ---------------------------------------------------------------------------
# Test 5 — governance fields are correctly populated
# ---------------------------------------------------------------------------

def test_governance_fields(
    calculator, asset_model, scenario, calendar, ma_result, rfr_curve
):
    base_bel = 100_000.0
    result = calculator.compute(
        asset_model=asset_model,
        assets_df=pd.DataFrame(),
        asset_cfs=pd.DataFrame(columns=["t", "asset_id", "cf"]),
        liability_cashflows={0: 3_000.0},
        rfr_curve=rfr_curve,
        ma_result=ma_result,
        base_bel_post_ma=base_bel,
        base_bel_series=[base_bel],
        scenario=scenario,
        calendar=calendar,
        base_assumptions=None,
        per_cohort_mps={},
        per_cohort_models={},
    )
    expected_mv = asset_model.total_market_value(scenario)
    assert result.base_asset_mv           == pytest.approx(expected_mv, rel=1e-6)
    assert result.base_bel_post_ma        == pytest.approx(base_bel)
    assert result.spread_up_bps           == pytest.approx(75.0)
    assert result.spread_down_bps         == pytest.approx(25.0)
    assert result.rate_up_bps             == pytest.approx(100.0)
    assert result.rate_down_bps           == pytest.approx(100.0)
    assert result.longevity_mortality_stress_factor == pytest.approx(0.20)
