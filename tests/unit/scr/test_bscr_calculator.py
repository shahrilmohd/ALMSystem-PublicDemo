"""tests/unit/scr/test_bscr_calculator.py — BSCRCalculator unit tests."""
from __future__ import annotations

import pandas as pd
import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.asset.bond import Bond
from engine.core.projection_calendar import ProjectionCalendar
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.matching_adjustment.ma_calculator import MAResult
from engine.scr.bscr_calculator import BSCRCalculator
from engine.scr.bscr_result import BSCRResult
from engine.scr.scr_assumptions import SCRStressAssumptions


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def assumptions() -> SCRStressAssumptions:
    return SCRStressAssumptions.sii_standard_formula()


@pytest.fixture
def rfr_curve() -> RiskFreeRateCurve:
    return RiskFreeRateCurve.flat(0.03)


@pytest.fixture
def scenario(rfr_curve: RiskFreeRateCurve) -> AssetScenarioPoint:
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


def _compute(
    calculator: BSCRCalculator,
    asset_model: AssetModel,
    scenario: AssetScenarioPoint,
    calendar: ProjectionCalendar,
    rfr_curve: RiskFreeRateCurve,
    ma_result: MAResult,
    **kwargs,
) -> BSCRResult:
    defaults = dict(
        asset_model=asset_model,
        assets_df=pd.DataFrame(),
        asset_cfs=pd.DataFrame(columns=["t", "asset_id", "cf"]),
        liability_cashflows={0: 3_000.0},
        rfr_curve=rfr_curve,
        ma_result=ma_result,
        base_bel_post_ma=100_000.0,
        base_bel_series=[100_000.0, 90_000.0, 80_000.0],
        scenario=scenario,
        calendar=calendar,
        base_assumptions=None,
        per_cohort_mps={},
        per_cohort_models={},
    )
    defaults.update(kwargs)
    return calculator.compute(**defaults)


# ---------------------------------------------------------------------------
# BC1: construction with sii_standard_formula() succeeds
# ---------------------------------------------------------------------------

def test_bc1_constructs_cleanly(assumptions: SCRStressAssumptions) -> None:
    calc = BSCRCalculator(assumptions)
    assert calc.assumptions is assumptions


# ---------------------------------------------------------------------------
# BC2: all-zero optional inputs → zero new-module SCRs
# ---------------------------------------------------------------------------

def test_bc2_zero_optional_inputs_zero_new_scrs(
    assumptions, asset_model, scenario, calendar, rfr_curve, ma_result
) -> None:
    calc = BSCRCalculator(assumptions)
    result = _compute(calc, asset_model, scenario, calendar, rfr_curve, ma_result)

    # New Step-26 sub-modules: no inputs provided → zero
    assert result.scr_lapse       == 0.0
    assert result.scr_expense     == 0.0
    assert result.scr_currency    == 0.0
    assert result.scr_counterparty == 0.0
    # BPA path: scr_mortality defaulted to 0.0
    assert result.scr_mortality   == 0.0


# ---------------------------------------------------------------------------
# BC3: non-zero lapse BEL sensitivity → lapse SCR propagates into result
# ---------------------------------------------------------------------------

def test_bc3_lapse_scr_propagates(
    assumptions, asset_model, scenario, calendar, rfr_curve, ma_result
) -> None:
    calc = BSCRCalculator(assumptions)
    # BEL rises under down-shock: adverse
    result = _compute(
        calc, asset_model, scenario, calendar, rfr_curve, ma_result,
        bel_lapse_down=120_000.0,  # BEL rose by 20k
        base_lapse_bel=100_000.0,
    )
    # scr_lapse = -(120k - 100k) flipped sign = -20k... wait
    # of_change_down = -(120k - 100k) = -20k (loss); SCR contrib = 20k
    assert result.scr_lapse == pytest.approx(20_000.0)
    assert result.bscr > 0.0


# ---------------------------------------------------------------------------
# BC4: scr_total = bscr + scr_operational invariant
# ---------------------------------------------------------------------------

def test_bc4_scr_total_invariant(
    assumptions, asset_model, scenario, calendar, rfr_curve, ma_result
) -> None:
    calc = BSCRCalculator(assumptions)
    result = _compute(calc, asset_model, scenario, calendar, rfr_curve, ma_result)
    assert result.scr_total == pytest.approx(result.bscr + result.scr_operational)


# ---------------------------------------------------------------------------
# BC5: result.assumptions matches the SCRStressAssumptions passed at construction
# ---------------------------------------------------------------------------

def test_bc5_assumptions_in_result(
    assumptions, asset_model, scenario, calendar, rfr_curve, ma_result
) -> None:
    calc = BSCRCalculator(assumptions)
    result = _compute(calc, asset_model, scenario, calendar, rfr_curve, ma_result)
    assert result.assumptions is assumptions
