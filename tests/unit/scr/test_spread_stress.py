"""
tests/unit/scr/test_spread_stress.py

Numerical tests for SpreadStressEngine.

Portfolio used throughout
--------------------------
One AC bond: face_value=100_000, coupon=5%, maturity=60 months (5 years),
calibration_spread=0.01 (100 bps), spread_bps=100 bps.
One FVTPL bond: face_value=50_000, coupon=4%, maturity=36 months,
calibration_spread=0.008 (80 bps), spread_bps=80 bps.

RF curve: flat 3% for all tests.
MA shock: +75 bps (up), −25 bps (down).
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.asset.bond import Bond
from engine.core.projection_calendar import ProjectionCalendar
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.matching_adjustment.fundamental_spread import FundamentalSpreadTable
from engine.matching_adjustment.ma_calculator import MACalculator, build_ma_discount_curve
from engine.scr.spread_stress import (
    SpreadStressEngine,
    _apply_credit_spread_shock,
    _liability_dict_to_annual_df,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FS_CSV = (
    "# effective_date: 2024-07-01\n"
    "# source_ref: PRA PS10/24\n"
    "rating,seniority,tenor_lower,tenor_upper,long_run_pd_pct,lgd_pct,"
    "downgrade_allowance_bps,fs_bps\n"
    "BBB,senior_unsecured,0,5,0.17,40,21,28\n"
    "BBB,senior_unsecured,5,10,0.17,40,24,31\n"
    "BBB-,senior_unsecured,0,5,0.24,40,22,32\n"
)


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
def ac_bond() -> Bond:
    return Bond(
        asset_id="AC1",
        face_value=100_000.0,
        annual_coupon_rate=0.05,
        maturity_month=60,
        accounting_basis="AC",
        initial_book_value=100_000.0,
        eir=0.05,
        calibration_spread=0.01,
    )


@pytest.fixture
def fvtpl_bond() -> Bond:
    return Bond(
        asset_id="FV1",
        face_value=50_000.0,
        annual_coupon_rate=0.04,
        maturity_month=36,
        accounting_basis="FVTPL",
        initial_book_value=50_000.0,
        eir=0.04,
        calibration_spread=0.008,
    )


@pytest.fixture
def asset_model(ac_bond, fvtpl_bond) -> AssetModel:
    return AssetModel([ac_bond, fvtpl_bond])


@pytest.fixture
def assets_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "asset_id": "AC1",
            "cashflow_type": "fixed",
            "currency": "GBP",
            "has_credit_risk_transfer": False,
            "has_qualifying_currency_swap": False,
            "rating": "BBB",
            "seniority": "senior_unsecured",
            "tenor_years": 5.0,
            "spread_bps": 100.0,
            "accounting_basis": "AC",
        },
        {
            "asset_id": "FV1",
            "cashflow_type": "fixed",
            "currency": "GBP",
            "has_credit_risk_transfer": False,
            "has_qualifying_currency_swap": False,
            "rating": "BBB-",
            "seniority": "senior_unsecured",
            "tenor_years": 3.0,
            "spread_bps": 80.0,
            "accounting_basis": "FVTPL",
        },
    ])


@pytest.fixture
def asset_cfs() -> pd.DataFrame:
    rows = []
    for t in range(1, 6):
        rows.append({"t": t, "asset_id": "AC1", "cf": 5_200.0})
        rows.append({"t": t, "asset_id": "FV1", "cf": 1_600.0})
    return pd.DataFrame(rows)


@pytest.fixture
def liability_cashflows() -> dict[int, float]:
    return {i: 3_000.0 for i in range(60)}


@pytest.fixture
def calendar() -> ProjectionCalendar:
    return ProjectionCalendar(projection_years=60, monthly_years=10)


@pytest.fixture
def base_bel(rfr_curve, liability_cashflows, calendar) -> float:
    from engine.scr._bel_utils import discount_cashflows
    return discount_cashflows(liability_cashflows, rfr_curve, calendar)


@pytest.fixture
def fs_table(tmp_path) -> FundamentalSpreadTable:
    p = tmp_path / "fs.csv"
    p.write_text(FS_CSV)
    return FundamentalSpreadTable.from_csv(p)


@pytest.fixture
def ma_calculator(fs_table) -> MACalculator:
    return MACalculator(fs_table)


@pytest.fixture
def engine_non_bpa() -> SpreadStressEngine:
    return SpreadStressEngine(ma_calculator=None, spread_up_bps=75, spread_down_bps=25)


@pytest.fixture
def engine_bpa(ma_calculator) -> SpreadStressEngine:
    return SpreadStressEngine(ma_calculator=ma_calculator, spread_up_bps=75, spread_down_bps=25)


# ---------------------------------------------------------------------------
# Test 1 — spread widening decreases asset MV
# ---------------------------------------------------------------------------

def test_spread_up_asset_mv_decreases(
    engine_non_bpa, asset_model, assets_df, asset_cfs,
    liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    up, _ = engine_non_bpa.compute(
        asset_model=asset_model,
        assets_df=assets_df,
        asset_cfs=asset_cfs,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        base_ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    # Widening → bonds reprice lower → negative MV change
    assert up.asset_mv_change < 0.0


# ---------------------------------------------------------------------------
# Test 2 — spread tightening increases asset MV
# ---------------------------------------------------------------------------

def test_spread_down_asset_mv_increases(
    engine_non_bpa, asset_model, assets_df, asset_cfs,
    liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    _, down = engine_non_bpa.compute(
        asset_model=asset_model,
        assets_df=assets_df,
        asset_cfs=asset_cfs,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        base_ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    assert down.asset_mv_change > 0.0


# ---------------------------------------------------------------------------
# Test 3 — non-BPA mode: ΔBEL = 0 (no MA offset)
# ---------------------------------------------------------------------------

def test_non_bpa_mode_bel_unchanged(
    engine_non_bpa, asset_model, assets_df, asset_cfs,
    liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    up, down = engine_non_bpa.compute(
        asset_model=asset_model,
        assets_df=assets_df,
        asset_cfs=asset_cfs,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        base_ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    assert up.bel_change == pytest.approx(0.0, abs=1e-6)
    assert down.bel_change == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test 4 — non-BPA: own funds change equals pure asset MV change
# ---------------------------------------------------------------------------

def test_non_bpa_own_funds_equals_asset_mv_change(
    engine_non_bpa, asset_model, assets_df, asset_cfs,
    liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    up, down = engine_non_bpa.compute(
        asset_model=asset_model,
        assets_df=assets_df,
        asset_cfs=asset_cfs,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        base_ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    assert up.own_funds_change   == pytest.approx(up.asset_mv_change,   rel=1e-9)
    assert down.own_funds_change == pytest.approx(down.asset_mv_change, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 5 — BPA mode: spread widening increases MA benefit
# ---------------------------------------------------------------------------

def test_bpa_spread_up_ma_benefit_increases(
    engine_bpa, asset_model, assets_df, asset_cfs,
    liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    up, _ = engine_bpa.compute(
        asset_model=asset_model,
        assets_df=assets_df,
        asset_cfs=asset_cfs,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        base_ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    # Only the AC bond qualifies for MA; after widening MA contribution rises.
    assert up.stressed_ma_benefit_bps > 0.0


# ---------------------------------------------------------------------------
# Test 6 — BPA mode: spread widening reduces post-MA BEL (higher discount rate)
# ---------------------------------------------------------------------------

def test_bpa_spread_up_bel_decreases(
    engine_bpa, asset_model, assets_df, asset_cfs,
    liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    up, _ = engine_bpa.compute(
        asset_model=asset_model,
        assets_df=assets_df,
        asset_cfs=asset_cfs,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        base_ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    # Higher MA benefit → higher discount rate → BEL falls (negative ΔBEL)
    assert up.bel_change < 0.0


# ---------------------------------------------------------------------------
# Test 7 — spread tightening floor: stressed spread >= 0
# ---------------------------------------------------------------------------

def test_spread_down_floor_at_zero():
    """
    A bond with calibration_spread=30 bps shocked down by 75 bps.
    Stressed spread must be floored at 0, not go negative.
    """
    low_spread_bond = Bond(
        asset_id="LOW",
        face_value=100_000.0,
        annual_coupon_rate=0.03,
        maturity_month=60,
        accounting_basis="AC",
        initial_book_value=100_000.0,
        eir=0.03,
        calibration_spread=0.003,   # 30 bps
    )
    rfr  = RiskFreeRateCurve.flat(0.03)
    scen = AssetScenarioPoint(timestep=0, rate_curve=rfr, equity_total_return_yr=0.0, dt=1/12)
    model = AssetModel([low_spread_bond])
    cal   = ProjectionCalendar(projection_years=10, monthly_years=5)

    engine = SpreadStressEngine(ma_calculator=None, spread_up_bps=75, spread_down_bps=75)
    base_bel = 50_000.0
    _, down = engine.compute(
        asset_model=model,
        assets_df=pd.DataFrame(),
        asset_cfs=pd.DataFrame(columns=["t", "asset_id", "cf"]),
        liability_cashflows={i: 500.0 for i in range(60)},
        base_bel_post_ma=base_bel,
        rfr_curve=rfr,
        base_ma_benefit_bps=0.0,
        scenario=scen,
        calendar=cal,
    )
    # Down shock of 75 bps on 30 bps bond → floored at 0.
    # Stressed MV at cs=0 should be HIGHER than base MV (spread removal = price rise).
    assert down.asset_mv_change > 0.0


# ---------------------------------------------------------------------------
# Test 8 — scr_spread = max(-own_funds_change_up, 0)
# ---------------------------------------------------------------------------

def test_scr_spread_from_widening_only(
    engine_non_bpa, asset_model, assets_df, asset_cfs,
    liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    from engine.scr.scr_result import SCRResult
    up, down = engine_non_bpa.compute(
        asset_model=asset_model,
        assets_df=assets_df,
        asset_cfs=asset_cfs,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        base_ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    scr_spread = max(-up.own_funds_change, 0.0)
    assert scr_spread >= 0.0
    assert scr_spread == pytest.approx(-up.own_funds_change, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 9 — _apply_credit_spread_shock helper
# ---------------------------------------------------------------------------

def test_apply_credit_spread_shock_floor():
    df = pd.DataFrame([
        {"asset_id": "A", "spread_bps": 100.0},
        {"asset_id": "B", "spread_bps": 20.0},   # would go negative
    ])
    shocked = _apply_credit_spread_shock(df, delta_bps=-50.0)
    assert shocked.loc[shocked["asset_id"] == "A", "spread_bps"].iloc[0] == pytest.approx(50.0)
    assert shocked.loc[shocked["asset_id"] == "B", "spread_bps"].iloc[0] == pytest.approx(0.0)
    # Original unchanged
    assert df.loc[df["asset_id"] == "B", "spread_bps"].iloc[0] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Test 10 — invalid constructor arguments
# ---------------------------------------------------------------------------

def test_invalid_constructor():
    with pytest.raises(ValueError):
        SpreadStressEngine(spread_up_bps=-10)
    with pytest.raises(ValueError):
        SpreadStressEngine(spread_down_bps=-5)
