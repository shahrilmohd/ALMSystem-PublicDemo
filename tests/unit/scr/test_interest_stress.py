"""
tests/unit/scr/test_interest_stress.py

Numerical tests for InterestStressEngine.

Portfolio used throughout
--------------------------
One AC bond: face_value=100_000, coupon=5%, maturity=60 months (5 years).
One FVTPL bond: face_value=50_000, coupon=4%, maturity=36 months (3 years).

RF curve: flat 3%.
Rate shocks: ±100 bps (up/down).
MA benefit: 0 bps for non-BPA tests; 50 bps for BPA-mode test.
"""
from __future__ import annotations

import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.asset.bond import Bond
from engine.core.projection_calendar import ProjectionCalendar
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.scr.interest_stress import InterestStressEngine, _shift_rfr_curve


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
def engine() -> InterestStressEngine:
    return InterestStressEngine(rate_up_bps=100.0, rate_down_bps=100.0)


# ---------------------------------------------------------------------------
# Test 1 — rate up: bond MV decreases (duration risk)
# ---------------------------------------------------------------------------

def test_rate_up_asset_mv_decreases(
    engine, asset_model, liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    up, _ = engine.compute(
        asset_model=asset_model,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    # Rising rates → bond prices fall → negative MV change
    assert up.asset_mv_change < 0.0


# ---------------------------------------------------------------------------
# Test 2 — rate down: bond MV increases
# ---------------------------------------------------------------------------

def test_rate_down_asset_mv_increases(
    engine, asset_model, liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    _, down = engine.compute(
        asset_model=asset_model,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    # Falling rates → bond prices rise → positive MV change
    assert down.asset_mv_change > 0.0


# ---------------------------------------------------------------------------
# Test 3 — rate up: BEL decreases (higher discount rate)
# ---------------------------------------------------------------------------

def test_rate_up_bel_decreases(
    engine, asset_model, liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    up, _ = engine.compute(
        asset_model=asset_model,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    # Higher discount rates → lower present value of liabilities
    assert up.bel_change < 0.0


# ---------------------------------------------------------------------------
# Test 4 — rate down: BEL increases (lower discount rate)
# ---------------------------------------------------------------------------

def test_rate_down_bel_increases(
    engine, asset_model, liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    _, down = engine.compute(
        asset_model=asset_model,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    # Lower discount rates → higher present value of liabilities
    assert down.bel_change > 0.0


# ---------------------------------------------------------------------------
# Test 5 — own_funds_change = asset_mv_change − bel_change (accounting identity)
# ---------------------------------------------------------------------------

def test_own_funds_change_accounting_identity(
    engine, asset_model, liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    up, down = engine.compute(
        asset_model=asset_model,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    assert up.own_funds_change   == pytest.approx(up.asset_mv_change   - up.bel_change,   rel=1e-9)
    assert down.own_funds_change == pytest.approx(down.asset_mv_change - down.bel_change, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 6 — rate down floor: stressed rates never go negative
# ---------------------------------------------------------------------------

def test_rate_down_floor_at_zero():
    """
    A flat 50 bps curve stressed down by 100 bps must floor at 0, not go negative.
    """
    low_rfr = RiskFreeRateCurve.flat(0.005)   # 50 bps
    shifted = _shift_rfr_curve(low_rfr, delta=-0.01)  # −100 bps
    # All knots should be clamped at 0
    for maturity, rate in shifted.spot_rates.items():
        assert rate >= 0.0, f"Rate at maturity {maturity} went negative: {rate}"


# ---------------------------------------------------------------------------
# Test 7 — MA benefit (non-zero) offsets BEL under rate stress
# ---------------------------------------------------------------------------

def test_ma_benefit_offsets_bel(
    asset_model, liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    """
    Passing ma_benefit_bps=50 raises the discount rate applied to liabilities.
    The stressed BEL (at higher discount rate) should be lower than the
    no-MA stressed BEL for the same rate shock.
    """
    engine_no_ma  = InterestStressEngine(rate_up_bps=100.0, rate_down_bps=100.0)
    engine_with_ma = InterestStressEngine(rate_up_bps=100.0, rate_down_bps=100.0)

    up_no_ma, _ = engine_no_ma.compute(
        asset_model=asset_model,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    up_with_ma, _ = engine_with_ma.compute(
        asset_model=asset_model,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        ma_benefit_bps=50.0,
        scenario=scenario,
        calendar=calendar,
    )
    # MA adds 50 bps on top → even lower BEL, so bel_change is more negative
    assert up_with_ma.bel_change < up_no_ma.bel_change


# ---------------------------------------------------------------------------
# Test 8 — invalid constructor arguments
# ---------------------------------------------------------------------------

def test_invalid_constructor():
    with pytest.raises(ValueError):
        InterestStressEngine(rate_up_bps=-10)
    with pytest.raises(ValueError):
        InterestStressEngine(rate_down_bps=-5)


# ---------------------------------------------------------------------------
# Test 9 — zero shock: no change
# ---------------------------------------------------------------------------

def test_zero_shock_no_change(
    asset_model, liability_cashflows, base_bel, rfr_curve, scenario, calendar,
):
    engine_zero = InterestStressEngine(rate_up_bps=0.0, rate_down_bps=0.0)
    up, down = engine_zero.compute(
        asset_model=asset_model,
        liability_cashflows=liability_cashflows,
        base_bel_post_ma=base_bel,
        rfr_curve=rfr_curve,
        ma_benefit_bps=0.0,
        scenario=scenario,
        calendar=calendar,
    )
    assert up.asset_mv_change   == pytest.approx(0.0, abs=1e-6)
    assert up.bel_change        == pytest.approx(0.0, abs=1e-6)
    assert down.asset_mv_change == pytest.approx(0.0, abs=1e-6)
    assert down.bel_change      == pytest.approx(0.0, abs=1e-6)
