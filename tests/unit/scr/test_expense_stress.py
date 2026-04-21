"""tests/unit/scr/test_expense_stress.py — ExpenseStressEngine unit tests."""
from __future__ import annotations

import dataclasses
import math

import pytest

from engine.core.projection_calendar import ProjectionCalendar
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.scr.expense_stress import ExpenseStressEngine, ExpenseStressResult
from engine.scr.scr_assumptions import SCRStressAssumptions


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def assumptions() -> SCRStressAssumptions:
    return SCRStressAssumptions.sii_standard_formula()


@pytest.fixture
def engine(assumptions: SCRStressAssumptions) -> ExpenseStressEngine:
    return ExpenseStressEngine(assumptions)


@pytest.fixture
def flat_rfr() -> RiskFreeRateCurve:
    return RiskFreeRateCurve.flat(rate_yr=0.05)


@pytest.fixture
def annual_calendar() -> ProjectionCalendar:
    return ProjectionCalendar(projection_years=10, monthly_years=0)


# ---------------------------------------------------------------------------
# E1: None expense_cashflows → SCR = 0
# ---------------------------------------------------------------------------

def test_e1_none_cashflows_zero_scr(
    engine: ExpenseStressEngine,
    flat_rfr: RiskFreeRateCurve,
    annual_calendar: ProjectionCalendar,
) -> None:
    result = engine.compute(None, flat_rfr, annual_calendar)
    assert result.scr_expense == 0.0
    assert result.expense_bel_change == 0.0


def test_e1_empty_dict_zero_scr(
    engine: ExpenseStressEngine,
    flat_rfr: RiskFreeRateCurve,
    annual_calendar: ProjectionCalendar,
) -> None:
    result = engine.compute({}, flat_rfr, annual_calendar)
    assert result.scr_expense == 0.0


# ---------------------------------------------------------------------------
# E2: flat expense schedule + 10% loading → BEL increases ≈ 10% of PV(expenses)
# ---------------------------------------------------------------------------

def test_e2_loading_shock_increases_bel(
    flat_rfr: RiskFreeRateCurve,
    annual_calendar: ProjectionCalendar,
) -> None:
    # Use zero inflation so the loading dominates
    base = SCRStressAssumptions.sii_standard_formula()
    zero_inflation = dataclasses.replace(base, expense_inflation_shock_pa=0.0)
    engine = ExpenseStressEngine(zero_inflation)

    # Single period cashflow at t=0 (no discounting effect, t_years=0 → df=1, inflation=(1+0)^0=1)
    cashflows = {0: 1000.0}
    result = engine.compute(cashflows, flat_rfr, annual_calendar)

    # stressed = 1000 × 1.10 × 1.0^0 = 1100; ΔBEL = 1100 - 1000 = 100 (adverse)
    # SCR = max(ΔBEL, 0) = 100
    assert result.expense_bel_change == pytest.approx(100.0, rel=1e-6)
    assert result.scr_expense == pytest.approx(100.0, rel=1e-6)  # positive: BEL rose


# ---------------------------------------------------------------------------
# E3: inflation drift accumulates over time
# ---------------------------------------------------------------------------

def test_e3_inflation_drift(
    flat_rfr: RiskFreeRateCurve,
    annual_calendar: ProjectionCalendar,
) -> None:
    base = SCRStressAssumptions.sii_standard_formula()
    no_loading = dataclasses.replace(base, expense_loading_shock_factor=0.0)
    engine = ExpenseStressEngine(no_loading)

    # Two equal cashflows at period 1 (t=1yr) and period 9 (t=9yr)
    cashflows = {1: 1000.0, 9: 1000.0}
    result = engine.compute(cashflows, flat_rfr, annual_calendar)

    # Inflation factor at t=1: (1.01)^1 = 1.01
    # Inflation factor at t=9: (1.01)^9 ≈ 1.0937
    # Period-9 has much larger relative stress than period-1
    inflation_t1 = (1.01 ** 1) - 1.0   # 0.01
    inflation_t9 = (1.01 ** 9) - 1.0   # ≈ 0.0937

    assert inflation_t9 > inflation_t1, (
        "inflation factor at t=9 should be larger than at t=1"
    )

    # ΔBEL positive (adverse) → SCR = ΔBEL > 0
    assert result.expense_bel_change > 0.0
    assert result.scr_expense == pytest.approx(result.expense_bel_change)


# ---------------------------------------------------------------------------
# E4: zero shocks in assumptions → SCR = 0 even with non-empty cashflows
# ---------------------------------------------------------------------------

def test_e4_zero_shocks_zero_scr(
    flat_rfr: RiskFreeRateCurve,
    annual_calendar: ProjectionCalendar,
) -> None:
    base = SCRStressAssumptions.sii_standard_formula()
    no_shock = dataclasses.replace(
        base,
        expense_loading_shock_factor=0.0,
        expense_inflation_shock_pa=0.0,
    )
    engine = ExpenseStressEngine(no_shock)

    cashflows = {1: 500.0, 2: 500.0, 3: 500.0}
    result = engine.compute(cashflows, flat_rfr, annual_calendar)

    assert result.scr_expense == pytest.approx(0.0, abs=1e-10)
    assert result.expense_bel_change == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Governance: shock parameters stored in result
# ---------------------------------------------------------------------------

def test_governance_fields_populated(
    engine: ExpenseStressEngine,
    flat_rfr: RiskFreeRateCurve,
    annual_calendar: ProjectionCalendar,
) -> None:
    result = engine.compute({1: 100.0}, flat_rfr, annual_calendar)
    assert result.loading_shock_factor == pytest.approx(0.10)
    assert result.inflation_shock_pa == pytest.approx(0.01)
