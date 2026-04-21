"""tests/unit/scr/test_risk_margin.py — RiskMarginCalculator unit tests."""
from __future__ import annotations

import dataclasses

import pytest

from engine.curves.rate_curve import RiskFreeRateCurve
from engine.scr.risk_margin import RiskMarginCalculator
from engine.scr.scr_assumptions import SCRStressAssumptions


@pytest.fixture
def calc() -> RiskMarginCalculator:
    return RiskMarginCalculator(SCRStressAssumptions.sii_standard_formula())


@pytest.fixture
def flat_rfr() -> RiskFreeRateCurve:
    return RiskFreeRateCurve.flat(rate_yr=0.05)


# ---------------------------------------------------------------------------
# RM1: zero SCR → RM = 0
# ---------------------------------------------------------------------------

def test_rm1_zero_scr(calc: RiskMarginCalculator, flat_rfr: RiskFreeRateCurve) -> None:
    rm = calc.compute(scr_t0=0.0, bel_series=[1000.0, 800.0, 600.0], rfr_curve=flat_rfr)
    assert rm == 0.0


# ---------------------------------------------------------------------------
# RM2: zero BEL at t=0 → RM = 0 (guard against division by zero)
# ---------------------------------------------------------------------------

def test_rm2_zero_bel_t0(calc: RiskMarginCalculator, flat_rfr: RiskFreeRateCurve) -> None:
    rm = calc.compute(scr_t0=500.0, bel_series=[0.0, 0.0, 0.0], rfr_curve=flat_rfr)
    assert rm == 0.0


# ---------------------------------------------------------------------------
# RM3: CoC = 0 in assumptions → RM = 0
# ---------------------------------------------------------------------------

def test_rm3_zero_coc(flat_rfr: RiskFreeRateCurve) -> None:
    base = SCRStressAssumptions.sii_standard_formula()
    no_coc = dataclasses.replace(base, cost_of_capital_rate=0.0)
    calc = RiskMarginCalculator(no_coc)
    rm = calc.compute(scr_t0=500.0, bel_series=[1000.0, 500.0], rfr_curve=flat_rfr)
    assert rm == 0.0


# ---------------------------------------------------------------------------
# RM4: single future period numerical anchor
# BEL_1/BEL_0 = 0.5, SCR_0=100, r=0.05, CoC=0.06
# RM = 0.06 × (100 × 0.5) / 1.05 ≈ 2.857
# ---------------------------------------------------------------------------

def test_rm4_numerical_anchor(flat_rfr: RiskFreeRateCurve) -> None:
    base = SCRStressAssumptions.sii_standard_formula()
    calc = RiskMarginCalculator(dataclasses.replace(base, cost_of_capital_rate=0.06))

    bel_series = [1000.0, 500.0]  # BEL_1 / BEL_0 = 0.5
    rm = calc.compute(scr_t0=100.0, bel_series=bel_series, rfr_curve=flat_rfr)

    # r_1yr from flat_rfr: df(12) = 1/(1.05) → r_1yr = 0.05
    # RM = 0.06 × (100 × 0.5) / 1.05 = 0.06 × 47.619... ≈ 2.857
    expected = 0.06 * (100.0 * 0.5) / 1.05
    assert rm == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# RM5: RM decreases as BEL run-off accelerates
# ---------------------------------------------------------------------------

def test_rm5_faster_runoff_lower_rm(flat_rfr: RiskFreeRateCurve) -> None:
    base = SCRStressAssumptions.sii_standard_formula()
    calc = RiskMarginCalculator(base)

    slow_runoff = [1000.0, 950.0, 900.0, 850.0, 800.0]
    fast_runoff = [1000.0, 600.0, 300.0, 100.0,  50.0]

    rm_slow = calc.compute(scr_t0=100.0, bel_series=slow_runoff, rfr_curve=flat_rfr)
    rm_fast = calc.compute(scr_t0=100.0, bel_series=fast_runoff, rfr_curve=flat_rfr)

    assert rm_fast < rm_slow, "faster BEL run-off should produce a lower risk margin"


# ---------------------------------------------------------------------------
# RM6: changing cost_of_capital_rate scales RM proportionally
# ---------------------------------------------------------------------------

def test_rm6_coc_scales_rm_proportionally(flat_rfr: RiskFreeRateCurve) -> None:
    base = SCRStressAssumptions.sii_standard_formula()
    calc_6pct  = RiskMarginCalculator(dataclasses.replace(base, cost_of_capital_rate=0.06))
    calc_12pct = RiskMarginCalculator(dataclasses.replace(base, cost_of_capital_rate=0.12))

    bel = [1000.0, 800.0, 600.0, 400.0, 200.0]
    rm_6  = calc_6pct.compute(scr_t0=500.0,  bel_series=bel, rfr_curve=flat_rfr)
    rm_12 = calc_12pct.compute(scr_t0=500.0, bel_series=bel, rfr_curve=flat_rfr)

    assert rm_12 == pytest.approx(rm_6 * 2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Edge: single-element BEL series → no future periods → RM = 0
# ---------------------------------------------------------------------------

def test_single_period_bel_series(calc: RiskMarginCalculator, flat_rfr: RiskFreeRateCurve) -> None:
    rm = calc.compute(scr_t0=100.0, bel_series=[1000.0], rfr_curve=flat_rfr)
    assert rm == 0.0
