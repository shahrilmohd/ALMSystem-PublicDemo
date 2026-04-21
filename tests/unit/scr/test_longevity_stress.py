"""
tests/unit/scr/test_longevity_stress.py

Numerical tests for LongevityStressEngine.

Approach
--------
Real mortality tables are not needed here.  We use a flat synthetic
MortalityBasis (q_x = 0.02, zero improvement) so that q_x values are
known exactly.

The stress scales base_table_male / base_table_female by (1 − factor),
which reduces every projected q(x, t) by exactly (factor × 100)%.

A stub liability model computes a simple annuity BEL using the module-level
q_x() function so that the change in BEL under stressed mortality is
analytically predictable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.bpa.assumptions import BPAAssumptions
from engine.liability.bpa.mortality import MortalityBasis, TABLE_LENGTH, q_x as mortality_q_x
from engine.scr.longevity_stress import (
    LongevityStressEngine,
    _build_stressed_assumptions,
    _build_stressed_mortality,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_mortality() -> MortalityBasis:
    """Zero-improvement flat mortality: q_x = 0.02 constant for all ages/years."""
    flat_qx = np.full(TABLE_LENGTH, 0.02, dtype=float)
    zero_rf = np.zeros(TABLE_LENGTH, dtype=float)
    return MortalityBasis(
        base_table_male=flat_qx.copy(),
        base_table_female=flat_qx.copy(),
        initial_improvement_male=zero_rf.copy(),
        initial_improvement_female=zero_rf.copy(),
        base_year=2023,
        ltr=0.0,
        convergence_period=20,
        ae_ratio_male=1.0,
        ae_ratio_female=1.0,
    )


@pytest.fixture
def base_assumptions(flat_mortality) -> BPAAssumptions:
    return BPAAssumptions.default(flat_mortality)


@pytest.fixture
def engine() -> LongevityStressEngine:
    return LongevityStressEngine(mortality_stress_factor=0.20)


# ---------------------------------------------------------------------------
# Minimal stub liability model for BEL calculation
# ---------------------------------------------------------------------------

class _StubModel:
    """
    Minimal liability model using module-level q_x().

    BEL = weight × Σ_{t=1}^{T} pension × tpx(t) × df(t)

    where tpx(t) = Π_{s=0}^{t-1} (1 − q_x(age+s, sex, year+s, basis))
    and df(t) = discount_curve.discount_factor(t * 12).

    Lower q_x (stressed basis) → higher tpx → higher BEL.
    """

    def __init__(self, age: float = 65.0, pension: float = 1_000.0, years: int = 20):
        self._age     = age
        self._pension = pension
        self._years   = years

    def get_bel(self, mps: pd.DataFrame, assumptions: BPAAssumptions, timestep: int) -> float:
        if mps.empty:
            return 0.0
        total_weight = float(mps["weight"].sum())
        year  = assumptions.valuation_year
        curve = assumptions.discount_curve
        basis = assumptions.mortality

        bel = 0.0
        tpx = 1.0
        for t in range(1, self._years + 1):
            qx = mortality_q_x(
                effective_age=self._age + (t - 1),
                sex="M",
                calendar_year=year + (t - 1),
                basis=basis,
            )
            tpx  *= 1.0 - qx
            df    = curve.discount_factor(t * 12)
            bel  += self._pension * tpx * df

        return total_weight * bel


def _stub_mps(weight: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame([{"weight": weight, "age": 65.0, "sex": "M", "pension": 1_000.0}])


# ---------------------------------------------------------------------------
# Test 1 — empty cohorts → zero SCR
# ---------------------------------------------------------------------------

def test_empty_cohorts_returns_zero(engine, base_assumptions):
    result = engine.compute(
        per_cohort_mps={},
        per_cohort_models={},
        base_bel_series=[10_000.0, 9_000.0, 8_000.0],
        base_assumptions=base_assumptions,
    )
    assert result.scr_longevity   == pytest.approx(0.0)
    assert result.stressed_bel_t0 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 2 — stressed BEL > base BEL (lower mortality → more survivors)
# ---------------------------------------------------------------------------

def test_stressed_bel_exceeds_base(engine, base_assumptions):
    model = _StubModel(age=65.0, pension=1_000.0, years=20)
    mps   = _stub_mps(weight=1.0)

    base_bel = model.get_bel(mps, base_assumptions, timestep=0)
    base_series = [base_bel, base_bel * 0.9, base_bel * 0.8]

    result = engine.compute(
        per_cohort_mps={"cohort_A": mps},
        per_cohort_models={"cohort_A": model},
        base_bel_series=base_series,
        base_assumptions=base_assumptions,
    )
    assert result.stressed_bel_t0 > base_bel


# ---------------------------------------------------------------------------
# Test 3 — scr_longevity = max(stressed_bel_t0 − base_bel_t0, 0)
# ---------------------------------------------------------------------------

def test_scr_longevity_formula(engine, base_assumptions):
    model = _StubModel(age=65.0, pension=1_000.0, years=20)
    mps   = _stub_mps(weight=1.0)

    base_bel    = model.get_bel(mps, base_assumptions, timestep=0)
    base_series = [base_bel]

    result = engine.compute(
        per_cohort_mps={"cohort_A": mps},
        per_cohort_models={"cohort_A": model},
        base_bel_series=base_series,
        base_assumptions=base_assumptions,
    )
    expected_scr = max(0.0, result.stressed_bel_t0 - base_bel)
    assert result.scr_longevity == pytest.approx(expected_scr, rel=1e-9)
    assert result.scr_longevity >= 0.0


# ---------------------------------------------------------------------------
# Test 4 — proportional scaling of stressed BEL series
# ---------------------------------------------------------------------------

def test_stressed_bel_series_proportional(engine, base_assumptions):
    model = _StubModel(age=65.0, pension=1_000.0, years=20)
    mps   = _stub_mps(weight=1.0)

    base_bel_t0 = model.get_bel(mps, base_assumptions, timestep=0)
    base_series = [base_bel_t0, base_bel_t0 * 0.90, base_bel_t0 * 0.80]

    result = engine.compute(
        per_cohort_mps={"cohort_A": mps},
        per_cohort_models={"cohort_A": model},
        base_bel_series=base_series,
        base_assumptions=base_assumptions,
    )
    ratio = result.stressed_bel_t0 / base_bel_t0
    for t, stressed_t in enumerate(result.stressed_bel_series):
        assert stressed_t == pytest.approx(base_series[t] * ratio, rel=1e-9)


# ---------------------------------------------------------------------------
# Test 5 — _build_stressed_mortality scales base tables by (1 − factor)
# ---------------------------------------------------------------------------

def test_build_stressed_mortality_scales_tables(flat_mortality):
    factor  = 0.20
    stressed = _build_stressed_mortality(flat_mortality, factor)

    # Base tables scaled
    np.testing.assert_allclose(
        stressed.base_table_male,
        flat_mortality.base_table_male * (1.0 - factor),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        stressed.base_table_female,
        flat_mortality.base_table_female * (1.0 - factor),
        rtol=1e-9,
    )
    # Other fields unchanged
    assert stressed.ltr               == flat_mortality.ltr
    assert stressed.convergence_period == flat_mortality.convergence_period
    assert stressed.ae_ratio_male     == flat_mortality.ae_ratio_male

    # Projected q_x is reduced by exactly 20%
    base_q     = mortality_q_x(65.0, "M", 2023, flat_mortality)
    stressed_q = mortality_q_x(65.0, "M", 2023, stressed)
    assert stressed_q == pytest.approx(base_q * (1.0 - factor), rel=1e-9)


# ---------------------------------------------------------------------------
# Test 6 — _build_stressed_assumptions preserves all fields except mortality
# ---------------------------------------------------------------------------

def test_build_stressed_assumptions_preserves_fields(base_assumptions, flat_mortality):
    factor   = 0.20
    stressed = _build_stressed_assumptions(base_assumptions, factor)

    assert stressed.valuation_year == base_assumptions.valuation_year
    assert stressed.inflation_rate == base_assumptions.inflation_rate
    assert stressed.discount_curve is base_assumptions.discount_curve
    assert stressed.tv_rate        == base_assumptions.tv_rate
    # Mortality tables are scaled
    np.testing.assert_allclose(
        stressed.mortality.base_table_male,
        base_assumptions.mortality.base_table_male * (1.0 - factor),
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# Test 7 — multiple cohorts: stressed BEL sums across all cohorts
# ---------------------------------------------------------------------------

def test_multiple_cohorts_sum(engine, base_assumptions):
    model_a = _StubModel(age=65.0, pension=1_000.0, years=20)
    model_b = _StubModel(age=70.0, pension=2_000.0, years=15)
    mps_a   = _stub_mps(weight=1.0)
    mps_b   = _stub_mps(weight=1.0)

    base_bel_a  = model_a.get_bel(mps_a, base_assumptions, timestep=0)
    base_bel_b  = model_b.get_bel(mps_b, base_assumptions, timestep=0)
    base_series = [base_bel_a + base_bel_b]

    result = engine.compute(
        per_cohort_mps={"A": mps_a, "B": mps_b},
        per_cohort_models={"A": model_a, "B": model_b},
        base_bel_series=base_series,
        base_assumptions=base_assumptions,
    )
    assert result.scr_longevity > 0.0


# ---------------------------------------------------------------------------
# Test 8 — invalid constructor arguments
# ---------------------------------------------------------------------------

def test_invalid_constructor():
    with pytest.raises(ValueError):
        LongevityStressEngine(mortality_stress_factor=0.0)
    with pytest.raises(ValueError):
        LongevityStressEngine(mortality_stress_factor=1.0)
    with pytest.raises(ValueError):
        LongevityStressEngine(mortality_stress_factor=1.5)
