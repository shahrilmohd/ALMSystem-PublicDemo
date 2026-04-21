"""
Unit tests for engine/liability/bpa/deferred.py

Numerical anchor strategy
--------------------------
All anchors use:
  - Zero improvement MortalityBasis (q_x = 0.02, constant)
  - RiskFreeRateCurve.flat(0.03)
  - inflation = 0.0  (revaluation_type="fixed", floor=0.0)
  - ill_health_rates = 0.0 for all ages
  - early_retirement_rate = 0.0, late_retirement_rate = 0.0  (no ERs)
  - tv_rate = 0.0 by default (varied in TV-specific tests)
  - expense_pa = 0.0

project_cashflows anchor (age=60, nra=65, tv_rate=0.10, weight=1, pension=1000):
----------------------------------------------------------------------------------
At age 60, ERA=55 ≤ 60 < NRA=65 and |60-65|=5 > 0.5 → early_retirement_rate=0.0
q_d_ann = 0.02, q_i_ann = 0.0, q_r_ann = 0.0, q_tv_ann = 0.10 (annual)
Period rates (dt=1.0):
  q_d = 0.02, q_i = 0.0, q_r = 0.0, q_tv = 0.10
  sum = 0.12 ≤ 1.0 → no scaling
tv_amount  = 1000 × 20 = 20,000
TV cashflow = weight × q_tv × tv_amount = 1.0 × 0.10 × 20,000 = 2,000.0
(revaluation_type="fixed", floor=0.0 → annual_rate=0.0 → rev_pension=1000)

get_decrements anchor (same setup):
-------------------------------------
deaths     = 1.0 × 0.02 = 0.02
lapses     = 1.0 × 0.10 = 0.10
maturities = 0.0

BEL anchor (NRA snap, 1-period calendar, tv_rate=0.0, no ill-health):
----------------------------------------------------------------------
age=64.6, nra=65.0 → |64.6-65|=0.4 < 0.5 → snap: q_r_ann=1.0
q_d=0.02, q_i=0, q_r=1.0, q_tv=0
sum = 1.02 > 1.0 → scale = 1/1.02
  q_d_scaled = 0.02/1.02
  q_r_scaled = 1.0/1.02
retiring_weight = 1.0 × q_r_scaled = 1/1.02
tv_weight       = 0

InPaymentLiability.get_bel for period 0 (1-period calendar, age=64.6, pension=1000):
  future_periods = [period 0]: time_in_years=0, year_fraction=1
  sp[0] = 1.0
  inflation_idx = (1+0)^0 = 1 then × (1+0)^1 = 1.0
  cf = 1000 × 1.0 × 1.0 × 1.0 × 1.0 = 1000
  df = discount_factor(12) = 1/1.03
  bel_ip = 1000 / 1.03

BEL_deferred = retiring_weight × bel_ip
             = (1/1.02) × (1000/1.03)
             = 1000 / (1.02 × 1.03)
"""
import numpy as np
import pandas as pd
import pytest

from engine.core.projection_calendar import ProjectionCalendar
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.base_liability import LiabilityCashflows, Decrements
from engine.liability.bpa.assumptions import BPAAssumptions, RetirementRates
from engine.liability.bpa.deferred import DeferredLiability, REQUIRED_COLUMNS
from engine.liability.bpa.mortality import TABLE_LENGTH, MortalityBasis


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def annual_1():
    return ProjectionCalendar(projection_years=1, monthly_years=0)


@pytest.fixture
def annual_10():
    return ProjectionCalendar(projection_years=10, monthly_years=0)


@pytest.fixture
def zero_improvement_basis():
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
    )


def make_assumptions(
    basis,
    discount_rate: float = 0.03,
    tv_rate: float = 0.0,
    early_ret_rate: float = 0.0,
    late_ret_rate: float = 0.0,
):
    ill = np.zeros(TABLE_LENGTH)
    return BPAAssumptions(
        mortality=basis,
        valuation_year=2023,
        discount_curve=RiskFreeRateCurve.flat(discount_rate),
        inflation_rate=0.0,
        rpi_rate=0.0,
        tv_rate=tv_rate,
        ill_health_rates=ill,
        retirement=RetirementRates(
            early_retirement_rate=early_ret_rate,
            late_retirement_rate=late_ret_rate,
        ),
        expense_pa=0.0,
    )


def make_mp(
    age: float = 60.0,
    sex: str = "M",
    weight: float = 1.0,
    deferred_pension_pa: float = 1000.0,
    era: float = 55.0,
    nra: float = 65.0,
    revaluation_type: str = "fixed",
    revaluation_cap: float = 0.05,
    revaluation_floor: float = 0.0,
    deferment_years: float = 5.0,
):
    return pd.DataFrame([{
        "mp_id":               "DEF001",
        "sex":                 sex,
        "age":                 age,
        "in_force_count":              weight,
        "deferred_pension_pa": deferred_pension_pa,
        "era":                 era,
        "nra":                 nra,
        "revaluation_type":    revaluation_type,
        "revaluation_cap":     revaluation_cap,
        "revaluation_floor":   revaluation_floor,
        "deferment_years":     deferment_years,
    }])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_missing_column_raises(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        bad = pd.DataFrame([{"mp_id": "X", "sex": "M", "age": 60.0}])
        with pytest.raises(ValueError, match="missing columns"):
            liability.get_bel(bad, make_assumptions(zero_improvement_basis), 0)

    def test_required_columns_present(self):
        assert "sex"                 in REQUIRED_COLUMNS
        assert "age"                 in REQUIRED_COLUMNS
        assert "deferred_pension_pa" in REQUIRED_COLUMNS
        assert "nra"                 in REQUIRED_COLUMNS
        assert "era"                 in REQUIRED_COLUMNS
        assert "revaluation_type"    in REQUIRED_COLUMNS
        assert "deferment_years"     in REQUIRED_COLUMNS

    def test_validation_in_project_cashflows(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        bad = pd.DataFrame([{"mp_id": "X"}])
        with pytest.raises(ValueError, match="missing columns"):
            liability.project_cashflows(bad, make_assumptions(zero_improvement_basis), 0)

    def test_validation_in_get_decrements(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        bad = pd.DataFrame([{"mp_id": "X"}])
        with pytest.raises(ValueError, match="missing columns"):
            liability.get_decrements(bad, make_assumptions(zero_improvement_basis), 0)


# ---------------------------------------------------------------------------
# project_cashflows
# ---------------------------------------------------------------------------

class TestProjectCashflows:

    def test_returns_liability_cashflows(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        result = liability.project_cashflows(
            make_mp(), make_assumptions(zero_improvement_basis, tv_rate=0.10), 0
        )
        assert isinstance(result, LiabilityCashflows)

    def test_premiums_death_maturity_zero(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        result = liability.project_cashflows(
            make_mp(), make_assumptions(zero_improvement_basis, tv_rate=0.10), 0
        )
        assert result.premiums == 0.0
        assert result.death_claims == 0.0
        assert result.maturity_payments == 0.0

    def test_zero_weight_zero_cashflow(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        result = liability.project_cashflows(
            make_mp(weight=0.0), make_assumptions(zero_improvement_basis, tv_rate=0.10), 0
        )
        assert result.surrender_payments == 0.0

    def test_zero_tv_rate_zero_surrender(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        result = liability.project_cashflows(
            make_mp(), make_assumptions(zero_improvement_basis, tv_rate=0.0), 0
        )
        assert result.surrender_payments == 0.0

    def test_timestep_recorded(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        result = liability.project_cashflows(
            make_mp(), make_assumptions(zero_improvement_basis), 3
        )
        assert result.timestep == 3

    def test_tv_cashflow_numerical_anchor(self, annual_10, zero_improvement_basis):
        """
        age=60, nra=65, era=55, tv_rate=0.10, pension=1000, weight=1, fixed revaluation (0%)
        No retirement rate → q_r=0, q_d=0.02, q_tv=0.10, sum=0.12 ≤ 1 → no scaling.
        tv_amount = 1000 × 20 = 20,000
        TV outgo = 1.0 × 0.10 × 20,000 = 2,000.0
        """
        liability = DeferredLiability(annual_10, tv_annuity_factor=20.0)
        assump = make_assumptions(zero_improvement_basis, tv_rate=0.10)
        result = liability.project_cashflows(make_mp(), assump, 0)
        assert result.surrender_payments == pytest.approx(2000.0, rel=1e-6)

    def test_tv_proportional_to_pension(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10, tv_annuity_factor=20.0)
        assump = make_assumptions(zero_improvement_basis, tv_rate=0.10)
        cf1 = liability.project_cashflows(make_mp(deferred_pension_pa=1000.0), assump, 0)
        cf2 = liability.project_cashflows(make_mp(deferred_pension_pa=2000.0), assump, 0)
        assert cf2.surrender_payments == pytest.approx(2.0 * cf1.surrender_payments, rel=1e-9)

    def test_tv_proportional_to_weight(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        assump = make_assumptions(zero_improvement_basis, tv_rate=0.10)
        cf1 = liability.project_cashflows(make_mp(weight=1.0), assump, 0)
        cf3 = liability.project_cashflows(make_mp(weight=3.0), assump, 0)
        assert cf3.surrender_payments == pytest.approx(3.0 * cf1.surrender_payments, rel=1e-9)

    def test_no_tv_before_era(self, annual_10, zero_improvement_basis):
        """
        TV rate is independent of ERA — TV elections can occur even before ERA.
        This test confirms TV IS non-zero when tv_rate > 0 regardless of age vs ERA.
        """
        liability = DeferredLiability(annual_10)
        # age=40, era=55 → not yet eligible to retire, but TV can still be elected
        assump = make_assumptions(zero_improvement_basis, tv_rate=0.05)
        result = liability.project_cashflows(make_mp(age=40.0, era=55.0), assump, 0)
        assert result.surrender_payments > 0.0


# ---------------------------------------------------------------------------
# get_decrements
# ---------------------------------------------------------------------------

class TestGetDecrements:

    def test_returns_decrements(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        dec = liability.get_decrements(
            make_mp(), make_assumptions(zero_improvement_basis, tv_rate=0.10), 0
        )
        assert isinstance(dec, Decrements)

    def test_in_force_identity(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        dec = liability.get_decrements(
            make_mp(), make_assumptions(zero_improvement_basis, tv_rate=0.05), 0
        )
        assert dec.in_force_end == pytest.approx(
            dec.in_force_start - dec.deaths - dec.lapses - dec.maturities, rel=1e-9
        )

    def test_deaths_numerical_anchor(self, annual_10, zero_improvement_basis):
        """
        q_d=0.02, q_i=0, q_r=0 (no early retirement rate), q_tv=0.10
        sum=0.12 → no scaling → deaths = 1.0 × 0.02 = 0.02
        """
        liability = DeferredLiability(annual_10)
        dec = liability.get_decrements(
            make_mp(), make_assumptions(zero_improvement_basis, tv_rate=0.10), 0
        )
        assert dec.deaths == pytest.approx(0.02, rel=1e-6)

    def test_lapses_numerical_anchor(self, annual_10, zero_improvement_basis):
        """lapses = q_tv = 0.10 (no scaling since sum=0.12)"""
        liability = DeferredLiability(annual_10)
        dec = liability.get_decrements(
            make_mp(), make_assumptions(zero_improvement_basis, tv_rate=0.10), 0
        )
        assert dec.lapses == pytest.approx(0.10, rel=1e-6)

    def test_maturities_zero_without_retirement(self, annual_10, zero_improvement_basis):
        """No early retirement rate → maturities = 0"""
        liability = DeferredLiability(annual_10)
        dec = liability.get_decrements(
            make_mp(), make_assumptions(zero_improvement_basis, tv_rate=0.05), 0
        )
        assert dec.maturities == 0.0

    def test_maturities_nonzero_at_nra(self, annual_1, zero_improvement_basis):
        """
        At NRA snap (age=64.6, nra=65), q_r_annual=1.0, sum > 1 → scaled.
        maturities = weight × q_r_scaled > 0.
        """
        liability = DeferredLiability(annual_1)
        dec = liability.get_decrements(
            make_mp(age=64.6, nra=65.0),
            make_assumptions(zero_improvement_basis),
            0,
        )
        assert dec.maturities > 0.0

    def test_decrement_sum_does_not_exceed_weight(self, annual_10, zero_improvement_basis):
        """deaths + lapses + maturities ≤ weight (scaling must hold)."""
        liability = DeferredLiability(annual_10)
        assump = make_assumptions(
            zero_improvement_basis, tv_rate=0.30,
            early_ret_rate=0.30, late_ret_rate=0.30
        )
        dec = liability.get_decrements(make_mp(age=60.0), assump, 0)
        total_out = dec.deaths + dec.lapses + dec.maturities
        assert total_out <= dec.in_force_start + 1e-12

    def test_timestep_recorded(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        dec = liability.get_decrements(make_mp(), make_assumptions(zero_improvement_basis), 5)
        assert dec.timestep == 5


# ---------------------------------------------------------------------------
# get_bel
# ---------------------------------------------------------------------------

class TestGetBel:

    def test_bel_positive_for_in_force_at_nra(self, annual_1, zero_improvement_basis):
        """At NRA snap all members retire → BEL > 0."""
        liability = DeferredLiability(annual_1)
        bel = liability.get_bel(
            make_mp(age=64.6, nra=65.0),
            make_assumptions(zero_improvement_basis),
            0,
        )
        assert bel > 0.0

    def test_bel_zero_weight_returns_zero(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        bel = liability.get_bel(
            make_mp(weight=0.0),
            make_assumptions(zero_improvement_basis),
            0,
        )
        assert bel == 0.0

    def test_bel_beyond_calendar_returns_zero(self, annual_1, zero_improvement_basis):
        """Timestep past the end of calendar → empty future → BEL = 0."""
        liability = DeferredLiability(annual_1)
        n = len(annual_1.periods)
        bel = liability.get_bel(
            make_mp(age=64.6, nra=65.0),
            make_assumptions(zero_improvement_basis),
            n,
        )
        assert bel == 0.0

    def test_bel_nra_snap_numerical_anchor(self, annual_1, zero_improvement_basis):
        """
        1-period annual calendar. age=64.6, nra=65, era=55, no TV, no ill-health.
        NRA snap → q_r_annual=1.0. q_d_annual=0.02, q_i=0, q_tv=0.
        sum = 1.02 → scale = 1/1.02
          q_d_scaled = 0.02/1.02
          q_r_scaled = 1.0/1.02
        retiring_weight = 1.0 × q_r_scaled = 1/1.02

        InPaymentLiability.get_bel(ret_mp with weight=1, pension=1000, r=0.03, t=0):
          1 future period: cf = 1000 × 1.0 × 1.0 × sp[0=1.0]
          df = 1/1.03
          bel_ip = 1000 / 1.03

        BEL_deferred = retiring_weight × bel_ip
                     = (1/1.02) × (1000/1.03)
                     = 1000 / (1.02 × 1.03)
        """
        liability = DeferredLiability(annual_1, tv_annuity_factor=20.0)
        assump = make_assumptions(zero_improvement_basis, discount_rate=0.03)
        bel = liability.get_bel(
            make_mp(age=64.6, nra=65.0, deferred_pension_pa=1000.0), assump, 0
        )
        expected = 1000.0 / (1.02 * 1.03)
        assert bel == pytest.approx(expected, rel=1e-6)

    def test_bel_proportional_to_weight(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        assump = make_assumptions(zero_improvement_basis)
        bel1 = liability.get_bel(make_mp(age=64.6, nra=65.0, weight=1.0), assump, 0)
        bel3 = liability.get_bel(make_mp(age=64.6, nra=65.0, weight=3.0), assump, 0)
        assert bel3 == pytest.approx(3.0 * bel1, rel=1e-9)

    def test_bel_proportional_to_pension(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        assump = make_assumptions(zero_improvement_basis)
        bel1 = liability.get_bel(
            make_mp(age=64.6, nra=65.0, deferred_pension_pa=1000.0), assump, 0
        )
        bel2 = liability.get_bel(
            make_mp(age=64.6, nra=65.0, deferred_pension_pa=2000.0), assump, 0
        )
        assert bel2 == pytest.approx(2.0 * bel1, rel=1e-9)

    def test_bel_decreases_as_timestep_advances(self, annual_10, zero_improvement_basis):
        """BEL at t=0 should exceed BEL at t=3 (fewer future periods)."""
        liability = DeferredLiability(annual_10)
        assump = make_assumptions(zero_improvement_basis, early_ret_rate=0.05)
        bel0 = liability.get_bel(make_mp(age=60.0), assump, 0)
        bel3 = liability.get_bel(make_mp(age=60.0), assump, 3)
        assert bel0 > bel3

    def test_bel_increases_with_lower_discount_rate(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        bel_lo = liability.get_bel(
            make_mp(age=64.6, nra=65.0),
            make_assumptions(zero_improvement_basis, discount_rate=0.02),
            0,
        )
        bel_hi = liability.get_bel(
            make_mp(age=64.6, nra=65.0),
            make_assumptions(zero_improvement_basis, discount_rate=0.05),
            0,
        )
        assert bel_lo > bel_hi

    def test_bel_zero_before_era(self, annual_10, zero_improvement_basis):
        """
        If NRA is beyond the calendar end and early_ret_rate=0 and tv_rate=0
        and the member never reaches ERA (age + projection < ERA), BEL = 0.
        """
        # 1-year calendar; age=40, era=55 — member cannot retire in projection
        cal = ProjectionCalendar(projection_years=1, monthly_years=0)
        liability = DeferredLiability(cal)
        assump = make_assumptions(zero_improvement_basis, tv_rate=0.0)
        bel = liability.get_bel(
            make_mp(age=40.0, era=55.0, nra=65.0), assump, 0
        )
        assert bel == 0.0

    def test_bel_tv_contribution(self, annual_10, zero_improvement_basis):
        """
        TV elections create positive BEL contribution.
        With tv_rate=0 vs tv_rate=0.05 (and early_ret_rate=0, no NRA snap in 10 yrs):
        age=55, nra=70, era=55. Early retirement available, but early_ret_rate=0.
        BEL with tv_rate>0 should exceed BEL with tv_rate=0 when member can't retire.
        """
        liability = DeferredLiability(annual_10, tv_annuity_factor=20.0)
        bel_no_tv = liability.get_bel(
            make_mp(age=55.0, era=55.0, nra=70.0),
            make_assumptions(zero_improvement_basis, tv_rate=0.0),
            0,
        )
        bel_with_tv = liability.get_bel(
            make_mp(age=55.0, era=55.0, nra=70.0),
            make_assumptions(zero_improvement_basis, tv_rate=0.05),
            0,
        )
        # BEL with TV includes discounted TV outgo in addition to any retirement BEL
        assert bel_with_tv > bel_no_tv

    def test_bel_with_cpi_revaluation_exceeds_fixed(self, annual_10, zero_improvement_basis):
        """Positive CPI revaluation increases the projected pension and hence BEL."""
        def make_mp_rv(rv_type: str, floor: float):
            return pd.DataFrame([{
                "mp_id": "DEF001", "sex": "M", "age": 64.6, "in_force_count": 1.0,
                "deferred_pension_pa": 1000.0,
                "era": 55.0, "nra": 65.0,
                "revaluation_type": rv_type,
                "revaluation_cap": 0.05,
                "revaluation_floor": floor,
                "deferment_years": 5.0,
            }])

        liability = DeferredLiability(annual_10)
        basis_assump = make_assumptions(zero_improvement_basis)
        # CPI inflation_rate = 0.03 via BPAAssumptions.inflation_rate
        import dataclasses
        assump_cpi = dataclasses.replace(basis_assump, inflation_rate=0.03, rpi_rate=0.03)

        bel_fixed = liability.get_bel(make_mp_rv("fixed", 0.0), basis_assump, 0)
        bel_cpi   = liability.get_bel(make_mp_rv("CPI",   0.0), assump_cpi,   0)
        assert bel_cpi > bel_fixed

    def test_reserve_equals_bel(self, annual_10, zero_improvement_basis):
        liability = DeferredLiability(annual_10)
        assump = make_assumptions(zero_improvement_basis)
        mp = make_mp(age=64.6, nra=65.0)
        assert liability.get_reserve(mp, assump, 0) == pytest.approx(
            liability.get_bel(mp, assump, 0), rel=1e-9
        )

    def test_bel_multiple_model_points_additive(self, annual_10, zero_improvement_basis):
        """BEL for two MPs combined = sum of individual BELs."""
        liability = DeferredLiability(annual_10)
        assump = make_assumptions(zero_improvement_basis, early_ret_rate=0.05)
        mp1 = make_mp(age=60.0, weight=1.0, deferred_pension_pa=1000.0)
        mp2 = make_mp(age=63.0, weight=2.0, deferred_pension_pa=1500.0)
        combined = pd.concat([mp1, mp2], ignore_index=True)

        bel1 = liability.get_bel(mp1, assump, 0)
        bel2 = liability.get_bel(mp2, assump, 0)
        bel_combined = liability.get_bel(combined, assump, 0)
        assert bel_combined == pytest.approx(bel1 + bel2, rel=1e-9)
