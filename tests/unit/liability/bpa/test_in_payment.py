"""
Unit tests for engine/liability/bpa/in_payment.py

Numerical anchor strategy
--------------------------
All BEL and cashflow anchors use:
  - zero_improvement_basis  (q_x = 0.02 constant, no improvement)
  - RiskFreeRateCurve.flat(0.03)  (3% flat discount)
  - inflation_rate = 0.0          (no LPI uplift, simplifies arithmetic)
  - expense_pa = 0.0              (isolate pension cashflow)
  - weight = 1.0, pension_pa = 1200.0, annual

Under these conditions:
  q_annual = 0.02
  Annual survival probability = 0.98
  BEL for one annual period = 1200 × 0.98 × DF(12m)
    DF(12m) = (1.03)^(-1) = 0.970874...

All expected values are computed from first principles and labelled
with the formula used so they can be independently verified.
"""
import math

import numpy as np
import pandas as pd
import pytest

from engine.core.projection_calendar import ProjectionCalendar
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.base_liability import LiabilityCashflows, Decrements
from engine.liability.bpa.assumptions import BPAAssumptions
from engine.liability.bpa.in_payment import InPaymentLiability, REQUIRED_COLUMNS
from engine.liability.bpa.mortality import MortalityBasis, TABLE_LENGTH


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def annual_calendar():
    """All-annual ProjectionCalendar: 10 periods, dt=1.0 each."""
    return ProjectionCalendar(projection_years=10, monthly_years=0)


@pytest.fixture
def zero_impr_basis():
    flat = np.full(TABLE_LENGTH, 0.02)
    zero = np.zeros(TABLE_LENGTH)
    return MortalityBasis(
        base_table_male=flat.copy(),
        base_table_female=flat.copy(),
        initial_improvement_male=zero.copy(),
        initial_improvement_female=zero.copy(),
        base_year=2023,
        ltr=0.0,
        convergence_period=20,
    )


@pytest.fixture
def assumptions(zero_impr_basis):
    ill = np.zeros(TABLE_LENGTH)
    return BPAAssumptions(
        mortality=zero_impr_basis,
        valuation_year=2023,
        discount_curve=RiskFreeRateCurve.flat(0.03),
        inflation_rate=0.0,
        rpi_rate=0.0,
        tv_rate=0.0,
        ill_health_rates=ill,
        expense_pa=0.0,
    )


@pytest.fixture
def single_mp():
    """One male age 65, weight 1.0, pension £1200 p.a., no LPI, no GMP."""
    return pd.DataFrame([{
        "mp_id":      "P001",
        "sex":        "M",
        "age":        65.0,
        "in_force_count":     1.0,
        "pension_pa": 1200.0,
        "lpi_cap":    0.0,
        "lpi_floor":  0.0,
        "gmp_pa":     0.0,
    }])


@pytest.fixture
def liability(annual_calendar):
    return InPaymentLiability(annual_calendar)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_missing_column_raises(self, liability, assumptions):
        bad = pd.DataFrame([{"mp_id": "X", "sex": "M", "age": 65.0}])
        with pytest.raises(ValueError, match="missing columns"):
            liability.project_cashflows(bad, assumptions, 0)

    def test_wrong_assumptions_type_raises(self, liability, single_mp):
        with pytest.raises(AssertionError):
            liability.project_cashflows(single_mp, object(), 0)

    def test_required_columns_set(self):
        assert "pension_pa" in REQUIRED_COLUMNS
        assert "lpi_cap" in REQUIRED_COLUMNS
        assert "in_force_count" in REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# project_cashflows
# ---------------------------------------------------------------------------

class TestProjectCashflows:

    def test_returns_liability_cashflows(self, liability, single_mp, assumptions):
        result = liability.project_cashflows(single_mp, assumptions, 0)
        assert isinstance(result, LiabilityCashflows)

    def test_timestep_recorded(self, liability, single_mp, assumptions):
        result = liability.project_cashflows(single_mp, assumptions, 3)
        assert result.timestep == 3

    def test_premiums_zero(self, liability, single_mp, assumptions):
        result = liability.project_cashflows(single_mp, assumptions, 0)
        assert result.premiums == 0.0

    def test_death_claims_zero(self, liability, single_mp, assumptions):
        result = liability.project_cashflows(single_mp, assumptions, 0)
        assert result.death_claims == 0.0

    def test_surrender_payments_zero(self, liability, single_mp, assumptions):
        result = liability.project_cashflows(single_mp, assumptions, 0)
        assert result.surrender_payments == 0.0

    def test_pension_outgo_numerical_anchor(self, liability, single_mp, assumptions):
        # Period 0: dt=1.0, t_years=0.0, inflation=0 → pension = 1200 × 1.0 × 1.0 × 1.0
        result = liability.project_cashflows(single_mp, assumptions, 0)
        assert result.maturity_payments == pytest.approx(1200.0, rel=1e-9)

    def test_pension_outgo_second_period(self, liability, single_mp, assumptions):
        # Period 1: dt=1.0, inflation=0 → pension = 1200 × 1.0 × 1.0 × 1.0 (no inflation)
        result = liability.project_cashflows(single_mp, assumptions, 1)
        assert result.maturity_payments == pytest.approx(1200.0, rel=1e-9)

    def test_expense_loaded(self, annual_calendar, zero_impr_basis):
        ill = np.zeros(TABLE_LENGTH)
        assumptions_with_expense = BPAAssumptions(
            mortality=zero_impr_basis,
            valuation_year=2023,
            discount_curve=RiskFreeRateCurve.flat(0.03),
            inflation_rate=0.0,
            rpi_rate=0.0,
            tv_rate=0.0,
            ill_health_rates=ill,
            expense_pa=150.0,
        )
        liability = InPaymentLiability(annual_calendar)
        mp = pd.DataFrame([{
            "mp_id": "P1", "sex": "M", "age": 65.0, "in_force_count": 2.0,
            "pension_pa": 0.0, "lpi_cap": 0.0, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        result = liability.project_cashflows(mp, assumptions_with_expense, 0)
        # expense = 150 × 1.0 (dt) × 2.0 (weight) = 300
        assert result.expenses == pytest.approx(300.0, rel=1e-9)

    def test_lpi_inflation_increases_pension(self, annual_calendar, zero_impr_basis):
        ill = np.zeros(TABLE_LENGTH)
        assumptions_lpi = BPAAssumptions(
            mortality=zero_impr_basis,
            valuation_year=2023,
            discount_curve=RiskFreeRateCurve.flat(0.03),
            inflation_rate=0.03,
            rpi_rate=0.03,
            tv_rate=0.0,
            ill_health_rates=ill,
            expense_pa=0.0,
        )
        liability = InPaymentLiability(annual_calendar)
        mp = pd.DataFrame([{
            "mp_id": "P1", "sex": "M", "age": 65.0, "in_force_count": 1.0,
            "pension_pa": 1200.0, "lpi_cap": 0.05, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        # Period 0: t_years=0 → inflation_idx = (1.03)^0 = 1.0 → pension = 1200
        cf0 = liability.project_cashflows(mp, assumptions_lpi, 0)
        # Period 1: t_years=1 → inflation_idx = 1.03 → pension = 1236
        cf1 = liability.project_cashflows(mp, assumptions_lpi, 1)
        assert cf0.maturity_payments == pytest.approx(1200.0, rel=1e-6)
        assert cf1.maturity_payments == pytest.approx(1200.0 * 1.03, rel=1e-6)

    def test_lpi_cap_applied(self, annual_calendar, zero_impr_basis):
        ill = np.zeros(TABLE_LENGTH)
        # Inflation > cap: pension should grow at cap rate only
        assumptions_highcpi = BPAAssumptions(
            mortality=zero_impr_basis,
            valuation_year=2023,
            discount_curve=RiskFreeRateCurve.flat(0.03),
            inflation_rate=0.08,   # 8% CPI > 5% cap
            rpi_rate=0.08,
            tv_rate=0.0,
            ill_health_rates=ill,
            expense_pa=0.0,
        )
        liability = InPaymentLiability(annual_calendar)
        mp = pd.DataFrame([{
            "mp_id": "P1", "sex": "M", "age": 65.0, "in_force_count": 1.0,
            "pension_pa": 1200.0, "lpi_cap": 0.05, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        cf1 = liability.project_cashflows(mp, assumptions_highcpi, 1)
        # Capped at 5%: period 1 pension = 1200 × (1.05)^1
        assert cf1.maturity_payments == pytest.approx(1200.0 * 1.05, rel=1e-6)

    def test_zero_weight_contributes_nothing(self, liability, assumptions):
        mp = pd.DataFrame([{
            "mp_id": "P1", "sex": "M", "age": 65.0, "in_force_count": 0.0,
            "pension_pa": 1200.0, "lpi_cap": 0.0, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        result = liability.project_cashflows(mp, assumptions, 0)
        assert result.maturity_payments == 0.0


# ---------------------------------------------------------------------------
# get_bel
# ---------------------------------------------------------------------------

class TestGetBel:

    def test_bel_positive_for_in_force_lives(self, liability, single_mp, assumptions):
        bel = liability.get_bel(single_mp, assumptions, 0)
        assert bel > 0.0

    def test_bel_zero_weight_returns_zero(self, liability, assumptions):
        mp = pd.DataFrame([{
            "mp_id": "P1", "sex": "M", "age": 65.0, "in_force_count": 0.0,
            "pension_pa": 1200.0, "lpi_cap": 0.0, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        assert liability.get_bel(mp, assumptions, 0) == 0.0

    def test_bel_at_last_period_is_zero(self, annual_calendar, single_mp, assumptions):
        liability = InPaymentLiability(annual_calendar)
        n = len(annual_calendar.periods)
        bel = liability.get_bel(single_mp, assumptions, n)
        assert bel == 0.0

    def test_bel_one_period_numerical_anchor(self, zero_impr_basis):
        """
        One-period calendar (annual):
          CF(t=1yr) = 1200 × 1.0(inflation) × 1.0(dt) × sp[0]=1.0
          DF(12m)   = (1.03)^-1
          BEL       = 1200 / 1.03
        """
        cal = ProjectionCalendar(projection_years=1, monthly_years=0)
        liability = InPaymentLiability(cal)
        mp = pd.DataFrame([{
            "mp_id": "P1", "sex": "M", "age": 65.0, "in_force_count": 1.0,
            "pension_pa": 1200.0, "lpi_cap": 0.0, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        ill = np.zeros(TABLE_LENGTH)
        assump = BPAAssumptions(
            mortality=zero_impr_basis,
            valuation_year=2023,
            discount_curve=RiskFreeRateCurve.flat(0.03),
            inflation_rate=0.0,
            rpi_rate=0.0,
            tv_rate=0.0,
            ill_health_rates=ill,
            expense_pa=0.0,
        )
        bel = liability.get_bel(mp, assump, 0)
        # sp[0] = 1.0 (survival at START of period 0, before any deaths)
        # Payment at end of year 1: t_payment = 12m
        # BEL = 1200 × 1.0 × 1.0 × 1.0 × (1.03)^-1
        expected = 1200.0 / 1.03
        assert bel == pytest.approx(expected, rel=1e-6)

    def test_bel_two_periods_numerical_anchor(self, zero_impr_basis):
        """
        Two annual periods, q=0.02, r=0.03, no inflation:
          sp[0] = 1.0 (start of period 0)
          sp[1] = 0.98 (start of period 1, after one year of q=0.02)
          BEL = 1200×1.0×DF(12m) + 1200×0.98×DF(24m)
              = 1200/1.03 + 1200×0.98/1.03^2
        """
        cal = ProjectionCalendar(projection_years=2, monthly_years=0)
        liability = InPaymentLiability(cal)
        mp = pd.DataFrame([{
            "mp_id": "P1", "sex": "M", "age": 65.0, "in_force_count": 1.0,
            "pension_pa": 1200.0, "lpi_cap": 0.0, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        ill = np.zeros(TABLE_LENGTH)
        assump = BPAAssumptions(
            mortality=zero_impr_basis,
            valuation_year=2023,
            discount_curve=RiskFreeRateCurve.flat(0.03),
            inflation_rate=0.0,
            rpi_rate=0.0,
            tv_rate=0.0,
            ill_health_rates=ill,
            expense_pa=0.0,
        )
        bel = liability.get_bel(mp, assump, 0)
        expected = 1200.0 / 1.03 + 1200.0 * 0.98 / (1.03 ** 2)
        assert bel == pytest.approx(expected, rel=1e-6)

    def test_bel_decreases_as_timestep_advances(self, liability, single_mp, assumptions):
        bel0 = liability.get_bel(single_mp, assumptions, 0)
        bel1 = liability.get_bel(single_mp, assumptions, 1)
        assert bel0 > bel1

    def test_bel_proportional_to_weight(self, liability, assumptions):
        mp1 = pd.DataFrame([{
            "mp_id": "P1", "sex": "M", "age": 65.0, "in_force_count": 1.0,
            "pension_pa": 1200.0, "lpi_cap": 0.0, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        mp2 = pd.DataFrame([{
            "mp_id": "P2", "sex": "M", "age": 65.0, "in_force_count": 3.0,
            "pension_pa": 1200.0, "lpi_cap": 0.0, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        assert liability.get_bel(mp2, assumptions, 0) == pytest.approx(
            3.0 * liability.get_bel(mp1, assumptions, 0), rel=1e-9
        )

    def test_bel_proportional_to_pension(self, liability, assumptions):
        mp_base = pd.DataFrame([{
            "mp_id": "P1", "sex": "M", "age": 65.0, "in_force_count": 1.0,
            "pension_pa": 1200.0, "lpi_cap": 0.0, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        mp_double = pd.DataFrame([{
            "mp_id": "P2", "sex": "M", "age": 65.0, "in_force_count": 1.0,
            "pension_pa": 2400.0, "lpi_cap": 0.0, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        assert liability.get_bel(mp_double, assumptions, 0) == pytest.approx(
            2.0 * liability.get_bel(mp_base, assumptions, 0), rel=1e-9
        )

    def test_higher_discount_rate_lowers_bel(self, annual_calendar, single_mp, zero_impr_basis):
        ill = np.zeros(TABLE_LENGTH)
        def make_assump(rate):
            return BPAAssumptions(
                mortality=zero_impr_basis,
                valuation_year=2023,
                discount_curve=RiskFreeRateCurve.flat(rate),
                inflation_rate=0.0, rpi_rate=0.0, tv_rate=0.0,
                ill_health_rates=ill, expense_pa=0.0,
            )
        l = InPaymentLiability(annual_calendar)
        bel_low  = l.get_bel(single_mp, make_assump(0.02), 0)
        bel_high = l.get_bel(single_mp, make_assump(0.05), 0)
        assert bel_low > bel_high

    def test_higher_mortality_lowers_bel(self, annual_calendar, single_mp):
        ill = np.zeros(TABLE_LENGTH)
        def make_basis(qx_val):
            arr = np.full(TABLE_LENGTH, qx_val)
            zero = np.zeros(TABLE_LENGTH)
            return MortalityBasis(arr, arr, zero, zero, ltr=0.0)
        def make_assump(qx_val):
            return BPAAssumptions(
                mortality=make_basis(qx_val),
                valuation_year=2023,
                discount_curve=RiskFreeRateCurve.flat(0.03),
                inflation_rate=0.0, rpi_rate=0.0, tv_rate=0.0,
                ill_health_rates=ill, expense_pa=0.0,
            )
        l = InPaymentLiability(annual_calendar)
        bel_low_mort  = l.get_bel(single_mp, make_assump(0.01), 0)
        bel_high_mort = l.get_bel(single_mp, make_assump(0.05), 0)
        assert bel_low_mort > bel_high_mort


# ---------------------------------------------------------------------------
# get_decrements
# ---------------------------------------------------------------------------

class TestGetDecrements:

    def test_returns_decrements(self, liability, single_mp, assumptions):
        result = liability.get_decrements(single_mp, assumptions, 0)
        assert isinstance(result, Decrements)

    def test_in_force_identity(self, liability, single_mp, assumptions):
        result = liability.get_decrements(single_mp, assumptions, 0)
        assert result.in_force_end == pytest.approx(
            result.in_force_start - result.deaths, rel=1e-9
        )

    def test_lapses_and_maturities_zero(self, liability, single_mp, assumptions):
        result = liability.get_decrements(single_mp, assumptions, 0)
        assert result.lapses == 0.0
        assert result.maturities == 0.0

    def test_deaths_numerical_anchor(self, liability, single_mp, assumptions):
        # q_annual=0.02, dt=1.0 → q_period=0.02, deaths = 1.0 × 0.02
        result = liability.get_decrements(single_mp, assumptions, 0)
        assert result.deaths == pytest.approx(0.02, rel=1e-6)

    def test_reserve_equals_bel(self, liability, single_mp, assumptions):
        assert liability.get_reserve(single_mp, assumptions, 0) == pytest.approx(
            liability.get_bel(single_mp, assumptions, 0), rel=1e-9
        )
