"""
Unit tests for engine/liability/bpa/enhanced.py

Key properties under test
--------------------------
1. rating_years = 0  → identical results to InPaymentLiability
2. rating_years > 0  → higher mortality than standard on an age-increasing basis
3. get_bel, project_cashflows, get_decrements all inherited correctly
4. Missing rating_years column is rejected
5. Larger rating → higher mortality → lower BEL

All tests use zero_improvement_basis (q_x constant) and flat 3% discount
so that expected values can be reasoned about simply.
"""
import numpy as np
import pandas as pd
import pytest

from engine.core.projection_calendar import ProjectionCalendar
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.bpa.assumptions import BPAAssumptions
from engine.liability.bpa.enhanced import EnhancedLiability, ENHANCED_REQUIRED_COLUMNS
from engine.liability.bpa.in_payment import InPaymentLiability
from engine.liability.bpa.mortality import MortalityBasis, TABLE_LENGTH, MIN_TABLE_AGE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def annual_calendar():
    return ProjectionCalendar(projection_years=10, monthly_years=0)


@pytest.fixture
def increasing_basis():
    """
    Mortality that increases with age: q_x = 0.001 + (age-16)*0.003, capped at 0.99.
    This ensures that an age-rated lookup (higher effective_age) produces
    materially higher mortality than the actual age.
    """
    arr_qx = np.array(
        [min(0.001 + i * 0.003, 0.99) for i in range(TABLE_LENGTH)], dtype=float
    )
    zero_rf = np.zeros(TABLE_LENGTH)
    return MortalityBasis(
        base_table_male=arr_qx.copy(),
        base_table_female=arr_qx.copy(),
        initial_improvement_male=zero_rf,
        initial_improvement_female=zero_rf,
        ltr=0.0,
    )


@pytest.fixture
def flat_basis():
    """Flat q_x = 0.02 — rating shift has no mortality effect (same q at all ages)."""
    arr = np.full(TABLE_LENGTH, 0.02)
    zero = np.zeros(TABLE_LENGTH)
    return MortalityBasis(arr.copy(), arr.copy(), zero.copy(), zero.copy(), ltr=0.0)


def make_assumptions(basis):
    ill = np.zeros(TABLE_LENGTH)
    return BPAAssumptions(
        mortality=basis,
        valuation_year=2023,
        discount_curve=RiskFreeRateCurve.flat(0.03),
        inflation_rate=0.0,
        rpi_rate=0.0,
        tv_rate=0.0,
        ill_health_rates=ill,
        expense_pa=0.0,
    )


def make_mp(age=65.0, rating_years=0.0, weight=1.0, pension_pa=1200.0):
    return pd.DataFrame([{
        "mp_id":        "E001",
        "sex":          "M",
        "age":          age,
        "in_force_count":       weight,
        "pension_pa":   pension_pa,
        "lpi_cap":      0.0,
        "lpi_floor":    0.0,
        "gmp_pa":       0.0,
        "rating_years": rating_years,
    }])


# ---------------------------------------------------------------------------
# 1. Column validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_missing_rating_years_raises(self, annual_calendar, flat_basis):
        liability = EnhancedLiability(annual_calendar)
        mp_no_rating = pd.DataFrame([{
            "mp_id": "E1", "sex": "M", "age": 65.0, "in_force_count": 1.0,
            "pension_pa": 1200.0, "lpi_cap": 0.0, "lpi_floor": 0.0, "gmp_pa": 0.0,
        }])
        with pytest.raises(ValueError, match="missing columns"):
            liability.get_decrement_rates(0, mp_no_rating)

    def test_enhanced_required_columns_superset_of_in_payment(self):
        from engine.liability.bpa.in_payment import REQUIRED_COLUMNS
        assert REQUIRED_COLUMNS.issubset(ENHANCED_REQUIRED_COLUMNS)
        assert "rating_years" in ENHANCED_REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# 2. rating_years = 0 produces same results as InPaymentLiability
# ---------------------------------------------------------------------------

class TestZeroRatingEquivalence:
    """With rating_years=0, EnhancedLiability must be identical to InPaymentLiability."""

    def test_bel_matches_in_payment(self, annual_calendar, flat_basis):
        assump = make_assumptions(flat_basis)
        mp_enhanced = make_mp(rating_years=0.0)
        mp_standard = mp_enhanced.drop(columns=["rating_years"])

        enhanced = EnhancedLiability(annual_calendar)
        standard = InPaymentLiability(annual_calendar)

        bel_e = enhanced.get_bel(mp_enhanced, assump, 0)
        bel_s = standard.get_bel(mp_standard, assump, 0)
        assert bel_e == pytest.approx(bel_s, rel=1e-9)

    def test_cashflows_match_in_payment(self, annual_calendar, flat_basis):
        assump = make_assumptions(flat_basis)
        mp_enhanced = make_mp(rating_years=0.0)
        mp_standard = mp_enhanced.drop(columns=["rating_years"])

        enhanced = EnhancedLiability(annual_calendar)
        standard = InPaymentLiability(annual_calendar)

        cf_e = enhanced.project_cashflows(mp_enhanced, assump, 0)
        cf_s = standard.project_cashflows(mp_standard, assump, 0)
        assert cf_e.maturity_payments == pytest.approx(cf_s.maturity_payments, rel=1e-9)

    def test_decrements_match_in_payment(self, annual_calendar, flat_basis):
        assump = make_assumptions(flat_basis)
        mp_enhanced = make_mp(rating_years=0.0)
        mp_standard = mp_enhanced.drop(columns=["rating_years"])

        enhanced = EnhancedLiability(annual_calendar)
        standard = InPaymentLiability(annual_calendar)

        dec_e = enhanced.get_decrements(mp_enhanced, assump, 0)
        dec_s = standard.get_decrements(mp_standard, assump, 0)
        assert dec_e.deaths == pytest.approx(dec_s.deaths, rel=1e-9)


# ---------------------------------------------------------------------------
# 3. Positive rating raises mortality and lowers BEL
# ---------------------------------------------------------------------------

class TestPositiveRating:

    def test_rated_mortality_higher_than_standard(self, annual_calendar, increasing_basis):
        """On an age-increasing basis, rating_years=5 must give higher q_death."""
        assump = make_assumptions(increasing_basis)
        liability = EnhancedLiability(annual_calendar)

        mp_std   = make_mp(age=65.0, rating_years=0.0)
        mp_rated = make_mp(age=65.0, rating_years=5.0)

        rates_std   = liability.get_decrement_rates_with_assumptions(0, mp_std,   assump)
        rates_rated = liability.get_decrement_rates_with_assumptions(0, mp_rated, assump)

        assert rates_rated["q_death"].iloc[0] > rates_std["q_death"].iloc[0]

    def test_rated_bel_lower_than_standard(self, annual_calendar, increasing_basis):
        """Higher mortality (shorter expected life) → lower BEL."""
        assump = make_assumptions(increasing_basis)
        liability = EnhancedLiability(annual_calendar)

        bel_std   = liability.get_bel(make_mp(rating_years=0.0), assump, 0)
        bel_rated = liability.get_bel(make_mp(rating_years=5.0), assump, 0)
        assert bel_rated < bel_std

    def test_larger_rating_lowers_bel_further(self, annual_calendar, increasing_basis):
        assump = make_assumptions(increasing_basis)
        liability = EnhancedLiability(annual_calendar)

        bel_5  = liability.get_bel(make_mp(rating_years=5.0),  assump, 0)
        bel_10 = liability.get_bel(make_mp(rating_years=10.0), assump, 0)
        assert bel_10 < bel_5

    def test_mortality_uses_rated_age_numerical_anchor(self, annual_calendar, increasing_basis):
        """
        q_death for enhanced life at rating_years=5 must equal q_death of
        InPaymentLiability at age+5.

        Verifies: effective_age = actual_age + rating_years is the only change.
        """
        assump = make_assumptions(increasing_basis)
        enhanced = EnhancedLiability(annual_calendar)
        standard = InPaymentLiability(annual_calendar)

        mp_enhanced = make_mp(age=65.0, rating_years=5.0)
        # Standard member at the rated age (70) — drop rating_years column
        mp_at_rated_age = make_mp(age=70.0, rating_years=0.0).drop(columns=["rating_years"])

        rates_e = enhanced.get_decrement_rates_with_assumptions(0, mp_enhanced, assump)
        rates_s = standard.get_decrement_rates_with_assumptions(0, mp_at_rated_age, assump)

        assert rates_e["q_death"].iloc[0] == pytest.approx(
            rates_s["q_death"].iloc[0], rel=1e-9
        )

    def test_bel_at_rated_age_equals_standard_bel_at_same_age(
        self, annual_calendar, increasing_basis
    ):
        """
        BEL for enhanced(age=65, rating=5) must equal BEL for standard(age=70).
        Confirms that the rating is a pure age shift with no other effect.
        """
        assump = make_assumptions(increasing_basis)
        enhanced = EnhancedLiability(annual_calendar)
        standard = InPaymentLiability(annual_calendar)

        mp_enhanced = make_mp(age=65.0, rating_years=5.0)
        mp_rated    = make_mp(age=70.0, rating_years=0.0).drop(columns=["rating_years"])

        bel_e = enhanced.get_bel(mp_enhanced, assump, 0)
        bel_s = standard.get_bel(mp_rated,    assump, 0)
        assert bel_e == pytest.approx(bel_s, rel=1e-6)


# ---------------------------------------------------------------------------
# 4. Inherited methods work correctly
# ---------------------------------------------------------------------------

class TestInheritedBehaviour:

    def test_reserve_equals_bel(self, annual_calendar, flat_basis):
        assump = make_assumptions(flat_basis)
        liability = EnhancedLiability(annual_calendar)
        mp = make_mp(rating_years=3.0)
        assert liability.get_reserve(mp, assump, 0) == pytest.approx(
            liability.get_bel(mp, assump, 0), rel=1e-9
        )

    def test_decrements_in_force_identity(self, annual_calendar, flat_basis):
        assump = make_assumptions(flat_basis)
        liability = EnhancedLiability(annual_calendar)
        dec = liability.get_decrements(make_mp(rating_years=2.0), assump, 0)
        assert dec.in_force_end == pytest.approx(
            dec.in_force_start - dec.deaths, rel=1e-9
        )

    def test_lpi_inflation_inherited(self, annual_calendar, flat_basis):
        """LPI uplift inherited from InPaymentLiability — rating_years does not affect it."""
        ill = np.zeros(TABLE_LENGTH)
        assump_lpi = BPAAssumptions(
            mortality=flat_basis,
            valuation_year=2023,
            discount_curve=RiskFreeRateCurve.flat(0.03),
            inflation_rate=0.03,
            rpi_rate=0.03,
            tv_rate=0.0,
            ill_health_rates=ill,
            expense_pa=0.0,
        )
        liability = EnhancedLiability(annual_calendar)
        mp = make_mp(rating_years=5.0)
        # Inject lpi_cap to allow inflation through
        mp["lpi_cap"] = 0.05

        cf0 = liability.project_cashflows(mp, assump_lpi, 0)
        cf1 = liability.project_cashflows(mp, assump_lpi, 1)
        # Period 1 pension = 1200 × (1.03)^1 (3% LPI-capped inflation)
        assert cf1.maturity_payments == pytest.approx(
            cf0.maturity_payments * 1.03, rel=1e-6
        )
