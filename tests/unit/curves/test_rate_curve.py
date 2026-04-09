"""
Unit tests for RiskFreeRateCurve.

Acronym glossary
----------------
DF     Discount Factor — present value of £1 paid at time t; DF(t) ∈ (0, 1].
UFR    Ultimate Forward Rate — the long-run forward rate to which Smith–Wilson
       extrapolation converges.
T_min  Smallest maturity supplied in spot_rates (years).
T_max  Largest maturity supplied in spot_rates (years).
SW     Smith–Wilson extrapolation method (EIOPA-style).
FF     Flat-forward extrapolation method.

Rules under test
----------------
Construction:
  1.  Empty spot_rates raises ValueError.
  2.  Non-positive maturity raises ValueError.
  3.  Negative spot rate raises ValueError.
  4.  Valid construction stores spot_rates, extrapolation, ufr, alpha.

discount_factor:
  5.  At t = 0 months, DF = 1.0.
  6.  At a knot maturity, DF equals the analytically computed value.
  7.  Log-linear interpolation between two knots.
  8.  Below-T_min extrapolation (linear log DF from t = 0 to T_min).
  9.  FF extrapolation beyond T_max.
  10. SW extrapolation converges toward UFR.

flat() class method:
  11. Produces DF = (1 + r)^(−t/12) for a flat annual rate r.
  12. flat(0.0) gives DF = 1.0 for all t.
  13. flat() with negative t_months returns 1.0.
  14. flat() passes kwargs to the constructor.

Monotonicity and sanity:
  15. DF is strictly decreasing for positive rates.
  16. DF is always in (0, 1] for non-negative rates and positive t.
"""
from __future__ import annotations

import math

import pytest

from engine.curves.rate_curve import ExtrapolationMethod, RiskFreeRateCurve


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestRiskFreeRateCurveConstruction:
    def test_empty_spot_rates_raises(self):
        with pytest.raises(ValueError, match="empty"):
            RiskFreeRateCurve(spot_rates={})

    def test_non_positive_maturity_raises(self):
        with pytest.raises(ValueError, match="positive"):
            RiskFreeRateCurve(spot_rates={0.0: 0.03})

    def test_negative_maturity_raises(self):
        with pytest.raises(ValueError, match="positive"):
            RiskFreeRateCurve(spot_rates={-1.0: 0.03})

    def test_negative_spot_rate_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            RiskFreeRateCurve(spot_rates={1.0: -0.01})

    def test_fields_stored(self):
        curve = RiskFreeRateCurve(
            spot_rates={1.0: 0.03, 5.0: 0.04},
            extrapolation=ExtrapolationMethod.SMITH_WILSON,
            ufr=0.045,
            alpha=0.12,
        )
        assert curve.spot_rates == {1.0: 0.03, 5.0: 0.04}
        assert curve.extrapolation == ExtrapolationMethod.SMITH_WILSON
        assert curve.ufr   == pytest.approx(0.045)
        assert curve.alpha == pytest.approx(0.12)


# ---------------------------------------------------------------------------
# discount_factor — basic
# ---------------------------------------------------------------------------

class TestDiscountFactorBasic:
    def test_at_zero_months_df_is_one(self):
        """DF(0) = 1.0 by definition."""
        curve = RiskFreeRateCurve.flat(0.05)
        assert curve.discount_factor(0.0) == pytest.approx(1.0)

    def test_negative_t_months_treated_as_zero(self):
        """Negative t_months → DF = 1.0."""
        curve = RiskFreeRateCurve.flat(0.05)
        assert curve.discount_factor(-6.0) == pytest.approx(1.0)

    def test_at_knot_one_year(self):
        """
        Knot at T = 1 yr, rate = 5%.
        DF(12m) = (1.05)^(−1).
        """
        curve    = RiskFreeRateCurve(spot_rates={1.0: 0.05, 5.0: 0.05})
        expected = 1.05 ** -1.0
        assert curve.discount_factor(12.0) == pytest.approx(expected, rel=1e-9)

    def test_at_knot_two_years(self):
        """
        Knot at T = 2 yr, rate = 4%.
        DF(24m) = (1.04)^(−2).
        """
        curve    = RiskFreeRateCurve(spot_rates={2.0: 0.04, 10.0: 0.05})
        expected = 1.04 ** -2.0
        assert curve.discount_factor(24.0) == pytest.approx(expected, rel=1e-9)

    def test_df_in_unit_interval(self):
        """DF must be in (0, 1] for all positive t."""
        curve = RiskFreeRateCurve(spot_rates={1.0: 0.03, 5.0: 0.04, 10.0: 0.05})
        for t_months in [1, 6, 12, 24, 60, 120, 240]:
            df = curve.discount_factor(t_months)
            assert 0.0 < df <= 1.0, f"DF = {df} out of (0,1] at t = {t_months}m"


# ---------------------------------------------------------------------------
# discount_factor — log-linear interpolation
# ---------------------------------------------------------------------------

class TestDiscountFactorInterpolation:
    def test_log_linear_midpoint(self):
        """
        Two knots: T1 = 1 yr (3%), T2 = 3 yr (5%).
        DF(T1) = 1.03^(−1),  DF(T2) = 1.05^(−3).

        At T = 2 yr (midpoint, weight = 0.5):
            log DF(2) = (log DF(T1) + log DF(T2)) / 2
        """
        df_t1 = 1.03 ** -1.0
        df_t2 = 1.05 ** -3.0
        expected = math.exp((math.log(df_t1) + math.log(df_t2)) / 2.0)

        curve = RiskFreeRateCurve(spot_rates={1.0: 0.03, 3.0: 0.05})
        assert curve.discount_factor(24.0) == pytest.approx(expected, rel=1e-9)

    def test_log_linear_one_third_weight(self):
        """
        Two knots: T1 = 3 yr (3%), T2 = 6 yr (4%).
        At T = 4 yr, weight = (4 − 3) / (6 − 3) = 1/3.

            log DF(4) = log DF(T1) + (1/3) × (log DF(T2) − log DF(T1))
        """
        df_t1    = 1.03 ** -3.0
        df_t2    = 1.04 ** -6.0
        expected = math.exp(math.log(df_t1) + (1 / 3) * (math.log(df_t2) - math.log(df_t1)))

        curve = RiskFreeRateCurve(spot_rates={3.0: 0.03, 6.0: 0.04})
        assert curve.discount_factor(48.0) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# discount_factor — below-T_min extrapolation
# ---------------------------------------------------------------------------

class TestBelowTminExtrapolation:
    def test_half_year_below_tmin(self):
        """
        Single knot at T_min = 1 yr, rate = 5%.
        At t = 6m (0.5 yr):
            log DF(0.5) = (0.5 / 1.0) × log DF(1) = 0.5 × log(1.05^(−1))
            DF(0.5)     = 1.05^(−0.5)
        """
        curve    = RiskFreeRateCurve.flat(0.05)
        expected = 1.05 ** -0.5
        assert curve.discount_factor(6.0) == pytest.approx(expected, rel=1e-9)

    def test_one_month_below_tmin(self):
        """At t = 1m: DF = (1 + r)^(−1/12)."""
        r        = 0.05
        curve    = RiskFreeRateCurve.flat(r)
        expected = (1.0 + r) ** (-1.0 / 12.0)
        assert curve.discount_factor(1.0) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# discount_factor — FF (flat-forward) extrapolation beyond T_max
# ---------------------------------------------------------------------------

class TestFlatForwardExtrapolation:
    def test_ff_single_knot(self):
        """
        Single knot at T_max = 1 yr, rate = 5%.
        FF extends at the continuously-compounded forward rate f = log(1.05),
        giving DF(t) = 1.05^(−t) for all t ≥ 0.
        """
        curve = RiskFreeRateCurve(
            spot_rates={1.0: 0.05},
            extrapolation=ExtrapolationMethod.FLAT_FORWARD,
        )
        for t_yr in [1.5, 2.0, 5.0, 10.0, 30.0]:
            expected = 1.05 ** -t_yr
            assert curve.discount_factor(t_yr * 12) == pytest.approx(
                expected, rel=1e-9
            ), f"FF failed at t = {t_yr} yr"

    def test_ff_two_knots_same_rate(self):
        """
        Two knots at the same rate r.
        FF must produce DF(t) = (1 + r)^(−t) everywhere,
        including beyond T_max.
        """
        r     = 0.04
        curve = RiskFreeRateCurve(
            spot_rates={1.0: r, 10.0: r},
            extrapolation=ExtrapolationMethod.FLAT_FORWARD,
        )
        for t_yr in [0.5, 1.0, 5.0, 10.0, 15.0]:
            expected = (1.0 + r) ** -t_yr
            assert curve.discount_factor(t_yr * 12) == pytest.approx(
                expected, rel=1e-8
            ), f"FF two-knot failed at t = {t_yr} yr"


# ---------------------------------------------------------------------------
# discount_factor — SW (Smith–Wilson) extrapolation beyond T_max
# ---------------------------------------------------------------------------

class TestSmithWilsonExtrapolation:
    def test_sw_converges_to_ufr(self):
        """
        The instantaneous forward rate must converge to the UFR at long
        maturities — this is the defining property of Smith-Wilson.

        Note: the implied SPOT rate converges much more slowly (it is a
        running average over all forward rates since t = 0, including the
        3–3.5% liquid range which drags it down). The correct assertion is
        on the FORWARD rate, not the spot rate.

        At t = 200 yr, the exponential decay term is:
            exp(−alpha × (200 − T_max)) = exp(−0.1 × 180) = exp(−18) ≈ 1.5×10⁻⁸

        So f(200) ≈ ω = log(1 + UFR), and exp(ω) − 1 = UFR exactly.
        We verify the implied annual forward rate is within 0.01% of UFR.
        """
        ufr   = 0.042
        curve = RiskFreeRateCurve(
            spot_rates={1.0: 0.03, 20.0: 0.035},
            extrapolation=ExtrapolationMethod.SMITH_WILSON,
            ufr=ufr,
            alpha=0.1,
        )
        # Estimate instantaneous forward rate at t = 200 yr using a small step.
        t_yr   = 200.0
        dt     = 0.01                         # 0.01 yr forward difference
        df_t   = curve.discount_factor(t_yr * 12)
        df_tdt = curve.discount_factor((t_yr + dt) * 12)
        fwd_cc = -math.log(df_tdt / df_t) / dt    # c.c. forward rate
        implied_annual_fwd = math.exp(fwd_cc) - 1  # convert to annual
        assert implied_annual_fwd == pytest.approx(ufr, abs=0.0001)

    def test_sw_differs_from_ff_beyond_tmax(self):
        """
        SW and FF agree at T_max (same base DF) but diverge beyond it because
        they use different forward rate shapes.
        """
        spot_rates = {1.0: 0.03, 20.0: 0.035}
        ff = RiskFreeRateCurve(spot_rates=spot_rates,
                               extrapolation=ExtrapolationMethod.FLAT_FORWARD)
        sw = RiskFreeRateCurve(spot_rates=spot_rates,
                               extrapolation=ExtrapolationMethod.SMITH_WILSON,
                               ufr=0.042, alpha=0.1)
        # At T_max they share the same DF
        assert ff.discount_factor(20 * 12) == pytest.approx(
            sw.discount_factor(20 * 12), rel=1e-9
        )
        # Beyond T_max they diverge
        assert ff.discount_factor(30 * 12) != pytest.approx(
            sw.discount_factor(30 * 12), rel=1e-4
        )


# ---------------------------------------------------------------------------
# flat() class method
# ---------------------------------------------------------------------------

class TestFlatClassMethod:
    def test_flat_df_at_one_year(self):
        """flat(0.05): DF(12m) = (1.05)^(−1)."""
        curve = RiskFreeRateCurve.flat(0.05)
        assert curve.discount_factor(12.0) == pytest.approx(1.05 ** -1.0, rel=1e-9)

    def test_flat_zero_rate_df_is_one(self):
        """flat(0.0): DF = 1.0 for all non-negative t."""
        curve = RiskFreeRateCurve.flat(0.0)
        for t_months in [0, 1, 6, 12, 60, 120]:
            assert curve.discount_factor(t_months) == pytest.approx(1.0, abs=1e-12)

    @pytest.mark.parametrize("annual_rate", [0.01, 0.03, 0.05, 0.08, 0.10])
    def test_flat_one_month_df_equals_annual_rate_to_twelfth_power(self, annual_rate):
        """flat(r): DF(1m) = (1 + r)^(−1/12) for any rate r."""
        curve    = RiskFreeRateCurve.flat(annual_rate)
        expected = (1.0 + annual_rate) ** (-1.0 / 12.0)
        assert curve.discount_factor(1.0) == pytest.approx(expected, rel=1e-9)

    def test_flat_passes_kwargs_to_constructor(self):
        """flat() must forward keyword arguments such as extrapolation and ufr."""
        curve = RiskFreeRateCurve.flat(
            0.03,
            extrapolation=ExtrapolationMethod.SMITH_WILSON,
            ufr=0.05,
        )
        assert curve.extrapolation == ExtrapolationMethod.SMITH_WILSON
        assert curve.ufr == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

class TestMonotonicity:
    def test_df_strictly_decreasing_for_positive_rates(self):
        """
        For a positive rate structure, DF must be strictly decreasing:
        a pound received later is worth strictly less today.
        """
        curve = RiskFreeRateCurve(spot_rates={1.0: 0.03, 10.0: 0.05, 30.0: 0.045})
        steps = [0, 1, 6, 12, 24, 60, 120, 240, 360]
        dfs   = [curve.discount_factor(t) for t in steps]
        for i in range(len(dfs) - 1):
            assert dfs[i] > dfs[i + 1], (
                f"DF not decreasing: DF({steps[i]}m) = {dfs[i]}, "
                f"DF({steps[i+1]}m) = {dfs[i+1]}"
            )
