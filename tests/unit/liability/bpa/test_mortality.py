"""
Unit tests for engine/liability/bpa/mortality.py

Numerical anchors use the flat synthetic basis (q_x=0.02, rf=0.02, LTR=0.01,
convergence_period=20) defined in conftest.py so that expected values can be
hand-verified without reference to licensed CMI tables.

Test groups
-----------
1. MortalityBasis construction — validation catches bad inputs
2. improvement_factor          — convergence formula, boundary years
3. q_x                        — calendar-time projection, AE ratios
4. survival_probs_variable_dt  — shape, values, dt scaling
5. Enhanced-life pattern       — age-shift behaves identically to actual age
"""
import math

import numpy as np
import pytest

from engine.liability.bpa.mortality import (
    MIN_TABLE_AGE,
    MAX_TABLE_AGE,
    TABLE_LENGTH,
    MortalityBasis,
    improvement_factor,
    q_x,
    survival_probs_variable_dt,
)


# ---------------------------------------------------------------------------
# 1. MortalityBasis construction
# ---------------------------------------------------------------------------

class TestMortalityBasisConstruction:

    def test_valid_construction(self, flat_basis):
        assert flat_basis.base_year == 2023
        assert flat_basis.ltr == 0.01
        assert len(flat_basis.base_table_male) == TABLE_LENGTH

    def test_wrong_array_length_raises(self):
        bad = np.full(50, 0.02)
        good = np.full(TABLE_LENGTH, 0.02)
        with pytest.raises(ValueError, match="length"):
            MortalityBasis(
                base_table_male=bad,
                base_table_female=good,
                initial_improvement_male=good,
                initial_improvement_female=good,
            )

    def test_negative_qx_raises(self):
        arr = np.full(TABLE_LENGTH, 0.02)
        bad = arr.copy()
        bad[10] = -0.001
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            MortalityBasis(
                base_table_male=bad,
                base_table_female=arr,
                initial_improvement_male=arr,
                initial_improvement_female=arr,
            )

    def test_ltr_out_of_range_raises(self, flat_basis):
        arr = np.full(TABLE_LENGTH, 0.02)
        with pytest.raises(ValueError, match="ltr"):
            MortalityBasis(
                base_table_male=arr,
                base_table_female=arr,
                initial_improvement_male=arr,
                initial_improvement_female=arr,
                ltr=0.20,   # > 0.10 ceiling
            )

    def test_zero_convergence_period_raises(self):
        arr = np.full(TABLE_LENGTH, 0.02)
        with pytest.raises(ValueError, match="convergence_period"):
            MortalityBasis(
                base_table_male=arr,
                base_table_female=arr,
                initial_improvement_male=arr,
                initial_improvement_female=arr,
                convergence_period=0,
            )

    def test_non_array_raises(self):
        arr = np.full(TABLE_LENGTH, 0.02)
        with pytest.raises(TypeError, match="numpy array"):
            MortalityBasis(
                base_table_male=[0.02] * TABLE_LENGTH,   # list, not ndarray
                base_table_female=arr,
                initial_improvement_male=arr,
                initial_improvement_female=arr,
            )


# ---------------------------------------------------------------------------
# 2. improvement_factor
# ---------------------------------------------------------------------------

class TestImprovementFactor:

    def test_at_base_year_equals_initial_rate(self, flat_basis):
        # At base_year, f = initial_rate regardless of LTR
        f = improvement_factor(65.0, "M", 2023, flat_basis)
        assert f == pytest.approx(0.02, rel=1e-9)

    def test_after_convergence_equals_ltr(self, flat_basis):
        # base_year + convergence_period = 2023 + 20 = 2043
        f = improvement_factor(65.0, "M", 2043, flat_basis)
        assert f == pytest.approx(0.01, rel=1e-9)

    def test_beyond_convergence_stays_at_ltr(self, flat_basis):
        f_at = improvement_factor(65.0, "M", 2043, flat_basis)
        f_beyond = improvement_factor(65.0, "M", 2060, flat_basis)
        assert f_at == pytest.approx(f_beyond, rel=1e-9)

    def test_midpoint_is_average_of_initial_and_ltr(self, flat_basis):
        # At base_year + convergence_period/2 = 2033, weight = 0.5
        f = improvement_factor(65.0, "M", 2033, flat_basis)
        expected = 0.01 + (0.02 - 0.01) * 0.5   # = 0.015
        assert f == pytest.approx(expected, rel=1e-9)

    def test_female_uses_female_initial_rate(self, flat_basis):
        # flat_basis has same initial for M and F; use asymmetric basis
        arr_f = np.full(TABLE_LENGTH, 0.02)
        arr_m = np.full(TABLE_LENGTH, 0.03)
        arr_qx = np.full(TABLE_LENGTH, 0.02)
        basis = MortalityBasis(
            base_table_male=arr_qx,
            base_table_female=arr_qx,
            initial_improvement_male=arr_m,
            initial_improvement_female=arr_f,
            base_year=2023,
            ltr=0.01,
            convergence_period=20,
        )
        assert improvement_factor(65.0, "M", 2023, basis) == pytest.approx(0.03)
        assert improvement_factor(65.0, "F", 2023, basis) == pytest.approx(0.02)

    def test_age_clamped_at_max(self, flat_basis):
        # Age 130 should clamp to 120 — no error
        f = improvement_factor(130.0, "M", 2030, flat_basis)
        assert 0.0 <= f <= 1.0

    def test_age_clamped_at_min(self, flat_basis):
        f = improvement_factor(0.0, "F", 2030, flat_basis)
        assert 0.0 <= f <= 1.0


# ---------------------------------------------------------------------------
# 3. q_x
# ---------------------------------------------------------------------------

class TestQx:

    def test_at_base_year_returns_base_table_value(self, flat_basis):
        # No improvement applied at base_year
        result = q_x(65.0, "M", 2023, flat_basis)
        assert result == pytest.approx(0.02, rel=1e-9)

    def test_improves_over_time(self, flat_basis):
        q_now = q_x(65.0, "M", 2023, flat_basis)
        q_future = q_x(65.0, "M", 2030, flat_basis)
        assert q_future < q_now

    def test_calendar_year_before_base_returns_base_value(self, flat_basis):
        result = q_x(65.0, "M", 2020, flat_basis)
        assert result == pytest.approx(0.02, rel=1e-9)

    def test_one_year_improvement_numerical_anchor(self, flat_basis):
        # In 2024 (1 year after base_year=2023), convergence weight = (20-1)/20 = 0.95
        # f(2024) = LTR + (initial - LTR) × 0.95 = 0.01 + 0.01 × 0.95 = 0.0195
        # q(65, 2024) = 0.02 × (1 - 0.0195) = 0.019610
        f_2024 = 0.01 + (0.02 - 0.01) * (19 / 20)
        result = q_x(65.0, "M", 2024, flat_basis)
        assert result == pytest.approx(0.02 * (1 - f_2024), rel=1e-9)

    def test_ten_year_improvement_numerical_anchor(self, flat_basis):
        # After 10 years, improvement factors converge linearly.
        # Compute expected product manually.
        basis = flat_basis
        product = 1.0
        for yr in range(2024, 2034):
            years_elapsed = yr - 2023
            weight = max(0.0, (20 - years_elapsed) / 20)
            f = 0.01 + (0.02 - 0.01) * weight
            product *= (1.0 - f)
        expected = 0.02 * product
        assert q_x(65.0, "M", 2033, flat_basis) == pytest.approx(expected, rel=1e-9)

    def test_ae_ratio_scales_result(self):
        arr = np.full(TABLE_LENGTH, 0.02)
        basis_90 = MortalityBasis(
            base_table_male=arr,
            base_table_female=arr,
            initial_improvement_male=arr,
            initial_improvement_female=arr,
            ae_ratio_male=0.90,
        )
        result = q_x(65.0, "M", 2023, basis_90)
        assert result == pytest.approx(0.02 * 0.90, rel=1e-9)

    def test_ae_ratio_female_independent(self):
        arr = np.full(TABLE_LENGTH, 0.02)
        basis = MortalityBasis(
            base_table_male=arr,
            base_table_female=arr,
            initial_improvement_male=arr,
            initial_improvement_female=arr,
            ae_ratio_male=1.0,
            ae_ratio_female=0.85,
        )
        assert q_x(65.0, "M", 2023, basis) == pytest.approx(0.02)
        assert q_x(65.0, "F", 2023, basis) == pytest.approx(0.02 * 0.85)

    def test_result_clipped_to_zero_one(self):
        # AE ratio > 1 and high base rate should not exceed 1.0
        arr_high = np.full(TABLE_LENGTH, 0.95)
        arr_rf = np.full(TABLE_LENGTH, 0.02)
        basis = MortalityBasis(
            base_table_male=arr_high,
            base_table_female=arr_high,
            initial_improvement_male=arr_rf,
            initial_improvement_female=arr_rf,
            ae_ratio_male=1.20,
        )
        result = q_x(65.0, "M", 2023, basis)
        assert result <= 1.0

    def test_zero_improvement_no_change(self, zero_improvement_basis):
        q_now = q_x(65.0, "M", 2023, zero_improvement_basis)
        q_far = q_x(65.0, "M", 2060, zero_improvement_basis)
        assert q_now == pytest.approx(q_far, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. survival_probs_variable_dt
# ---------------------------------------------------------------------------

class TestSurvivalProbs:

    def test_first_element_always_one(self, flat_basis):
        sp = survival_probs_variable_dt(65.0, "M", np.full(12, 1/12), 2023, flat_basis)
        assert sp[0] == pytest.approx(1.0, rel=1e-9)

    def test_length_is_n_periods_plus_one(self, flat_basis):
        sp = survival_probs_variable_dt(65.0, "M", np.full(24, 1/12), 2023, flat_basis)
        assert len(sp) == 25

    def test_monotonically_decreasing(self, flat_basis):
        sp = survival_probs_variable_dt(65.0, "M", np.full(12, 1/12), 2023, flat_basis)
        assert np.all(np.diff(sp) <= 0.0)

    def test_annual_dt_single_period_numerical_anchor(self, zero_improvement_basis):
        # With zero improvement, q_x = 0.02 always.
        # Annual dt=1.0: q_period = 0.02, survival after 1 year = 0.98.
        sp = survival_probs_variable_dt(65.0, "M", np.array([1.0]), 2023, zero_improvement_basis)
        assert sp[1] == pytest.approx(0.98, rel=1e-9)

    def test_monthly_dt_twelve_periods_approx_annual(self, zero_improvement_basis):
        # 12 monthly steps with q_period = 1-(1-0.02)^(1/12) each.
        # Should closely approximate annual survival of 0.98.
        sp_monthly = survival_probs_variable_dt(65.0, "M", np.full(12, 1/12), 2023, zero_improvement_basis)
        sp_annual  = survival_probs_variable_dt(65.0, "M", np.array([1.0]),    2023, zero_improvement_basis)
        # Monthly compounding of (1-0.02)^(1/12) twelve times = (1-0.02)^1
        assert sp_monthly[12] == pytest.approx(sp_annual[1], rel=1e-6)

    def test_zero_periods_returns_single_one(self, flat_basis):
        sp = survival_probs_variable_dt(65.0, "M", np.array([]), 2023, flat_basis)
        assert len(sp) == 1
        assert sp[0] == pytest.approx(1.0)

    def test_high_age_clamped_no_error(self, flat_basis):
        # Age 115 with 10-year projection → effective ages up to 125, clamped
        sp = survival_probs_variable_dt(115.0, "M", np.full(10, 1.0), 2023, flat_basis)
        assert len(sp) == 11
        assert np.all(sp >= 0.0)
        assert np.all(sp <= 1.0)

    def test_hybrid_calendar_matches_equivalent_uniform(self, zero_improvement_basis):
        # 6 monthly + 1 annual period = 18 months total.
        # Should give identical survival to 18 monthly periods over same duration.
        dt_hybrid  = np.array([1/12] * 6 + [1.0])
        dt_monthly = np.full(18, 1/12)
        sp_hybrid  = survival_probs_variable_dt(65.0, "M", dt_hybrid,  2023, zero_improvement_basis)
        sp_monthly = survival_probs_variable_dt(65.0, "M", dt_monthly, 2023, zero_improvement_basis)
        assert sp_hybrid[-1] == pytest.approx(sp_monthly[-1], rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Enhanced-life pattern (age shift)
# ---------------------------------------------------------------------------

class TestEnhancedLifePattern:
    """
    Enhanced lives pass effective_age = actual_age + rating_years.
    These tests verify that the functions behave identically whether
    the caller shifts the age or whether we directly call with the
    rated age — there is no separate code path.
    """

    def test_qx_with_shift_matches_direct_lookup(self, flat_basis):
        # q_x(65+5, ...) must equal q_x(70, ...)
        assert q_x(70.0, "M", 2023, flat_basis) == q_x(65.0 + 5.0, "M", 2023, flat_basis)

    def test_improvement_factor_with_shift(self, flat_basis):
        f_shifted = improvement_factor(65.0 + 5.0, "M", 2030, flat_basis)
        f_direct  = improvement_factor(70.0,        "M", 2030, flat_basis)
        assert f_shifted == pytest.approx(f_direct, rel=1e-9)

    def test_survival_probs_with_age_rating(self, flat_basis):
        # flat basis: q_x is same for all ages, so survival with any shift
        # is identical — confirming that the shift does not break anything.
        dt_arr = np.full(12, 1/12)
        sp_base     = survival_probs_variable_dt(65.0,       "M", dt_arr, 2023, flat_basis)
        sp_enhanced = survival_probs_variable_dt(65.0 + 5.0, "M", dt_arr, 2023, flat_basis)
        # flat basis → same q_x at all ages → identical survival curves
        np.testing.assert_allclose(sp_base, sp_enhanced, rtol=1e-9)

    def test_age_rating_increases_mortality_on_realistic_basis(self):
        # Build an increasing-with-age basis to confirm that the shift
        # actually produces higher mortality for enhanced lives.
        arr_rf = np.full(TABLE_LENGTH, 0.01)
        # qx increases linearly with age: 0.001 at 16, 0.001 + (age-16)*0.005 capped at 1
        qx_vals = np.array([min(0.001 + (i * 0.005), 0.999) for i in range(TABLE_LENGTH)])
        basis = MortalityBasis(
            base_table_male=qx_vals,
            base_table_female=qx_vals,
            initial_improvement_male=arr_rf,
            initial_improvement_female=arr_rf,
            ltr=0.01,
        )
        q_standard = q_x(65.0, "M", 2023, basis)
        q_enhanced = q_x(65.0 + 5.0, "M", 2023, basis)   # rated age 70
        assert q_enhanced > q_standard
