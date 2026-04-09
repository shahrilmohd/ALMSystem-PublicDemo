"""
tests/unit/core/test_projection_calendar.py

Numerical tests for ProjectionCalendar and ProjectionPeriod.

All assertions with floating-point arithmetic use pytest.approx with
rel=1e-9 unless the result is expected to be exact (e.g. integer
multiples of 12 months or pure integer annual times).
"""
import pytest

from engine.core.projection_calendar import ProjectionCalendar, ProjectionPeriod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cal_60_10() -> ProjectionCalendar:
    """60-year projection, 10 monthly years — the standard BPA calendar."""
    return ProjectionCalendar(projection_years=60, monthly_years=10)


@pytest.fixture
def cal_5_5() -> ProjectionCalendar:
    """5-year projection, all monthly — edge case: no annual periods."""
    return ProjectionCalendar(projection_years=5, monthly_years=5)


@pytest.fixture
def cal_2_0() -> ProjectionCalendar:
    """2-year projection, no monthly periods — all annual."""
    return ProjectionCalendar(projection_years=2, monthly_years=0)


# ---------------------------------------------------------------------------
# Period counts
# ---------------------------------------------------------------------------

class TestPeriodCounts:
    def test_standard_total(self, cal_60_10):
        # 120 monthly + 50 annual = 170
        assert cal_60_10.n_periods == 170

    def test_standard_monthly_count(self, cal_60_10):
        monthly = [p for p in cal_60_10.periods if p.is_monthly]
        assert len(monthly) == 120

    def test_standard_annual_count(self, cal_60_10):
        annual = [p for p in cal_60_10.periods if not p.is_monthly]
        assert len(annual) == 50

    def test_all_monthly(self, cal_5_5):
        # 5 years × 12 + 0 annual = 60
        assert cal_5_5.n_periods == 60
        assert all(p.is_monthly for p in cal_5_5.periods)

    def test_all_annual(self, cal_2_0):
        # 0 monthly + 2 annual = 2
        assert cal_2_0.n_periods == 2
        assert all(not p.is_monthly for p in cal_2_0.periods)

    def test_period_indices_are_contiguous(self, cal_60_10):
        for i, p in enumerate(cal_60_10.periods):
            assert p.period_index == i


# ---------------------------------------------------------------------------
# Period attributes — monthly block
# ---------------------------------------------------------------------------

class TestMonthlyPeriodAttributes:
    def test_first_period_year_fraction(self, cal_60_10):
        assert cal_60_10.periods[0].year_fraction == pytest.approx(1 / 12, rel=1e-12)

    def test_first_period_is_monthly(self, cal_60_10):
        assert cal_60_10.periods[0].is_monthly is True

    def test_first_period_time_zero(self, cal_60_10):
        assert cal_60_10.periods[0].time_in_years == 0.0

    def test_last_monthly_period_index(self, cal_60_10):
        assert cal_60_10.periods[119].period_index == 119

    def test_last_monthly_period_is_monthly(self, cal_60_10):
        assert cal_60_10.periods[119].is_monthly is True

    def test_last_monthly_period_time(self, cal_60_10):
        # 119 / 12 — not exactly representable but close
        assert cal_60_10.periods[119].time_in_years == pytest.approx(119 / 12, rel=1e-12)

    def test_period_12_time_is_one_year_exact(self, cal_60_10):
        # 12 / 12.0 == 1.0 exactly in IEEE 754
        assert cal_60_10.periods[12].time_in_years == 1.0

    def test_all_monthly_year_fractions_equal(self, cal_60_10):
        monthly = [p for p in cal_60_10.periods if p.is_monthly]
        for p in monthly:
            assert p.year_fraction == pytest.approx(1 / 12, rel=1e-12)


# ---------------------------------------------------------------------------
# Period attributes — annual block
# ---------------------------------------------------------------------------

class TestAnnualPeriodAttributes:
    def test_first_annual_period_index(self, cal_60_10):
        assert cal_60_10.periods[120].period_index == 120

    def test_first_annual_is_not_monthly(self, cal_60_10):
        assert cal_60_10.periods[120].is_monthly is False

    def test_first_annual_year_fraction(self, cal_60_10):
        assert cal_60_10.periods[120].year_fraction == 1.0

    def test_first_annual_time_exactly_ten(self, cal_60_10):
        # This must be exactly 10.0 — integer, no float accumulation
        assert cal_60_10.periods[120].time_in_years == 10.0

    def test_second_annual_time(self, cal_60_10):
        assert cal_60_10.periods[121].time_in_years == 11.0

    def test_last_annual_time(self, cal_60_10):
        # Period 169 is the last (index 120 + 49 = 169), time = 59.0
        assert cal_60_10.periods[169].time_in_years == 59.0

    def test_all_annual_year_fractions_equal_one(self, cal_60_10):
        annual = [p for p in cal_60_10.periods if not p.is_monthly]
        for p in annual:
            assert p.year_fraction == 1.0


# ---------------------------------------------------------------------------
# time_at()
# ---------------------------------------------------------------------------

class TestTimeAt:
    def test_time_at_zero(self, cal_60_10):
        assert cal_60_10.time_at(0) == 0.0

    def test_time_at_monthly_annual_boundary(self, cal_60_10):
        # Period 120 is the first annual period — must be exactly 10.0
        assert cal_60_10.time_at(120) == 10.0

    def test_time_at_end_of_projection(self, cal_60_10):
        # n_periods = 170 → end of final period = 60.0 years
        assert cal_60_10.time_at(cal_60_10.n_periods) == 60.0

    def test_time_at_one_year(self, cal_60_10):
        # Period 12 starts at exactly 1 year
        assert cal_60_10.time_at(12) == 1.0

    def test_time_at_negative_raises(self, cal_60_10):
        with pytest.raises(IndexError):
            cal_60_10.time_at(-1)

    def test_time_at_beyond_end_raises(self, cal_60_10):
        with pytest.raises(IndexError):
            cal_60_10.time_at(cal_60_10.n_periods + 1)


# ---------------------------------------------------------------------------
# discount_factor()
# ---------------------------------------------------------------------------

class TestDiscountFactor:
    def test_period_zero_always_one(self, cal_60_10):
        assert cal_60_10.discount_factor(0, annual_rate=0.05) == 1.0

    def test_zero_rate_always_one(self, cal_60_10):
        assert cal_60_10.discount_factor(24, annual_rate=0.0) == 1.0

    def test_one_year_monthly_exact(self, cal_60_10):
        # 12 monthly periods = 1 year → DF = 1/1.05
        expected = 1.0 / 1.05
        assert cal_60_10.discount_factor(12, annual_rate=0.05) == pytest.approx(
            expected, rel=1e-10
        )

    def test_two_years_monthly(self, cal_60_10):
        expected = 1.0 / (1.05 ** 2)
        assert cal_60_10.discount_factor(24, annual_rate=0.05) == pytest.approx(
            expected, rel=1e-10
        )

    def test_transition_boundary(self, cal_60_10):
        # Period 120 starts at 10.0 years
        expected = 1.0 / (1.05 ** 10)
        assert cal_60_10.discount_factor(120, annual_rate=0.05) == pytest.approx(
            expected, rel=1e-10
        )

    def test_first_annual_period(self, cal_60_10):
        # Period 121 starts at 11.0 years
        expected = 1.0 / (1.05 ** 11)
        assert cal_60_10.discount_factor(121, annual_rate=0.05) == pytest.approx(
            expected, rel=1e-10
        )

    def test_end_of_projection(self, cal_60_10):
        # period_index = n_periods = 170 → t = 60.0 years
        expected = 1.0 / (1.05 ** 60)
        assert cal_60_10.discount_factor(
            cal_60_10.n_periods, annual_rate=0.05
        ) == pytest.approx(expected, rel=1e-10)

    def test_monotonically_decreasing(self, cal_60_10):
        rate = 0.04
        dfs = [cal_60_10.discount_factor(i, rate) for i in range(0, 170, 10)]
        for a, b in zip(dfs, dfs[1:]):
            assert a > b

    def test_invalid_rate_raises(self, cal_60_10):
        with pytest.raises(ValueError):
            cal_60_10.discount_factor(10, annual_rate=-1.0)

    def test_out_of_range_period_raises(self, cal_60_10):
        with pytest.raises(IndexError):
            cal_60_10.discount_factor(999, annual_rate=0.05)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestConstructorValidation:
    def test_zero_projection_years_raises(self):
        with pytest.raises(ValueError, match="projection_years"):
            ProjectionCalendar(projection_years=0)

    def test_negative_projection_years_raises(self):
        with pytest.raises(ValueError, match="projection_years"):
            ProjectionCalendar(projection_years=-5)

    def test_negative_monthly_years_raises(self):
        with pytest.raises(ValueError, match="monthly_years"):
            ProjectionCalendar(projection_years=60, monthly_years=-1)

    def test_monthly_exceeds_projection_raises(self):
        with pytest.raises(ValueError, match="monthly_years"):
            ProjectionCalendar(projection_years=5, monthly_years=10)

    def test_monthly_equals_projection_allowed(self):
        # All monthly — valid
        cal = ProjectionCalendar(projection_years=3, monthly_years=3)
        assert cal.n_periods == 36

    def test_monthly_zero_allowed(self):
        # All annual — valid
        cal = ProjectionCalendar(projection_years=10, monthly_years=0)
        assert cal.n_periods == 10
        assert all(not p.is_monthly for p in cal.periods)


# ---------------------------------------------------------------------------
# ProjectionPeriod is immutable
# ---------------------------------------------------------------------------

class TestProjectionPeriodImmutability:
    def test_frozen_dataclass(self, cal_60_10):
        p = cal_60_10.periods[0]
        with pytest.raises(Exception):
            p.period_index = 99  # type: ignore[misc]
