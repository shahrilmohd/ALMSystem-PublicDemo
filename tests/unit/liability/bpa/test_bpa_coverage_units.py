"""
tests/unit/liability/bpa/test_bpa_coverage_units.py

Unit tests for BPACoverageUnitProvider (DECISIONS.md §35).

Test matrix
-----------
Construction:
  1.  Raises ValueError when period_outgos and period_end_times_years differ in length.
  2.  Raises ValueError when locked_in_rate <= -1.
  3.  Accepts locked_in_rate of 0.0 (no discounting).
  4.  Accepts negative outgos without raising.

units_consumed:
  5.  Numerical correctness: flat outgo 100, r=5%, annual periods.
  6.  Returns 0.0 for t < 0 (out-of-bounds low).
  7.  Returns 0.0 for t >= N (out-of-bounds high).

units_remaining:
  8.  units_remaining(0) equals total_coverage_units.
  9.  units_remaining(N) == 0.0 (end of projection).
  10. units_remaining(t+1) == units_remaining(t) - units_consumed(t) for all t.
  11. units_remaining is monotonically non-increasing.
  12. Returns 0.0 for t > N (out-of-bounds).

total_coverage_units:
  13. Equals sum of all units_consumed values.
  14. Zero when all outgos are zero.

Protocol:
  15. Satisfies CoverageUnitProvider protocol (duck-typing check).

Edge cases:
  16. Single-period provider: consumed(0) == remaining(0), remaining(1) == 0.
  17. Fractional year-end times (monthly periods): r=3%, 6-month periods.
  18. Release fraction sums correctly over all periods.
"""
from __future__ import annotations

import math

import pytest

from engine.ifrs17.coverage_units import CoverageUnitProvider
from engine.liability.bpa.coverage_units import BPACoverageUnitProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_annual(outgos: list[float], r: float) -> BPACoverageUnitProvider:
    """Provider with annual periods: t_end[i] = i + 1."""
    n = len(outgos)
    return BPACoverageUnitProvider(
        period_outgos          = outgos,
        locked_in_rate         = r,
        period_end_times_years = [float(i + 1) for i in range(n)],
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            BPACoverageUnitProvider(
                period_outgos          = [100.0, 100.0],
                locked_in_rate         = 0.05,
                period_end_times_years = [1.0],   # wrong length
            )

    def test_locked_in_rate_minus_one_raises(self):
        with pytest.raises(ValueError, match="locked_in_rate"):
            BPACoverageUnitProvider(
                period_outgos          = [100.0],
                locked_in_rate         = -1.0,
                period_end_times_years = [1.0],
            )

    def test_locked_in_rate_below_minus_one_raises(self):
        with pytest.raises(ValueError, match="locked_in_rate"):
            BPACoverageUnitProvider(
                period_outgos          = [100.0],
                locked_in_rate         = -1.5,
                period_end_times_years = [1.0],
            )

    def test_zero_locked_in_rate_accepted(self):
        p = _make_annual([100.0, 100.0], r=0.0)
        # No discounting: consumed = outgo, remaining = reverse cumsum
        assert p.units_consumed(0) == pytest.approx(100.0)
        assert p.units_consumed(1) == pytest.approx(100.0)

    def test_negative_outgo_accepted(self):
        # Negative outgo is unusual but not an error
        p = _make_annual([-50.0, 100.0], r=0.05)
        assert p.units_consumed(0) < 0.0
        assert p.units_consumed(1) > 0.0


# ---------------------------------------------------------------------------
# units_consumed — numerical correctness
# ---------------------------------------------------------------------------

class TestUnitsConsumed:
    """
    Flat outgo = 100, r = 5%, annual periods.
    units_consumed[t] = 100 × (1.05)^(-(t+1))
    """

    @pytest.fixture
    def provider(self) -> BPACoverageUnitProvider:
        return _make_annual([100.0, 100.0, 100.0], r=0.05)

    def test_period_0(self, provider):
        expected = 100.0 / 1.05
        assert provider.units_consumed(0) == pytest.approx(expected, rel=1e-9)

    def test_period_1(self, provider):
        expected = 100.0 / (1.05 ** 2)
        assert provider.units_consumed(1) == pytest.approx(expected, rel=1e-9)

    def test_period_2(self, provider):
        expected = 100.0 / (1.05 ** 3)
        assert provider.units_consumed(2) == pytest.approx(expected, rel=1e-9)

    def test_out_of_bounds_low(self, provider):
        assert provider.units_consumed(-1) == 0.0

    def test_out_of_bounds_high(self, provider):
        assert provider.units_consumed(3) == 0.0
        assert provider.units_consumed(100) == 0.0


# ---------------------------------------------------------------------------
# units_remaining
# ---------------------------------------------------------------------------

class TestUnitsRemaining:
    @pytest.fixture
    def provider(self) -> BPACoverageUnitProvider:
        return _make_annual([100.0, 100.0, 100.0], r=0.05)

    def test_remaining_zero_at_end(self, provider):
        assert provider.units_remaining(3) == pytest.approx(0.0, abs=1e-12)

    def test_remaining_out_of_bounds_high(self, provider):
        assert provider.units_remaining(4) == 0.0

    def test_remaining_equals_total_at_t0(self, provider):
        assert provider.units_remaining(0) == pytest.approx(
            provider.total_coverage_units, rel=1e-12
        )

    def test_remaining_decreases_by_consumed(self, provider):
        for t in range(3):
            drop = provider.units_remaining(t) - provider.units_remaining(t + 1)
            assert drop == pytest.approx(provider.units_consumed(t), rel=1e-9)

    def test_remaining_monotonically_non_increasing(self, provider):
        vals = [provider.units_remaining(t) for t in range(4)]
        for a, b in zip(vals, vals[1:]):
            assert a >= b - 1e-12

    def test_exact_total(self, provider):
        # total = 100/1.05 + 100/1.05² + 100/1.05³
        expected = sum(100.0 / (1.05 ** k) for k in range(1, 4))
        assert provider.total_coverage_units == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# total_coverage_units
# ---------------------------------------------------------------------------

class TestTotalCoverageUnits:
    def test_equals_sum_of_all_consumed(self):
        p = _make_annual([80.0, 120.0, 100.0], r=0.04)
        total_consumed = sum(p.units_consumed(t) for t in range(3))
        assert p.total_coverage_units == pytest.approx(total_consumed, rel=1e-12)

    def test_zero_when_all_outgos_zero(self):
        p = _make_annual([0.0, 0.0, 0.0], r=0.05)
        assert p.total_coverage_units == pytest.approx(0.0, abs=1e-15)

    def test_equals_units_remaining_at_t0(self):
        p = _make_annual([50.0, 50.0], r=0.03)
        assert p.total_coverage_units == pytest.approx(p.units_remaining(0), rel=1e-12)


# ---------------------------------------------------------------------------
# CoverageUnitProvider protocol satisfaction
# ---------------------------------------------------------------------------

class TestProtocol:
    def test_satisfies_coverage_unit_provider_protocol(self):
        p: CoverageUnitProvider = _make_annual([100.0], r=0.05)
        # Protocol requires callable units_consumed and units_remaining
        assert callable(p.units_consumed)
        assert callable(p.units_remaining)

    def test_units_consumed_returns_float(self):
        p = _make_annual([100.0], r=0.05)
        result = p.units_consumed(0)
        assert isinstance(result, float)

    def test_units_remaining_returns_float(self):
        p = _make_annual([100.0], r=0.05)
        result = p.units_remaining(0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_period(self):
        """1 period: consumed(0) = remaining(0), remaining(1) = 0."""
        p = BPACoverageUnitProvider(
            period_outgos          = [200.0],
            locked_in_rate         = 0.05,
            period_end_times_years = [0.5],   # 6-month period
        )
        expected = 200.0 * (1.05 ** -0.5)
        assert p.units_consumed(0) == pytest.approx(expected, rel=1e-9)
        assert p.units_remaining(0) == pytest.approx(expected, rel=1e-9)
        assert p.units_remaining(1) == pytest.approx(0.0, abs=1e-12)

    def test_monthly_periods(self):
        """
        12 monthly periods, flat outgo = 10, r = 3%.
        t_end[i] = (i+1)/12.
        """
        outgos = [10.0] * 12
        t_ends = [(i + 1) / 12.0 for i in range(12)]
        p = BPACoverageUnitProvider(
            period_outgos          = outgos,
            locked_in_rate         = 0.03,
            period_end_times_years = t_ends,
        )
        # Manually verify period 0: 10 × (1.03)^(-1/12)
        expected_0 = 10.0 * (1.03 ** (-1.0 / 12))
        assert p.units_consumed(0) == pytest.approx(expected_0, rel=1e-9)
        # Remaining at t=12 (end) must be 0
        assert p.units_remaining(12) == pytest.approx(0.0, abs=1e-12)

    def test_release_fraction_last_period_is_one(self):
        """At the final period, all remaining units are consumed → fraction = 1."""
        p = _make_annual([100.0, 100.0, 100.0], r=0.05)
        n = 3
        fraction = p.units_consumed(n - 1) / p.units_remaining(n - 1)
        assert fraction == pytest.approx(1.0, rel=1e-9)

    def test_sum_of_consumed_equals_total(self):
        """Σ units_consumed(t) == total_coverage_units."""
        p = _make_annual([60.0, 80.0, 100.0, 40.0], r=0.06)
        total = sum(p.units_consumed(t) for t in range(4))
        assert total == pytest.approx(p.total_coverage_units, rel=1e-12)
