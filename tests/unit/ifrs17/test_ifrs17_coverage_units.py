"""
tests/unit/ifrs17/test_ifrs17_coverage_units.py

Tests for the CoverageUnitProvider protocol.

Three things are tested:
  1. A concrete implementation satisfies the protocol at runtime.
  2. The BPA contract: units_remaining(t) decreases monotonically and
     units_consumed(t) equals the drop in remaining units between periods.
  3. units_remaining reaches 0 at the end of the projection.

These tests use a simple synthetic BPA-style provider where the PV of
future outgo is known analytically — a flat annual outgo of 10, discounted
at 5%, over 3 periods.
"""
import pytest

from engine.ifrs17.coverage_units import CoverageUnitProvider


# ---------------------------------------------------------------------------
# Synthetic implementation
# ---------------------------------------------------------------------------

class FlatOutgoCoverageUnitProvider:
    """
    3-period BPA provider. Outgo = 10 each period, locked_in_rate = 5% p.a.

    PV at locked-in rate 5%:
      Period 0 remaining (t=0..2):
        PV = 10/1.05 + 10/1.05² + 10/1.05³ = 9.524 + 9.070 + 8.638 = 27.232

      Period 1 remaining (t=1..2):
        PV = 10/1.05 + 10/1.05² = 9.524 + 9.070 = 18.594   (re-baselined to t=0 for simplicity)

    For this test we use pre-computed integer-friendly values:
      remaining = [30, 20, 10, 0]   (simplified for test clarity)
      consumed  = [10, 10, 10]
    """

    _remaining = [30.0, 20.0, 10.0, 0.0]
    _consumed  = [10.0, 10.0, 10.0]

    def units_consumed(self, t: int) -> float:
        return self._consumed[t]

    def units_remaining(self, t: int) -> float:
        return self._remaining[t]


@pytest.fixture
def provider() -> FlatOutgoCoverageUnitProvider:
    return FlatOutgoCoverageUnitProvider()


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------

class TestProtocolSatisfaction:
    def test_concrete_class_satisfies_protocol(self, provider):
        # Static type checkers enforce this; at runtime we verify callable
        p: CoverageUnitProvider = provider  # type: ignore[assignment]
        assert callable(p.units_consumed)
        assert callable(p.units_remaining)

    def test_units_consumed_returns_float(self, provider):
        result = provider.units_consumed(0)
        assert isinstance(result, float)

    def test_units_remaining_returns_float(self, provider):
        result = provider.units_remaining(0)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# BPA contract: remaining decreases, consumed = drop in remaining
# ---------------------------------------------------------------------------

class TestBpaContract:
    def test_remaining_decreases_monotonically(self, provider):
        remainders = [provider.units_remaining(t) for t in range(4)]
        for a, b in zip(remainders, remainders[1:]):
            assert a >= b

    def test_consumed_equals_drop_in_remaining(self, provider):
        for t in range(3):
            drop = provider.units_remaining(t) - provider.units_remaining(t + 1)
            assert provider.units_consumed(t) == pytest.approx(drop, rel=1e-10)

    def test_remaining_zero_at_end(self, provider):
        assert provider.units_remaining(3) == 0.0

    def test_consumed_non_negative(self, provider):
        for t in range(3):
            assert provider.units_consumed(t) >= 0.0

    def test_remaining_non_negative(self, provider):
        for t in range(4):
            assert provider.units_remaining(t) >= 0.0

    def test_total_consumed_equals_opening_remaining(self, provider):
        total_consumed = sum(provider.units_consumed(t) for t in range(3))
        assert total_consumed == pytest.approx(provider.units_remaining(0), rel=1e-10)


# ---------------------------------------------------------------------------
# Release fraction derived from provider values
# ---------------------------------------------------------------------------

class TestReleaseFraction:
    def test_release_fraction_period_0(self, provider):
        # consumed(0) / remaining(0) = 10 / 30 = 1/3
        fraction = provider.units_consumed(0) / provider.units_remaining(0)
        assert fraction == pytest.approx(1 / 3, rel=1e-10)

    def test_release_fraction_final_period(self, provider):
        # At period 2, all remaining units are consumed → fraction = 1
        fraction = provider.units_consumed(2) / provider.units_remaining(2)
        assert fraction == pytest.approx(1.0, rel=1e-10)
