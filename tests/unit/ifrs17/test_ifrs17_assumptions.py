"""
tests/unit/ifrs17/test_ifrs17_assumptions.py

Tests for LockedInAssumptions and CurrentAssumptions.
AssumptionProvider is a Protocol — no instantiation test needed.
"""
import pytest
from datetime import date

from engine.ifrs17.assumptions import (
    AssumptionProvider,
    CurrentAssumptions,
    LockedInAssumptions,
)


# ---------------------------------------------------------------------------
# LockedInAssumptions
# ---------------------------------------------------------------------------

class TestLockedInAssumptions:
    def test_fields_stored(self):
        a = LockedInAssumptions(
            cohort_id      = "pensioner",
            inception_date = date(2024, 3, 31),
            locked_in_rate = 0.052,
        )
        assert a.cohort_id      == "pensioner"
        assert a.inception_date == date(2024, 3, 31)
        assert a.locked_in_rate == 0.052

    def test_immutable(self):
        a = LockedInAssumptions("x", date(2024, 1, 1), 0.05)
        with pytest.raises(Exception):
            a.locked_in_rate = 0.06  # type: ignore[misc]

    def test_zero_rate_allowed(self):
        a = LockedInAssumptions("x", date(2024, 1, 1), 0.0)
        assert a.locked_in_rate == 0.0

    def test_negative_rate_above_minus_one_allowed(self):
        a = LockedInAssumptions("x", date(2024, 1, 1), -0.005)
        assert a.locked_in_rate == -0.005

    def test_rate_minus_one_raises(self):
        with pytest.raises(ValueError, match="locked_in_rate"):
            LockedInAssumptions("x", date(2024, 1, 1), -1.0)

    def test_rate_below_minus_one_raises(self):
        with pytest.raises(ValueError, match="locked_in_rate"):
            LockedInAssumptions("x", date(2024, 1, 1), -1.5)


# ---------------------------------------------------------------------------
# CurrentAssumptions
# ---------------------------------------------------------------------------

class TestCurrentAssumptions:
    def test_fields_stored(self):
        a = CurrentAssumptions(
            t               = 12,
            current_rate    = 0.045,
            mortality_table = "S3PMA_CMI2023",
            inflation_index = 0.025,
        )
        assert a.t               == 12
        assert a.current_rate    == 0.045
        assert a.mortality_table == "S3PMA_CMI2023"
        assert a.inflation_index == 0.025

    def test_immutable(self):
        a = CurrentAssumptions(0, 0.05, "S3PMA", 0.02)
        with pytest.raises(Exception):
            a.current_rate = 0.06  # type: ignore[misc]

    def test_t_zero_allowed(self):
        a = CurrentAssumptions(t=0, current_rate=0.05, mortality_table="S3", inflation_index=0.02)
        assert a.t == 0

    def test_negative_t_raises(self):
        with pytest.raises(ValueError, match="t must be"):
            CurrentAssumptions(t=-1, current_rate=0.05, mortality_table="S3", inflation_index=0.02)

    def test_current_rate_minus_one_raises(self):
        with pytest.raises(ValueError, match="current_rate"):
            CurrentAssumptions(t=0, current_rate=-1.0, mortality_table="S3", inflation_index=0.02)

    def test_zero_inflation_allowed(self):
        a = CurrentAssumptions(t=0, current_rate=0.05, mortality_table="S3", inflation_index=0.0)
        assert a.inflation_index == 0.0


# ---------------------------------------------------------------------------
# AssumptionProvider is a structural Protocol — verify duck-typed impl works
# ---------------------------------------------------------------------------

class TestAssumptionProviderProtocol:
    def test_concrete_implementation_satisfies_protocol(self):
        """A concrete class with the right methods satisfies the Protocol."""

        class FlatProvider:
            def get_locked_in(self, cohort_id: str) -> LockedInAssumptions:
                return LockedInAssumptions(cohort_id, date(2024, 1, 1), 0.05)

            def get_current(self, t: int) -> CurrentAssumptions:
                return CurrentAssumptions(t, 0.048, "S3PMA_CMI2023", 0.025)

        provider: AssumptionProvider = FlatProvider()  # type: ignore[assignment]
        li = provider.get_locked_in("pensioner")
        cu = provider.get_current(5)
        assert li.locked_in_rate == 0.05
        assert cu.t == 5
