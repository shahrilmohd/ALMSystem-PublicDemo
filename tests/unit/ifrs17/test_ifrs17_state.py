"""
tests/unit/ifrs17/test_ifrs17_state.py

Tests for Ifrs17State — immutability, validation constraints, and field values.
"""
import pytest
from datetime import date

from engine.ifrs17.state import Ifrs17State


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**overrides) -> Ifrs17State:
    defaults = dict(
        cohort_id                = "pensioner",
        valuation_date           = date(2024, 12, 31),
        csm_balance              = 500.0,
        loss_component           = 0.0,
        remaining_coverage_units = 800.0,
        total_coverage_units     = 1000.0,
        locked_in_rate           = 0.05,
        inception_date           = date(2024, 1, 1),
    )
    defaults.update(overrides)
    return Ifrs17State(**defaults)


# ---------------------------------------------------------------------------
# Construction and field access
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_fields_stored_correctly(self):
        s = make_state()
        assert s.cohort_id                == "pensioner"
        assert s.valuation_date           == date(2024, 12, 31)
        assert s.csm_balance              == 500.0
        assert s.loss_component           == 0.0
        assert s.remaining_coverage_units == 800.0
        assert s.total_coverage_units     == 1000.0
        assert s.locked_in_rate           == 0.05
        assert s.inception_date           == date(2024, 1, 1)

    def test_onerous_group_csm_zero(self):
        s = make_state(csm_balance=0.0, loss_component=120.0)
        assert s.loss_component == 120.0
        assert s.csm_balance    == 0.0

    def test_valuation_equals_inception_allowed(self):
        s = make_state(
            valuation_date=date(2024, 1, 1),
            inception_date=date(2024, 1, 1),
        )
        assert s.valuation_date == s.inception_date

    def test_remaining_equals_total_allowed(self):
        # First period — nothing consumed yet
        s = make_state(remaining_coverage_units=1000.0, total_coverage_units=1000.0)
        assert s.remaining_coverage_units == s.total_coverage_units

    def test_remaining_zero_allowed(self):
        # All coverage consumed — end of contract
        s = make_state(remaining_coverage_units=0.0)
        assert s.remaining_coverage_units == 0.0


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

class TestImmutability:
    def test_cannot_set_csm_balance(self):
        s = make_state()
        with pytest.raises(Exception):
            s.csm_balance = 999.0  # type: ignore[misc]

    def test_cannot_set_cohort_id(self):
        s = make_state()
        with pytest.raises(Exception):
            s.cohort_id = "deferred"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation — csm_balance
# ---------------------------------------------------------------------------

class TestCsmBalanceValidation:
    def test_negative_csm_raises(self):
        with pytest.raises(ValueError, match="csm_balance"):
            make_state(csm_balance=-1.0)

    def test_zero_csm_allowed(self):
        s = make_state(csm_balance=0.0)
        assert s.csm_balance == 0.0


# ---------------------------------------------------------------------------
# Validation — loss_component
# ---------------------------------------------------------------------------

class TestLossComponentValidation:
    def test_negative_loss_component_raises(self):
        with pytest.raises(ValueError, match="loss_component"):
            make_state(loss_component=-0.01)

    def test_both_csm_and_loss_nonzero_raises(self):
        with pytest.raises(ValueError, match="both be non-zero"):
            make_state(csm_balance=100.0, loss_component=50.0)


# ---------------------------------------------------------------------------
# Validation — coverage units
# ---------------------------------------------------------------------------

class TestCoverageUnitValidation:
    def test_negative_remaining_raises(self):
        with pytest.raises(ValueError, match="remaining_coverage_units"):
            make_state(remaining_coverage_units=-1.0)

    def test_zero_total_raises(self):
        with pytest.raises(ValueError, match="total_coverage_units"):
            make_state(total_coverage_units=0.0)

    def test_negative_total_raises(self):
        with pytest.raises(ValueError, match="total_coverage_units"):
            make_state(total_coverage_units=-100.0)

    def test_remaining_exceeds_total_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            make_state(remaining_coverage_units=1001.0, total_coverage_units=1000.0)


# ---------------------------------------------------------------------------
# Validation — locked_in_rate
# ---------------------------------------------------------------------------

class TestLockedInRateValidation:
    def test_rate_minus_one_raises(self):
        with pytest.raises(ValueError, match="locked_in_rate"):
            make_state(locked_in_rate=-1.0)

    def test_zero_rate_allowed(self):
        s = make_state(locked_in_rate=0.0)
        assert s.locked_in_rate == 0.0

    def test_negative_rate_above_minus_one_allowed(self):
        # Negative rates are technically valid (ZIRP/NIRP environments)
        s = make_state(locked_in_rate=-0.005)
        assert s.locked_in_rate == -0.005


# ---------------------------------------------------------------------------
# Validation — dates
# ---------------------------------------------------------------------------

class TestDateValidation:
    def test_valuation_before_inception_raises(self):
        with pytest.raises(ValueError, match="valuation_date"):
            make_state(
                valuation_date=date(2023, 12, 31),
                inception_date=date(2024, 1, 1),
            )
