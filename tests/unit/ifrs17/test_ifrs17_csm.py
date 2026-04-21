"""
tests/unit/ifrs17/test_ifrs17_csm.py

Numerical tests for CsmTracker.

All expected values are computed by hand and commented inline so they can
be independently verified by an actuary without running the code.
"""
import pytest

from engine.ifrs17.csm import CsmTracker, CsmStepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tracker(opening_csm: float = 100.0, rate: float = 0.05) -> CsmTracker:
    return CsmTracker(opening_csm=opening_csm, locked_in_rate=rate)


# ---------------------------------------------------------------------------
# Two-period annual contract — the canonical test case from the plan
#
#   CSM_0 = 100, locked_in_rate = 5% p.a., year_fraction = 1.0
#   total CU = 100  (units_remaining_opening values: 100, 40)
#
#   Period 1: consumed=60, remaining_opening=100
#     accretion  = 100 × 0.05 × 1.0       = 5.00
#     after_acc  = 105.00
#     adjustment = 0  →  after_adj = 105.00
#     release    = 105.00 × (60/100)       = 63.00
#     closing    = 105.00 × (40/100)       = 42.00
#
#   Period 2: consumed=40, remaining_opening=40
#     accretion  = 42.00 × 0.05 × 1.0     = 2.10
#     after_acc  = 44.10
#     release    = 44.10 × (40/40)         = 44.10
#     closing    = 0.00
# ---------------------------------------------------------------------------

class TestTwoPeriodAnnualContract:
    @pytest.fixture
    def tracker(self) -> CsmTracker:
        return make_tracker(opening_csm=100.0, rate=0.05)

    def test_period1_accretion(self, tracker):
        r = tracker.step(units_consumed=60, units_remaining_opening=100, year_fraction=1.0)
        assert r.csm_accretion == pytest.approx(5.0, rel=1e-10)

    def test_period1_release(self, tracker):
        r = tracker.step(units_consumed=60, units_remaining_opening=100, year_fraction=1.0)
        assert r.csm_release == pytest.approx(63.0, rel=1e-10)

    def test_period1_closing(self, tracker):
        r = tracker.step(units_consumed=60, units_remaining_opening=100, year_fraction=1.0)
        assert r.csm_closing == pytest.approx(42.0, rel=1e-10)

    def test_period1_no_adjustment(self, tracker):
        r = tracker.step(units_consumed=60, units_remaining_opening=100, year_fraction=1.0)
        assert r.csm_adjustment_non_financial == pytest.approx(0.0, abs=1e-12)

    def test_period1_no_onerous_excess(self, tracker):
        r = tracker.step(units_consumed=60, units_remaining_opening=100, year_fraction=1.0)
        assert r.onerous_excess == 0.0

    def test_period2_accretion(self, tracker):
        tracker.step(units_consumed=60, units_remaining_opening=100, year_fraction=1.0)
        r = tracker.step(units_consumed=40, units_remaining_opening=40, year_fraction=1.0)
        assert r.csm_accretion == pytest.approx(2.1, rel=1e-10)

    def test_period2_release_equals_full_balance(self, tracker):
        tracker.step(units_consumed=60, units_remaining_opening=100, year_fraction=1.0)
        r = tracker.step(units_consumed=40, units_remaining_opening=40, year_fraction=1.0)
        assert r.csm_release == pytest.approx(44.1, rel=1e-10)

    def test_period2_closing_is_zero(self, tracker):
        tracker.step(units_consumed=60, units_remaining_opening=100, year_fraction=1.0)
        r = tracker.step(units_consumed=40, units_remaining_opening=40, year_fraction=1.0)
        assert r.csm_closing == pytest.approx(0.0, abs=1e-10)

    def test_balance_updates_after_each_step(self, tracker):
        assert tracker.balance == pytest.approx(100.0)
        tracker.step(units_consumed=60, units_remaining_opening=100, year_fraction=1.0)
        assert tracker.balance == pytest.approx(42.0, rel=1e-10)
        tracker.step(units_consumed=40, units_remaining_opening=40, year_fraction=1.0)
        assert tracker.balance == pytest.approx(0.0, abs=1e-10)

    def test_opening_field_matches_prior_closing(self, tracker):
        r1 = tracker.step(units_consumed=60, units_remaining_opening=100, year_fraction=1.0)
        r2 = tracker.step(units_consumed=40, units_remaining_opening=40, year_fraction=1.0)
        assert r2.csm_opening == pytest.approx(r1.csm_closing, rel=1e-10)


# ---------------------------------------------------------------------------
# Monthly period: year_fraction = 1/12
#
#   CSM_0 = 120, rate = 6% p.a., year_fraction = 1/12
#   units_remaining_opening = 60, consumed = 5
#
#   accretion  = 120 × 0.06 × (1/12) = 0.60
#   after_acc  = 120.60
#   release    = 120.60 × (5/60)     = 10.05
#   closing    = 120.60 × (55/60)    = 110.55
# ---------------------------------------------------------------------------

class TestMonthlyPeriod:
    def test_monthly_accretion(self):
        tracker = make_tracker(opening_csm=120.0, rate=0.06)
        r = tracker.step(
            units_consumed=5, units_remaining_opening=60, year_fraction=1 / 12
        )
        assert r.csm_accretion == pytest.approx(120.0 * 0.06 / 12, rel=1e-10)

    def test_monthly_release(self):
        tracker = make_tracker(opening_csm=120.0, rate=0.06)
        r = tracker.step(
            units_consumed=5, units_remaining_opening=60, year_fraction=1 / 12
        )
        expected_release = (120.0 + 120.0 * 0.06 / 12) * (5 / 60)
        assert r.csm_release == pytest.approx(expected_release, rel=1e-10)

    def test_monthly_closing(self):
        tracker = make_tracker(opening_csm=120.0, rate=0.06)
        r = tracker.step(
            units_consumed=5, units_remaining_opening=60, year_fraction=1 / 12
        )
        expected_closing = (120.0 + 120.0 * 0.06 / 12) * (55 / 60)
        assert r.csm_closing == pytest.approx(expected_closing, rel=1e-10)


# ---------------------------------------------------------------------------
# Non-financial assumption change — CSM adjustment
#
#   CSM_0 = 100, rate = 5%, year_fraction = 1.0
#   fcf_change_non_financial = +20 (FCF worsened by 20 → reduces CSM)
#
#   accretion       = 100 × 0.05 = 5.0
#   after_acc       = 105.0
#   after_adj       = 105.0 − 20  = 85.0
#   consumed=50, remaining=100
#   release         = 85.0 × (50/100) = 42.5
#   closing         = 85.0 × (50/100) = 42.5
# ---------------------------------------------------------------------------

class TestNonFinancialAdjustment:
    def test_positive_fcf_change_reduces_csm(self):
        tracker = make_tracker(opening_csm=100.0, rate=0.05)
        r = tracker.step(
            units_consumed=50,
            units_remaining_opening=100,
            year_fraction=1.0,
            fcf_change_non_financial=20.0,
        )
        assert r.csm_release == pytest.approx(42.5, rel=1e-10)
        assert r.csm_closing == pytest.approx(42.5, rel=1e-10)

    def test_negative_fcf_change_increases_csm(self):
        # FCF improved by 10 → CSM increases by 10
        # after_adj = 105 + 10 = 115; release = 115 × 0.5 = 57.5
        tracker = make_tracker(opening_csm=100.0, rate=0.05)
        r = tracker.step(
            units_consumed=50,
            units_remaining_opening=100,
            year_fraction=1.0,
            fcf_change_non_financial=-10.0,
        )
        assert r.csm_release == pytest.approx(57.5, rel=1e-10)

    def test_adjustment_field_sign_convention(self):
        # fcf_change_non_financial = +20 (worse) → field = -20 (CSM reduced by 20)
        tracker = make_tracker(opening_csm=100.0, rate=0.05)
        r = tracker.step(
            units_consumed=50,
            units_remaining_opening=100,
            year_fraction=1.0,
            fcf_change_non_financial=20.0,
        )
        assert r.csm_adjustment_non_financial == pytest.approx(-20.0, rel=1e-10)


# ---------------------------------------------------------------------------
# Onerous group: FCF increase exceeds available CSM
#
#   CSM_0 = 50, rate = 5%, year_fraction = 1.0
#   fcf_change_non_financial = +80  (FCF worsened by 80)
#
#   after_acc      = 50 × 1.05      = 52.5
#   attempted_adj  = 52.5 − 80      = −27.5  → floor at 0
#   onerous_excess = 27.5
#   release        = 0 (CSM = 0)
#   closing        = 0
# ---------------------------------------------------------------------------

class TestOnerousGroup:
    def test_onerous_excess_reported(self):
        tracker = make_tracker(opening_csm=50.0, rate=0.05)
        r = tracker.step(
            units_consumed=50,
            units_remaining_opening=100,
            year_fraction=1.0,
            fcf_change_non_financial=80.0,
        )
        assert r.onerous_excess == pytest.approx(27.5, rel=1e-10)

    def test_csm_closing_floored_at_zero(self):
        tracker = make_tracker(opening_csm=50.0, rate=0.05)
        r = tracker.step(
            units_consumed=50,
            units_remaining_opening=100,
            year_fraction=1.0,
            fcf_change_non_financial=80.0,
        )
        assert r.csm_closing == pytest.approx(0.0, abs=1e-10)

    def test_no_release_when_onerous(self):
        tracker = make_tracker(opening_csm=50.0, rate=0.05)
        r = tracker.step(
            units_consumed=50,
            units_remaining_opening=100,
            year_fraction=1.0,
            fcf_change_non_financial=80.0,
        )
        assert r.csm_release == pytest.approx(0.0, abs=1e-10)

    def test_balance_zero_after_onerous(self):
        tracker = make_tracker(opening_csm=50.0, rate=0.05)
        tracker.step(
            units_consumed=50,
            units_remaining_opening=100,
            year_fraction=1.0,
            fcf_change_non_financial=80.0,
        )
        assert tracker.balance == pytest.approx(0.0, abs=1e-10)

    def test_zero_opening_csm_no_accretion_no_release(self):
        tracker = make_tracker(opening_csm=0.0, rate=0.05)
        r = tracker.step(units_consumed=10, units_remaining_opening=100, year_fraction=1.0)
        assert r.csm_opening   == 0.0
        assert r.csm_accretion == 0.0
        assert r.csm_release   == 0.0
        assert r.csm_closing   == 0.0


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_opening_csm_raises(self):
        with pytest.raises(ValueError, match="opening_csm"):
            CsmTracker(opening_csm=-1.0, locked_in_rate=0.05)

    def test_locked_in_rate_minus_one_raises(self):
        with pytest.raises(ValueError, match="locked_in_rate"):
            CsmTracker(opening_csm=100.0, locked_in_rate=-1.0)

    def test_zero_units_remaining_raises(self):
        tracker = make_tracker()
        with pytest.raises(ValueError, match="units_remaining_opening"):
            tracker.step(units_consumed=10, units_remaining_opening=0.0, year_fraction=1.0)

    def test_negative_units_consumed_raises(self):
        tracker = make_tracker()
        with pytest.raises(ValueError, match="units_consumed"):
            tracker.step(units_consumed=-1, units_remaining_opening=100, year_fraction=1.0)

    def test_zero_year_fraction_raises(self):
        tracker = make_tracker()
        with pytest.raises(ValueError, match="year_fraction"):
            tracker.step(units_consumed=10, units_remaining_opening=100, year_fraction=0.0)
