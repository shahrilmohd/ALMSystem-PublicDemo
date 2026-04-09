"""
tests/unit/ifrs17/test_ifrs17_loss_component.py

Numerical tests for LossComponentTracker.

All expected values are hand-computed and commented inline.
"""
import pytest

from engine.ifrs17.loss_component import LossComponentTracker, LossComponentStepResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tracker(opening: float = 0.0) -> LossComponentTracker:
    return LossComponentTracker(opening_loss_component=opening)


# ---------------------------------------------------------------------------
# Onerous contract at inception
#
#   FCF_0 = +50 → loss_component_0 = 50, CSM_0 = 0
#
#   Period 1: actual_outgo=20, total_remaining=50
#     release_fraction = 20/50 = 0.40
#     release          = 50 × 0.40 = 20.00
#     closing          = 50 − 20   = 30.00
#
#   Period 2: actual_outgo=30, total_remaining=30
#     release_fraction = 30/30 = 1.00
#     release          = 30 × 1.00 = 30.00
#     closing          = 30 − 30   = 0.00
# ---------------------------------------------------------------------------

class TestOnerousAtInception:
    @pytest.fixture
    def tracker(self) -> LossComponentTracker:
        return make_tracker(opening=50.0)

    def test_period1_release(self, tracker):
        r = tracker.step(actual_outgo=20, total_remaining_outgo=50)
        assert r.release == pytest.approx(20.0, rel=1e-10)

    def test_period1_closing(self, tracker):
        r = tracker.step(actual_outgo=20, total_remaining_outgo=50)
        assert r.closing == pytest.approx(30.0, rel=1e-10)

    def test_period1_opening_field(self, tracker):
        r = tracker.step(actual_outgo=20, total_remaining_outgo=50)
        assert r.opening == pytest.approx(50.0, rel=1e-10)

    def test_period1_addition_is_zero(self, tracker):
        r = tracker.step(actual_outgo=20, total_remaining_outgo=50)
        assert r.addition == 0.0

    def test_period2_release_fully_exhausts(self, tracker):
        tracker.step(actual_outgo=20, total_remaining_outgo=50)
        r = tracker.step(actual_outgo=30, total_remaining_outgo=30)
        assert r.release  == pytest.approx(30.0, rel=1e-10)
        assert r.closing  == pytest.approx(0.0, abs=1e-10)

    def test_balance_updates_correctly(self, tracker):
        assert tracker.balance == 50.0
        tracker.step(actual_outgo=20, total_remaining_outgo=50)
        assert tracker.balance == pytest.approx(30.0, rel=1e-10)
        tracker.step(actual_outgo=30, total_remaining_outgo=30)
        assert tracker.balance == pytest.approx(0.0, abs=1e-10)

    def test_opening_of_period2_matches_closing_of_period1(self, tracker):
        r1 = tracker.step(actual_outgo=20, total_remaining_outgo=50)
        r2 = tracker.step(actual_outgo=30, total_remaining_outgo=30)
        assert r2.opening == pytest.approx(r1.closing, rel=1e-10)


# ---------------------------------------------------------------------------
# Non-onerous group — loss component stays zero throughout
# ---------------------------------------------------------------------------

class TestNonOnerousGroup:
    def test_zero_opening_returns_all_zeros(self):
        tracker = make_tracker(opening=0.0)
        r = tracker.step(actual_outgo=100, total_remaining_outgo=500)
        assert r.opening  == 0.0
        assert r.addition == 0.0
        assert r.release  == 0.0
        assert r.closing  == 0.0

    def test_balance_stays_zero(self):
        tracker = make_tracker(opening=0.0)
        for _ in range(5):
            tracker.step(actual_outgo=10, total_remaining_outgo=100)
        assert tracker.balance == 0.0


# ---------------------------------------------------------------------------
# add_onerous_excess — mid-projection deterioration
#
#   Group starts profitable (no loss component).
#   At period 2, CsmTracker reports onerous_excess = 27.5.
#   GmmEngine calls add_onerous_excess(27.5) before period 2 step.
#
#   Period 2 step: actual_outgo=10, total_remaining=40
#     opening   = 27.5
#     fraction  = 10/40 = 0.25
#     release   = 27.5 × 0.25 = 6.875
#     closing   = 27.5 − 6.875 = 20.625
# ---------------------------------------------------------------------------

class TestAddOnerousExcess:
    def test_excess_added_to_balance(self):
        tracker = make_tracker(opening=0.0)
        tracker.add_onerous_excess(27.5)
        assert tracker.balance == pytest.approx(27.5, rel=1e-10)

    def test_step_after_excess_release(self):
        tracker = make_tracker(opening=0.0)
        tracker.add_onerous_excess(27.5)
        r = tracker.step(actual_outgo=10, total_remaining_outgo=40)
        assert r.opening  == pytest.approx(27.5,  rel=1e-10)
        assert r.release  == pytest.approx(6.875, rel=1e-10)
        assert r.closing  == pytest.approx(20.625, rel=1e-10)

    def test_zero_excess_is_noop(self):
        tracker = make_tracker(opening=10.0)
        tracker.add_onerous_excess(0.0)
        assert tracker.balance == pytest.approx(10.0)

    def test_negative_excess_raises(self):
        tracker = make_tracker()
        with pytest.raises(ValueError, match="excess"):
            tracker.add_onerous_excess(-1.0)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_negative_opening_raises(self):
        with pytest.raises(ValueError, match="opening_loss_component"):
            LossComponentTracker(opening_loss_component=-0.01)

    def test_negative_actual_outgo_raises(self):
        tracker = make_tracker(opening=10.0)
        with pytest.raises(ValueError, match="actual_outgo"):
            tracker.step(actual_outgo=-1.0, total_remaining_outgo=100)

    def test_zero_total_remaining_raises(self):
        tracker = make_tracker(opening=10.0)
        with pytest.raises(ValueError, match="total_remaining_outgo"):
            tracker.step(actual_outgo=5.0, total_remaining_outgo=0.0)
