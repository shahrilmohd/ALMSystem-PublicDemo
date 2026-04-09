"""
tests/unit/ifrs17/test_ifrs17_gmm.py

End-to-end numerical tests for GmmEngine.

All expected values are hand-computed from the canonical 3-period contract
defined at the top of this module. The test verifies that GmmEngine correctly
orchestrates CsmTracker, LossComponentTracker, and assembles LRC/LIC output.

3-Period Profitable Contract
============================
  Inception:  FCF_0 = -90  →  CSM_0 = 90
  locked_in_rate = 4% p.a.
  total_coverage_units = 90

  CU schedule (remaining at START of each period, units consumed each period):
    t=0: remaining=90, consumed=45
    t=1: remaining=45, consumed=30
    t=2: remaining=15, consumed=15

  RA = 5, 3, 0 at t=0,1,2 (simplified for test clarity)
  BEL current/locked = same (no financial assumption changes) = 70, 40, 0

  Period 0 (t=0):  year_fraction=1.0
    accretion  = 90 × 0.04        = 3.60
    after_acc  = 93.60
    release    = 93.60 × (45/90)  = 46.80
    closing    = 93.60 × (45/90)  = 46.80
    LRC        = 70 + 5 + 46.80   = 121.80

  Period 1 (t=1):  year_fraction=1.0
    accretion  = 46.80 × 0.04     = 1.872
    after_acc  = 48.672
    release    = 48.672 × (30/45) = 32.448
    closing    = 48.672 × (15/45) = 16.224
    LRC        = 40 + 3 + 16.224  = 59.224

  Period 2 (t=2):  year_fraction=1.0
    accretion  = 16.224 × 0.04    = 0.64896
    after_acc  = 16.87296
    release    = 16.87296 × (15/15) = 16.87296
    closing    = 0.0
    LRC        = 0 + 0 + 0        = 0.0
"""
import pytest
from datetime import date

from engine.ifrs17.gmm import GmmEngine, GmmStepResult
from engine.ifrs17.coverage_units import CoverageUnitProvider
from engine.ifrs17.state import Ifrs17State


# ---------------------------------------------------------------------------
# Synthetic CoverageUnitProvider for the 3-period contract
# ---------------------------------------------------------------------------

class ThreePeriodProvider:
    """
    CU schedule:
      t=0: remaining=90, consumed=45
      t=1: remaining=45, consumed=30
      t=2: remaining=15, consumed=15
      t=3: remaining=0  (end of projection)
    """
    _remaining = {0: 90.0, 1: 45.0, 2: 15.0, 3: 0.0}
    _consumed  = {0: 45.0, 1: 30.0, 2: 15.0}

    def units_consumed(self, t: int) -> float:
        return self._consumed[t]

    def units_remaining(self, t: int) -> float:
        return self._remaining.get(t, 0.0)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

INCEPTION = date(2024, 1, 1)
VALUATION = date(2024, 12, 31)


def make_opening_state(
    csm: float = 90.0,
    loss: float = 0.0,
    remaining_cu: float = 90.0,
) -> Ifrs17State:
    return Ifrs17State(
        cohort_id                = "pensioner",
        valuation_date           = INCEPTION,
        csm_balance              = csm,
        loss_component           = loss,
        remaining_coverage_units = remaining_cu,
        total_coverage_units     = 90.0,
        locked_in_rate           = 0.04,
        inception_date           = INCEPTION,
    )


def make_engine(
    csm: float = 90.0,
    loss: float = 0.0,
) -> GmmEngine:
    state = make_opening_state(csm=csm, loss=loss)
    return GmmEngine(
        contract_groups         = ["pensioner"],
        opening_states          = {"pensioner": state},
        coverage_unit_providers = {"pensioner": ThreePeriodProvider()},
    )


def step(engine: GmmEngine, t: int, bel: float = 70.0, ra: float = 5.0,
         remaining_cu: float = 90.0, **kwargs) -> GmmStepResult:
    """Convenience wrapper with sensible defaults."""
    return engine.step(
        cohort_id                = "pensioner",
        t                        = t,
        bel_current              = bel,
        bel_locked               = bel,
        risk_adjustment          = ra,
        remaining_coverage_units = remaining_cu,
        year_fraction            = 1.0,
        insurance_finance_pl     = 0.0,
        insurance_finance_oci    = 0.0,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Period 0
# ---------------------------------------------------------------------------

class TestPeriod0:
    @pytest.fixture
    def result(self) -> GmmStepResult:
        return step(make_engine(), t=0, bel=70.0, ra=5.0, remaining_cu=90.0)

    def test_csm_opening(self, result):
        assert result.csm_opening == pytest.approx(90.0)

    def test_csm_accretion(self, result):
        # 90 × 0.04 = 3.60
        assert result.csm_accretion == pytest.approx(3.60, rel=1e-10)

    def test_csm_release(self, result):
        # 93.60 × (45/90) = 46.80
        assert result.csm_release == pytest.approx(46.80, rel=1e-10)

    def test_csm_closing(self, result):
        # 93.60 × (45/90) = 46.80
        assert result.csm_closing == pytest.approx(46.80, rel=1e-10)

    def test_lrc(self, result):
        # 70 + 5 + 46.80 = 121.80
        assert result.lrc == pytest.approx(121.80, rel=1e-10)

    def test_loss_component_all_zero(self, result):
        assert result.loss_component_opening  == 0.0
        assert result.loss_component_release  == 0.0
        assert result.loss_component_closing  == 0.0

    def test_p_and_l_csm_release(self, result):
        assert result.p_and_l_csm_release == pytest.approx(46.80, rel=1e-10)


# ---------------------------------------------------------------------------
# Period 1
# ---------------------------------------------------------------------------

class TestPeriod1:
    @pytest.fixture
    def result(self) -> GmmStepResult:
        engine = make_engine()
        step(engine, t=0, bel=70.0, ra=5.0, remaining_cu=90.0)
        return step(engine, t=1, bel=40.0, ra=3.0, remaining_cu=45.0)

    def test_csm_opening(self, result):
        assert result.csm_opening == pytest.approx(46.80, rel=1e-10)

    def test_csm_accretion(self, result):
        # 46.80 × 0.04 = 1.872
        assert result.csm_accretion == pytest.approx(1.872, rel=1e-10)

    def test_csm_release(self, result):
        # 48.672 × (30/45) = 32.448
        assert result.csm_release == pytest.approx(32.448, rel=1e-10)

    def test_csm_closing(self, result):
        # 48.672 × (15/45) = 16.224
        assert result.csm_closing == pytest.approx(16.224, rel=1e-10)

    def test_lrc(self, result):
        # 40 + 3 + 16.224 = 59.224
        assert result.lrc == pytest.approx(59.224, rel=1e-10)


# ---------------------------------------------------------------------------
# Period 2
# ---------------------------------------------------------------------------

class TestPeriod2:
    @pytest.fixture
    def result(self) -> GmmStepResult:
        engine = make_engine()
        step(engine, t=0, bel=70.0, ra=5.0, remaining_cu=90.0)
        step(engine, t=1, bel=40.0, ra=3.0, remaining_cu=45.0)
        return step(engine, t=2, bel=0.0, ra=0.0, remaining_cu=15.0)

    def test_csm_accretion(self, result):
        # 16.224 × 0.04 = 0.64896
        assert result.csm_accretion == pytest.approx(0.64896, rel=1e-10)

    def test_csm_release_exhausts_balance(self, result):
        # All remaining CSM released: 16.87296 × (15/15) = 16.87296
        assert result.csm_release == pytest.approx(16.87296, rel=1e-10)

    def test_csm_closing_zero(self, result):
        assert result.csm_closing == pytest.approx(0.0, abs=1e-10)

    def test_lrc_zero(self, result):
        # BEL=0, RA=0, CSM=0
        assert result.lrc == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Onerous group at inception
#
#   FCF_0 = +50 → loss_component_0 = 50, csm_0 = 0
#   Period 0: actual_outgo=20, total_remaining=50
#     lc_release = 50 × (20/50) = 20, closing = 30
# ---------------------------------------------------------------------------

class TestOnerousAtInception:
    def test_loss_component_release(self):
        state = Ifrs17State(
            cohort_id                = "pensioner",
            valuation_date           = INCEPTION,
            csm_balance              = 0.0,
            loss_component           = 50.0,
            remaining_coverage_units = 90.0,
            total_coverage_units     = 90.0,
            locked_in_rate           = 0.04,
            inception_date           = INCEPTION,
        )
        engine = GmmEngine(
            contract_groups         = ["pensioner"],
            opening_states          = {"pensioner": state},
            coverage_unit_providers = {"pensioner": ThreePeriodProvider()},
        )
        r = engine.step(
            cohort_id                = "pensioner",
            t                        = 0,
            bel_current              = 70.0,
            bel_locked               = 70.0,
            risk_adjustment          = 5.0,
            remaining_coverage_units = 90.0,
            year_fraction            = 1.0,
            actual_outgo             = 20.0,
            total_remaining_outgo    = 50.0,
            insurance_finance_pl     = 0.0,
            insurance_finance_oci    = 0.0,
        )
        assert r.loss_component_opening  == pytest.approx(50.0)
        assert r.loss_component_release  == pytest.approx(20.0, rel=1e-10)
        assert r.loss_component_closing  == pytest.approx(30.0, rel=1e-10)
        assert r.csm_opening             == 0.0
        assert r.csm_release             == 0.0


# ---------------------------------------------------------------------------
# closing_state()
# ---------------------------------------------------------------------------

class TestClosingState:
    def test_closing_state_csm_matches_tracker(self):
        engine = make_engine()
        step(engine, t=0, bel=70.0, ra=5.0, remaining_cu=90.0)
        state = engine.closing_state("pensioner", VALUATION, INCEPTION)
        assert state.csm_balance == pytest.approx(46.80, rel=1e-10)
        assert state.cohort_id   == "pensioner"
        assert state.valuation_date == VALUATION

    def test_closing_state_after_all_periods_csm_zero(self):
        engine = make_engine()
        step(engine, t=0, bel=70.0, ra=5.0, remaining_cu=90.0)
        step(engine, t=1, bel=40.0, ra=3.0, remaining_cu=45.0)
        step(engine, t=2, bel=0.0,  ra=0.0, remaining_cu=15.0)
        state = engine.closing_state("pensioner", VALUATION, INCEPTION)
        assert state.csm_balance == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_unknown_cohort_id_raises(self):
        engine = make_engine()
        with pytest.raises(ValueError, match="cohort_id"):
            engine.step(
                cohort_id="unknown", t=0,
                bel_current=70, bel_locked=70,
                risk_adjustment=5, remaining_coverage_units=90,
            )

    def test_negative_risk_adjustment_raises(self):
        engine = make_engine()
        with pytest.raises(ValueError, match="risk_adjustment"):
            engine.step(
                cohort_id="pensioner", t=0,
                bel_current=70, bel_locked=70,
                risk_adjustment=-1.0, remaining_coverage_units=90,
            )

    def test_missing_state_raises_at_construction(self):
        with pytest.raises(ValueError, match="Ifrs17State"):
            GmmEngine(
                contract_groups         = ["pensioner", "deferred"],
                opening_states          = {"pensioner": make_opening_state()},
                coverage_unit_providers = {
                    "pensioner": ThreePeriodProvider(),
                    "deferred":  ThreePeriodProvider(),
                },
            )

    def test_missing_provider_raises_at_construction(self):
        with pytest.raises(ValueError, match="CoverageUnitProvider"):
            GmmEngine(
                contract_groups         = ["pensioner", "deferred"],
                opening_states          = {
                    "pensioner": make_opening_state(),
                    "deferred":  make_opening_state(),
                },
                coverage_unit_providers = {"pensioner": ThreePeriodProvider()},
            )
