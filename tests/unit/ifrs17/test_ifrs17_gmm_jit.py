"""
tests/unit/ifrs17/test_ifrs17_gmm_jit.py

Regression tests for the JAX JIT-compiled GMM inner step.

Purpose (DECISIONS.md §32 Step 1)
-----------------------------------
1. Confirm JAX is an importable, working dependency with float64 active.
2. Confirm _gmm_step_inner produces numerically identical results to the
   hand-computed anchors from test_ifrs17_gmm.py — both profitable and
   onerous-at-inception cases.
3. Confirm that GmmEngine.step() wiring is correct end-to-end: the closing
   CSM from period k is correctly used as the opening CSM for period k+1.
4. Confirm the onerous mid-run path (fcf_change > available CSM): CSM is
   floored at zero and the excess flows into the loss component via the
   jnp.where / jnp.maximum branches in _gmm_step_inner.
"""
import pytest
from datetime import date

from engine.ifrs17._gmm_jit import _gmm_step_inner, JAX_AVAILABLE
from engine.ifrs17.gmm import GmmEngine
from engine.ifrs17.state import Ifrs17State


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

INCEPTION = date(2024, 1, 1)


class ThreePeriodProvider:
    """Same CU schedule as test_ifrs17_gmm.py."""
    _remaining = {0: 90.0, 1: 45.0, 2: 15.0, 3: 0.0}
    _consumed  = {0: 45.0, 1: 30.0, 2: 15.0}

    def units_consumed(self, t: int) -> float:
        return self._consumed[t]

    def units_remaining(self, t: int) -> float:
        return self._remaining.get(t, 0.0)


def make_engine(csm: float = 90.0, loss: float = 0.0) -> GmmEngine:
    state = Ifrs17State(
        cohort_id                = "pensioner",
        valuation_date           = INCEPTION,
        csm_balance              = csm,
        loss_component           = loss,
        remaining_coverage_units = 90.0,
        total_coverage_units     = 90.0,
        locked_in_rate           = 0.04,
        inception_date           = INCEPTION,
    )
    return GmmEngine(
        contract_groups         = ["pensioner"],
        opening_states          = {"pensioner": state},
        coverage_unit_providers = {"pensioner": ThreePeriodProvider()},
    )


# ---------------------------------------------------------------------------
# 1. JAX availability and precision
# ---------------------------------------------------------------------------

class TestJaxAvailability:

    def test_jax_importable(self):
        assert JAX_AVAILABLE, "JAX is not installed — run: uv add jax"

    def test_float64_enabled(self):
        import jax.numpy as jnp
        x = jnp.array(1.0)
        assert x.dtype == jnp.float64, (
            f"Expected float64, got {x.dtype}. "
            "jax_enable_x64 must be set before any JAX array is created."
        )


# ---------------------------------------------------------------------------
# 2. Direct numerical regression on _gmm_step_inner
#
# Profitable case — Period 0 inputs from the 3-period contract:
#   csm_opening=90, locked_in_rate=4%, year_fraction=1.0
#   units_consumed=45, units_remaining=90
#   fcf_change=0 (no assumption change), lc_opening=0
#   bel_current=70, risk_adjustment=5
#
#   accretion  = 90 × 0.04        = 3.60
#   after_acc  = 93.60
#   release    = 93.60 × (45/90)  = 46.80
#   csm_closing= 46.80
#   lrc        = 70 + 5 + 46.80   = 121.80
# ---------------------------------------------------------------------------

class TestGmmStepInnerProfitable:

    @pytest.fixture(scope="class")
    def result(self):
        return _gmm_step_inner(
            90.0,   # csm_opening
            0.04,   # locked_in_rate
            1.0,    # year_fraction
            45.0,   # units_consumed
            90.0,   # units_remaining_opening
            0.0,    # fcf_change_non_financial
            0.0,    # lc_opening
            0.0,    # actual_outgo
            1.0,    # total_remaining_outgo (safe default when lc=0)
            70.0,   # bel_current
            5.0,    # risk_adjustment
        )

    def test_csm_accretion(self, result):
        assert float(result[0]) == pytest.approx(3.60, rel=1e-10)

    def test_csm_adjustment_non_financial_zero(self, result):
        # No assumption change — reported adjustment is zero.
        assert float(result[1]) == pytest.approx(0.0, abs=1e-10)

    def test_csm_release(self, result):
        assert float(result[2]) == pytest.approx(46.80, rel=1e-10)

    def test_csm_closing(self, result):
        assert float(result[3]) == pytest.approx(46.80, rel=1e-10)

    def test_loss_component_all_zero(self, result):
        assert float(result[4]) == pytest.approx(0.0, abs=1e-10)  # lc after addition
        assert float(result[5]) == pytest.approx(0.0, abs=1e-10)  # lc release
        assert float(result[6]) == pytest.approx(0.0, abs=1e-10)  # lc closing

    def test_lrc(self, result):
        assert float(result[7]) == pytest.approx(121.80, rel=1e-10)


# ---------------------------------------------------------------------------
# Onerous-at-inception case:
#   csm_opening=0, lc_opening=50, actual_outgo=20, total_remaining=50
#   → lc_release = 50 × (20/50) = 20, lc_closing = 30
# ---------------------------------------------------------------------------

class TestGmmStepInnerOnerousAtInception:

    @pytest.fixture(scope="class")
    def result(self):
        return _gmm_step_inner(
            0.0,    # csm_opening
            0.04,   # locked_in_rate
            1.0,    # year_fraction
            45.0,   # units_consumed
            90.0,   # units_remaining_opening
            0.0,    # fcf_change_non_financial
            50.0,   # lc_opening
            20.0,   # actual_outgo
            50.0,   # total_remaining_outgo
            70.0,   # bel_current
            5.0,    # risk_adjustment
        )

    def test_csm_zero_throughout(self, result):
        assert float(result[0]) == pytest.approx(0.0, abs=1e-10)  # accretion
        assert float(result[2]) == pytest.approx(0.0, abs=1e-10)  # release
        assert float(result[3]) == pytest.approx(0.0, abs=1e-10)  # closing

    def test_lc_release(self, result):
        assert float(result[5]) == pytest.approx(20.0, rel=1e-10)

    def test_lc_closing(self, result):
        assert float(result[6]) == pytest.approx(30.0, rel=1e-10)


# ---------------------------------------------------------------------------
# 3. GmmEngine end-to-end wiring regression
#
# Runs the full 3-period contract through GmmEngine.step() and checks
# that period k+1 correctly picks up the closing balance from period k.
# Verifies the _set_csm / _set_balance state mutation is working.
# ---------------------------------------------------------------------------

class TestGmmEngineJitWiring:

    @pytest.fixture(scope="class")
    def results(self):
        engine = make_engine()
        r0 = engine.step(
            cohort_id="pensioner", t=0,
            bel_current=70.0, bel_locked=70.0,
            risk_adjustment=5.0, remaining_coverage_units=90.0,
            year_fraction=1.0,
            insurance_finance_pl=0.0, insurance_finance_oci=0.0,
        )
        r1 = engine.step(
            cohort_id="pensioner", t=1,
            bel_current=40.0, bel_locked=40.0,
            risk_adjustment=3.0, remaining_coverage_units=45.0,
            year_fraction=1.0,
            insurance_finance_pl=0.0, insurance_finance_oci=0.0,
        )
        r2 = engine.step(
            cohort_id="pensioner", t=2,
            bel_current=0.0, bel_locked=0.0,
            risk_adjustment=0.0, remaining_coverage_units=15.0,
            year_fraction=1.0,
            insurance_finance_pl=0.0, insurance_finance_oci=0.0,
        )
        return r0, r1, r2

    def test_period0_csm_closing(self, results):
        assert results[0].csm_closing == pytest.approx(46.80, rel=1e-10)

    def test_period1_csm_opening_matches_period0_closing(self, results):
        # This is the key wiring test: period 1 opening must equal period 0 closing.
        assert results[1].csm_opening == pytest.approx(46.80, rel=1e-10)

    def test_period1_csm_closing(self, results):
        # 48.672 × (15/45) = 16.224
        assert results[1].csm_closing == pytest.approx(16.224, rel=1e-10)

    def test_period2_csm_opening_matches_period1_closing(self, results):
        assert results[2].csm_opening == pytest.approx(16.224, rel=1e-10)

    def test_period2_csm_closing_zero(self, results):
        assert results[2].csm_closing == pytest.approx(0.0, abs=1e-10)

    def test_lrc_sequence(self, results):
        assert results[0].lrc == pytest.approx(121.80,  rel=1e-10)
        assert results[1].lrc == pytest.approx(59.224,  rel=1e-10)
        assert results[2].lrc == pytest.approx(0.0,     abs=1e-10)

    def test_repeated_calls_with_different_inputs_no_error(self):
        # JAX handles different input values without retracing on shapes.
        for bel in [70.0, 55.0, 10.0]:
            eng = make_engine()
            eng.step(
                cohort_id="pensioner", t=0,
                bel_current=bel, bel_locked=bel,
                risk_adjustment=5.0, remaining_coverage_units=90.0,
                year_fraction=1.0,
                insurance_finance_pl=0.0, insurance_finance_oci=0.0,
            )


# ---------------------------------------------------------------------------
# 4. Onerous mid-run — FCF deterioration exceeds available CSM
#
# Inputs:
#   csm_opening=90, fcf_change=200 (large deterioration)
#   csm_after_acc = 90 + 3.60 = 93.60
#   raw adj       = 93.60 - 200 = -106.40  → onerous excess = 106.40
#   csm_closing   = 0.0 (floored)
#   lc_opening    = 0 + 106.40 = 106.40
#   lc_release    = 106.40 × (20/100) = 21.28
#   lc_closing    = 106.40 - 21.28 = 85.12
#
# This exercises the jnp.where / jnp.maximum branches in _gmm_step_inner.
# ---------------------------------------------------------------------------

class TestOnerousMidRun:

    @pytest.fixture(scope="class")
    def result(self):
        return _gmm_step_inner(
            90.0,   # csm_opening
            0.04,   # locked_in_rate
            1.0,    # year_fraction
            45.0,   # units_consumed
            90.0,   # units_remaining_opening
            200.0,  # fcf_change_non_financial — severe deterioration
            0.0,    # lc_opening (group was previously profitable)
            20.0,   # actual_outgo
            100.0,  # total_remaining_outgo
            70.0,   # bel_current
            5.0,    # risk_adjustment
        )

    def test_csm_floored_at_zero(self, result):
        assert float(result[3]) == pytest.approx(0.0, abs=1e-10)

    def test_lc_receives_onerous_excess(self, result):
        # lc_opening_after_addition = 0 + 106.40
        assert float(result[4]) == pytest.approx(106.40, rel=1e-6)

    def test_lc_release(self, result):
        # 106.40 × (20/100) = 21.28
        assert float(result[5]) == pytest.approx(21.28, rel=1e-6)

    def test_lc_closing(self, result):
        assert float(result[6]) == pytest.approx(85.12, rel=1e-6)

    def test_csm_release_zero_when_floored(self, result):
        # Once CSM is floored at 0, nothing can be released.
        assert float(result[2]) == pytest.approx(0.0, abs=1e-10)
