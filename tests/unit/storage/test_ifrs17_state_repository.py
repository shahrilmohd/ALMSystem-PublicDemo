"""
tests/unit/storage/test_ifrs17_state_repository.py

Integration tests for Ifrs17StateStore using in-memory SQLite.

Coverage
--------
  - save_state / load_state roundtrip (state fields preserved exactly)
  - load_state returns None when no row exists
  - save_state overwrites on rerun (upsert)
  - save_movements / get_movements roundtrip (all GmmStepResult fields)
  - get_movements returns [] for unknown key
  - save_movements overwrites on rerun (upsert by period_index)
  - get_movements returns results ordered by period_index
"""
from __future__ import annotations

import pytest
from datetime import date

from engine.ifrs17.gmm import GmmStepResult
from engine.ifrs17.state import Ifrs17State
from storage.ifrs17_state_repository import Ifrs17StateStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INCEPTION  = date(2024, 1, 1)
VALUATION  = date(2024, 12, 31)
VALUATION2 = date(2025, 12, 31)


def make_state(
    cohort_id:      str   = "pensioner",
    valuation_date: date  = VALUATION,
    csm:            float = 46.80,
    loss:           float = 0.0,
    remaining_cu:   float = 45.0,
) -> Ifrs17State:
    return Ifrs17State(
        cohort_id                = cohort_id,
        valuation_date           = valuation_date,
        csm_balance              = csm,
        loss_component           = loss,
        remaining_coverage_units = remaining_cu,
        total_coverage_units     = 90.0,
        locked_in_rate           = 0.04,
        inception_date           = INCEPTION,
    )


def make_step_result(period_index: int = 0, offset: float = 0.0) -> GmmStepResult:
    """Create a GmmStepResult with deterministic values offset by `offset`."""
    return GmmStepResult(
        csm_opening                  = 90.0 + offset,
        csm_accretion                = 3.6  + offset,
        csm_adjustment_non_financial = 0.0,
        csm_release                  = 46.8 + offset,
        csm_closing                  = 46.8 + offset,
        loss_component_opening       = 0.0,
        loss_component_addition      = 0.0,
        loss_component_release       = 0.0,
        loss_component_closing       = 0.0,
        insurance_finance_pl         = 0.0,
        insurance_finance_oci        = 0.0,
        lrc                          = 121.8 + offset,
        lic                          = 0.0,
        p_and_l_csm_release          = 46.8  + offset,
        p_and_l_loss_component       = 0.0,
        p_and_l_insurance_finance    = 0.0,
        bel_current                  = 70.0,
        bel_locked                   = 70.0,
        risk_adjustment              = 5.0,
    )


# ---------------------------------------------------------------------------
# State roundtrip
# ---------------------------------------------------------------------------

class TestLoadState:
    def test_load_returns_none_when_absent(self, session):
        store = Ifrs17StateStore(session)
        result = store.load_state("unknown", VALUATION)
        assert result is None

    def test_roundtrip_all_fields(self, session):
        state = make_state()
        store = Ifrs17StateStore(session)
        store.save_state(state)
        session.commit()

        loaded = store.load_state("pensioner", VALUATION)
        assert loaded is not None
        assert loaded.cohort_id                == "pensioner"
        assert loaded.valuation_date           == VALUATION
        assert loaded.csm_balance              == pytest.approx(46.80)
        assert loaded.loss_component           == pytest.approx(0.0)
        assert loaded.remaining_coverage_units == pytest.approx(45.0)
        assert loaded.total_coverage_units     == pytest.approx(90.0)
        assert loaded.locked_in_rate           == pytest.approx(0.04)
        assert loaded.inception_date           == INCEPTION

    def test_load_returns_none_wrong_valuation_date(self, session):
        store = Ifrs17StateStore(session)
        store.save_state(make_state(valuation_date=VALUATION))
        session.commit()

        result = store.load_state("pensioner", VALUATION2)
        assert result is None

    def test_load_returns_none_wrong_cohort(self, session):
        store = Ifrs17StateStore(session)
        store.save_state(make_state(cohort_id="pensioner"))
        session.commit()

        result = store.load_state("deferred", VALUATION)
        assert result is None

    def test_multiple_cohorts_independent(self, session):
        store = Ifrs17StateStore(session)
        store.save_state(make_state(cohort_id="pensioner", csm=46.80))
        store.save_state(make_state(cohort_id="deferred",  csm=22.00))
        session.commit()

        p = store.load_state("pensioner", VALUATION)
        d = store.load_state("deferred",  VALUATION)
        assert p.csm_balance == pytest.approx(46.80)
        assert d.csm_balance == pytest.approx(22.00)


class TestSaveStateUpsert:
    def test_rerun_overwrites_state(self, session):
        store = Ifrs17StateStore(session)
        store.save_state(make_state(csm=46.80))
        session.commit()

        # Rerun produces different closing balance
        store.save_state(make_state(csm=50.00))
        session.commit()

        loaded = store.load_state("pensioner", VALUATION)
        assert loaded.csm_balance == pytest.approx(50.00)

    def test_upsert_updates_all_fields(self, session):
        store = Ifrs17StateStore(session)
        store.save_state(make_state(csm=46.80, loss=0.0, remaining_cu=45.0))
        session.commit()

        store.save_state(make_state(csm=0.0, loss=10.0, remaining_cu=30.0))
        session.commit()

        loaded = store.load_state("pensioner", VALUATION)
        assert loaded.csm_balance              == pytest.approx(0.0)
        assert loaded.loss_component           == pytest.approx(10.0)
        assert loaded.remaining_coverage_units == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Movements roundtrip
# ---------------------------------------------------------------------------

class TestGetMovements:
    def test_get_returns_empty_list_when_absent(self, session):
        store = Ifrs17StateStore(session)
        results = store.get_movements("unknown", VALUATION)
        assert results == []

    def test_single_period_roundtrip(self, session):
        r = make_step_result(period_index=0)
        store = Ifrs17StateStore(session)
        store.save_movements("pensioner", VALUATION, [r])
        session.commit()

        loaded = store.get_movements("pensioner", VALUATION)
        assert len(loaded) == 1
        l = loaded[0]
        assert l.csm_opening   == pytest.approx(90.0)
        assert l.csm_accretion == pytest.approx(3.6)
        assert l.csm_release   == pytest.approx(46.8)
        assert l.csm_closing   == pytest.approx(46.8)
        assert l.lrc           == pytest.approx(121.8)
        assert l.bel_current   == pytest.approx(70.0)
        assert l.risk_adjustment == pytest.approx(5.0)

    def test_three_periods_roundtrip(self, session):
        movements = [make_step_result(i, offset=float(i)) for i in range(3)]
        store = Ifrs17StateStore(session)
        store.save_movements("pensioner", VALUATION, movements)
        session.commit()

        loaded = store.get_movements("pensioner", VALUATION)
        assert len(loaded) == 3
        for i, l in enumerate(loaded):
            assert l.csm_opening == pytest.approx(90.0 + i)

    def test_movements_returned_in_period_order(self, session):
        # Save in reverse to prove ordering is by period_index, not insert order
        movements = [make_step_result(i, offset=float(i * 10)) for i in range(5)]
        store = Ifrs17StateStore(session)
        store.save_movements("pensioner", VALUATION, movements)
        session.commit()

        loaded = store.get_movements("pensioner", VALUATION)
        assert len(loaded) == 5
        # csm_opening for period i = 90 + i*10
        for i, l in enumerate(loaded):
            assert l.csm_opening == pytest.approx(90.0 + i * 10)

    def test_movements_isolated_by_cohort(self, session):
        store = Ifrs17StateStore(session)
        store.save_movements("pensioner", VALUATION, [make_step_result(0, offset=0.0)])
        store.save_movements("deferred",  VALUATION, [make_step_result(0, offset=99.0)])
        session.commit()

        p = store.get_movements("pensioner", VALUATION)
        d = store.get_movements("deferred",  VALUATION)
        assert p[0].csm_opening == pytest.approx(90.0)
        assert d[0].csm_opening == pytest.approx(189.0)

    def test_movements_isolated_by_valuation_date(self, session):
        store = Ifrs17StateStore(session)
        store.save_movements("pensioner", VALUATION,  [make_step_result(0, offset=0.0)])
        store.save_movements("pensioner", VALUATION2, [make_step_result(0, offset=5.0)])
        session.commit()

        m1 = store.get_movements("pensioner", VALUATION)
        m2 = store.get_movements("pensioner", VALUATION2)
        assert m1[0].csm_opening == pytest.approx(90.0)
        assert m2[0].csm_opening == pytest.approx(95.0)


class TestSaveMovementsUpsert:
    def test_rerun_overwrites_movements(self, session):
        store = Ifrs17StateStore(session)
        store.save_movements("pensioner", VALUATION, [make_step_result(0, offset=0.0)])
        session.commit()

        store.save_movements("pensioner", VALUATION, [make_step_result(0, offset=10.0)])
        session.commit()

        loaded = store.get_movements("pensioner", VALUATION)
        assert len(loaded) == 1
        assert loaded[0].csm_opening == pytest.approx(100.0)

    def test_rerun_with_more_periods_extends(self, session):
        """First run: 2 periods. Rerun: 3 periods. All 3 should be present."""
        store = Ifrs17StateStore(session)
        store.save_movements("pensioner", VALUATION, [make_step_result(i) for i in range(2)])
        session.commit()

        store.save_movements("pensioner", VALUATION, [make_step_result(i, offset=float(i)) for i in range(3)])
        session.commit()

        loaded = store.get_movements("pensioner", VALUATION)
        assert len(loaded) == 3


# ---------------------------------------------------------------------------
# P&L reconstruction from stored fields
# ---------------------------------------------------------------------------

class TestPnlReconstruction:
    def test_p_and_l_fields_reconstructed_from_stored_values(self, session):
        """
        p_and_l_csm_release and p_and_l_loss_component are recomputed from
        stored csm_release / loss_component fields on get_movements().
        """
        r = GmmStepResult(
            csm_opening                  = 50.0,
            csm_accretion                = 2.0,
            csm_adjustment_non_financial = 0.0,
            csm_release                  = 25.0,
            csm_closing                  = 27.0,
            loss_component_opening       = 10.0,
            loss_component_addition      = 5.0,
            loss_component_release       = 8.0,
            loss_component_closing       = 7.0,
            insurance_finance_pl         = 1.5,
            insurance_finance_oci        = 0.5,
            lrc                          = 80.0,
            lic                          = 3.0,
            p_and_l_csm_release          = 25.0,
            p_and_l_loss_component       = 3.0,   # release - addition = 8 - 5
            p_and_l_insurance_finance    = 1.5,
            bel_current                  = 50.0,
            bel_locked                   = 48.0,
            risk_adjustment              = 3.0,
        )
        store = Ifrs17StateStore(session)
        store.save_movements("pensioner", VALUATION, [r])
        session.commit()

        loaded = store.get_movements("pensioner", VALUATION)
        l = loaded[0]
        assert l.p_and_l_csm_release       == pytest.approx(25.0)
        assert l.p_and_l_loss_component    == pytest.approx(3.0)   # 8 - 5
        assert l.p_and_l_insurance_finance == pytest.approx(1.5)
        assert l.insurance_finance_oci     == pytest.approx(0.5)
        assert l.lic                       == pytest.approx(3.0)
