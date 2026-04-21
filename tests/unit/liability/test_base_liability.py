"""
Unit tests for LiabilityCashflows, Decrements, and BaseLiability.

Rules under test
----------------
LiabilityCashflows.net_outgo:
  1. Positive when outgo > income (claims > premiums).
  2. Negative when income > outgo (premiums > claims).
  3. Zero when outgo equals income.
  4. Correct when all components are non-zero.

Decrements:
  5. All fields stored correctly.
  6. Dataclass equality works (two identical instances are equal).

BaseLiability:
  7. Cannot be instantiated directly (it is abstract).
"""
from __future__ import annotations

import pytest

from engine.liability.base_liability import (
    BaseLiability,
    Decrements,
    LiabilityCashflows,
)


# ---------------------------------------------------------------------------
# LiabilityCashflows.net_outgo
# ---------------------------------------------------------------------------

class TestLiabilityCashflowsNetOutgo:
    def test_positive_when_outgo_exceeds_income(self):
        # Claims 20,000 > premiums 10,000 → net outgo positive
        cf = LiabilityCashflows(
            timestep=1,
            premiums=10_000.0,
            death_claims=20_000.0,
            surrender_payments=0.0,
            maturity_payments=0.0,
            expenses=0.0,
        )
        assert cf.net_outgo == pytest.approx(10_000.0)

    def test_negative_when_income_exceeds_outgo(self):
        # Premiums 50,000 > all outgo components 0 → net outgo negative
        cf = LiabilityCashflows(
            timestep=1,
            premiums=50_000.0,
            death_claims=0.0,
            surrender_payments=0.0,
            maturity_payments=0.0,
            expenses=0.0,
        )
        assert cf.net_outgo == pytest.approx(-50_000.0)

    def test_zero_when_outgo_equals_income(self):
        cf = LiabilityCashflows(
            timestep=1,
            premiums=30_000.0,
            death_claims=10_000.0,
            surrender_payments=5_000.0,
            maturity_payments=10_000.0,
            expenses=5_000.0,
        )
        assert cf.net_outgo == pytest.approx(0.0)

    def test_all_components_included(self):
        cf = LiabilityCashflows(
            timestep=1,
            premiums=100.0,
            death_claims=20.0,
            surrender_payments=30.0,
            maturity_payments=40.0,
            expenses=50.0,
        )
        # net = 20 + 30 + 40 + 50 − 100 = 40
        assert cf.net_outgo == pytest.approx(40.0)

    def test_timestep_stored(self):
        cf = LiabilityCashflows(
            timestep=7,
            premiums=0.0, death_claims=0.0,
            surrender_payments=0.0, maturity_payments=0.0, expenses=0.0,
        )
        assert cf.timestep == 7


# ---------------------------------------------------------------------------
# Decrements
# ---------------------------------------------------------------------------

class TestDecrements:
    def test_all_fields_stored(self):
        d = Decrements(
            timestep=3,
            in_force_start=100.0,
            deaths=1.0,
            lapses=4.95,
            maturities=0.0,
            in_force_end=94.05,
        )
        assert d.timestep       == 3
        assert d.in_force_start == pytest.approx(100.0)
        assert d.deaths         == pytest.approx(1.0)
        assert d.lapses         == pytest.approx(4.95)
        assert d.maturities     == pytest.approx(0.0)
        assert d.in_force_end   == pytest.approx(94.05)

    def test_dataclass_equality(self):
        d1 = Decrements(1, 100.0, 1.0, 4.0, 0.0, 95.0)
        d2 = Decrements(1, 100.0, 1.0, 4.0, 0.0, 95.0)
        assert d1 == d2

    def test_dataclass_inequality(self):
        d1 = Decrements(1, 100.0, 1.0, 4.0, 0.0, 95.0)
        d2 = Decrements(1, 100.0, 2.0, 4.0, 0.0, 94.0)
        assert d1 != d2


# ---------------------------------------------------------------------------
# BaseLiability — abstract
# ---------------------------------------------------------------------------

class TestBaseLiabilityIsAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseLiability()  # type: ignore[abstract]

    def test_subclass_without_all_methods_is_abstract(self):
        class Incomplete(BaseLiability):
            def project_cashflows(self, mp, assumptions, timestep):
                pass
            # missing get_bel, get_reserve, get_decrements

        with pytest.raises(TypeError):
            Incomplete()
