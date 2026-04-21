"""tests/unit/scr/test_currency_stress.py — CurrencyStressEngine unit tests."""
from __future__ import annotations

import dataclasses

import pytest

from engine.scr.currency_stress import CurrencyStressEngine, CurrencyStressResult
from engine.scr.scr_assumptions import SCRStressAssumptions


@pytest.fixture
def engine() -> CurrencyStressEngine:
    return CurrencyStressEngine(SCRStressAssumptions.sii_standard_formula())


# ---------------------------------------------------------------------------
# C1: empty dict → SCR = 0
# ---------------------------------------------------------------------------

def test_c1_empty_dict_zero_scr(engine: CurrencyStressEngine) -> None:
    result = engine.compute({})
    assert result.scr_currency == 0.0
    assert result.total_absolute_exposure_gbp == 0.0


def test_c1_none_zero_scr(engine: CurrencyStressEngine) -> None:
    result = engine.compute(None)
    assert result.scr_currency == 0.0


# ---------------------------------------------------------------------------
# C2: single exposure of 100 → SCR = 25 with sii_standard_formula (factor=0.25)
# ---------------------------------------------------------------------------

def test_c2_single_exposure(engine: CurrencyStressEngine) -> None:
    result = engine.compute({"USD": 100.0})
    assert result.scr_currency == pytest.approx(25.0)
    assert result.total_absolute_exposure_gbp == pytest.approx(100.0)


def test_c2_single_negative_exposure(engine: CurrencyStressEngine) -> None:
    # Short EUR position — absolute value used
    result = engine.compute({"EUR": -100.0})
    assert result.scr_currency == pytest.approx(25.0)
    assert result.total_absolute_exposure_gbp == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# C3: two exposures with opposite signs → total = sum of absolute values
# ---------------------------------------------------------------------------

def test_c3_opposite_signs(engine: CurrencyStressEngine) -> None:
    result = engine.compute({"USD": 200.0, "EUR": -300.0})
    # total_abs = 200 + 300 = 500; SCR = 500 × 0.25 = 125
    assert result.total_absolute_exposure_gbp == pytest.approx(500.0)
    assert result.scr_currency == pytest.approx(125.0)


def test_c3_multiple_currencies_summed(engine: CurrencyStressEngine) -> None:
    result = engine.compute({"USD": 100.0, "EUR": 100.0, "JPY": 50.0})
    assert result.total_absolute_exposure_gbp == pytest.approx(250.0)
    assert result.scr_currency == pytest.approx(62.5)


# ---------------------------------------------------------------------------
# C4: zero currency_shock_factor in assumptions → SCR = 0
# ---------------------------------------------------------------------------

def test_c4_zero_shock_factor() -> None:
    base = SCRStressAssumptions.sii_standard_formula()
    zero = dataclasses.replace(base, currency_shock_factor=0.0)
    engine = CurrencyStressEngine(zero)
    result = engine.compute({"USD": 1_000_000.0})
    assert result.scr_currency == 0.0
    # Still reports exposure correctly
    assert result.total_absolute_exposure_gbp == pytest.approx(1_000_000.0)


# ---------------------------------------------------------------------------
# Governance: shock factor stored in result
# ---------------------------------------------------------------------------

def test_governance_shock_factor_in_result(engine: CurrencyStressEngine) -> None:
    result = engine.compute({"USD": 100.0})
    assert result.shock_factor == pytest.approx(0.25)
