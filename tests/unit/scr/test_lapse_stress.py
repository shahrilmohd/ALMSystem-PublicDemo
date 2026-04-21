"""tests/unit/scr/test_lapse_stress.py — LapseStressEngine unit tests."""
from __future__ import annotations

import pytest

from engine.scr.lapse_stress import LapseStressEngine, LapseStressResult
from engine.scr.scr_assumptions import SCRStressAssumptions


@pytest.fixture
def engine() -> LapseStressEngine:
    return LapseStressEngine(SCRStressAssumptions.sii_standard_formula())


# ---------------------------------------------------------------------------
# L1: all inputs zero → SCR = 0 (in-payment annuity / no-lapse product)
# ---------------------------------------------------------------------------

def test_l1_all_zero_inputs(engine: LapseStressEngine) -> None:
    result = engine.compute(
        bel_lapse_up=0.0,
        bel_lapse_down=0.0,
        base_bel=0.0,
        mass_lapse_bel_reduction=0.0,
        mass_lapse_asset_outflow=0.0,
    )
    assert result.scr_lapse == 0.0


def test_l1_zero_fields_populated_correctly(engine: LapseStressEngine) -> None:
    result = engine.compute(0.0, 0.0, 0.0, 0.0, 0.0)
    assert result.lapse_up_own_funds_change == 0.0
    assert result.lapse_down_own_funds_change == 0.0
    assert result.mass_lapse_own_funds_change == 0.0


# ---------------------------------------------------------------------------
# L2: bel_lapse_up < base_bel → of_change_up positive (gain; SCR contribution = 0)
# ---------------------------------------------------------------------------

def test_l2_lapse_up_gain(engine: LapseStressEngine) -> None:
    # BEL falls under higher lapse: own funds rise (gain), not adverse
    result = engine.compute(
        bel_lapse_up=800.0,   # BEL fell → gain
        bel_lapse_down=1000.0,
        base_bel=1000.0,
        mass_lapse_bel_reduction=0.0,
        mass_lapse_asset_outflow=0.0,
    )
    # of_change_up = -(800 - 1000) = +200 (gain)
    assert result.lapse_up_own_funds_change == pytest.approx(200.0)
    # SCR contribution from up-shock = -of_change_up = -200 < 0, floored to 0
    assert result.scr_lapse == 0.0  # down-shock also zero (no change), mass zero


# ---------------------------------------------------------------------------
# L3: bel_lapse_down > base_bel → of_change_down negative (loss → positive SCR)
# ---------------------------------------------------------------------------

def test_l3_lapse_down_loss(engine: LapseStressEngine) -> None:
    # BEL rises under lower lapse: own funds fall (loss → SCR)
    result = engine.compute(
        bel_lapse_up=1000.0,
        bel_lapse_down=1200.0,  # BEL rose → loss
        base_bel=1000.0,
        mass_lapse_bel_reduction=0.0,
        mass_lapse_asset_outflow=0.0,
    )
    # of_change_down = -(1200 - 1000) = -200 (loss)
    assert result.lapse_down_own_funds_change == pytest.approx(-200.0)
    # SCR = -(-200) = 200
    assert result.scr_lapse == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# L4: mass lapse: bel_reduction > asset_outflow → net gain (no SCR)
# ---------------------------------------------------------------------------

def test_l4_mass_lapse_net_gain(engine: LapseStressEngine) -> None:
    result = engine.compute(
        bel_lapse_up=1000.0,
        bel_lapse_down=1000.0,
        base_bel=1000.0,
        mass_lapse_bel_reduction=300.0,   # BEL fell by 300
        mass_lapse_asset_outflow=200.0,   # only 200 paid out
    )
    # of_change_mass = 300 - 200 = 100 (net gain, BEL fell more than TV paid)
    assert result.mass_lapse_own_funds_change == pytest.approx(100.0)
    assert result.scr_lapse == 0.0


# ---------------------------------------------------------------------------
# L5: mass lapse: bel_reduction < asset_outflow → net loss → SCR
# ---------------------------------------------------------------------------

def test_l5_mass_lapse_net_loss(engine: LapseStressEngine) -> None:
    result = engine.compute(
        bel_lapse_up=1000.0,
        bel_lapse_down=1000.0,
        base_bel=1000.0,
        mass_lapse_bel_reduction=100.0,   # BEL fell by 100
        mass_lapse_asset_outflow=300.0,   # but 300 paid out → net loss
    )
    # of_change_mass = 100 - 300 = -200 (net loss)
    assert result.mass_lapse_own_funds_change == pytest.approx(-200.0)
    assert result.scr_lapse == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# L6: SCR = max of three scenarios, always ≥ 0
# ---------------------------------------------------------------------------

def test_l6_scr_is_max_of_three(engine: LapseStressEngine) -> None:
    result = engine.compute(
        bel_lapse_up=900.0,     # of_change_up = 100 (gain) → SCR contrib = 0
        bel_lapse_down=1150.0,  # of_change_down = -150 → SCR contrib = 150
        base_bel=1000.0,
        mass_lapse_bel_reduction=50.0,
        mass_lapse_asset_outflow=250.0,  # of_change_mass = -200 → SCR contrib = 200
    )
    assert result.scr_lapse == pytest.approx(200.0)


def test_l6_scr_nonnegative_always() -> None:
    engine = LapseStressEngine(SCRStressAssumptions.sii_standard_formula())
    # All scenarios favourable
    result = engine.compute(
        bel_lapse_up=800.0,
        bel_lapse_down=900.0,
        base_bel=1000.0,
        mass_lapse_bel_reduction=500.0,
        mass_lapse_asset_outflow=100.0,
    )
    assert result.scr_lapse >= 0.0


# ---------------------------------------------------------------------------
# L7: governance fields are populated from assumptions
# ---------------------------------------------------------------------------

def test_l7_governance_fields(engine: LapseStressEngine) -> None:
    result = engine.compute(0.0, 0.0, 0.0, 0.0, 0.0)
    assert result.permanent_shock_factor == pytest.approx(0.50)
    assert result.mass_lapse_shock_factor == pytest.approx(0.30)


def test_l7_custom_assumptions_governance() -> None:
    import dataclasses
    base = SCRStressAssumptions.sii_standard_formula()
    custom = dataclasses.replace(base, lapse_permanent_shock_factor=0.25, lapse_mass_shock_factor=0.20)
    engine = LapseStressEngine(custom)
    result = engine.compute(0.0, 0.0, 0.0, 0.0, 0.0)
    assert result.permanent_shock_factor == pytest.approx(0.25)
    assert result.mass_lapse_shock_factor == pytest.approx(0.20)
