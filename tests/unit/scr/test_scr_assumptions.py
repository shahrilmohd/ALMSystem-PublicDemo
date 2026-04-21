"""tests/unit/scr/test_scr_assumptions.py — SCRStressAssumptions unit tests."""
from __future__ import annotations

import dataclasses
import math

import pytest

from engine.scr.scr_assumptions import SCRStressAssumptions


# ---------------------------------------------------------------------------
# SA1: sii_standard_formula() regulatory values
# ---------------------------------------------------------------------------

def test_sa1_spread_stress_defaults():
    a = SCRStressAssumptions.sii_standard_formula()
    assert a.spread_up_bps == 75.0
    assert a.spread_down_bps == 25.0


def test_sa1_interest_stress_defaults():
    a = SCRStressAssumptions.sii_standard_formula()
    assert a.rate_up_bps == 100.0
    assert a.rate_down_bps == 100.0


def test_sa1_longevity_defaults():
    a = SCRStressAssumptions.sii_standard_formula()
    assert a.longevity_mortality_stress_factor == 0.20


def test_sa1_lapse_defaults():
    a = SCRStressAssumptions.sii_standard_formula()
    assert a.lapse_permanent_shock_factor == 0.50
    assert a.lapse_mass_shock_factor == 0.30
    assert a.lapse_tv_to_bel_ratio == 1.0


def test_sa1_expense_defaults():
    a = SCRStressAssumptions.sii_standard_formula()
    assert a.expense_loading_shock_factor == 0.10
    assert a.expense_inflation_shock_pa == 0.01


def test_sa1_currency_defaults():
    a = SCRStressAssumptions.sii_standard_formula()
    assert a.currency_shock_factor == 0.25


def test_sa1_op_risk_defaults():
    a = SCRStressAssumptions.sii_standard_formula()
    assert a.op_risk_bscr_cap_factor == 0.30
    assert a.op_risk_bel_factor == 0.0045


def test_sa1_coc_default():
    a = SCRStressAssumptions.sii_standard_formula()
    assert a.cost_of_capital_rate == 0.06


# ---------------------------------------------------------------------------
# SA2: correlation matrices are symmetric
# ---------------------------------------------------------------------------

def _check_symmetric(matrix: tuple, name: str) -> None:
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            assert math.isclose(matrix[i][j], matrix[j][i], abs_tol=1e-9), (
                f"{name}[{i}][{j}]={matrix[i][j]} != [{j}][{i}]={matrix[j][i]}"
            )


def test_sa2_market_corr_symmetric():
    a = SCRStressAssumptions.sii_standard_formula()
    _check_symmetric(a.market_corr, "market_corr")


def test_sa2_life_corr_symmetric():
    a = SCRStressAssumptions.sii_standard_formula()
    _check_symmetric(a.life_corr, "life_corr")


def test_sa2_module_corr_symmetric():
    a = SCRStressAssumptions.sii_standard_formula()
    _check_symmetric(a.module_corr, "module_corr")


# ---------------------------------------------------------------------------
# SA3: all diagonal entries = 1.0
# ---------------------------------------------------------------------------

def _check_diagonal_ones(matrix: tuple, name: str) -> None:
    for i, row in enumerate(matrix):
        assert math.isclose(row[i], 1.0), f"{name}[{i}][{i}]={row[i]} != 1.0"


def test_sa3_market_corr_diagonal():
    a = SCRStressAssumptions.sii_standard_formula()
    _check_diagonal_ones(a.market_corr, "market_corr")


def test_sa3_life_corr_diagonal():
    a = SCRStressAssumptions.sii_standard_formula()
    _check_diagonal_ones(a.life_corr, "life_corr")


def test_sa3_module_corr_diagonal():
    a = SCRStressAssumptions.sii_standard_formula()
    _check_diagonal_ones(a.module_corr, "module_corr")


# ---------------------------------------------------------------------------
# SA3b: matrix dimensions
# ---------------------------------------------------------------------------

def test_sa3b_market_corr_3x3():
    a = SCRStressAssumptions.sii_standard_formula()
    assert len(a.market_corr) == 3
    assert all(len(row) == 3 for row in a.market_corr)


def test_sa3b_life_corr_4x4():
    a = SCRStressAssumptions.sii_standard_formula()
    assert len(a.life_corr) == 4
    assert all(len(row) == 4 for row in a.life_corr)


def test_sa3b_module_corr_3x3():
    a = SCRStressAssumptions.sii_standard_formula()
    assert len(a.module_corr) == 3
    assert all(len(row) == 3 for row in a.module_corr)


# ---------------------------------------------------------------------------
# SA3c: specific SII DR cross-correlation values
# ---------------------------------------------------------------------------

def test_sa3c_life_corr_mortality_longevity_negative():
    """SII DR Annex IV: corr(mortality, longevity) = -0.25."""
    a = SCRStressAssumptions.sii_standard_formula()
    assert math.isclose(a.life_corr[0][1], -0.25)
    assert math.isclose(a.life_corr[1][0], -0.25)


def test_sa3c_market_corr_interest_spread():
    """SII DR Annex V: corr(interest, spread) = 0.50."""
    a = SCRStressAssumptions.sii_standard_formula()
    assert math.isclose(a.market_corr[0][1], 0.50)


# ---------------------------------------------------------------------------
# SA4: dataclasses.replace() override
# ---------------------------------------------------------------------------

def test_sa4_replace_scalar_field():
    base = SCRStressAssumptions.sii_standard_formula()
    custom = dataclasses.replace(base, spread_up_bps=50.0)
    assert custom.spread_up_bps == 50.0
    assert custom.spread_down_bps == base.spread_down_bps


def test_sa4_replace_preserves_matrices():
    base = SCRStressAssumptions.sii_standard_formula()
    custom = dataclasses.replace(base, cost_of_capital_rate=0.08)
    assert custom.market_corr == base.market_corr
    assert custom.life_corr == base.life_corr
    assert custom.module_corr == base.module_corr


# ---------------------------------------------------------------------------
# SA5: immutability — FrozenInstanceError
# ---------------------------------------------------------------------------

def test_sa5_frozen_instance_error():
    a = SCRStressAssumptions.sii_standard_formula()
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.spread_up_bps = 999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SA6: __post_init__ validation
# ---------------------------------------------------------------------------

def test_sa6_negative_spread_up_raises():
    base = SCRStressAssumptions.sii_standard_formula()
    with pytest.raises(ValueError, match="spread_up_bps"):
        dataclasses.replace(base, spread_up_bps=-1.0)


def test_sa6_longevity_factor_zero_raises():
    base = SCRStressAssumptions.sii_standard_formula()
    with pytest.raises(ValueError, match="longevity_mortality_stress_factor"):
        dataclasses.replace(base, longevity_mortality_stress_factor=0.0)


def test_sa6_longevity_factor_one_raises():
    base = SCRStressAssumptions.sii_standard_formula()
    with pytest.raises(ValueError, match="longevity_mortality_stress_factor"):
        dataclasses.replace(base, longevity_mortality_stress_factor=1.0)


def test_sa6_lapse_permanent_negative_raises():
    base = SCRStressAssumptions.sii_standard_formula()
    with pytest.raises(ValueError, match="lapse_permanent_shock_factor"):
        dataclasses.replace(base, lapse_permanent_shock_factor=-0.1)


def test_sa6_currency_shock_above_one_raises():
    base = SCRStressAssumptions.sii_standard_formula()
    with pytest.raises(ValueError, match="currency_shock_factor"):
        dataclasses.replace(base, currency_shock_factor=1.5)


def test_sa6_asymmetric_matrix_raises():
    base = SCRStressAssumptions.sii_standard_formula()
    bad_corr = (
        (1.00, 0.99, 0.25),
        (0.50, 1.00, 0.25),
        (0.25, 0.25, 1.00),
    )
    with pytest.raises(ValueError, match="market_corr"):
        dataclasses.replace(base, market_corr=bad_corr)


def test_sa6_non_unit_diagonal_raises():
    base = SCRStressAssumptions.sii_standard_formula()
    bad_corr = (
        (0.90, 0.50, 0.25),
        (0.50, 1.00, 0.25),
        (0.25, 0.25, 1.00),
    )
    with pytest.raises(ValueError, match="market_corr"):
        dataclasses.replace(base, market_corr=bad_corr)


def test_sa6_wrong_size_matrix_raises():
    base = SCRStressAssumptions.sii_standard_formula()
    bad_corr = (
        (1.00, 0.50),
        (0.50, 1.00),
    )
    with pytest.raises(ValueError, match="market_corr"):
        dataclasses.replace(base, market_corr=bad_corr)
