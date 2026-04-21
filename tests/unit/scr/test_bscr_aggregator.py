"""tests/unit/scr/test_bscr_aggregator.py — BSCRAggregator unit tests."""
from __future__ import annotations

import dataclasses
import math

import pytest

from engine.scr.bscr_aggregator import BSCRAggregator
from engine.scr.scr_assumptions import SCRStressAssumptions


@pytest.fixture
def agg() -> BSCRAggregator:
    return BSCRAggregator(SCRStressAssumptions.sii_standard_formula())


def _call(agg: BSCRAggregator, **kwargs) -> tuple[float, float, float, float]:
    defaults = dict(
        scr_interest=0.0, scr_spread=0.0, scr_currency=0.0,
        scr_mortality=0.0, scr_longevity=0.0, scr_lapse=0.0, scr_expense=0.0,
        scr_counterparty=0.0, base_bel_post_ma=0.0,
    )
    defaults.update(kwargs)
    return agg.aggregate(**defaults)


# ---------------------------------------------------------------------------
# BA1: all sub-modules zero → all aggregates zero
# ---------------------------------------------------------------------------

def test_ba1_all_zero(agg: BSCRAggregator) -> None:
    scr_market, scr_life, bscr, scr_op = _call(agg)
    assert scr_market == 0.0
    assert scr_life   == 0.0
    assert bscr       == 0.0
    assert scr_op     == 0.0


# ---------------------------------------------------------------------------
# BA2: single sub-module non-zero → BSCR = that sub-module's SCR
# ---------------------------------------------------------------------------

def test_ba2_single_nonzero_interest(agg: BSCRAggregator) -> None:
    scr_market, scr_life, bscr, _ = _call(agg, scr_interest=100.0, base_bel_post_ma=10_000.0)
    assert scr_market == pytest.approx(100.0)
    assert scr_life   == pytest.approx(0.0)
    assert bscr       == pytest.approx(100.0)


def test_ba2_single_nonzero_longevity(agg: BSCRAggregator) -> None:
    scr_market, scr_life, bscr, _ = _call(agg, scr_longevity=200.0, base_bel_post_ma=10_000.0)
    assert scr_life   == pytest.approx(200.0)
    assert scr_market == pytest.approx(0.0)
    assert bscr       == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# BA3: identity correlation matrix → BSCR = sqrt(A² + B²)
# ---------------------------------------------------------------------------

def test_ba3_identity_correlation() -> None:
    base = SCRStressAssumptions.sii_standard_formula()
    identity_3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    identity_4 = (
        (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0),
    )
    custom = dataclasses.replace(
        base,
        market_corr=identity_3,
        life_corr=identity_4,
        module_corr=identity_3,
    )
    agg = BSCRAggregator(custom)

    scr_market, scr_life, bscr, _ = _call(agg, scr_interest=30.0, scr_spread=40.0,
                                           base_bel_post_ma=100.0)
    # market: sqrt(30² + 40²) = 50.0
    assert scr_market == pytest.approx(50.0)
    # bscr: sqrt(50² + 0² + 0²) = 50.0
    assert bscr == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# BA4: all-ones correlation matrix → BSCR = A + B (perfect correlation)
# ---------------------------------------------------------------------------

def test_ba4_all_ones_correlation() -> None:
    base = SCRStressAssumptions.sii_standard_formula()
    ones_3 = ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    ones_4 = (
        (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 1.0),
    )
    custom = dataclasses.replace(base, market_corr=ones_3, life_corr=ones_4, module_corr=ones_3)
    agg = BSCRAggregator(custom)

    scr_market, scr_life, bscr, _ = _call(agg, scr_interest=30.0, scr_spread=40.0,
                                           base_bel_post_ma=100.0)
    # market: sqrt((30+40)^2) = 70
    assert scr_market == pytest.approx(70.0)


# ---------------------------------------------------------------------------
# BA5: market aggregation with SII defaults — numerical anchor
# ---------------------------------------------------------------------------

def test_ba5_market_aggregation_numerical(agg: BSCRAggregator) -> None:
    # interest=50, spread=50, currency=0, ρ_IS=0.50
    # scr_market = sqrt(50² + 50² + 2×0.5×50×50) = sqrt(2500+2500+2500) = sqrt(7500) ≈ 86.60
    scr_market, _, _, _ = _call(agg, scr_interest=50.0, scr_spread=50.0, base_bel_post_ma=10_000.0)
    expected = math.sqrt(7500.0)
    assert scr_market == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# BA6: BPA path — scr_mortality=0, scr_longevity=100 → life SCR = 100
# ---------------------------------------------------------------------------

def test_ba6_bpa_path(agg: BSCRAggregator) -> None:
    _, scr_life, _, _ = _call(agg, scr_longevity=100.0, base_bel_post_ma=10_000.0)
    assert scr_life == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# BA7: conventional path — scr_longevity=0, scr_mortality=100 → life SCR = 100
# ---------------------------------------------------------------------------

def test_ba7_conventional_path(agg: BSCRAggregator) -> None:
    _, scr_life, _, _ = _call(agg, scr_mortality=100.0, base_bel_post_ma=10_000.0)
    assert scr_life == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# BA8: mortality + longevity both non-zero → life SCR < sum (negative corr)
# ---------------------------------------------------------------------------

def test_ba8_mortality_longevity_negative_corr(agg: BSCRAggregator) -> None:
    # ρ(mortality, longevity) = -0.25 per SII DR Annex IV
    # life = sqrt(100² + 100² + 2×(-0.25)×100×100) = sqrt(10000+10000-5000) = sqrt(15000) ≈ 122.47
    _, scr_life, _, _ = _call(agg, scr_mortality=100.0, scr_longevity=100.0, base_bel_post_ma=0.0)
    naive_sum = 200.0
    assert scr_life < naive_sum, "negative correlation should reduce combined life SCR"
    expected = math.sqrt(10_000 + 10_000 + 2 * (-0.25) * 10_000)
    assert scr_life == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# BA9: op risk BEL term binds
# ---------------------------------------------------------------------------

def test_ba9_op_risk_bel_binds(agg: BSCRAggregator) -> None:
    # BSCR=1000, BEL=5000; op_bscr_cap=0.30 → 300; op_bel=0.0045 → 22.5 → min = 22.5
    _, _, bscr, scr_op = _call(agg, scr_interest=1000.0, base_bel_post_ma=5_000.0)
    assert bscr == pytest.approx(1000.0)
    assert scr_op == pytest.approx(22.5, rel=1e-4)


# ---------------------------------------------------------------------------
# BA10: op risk BSCR cap binds
# ---------------------------------------------------------------------------

def test_ba10_op_risk_bscr_cap_binds(agg: BSCRAggregator) -> None:
    # BSCR=1000, BEL=100000; cap → 300; bel → 450 → min = 300
    _, _, bscr, scr_op = _call(agg, scr_interest=1000.0, base_bel_post_ma=100_000.0)
    assert bscr == pytest.approx(1000.0)
    assert scr_op == pytest.approx(300.0, rel=1e-4)


# ---------------------------------------------------------------------------
# BA11: overriding a correlation matrix via custom assumptions changes output
# ---------------------------------------------------------------------------

def test_ba11_custom_correlation_changes_output() -> None:
    base = SCRStressAssumptions.sii_standard_formula()

    # Override module_corr to identity (no between-module diversification)
    identity_3 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    custom = dataclasses.replace(base, module_corr=identity_3)

    agg_sii    = BSCRAggregator(base)
    agg_custom = BSCRAggregator(custom)

    kwargs = dict(scr_interest=100.0, scr_longevity=200.0, base_bel_post_ma=10_000.0)
    _, _, bscr_sii,    _ = _call(agg_sii,    **kwargs)
    _, _, bscr_custom, _ = _call(agg_custom, **kwargs)

    # identity → BSCR = sqrt(market² + life²); SII has positive off-diag → higher BSCR
    assert bscr_custom != pytest.approx(bscr_sii, rel=0.01), (
        "changing module_corr should change BSCR"
    )
