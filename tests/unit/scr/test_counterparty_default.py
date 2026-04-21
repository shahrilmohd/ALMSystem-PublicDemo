"""tests/unit/scr/test_counterparty_default.py — CounterpartyDefaultEngine unit tests."""
from __future__ import annotations

import math

import pytest

from engine.scr.counterparty_default import (
    CounterpartyDefaultEngine,
    CounterpartyDefaultResult,
    CounterpartyExposure,
)


@pytest.fixture
def engine() -> CounterpartyDefaultEngine:
    return CounterpartyDefaultEngine()


# ---------------------------------------------------------------------------
# CD1: None → zero
# ---------------------------------------------------------------------------

def test_cd1_none_zero(engine: CounterpartyDefaultEngine) -> None:
    result = engine.compute(None)
    assert result.scr_counterparty == 0.0
    assert result.total_lgd_gbp == 0.0
    assert result.num_counterparties == 0


def test_cd1_empty_list_zero(engine: CounterpartyDefaultEngine) -> None:
    result = engine.compute([])
    assert result.scr_counterparty == 0.0
    assert result.num_counterparties == 0


# ---------------------------------------------------------------------------
# CD2: single BBB counterparty (PD=0.005, LGD=100) → SCR ≈ 21.2
# ---------------------------------------------------------------------------

def test_cd2_single_bbb_counterparty(engine: CounterpartyDefaultEngine) -> None:
    # V = 100² × 0.005 × 0.995 = 10000 × 0.004975 = 49.75
    # 3 × sqrt(49.75) ≈ 3 × 7.0534 ≈ 21.16
    exp = CounterpartyExposure(name="BankA", lgd_gbp=100.0, pd=0.005)
    result = engine.compute([exp])

    v = 100.0 ** 2 * 0.005 * 0.995
    expected_scr = min(3.0 * math.sqrt(v), 100.0)

    assert result.scr_counterparty == pytest.approx(expected_scr, rel=1e-6)
    assert result.total_lgd_gbp == pytest.approx(100.0)
    assert result.num_counterparties == 1


def test_cd2_scr_approx_21_2() -> None:
    engine = CounterpartyDefaultEngine()
    exp = CounterpartyExposure(name="BankA", lgd_gbp=100.0, pd=0.005)
    result = engine.compute([exp])
    assert result.scr_counterparty == pytest.approx(21.16, abs=0.01)


# ---------------------------------------------------------------------------
# CD3: SCR ≤ total LGD cap enforced
# ---------------------------------------------------------------------------

def test_cd3_lgd_cap_enforced() -> None:
    engine = CounterpartyDefaultEngine()
    # Very high PD means 3×sqrt(V) can exceed total_lgd
    # PD=0.5: V = LGD² × 0.5 × 0.5 = LGD² × 0.25; 3×sqrt(V) = 3 × LGD × 0.5 = 1.5 × LGD > LGD
    exp = CounterpartyExposure(name="Junk", lgd_gbp=1000.0, pd=0.5)
    result = engine.compute([exp])
    assert result.scr_counterparty <= result.total_lgd_gbp


def test_cd3_cap_value_is_total_lgd() -> None:
    engine = CounterpartyDefaultEngine()
    exp = CounterpartyExposure(name="Junk", lgd_gbp=500.0, pd=0.5)
    result = engine.compute([exp])
    # 3*sqrt(500^2 * 0.25) = 3 * 250 = 750 > 500 → capped at 500
    assert result.scr_counterparty == pytest.approx(500.0)
    assert result.total_lgd_gbp == pytest.approx(500.0)


# ---------------------------------------------------------------------------
# CD4: PD = 0 → SCR = 0
# ---------------------------------------------------------------------------

def test_cd4_zero_pd_zero_scr(engine: CounterpartyDefaultEngine) -> None:
    exp = CounterpartyExposure(name="ZeroPD", lgd_gbp=10_000.0, pd=0.0)
    result = engine.compute([exp])
    assert result.scr_counterparty == 0.0


# ---------------------------------------------------------------------------
# Multi-counterparty: variance is sum of individual variances (no cross terms)
# ---------------------------------------------------------------------------

def test_multi_counterparty_num_recorded(engine: CounterpartyDefaultEngine) -> None:
    exps = [
        CounterpartyExposure(name="A", lgd_gbp=100.0, pd=0.005),
        CounterpartyExposure(name="B", lgd_gbp=200.0, pd=0.010),
    ]
    result = engine.compute(exps)
    assert result.num_counterparties == 2
    assert result.total_lgd_gbp == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Validation: CounterpartyExposure rejects invalid inputs
# ---------------------------------------------------------------------------

def test_negative_lgd_raises() -> None:
    with pytest.raises(ValueError, match="lgd_gbp"):
        CounterpartyExposure(name="Bad", lgd_gbp=-1.0, pd=0.005)


def test_pd_above_one_raises() -> None:
    with pytest.raises(ValueError, match="pd"):
        CounterpartyExposure(name="Bad", lgd_gbp=100.0, pd=1.5)


def test_negative_pd_raises() -> None:
    with pytest.raises(ValueError, match="pd"):
        CounterpartyExposure(name="Bad", lgd_gbp=100.0, pd=-0.1)
