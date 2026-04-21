"""
tests/unit/strategy/test_buy_and_hold_strategy.py

Tests for BuyAndHoldStrategy (DECISIONS.md §46).

Rules under test
----------------
1.  rebalancing_needed() returns False with an empty portfolio.
2.  rebalancing_needed() returns False even when portfolio is severely imbalanced.
3.  compute_rebalancing_trades() returns [] with a balanced portfolio.
4.  compute_rebalancing_trades() returns [] even when portfolio is severely imbalanced.
5.  compute_forced_sells() returns FVTPL sell orders when cash shortfall exists.
6.  compute_forced_sells() does not generate AC sell orders when force_sell_ac=False (default).
7.  compute_forced_sells() generates AC sell orders only when force_sell_ac=True
    and FVTPL proceeds are insufficient to cover shortfall.
8.  strategy.name == "BuyAndHoldStrategy".
"""
from __future__ import annotations

import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.bond import Bond
from engine.asset.base_asset import AssetScenarioPoint
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.strategy.buy_and_hold_strategy import BuyAndHoldStrategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scenario() -> AssetScenarioPoint:
    return AssetScenarioPoint(
        timestep=0,
        rate_curve=RiskFreeRateCurve.flat(0.03),
        equity_total_return_yr=0.0,
    )


def make_ac_bond(asset_id: str = "ac_01", face: float = 1_000_000.0) -> Bond:
    return Bond(asset_id, face, 0.05, 120, "AC", face * 0.95)


def make_fvtpl_bond(asset_id: str = "fv_01", face: float = 500_000.0) -> Bond:
    return Bond(asset_id, face, 0.05, 60, "FVTPL", face)


def make_portfolio_ac_only() -> AssetModel:
    am = AssetModel()
    am.add_asset(make_ac_bond("ac_01", 1_000_000.0))
    am.add_asset(make_ac_bond("ac_02", 1_000_000.0))
    return am


def make_portfolio_fvtpl_only() -> AssetModel:
    am = AssetModel()
    am.add_asset(make_fvtpl_bond("fv_01", 500_000.0))
    am.add_asset(make_fvtpl_bond("fv_02", 500_000.0))
    return am


def make_mixed_portfolio() -> AssetModel:
    am = AssetModel()
    am.add_asset(make_ac_bond("ac_01", 2_000_000.0))
    am.add_asset(make_fvtpl_bond("fv_01", 500_000.0))
    return am


# ---------------------------------------------------------------------------
# Test 1 & 2 — rebalancing_needed always False
# ---------------------------------------------------------------------------

class TestRebalancingNeeded:

    def test_returns_false_empty_portfolio(self):
        """Test 1 — always False even with no assets."""
        strategy = BuyAndHoldStrategy()
        am = AssetModel()
        assert strategy.rebalancing_needed(am, make_scenario()) is False

    def test_returns_false_severely_imbalanced_portfolio(self):
        """Test 2 — False even when portfolio is 100% in one asset class."""
        strategy = BuyAndHoldStrategy()
        am = make_portfolio_ac_only()  # 100% AC bonds, no FVTPL, no equities
        assert strategy.rebalancing_needed(am, make_scenario()) is False


# ---------------------------------------------------------------------------
# Test 3 & 4 — compute_rebalancing_trades always []
# ---------------------------------------------------------------------------

class TestComputeRebalancingTrades:

    def test_returns_empty_list_balanced_portfolio(self):
        """Test 3 — always [] regardless of portfolio state."""
        strategy = BuyAndHoldStrategy()
        am = make_mixed_portfolio()
        assert strategy.compute_rebalancing_trades(am, make_scenario()) == []

    def test_returns_empty_list_severely_imbalanced(self):
        """Test 4 — [] even when portfolio is 100% AC (no FVTPL at all)."""
        strategy = BuyAndHoldStrategy()
        am = make_portfolio_ac_only()
        assert strategy.compute_rebalancing_trades(am, make_scenario()) == []


# ---------------------------------------------------------------------------
# Test 5 — forced sells: FVTPL sold for cash shortfall
# ---------------------------------------------------------------------------

class TestForcedSells:

    def test_fvtpl_sold_for_cash_shortfall(self):
        """Test 5 — FVTPL bond is sold when cash shortfall exists."""
        strategy = BuyAndHoldStrategy()
        am = make_portfolio_fvtpl_only()  # two FVTPL bonds ~£500k each
        scenario = make_scenario()

        orders = strategy.compute_forced_sells(am, shortfall=300_000.0, scenario=scenario)

        assert len(orders) > 0
        total_raised = sum(abs(o.trade_amount) for o in orders)
        assert total_raised == pytest.approx(300_000.0, rel=0.01)
        for o in orders:
            assert o.reason == "FORCED_SELL_CASH_SHORTFALL"

    def test_no_orders_when_shortfall_zero(self):
        """Zero shortfall produces no orders."""
        strategy = BuyAndHoldStrategy()
        am = make_mixed_portfolio()
        orders = strategy.compute_forced_sells(am, shortfall=0.0, scenario=make_scenario())
        assert orders == []

    def test_no_orders_when_shortfall_negative(self):
        """Negative shortfall (surplus) produces no orders."""
        strategy = BuyAndHoldStrategy()
        am = make_mixed_portfolio()
        orders = strategy.compute_forced_sells(am, shortfall=-100.0, scenario=make_scenario())
        assert orders == []

    def test_ac_not_sold_when_force_sell_ac_false(self):
        """Test 6 — AC bonds are NOT sold when force_sell_ac=False (default)."""
        strategy = BuyAndHoldStrategy(force_sell_ac=False)
        am = make_portfolio_ac_only()  # only AC bonds — no FVTPL available
        scenario = make_scenario()

        orders = strategy.compute_forced_sells(am, shortfall=500_000.0, scenario=scenario)

        # No orders should be generated — AC bonds are protected
        ac_ids = {a.asset_id for a in am.assets_by_basis("AC")}
        for o in orders:
            assert o.asset_id not in ac_ids, (
                f"AC bond {o.asset_id} was sold despite force_sell_ac=False"
            )

    def test_ac_sold_when_force_sell_ac_true_and_fvtpl_insufficient(self):
        """
        Test 7 — AC bonds ARE sold when force_sell_ac=True and FVTPL
        proceeds alone cannot cover the shortfall.
        """
        strategy = BuyAndHoldStrategy(force_sell_ac=True)
        am = make_mixed_portfolio()   # AC bond £2m + FVTPL bond £500k
        scenario = make_scenario()
        # Request £1m: FVTPL ~£500k is not enough alone → AC must contribute
        orders = strategy.compute_forced_sells(am, shortfall=1_000_000.0, scenario=scenario)

        total_raised = sum(abs(o.trade_amount) for o in orders)
        assert total_raised == pytest.approx(1_000_000.0, rel=0.01)

        sold_ids = {o.asset_id for o in orders}
        # FVTPL bond should be sold
        assert "fv_01" in sold_ids
        # AC bond should also appear (needed to cover remainder)
        assert "ac_01" in sold_ids


# ---------------------------------------------------------------------------
# Test 8 — strategy.name
# ---------------------------------------------------------------------------

class TestStrategyName:

    def test_name_is_class_name(self):
        """Test 8 — strategy.name returns 'BuyAndHoldStrategy'."""
        strategy = BuyAndHoldStrategy()
        assert strategy.name == "BuyAndHoldStrategy"
