"""
Unit tests for engine/strategy/investment_strategy.py.

Key architectural rule under test: AC-designated bonds must NOT be sold
for routine SAA rebalancing (DECISIONS.md Section 7).
"""
from __future__ import annotations

import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.bond import Bond
from engine.asset.equity import Equity
from engine.asset.base_asset import AssetScenarioPoint
from engine.config.fund_config import AssetClassWeights
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.strategy.investment_strategy import InvestmentStrategy, TradeOrder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scenario(timestep: int = 0) -> AssetScenarioPoint:
    return AssetScenarioPoint(
        timestep=timestep,
        rate_curve=RiskFreeRateCurve.flat(0.03),
        equity_total_return_yr=0.08,
    )


def make_weights(bonds: float = 0.6, equities: float = 0.3,
                 cash: float = 0.1, derivatives: float = 0.0) -> AssetClassWeights:
    return AssetClassWeights(
        bonds=bonds, equities=equities, cash=cash, derivatives=derivatives
    )


def make_ac_bond(asset_id: str, face: float = 1_000_000.0) -> Bond:
    return Bond(asset_id, face, 0.05, 36, "AC", face * 0.95)


def make_fvtpl_bond(asset_id: str, face: float = 1_000_000.0) -> Bond:
    return Bond(asset_id, face, 0.05, 36, "FVTPL", face)


def make_equity(asset_id: str, mv: float = 500_000.0) -> Equity:
    return Equity(asset_id, mv, dividend_yield_yr=0.03)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestStrategyConstruction:

    def test_valid_construction(self):
        strat = InvestmentStrategy(make_weights(), rebalancing_tolerance=0.05)
        assert strat.rebalancing_tolerance == 0.05
        assert strat.force_sell_ac is False

    def test_invalid_tolerance_raises(self):
        with pytest.raises(ValueError, match="rebalancing_tolerance"):
            InvestmentStrategy(make_weights(), rebalancing_tolerance=1.5)

    def test_name_property(self):
        strat = InvestmentStrategy(make_weights())
        assert strat.name == "InvestmentStrategy"


# ---------------------------------------------------------------------------
# AC sell constraint — the critical rule (DECISIONS.md Section 7)
# ---------------------------------------------------------------------------

class TestACConstraint:

    def test_ac_bond_not_sold_in_routine_rebalancing(self):
        """
        When bond class is over-weight and the only bond is AC,
        no sell order should be generated (AC constraint enforced).
        """
        # 100% bonds (all AC), target is 60/40 bonds/equities.
        # Strategy would need to sell bonds, but they're all AC.
        ac_bond = make_ac_bond("b_ac", 1_000_000.0)
        am      = AssetModel([ac_bond])
        strat   = InvestmentStrategy(
            make_weights(bonds=0.6, equities=0.4, cash=0.0),
            rebalancing_tolerance=0.0,
            force_sell_ac=False,
        )
        s      = make_scenario()
        orders = strat.compute_rebalancing_trades(am, s)

        # No sell orders for the AC bond
        sell_orders = [o for o in orders if o.trade_amount < 0]
        ac_sells    = [o for o in sell_orders if o.asset_id == "b_ac"]
        assert len(ac_sells) == 0

    def test_fvtpl_bond_sold_when_over_weight(self):
        """
        FVTPL bonds can be sold in routine SAA rebalancing.
        """
        # Only FVTPL bonds, over-weight in bonds class.
        fv_bond = make_fvtpl_bond("b_fv", 1_000_000.0)
        am      = AssetModel([fv_bond])
        strat   = InvestmentStrategy(
            make_weights(bonds=0.5, equities=0.0, cash=0.5),
            rebalancing_tolerance=0.0,
            force_sell_ac=False,
        )
        orders = strat.compute_rebalancing_trades(am, make_scenario())
        sell_orders = [o for o in orders if o.trade_amount < 0]
        fv_sells    = [o for o in sell_orders if o.asset_id == "b_fv"]
        assert len(fv_sells) == 1

    def test_force_sell_ac_overrides_constraint(self):
        """
        With force_sell_ac=True, AC bonds CAN be sold in rebalancing.
        """
        ac_bond = make_ac_bond("b_ac", 1_000_000.0)
        am      = AssetModel([ac_bond])
        strat   = InvestmentStrategy(
            make_weights(bonds=0.5, equities=0.0, cash=0.5),
            rebalancing_tolerance=0.0,
            force_sell_ac=True,
        )
        orders   = strat.compute_rebalancing_trades(am, make_scenario())
        ac_sells = [o for o in orders if o.asset_id == "b_ac" and o.trade_amount < 0]
        assert len(ac_sells) == 1

    def test_mixed_portfolio_sells_only_fvtpl(self):
        """
        Portfolio with AC + FVTPL bonds in bond class over-weight.
        Only the FVTPL bond should be in the sell order.
        """
        ac_bond = make_ac_bond("b_ac", 800_000.0)
        fv_bond = make_fvtpl_bond("b_fv", 800_000.0)
        eq      = make_equity("eq1", 400_000.0)
        am      = AssetModel([ac_bond, fv_bond, eq])
        # Current: bonds=80% (1.6m / 2m), equities=20%.  Target: bonds=40%, equities=60%.
        strat  = InvestmentStrategy(
            make_weights(bonds=0.4, equities=0.6, cash=0.0),
            rebalancing_tolerance=0.0,
        )
        orders    = strat.compute_rebalancing_trades(am, make_scenario())
        sell_ids  = {o.asset_id for o in orders if o.trade_amount < 0}
        assert "b_fv" in sell_ids
        assert "b_ac" not in sell_ids


# ---------------------------------------------------------------------------
# Rebalancing logic
# ---------------------------------------------------------------------------

class TestRebalancingLogic:

    def test_no_rebalancing_when_within_tolerance(self):
        """No orders when weights are within the tolerance band."""
        b  = make_fvtpl_bond("b1", 600_000.0)
        eq = make_equity("eq1", 400_000.0)
        am = AssetModel([b, eq])
        # Current weights: bonds=60%, equities=40%. Target: bonds=60%, equities=40%.
        strat  = InvestmentStrategy(
            make_weights(bonds=0.6, equities=0.4, cash=0.0),
            rebalancing_tolerance=0.05,
        )
        orders = strat.compute_rebalancing_trades(am, make_scenario())
        assert orders == []

    def test_rebalancing_needed_detects_drift(self):
        b  = make_fvtpl_bond("b1", 900_000.0)
        eq = make_equity("eq1", 100_000.0)
        am = AssetModel([b, eq])
        strat = InvestmentStrategy(
            make_weights(bonds=0.6, equities=0.4, cash=0.0),
            rebalancing_tolerance=0.05,
        )
        assert strat.rebalancing_needed(am, make_scenario()) is True

    def test_rebalancing_not_needed_at_target(self):
        b  = make_fvtpl_bond("b1", 600_000.0)
        eq = make_equity("eq1", 400_000.0)
        am = AssetModel([b, eq])
        strat = InvestmentStrategy(
            make_weights(bonds=0.6, equities=0.4, cash=0.0),
            rebalancing_tolerance=0.05,
        )
        assert strat.rebalancing_needed(am, make_scenario()) is False

    def test_empty_portfolio_no_orders(self):
        am    = AssetModel()
        strat = InvestmentStrategy(make_weights())
        assert strat.compute_rebalancing_trades(am, make_scenario()) == []

    def test_sell_order_target_value_less_than_current(self):
        """Sell orders must have target_value < current_mv."""
        fv = make_fvtpl_bond("b1", 1_000_000.0)
        am = AssetModel([fv])
        strat = InvestmentStrategy(
            make_weights(bonds=0.5, equities=0.0, cash=0.5),
            rebalancing_tolerance=0.0,
        )
        s      = make_scenario()
        orders = strat.compute_rebalancing_trades(am, s)
        for order in orders:
            if order.trade_amount < 0:
                assert order.target_value < fv.market_value(s)

    def test_buy_order_target_value_greater_than_current(self):
        """Buy orders must have target_value > current_mv."""
        fv = make_fvtpl_bond("b1", 400_000.0)
        eq = make_equity("eq1", 100_000.0)
        am = AssetModel([fv, eq])
        # Target: bonds=80%, equities=20%.  Currently bonds=80%, eq=20% → no rebal needed
        # Make it so bonds under-weight
        strat = InvestmentStrategy(
            make_weights(bonds=0.9, equities=0.1, cash=0.0),
            rebalancing_tolerance=0.0,
        )
        s      = make_scenario()
        orders = strat.compute_rebalancing_trades(am, s)
        for order in orders:
            if order.trade_amount > 0:
                asset    = am.get_asset(order.asset_id)
                curr_mv  = asset.market_value(s)
                assert order.target_value > curr_mv


# ---------------------------------------------------------------------------
# Forced sells — cash shortfall
# ---------------------------------------------------------------------------

class TestForcedSells:

    def test_forced_sell_fvtpl_first(self):
        """
        Cash shortfall: FVTPL bonds sold before AC bonds.
        """
        ac_bond = make_ac_bond("b_ac", 1_000_000.0)
        fv_bond = make_fvtpl_bond("b_fv", 500_000.0)
        am      = AssetModel([ac_bond, fv_bond])
        strat   = InvestmentStrategy(make_weights(), force_sell_ac=False)

        shortfall = 200_000.0
        orders    = strat.compute_forced_sells(am, shortfall, make_scenario())

        # Should only sell FVTPL to cover 200k shortfall
        assert len(orders) >= 1
        sold_ids = {o.asset_id for o in orders}
        assert "b_fv" in sold_ids
        assert "b_ac" not in sold_ids

    def test_forced_sell_covers_shortfall(self):
        fv_bond   = make_fvtpl_bond("b_fv", 1_000_000.0)
        am        = AssetModel([fv_bond])
        strat     = InvestmentStrategy(make_weights())
        s         = make_scenario()
        shortfall = 300_000.0
        orders    = strat.compute_forced_sells(am, shortfall, s)

        total_raised = sum(-o.trade_amount for o in orders)
        assert total_raised >= shortfall - 1.0   # within £1 rounding

    def test_forced_sell_ac_only_when_fvtpl_insufficient(self):
        """
        If FVTPL is insufficient and force_sell_ac=True, AC bonds are also sold.
        """
        ac_bond = make_ac_bond("b_ac", 1_000_000.0)
        fv_bond = make_fvtpl_bond("b_fv", 100_000.0)
        am      = AssetModel([ac_bond, fv_bond])
        strat   = InvestmentStrategy(make_weights(), force_sell_ac=True)
        s       = make_scenario()

        # Shortfall larger than FVTPL portfolio
        shortfall = 500_000.0
        orders    = strat.compute_forced_sells(am, shortfall, s)

        total_raised = sum(-o.trade_amount for o in orders)
        assert total_raised >= shortfall - 1.0
        sold_ids = {o.asset_id for o in orders}
        assert "b_ac" in sold_ids   # AC bond needed to fill the gap

    def test_forced_sell_zero_shortfall_returns_empty(self):
        am    = AssetModel([make_fvtpl_bond("b1")])
        strat = InvestmentStrategy(make_weights())
        assert strat.compute_forced_sells(am, 0.0, make_scenario()) == []

    def test_forced_sell_negative_shortfall_returns_empty(self):
        am    = AssetModel([make_fvtpl_bond("b1")])
        strat = InvestmentStrategy(make_weights())
        assert strat.compute_forced_sells(am, -100.0, make_scenario()) == []

    def test_forced_sell_ac_not_sold_when_force_sell_ac_false(self):
        """With force_sell_ac=False, AC bonds are never touched even if shortfall persists."""
        ac_bond   = make_ac_bond("b_ac", 1_000_000.0)
        am        = AssetModel([ac_bond])       # only AC in portfolio
        strat     = InvestmentStrategy(make_weights(), force_sell_ac=False)
        orders    = strat.compute_forced_sells(am, 500_000.0, make_scenario())
        # No orders — cannot sell AC, shortfall persists (warning logged)
        ac_sells  = [o for o in orders if o.asset_id == "b_ac"]
        assert len(ac_sells) == 0


# ---------------------------------------------------------------------------
# TradeOrder
# ---------------------------------------------------------------------------

class TestTradeOrder:

    def test_trade_order_fields(self):
        order = TradeOrder(
            asset_id="b1",
            target_value=800_000.0,
            trade_amount=-200_000.0,
            reason="SAA_REBALANCE_SELL",
        )
        assert order.asset_id == "b1"
        assert order.trade_amount == -200_000.0
        assert order.reason == "SAA_REBALANCE_SELL"
