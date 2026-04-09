"""
Investment strategy: SAA rebalancing with AC bond constraint.

InvestmentStrategy computes buy/sell trade orders to move the asset
portfolio towards its Strategic Asset Allocation (SAA) target weights.

AC bond constraint (DECISIONS.md Section 7 / ALM_Architecture.md §8.6)
-----------------------------------------------------------------------
AC-designated bonds must NOT be sold for routine SAA rebalancing.
Selling an AC bond before maturity:
  1. Crystallises a gain/loss previously unrecognised in P&L.
  2. May trigger an IFRS 9 tainting review of the entire AC portfolio.
  3. Has different tax treatment from FVTPL sales.

The constraint is enforced here — not in Bond.rebalance() — so it
applies unconditionally to all SAA-driven trade decisions.

Exceptions (AC bond may be sold):
  - Bond reaches maturity (handled automatically by AssetModel).
  - Explicit force_sell_ac=True in run config (e.g. asset disposal scenario).
  - Cash shortfall: FVTPL assets are sold first; AC only if still insufficient.

Rebalancing logic (per ALM_Architecture.md Section 6.2 Step 5)
---------------------------------------------------------------
1. Compute current actual weights from portfolio market values.
2. Compare to SAA target weights; skip if within rebalancing_tolerance.
3. Identify over-weight and under-weight asset classes.
4. Build sell orders: over-weight positions, excluding AC bonds.
5. Build buy orders: scale existing under-weight positions proportionally.
6. If cash shortfall flag is set: build forced sell orders (FVTPL first).

Trade output
------------
compute_rebalancing_trades() → list[TradeOrder]
    One entry per asset with a non-zero trade.  Fund.py executes each trade
    by calling asset.rebalance(trade.target_value, scenario).

compute_forced_sells() → list[TradeOrder]
    Called separately when Fund detects cash_balance < 0.
    Sells FVTPL assets first; AC only when force_sell_ac=True and still
    insufficient.  Returns the orders needed to restore cash_balance ≥ 0.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.config.fund_config import AssetClassWeights
from engine.strategy.base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class TradeOrder:
    """
    A single buy or sell order for one asset.

    asset_id     : Identifier of the asset to trade.
    target_value : Desired market value after the trade.
                   Pass to asset.rebalance(target_value, scenario).
    trade_amount : Expected cash impact (target_value − current_mv).
                   Positive = buy; negative = sell.
    reason       : Human-readable reason code for audit trail.
    """
    asset_id:     str
    target_value: float
    trade_amount: float
    reason:       str = ""


class InvestmentStrategy(BaseStrategy):
    """
    SAA rebalancing strategy with AC bond sell constraint.

    Parameters
    ----------
    saa_weights : AssetClassWeights
        Strategic Asset Allocation target weights (bonds, equities,
        derivatives, cash).  Must sum to 1.0 (validated by Pydantic).

    rebalancing_tolerance : float
        Percentage band (e.g. 0.05 = ±5%) around SAA target weights.
        If no asset class drifts outside this band, rebalancing is skipped.
        Default: 0.05.

    force_sell_ac : bool
        If True, AC bonds may be sold during routine SAA rebalancing.
        ONLY set in explicit asset disposal run configs.  Default: False.
    """

    def __init__(
        self,
        saa_weights: AssetClassWeights,
        rebalancing_tolerance: float = 0.05,
        force_sell_ac: bool = False,
    ) -> None:
        if not (0.0 <= rebalancing_tolerance <= 1.0):
            raise ValueError(
                f"rebalancing_tolerance must be in [0, 1], "
                f"got {rebalancing_tolerance}"
            )
        self.saa_weights           = saa_weights
        self.rebalancing_tolerance = rebalancing_tolerance
        self.force_sell_ac         = force_sell_ac

    # -----------------------------------------------------------------------
    # Primary interface
    # -----------------------------------------------------------------------

    def compute_rebalancing_trades(
        self,
        asset_model: AssetModel,
        scenario: AssetScenarioPoint,
    ) -> list[TradeOrder]:
        """
        Compute SAA rebalancing orders for this time step.

        Returns an empty list if:
          - The portfolio is empty.
          - No asset class is outside its tolerance band.

        AC bonds are excluded from sell orders unless force_sell_ac=True.
        A warning is logged when the AC constraint prevents full rebalancing.

        Args:
            asset_model : Current asset portfolio.
            scenario    : Economic conditions for this timestep (for MV calc).

        Returns:
            List of TradeOrder.  May be empty.  Fund executes each order via
            asset.rebalance(order.target_value, scenario).
        """
        if len(asset_model) == 0:
            return []

        total_mv = asset_model.total_market_value(scenario)
        if total_mv <= 0.0:
            return []

        # --- Current weights by asset class ---
        mv_by_class = asset_model.market_value_by_class(scenario)
        target_weights = self.saa_weights.model_dump()  # {"bonds": w, "equities": w, ...}

        # --- Check whether any class is outside tolerance ---
        rebalance_needed = False
        for cls, target_w in target_weights.items():
            actual_w = mv_by_class.get(cls, 0.0) / total_mv
            if abs(actual_w - target_w) > self.rebalancing_tolerance:
                rebalance_needed = True
                break

        if not rebalance_needed:
            return []

        # --- Compute target MV per asset class ---
        target_mv_by_class = {
            cls: total_mv * w for cls, w in target_weights.items()
        }

        orders: list[TradeOrder] = []

        # --- Build orders per asset ---
        # Strategy: within each asset class, scale all existing assets
        # proportionally to reach the target class total MV.
        # AC constraint: AC bonds are NOT included in sell candidates.
        for cls, target_class_mv in target_mv_by_class.items():
            class_assets = asset_model.assets_by_class(cls)
            if not class_assets:
                continue

            current_class_mv = mv_by_class.get(cls, 0.0)
            delta = target_class_mv - current_class_mv

            if abs(delta) < 1e-6:
                continue  # already at target

            if delta < 0:
                # Need to SELL in this class.
                # Separate sellable (FVTPL/FVOCI) from constrained (AC).
                sellable = [
                    a for a in class_assets
                    if a.accounting_basis != "AC" or self.force_sell_ac
                ]
                constrained = [
                    a for a in class_assets
                    if a.accounting_basis == "AC" and not self.force_sell_ac
                ]

                sellable_mv = sum(a.market_value(scenario) for a in sellable)
                constrained_mv = sum(a.market_value(scenario) for a in constrained)

                if sellable_mv <= 0.0:
                    if constrained_mv > 0.0:
                        logger.warning(
                            "AC constraint prevents rebalancing of class '%s': "
                            "all holdings are AC-designated and force_sell_ac=False. "
                            "Target weight cannot be reached.",
                            cls,
                        )
                    continue

                # Can we sell enough from sellable assets alone?
                required_sell = abs(delta)
                if sellable_mv < required_sell:
                    logger.warning(
                        "AC constraint partially prevents rebalancing of class '%s': "
                        "sellable MV (%.2f) < required sell (%.2f).  "
                        "Selling all sellable assets; AC bonds not sold.",
                        cls, sellable_mv, required_sell,
                    )
                    # Sell all sellable assets.
                    for asset in sellable:
                        orders.append(TradeOrder(
                            asset_id=asset.asset_id,
                            target_value=0.0,
                            trade_amount=-asset.market_value(scenario),
                            reason="SAA_REBALANCE_SELL_ALL_SELLABLE",
                        ))
                else:
                    # Scale down sellable assets proportionally.
                    sell_fraction = required_sell / sellable_mv
                    for asset in sellable:
                        current_mv = asset.market_value(scenario)
                        new_mv     = current_mv * (1.0 - sell_fraction)
                        orders.append(TradeOrder(
                            asset_id=asset.asset_id,
                            target_value=new_mv,
                            trade_amount=new_mv - current_mv,
                            reason="SAA_REBALANCE_SELL",
                        ))

            else:
                # Need to BUY in this class.
                # Scale up all assets in the class proportionally.
                if current_class_mv <= 0.0:
                    # No existing holdings — cannot buy proportionally.
                    # Fund must decide which specific assets to purchase.
                    logger.warning(
                        "Cannot rebalance into class '%s': no existing holdings "
                        "to scale.  Fund must inject new assets.",
                        cls,
                    )
                    continue

                buy_fraction = delta / current_class_mv
                for asset in class_assets:
                    current_mv = asset.market_value(scenario)
                    new_mv     = current_mv * (1.0 + buy_fraction)
                    orders.append(TradeOrder(
                        asset_id=asset.asset_id,
                        target_value=new_mv,
                        trade_amount=new_mv - current_mv,
                        reason="SAA_REBALANCE_BUY",
                    ))

        return orders

    def compute_forced_sells(
        self,
        asset_model: AssetModel,
        shortfall: float,
        scenario: AssetScenarioPoint,
    ) -> list[TradeOrder]:
        """
        Compute forced sell orders to cover a cash shortfall.

        Called by Fund when cash_balance < 0 (Step 3b / Step 5f of the
        per-time-step sequence).

        Order of preference:
          1. FVTPL assets (sold first — no AC constraint).
          2. FVOCI assets.
          3. AC assets (only if force_sell_ac=True and still insufficient).

        AC assets are not sold unless force_sell_ac=True.  If the shortfall
        cannot be covered without selling AC assets (and force_sell_ac=False),
        a warning is logged and the maximum possible cash is raised from
        non-AC assets only.

        Args:
            asset_model : Current asset portfolio.
            shortfall   : Cash needed (positive number, > 0).
            scenario    : Current economic conditions.

        Returns:
            List of TradeOrder sufficient to cover shortfall (or as close as
            possible under the AC constraint).
        """
        if shortfall <= 0.0:
            return []

        orders:       list[TradeOrder] = []
        still_needed: float            = shortfall

        def sell_from(assets_to_sell: list) -> None:
            nonlocal still_needed
            for asset in assets_to_sell:
                if still_needed <= 0.0:
                    break
                current_mv = asset.market_value(scenario)
                if current_mv <= 0.0:
                    continue
                sell_amount = min(still_needed, current_mv)
                new_mv      = current_mv - sell_amount
                orders.append(TradeOrder(
                    asset_id=asset.asset_id,
                    target_value=new_mv,
                    trade_amount=-sell_amount,
                    reason="FORCED_SELL_CASH_SHORTFALL",
                ))
                still_needed -= sell_amount

        # Preference 1: FVTPL
        sell_from(asset_model.assets_by_basis("FVTPL"))

        # Preference 2: FVOCI
        if still_needed > 0.0:
            sell_from(asset_model.assets_by_basis("FVOCI"))

        # Preference 3: AC (only if force_sell_ac=True)
        if still_needed > 0.0:
            if self.force_sell_ac:
                sell_from(asset_model.assets_by_basis("AC"))
            else:
                logger.warning(
                    "Cash shortfall of %.2f cannot be fully covered: "
                    "remaining %.2f requires selling AC bonds, but "
                    "force_sell_ac=False.  Shortfall will persist.",
                    shortfall,
                    still_needed,
                )

        return orders

    def rebalancing_needed(
        self,
        asset_model: AssetModel,
        scenario: AssetScenarioPoint,
    ) -> bool:
        """
        Return True if any asset class is outside its SAA tolerance band.

        Convenience method for Fund to decide whether to call
        compute_rebalancing_trades() this period.

        Args:
            asset_model : Current asset portfolio.
            scenario    : Current economic conditions.
        """
        if len(asset_model) == 0:
            return False
        total_mv = asset_model.total_market_value(scenario)
        if total_mv <= 0.0:
            return False

        mv_by_class    = asset_model.market_value_by_class(scenario)
        target_weights = self.saa_weights.model_dump()

        for cls, target_w in target_weights.items():
            actual_w = mv_by_class.get(cls, 0.0) / total_mv
            if abs(actual_w - target_w) > self.rebalancing_tolerance:
                return True
        return False
