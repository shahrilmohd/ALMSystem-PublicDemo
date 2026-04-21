"""
engine/strategy/buy_and_hold_strategy.py
=========================================
BuyAndHoldStrategy — no-op rebalancing for BPA MA portfolios.

Design rationale (DECISIONS.md §46)
-------------------------------------
BPA MA portfolios are held to maturity by regulatory design.  Active
rebalancing would:

1. Break the PRA cashflow matching test (condition 4, §22) — any trade
   changes the asset CF schedule, requiring the test to be re-run and
   potentially invalidating the MA benefit.
2. Risk IFRS 9 tainting of the AC portfolio — selling AC bonds before
   maturity crystallises gains/losses and may trigger a portfolio-wide
   reclassification review (§7).
3. Require re-certification of MA eligibility on every reporting date,
   which is operationally impractical and inconsistent with PRA practice.

BuyAndHoldStrategy satisfies the Fund interface (Fund always receives a
strategy object) but suppresses all routine rebalancing.  Forced sells for
cash shortfalls remain active — they are handled identically to
InvestmentStrategy: FVTPL first, FVOCI second, AC only when
force_sell_ac=True and FVTPL/FVOCI are insufficient.

This class is injected by BPARun only.  It must never be injected into a
conventional DeterministicRun or StochasticRun — those require active SAA
rebalancing to maintain target risk exposure.
"""
from __future__ import annotations

import logging

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.strategy.base_strategy import BaseStrategy
from engine.strategy.investment_strategy import TradeOrder

logger = logging.getLogger(__name__)


class BuyAndHoldStrategy(BaseStrategy):
    """
    No-op rebalancing strategy for BPA MA portfolios.

    ``compute_rebalancing_trades()`` always returns ``[]``.
    ``rebalancing_needed()``         always returns ``False``.
    ``compute_forced_sells()``        behaves identically to
        ``InvestmentStrategy.compute_forced_sells()``:
        FVTPL first, FVOCI second, AC only when ``force_sell_ac=True``.

    Parameters
    ----------
    force_sell_ac : bool
        When True, AC bonds may be sold as a last resort during a cash
        shortfall after all FVTPL and FVOCI assets are exhausted.
        Default False — AC bonds are fully protected, matching the MA
        regulatory intent of holding them to maturity.
    """

    def __init__(self, force_sell_ac: bool = False) -> None:
        self.force_sell_ac = force_sell_ac

    # -----------------------------------------------------------------------
    # Rebalancing — always no-op
    # -----------------------------------------------------------------------

    def rebalancing_needed(
        self,
        asset_model: AssetModel,
        scenario: AssetScenarioPoint,
    ) -> bool:
        """Always returns False — BPA MA portfolios are never rebalanced."""
        return False

    def compute_rebalancing_trades(
        self,
        asset_model: AssetModel,
        scenario: AssetScenarioPoint,
    ) -> list[TradeOrder]:
        """Always returns [] — BPA MA portfolios are held to maturity."""
        return []

    # -----------------------------------------------------------------------
    # Forced sells — active for cash shortfalls
    # -----------------------------------------------------------------------

    def compute_forced_sells(
        self,
        asset_model: AssetModel,
        shortfall: float,
        scenario: AssetScenarioPoint,
    ) -> list[TradeOrder]:
        """
        Compute forced sell orders to cover a cash shortfall.

        Order of preference:
          1. FVTPL assets (sold first — no AC constraint).
          2. FVOCI assets.
          3. AC assets (only if force_sell_ac=True and still insufficient).

        If the shortfall cannot be covered without selling AC bonds and
        ``force_sell_ac=False``, a warning is logged and the maximum
        recoverable cash from non-AC assets is raised.  The residual
        shortfall persists — the fund will carry a negative cash balance.

        Parameters
        ----------
        asset_model : AssetModel
            Current asset portfolio.
        shortfall : float
            Cash needed (positive, > 0).
        scenario : AssetScenarioPoint
            Current economic conditions for market value lookup.

        Returns
        -------
        list[TradeOrder]
            Orders sufficient to cover the shortfall (or as close as
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

        # Preference 3: AC (only when explicitly permitted)
        if still_needed > 0.0:
            if self.force_sell_ac:
                sell_from(asset_model.assets_by_basis("AC"))
            else:
                logger.warning(
                    "BuyAndHoldStrategy: cash shortfall of %.2f cannot be "
                    "fully covered. Remaining %.2f requires selling AC bonds, "
                    "but force_sell_ac=False. Shortfall will persist.",
                    shortfall,
                    still_needed,
                )

        return orders
