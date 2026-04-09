"""
Fund — coordinates asset and liability models for one projection period.

Design
------
Fund receives its asset model, liability model, and investment strategy at
construction (dependency injection — CLAUDE.md Rule 5). It never loads data,
reads files, or holds run configuration.

Module dependencies
-------------------
Fund imports from four modules. Here is exactly what it needs from each:

    base_liability.py
        BaseLiability       — type annotation for liability_model parameter.
                              Fund calls: get_decrements(), project_cashflows().
        LiabilityCashflows  — return type of project_cashflows().
        Decrements          — return type of get_decrements().

    asset_model.py
        AssetModel          — type annotation for asset_model parameter.
                              Fund calls: collect_cashflows(scenario),
                              step_time(scenario), aggregate_pnl(),
                              market_value_by_basis(scenario),
                              total_market_value(scenario),
                              total_book_value(), get_asset(asset_id).

    investment_strategy.py
        InvestmentStrategy  — type annotation for investment_strategy parameter.
                              Fund calls: rebalancing_needed(asset_model, scenario),
                              compute_rebalancing_trades(asset_model, scenario),
                              compute_forced_sells(asset_model, shortfall, scenario).

    base_asset.py
        AssetScenarioPoint  — data class carrying rate_curve, equity_total_return_yr,
                              and timestep for the current period.
                              Fund passes this object unchanged to every AssetModel
                              and InvestmentStrategy method call.
                              BaseAsset itself is NOT imported — Fund never
                              interacts with individual assets directly except
                              via asset_model.get_asset(id).rebalance(), which
                              is an individual asset call needed only to execute
                              trade orders from the strategy.

Rate curve data flow
--------------------
Fund receives AssetScenarioPoint which carries a RiskFreeRateCurve.
Fund passes the full scenario object down to AssetModel methods.
AssetModel passes it to each individual bond/equity.
Bond.market_value() calls scenario.rate_curve.discount_factor(month)
to discount its future cash flows and arrive at a market value.
The rate_curve never leaves AssetScenarioPoint until Bond actually
needs to price itself.

    Fund.step_time(scenario)
        └─► asset_model.collect_cashflows(scenario)
                └─► bond.project_cashflows(scenario)
                        └─► bond.market_value(scenario)
                                └─► scenario.rate_curve.discount_factor(t)
        └─► asset_model.step_time(scenario)
                └─► bond.step_time(scenario)
                        └─► bond.market_value(scenario)   [FVTPL/FVOCI repricing]
                                └─► scenario.rate_curve.discount_factor(t)
        └─► asset_model.market_value_by_basis(scenario)
                └─► bond.market_value(scenario)
                        └─► scenario.rate_curve.discount_factor(t)

Per-step sequence (Fund.step_time())
-------------------------------------
1. Collect liability cashflows and decrements from the liability model.
2. Collect asset income (coupons, dividends, maturity proceeds) via
   AssetModel.collect_cashflows(). Matured bonds are removed here.
3. Update cash balance: cash += asset_income; cash -= net_liability_outflow.
4. Forced sells (if cash < 0): call investment_strategy.compute_forced_sells();
   execute via asset.rebalance(); update cash for proceeds received.
5. SAA rebalancing (if drift > tolerance): call compute_rebalancing_trades();
   execute via asset.rebalance(); update cash for each buy/sell.
6. Call asset_model.step_time(scenario) — advance book values and capture
   realised G/L from the rebalancing trades executed in steps 4-5.
   IMPORTANT: step_time() MUST follow rebalancing. Bond.rebalance() accumulates
   _realised_gl_this_period; Bond.step_time() incorporates it into _last_pnl
   and resets it. Reversing the order misplaces realised G/L. (DECISIONS.md §11)
7. Collect period P&L and closing valuations via aggregate_pnl() and
   market_value_by_basis().

BEL is NOT computed here. It requires a full forward projection of future
cashflows and is computed by the run mode in a backward pass.
(See DECISIONS.md §12 and DeterministicRun.execute().)

Architectural rules
-------------------
- engine/ has zero imports from frontend/, api/, or worker/.
- Fund never reads files or databases — all inputs are injected.
- Results are never stored inside Fund — step_time() returns FundTimestepResult.
- The time loop lives in the run mode, not inside Fund.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.liability.base_liability import BaseLiability, Decrements, LiabilityCashflows
from engine.strategy.investment_strategy import InvestmentStrategy


# ---------------------------------------------------------------------------
# Result data types
# ---------------------------------------------------------------------------

@dataclass
class AssetTimestepResult:
    """
    Portfolio-level asset outputs for one (scenario_id, timestep).

    All monetary values are in the fund's currency unit.
    Income and G/L fields cover the period ending at this timestep.
    By-basis MV fields are closing values after all trades for the period.
    """
    total_market_value: float  # Total MV of all non-cash assets
    total_book_value:   float  # Total BV (amortised for AC; = MV for FVTPL/FVOCI)
    cash_balance:       float  # Fund cash after all movements this period

    # Investment income recognised in P&L
    eir_income:         float  # AC/FVOCI: EIR amortisation
    coupon_income:      float  # FVTPL: coupon cash received
    dividend_income:    float  # Equities: dividend cash received

    # Capital gains recognised in P&L
    unrealised_gl:      float  # FVTPL only (disclosed-only for AC; OCI for FVOCI)
    realised_gl:        float  # All bases: on rebalancing / forced sells

    # OCI balance (FVOCI cumulative since inception)
    oci_reserve:        float

    # Closing market value by accounting basis
    mv_ac:              float
    mv_fvtpl:           float
    mv_fvoci:           float


@dataclass
class FundTimestepResult:
    """
    Combined liability and asset outputs for one projection period.

    Produced by Fund.step_time() and consumed by the run mode to populate
    ResultStore. BEL is excluded — computed by the run mode in a backward pass.
    """
    cashflows:  LiabilityCashflows
    decrements: Decrements
    asset:      AssetTimestepResult


# ---------------------------------------------------------------------------
# Fund
# ---------------------------------------------------------------------------

class Fund:
    """
    Coordinates asset and liability models for a single fund projection.

    All dependencies are injected at construction. Fund does not read files,
    access databases, or hold run configuration.

    Parameters
    ----------
    asset_model : AssetModel
        Portfolio container. Mutated in-place each step_time() call.
        AssetModel passes scenario (including rate_curve) to each bond/equity
        for market value calculations and book value advancement.
    liability_model : BaseLiability
        Stateless liability projection model. Called with model_points and
        assumptions each step_time() call.
    investment_strategy : InvestmentStrategy
        Computes rebalancing orders and forced sells. Injected per CLAUDE.md Rule 5.
    initial_cash : float
        Opening cash balance at the start of the projection (£). Default: 0.0.
    """

    def __init__(
        self,
        asset_model:         AssetModel,
        liability_model:     BaseLiability,
        investment_strategy: InvestmentStrategy,
        initial_cash:        float = 0.0,
    ) -> None:
        self._asset_model         = asset_model
        self._liability_model     = liability_model
        self._investment_strategy = investment_strategy
        self._cash_balance        = initial_cash
        self._logger              = logging.getLogger(self.__class__.__name__)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def cash_balance(self) -> float:
        """Current fund cash balance. Updated by each step_time() call."""
        return self._cash_balance

    @property
    def asset_model(self) -> AssetModel:
        """Read-only access to the asset portfolio for inspection by run modes."""
        return self._asset_model

    # -----------------------------------------------------------------------
    # step_time — single projection period
    # -----------------------------------------------------------------------

    def step_time(
        self,
        scenario:     AssetScenarioPoint,
        model_points: pd.DataFrame,
        assumptions:  Any,
    ) -> FundTimestepResult:
        """
        Advance the fund by one projection period.

        The scenario carries the rate_curve used to price every bond in the
        portfolio. Fund passes the full scenario object to AssetModel methods.
        AssetModel passes it to each individual bond. Bond.market_value() calls
        scenario.rate_curve.discount_factor() to discount its cash flows.

        Parameters
        ----------
        scenario : AssetScenarioPoint
            Current-period risk-free yield curve and equity return assumption.
            Passed unchanged to all AssetModel and InvestmentStrategy calls.
        model_points : pd.DataFrame
            In-force liability data at the START of this period.
            Not mutated here — the run mode advances model points between steps.
        assumptions : Any
            Liability assumption object (e.g. ConventionalAssumptions).
            Passed unchanged to the liability model.

        Returns
        -------
        FundTimestepResult
            Liability cashflows, decrements, and portfolio-level asset result.
            BEL is excluded — the run mode computes it in a backward pass.
        """
        t = scenario.timestep

        # ------------------------------------------------------------------
        # Step 1 — Liability cashflows and decrements
        # ------------------------------------------------------------------
        decrements = self._liability_model.get_decrements(model_points, assumptions, t)
        cashflows  = self._liability_model.project_cashflows(model_points, assumptions, t)

        # ------------------------------------------------------------------
        # Step 2 — Collect asset income
        # scenario (with rate_curve) flows through here: collect_cashflows
        # passes scenario to each bond, which calls scenario.rate_curve
        # .discount_factor() to price itself and compute coupon / maturity
        # proceeds. Matured bonds are removed from the portfolio here.
        # ------------------------------------------------------------------
        asset_cfs    = self._asset_model.collect_cashflows(scenario)
        asset_income = (
            asset_cfs.coupon_income
            + asset_cfs.dividend_income
            + asset_cfs.maturity_proceeds
        )

        # ------------------------------------------------------------------
        # Step 3 — Update cash balance
        # net_outgo > 0: fund pays out more than it receives (outflow).
        # net_outgo < 0: premiums exceed claims (cash inflow to fund).
        # ------------------------------------------------------------------
        self._cash_balance += asset_income
        self._cash_balance -= cashflows.net_outgo

        # ------------------------------------------------------------------
        # Step 4 — Forced sells if cash balance is negative
        # FVTPL / FVOCI sold first; AC only if force_sell_ac=True.
        # asset_model.get_asset(id) retrieves the bond/equity object;
        # asset.rebalance(target_value, scenario) uses scenario.rate_curve
        # to compute current MV, then scales the position and returns the
        # trade amount (negative = sell, positive = buy).
        # ------------------------------------------------------------------
        if self._cash_balance < 0.0:
            shortfall = -self._cash_balance
            forced = self._investment_strategy.compute_forced_sells(
                self._asset_model, shortfall, scenario
            )
            for order in forced:
                asset = self._asset_model.get_asset(order.asset_id)
                trade = asset.rebalance(order.target_value, scenario)
                self._cash_balance -= trade   # sell → trade < 0 → cash increases
            if self._cash_balance < -1.0:     # £1 rounding tolerance
                self._logger.warning(
                    "Timestep %d: cash shortfall of £%.2f remains after forced sells "
                    "(AC constraint may be preventing required sales).",
                    t, -self._cash_balance,
                )

        # ------------------------------------------------------------------
        # Step 5 — SAA rebalancing
        # rebalancing_needed() computes actual weights using scenario.rate_curve
        # to price each asset, then compares against SAA targets.
        # compute_rebalancing_trades() returns TradeOrder objects for existing
        # assets. AC bonds excluded from sell orders (DECISIONS.md §7).
        # ------------------------------------------------------------------
        if self._investment_strategy.rebalancing_needed(self._asset_model, scenario):
            rebal_orders = self._investment_strategy.compute_rebalancing_trades(
                self._asset_model, scenario
            )
            for order in rebal_orders:
                asset = self._asset_model.get_asset(order.asset_id)
                trade = asset.rebalance(order.target_value, scenario)
                self._cash_balance -= trade   # buy → trade > 0 → cash decreases

        # ------------------------------------------------------------------
        # Step 6 — Advance book values (MUST follow all rebalancing)
        # asset_model.step_time(scenario) calls bond.step_time(scenario) on
        # every remaining asset. Each bond uses scenario.rate_curve to reprice
        # itself (FVTPL/FVOCI: new_book_value = new_market_value) and
        # incorporates _realised_gl_this_period (set by rebalance()) into P&L.
        # ------------------------------------------------------------------
        self._asset_model.step_time(scenario)

        # ------------------------------------------------------------------
        # Step 7 — Collect period P&L and closing valuations
        # aggregate_pnl() reads _last_pnl from each asset (set by step_time).
        # market_value_by_basis() calls bond.market_value(scenario) on each
        # asset, again using scenario.rate_curve for pricing.
        # ------------------------------------------------------------------
        pnl         = self._asset_model.aggregate_pnl()
        mv_by_basis = self._asset_model.market_value_by_basis(scenario)

        asset_result = AssetTimestepResult(
            total_market_value=self._asset_model.total_market_value(scenario),
            total_book_value=self._asset_model.total_book_value(),
            cash_balance=self._cash_balance,
            eir_income=pnl.get("eir_income", 0.0),
            coupon_income=pnl.get("coupon_received", 0.0),
            dividend_income=pnl.get("dividend_income", 0.0),
            unrealised_gl=pnl.get("unrealised_gl", 0.0),
            realised_gl=pnl.get("realised_gl", 0.0),
            oci_reserve=pnl.get("oci_reserve", 0.0),
            mv_ac=mv_by_basis.get("AC", 0.0),
            mv_fvtpl=mv_by_basis.get("FVTPL", 0.0),
            mv_fvoci=mv_by_basis.get("FVOCI", 0.0),
        )

        return FundTimestepResult(
            cashflows=cashflows,
            decrements=decrements,
            asset=asset_result,
        )
