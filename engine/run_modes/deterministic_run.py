"""
DeterministicRun — full ALM projection over a single economic scenario.

Purpose
-------
DeterministicRun is the first end-to-end run mode that coordinates both
assets and liabilities. It runs a monthly loop over the full projection term,
calling Fund.step_time() at each timestep and writing results to ResultStore.

Used for:
    - Full P&L validation against independently-calculated values
    - Asset/liability cash flow matching analysis
    - Bonus crediting capacity analysis
    - Single-scenario sensitivity runs (e.g. rate +100bps)

Architecture notes
------------------
Data is INJECTED into the constructor. model_points, assumptions, asset_model,
and investment_strategy are all passed in as parameters — this class never
reads files or databases. The data loaders (Steps 6 and later) are the layer
that reads from files and then calls this constructor.

Time loop
---------
The monthly loop lives in execute(). DeterministicRun owns the loop directly
(not via engine/core/top.py) because top.py is not yet built. When top.py is
added (Phase 1 completion), DeterministicRun will delegate to it.

Two-pass BEL design (same as LiabilityOnlyRun — see DECISIONS.md §12)
-----------------------------------------------------------------------
Pass 1 — Forward (per month):
    Call Fund.step_time() to advance the full fund (assets + liabilities).
    Collect and store FundTimestepResult. Advance model points.

Pass 2 — Backward (vectorised float arithmetic):
    BEL(t) = Σ_{s=0}^{T-1-t} net_outgo[t+s] × DF(s+1)
    where DF(k) = assumptions.rate_curve.discount_factor(k).
    All operands are plain Python floats — no pandas overhead.
    For a 120-month projection: 7,260 float multiplications (microseconds).

Rate curve and equity return
----------------------------
The AssetScenarioPoint passed to Fund at each timestep is built from:
    rate_curve          — assumptions.rate_curve (same RiskFreeRateCurve used
                          for BEL discounting; one consistent risk-free curve
                          for the deterministic scenario).
    equity_total_return_yr — injected as equity_return_yr constructor parameter.

This design is forward-compatible with StochasticRun (Step 9), where each
scenario supplies its own path of rate curves and equity returns.

MP state advancement
--------------------
The original self._model_points is never mutated. execute() works on a copy.
After each Fund.step_time() call, model points are advanced by one calendar
month using the same logic as LiabilityOnlyRun._advance_model_points().
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig, RunType
from engine.core.fund import Fund, FundTimestepResult
from engine.liability.conventional import Conventional, ConventionalAssumptions
from engine.results.result_store import ResultStore, TimestepResult
from engine.run_modes.base_run import BaseRun, ProgressCallback
from engine.strategy.investment_strategy import InvestmentStrategy


class DeterministicRun(BaseRun):
    """
    Full ALM projection over a single deterministic economic scenario.

    Parameters
    ----------
    config : RunConfig
        Validated master run configuration. run_type must be DETERMINISTIC.
    fund_config : FundConfig
        Validated fund configuration.
    model_points : pd.DataFrame
        Pre-loaded, validated liability model point DataFrame.
    assumptions : ConventionalAssumptions
        Pre-built assumption object. assumptions.rate_curve is used for both
        BEL discounting and asset pricing (AssetScenarioPoint.rate_curve).
    asset_model : AssetModel
        Pre-built portfolio. Mutated in-place during projection. Contains all
        bonds and equities at their opening valuation-date state.
    investment_strategy : InvestmentStrategy
        Pre-built strategy (injected per CLAUDE.md Rule 5). Determines SAA
        rebalancing orders and forced sell logic each period.
    equity_return_yr : float
        Annual total return assumption for equities. Constant across all
        timesteps for the deterministic scenario. Default: 0.0.
    initial_cash : float
        Opening cash balance at the start of the projection (£). Default: 0.0.
    progress_callback : ProgressCallback, optional
        Called at key milestones with (fraction, message).
    """

    def __init__(
        self,
        config:              RunConfig,
        fund_config:         FundConfig,
        model_points:        pd.DataFrame,
        assumptions:         ConventionalAssumptions,
        asset_model:         AssetModel,
        investment_strategy: InvestmentStrategy,
        equity_return_yr:    float = 0.0,
        initial_cash:        float = 0.0,
        progress_callback:   Optional[ProgressCallback] = None,
    ) -> None:
        super().__init__(config, fund_config, progress_callback)
        self._model_points        = model_points
        self._assumptions         = assumptions
        self._asset_model         = asset_model
        self._investment_strategy = investment_strategy
        self._equity_return_yr    = equity_return_yr
        self._initial_cash        = initial_cash
        self._liability_model: Conventional           = Conventional()
        self._fund: Optional[Fund]                    = None
        self._store: Optional[ResultStore]            = None

    # -----------------------------------------------------------------------
    # validate_config
    # -----------------------------------------------------------------------

    def validate_config(self) -> None:
        """
        Verify run_type is DETERMINISTIC before any I/O starts.

        Raises
        ------
        ValueError
            If config.run_type is not RunType.DETERMINISTIC.
        """
        if self._config.run_type != RunType.DETERMINISTIC:
            raise ValueError(
                f"DeterministicRun requires run_type='deterministic', "
                f"got '{self._config.run_type.value}'."
            )

    # -----------------------------------------------------------------------
    # setup
    # -----------------------------------------------------------------------

    def setup(self) -> None:
        """
        Instantiate Fund and ResultStore.

        Fund wraps the injected asset_model, a fresh Conventional liability
        model, and the injected investment_strategy.
        """
        self._liability_model = Conventional()
        self._fund = Fund(
            asset_model=self._asset_model,
            liability_model=self._liability_model,
            investment_strategy=self._investment_strategy,
            initial_cash=self._initial_cash,
        )
        self._store = ResultStore(run_id=self._config.run_id)
        self._logger.info(
            "Setup complete: Fund ready, ResultStore run_id=%s",
            self._config.run_id,
        )

    # -----------------------------------------------------------------------
    # execute — two-pass monthly projection loop
    # -----------------------------------------------------------------------

    def execute(self) -> None:
        """
        Run the monthly projection loop using the two-pass BEL design.

        Pass 1 — Forward (Fund loop):
            At each month t, build AssetScenarioPoint using
            assumptions.rate_curve and equity_return_yr. Call Fund.step_time()
            which advances both asset and liability state and returns a
            FundTimestepResult. Advance model points to t+1.

        Pass 2 — BEL backward summation:
            BEL(t) = Σ_{s=0}^{T-1-t} net_outgo[t+s] × DF(s+1)
            where DF(k) = assumptions.rate_curve.discount_factor(k).
            Pure float arithmetic — no pandas overhead.

        Reserve = BEL (Phase 1 rule).
        """
        total_months = self._config.projection.projection_term_years * 12
        mp = self._model_points.copy()

        # Pre-compute discount factors once (re-used in Pass 2)
        discount_factors = [
            self._assumptions.rate_curve.discount_factor(s + 1)
            for s in range(total_months)
        ]

        # ------------------------------------------------------------------
        # Pass 1 — Forward: advance Fund month by month
        # ------------------------------------------------------------------
        forward: list[FundTimestepResult] = []

        for t in range(total_months):
            scenario = AssetScenarioPoint(
                timestep=t,
                rate_curve=self._assumptions.rate_curve,
                equity_total_return_yr=self._equity_return_yr,
            )
            result = self._fund.step_time(scenario, mp, self._assumptions)
            forward.append(result)
            mp = self._advance_model_points(mp)
            self.report_progress(
                0.05 + 0.45 * (t + 1) / total_months,
                f"Month {t + 1}/{total_months} (forward pass)",
            )

        # ------------------------------------------------------------------
        # Pass 2 — BEL: backward summation of discounted future net_outgos
        # BEL(t) = Σ_{s=0}^{T-1-t} net_outgo[t+s] × DF(s+1)
        # ------------------------------------------------------------------
        net_outgos = [r.cashflows.net_outgo for r in forward]
        bels: list[float] = [
            sum(
                net_outgos[t + s] * discount_factors[s]
                for s in range(total_months - t)
            )
            for t in range(total_months)
        ]

        # ------------------------------------------------------------------
        # Store results
        # ------------------------------------------------------------------
        for t, (fund_result, bel) in enumerate(zip(forward, bels)):
            ar = fund_result.asset
            self._store.store(TimestepResult(
                run_id=self._config.run_id,
                scenario_id=0,
                timestep=t,
                cashflows=fund_result.cashflows,
                decrements=fund_result.decrements,
                bel=bel,
                reserve=bel,
                # Asset fields
                total_market_value=ar.total_market_value,
                total_book_value=ar.total_book_value,
                cash_balance=ar.cash_balance,
                eir_income=ar.eir_income,
                coupon_income=ar.coupon_income,
                dividend_income=ar.dividend_income,
                unrealised_gl=ar.unrealised_gl,
                realised_gl=ar.realised_gl,
                oci_reserve=ar.oci_reserve,
                mv_ac=ar.mv_ac,
                mv_fvtpl=ar.mv_fvtpl,
                mv_fvoci=ar.mv_fvoci,
            ))
            self.report_progress(
                0.50 + 0.45 * (t + 1) / total_months,
                f"Month {t + 1}/{total_months} (BEL + store)",
            )

        self._logger.info(
            "Projection complete: %d months, %d results stored",
            total_months,
            self._store.result_count(),
        )

    # -----------------------------------------------------------------------
    # teardown — write outputs to disk
    # -----------------------------------------------------------------------

    def teardown(self) -> None:
        """
        Write ResultStore output to disk with output filters applied.

        Filters (same as LiabilityOnlyRun):
          1. output_horizon_years — clip to first N years if set.
          2. output_timestep — MONTHLY: all rows; ANNUAL: year-end snapshots
             (t=11,23,...); QUARTERLY: quarter-end snapshots (t=2,5,8,11,...).

        Output file:
            {output_dir}/{run_id}_deterministic_results.csv   (default)
            {output_dir}/{run_id}_deterministic_results.parquet
        """
        output_dir = Path(self._config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = self._store.as_dataframe()

        # Horizon filter
        if self._config.output.output_horizon_years is not None:
            max_t = self._config.output.output_horizon_years * 12 - 1
            df = df[df["timestep"] <= max_t].reset_index(drop=True)

        # Timestep filter
        ts = self._config.output.output_timestep.value
        if ts == "annual":
            df = df[((df["timestep"] + 1) % 12) == 0].reset_index(drop=True)
        elif ts == "quarterly":
            df = df[((df["timestep"] + 1) % 3) == 0].reset_index(drop=True)

        stem = self._output_stem("results")
        fmt  = self._config.output.result_format.value
        if fmt == "parquet":
            path = output_dir / f"{stem}.parquet"
            df.to_parquet(path, index=False)
        else:
            path = output_dir / f"{stem}.csv"
            df.to_csv(path, index=False)

        self._logger.info("Results written to %s (%d rows)", path, len(df))

    # -----------------------------------------------------------------------
    # Public read-only accessors
    # -----------------------------------------------------------------------

    @property
    def store(self) -> Optional[ResultStore]:
        """The ResultStore populated by execute(). None before setup()."""
        return self._store

    @property
    def fund(self) -> Optional[Fund]:
        """The Fund instance created by setup(). None before setup()."""
        return self._fund

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _advance_model_points(self, mp: pd.DataFrame) -> pd.DataFrame:
        """
        Advance all model point rows by one calendar month.

        Delegates to Conventional._advance_model_points, which applies the
        full decrement cascade (deaths → lapses → maturities) via
        _apply_decrements before advancing duration, age, and bonus accrual.

        Returns a new DataFrame.  The input mp is not mutated.
        """
        return self._liability_model._advance_model_points(mp, self._assumptions)
