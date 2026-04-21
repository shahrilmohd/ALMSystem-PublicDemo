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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig, RunType
from engine.core.fund import Fund, FundTimestepResult
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.conventional import Conventional, ConventionalAssumptions
from engine.liability.liability_state import conventional_state_from_mps
from engine.matching_adjustment.ma_calculator import MACalculator, MAResult, build_ma_discount_curve
from engine.results.result_store import ResultStore, TimestepResult
from engine.run_modes.base_run import BaseRun, ProgressCallback
from engine.strategy.bonus_strategy import BonusStrategy
from engine.strategy.investment_strategy import InvestmentStrategy


# ---------------------------------------------------------------------------
# MA calibration result
# ---------------------------------------------------------------------------

@dataclass
class MACalibrationResult:
    """
    Output of DeterministicRun._calibrate_ma().

    Carries both BEL values, both discount curves, and the full MAResult
    for audit trail and attribution reporting (DECISIONS.md §21).

    Attributes
    ----------
    ma_result : MAResult
        Full output from MACalculator.compute() — includes ma_benefit_bps,
        eligible_asset_ids, cashflow_test_passes, per_asset_contributions,
        FS governance metadata.
    pre_ma_curve : RiskFreeRateCurve
        The input EIOPA RFR curve (plain risk-free, no MA shift).
    post_ma_curve : RiskFreeRateCurve
        pre_ma_curve parallel-shifted up by ma_benefit_bps (DECISIONS.md §21).
    pre_ma_bel : float
        BEL discounted at pre_ma_curve (plain RFR).
    post_ma_bel : float
        BEL discounted at post_ma_curve (RFR + MA benefit).
        This is the regulatory Solvency II BEL for a firm with MA approval.
    cashflow_recheck_passes : bool
        True if condition 4 (cumulative cashflow matching) still passes
        when liability CFs are scaled to reflect the post-MA BEL reduction.
    """
    ma_result:               MAResult
    pre_ma_curve:            RiskFreeRateCurve
    post_ma_curve:           RiskFreeRateCurve
    pre_ma_bel:              float
    post_ma_bel:             float
    cashflow_recheck_passes: bool


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
        # Optional MA calibration — all three must be set together.
        # When None (default), _calibrate_ma() is never called and execute()
        # is byte-for-byte identical to the pre-Step-20 implementation.
        ma_calculator:       Optional[MACalculator]    = None,
        assets_df:           Optional[pd.DataFrame]    = None,
        asset_cfs:           Optional[pd.DataFrame]    = None,
        bonus_strategy:      Optional[BonusStrategy]   = None,
    ) -> None:
        super().__init__(config, fund_config, progress_callback)
        self._model_points        = model_points
        self._assumptions         = assumptions
        self._asset_model         = asset_model
        self._investment_strategy = investment_strategy
        self._equity_return_yr    = equity_return_yr
        self._initial_cash        = initial_cash
        self._ma_calculator       = ma_calculator
        self._assets_df           = assets_df
        self._asset_cfs           = asset_cfs
        self._bonus_strategy      = bonus_strategy
        self._ma_calibration: Optional[MACalibrationResult] = None
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
        assert self._fund is not None
        assert self._store is not None
        total_months = self._config.projection.projection_term_years * 12
        mp = self._model_points.copy()

        # ------------------------------------------------------------------
        # MA pre-pass — only when ma_calculator is provided (BPA/MA runs).
        # Produces post_ma_curve used for both asset pricing and BEL discount.
        # Non-MA runs skip this entirely — zero behaviour change.
        # ------------------------------------------------------------------
        if self._ma_calculator is not None:
            self._ma_calibration = self._calibrate_ma(
                model_points  = self._model_points.copy(),
                assets_df     = self._assets_df,
                asset_cfs     = self._asset_cfs,
                ma_calculator = self._ma_calculator,
                total_months  = total_months,
            )
            active_curve = self._ma_calibration.post_ma_curve
            self.report_progress(0.05, "MA calibration complete")
        else:
            active_curve = self._assumptions.rate_curve

        # Pre-compute discount factors once (re-used in Pass 2).
        # For MA runs these use the post-MA curve; for non-MA runs the plain RFR.
        discount_factors = [
            active_curve.discount_factor(s + 1)
            for s in range(total_months)
        ]

        # Pre-MA discount factors — only needed when MA is active for attribution.
        if self._ma_calibration is not None:
            pre_ma_discount_factors = [
                self._ma_calibration.pre_ma_curve.discount_factor(s + 1)
                for s in range(total_months)
            ]
        else:
            pre_ma_discount_factors = None

        # ------------------------------------------------------------------
        # Pass 1 — Forward: advance Fund month by month
        # ------------------------------------------------------------------
        forward: list[FundTimestepResult] = []

        # Bonus strategy path: initialise vectorised liability state (n=1).
        # Smoothed return EMA is maintained here (not inside BonusStrategy)
        # because it is time-series state, not group state. See DECISIONS.md §64.
        if self._bonus_strategy is not None:
            bs_states = conventional_state_from_mps(mp, n_scenarios=1)
            smoothed_returns = np.zeros(1)

        for t in range(total_months):
            if self._bonus_strategy is not None:
                earned_ret_yr      = np.array([self._equity_return_yr])
                earned_ret_monthly = (1.0 + earned_ret_yr) ** (1.0 / 12.0) - 1.0
                smoothed_returns   = self._bonus_strategy.update_smoothed_returns(
                    smoothed_returns, earned_ret_yr
                )
                guaranteed_benefits = (
                    mp["sum_assured"].to_numpy(dtype=float)
                    + np.asarray(bs_states.accrued_bonus)
                )
                bonus_rates          = self._bonus_strategy.declare_reversionary(smoothed_returns)
                terminal_bonus_rates = self._bonus_strategy.compute_terminal_bonus_rate(
                    np.asarray(bs_states.asset_share), guaranteed_benefits
                )
                bs_states, step_cashflows, step_decrements = self._liability_model.batch_step(
                    bs_states, mp, bonus_rates, self._assumptions, t,
                    terminal_bonus_rates=terminal_bonus_rates,
                    earned_returns_monthly=earned_ret_monthly,
                )
                scenario_point = AssetScenarioPoint(
                    timestep=t,
                    rate_curve=active_curve,
                    equity_total_return_yr=self._equity_return_yr,
                )
                result = self._fund.step_time_with_liability(
                    scenario_point, step_cashflows[0], step_decrements[0]
                )
                mp = self._advance_model_points_no_bonus(mp)
            else:
                scenario = AssetScenarioPoint(
                    timestep=t,
                    rate_curve=active_curve,
                    equity_total_return_yr=self._equity_return_yr,
                )
                result = self._fund.step_time(scenario, mp, self._assumptions)
                mp = self._advance_model_points(mp)

            forward.append(result)
            self.report_progress(
                0.05 + 0.45 * (t + 1) / total_months,
                f"Month {t + 1}/{total_months} (forward pass)",
            )

        # ------------------------------------------------------------------
        # Pass 2 — BEL: backward summation of discounted future net_outgos
        # BEL(t) = Σ_{s=0}^{T-1-t} net_outgo[t+s] × DF(s+1)
        # For MA runs: also compute pre_ma_bel for attribution output.
        # ------------------------------------------------------------------
        net_outgos = [r.cashflows.net_outgo for r in forward]
        bels: list[float] = [
            sum(
                net_outgos[t + s] * discount_factors[s]
                for s in range(total_months - t)
            )
            for t in range(total_months)
        ]

        if pre_ma_discount_factors is not None:
            pre_ma_bels: Optional[list[float]] = [
                sum(
                    net_outgos[t + s] * pre_ma_discount_factors[s]
                    for s in range(total_months - t)
                )
                for t in range(total_months)
            ]
        else:
            pre_ma_bels = None

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
                # MA attribution — None for non-MA runs
                bel_pre_ma  = pre_ma_bels[t] if pre_ma_bels is not None else None,
                bel_post_ma = bel            if pre_ma_bels is not None else None,
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
        assert self._store is not None
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

    @property
    def ma_calibration(self) -> Optional[MACalibrationResult]:
        """
        MA calibration result from the pre-pass, or None if MA is not active.
        Populated after execute() completes (when ma_calculator was provided).
        """
        return self._ma_calibration

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

    def _advance_model_points_no_bonus(self, mp: pd.DataFrame) -> pd.DataFrame:
        """Advance MP by one month: duration, age, in_force — but NOT PAR bonus accrual.

        Used when batch_step owns accrued_bonus updates to prevent double-counting.
        """
        mp = mp.copy()

        mp["policy_duration_mths"] = mp["policy_duration_mths"] + 1

        remaining_term = mp["policy_term_yr"] * 12 - mp["policy_duration_mths"]
        mp.loc[remaining_term <= 0, "in_force_count"] = 0.0

        anniversary_mask = (mp["policy_duration_mths"] % 12) == 0
        mp.loc[anniversary_mask, "attained_age"] = (
            mp.loc[anniversary_mask, "attained_age"] + 1
        )

        return mp

    def _calibrate_ma(
        self,
        model_points:   pd.DataFrame,
        assets_df:      pd.DataFrame,
        asset_cfs:      pd.DataFrame,
        ma_calculator:  MACalculator,
        total_months:   int,
    ) -> MACalibrationResult:
        """
        MA pre-pass calibration sequence (DECISIONS.md §21, Steps 0a–0g).

        Called once before Pass 1 when ma_calculator is not None.
        Produces the post-MA discount curve that is locked for the entire
        subsequent projection.

        Parameters
        ----------
        model_points : pd.DataFrame
            Liability model points at the valuation date (copy — not mutated).
        assets_df : pd.DataFrame
            Asset metadata DataFrame for MACalculator eligibility assessment.
        asset_cfs : pd.DataFrame
            Per-asset cashflows net of default allowance.
            Columns: t (int, year), asset_id (str), cf (float).
        ma_calculator : MACalculator
            Pre-built MACalculator with the loaded FS table.
        total_months : int
            Projection horizon in months.

        Returns
        -------
        MACalibrationResult
        """
        rfr_curve = self._assumptions.rate_curve

        # ------------------------------------------------------------------
        # Step 0a — Project liability CFs at plain RFR to get CF schedule.
        # Run a liability-only forward pass (no Fund, no assets).
        # Collect monthly net_outgo; annualise for the matching test.
        # ------------------------------------------------------------------
        monthly_net_outgos: list[float] = []
        mp = model_points.copy()
        for t in range(total_months):
            cf = self._liability_model.project_cashflows(mp, self._assumptions, t)
            monthly_net_outgos.append(cf.net_outgo)
            mp = self._advance_model_points(mp)

        # Backward BEL summation at pre-MA rates
        pre_ma_dfs = [rfr_curve.discount_factor(s + 1) for s in range(total_months)]
        pre_ma_bel = sum(
            monthly_net_outgos[s] * pre_ma_dfs[s]
            for s in range(total_months)
        )

        # Annualise: sum monthly CFs into annual year-index buckets (year 1, 2, …)
        annual_years = (total_months + 11) // 12
        liab_cfs_annual_rows = []
        for yr in range(1, annual_years + 1):
            start = (yr - 1) * 12
            end   = min(yr * 12, total_months)
            total_cf = sum(monthly_net_outgos[start:end])
            liab_cfs_annual_rows.append({"t": yr, "cf": total_cf})
        liab_cfs_annual = pd.DataFrame(liab_cfs_annual_rows)

        # ------------------------------------------------------------------
        # Steps 0c–0d — Run MACalculator to get MA benefit.
        # asset_cfs is already in annual format (t = year, per the contract).
        # ------------------------------------------------------------------
        ma_result = ma_calculator.compute(
            assets_df          = assets_df,
            asset_cfs          = asset_cfs,
            liability_cfs      = liab_cfs_annual,
            liability_currency = "GBP",
            net_of_default     = True,
        )

        self._logger.info(
            "MA calibration: benefit=%.2f bps, eligible assets=%d, "
            "CF test passes=%s",
            ma_result.ma_benefit_bps,
            len(ma_result.eligible_asset_ids),
            ma_result.cashflow_test_passes,
        )

        if not ma_result.cashflow_test_passes:
            self._logger.warning(
                "MA cashflow matching test FAILED (condition 4). "
                "Failing periods: %s. "
                "Projection continues but the actuary must review the "
                "asset portfolio before relying on these results.",
                ma_result.failing_periods,
            )

        # ------------------------------------------------------------------
        # Step 0e — Build post-MA discount curve (parallel shift).
        # ------------------------------------------------------------------
        post_ma_curve = build_ma_discount_curve(rfr_curve, ma_result.ma_benefit_bps)

        # ------------------------------------------------------------------
        # Step 0f — Recompute BEL at post-MA rate; recheck CF matching.
        # Scale liability CFs by (post_ma_bel / pre_ma_bel) for the recheck.
        # ------------------------------------------------------------------
        post_ma_dfs = [post_ma_curve.discount_factor(s + 1) for s in range(total_months)]
        post_ma_bel = sum(
            monthly_net_outgos[s] * post_ma_dfs[s]
            for s in range(total_months)
        )

        if pre_ma_bel > 0.0:
            scale = post_ma_bel / pre_ma_bel
            scaled_liab_cfs = liab_cfs_annual.copy()
            scaled_liab_cfs["cf"] = scaled_liab_cfs["cf"] * scale
        else:
            scaled_liab_cfs = liab_cfs_annual.copy()

        # Re-run condition 4 on post-MA liability CFs
        # Build eligible asset CFs subset for the recheck
        eligible_ids = set(ma_result.eligible_asset_ids)
        eligible_asset_cfs = asset_cfs.loc[asset_cfs["asset_id"].isin(eligible_ids)]
        recheck_passes, _ = ma_calculator._eligibility.check_cashflow_matching(
            eligible_asset_cfs, scaled_liab_cfs, net_of_default=True
        )

        if not recheck_passes:
            self._logger.warning(
                "MA cashflow matching RECHECK (Step 0f) failed after "
                "post-MA BEL recomputation. Projection continues.",
            )

        # ------------------------------------------------------------------
        # Step 0g — Return locked calibration result.
        # ------------------------------------------------------------------
        return MACalibrationResult(
            ma_result               = ma_result,
            pre_ma_curve            = rfr_curve,
            post_ma_curve           = post_ma_curve,
            pre_ma_bel              = pre_ma_bel,
            post_ma_bel             = post_ma_bel,
            cashflow_recheck_passes = recheck_passes,
        )
