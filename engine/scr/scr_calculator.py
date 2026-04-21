"""
engine/scr/scr_calculator.py
=============================
SCRCalculator — orchestrates all three stress engines and assembles SCRResult.

Design (DECISIONS.md §51)
--------------------------
SCRCalculator is injected into BPARun at construction. After the base
projection completes (Phases 0–2), BPARun calls SCRCalculator.compute()
with the base-run outputs. The three stress engines run in sequence and
their results are assembled into one immutable SCRResult.

No BSCR aggregation is performed here. Full correlation-matrix aggregation
across all SII sub-modules is deferred to Step 26 (Phase 4).
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.core.projection_calendar import ProjectionCalendar
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.bpa.assumptions import BPAAssumptions
from engine.matching_adjustment.ma_calculator import MAResult
from engine.scr.interest_stress import InterestStressEngine
from engine.scr.longevity_stress import LongevityStressEngine
from engine.scr.scr_result import SCRResult
from engine.scr.spread_stress import SpreadStressEngine


class SCRCalculator:
    """
    Orchestrate all three SCR stress engines and return SCRResult.

    Parameters
    ----------
    spread_engine : SpreadStressEngine
        Credit spread stress engine (MA optional injection).
    interest_engine : InterestStressEngine
        Interest rate stress engine.
    longevity_engine : LongevityStressEngine
        Longevity stress engine (20% mortality improvement).
    """

    def __init__(
        self,
        spread_engine:    SpreadStressEngine,
        interest_engine:  InterestStressEngine,
        longevity_engine: LongevityStressEngine,
    ) -> None:
        self._spread    = spread_engine
        self._interest  = interest_engine
        self._longevity = longevity_engine

    def compute(
        self,
        asset_model:          AssetModel,
        assets_df:            pd.DataFrame,
        asset_cfs:            pd.DataFrame,
        liability_cashflows:  dict[int, float],
        rfr_curve:            RiskFreeRateCurve,
        ma_result:            MAResult,
        base_bel_post_ma:     float,
        base_bel_series:      list[float],
        scenario:             AssetScenarioPoint,
        calendar:             ProjectionCalendar,
        base_assumptions:     BPAAssumptions,
        per_cohort_mps:       dict[str, pd.DataFrame],
        per_cohort_models:    dict[str, Any],
        liability_currency:   str = "GBP",
    ) -> SCRResult:
        """
        Run all three stress engines and assemble the result.

        Parameters
        ----------
        asset_model : AssetModel
            Live asset portfolio at the valuation date.
        assets_df : pd.DataFrame
            One row per asset with credit spread (spread_bps) and eligibility
            columns. Used by SpreadStressEngine for MA recomputation.
        asset_cfs : pd.DataFrame
            Asset cashflows net of default allowance.
            Columns: t (int, year), asset_id (str), cf (float).
        liability_cashflows : dict[int, float]
            {period_index: net_outgo} — total liability CF schedule from
            BPARun._total_liability_cashflows. Unchanged under all stresses.
        rfr_curve : RiskFreeRateCurve
            Base EIOPA risk-free curve (pre-MA). Not mutated by any engine.
        ma_result : MAResult
            MA computation from the base run. ma_benefit_bps is the base
            portfolio-level MA benefit passed to the stress engines.
        base_bel_post_ma : float
            Total post-MA BEL at t=0 from the base run. Reference for ΔBEL.
        base_bel_series : list[float]
            Per-period post-MA BEL from the base run (length = n_periods).
            Used by LongevityStressEngine for the proportional scaling.
        scenario : AssetScenarioPoint
            Valuation date scenario. Provides RF curve for bond repricing.
        calendar : ProjectionCalendar
            Projection calendar for BEL discounting.
        base_assumptions : BPAAssumptions
            Base run assumptions. discount_curve is the post-MA adjusted curve.
            Used by LongevityStressEngine to build stressed assumptions.
        per_cohort_mps : dict[str, pd.DataFrame]
            {cohort_id: mps_at_t0} — opening model points per cohort.
            Used by LongevityStressEngine to call get_bel().
        per_cohort_models : dict[str, Any]
            {cohort_id: liability_model} — liability model per cohort.
        liability_currency : str
            ISO currency code. Passed to SpreadStressEngine. Default "GBP".

        Returns
        -------
        SCRResult
            Immutable record of all three stress results.
        """
        base_asset_mv = asset_model.total_market_value(scenario)

        # --- Spread stress ---
        sp_up, sp_down = self._spread.compute(
            asset_model=asset_model,
            assets_df=assets_df,
            asset_cfs=asset_cfs,
            liability_cashflows=liability_cashflows,
            base_bel_post_ma=base_bel_post_ma,
            rfr_curve=rfr_curve,
            base_ma_benefit_bps=ma_result.ma_benefit_bps,
            scenario=scenario,
            calendar=calendar,
            liability_currency=liability_currency,
        )

        # --- Interest rate stress ---
        ir_up, ir_down = self._interest.compute(
            asset_model=asset_model,
            liability_cashflows=liability_cashflows,
            base_bel_post_ma=base_bel_post_ma,
            rfr_curve=rfr_curve,
            ma_benefit_bps=ma_result.ma_benefit_bps,
            scenario=scenario,
            calendar=calendar,
        )

        # --- Longevity stress ---
        long_result = self._longevity.compute(
            per_cohort_mps=per_cohort_mps,
            per_cohort_models=per_cohort_models,
            base_bel_series=base_bel_series,
            base_assumptions=base_assumptions,
        )

        # --- Assemble SCRResult ---
        return SCRResult(
            # Spread
            scr_spread_up_own_funds_change=sp_up.own_funds_change,
            scr_spread_down_own_funds_change=sp_down.own_funds_change,
            scr_spread=max(-sp_up.own_funds_change, 0.0),
            spread_up_asset_mv_change=sp_up.asset_mv_change,
            spread_up_bel_change=sp_up.bel_change,
            spread_down_asset_mv_change=sp_down.asset_mv_change,
            spread_down_bel_change=sp_down.bel_change,
            # Interest
            scr_interest_up_own_funds_change=ir_up.own_funds_change,
            scr_interest_down_own_funds_change=ir_down.own_funds_change,
            scr_interest=max(-ir_up.own_funds_change, -ir_down.own_funds_change, 0.0),
            rate_up_asset_mv_change=ir_up.asset_mv_change,
            rate_up_bel_change=ir_up.bel_change,
            rate_down_asset_mv_change=ir_down.asset_mv_change,
            rate_down_bel_change=ir_down.bel_change,
            # Longevity
            scr_longevity=long_result.scr_longevity,
            longevity_stressed_bel_series=long_result.stressed_bel_series,
            # Governance
            base_asset_mv=base_asset_mv,
            base_bel_post_ma=base_bel_post_ma,
            spread_up_bps=self._spread.spread_up_bps,
            spread_down_bps=self._spread.spread_down_bps,
            rate_up_bps=self._interest.rate_up_bps,
            rate_down_bps=self._interest.rate_down_bps,
            longevity_mortality_stress_factor=self._longevity.mortality_stress_factor,
        )
