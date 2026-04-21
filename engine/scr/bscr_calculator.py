"""
engine/scr/bscr_calculator.py
==============================
BSCRCalculator — full SII standard formula SCR orchestrator.

Design (DECISIONS.md §63)
--------------------------
Constructed from a single ``SCRStressAssumptions`` instance. Builds all stress
engines, the BSCR aggregator, and the Risk Margin calculator internally from
that single assumption table — no other configuration is required.

Relationship to SCRCalculator
------------------------------
``SCRCalculator`` (Step 22) is kept unchanged for backward compatibility with
existing callers. ``BSCRCalculator`` extends the scope to all Step-26 modules
and produces a richer ``BSCRResult``. ``BPARun`` accepts both:
- ``scr_calculator`` → ``scr_result: SCRResult`` (existing)
- ``bscr_calculator`` (optional) → ``bscr_result: BSCRResult | None`` (new)

Engines constructed at __init__ time
--------------------------------------
SpreadStressEngine      — credit spread stress (Art 176–180)
InterestStressEngine    — interest rate stress (Art 166–169)
LongevityStressEngine   — longevity/mortality improvement (Art 137–138)
LapseStressEngine       — lapse permanent + mass shock (Art 142–144)
ExpenseStressEngine     — expense loading + inflation (Art 145–146)
CurrencyStressEngine    — FX shock (Art 188)
CounterpartyDefaultEngine — Type 1 default (Art 200–210)
BSCRAggregator          — quadratic form aggregation from assumption table
RiskMarginCalculator    — CoC run-off (Art 37–39)
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
from engine.scr.bscr_aggregator import BSCRAggregator
from engine.scr.bscr_result import BSCRResult
from engine.scr.counterparty_default import CounterpartyDefaultEngine, CounterpartyExposure
from engine.scr.currency_stress import CurrencyStressEngine
from engine.scr.expense_stress import ExpenseStressEngine
from engine.scr.interest_stress import InterestStressEngine
from engine.scr.lapse_stress import LapseStressEngine
from engine.scr.longevity_stress import LongevityStressEngine
from engine.scr.risk_margin import RiskMarginCalculator
from engine.scr.scr_assumptions import SCRStressAssumptions
from engine.scr.spread_stress import SpreadStressEngine


class BSCRCalculator:
    """
    Full SII standard formula SCR calculator built from a single
    ``SCRStressAssumptions`` instance.

    Parameters
    ----------
    assumptions : SCRStressAssumptions
        Central assumption table. All nine engines are configured from this
        single object — no other configuration arguments are needed.
    """

    def __init__(self, assumptions: SCRStressAssumptions) -> None:
        self._assumptions = assumptions

        self._spread_engine = SpreadStressEngine(
            spread_up_bps=assumptions.spread_up_bps,
            spread_down_bps=assumptions.spread_down_bps,
        )
        self._interest_engine = InterestStressEngine(
            rate_up_bps=assumptions.rate_up_bps,
            rate_down_bps=assumptions.rate_down_bps,
        )
        self._longevity_engine = LongevityStressEngine(
            mortality_stress_factor=assumptions.longevity_mortality_stress_factor,
        )
        self._lapse_engine        = LapseStressEngine(assumptions)
        self._expense_engine      = ExpenseStressEngine(assumptions)
        self._currency_engine     = CurrencyStressEngine(assumptions)
        self._counterparty_engine = CounterpartyDefaultEngine()
        self._aggregator          = BSCRAggregator(assumptions)
        self._risk_margin_calc    = RiskMarginCalculator(assumptions)

    def compute(
        self,
        # ---- Existing SCRCalculator parameters ----
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
        # ---- New Step-26 parameters ----
        # Lapse stress: pre-computed BEL sensitivity scalars
        bel_lapse_up:               float = 0.0,
        bel_lapse_down:             float = 0.0,
        base_lapse_bel:             float = 0.0,
        mass_lapse_bel_reduction:   float = 0.0,
        mass_lapse_asset_outflow:   float = 0.0,
        # Mortality SCR (conventional products; 0.0 for BPA)
        scr_mortality:              float = 0.0,
        # Optional Step-26 inputs
        expense_cashflows:                  dict[int, float] | None = None,
        net_foreign_currency_exposures:     dict[str, float] | None = None,
        counterparty_exposures:             list[CounterpartyExposure] | None = None,
    ) -> BSCRResult:
        """
        Run all SII stress engines and assemble a full ``BSCRResult``.

        Lapse scalars (bel_lapse_up, bel_lapse_down, base_lapse_bel,
        mass_lapse_bel_reduction, mass_lapse_asset_outflow) must be computed
        upstream by the run mode — this engine is product-agnostic.

        scr_mortality is 0.0 for BPA (longevity is the life biometric risk);
        set it non-zero for conventional products when MortalityStressEngine
        is added in a future step.
        """
        base_asset_mv = asset_model.total_market_value(scenario)

        sp_up, sp_down = self._spread_engine.compute(
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
        scr_spread = max(-sp_up.own_funds_change, 0.0)

        ir_up, ir_down = self._interest_engine.compute(
            asset_model=asset_model,
            liability_cashflows=liability_cashflows,
            base_bel_post_ma=base_bel_post_ma,
            rfr_curve=rfr_curve,
            ma_benefit_bps=ma_result.ma_benefit_bps,
            scenario=scenario,
            calendar=calendar,
        )
        scr_interest = max(-ir_up.own_funds_change, -ir_down.own_funds_change, 0.0)

        long_result = self._longevity_engine.compute(
            per_cohort_mps=per_cohort_mps,
            per_cohort_models=per_cohort_models,
            base_bel_series=base_bel_series,
            base_assumptions=base_assumptions,
        )
        scr_longevity = long_result.scr_longevity

        lapse_result = self._lapse_engine.compute(
            bel_lapse_up=bel_lapse_up,
            bel_lapse_down=bel_lapse_down,
            base_bel=base_lapse_bel,
            mass_lapse_bel_reduction=mass_lapse_bel_reduction,
            mass_lapse_asset_outflow=mass_lapse_asset_outflow,
        )

        expense_result = self._expense_engine.compute(
            expense_cashflows=expense_cashflows,
            rfr_curve=rfr_curve,
            calendar=calendar,
        )

        currency_result = self._currency_engine.compute(
            net_foreign_currency_exposures=net_foreign_currency_exposures,
        )

        cpty_result = self._counterparty_engine.compute(counterparty_exposures)

        scr_market, scr_life, bscr, scr_op = self._aggregator.aggregate(
            scr_interest=scr_interest,
            scr_spread=scr_spread,
            scr_currency=currency_result.scr_currency,
            scr_mortality=scr_mortality,
            scr_longevity=scr_longevity,
            scr_lapse=lapse_result.scr_lapse,
            scr_expense=expense_result.scr_expense,
            scr_counterparty=cpty_result.scr_counterparty,
            base_bel_post_ma=base_bel_post_ma,
        )
        scr_total = bscr + scr_op

        risk_margin = self._risk_margin_calc.compute(
            scr_t0=scr_total,
            bel_series=base_bel_series,
            rfr_curve=rfr_curve,
        )

        return BSCRResult(
            scr_spread=scr_spread,
            scr_interest=scr_interest,
            scr_longevity=scr_longevity,
            longevity_stressed_bel_series=long_result.stressed_bel_series,
            scr_mortality=scr_mortality,
            scr_lapse=lapse_result.scr_lapse,
            scr_expense=expense_result.scr_expense,
            scr_currency=currency_result.scr_currency,
            scr_counterparty=cpty_result.scr_counterparty,
            scr_market=scr_market,
            scr_life=scr_life,
            bscr=bscr,
            scr_operational=scr_op,
            scr_total=scr_total,
            risk_margin=risk_margin,
            base_asset_mv=base_asset_mv,
            base_bel_post_ma=base_bel_post_ma,
            assumptions=self._assumptions,
        )

    @property
    def assumptions(self) -> SCRStressAssumptions:
        return self._assumptions
