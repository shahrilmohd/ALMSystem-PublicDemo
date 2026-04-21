"""
engine/run_modes/bpa_run.py
============================
BPARun — run orchestrator for BPA MA portfolios.

Design (DECISIONS.md §21, §27, §46)
-------------------------------------
BPARun is a concrete BaseRun subclass that projects BPA annuity liabilities
against a matched asset portfolio with a Matching Adjustment benefit applied
to the discount curve.

Key differences from DeterministicRun
---------------------------------------
1. Liability model: four BPA liability classes (InPayment, Deferred,
   Dependant, Enhanced) wrapped in _BPACompositeLiability instead of
   the single Conventional class.
2. Timestep loop: hybrid calendar (monthly for first N years, annual
   thereafter) via ProjectionCalendar rather than a fixed monthly loop.
3. Investment strategy: BuyAndHoldStrategy — no routine rebalancing
   (DECISIONS.md §46).
4. MA pre-pass: _calibrate_ma_bpa() runs before the main projection loop
   to produce the post-MA discount curve (DECISIONS.md §21, Steps 0a–0g).
5. Per-cohort results: one TimestepResult per (cohort_id, period_index)
   stored in ResultStore using the cohort_id index (DECISIONS.md §17).
6. Two BEL series: bel_pre_ma (plain RFR) and bel_post_ma (RFR + MA
   benefit) stored for every period to support IFRS 17 and attribution.

_BPACompositeLiability
-----------------------
A private adapter class (not exported) that makes the four BPA liability
classes appear as a single BaseLiability to Fund. The composite:
  - Stores per-class per-deal model point DataFrames internally.
  - On each project_cashflows() / get_decrements() call it iterates over
    all classes and all deals, summing the results.
  - Tracks per-cohort net_outgos during the forward pass so BPARun can
    compute per-cohort BEL in the backward pass.
  - Has an advance(timestep, assumptions) method (called by BPARun after
    each Fund.step_time()) that reduces model point weights using the
    period decrement rates for each class.

This adapter is Step 20 temporary — replaced by per-cohort GmmEngine
wiring in Step 21.  It is intentionally minimal: its only purpose is to
satisfy the Fund interface without restructuring Fund itself.
"""
from __future__ import annotations

import logging
import dataclasses
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from engine.asset.asset_model import AssetModel
from engine.asset.base_asset import AssetScenarioPoint
from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig, RunType
from engine.core.fund import Fund
from engine.core.projection_calendar import ProjectionCalendar
from engine.ifrs17.gmm import GmmEngine
from engine.ifrs17.risk_adjustment import CostOfCapitalRA
from engine.ifrs17.state import Ifrs17State
from engine.liability.bpa.coverage_units import BPACoverageUnitProvider
from engine.liability.base_liability import BaseLiability, Decrements, LiabilityCashflows
from engine.liability.bpa.assumptions import BPAAssumptions
from engine.liability.bpa.deferred import DeferredLiability
from engine.liability.bpa.dependant import DependantLiability
from engine.liability.bpa.enhanced import EnhancedLiability
from engine.liability.bpa.in_payment import InPaymentLiability
from engine.liability.bpa.mortality import q_x
from engine.liability.bpa.registry import BPADealRegistry, make_cohort_id
from engine.matching_adjustment.fundamental_spread import FundamentalSpreadTable
from engine.matching_adjustment.ma_calculator import (
    MACalculator,
    MAResult,
    build_ma_discount_curve,
)
from engine.results.result_store import ResultStore, TimestepResult
from engine.run_modes.base_run import BaseRun, ProgressCallback
from engine.run_modes.deterministic_run import MACalibrationResult
from engine.scr.bscr_calculator import BSCRCalculator
from engine.scr.bscr_result import BSCRResult
from engine.scr.scr_calculator import SCRCalculator
from engine.scr.scr_result import SCRResult
from engine.strategy.buy_and_hold_strategy import BuyAndHoldStrategy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Population type constants — map to make_cohort_id() suffix
# ---------------------------------------------------------------------------

_POP_PENSIONER  = "pensioner"
_POP_DEFERRED   = "deferred"
_POP_DEPENDANT  = "dependant"
_POP_ENHANCED   = "enhanced"

# Column name that holds the live count / weight for each population type.
# in_payment / deferred / enhanced use "in_force_count" (integer head count).
# dependant uses "weight" (fractional probability = member_weight × dep_proportion).
_LIVES_COL: dict[str, str] = {
    _POP_PENSIONER: "in_force_count",
    _POP_DEFERRED:  "in_force_count",
    _POP_DEPENDANT: "weight",
    _POP_ENHANCED:  "in_force_count",
}


# ---------------------------------------------------------------------------
# Weight advancement helpers (module-private)
# ---------------------------------------------------------------------------

def _advance_in_payment(
    model:       InPaymentLiability,
    mps:         pd.DataFrame,
    assumptions: BPAAssumptions,
    timestep:    int,
) -> pd.DataFrame:
    """
    Advance in-payment (or enhanced) model points by one period.

    Applies per-model-point mortality: new_weight = old_weight × (1 − q_death).
    Uses InPaymentLiability.get_decrement_rates_with_assumptions() which
    returns per-mp q_death as a DataFrame.
    """
    if mps.empty or mps["in_force_count"].sum() <= 0.0:
        return mps.copy()
    rates   = model.get_decrement_rates_with_assumptions(timestep, mps, assumptions)
    new_mps = mps.copy()
    q_map   = rates.set_index("mp_id")["q_death"]
    new_mps["in_force_count"] = new_mps.apply(
        lambda row: row["in_force_count"] * max(0.0, 1.0 - q_map.get(row["mp_id"], 0.0)),
        axis=1,
    )
    return new_mps


def _advance_deferred(
    model:       DeferredLiability,
    mps:         pd.DataFrame,
    assumptions: BPAAssumptions,
    timestep:    int,
) -> pd.DataFrame:
    """
    Advance deferred model points by one period.

    Applies all four decrements (death, ill-health, retirement, TV).
    Survival rate = 1 − q_death − q_ill − q_retire − q_tv per model point.
    Uses the public get_decrement_rates_with_assumptions() interface.
    """
    if mps.empty or mps["in_force_count"].sum() <= 0.0:
        return mps.copy()
    rates   = model.get_decrement_rates_with_assumptions(timestep, mps, assumptions)
    new_mps = mps.copy()
    q_map   = rates.set_index("mp_id")
    new_mps["in_force_count"] = new_mps.apply(
        lambda row: row["in_force_count"] * max(
            0.0,
            1.0
            - q_map.loc[row["mp_id"], "q_death"]
            - q_map.loc[row["mp_id"], "q_ill"]
            - q_map.loc[row["mp_id"], "q_retire"]
            - q_map.loc[row["mp_id"], "q_tv"],
        ),
        axis=1,
    )
    return new_mps


def _advance_dependant(
    model:       DependantLiability,
    mps:         pd.DataFrame,
    assumptions: BPAAssumptions,
    timestep:    int,
) -> pd.DataFrame:
    """
    Advance dependant model points by one period.

    Dependant weight = member_weight × dependant_proportion (set by the data
    loader).  Reduces by member period mortality so that future trigger
    probabilities are computed on the surviving member pool.
    Uses the public get_decrement_rates_with_assumptions() interface.
    """
    if mps.empty or mps["weight"].sum() <= 0.0:
        return mps.copy()
    rates   = model.get_decrement_rates_with_assumptions(timestep, mps, assumptions)
    new_mps = mps.copy()
    q_map   = rates.set_index("mp_id")["q_death"]
    new_mps["weight"] = new_mps.apply(
        lambda row: row["weight"] * max(0.0, 1.0 - q_map.get(row["mp_id"], 0.0)),
        axis=1,
    )
    return new_mps


# ---------------------------------------------------------------------------
# _BPACompositeLiability
# ---------------------------------------------------------------------------

class _BPACompositeLiability(BaseLiability):
    """
    Aggregates cashflows and decrements from the four BPA liability classes
    so that Fund receives a single BaseLiability-compliant object.

    All four sub-models are called in sequence; their LiabilityCashflows and
    Decrements are summed field-by-field.  Model point state (weights) is
    maintained internally and advanced by BPARun after each period via
    advance().

    Model point state
    -----------------
    Stored as nested dicts: {population_type: {deal_id: DataFrame}}.
    The outer dict is keyed by _POP_* constants; the inner dict by deal_id.

    Per-cohort tracking
    -------------------
    project_cashflows() appends each cohort's net_outgo to
    _cohort_net_outgos[cohort_id] so BPARun can compute per-cohort BEL
    in the backward pass without re-running the projection.

    Ignored argument
    ----------------
    Fund.step_time() passes model_points to every BaseLiability method.
    _BPACompositeLiability ignores this argument — it uses its internally
    managed per-class mps instead.  BPARun passes pd.DataFrame() as a
    dummy to avoid confusion.
    """

    def __init__(
        self,
        in_payment_model: InPaymentLiability,
        deferred_model:   DeferredLiability,
        dependant_model:  DependantLiability,
        enhanced_model:   EnhancedLiability,
        in_payment_mps:   dict[str, pd.DataFrame],
        deferred_mps:     dict[str, pd.DataFrame],
        dependant_mps:    dict[str, pd.DataFrame],
        enhanced_mps:     dict[str, pd.DataFrame],
    ) -> None:
        self._models: dict[str, Any] = {
            _POP_PENSIONER: in_payment_model,
            _POP_DEFERRED:  deferred_model,
            _POP_DEPENDANT: dependant_model,
            _POP_ENHANCED:  enhanced_model,
        }
        # Live (mutable) model point store — advanced each period
        self._mps: dict[str, dict[str, pd.DataFrame]] = {
            _POP_PENSIONER: {k: v.copy() for k, v in in_payment_mps.items()},
            _POP_DEFERRED:  {k: v.copy() for k, v in deferred_mps.items()},
            _POP_DEPENDANT: {k: v.copy() for k, v in dependant_mps.items()},
            _POP_ENHANCED:  {k: v.copy() for k, v in enhanced_mps.items()},
        }
        # Per-cohort net_outgo accumulator (populated during forward pass)
        self._cohort_net_outgos: dict[str, list[float]] = {}
        self._calendar: Optional[ProjectionCalendar] = None

    # ------------------------------------------------------------------
    # Internal: pass calendar reference after it is built
    # ------------------------------------------------------------------

    def _set_calendar(self, calendar: ProjectionCalendar) -> None:
        """Called by BPARun.setup() immediately after construction."""
        self._calendar = calendar

    # ------------------------------------------------------------------
    # BaseLiability — project_cashflows
    # ------------------------------------------------------------------

    def project_cashflows(
        self,
        model_points: pd.DataFrame,   # ignored — internal state used
        assumptions:  Any,
        timestep:     int,
    ) -> LiabilityCashflows:
        total_premiums   = 0.0
        total_death      = 0.0
        total_surrender  = 0.0
        total_maturity   = 0.0
        total_expenses   = 0.0

        for pop_type, model in self._models.items():
            for deal_id, mps in self._mps[pop_type].items():
                if mps.empty or mps[_LIVES_COL[pop_type]].sum() <= 0.0:
                    # Still record zero for this cohort so the cohort_id
                    # appears in the store even in zero-weight periods.
                    cohort_id = make_cohort_id(deal_id, pop_type)
                    if cohort_id not in self._cohort_net_outgos:
                        self._cohort_net_outgos[cohort_id] = []
                    self._cohort_net_outgos[cohort_id].append(0.0)
                    continue

                cf = model.project_cashflows(mps, assumptions, timestep)
                total_premiums  += cf.premiums
                total_death     += cf.death_claims
                total_surrender += cf.surrender_payments
                total_maturity  += cf.maturity_payments
                total_expenses  += cf.expenses

                # Track per-cohort for backward BEL pass
                cohort_id = make_cohort_id(deal_id, pop_type)
                if cohort_id not in self._cohort_net_outgos:
                    self._cohort_net_outgos[cohort_id] = []
                self._cohort_net_outgos[cohort_id].append(cf.net_outgo)

        return LiabilityCashflows(
            timestep          = timestep,
            premiums          = total_premiums,
            death_claims      = total_death,
            surrender_payments= total_surrender,
            maturity_payments = total_maturity,
            expenses          = total_expenses,
        )

    # ------------------------------------------------------------------
    # BaseLiability — get_decrements
    # ------------------------------------------------------------------

    def get_decrements(
        self,
        model_points: pd.DataFrame,   # ignored
        assumptions:  Any,
        timestep:     int,
    ) -> Decrements:
        total_in_force = 0.0
        total_deaths   = 0.0
        total_lapses   = 0.0
        total_mats     = 0.0

        for pop_type, model in self._models.items():
            for deal_id, mps in self._mps[pop_type].items():
                if mps.empty or mps[_LIVES_COL[pop_type]].sum() <= 0.0:
                    continue
                dec = model.get_decrements(mps, assumptions, timestep)
                total_in_force += dec.in_force_start
                total_deaths   += dec.deaths
                total_lapses   += dec.lapses
                total_mats     += dec.maturities

        return Decrements(
            timestep       = timestep,
            in_force_start = total_in_force,
            deaths         = total_deaths,
            lapses         = total_lapses,
            maturities     = total_mats,
            in_force_end   = total_in_force - total_deaths - total_lapses - total_mats,
        )

    # ------------------------------------------------------------------
    # BaseLiability — get_bel (full projection from timestep)
    # ------------------------------------------------------------------

    def get_bel(
        self,
        model_points: pd.DataFrame,   # ignored
        assumptions:  Any,
        timestep:     int,
    ) -> float:
        total_bel = 0.0
        for pop_type, model in self._models.items():
            for deal_id, mps in self._mps[pop_type].items():
                if mps.empty or mps[_LIVES_COL[pop_type]].sum() <= 0.0:
                    continue
                total_bel += model.get_bel(mps, assumptions, timestep)
        return total_bel

    def get_reserve(
        self,
        model_points: pd.DataFrame,
        assumptions:  Any,
        timestep:     int,
    ) -> float:
        return self.get_bel(model_points, assumptions, timestep)

    # ------------------------------------------------------------------
    # Weight advancement — called by BPARun after each step_time()
    # ------------------------------------------------------------------

    def advance(self, timestep: int, assumptions: BPAAssumptions) -> None:
        """
        Advance all model point weights by one period using the correct
        decrement logic for each liability class.

        Decrements are independent across liability class boundaries:
          - in-payment / enhanced: death only
          - deferred: death + ill-health + retirement + TV
          - dependant: member mortality reduces trigger pool

        The composite updates its internal mps in-place (new DataFrames
        are assigned; the previous DataFrames are discarded).
        """
        assert self._calendar is not None, "_set_calendar() must be called before advance()"
        new_ip: dict[str, pd.DataFrame] = {}
        new_df: dict[str, pd.DataFrame] = {}
        new_dp: dict[str, pd.DataFrame] = {}
        new_en: dict[str, pd.DataFrame] = {}

        ip_model = self._models[_POP_PENSIONER]
        df_model = self._models[_POP_DEFERRED]
        dp_model = self._models[_POP_DEPENDANT]
        en_model = self._models[_POP_ENHANCED]

        for deal_id, mps in self._mps[_POP_PENSIONER].items():
            new_ip[deal_id] = _advance_in_payment(ip_model, mps, assumptions, timestep)
        for deal_id, mps in self._mps[_POP_DEFERRED].items():
            new_df[deal_id] = _advance_deferred(df_model, mps, assumptions, timestep)
        for deal_id, mps in self._mps[_POP_DEPENDANT].items():
            new_dp[deal_id] = _advance_dependant(dp_model, mps, assumptions, timestep)
        for deal_id, mps in self._mps[_POP_ENHANCED].items():
            new_en[deal_id] = _advance_in_payment(en_model, mps, assumptions, timestep)

        self._mps[_POP_PENSIONER] = new_ip
        self._mps[_POP_DEFERRED]  = new_df
        self._mps[_POP_DEPENDANT] = new_dp
        self._mps[_POP_ENHANCED]  = new_en

    # ------------------------------------------------------------------
    # Cohort introspection
    # ------------------------------------------------------------------

    def all_cohort_ids(self) -> list[str]:
        """Return all cohort_ids managed by this composite (sorted)."""
        ids: list[str] = []
        for pop_type in (_POP_PENSIONER, _POP_DEFERRED, _POP_DEPENDANT, _POP_ENHANCED):
            for deal_id in self._mps[pop_type]:
                ids.append(make_cohort_id(deal_id, pop_type))
        return sorted(ids)


# ---------------------------------------------------------------------------
# BPARun
# ---------------------------------------------------------------------------

class BPARun(BaseRun):
    """
    BPA MA portfolio run orchestrator.

    Wires the four BPA liability classes, a BuyAndHoldStrategy asset
    portfolio, and the MA pre-pass into a hybrid-timestep projection loop.

    Parameters
    ----------
    config : RunConfig
        Validated master run config.  run_type must be RunType.BPA.
    fund_config : FundConfig
        Validated fund configuration.
    in_payment_mps : dict[str, pd.DataFrame]
        In-payment pensioner model points, keyed by deal_id.
    deferred_mps : dict[str, pd.DataFrame]
        Deferred member model points, keyed by deal_id.
    dependant_mps : dict[str, pd.DataFrame]
        Dependant model points, keyed by deal_id.
    enhanced_mps : dict[str, pd.DataFrame]
        Enhanced life model points, keyed by deal_id.
    assumptions : BPAAssumptions
        Fully constructed BPA assumption set (pre-MA; BPARun replaces
        discount_curve with the post-MA curve before the main loop).
    asset_model : AssetModel
        Pre-built BPA asset portfolio (AC bonds + any FVTPL liquidity buffer).
    assets_df : pd.DataFrame
        Asset metadata for MACalculator eligibility assessment.
    asset_cfs : pd.DataFrame
        Per-asset cashflows net of default allowance (annual granularity;
        columns: t int, asset_id str, cf float).
    fs_table : FundamentalSpreadTable
        Loaded PRA fundamental spread table.
    deal_registry : BPADealRegistry
        Registry of all active BPA deals.
    ifrs17_state_store : Ifrs17StateStore, optional
        IFRS 17 state persistence (load + save per cohort). When None the
        IFRS 17 GMM engine is not run (useful in unit tests and dry runs).
    scr_calculator : SCRCalculator, optional
        SCR stress engine (spread, interest, longevity). When None the SCR
        pass is skipped and the IFRS 17 RA falls back to zero SCR_longevity
        series. Inject a fully wired SCRCalculator for production runs
        (DECISIONS.md §49, §51).
    ma_highly_predictable_cap : float
        Highly-predictable asset weight cap for MA eligibility. Default 0.35.
    projection_years : int
        Total projection horizon in years. Default 60.
    monthly_years : int
        Number of years projected at monthly granularity. Default 10.
    initial_cash : float
        Opening fund cash balance. Default 0.0.
    progress_callback : ProgressCallback, optional
        Called at key milestones with (fraction, message).
    """

    def __init__(
        self,
        config:                     RunConfig,
        fund_config:                FundConfig,
        in_payment_mps:             dict[str, pd.DataFrame],
        deferred_mps:               dict[str, pd.DataFrame],
        dependant_mps:              dict[str, pd.DataFrame],
        enhanced_mps:               dict[str, pd.DataFrame],
        assumptions:                BPAAssumptions,
        asset_model:                AssetModel,
        assets_df:                  pd.DataFrame,
        asset_cfs:                  pd.DataFrame,
        fs_table:                   FundamentalSpreadTable,
        deal_registry:              BPADealRegistry,
        ifrs17_state_store:         Optional[Any] = None,
        scr_calculator:             Optional[SCRCalculator] = None,
        bscr_calculator:            Optional[BSCRCalculator] = None,
        ma_highly_predictable_cap:  float = 0.35,
        projection_years:           int   = 60,
        monthly_years:              int   = 10,
        initial_cash:               float = 0.0,
        progress_callback:          Optional[ProgressCallback] = None,
    ) -> None:
        super().__init__(config, fund_config, progress_callback)
        self._in_payment_mps_init  = in_payment_mps
        self._deferred_mps_init    = deferred_mps
        self._dependant_mps_init   = dependant_mps
        self._enhanced_mps_init    = enhanced_mps
        self._assumptions          = assumptions
        self._asset_model          = asset_model
        self._assets_df            = assets_df
        self._asset_cfs            = asset_cfs
        self._fs_table             = fs_table
        self._deal_registry        = deal_registry
        self._ifrs17_state_store   = ifrs17_state_store
        self._scr_calculator       = scr_calculator
        self._bscr_calculator      = bscr_calculator
        self._ma_hp_cap            = ma_highly_predictable_cap
        self._projection_years     = projection_years
        self._monthly_years        = monthly_years
        self._initial_cash         = initial_cash

        # Populated by setup()
        self._calendar:     Optional[ProjectionCalendar]    = None
        self._composite:    Optional[_BPACompositeLiability] = None
        self._fund:         Optional[Fund]                  = None
        self._store:        Optional[ResultStore]           = None
        self._ma_calculator: Optional[MACalculator]         = None

        # Populated by execute()
        self._ma_calibration: Optional[MACalibrationResult] = None
        self._scr_result:     Optional[SCRResult]           = None
        self._bscr_result:    Optional[BSCRResult]          = None

    # -----------------------------------------------------------------------
    # validate_config
    # -----------------------------------------------------------------------

    def validate_config(self) -> None:
        """Raise ValueError if config.run_type is not RunType.BPA."""
        if self._config.run_type != RunType.BPA:
            raise ValueError(
                f"BPARun requires run_type='bpa', got '{self._config.run_type.value}'."
            )

    # -----------------------------------------------------------------------
    # setup
    # -----------------------------------------------------------------------

    def setup(self) -> None:
        """
        Instantiate all model components.

        1. ProjectionCalendar — hybrid timestep schedule.
        2. BuyAndHoldStrategy — no routine rebalancing.
        3. MACalculator — MA eligibility + benefit engine.
        4. BPA liability models — one per class.
        5. _BPACompositeLiability — wraps all four for Fund.
        6. Fund — asset model + composite + buy-and-hold strategy.
        7. ResultStore — result collector.
        """
        self._calendar = ProjectionCalendar(
            projection_years=self._projection_years,
            monthly_years=self._monthly_years,
        )

        strategy = BuyAndHoldStrategy()

        self._ma_calculator = MACalculator(
            fs_table=self._fs_table,
            ma_highly_predictable_cap=self._ma_hp_cap,
        )

        in_payment_model = InPaymentLiability(self._calendar)
        deferred_model   = DeferredLiability(self._calendar)
        dependant_model  = DependantLiability(self._calendar)
        enhanced_model   = EnhancedLiability(self._calendar)

        self._composite = _BPACompositeLiability(
            in_payment_model = in_payment_model,
            deferred_model   = deferred_model,
            dependant_model  = dependant_model,
            enhanced_model   = enhanced_model,
            in_payment_mps   = self._in_payment_mps_init,
            deferred_mps     = self._deferred_mps_init,
            dependant_mps    = self._dependant_mps_init,
            enhanced_mps     = self._enhanced_mps_init,
        )
        self._composite._set_calendar(self._calendar)

        self._fund = Fund(
            asset_model         = self._asset_model,
            liability_model     = self._composite,
            investment_strategy = strategy,
            initial_cash        = self._initial_cash,
        )
        self._store = ResultStore(run_id=self._config.run_id)

        self._logger.info(
            "BPARun setup: calendar=%d periods (%d monthly + %d annual), run_id=%s",
            self._calendar.n_periods,
            self._monthly_years * 12,
            self._projection_years - self._monthly_years,
            self._config.run_id,
        )

    # -----------------------------------------------------------------------
    # execute — MA pre-pass + hybrid loop + backward BEL
    # -----------------------------------------------------------------------

    def execute(self) -> None:
        """
        Run the BPA projection.

        Phase 0 — MA calibration (pre-pass):
            Run _calibrate_ma_bpa() to lock the post-MA discount curve.
            Construct post-MA BPAAssumptions for the main projection.

        Phase 1 — Forward (hybrid timestep loop):
            At each period, build AssetScenarioPoint with dt=period.year_fraction
            and call Fund.step_time().  Advance composite model point weights.

        Phase 2 — Backward BEL:
            For each cohort: compute pre-MA and post-MA BEL series using
            per-cohort net_outgos tracked by the composite.

        Phase 3 — SCR stress (optional, requires scr_calculator injected):
            Run SCRCalculator.compute() with base-run outputs.  Stores
            SCRResult and derives per-cohort stressed BEL series for the
            IFRS 17 RA.  Skipped when scr_calculator is None.

        Store results:
            One TimestepResult per (cohort_id, period_index) per period.

        Phase 4 — IFRS 17 GMM (optional, requires ifrs17_state_store):
            Run GmmEngine and CostOfCapitalRA for all cohorts.
        """
        assert self._calendar is not None, "call setup() before execute()"
        assert self._composite is not None
        assert self._fund is not None
        assert self._store is not None
        assert self._ma_calculator is not None

        # ------------------------------------------------------------------
        # MA pre-pass
        # ------------------------------------------------------------------
        self._ma_calibration = self._calibrate_ma_bpa()
        pre_ma_curve  = self._ma_calibration.pre_ma_curve
        post_ma_curve = self._ma_calibration.post_ma_curve
        self.report_progress(0.05, "MA calibration complete")

        # Build post-MA assumptions: same as input but with post-MA discount curve.
        # BPA assumptions object is frozen; construct a new one with replaced curve.
        import dataclasses
        post_ma_assumptions = dataclasses.replace(
            self._assumptions, discount_curve=post_ma_curve
        )

        # Pre-compute per-period spot discount factors at END of each period
        # for both curves — used in the backward BEL pass.
        pre_ma_dfs  = [
            pre_ma_curve.discount_factor(
                (p.time_in_years + p.year_fraction) * 12.0
            )
            for p in self._calendar.periods
        ]
        post_ma_dfs = [
            post_ma_curve.discount_factor(
                (p.time_in_years + p.year_fraction) * 12.0
            )
            for p in self._calendar.periods
        ]
        # DF to the START of each period (used for forward BEL division).
        # period 0 starts at t=0 → DF=1.0.
        pre_ma_dfs_start  = [
            pre_ma_curve.discount_factor(p.time_in_years * 12.0)
            for p in self._calendar.periods
        ]
        post_ma_dfs_start = [
            post_ma_curve.discount_factor(p.time_in_years * 12.0)
            for p in self._calendar.periods
        ]

        # ------------------------------------------------------------------
        # Pass 1 — Forward: hybrid timestep loop
        # ------------------------------------------------------------------
        from engine.core.fund import FundTimestepResult
        forward: list[FundTimestepResult] = []
        n_periods = self._calendar.n_periods
        dummy_mp  = pd.DataFrame()  # composite ignores this argument

        for period in self._calendar.periods:
            t  = period.period_index
            dt = period.year_fraction

            scenario = AssetScenarioPoint(
                timestep               = t,
                rate_curve             = post_ma_curve,
                equity_total_return_yr = 0.0,
                dt                     = dt,
            )
            result = self._fund.step_time(scenario, dummy_mp, post_ma_assumptions)
            forward.append(result)
            self._composite.advance(t, post_ma_assumptions)

            self.report_progress(
                0.05 + 0.45 * (t + 1) / n_periods,
                f"Period {t + 1}/{n_periods} (forward pass)",
            )

        # ------------------------------------------------------------------
        # Pass 2 — Backward BEL per cohort
        # ------------------------------------------------------------------
        # For each cohort, compute:
        #   pre_ma_bel(t)  = Σ_{s≥t} outgo[s] × (DF_pre_end[s]  / DF_pre_start[t])
        #   post_ma_bel(t) = Σ_{s≥t} outgo[s] × (DF_post_end[s] / DF_post_start[t])
        #
        # When DF_start[t] ≈ 0 (very long projection, negligible), use 1.0 as floor.

        cohort_ids = self._composite.all_cohort_ids()
        per_cohort_bel_pre:  dict[str, list[float]] = {}
        per_cohort_bel_post: dict[str, list[float]] = {}

        for cohort_id in cohort_ids:
            outgos = self._composite._cohort_net_outgos.get(cohort_id, [0.0] * n_periods)
            pre_bels:  list[float] = []
            post_bels: list[float] = []
            for t in range(n_periods):
                df_start_pre  = max(pre_ma_dfs_start[t],  1e-15)
                df_start_post = max(post_ma_dfs_start[t], 1e-15)
                pre_bel  = sum(
                    outgos[s] * pre_ma_dfs[s]  / df_start_pre
                    for s in range(t, n_periods)
                )
                post_bel = sum(
                    outgos[s] * post_ma_dfs[s] / df_start_post
                    for s in range(t, n_periods)
                )
                pre_bels.append(pre_bel)
                post_bels.append(post_bel)
            per_cohort_bel_pre[cohort_id]  = pre_bels
            per_cohort_bel_post[cohort_id] = post_bels

        # ------------------------------------------------------------------
        # Store results — one TimestepResult per (cohort_id × period)
        # ------------------------------------------------------------------
        for i, (period, fund_result) in enumerate(zip(self._calendar.periods, forward)):
            t  = period.period_index
            ar = fund_result.asset
            for cohort_id in cohort_ids:
                post_bel = per_cohort_bel_post[cohort_id][i]
                pre_bel  = per_cohort_bel_pre[cohort_id][i]
                self._store.store(TimestepResult(
                    run_id            = self._config.run_id,
                    scenario_id       = 0,
                    timestep          = t,
                    cohort_id         = cohort_id,
                    cashflows         = fund_result.cashflows,
                    decrements        = fund_result.decrements,
                    bel               = post_bel,
                    reserve           = post_bel,
                    # Asset fields (shared across cohorts — same fund)
                    total_market_value= ar.total_market_value,
                    total_book_value  = ar.total_book_value,
                    cash_balance      = ar.cash_balance,
                    eir_income        = ar.eir_income,
                    coupon_income     = ar.coupon_income,
                    dividend_income   = ar.dividend_income,
                    unrealised_gl     = ar.unrealised_gl,
                    realised_gl       = ar.realised_gl,
                    oci_reserve       = ar.oci_reserve,
                    mv_ac             = ar.mv_ac,
                    mv_fvtpl          = ar.mv_fvtpl,
                    mv_fvoci          = ar.mv_fvoci,
                    # MA attribution
                    bel_pre_ma        = pre_bel,
                    bel_post_ma       = post_bel,
                ))

            self.report_progress(
                0.50 + 0.45 * (t + 1) / n_periods,
                f"Period {t + 1}/{n_periods} (BEL + store)",
            )

        # ------------------------------------------------------------------
        # Phase 3 — SCR stress (optional)
        # ------------------------------------------------------------------
        # per_cohort_stressed_bel maps cohort_id → list[float] of the
        # longevity-stressed post-MA BEL, same length as n_periods.
        # When SCRCalculator is not injected the series is all-zeros and
        # CostOfCapitalRA will return zero RA (conservative fall-through).
        per_cohort_stressed_bel: dict[str, list[float]] = {
            cid: [0.0] * n_periods for cid in cohort_ids
        }

        if self._scr_calculator is not None:
            # Total liability CFs for spread / interest stress
            total_liability_cashflows: dict[int, float] = {
                t: sum(
                    self._composite._cohort_net_outgos.get(cid, [0.0] * n_periods)[t]
                    for cid in cohort_ids
                )
                for t in range(n_periods)
            }

            # Total base post-MA BEL series (sum across cohorts)
            total_bel_series_post: list[float] = [
                sum(per_cohort_bel_post[cid][t] for cid in cohort_ids)
                for t in range(n_periods)
            ]
            base_bel_post_ma_t0 = total_bel_series_post[0] if total_bel_series_post else 0.0

            # Valuation-date scenario: use pre-MA curve (asset pricing is RF
            # + calibration_spread; MA benefit is a liability-side offset only)
            opening_scenario = AssetScenarioPoint(
                timestep               = 0,
                rate_curve             = pre_ma_curve,
                equity_total_return_yr = 0.0,
                dt                     = self._calendar.periods[0].year_fraction,
            )

            # Opening model points and models per cohort (t=0, pre-advancement)
            per_cohort_mps_t0:   dict[str, pd.DataFrame] = {}
            per_cohort_models_t0: dict[str, Any] = {}
            for pop_type, model in self._composite._models.items():
                init_map = {
                    _POP_PENSIONER: self._in_payment_mps_init,
                    _POP_DEFERRED:  self._deferred_mps_init,
                    _POP_DEPENDANT: self._dependant_mps_init,
                    _POP_ENHANCED:  self._enhanced_mps_init,
                }[pop_type]
                for deal_id, mps in init_map.items():
                    cohort_id = make_cohort_id(deal_id, pop_type)
                    per_cohort_mps_t0[cohort_id]    = mps.copy()
                    per_cohort_models_t0[cohort_id] = model

            self._scr_result = self._scr_calculator.compute(
                asset_model          = self._asset_model,
                assets_df            = self._assets_df,
                asset_cfs            = self._asset_cfs,
                liability_cashflows  = total_liability_cashflows,
                rfr_curve            = pre_ma_curve,
                ma_result            = self._ma_calibration.ma_result,
                base_bel_post_ma     = base_bel_post_ma_t0,
                base_bel_series      = total_bel_series_post,
                scenario             = opening_scenario,
                calendar             = self._calendar,
                base_assumptions     = post_ma_assumptions,
                per_cohort_mps       = per_cohort_mps_t0,
                per_cohort_models    = per_cohort_models_t0,
            )
            self.report_progress(0.97, "SCR stress complete")

            # Distribute the total stressed BEL series to per-cohort proportionally.
            # Each cohort receives the same fraction of the stressed series as its
            # share of the base post-MA BEL at t=0.
            total_bel_t0 = base_bel_post_ma_t0
            for cohort_id in cohort_ids:
                cohort_bel_t0 = per_cohort_bel_post[cohort_id][0]
                fraction = cohort_bel_t0 / total_bel_t0 if total_bel_t0 > 0.0 else 0.0
                per_cohort_stressed_bel[cohort_id] = [
                    v * fraction
                    for v in self._scr_result.longevity_stressed_bel_series
                ]

        # ------------------------------------------------------------------
        # Phase 3b — Full BSCR (optional, Step 26: BSCRCalculator injected)
        # ------------------------------------------------------------------
        # per_cohort_mps_t0, per_cohort_models_t0, post_ma_assumptions, and
        # base_bel_post_ma_t0 are only defined inside the scr_calculator block above.
        if self._bscr_calculator is not None and self._scr_calculator is not None:
            # Lapse scalars: compute deferred BEL under stressed retirement/TV rates.
            shock = self._bscr_calculator.assumptions.lapse_permanent_shock_factor
            mass  = self._bscr_calculator.assumptions.lapse_mass_shock_factor
            tv_bel_ratio = self._bscr_calculator.assumptions.lapse_tv_to_bel_ratio

            bel_lapse_up   = 0.0
            bel_lapse_down = 0.0
            base_def_bel   = 0.0

            deferred_model = self._composite._models.get(_POP_DEFERRED)
            if deferred_model is not None:
                for deal_id, mps in self._deferred_mps_init.items():
                    if mps.empty or mps["in_force_count"].sum() <= 0.0:
                        continue
                    base_bel_d = deferred_model.get_bel(mps, post_ma_assumptions, 0)
                    base_def_bel += base_bel_d

                    ret_up = dataclasses.replace(
                        post_ma_assumptions.retirement,
                        early_retirement_rate=min(1.0, post_ma_assumptions.retirement.early_retirement_rate * (1.0 + shock)),
                        late_retirement_rate=min(1.0, post_ma_assumptions.retirement.late_retirement_rate  * (1.0 + shock)),
                    )
                    ret_down = dataclasses.replace(
                        post_ma_assumptions.retirement,
                        early_retirement_rate=max(0.0, post_ma_assumptions.retirement.early_retirement_rate * (1.0 - shock)),
                        late_retirement_rate=max(0.0, post_ma_assumptions.retirement.late_retirement_rate  * (1.0 - shock)),
                    )
                    assump_up   = dataclasses.replace(post_ma_assumptions, retirement=ret_up)
                    assump_down = dataclasses.replace(post_ma_assumptions, retirement=ret_down)

                    bel_lapse_up   += deferred_model.get_bel(mps, assump_up,   0)
                    bel_lapse_down += deferred_model.get_bel(mps, assump_down, 0)

            # Mass lapse: asset outflow ≈ TV/SV payments; BEL reduction = 0 for
            # in-payment annuitants (Art 142 — annuities cannot surrender mid-term).
            mass_lapse_asset_outflow = mass * tv_bel_ratio * base_def_bel

            self._bscr_result = self._bscr_calculator.compute(
                asset_model          = self._asset_model,
                assets_df            = self._assets_df,
                asset_cfs            = self._asset_cfs,
                liability_cashflows  = total_liability_cashflows,
                rfr_curve            = pre_ma_curve,
                ma_result            = self._ma_calibration.ma_result,
                base_bel_post_ma     = base_bel_post_ma_t0,
                base_bel_series      = total_bel_series_post,
                scenario             = opening_scenario,
                calendar             = self._calendar,
                base_assumptions     = post_ma_assumptions,
                per_cohort_mps       = per_cohort_mps_t0,
                per_cohort_models    = per_cohort_models_t0,
                bel_lapse_up         = bel_lapse_up,
                bel_lapse_down       = bel_lapse_down,
                base_lapse_bel       = base_def_bel,
                mass_lapse_bel_reduction=0.0,
                mass_lapse_asset_outflow=mass_lapse_asset_outflow,
                scr_mortality        = 0.0,  # BPA: no mortality SCR
            )
            self.report_progress(0.98, "Full BSCR complete")

            bscr_series = self._bscr_result.longevity_stressed_bel_series
            if bscr_series and base_bel_post_ma_t0 > 0.0:
                for cohort_id in cohort_ids:
                    cohort_bel_t0 = per_cohort_bel_post[cohort_id][0]
                    fraction = cohort_bel_t0 / base_bel_post_ma_t0
                    per_cohort_stressed_bel[cohort_id] = [
                        v * fraction for v in bscr_series
                    ]

        # ------------------------------------------------------------------
        # IFRS 17 — run GmmEngine and save state + movements (Step 21)
        # ------------------------------------------------------------------
        if self._ifrs17_state_store is not None:
            self._run_gmm_engine(
                cohort_ids              = cohort_ids,
                per_cohort_bel_post     = per_cohort_bel_post,
                per_cohort_stressed_bel = per_cohort_stressed_bel,
            )

        self._logger.info(
            "BPA projection complete: %d periods, %d cohorts, %d results stored",
            n_periods,
            len(cohort_ids),
            self._store.result_count(),
        )

    # -----------------------------------------------------------------------
    # teardown
    # -----------------------------------------------------------------------

    def teardown(self) -> None:
        """
        Write projection results and governance summaries to disk.

        Outputs
        -------
        {output_dir}/{run_id}_bpa_results.csv        — full per-cohort time-series
        {output_dir}/{run_id}_bpa_ma_summary.csv     — MA governance summary (1 row)
        {output_dir}/{run_id}_bpa_cohort_summary.csv — per-cohort day-one BEL summary
        {output_dir}/{run_id}_bpa_scr_result.csv     — SCR stress results (when available)
        """
        assert self._store is not None
        assert self._ma_calibration is not None
        assert self._calendar is not None

        output_dir = Path(self._config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ── Results time-series ────────────────────────────────────────────
        df = self._store.as_dataframe()
        if self._config.output.output_horizon_years is not None:
            max_t = self._config.output.output_horizon_years * self._calendar.n_periods // self._projection_years
            df = df[df["timestep"] <= max_t].reset_index(drop=True)

        ts_filter = self._config.output.output_timestep.value
        if ts_filter == "annual":
            n_monthly = self._monthly_years * 12
            df = df[df["timestep"] >= n_monthly].reset_index(drop=True)

        stem = self._output_stem("bpa_results")
        fmt  = self._config.output.result_format.value
        if fmt == "parquet":
            results_path = output_dir / f"{stem}.parquet"
            df.to_parquet(results_path, index=False)
        else:
            results_path = output_dir / f"{stem}.csv"
            df.to_csv(results_path, index=False)
        self._logger.info("BPA results written to %s (%d rows)", results_path, len(df))

        # ── MA governance summary ─────────────────────────────────────────
        ma_r = self._ma_calibration.ma_result
        ma_summary = pd.DataFrame([{
            "ma_benefit_bps":          ma_r.ma_benefit_bps,
            "eligible_asset_count":    len(ma_r.eligible_asset_ids),
            "cashflow_test_passes":    ma_r.cashflow_test_passes,
            "failing_periods":         str(ma_r.failing_periods),
            "fs_table_effective_date": str(ma_r.fs_table_effective_date),
            "fs_table_source_ref":     ma_r.fs_table_source_ref,
            "pre_ma_bel":              self._ma_calibration.pre_ma_bel,
            "post_ma_bel":             self._ma_calibration.post_ma_bel,
        }])
        summary_path = output_dir / f"{self._output_stem('bpa_ma_summary')}.csv"
        ma_summary.to_csv(summary_path, index=False)
        self._logger.info("MA summary written to %s", summary_path)

        # ── Per-cohort day-one BEL summary ────────────────────────────────
        self._write_cohort_summary(output_dir, df)

        # ── SCR result (when SCRCalculator was injected) ──────────────────
        if self._scr_result is not None:
            self._write_scr_summary(output_dir)

    # -----------------------------------------------------------------------
    # teardown helpers
    # -----------------------------------------------------------------------

    def _write_cohort_summary(self, output_dir: Path, results_df: pd.DataFrame) -> None:
        """
        Write {run_id}_bpa_cohort_summary.csv — one row per cohort_id.

        Columns represent the day-one (t=0) balance sheet position per
        contract group: opening BEL, MA reduction, in-force lives, and
        optionally the longevity-stressed BEL when SCR was computed.

        Source data
        -----------
        - bel_post_ma_t0 / bel_pre_ma_t0  : read back from stored results at timestep=0
        - in_force_lives_t0                : sum of weights from initial model point DataFrames
        - last_nonzero_period              : highest timestep with bel_post_ma > 0
        - scr_longevity_stressed_bel_t0    : period-0 entry of longevity stressed BEL series
                                             (from SCRResult, distributed proportionally to cohorts)
        """
        assert self._store is not None
        cohort_ids = self._store.cohort_ids()
        if not cohort_ids:
            return

        # Opening lives per cohort from initial model points
        initial_lives: dict[str, float] = {}
        for pop_type, init_map in [
            (_POP_PENSIONER, self._in_payment_mps_init),
            (_POP_DEFERRED,  self._deferred_mps_init),
            (_POP_DEPENDANT, self._dependant_mps_init),
            (_POP_ENHANCED,  self._enhanced_mps_init),
        ]:
            lives_col = _LIVES_COL[pop_type]
            for deal_id, mps in init_map.items():
                cid = make_cohort_id(deal_id, pop_type)
                initial_lives[cid] = float(mps[lives_col].sum()) if not mps.empty else 0.0

        # Period-0 longevity-stressed BEL per cohort (when SCR was computed)
        stressed_bel_t0: dict[str, Optional[float]] = {cid: None for cid in cohort_ids}
        if self._scr_result is not None:
            # Reconstruct per-cohort fraction using period-0 bel_post_ma from store
            total_bel_t0 = self._scr_result.base_bel_post_ma
            for cid in cohort_ids:
                try:
                    r = self._store.get(0, 0, cid)
                    cohort_bel_t0 = r.bel_post_ma or 0.0
                except KeyError:
                    cohort_bel_t0 = 0.0
                fraction = cohort_bel_t0 / total_bel_t0 if total_bel_t0 > 0.0 else 0.0
                stressed_bel_t0[cid] = (
                    self._scr_result.longevity_stressed_bel_series[0] * fraction
                    if self._scr_result.longevity_stressed_bel_series
                    else 0.0
                )

        rows = []
        for cid in cohort_ids:
            # t=0 BELs from stored results
            try:
                r0 = self._store.get(0, 0, cid)
                bel_post = r0.bel_post_ma or r0.bel or 0.0
                bel_pre  = r0.bel_pre_ma  or r0.bel or 0.0
            except KeyError:
                bel_post = bel_pre = 0.0

            # Last period with non-zero BEL
            cohort_rows = results_df[results_df["cohort_id"] == cid]
            nonzero = cohort_rows[cohort_rows["bel_post_ma"].fillna(0.0) > 0.0]
            last_nonzero = int(nonzero["timestep"].max()) if not nonzero.empty else 0

            ma_reduction  = bel_pre - bel_post
            ma_reduction_pct = (ma_reduction / bel_pre * 100.0) if bel_pre > 0.0 else 0.0

            # Parse deal_id and population_type from cohort_id
            parts = cid.rsplit("_", 1)
            pop_type = parts[-1] if len(parts) == 2 else ""
            deal_id  = parts[0]  if len(parts) == 2 else cid

            rows.append({
                "cohort_id":                    cid,
                "deal_id":                      deal_id,
                "population_type":              pop_type,
                "bel_post_ma_t0":               bel_post,
                "bel_pre_ma_t0":                bel_pre,
                "ma_reduction_t0":              ma_reduction,
                "ma_reduction_pct":             round(ma_reduction_pct, 4),
                "in_force_lives_t0":            initial_lives.get(cid, 0.0),
                "last_nonzero_period":          last_nonzero,
                "scr_longevity_stressed_bel_t0": stressed_bel_t0.get(cid),
            })

        cohort_path = output_dir / f"{self._output_stem('bpa_cohort_summary')}.csv"
        pd.DataFrame(rows).to_csv(cohort_path, index=False)
        self._logger.info(
            "BPA cohort summary written to %s (%d cohorts)", cohort_path, len(rows)
        )

    def _write_scr_summary(self, output_dir: Path) -> None:
        """
        Write {run_id}_bpa_scr_result.csv — one row per run.

        Contains all SCR sub-module results plus a partial BSCR calculated
        using the SII standard formula correlation matrix for the three
        modules built in Step 22 (spread, interest, longevity).

        SII standard formula correlation matrix (Article 164, Annex IV):
            Spread – Interest:  0.00  (independent risk drivers)
            Spread – Longevity: 0.25
            Interest – Longevity: 0.25

        bscr_partial = sqrt(
            SCR_sp²  + SCR_int²  + SCR_lon²
            + 2×0.25×SCR_sp×SCR_lon
            + 2×0.25×SCR_int×SCR_lon
        )

        This is explicitly labelled "partial" to distinguish it from a full
        BSCR; the remaining SII sub-modules (lapse, expense, operational,
        counterparty, currency) are added in Phase 4.
        """
        import math
        assert self._scr_result is not None
        assert self._ma_calibration is not None
        r = self._scr_result

        sp  = r.scr_spread
        ir  = r.scr_interest
        lon = r.scr_longevity
        bscr_partial = math.sqrt(
            sp**2 + ir**2 + lon**2
            + 2.0 * 0.25 * sp * lon
            + 2.0 * 0.25 * ir * lon
        )

        row = {
            "run_id":                          self._config.run_id,
            # Spread stress
            "scr_spread":                      sp,
            "spread_up_own_funds_change":       r.scr_spread_up_own_funds_change,
            "spread_down_own_funds_change":     r.scr_spread_down_own_funds_change,
            "spread_up_asset_mv_change":        r.spread_up_asset_mv_change,
            "spread_up_bel_change":             r.spread_up_bel_change,
            "spread_up_bps":                   r.spread_up_bps,
            "spread_down_bps":                 r.spread_down_bps,
            # Interest stress
            "scr_interest":                    ir,
            "rate_up_own_funds_change":         r.scr_interest_up_own_funds_change,
            "rate_down_own_funds_change":       r.scr_interest_down_own_funds_change,
            "rate_up_asset_mv_change":          r.rate_up_asset_mv_change,
            "rate_up_bel_change":               r.rate_up_bel_change,
            "rate_down_asset_mv_change":        r.rate_down_asset_mv_change,
            "rate_down_bel_change":             r.rate_down_bel_change,
            "rate_up_bps":                     r.rate_up_bps,
            "rate_down_bps":                   r.rate_down_bps,
            # Longevity stress
            "scr_longevity":                   lon,
            "longevity_stressed_bel_t0":       (
                r.longevity_stressed_bel_series[0]
                if r.longevity_stressed_bel_series else 0.0
            ),
            "longevity_mortality_stress_factor": r.longevity_mortality_stress_factor,
            # Base inputs
            "base_asset_mv":                   r.base_asset_mv,
            "base_bel_post_ma":                r.base_bel_post_ma,
            "ma_benefit_bps_base":             self._ma_calibration.ma_result.ma_benefit_bps,
            # Partial BSCR
            "bscr_partial":                    bscr_partial,
        }
        scr_path = output_dir / f"{self._output_stem('bpa_scr_result')}.csv"
        pd.DataFrame([row]).to_csv(scr_path, index=False)
        self._logger.info("BPA SCR result written to %s", scr_path)

    # -----------------------------------------------------------------------
    # Public accessors
    # -----------------------------------------------------------------------

    @property
    def store(self) -> Optional[ResultStore]:
        """ResultStore populated by execute(). None before setup()."""
        return self._store

    @property
    def fund(self) -> Optional[Fund]:
        """Fund instance created by setup(). None before setup()."""
        return self._fund

    @property
    def calendar(self) -> Optional[ProjectionCalendar]:
        """ProjectionCalendar created by setup(). None before setup()."""
        return self._calendar

    @property
    def ma_calibration(self) -> Optional[MACalibrationResult]:
        """MA calibration result from the pre-pass. None before execute()."""
        return self._ma_calibration

    @property
    def scr_result(self) -> Optional[SCRResult]:
        """SCR stress result. None when scr_calculator was not injected or before execute()."""
        return self._scr_result

    @property
    def bscr_result(self) -> Optional[BSCRResult]:
        """Full BSCR result. None when bscr_calculator was not injected or before execute()."""
        return self._bscr_result

    # -----------------------------------------------------------------------
    # Internal — MA pre-pass
    # -----------------------------------------------------------------------

    def _calibrate_ma_bpa(self) -> MACalibrationResult:
        """
        MA pre-pass calibration sequence for BPA liability classes
        (DECISIONS.md §21, Steps 0a–0g).

        Uses temporary copies of model points so the composite's live state
        is not consumed during the pre-pass.

        Returns
        -------
        MACalibrationResult
        """
        rfr_curve = self._assumptions.discount_curve

        # Build temporary model point copies for the pre-pass
        ip_mps = {k: v.copy() for k, v in self._in_payment_mps_init.items()}
        df_mps = {k: v.copy() for k, v in self._deferred_mps_init.items()}
        dp_mps = {k: v.copy() for k, v in self._dependant_mps_init.items()}
        en_mps = {k: v.copy() for k, v in self._enhanced_mps_init.items()}

        # Build one-off liability model instances for the pre-pass
        # (share the same calendar but are otherwise independent)
        assert self._calendar is not None
        ip_model = InPaymentLiability(self._calendar)
        df_model = DeferredLiability(self._calendar)
        dp_model = DependantLiability(self._calendar)
        en_model = EnhancedLiability(self._calendar)

        # Step 0a — Project liability CFs at plain RFR to get CF schedule
        period_net_outgos: list[float] = []

        for period in self._calendar.periods:
            t   = period.period_index
            total_outgo = 0.0
            for deal_id, mps in ip_mps.items():
                if not mps.empty and mps["in_force_count"].sum() > 0:
                    cf = ip_model.project_cashflows(mps, self._assumptions, t)
                    total_outgo += cf.net_outgo
            for deal_id, mps in df_mps.items():
                if not mps.empty and mps["in_force_count"].sum() > 0:
                    cf = df_model.project_cashflows(mps, self._assumptions, t)
                    total_outgo += cf.net_outgo
            for deal_id, mps in dp_mps.items():
                if not mps.empty and mps["weight"].sum() > 0:
                    cf = dp_model.project_cashflows(mps, self._assumptions, t)
                    total_outgo += cf.net_outgo
            for deal_id, mps in en_mps.items():
                if not mps.empty and mps["in_force_count"].sum() > 0:
                    cf = en_model.project_cashflows(mps, self._assumptions, t)
                    total_outgo += cf.net_outgo
            period_net_outgos.append(total_outgo)

            # Advance temporary model points
            ip_mps = {k: _advance_in_payment(ip_model, v, self._assumptions, t)
                      for k, v in ip_mps.items()}
            df_mps = {k: _advance_deferred(df_model, v, self._assumptions, t)
                      for k, v in df_mps.items()}
            dp_mps = {k: _advance_dependant(dp_model, v, self._assumptions, t)
                      for k, v in dp_mps.items()}
            en_mps = {k: _advance_in_payment(en_model, v, self._assumptions, t)
                      for k, v in en_mps.items()}

        # Pre-MA BEL via backward summation (absolute DFs from t=0)
        pre_ma_dfs = [
            rfr_curve.discount_factor((p.time_in_years + p.year_fraction) * 12.0)
            for p in self._calendar.periods
        ]
        pre_ma_bel = sum(
            period_net_outgos[s] * pre_ma_dfs[s]
            for s in range(len(period_net_outgos))
        )

        # Annualise to annual year buckets for condition 4 cashflow matching test
        # (MACalculator expects annual CFs with column t = integer year).
        total_years        = self._projection_years
        liab_cfs_rows: list[dict] = []
        for yr in range(1, total_years + 1):
            # Collect periods whose END falls in this year
            year_cf = 0.0
            for period in self._calendar.periods:
                t_end = period.time_in_years + period.year_fraction
                if (yr - 1) < t_end <= yr:
                    year_cf += period_net_outgos[period.period_index]
            liab_cfs_rows.append({"t": yr, "cf": year_cf})
        liab_cfs_annual = pd.DataFrame(liab_cfs_rows)

        # Steps 0c–0d — Run MACalculator
        assert self._ma_calculator is not None
        ma_result = self._ma_calculator.compute(
            assets_df          = self._assets_df,
            asset_cfs          = self._asset_cfs,
            liability_cfs      = liab_cfs_annual,
            liability_currency = "GBP",
            net_of_default     = True,
        )
        self._logger.info(
            "MA calibration: benefit=%.2f bps, eligible assets=%d, CF test=%s",
            ma_result.ma_benefit_bps,
            len(ma_result.eligible_asset_ids),
            ma_result.cashflow_test_passes,
        )
        if not ma_result.cashflow_test_passes:
            self._logger.warning(
                "MA cashflow matching test FAILED (condition 4). "
                "Failing periods: %s. Projection continues but the actuary "
                "must review the asset portfolio before relying on these results.",
                ma_result.failing_periods,
            )

        # Step 0e — Build post-MA discount curve
        post_ma_curve = build_ma_discount_curve(rfr_curve, ma_result.ma_benefit_bps)

        # Step 0f — Recompute BEL and recheck CF matching
        post_ma_dfs_pre = [
            post_ma_curve.discount_factor((p.time_in_years + p.year_fraction) * 12.0)
            for p in self._calendar.periods
        ]
        post_ma_bel = sum(
            period_net_outgos[s] * post_ma_dfs_pre[s]
            for s in range(len(period_net_outgos))
        )

        if pre_ma_bel > 0.0:
            scale = post_ma_bel / pre_ma_bel
            scaled_liab_cfs           = liab_cfs_annual.copy()
            scaled_liab_cfs["cf"]     = scaled_liab_cfs["cf"] * scale
        else:
            scaled_liab_cfs = liab_cfs_annual.copy()

        eligible_ids       = set(ma_result.eligible_asset_ids)
        eligible_asset_cfs = self._asset_cfs.loc[
            self._asset_cfs["asset_id"].isin(eligible_ids)
        ]
        recheck_passes, _ = self._ma_calculator._eligibility.check_cashflow_matching(
            eligible_asset_cfs, scaled_liab_cfs, net_of_default=True
        )
        if not recheck_passes:
            self._logger.warning(
                "MA cashflow matching RECHECK (Step 0f) failed after "
                "post-MA BEL recomputation. Projection continues.",
            )

        # Step 0g — Return locked calibration result
        return MACalibrationResult(
            ma_result               = ma_result,
            pre_ma_curve            = rfr_curve,
            post_ma_curve           = post_ma_curve,
            pre_ma_bel              = pre_ma_bel,
            post_ma_bel             = post_ma_bel,
            cashflow_recheck_passes = recheck_passes,
        )

    # -----------------------------------------------------------------------
    # Internal — IFRS 17 GMM engine (Step 21)
    # -----------------------------------------------------------------------

    def _run_gmm_engine(
        self,
        cohort_ids:              list[str],
        per_cohort_bel_post:     dict[str, list[float]],
        per_cohort_stressed_bel: dict[str, list[float]],
    ) -> None:
        """
        Run GmmEngine for all cohorts and persist state + movements.

        Called from execute() after the SCR pass, when the full per-cohort
        outgo, BEL, and stressed BEL series are available.

        Steps (DECISIONS.md §28, §34, §35, §49, §50, §51):
          B1. Derive locked-in rate from MA calibration.
          B2. Compute bel_locked per cohort (flat locked-in rate).
          B3. Build BPACoverageUnitProvider per cohort.
          B4. Load or create opening Ifrs17State per cohort.
          B5. Build GmmEngine.
          B6. Precompute RA series per cohort using SCR_longevity series
              (DECISIONS.md §49, §51).  SCR_longevity(t) = max(stressed_bel(t)
              − base_bel(t), 0) per cohort per period.
          B7. Step GmmEngine once per period per cohort.
          B8. Save closing Ifrs17State and movements.

        Parameters
        ----------
        cohort_ids : list[str]
        per_cohort_bel_post : dict[str, list[float]]
            Post-MA BEL per cohort per period (bel_current input to GmmEngine).
        per_cohort_stressed_bel : dict[str, list[float]]
            Longevity-stressed post-MA BEL per cohort per period.
            Derived from SCRResult.longevity_stressed_bel_series distributed
            proportionally across cohorts.  All-zeros when scr_calculator
            is None (Phase 3 skipped).
        """
        assert self._calendar is not None
        assert self._composite is not None
        assert self._ma_calibration is not None
        assert self._ifrs17_state_store is not None

        n_periods      = self._calendar.n_periods
        valuation_date = self._config.projection.valuation_date

        # ------------------------------------------------------------------
        # B1 — locked-in rate (uniform across cohorts for this run)
        # ------------------------------------------------------------------
        locked_in_rate = (
            self._ma_calibration.ma_result.ma_benefit_bps / 10_000.0
            + self._assumptions.discount_curve.spot_rates.get(10.0, 0.03)
        )

        # Pre-compute DF series for bel_locked: (1+r)^(-t) from t=0 to
        # end and start of each period respectively.
        locked_dfs_end = [
            (1.0 + locked_in_rate) ** (-(p.time_in_years + p.year_fraction))
            for p in self._calendar.periods
        ]
        locked_dfs_start = [
            (1.0 + locked_in_rate) ** (-p.time_in_years)
            for p in self._calendar.periods
        ]

        # ------------------------------------------------------------------
        # B2 — bel_locked per cohort: same backward sum as bel_post_ma but
        #       using the flat locked-in rate instead of the post-MA curve.
        # ------------------------------------------------------------------
        period_end_times = [
            p.time_in_years + p.year_fraction for p in self._calendar.periods
        ]

        per_cohort_bel_locked: dict[str, list[float]] = {}
        for cohort_id in cohort_ids:
            outgos = self._composite._cohort_net_outgos.get(
                cohort_id, [0.0] * n_periods
            )
            locked_bels: list[float] = []
            for t in range(n_periods):
                df_start = max(locked_dfs_start[t], 1e-15)
                bel_l = sum(
                    outgos[s] * locked_dfs_end[s] / df_start
                    for s in range(t, n_periods)
                )
                locked_bels.append(bel_l)
            per_cohort_bel_locked[cohort_id] = locked_bels

        # ------------------------------------------------------------------
        # B3 — BPACoverageUnitProvider per cohort
        # ------------------------------------------------------------------
        providers: dict[str, BPACoverageUnitProvider] = {}
        for cohort_id in cohort_ids:
            outgos = self._composite._cohort_net_outgos.get(
                cohort_id, [0.0] * n_periods
            )
            providers[cohort_id] = BPACoverageUnitProvider(
                period_outgos          = outgos,
                locked_in_rate         = locked_in_rate,
                period_end_times_years = period_end_times,
            )

        # ------------------------------------------------------------------
        # B4 — Load or create opening Ifrs17State per cohort
        # ------------------------------------------------------------------
        opening_states: dict[str, Ifrs17State] = {}
        inception_dates: dict[str, date] = {}

        for cohort_id in cohort_ids:
            # Try loading prior state (will be None on first run)
            state = self._ifrs17_state_store.load_state(cohort_id, valuation_date)

            # Resolve inception date from deal registry
            deal_id = cohort_id.rsplit("_", 1)[0]
            try:
                meta = self._deal_registry.get(deal_id)
                inception = meta.inception_date
            except KeyError:
                inception = valuation_date
            inception_dates[cohort_id] = inception

            if state is None:
                # First run — create inception state (DECISIONS.md §50: CSM=0)
                total_cu = max(providers[cohort_id].total_coverage_units, 1e-15)
                state = Ifrs17State(
                    cohort_id                = cohort_id,
                    valuation_date           = inception,
                    csm_balance              = 0.0,
                    loss_component           = 0.0,
                    remaining_coverage_units = total_cu,
                    total_coverage_units     = total_cu,
                    locked_in_rate           = locked_in_rate,
                    inception_date           = inception,
                )
            opening_states[cohort_id] = state

        # ------------------------------------------------------------------
        # B5 — Build GmmEngine
        # ------------------------------------------------------------------
        gmm_engine = GmmEngine(
            contract_groups         = cohort_ids,
            opening_states          = opening_states,
            coverage_unit_providers = providers,
        )

        # ------------------------------------------------------------------
        # B6 — Precompute RA per cohort per period (SCR_longevity series, §49)
        #
        # SCR_longevity(t) = max(stressed_bel(t) − base_bel(t), 0).
        # When per_cohort_stressed_bel is all-zeros (no scr_calculator
        # injected), future_scrs are all zero and RA = 0.
        # ------------------------------------------------------------------
        ra_calculator = CostOfCapitalRA()
        per_cohort_ra: dict[str, list[float]] = {}

        for cohort_id in cohort_ids:
            post_bels    = per_cohort_bel_post[cohort_id]
            stressed_bels = per_cohort_stressed_bel.get(cohort_id, [0.0] * n_periods)
            ra_series: list[float] = []

            for t in range(n_periods):
                df_start_t = max(locked_dfs_start[t], 1e-15)
                # SCR_longevity at each future period: max(stressed − base, 0)
                future_scrs = [
                    max(stressed_bels[s] - post_bels[s], 0.0)
                    for s in range(t, n_periods)
                ]
                # Conditional DFs from start of period t to end of period s
                future_dfs = [
                    locked_dfs_end[s] / df_start_t
                    for s in range(t, n_periods)
                ]
                if future_scrs and all(0.0 < d <= 1.0 for d in future_dfs):
                    ra = ra_calculator.compute(future_scrs, future_dfs)
                else:
                    ra = 0.0
                ra_series.append(ra)

            per_cohort_ra[cohort_id] = ra_series

        # ------------------------------------------------------------------
        # B7 — Step GmmEngine: one call per cohort per period
        # ------------------------------------------------------------------
        per_cohort_movements: dict[str, list] = {cid: [] for cid in cohort_ids}

        for i, period in enumerate(self._calendar.periods):
            t = period.period_index
            for cohort_id in cohort_ids:
                outgos = self._composite._cohort_net_outgos.get(
                    cohort_id, [0.0] * n_periods
                )
                remaining_cu = max(providers[cohort_id].units_remaining(t), 1e-15)
                total_remaining_outgo = max(
                    sum(outgos[s] for s in range(t, n_periods)), 1e-15
                )

                gmm_result = gmm_engine.step(
                    cohort_id                = cohort_id,
                    t                        = t,
                    bel_current              = per_cohort_bel_post[cohort_id][i],
                    bel_locked               = per_cohort_bel_locked[cohort_id][i],
                    risk_adjustment          = per_cohort_ra[cohort_id][i],
                    remaining_coverage_units = remaining_cu,
                    year_fraction            = period.year_fraction,
                    actual_outgo             = outgos[i] if i < len(outgos) else 0.0,
                    total_remaining_outgo    = total_remaining_outgo,
                )
                per_cohort_movements[cohort_id].append(gmm_result)

        # ------------------------------------------------------------------
        # B8 — Save closing state and full movement table per cohort
        # ------------------------------------------------------------------
        for cohort_id in cohort_ids:
            movements   = per_cohort_movements[cohort_id]
            last        = movements[-1]
            inception   = inception_dates[cohort_id]
            remaining_cu_end = max(
                providers[cohort_id].units_remaining(n_periods), 0.0
            )

            closing_state = Ifrs17State(
                cohort_id                = cohort_id,
                valuation_date           = valuation_date,
                csm_balance              = last.csm_closing,
                loss_component           = last.loss_component_closing,
                remaining_coverage_units = remaining_cu_end,
                total_coverage_units     = opening_states[cohort_id].total_coverage_units,
                locked_in_rate           = locked_in_rate,
                inception_date           = inception,
            )
            self._ifrs17_state_store.save_state(closing_state)
            self._ifrs17_state_store.save_movements(
                cohort_id, valuation_date, movements
            )

        self._logger.info(
            "IFRS 17 GMM complete: %d cohorts, %d periods each; "
            "state and movements saved.",
            len(cohort_ids),
            n_periods,
        )
