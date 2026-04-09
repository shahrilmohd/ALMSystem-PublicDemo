"""
engine/ifrs17/gmm.py — GmmEngine and GmmStepResult

Orchestrates all IFRS 17 General Measurement Model (GMM) components for one
or more contract groups through a single projection run.

GmmEngine is product-agnostic. Products inject:
  - opening Ifrs17State per contract group (loaded from Ifrs17StateStore)
  - CoverageUnitProvider per contract group (computes period CU values)
  - locked-in rates per contract group (from Ifrs17State)

The run mode orchestrator calls step() once per period per contract group,
supplying the current-period BEL values and FCF attribution inputs. At run
end, GmmEngine exposes the closing state for each contract group so the run
mode can persist it via Ifrs17StateStore.

See DECISIONS.md §28, §33, §34 for the full design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.ifrs17._gmm_jit import _gmm_step_inner
from engine.ifrs17.coverage_units import CoverageUnitProvider
from engine.ifrs17.csm import CsmTracker
from engine.ifrs17.loss_component import LossComponentTracker
from engine.ifrs17.state import Ifrs17State


@dataclass(frozen=True)
class GmmStepResult:
    """
    All IFRS 17 GMM movements for a single contract group in a single period.

    Fields — CSM movements
    ----------------------
    csm_opening : float
        CSM at the start of the period.
    csm_accretion : float
        Interest accretion at the locked-in rate × year_fraction.
    csm_adjustment_non_financial : float
        CSM adjustment from non-financial FCF changes (mortality, inflation).
        Negative = CSM reduced; positive = CSM increased.
    csm_release : float
        Amount released to P&L (always >= 0).
    csm_closing : float
        CSM at end of period (always >= 0).

    Fields — Loss component movements
    ----------------------------------
    loss_component_opening : float
    loss_component_addition : float
        Amount added from onerous excess (transferred from CSM).
    loss_component_release : float
        Amount released to P&L.
    loss_component_closing : float

    Fields — Insurance finance income/expense (DECISIONS.md §34)
    -------------------------------------------------------------
    insurance_finance_pl : float
        BEL unwinding at the locked-in rate = bel_locked(t) - bel_locked(t-1).
        Recognised in P&L. Supplied by the run mode (requires prior-period
        bel_locked); GmmEngine stores it as reported.
    insurance_finance_oci : float
        Change in the current-vs-locked gap = (bel_current - bel_locked) this
        period minus (bel_current - bel_locked) prior period.
        Recognised in OCI (if OCI election made). Supplied by the run mode.

    Fields — Balance sheet
    ----------------------
    lrc : float
        Liability for Remaining Coverage = bel_current + risk_adjustment + csm_closing.
    lic : float
        Liability for Incurred Claims = supplied by run mode (past-service BEL + RA).
        Zero for contract groups with no incurred-but-not-reported claims.

    Fields — P&L summary
    ---------------------
    p_and_l_csm_release : float
        P&L contribution from CSM release (= csm_release).
    p_and_l_loss_component : float
        P&L contribution from loss component release and any new onerous excess.
        = loss_component_release - loss_component_addition  (net P&L impact).
    p_and_l_insurance_finance : float
        = insurance_finance_pl  (the OCI portion is not in P&L).

    Input echoes (for traceability)
    --------------------------------
    bel_current : float
    bel_locked : float
    risk_adjustment : float
    """

    # CSM
    csm_opening:                   float
    csm_accretion:                 float
    csm_adjustment_non_financial:  float
    csm_release:                   float
    csm_closing:                   float
    # Loss component
    loss_component_opening:        float
    loss_component_addition:       float
    loss_component_release:        float
    loss_component_closing:        float
    # Finance income/expense
    insurance_finance_pl:          float
    insurance_finance_oci:         float
    # Balance sheet
    lrc:                           float
    lic:                           float
    # P&L summary
    p_and_l_csm_release:           float
    p_and_l_loss_component:        float
    p_and_l_insurance_finance:     float
    # Input echoes
    bel_current:                   float
    bel_locked:                    float
    risk_adjustment:               float


class GmmEngine:
    """
    IFRS 17 GMM orchestrator for one or more contract groups.

    Instantiated once per projection run by the run mode orchestrator.
    The run mode loads Ifrs17State from Ifrs17StateStore before constructing
    GmmEngine, and saves the closing state after the final step().

    Parameters
    ----------
    contract_groups : list[str]
        cohort_id values managed by this engine instance.
    opening_states : dict[str, Ifrs17State]
        Opening IFRS 17 state per cohort_id, loaded from Ifrs17StateStore.
        For a first run (inception), the state is constructed from FCF at t=0
        by the run mode before GmmEngine is created.
    coverage_unit_providers : dict[str, CoverageUnitProvider]
        CoverageUnitProvider per cohort_id. Injected by the product layer.
    """

    def __init__(
        self,
        contract_groups:          list[str],
        opening_states:           dict[str, Ifrs17State],
        coverage_unit_providers:  dict[str, CoverageUnitProvider],
    ) -> None:
        missing_states    = set(contract_groups) - set(opening_states)
        missing_providers = set(contract_groups) - set(coverage_unit_providers)
        if missing_states:
            raise ValueError(
                f"No opening Ifrs17State for cohort_ids: {sorted(missing_states)}"
            )
        if missing_providers:
            raise ValueError(
                f"No CoverageUnitProvider for cohort_ids: {sorted(missing_providers)}"
            )

        self._groups    = contract_groups
        self._states    = opening_states
        self._providers = coverage_unit_providers

        # Per-group trackers — initialised from opening state
        self._csm_trackers: dict[str, CsmTracker] = {
            cid: CsmTracker(
                opening_csm   = opening_states[cid].csm_balance,
                locked_in_rate= opening_states[cid].locked_in_rate,
            )
            for cid in contract_groups
        }
        self._lc_trackers: dict[str, LossComponentTracker] = {
            cid: LossComponentTracker(
                opening_loss_component=opening_states[cid].loss_component,
            )
            for cid in contract_groups
        }

        # Prior-period bel_locked per group for finance income computation.
        # Initialised to None; the run mode must supply prior values or
        # pass insurance_finance_pl / insurance_finance_oci directly.
        self._prior_bel_locked: dict[str, float | None] = {
            cid: None for cid in contract_groups
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def step(
        self,
        cohort_id:                 str,
        t:                         int,
        bel_current:               float,
        bel_locked:                float,
        risk_adjustment:           float,
        remaining_coverage_units:  float,
        year_fraction:             float  = 1.0,
        fcf_change_non_financial:  float  = 0.0,
        actual_outgo:              float  = 0.0,
        total_remaining_outgo:     float  = 1.0,
        lic:                       float  = 0.0,
        insurance_finance_pl:      float | None = None,
        insurance_finance_oci:     float | None = None,
    ) -> GmmStepResult:
        """
        Advance one contract group one projection period.

        Parameters
        ----------
        cohort_id : str
            Contract group to advance. Must be in contract_groups.
        t : int
            0-based projection period index.
        bel_current : float
            BEL discounted at the current-period rate (balance sheet value).
        bel_locked : float
            BEL discounted at the locked-in rate (for finance income split).
        risk_adjustment : float
            Risk Adjustment for this period (from CostOfCapitalRA or similar).
            Must be >= 0.
        remaining_coverage_units : float
            Coverage units remaining at the START of this period
            (from CoverageUnitProvider.units_remaining(t)). Must be > 0.
        year_fraction : float
            Period length in years (from ProjectionPeriod.year_fraction).
            Default 1.0 (annual). Must be > 0.
        fcf_change_non_financial : float
            FCF change for future service from non-financial assumptions.
            Positive = FCF worsened. Default 0.0.
        actual_outgo : float
            Expected cash outgo this period. Used for loss component release.
            Must be >= 0. Default 0.0.
        total_remaining_outgo : float
            Total expected outgo from this period to end. Used for loss
            component release fraction. Must be > 0. Default 1.0 (safe no-op
            when loss_component is zero).
        lic : float
            Liability for Incurred Claims (past-service BEL + RA).
            Default 0.0.
        insurance_finance_pl : float | None
            Pre-computed insurance finance P&L (BEL unwinding at locked-in
            rate). If None, computed internally as
            bel_locked - prior_bel_locked (or 0 on the first period).
        insurance_finance_oci : float | None
            Pre-computed OCI component. If None, computed internally as
            change in (bel_current - bel_locked) gap vs prior period.

        Returns
        -------
        GmmStepResult
        """
        if cohort_id not in self._groups:
            raise ValueError(
                f"Unknown cohort_id '{cohort_id}'. "
                f"Known groups: {self._groups}"
            )
        if risk_adjustment < 0.0:
            raise ValueError(
                f"risk_adjustment must be >= 0, got {risk_adjustment}"
            )

        csm_tracker    = self._csm_trackers[cohort_id]
        lc_tracker     = self._lc_trackers[cohort_id]
        units_consumed = self._providers[cohort_id].units_consumed(t)
        locked_in_rate = self._states[cohort_id].locked_in_rate

        # --- Insurance finance income/expense ---------------------------
        # Resolved before the JIT boundary: depends on prior_bel_locked
        # mutable state which cannot enter the pure function.
        prior_locked = self._prior_bel_locked[cohort_id]

        if insurance_finance_pl is None:
            finance_pl = (bel_locked - prior_locked) if prior_locked is not None else 0.0
        else:
            finance_pl = float(insurance_finance_pl)

        if insurance_finance_oci is None:
            current_gap = bel_current - bel_locked
            if prior_locked is not None:
                # Simplified: caller should supply for precise OCI split.
                finance_oci = current_gap
            else:
                finance_oci = 0.0
        else:
            finance_oci = float(insurance_finance_oci)

        self._prior_bel_locked[cohort_id] = bel_locked

        # --- JIT-compiled arithmetic ------------------------------------
        # Snapshot the opening balances before the pure function runs.
        csm_opening_val = csm_tracker.balance
        lc_opening_val  = lc_tracker.balance

        (
            csm_accretion,
            csm_adj_nf,
            csm_release,
            csm_closing,
            lc_opening_after_addition,
            lc_release,
            lc_closing,
            lrc,
        ) = _gmm_step_inner(
            csm_opening_val,
            locked_in_rate,
            year_fraction,
            units_consumed,
            remaining_coverage_units,
            fcf_change_non_financial,
            lc_opening_val,
            actual_outgo,
            total_remaining_outgo,
            bel_current,
            risk_adjustment,
        )

        # Onerous excess added inside _gmm_step_inner = difference between
        # the lc balance entering the function and lc_opening_after_addition.
        lc_addition = float(lc_opening_after_addition) - lc_opening_val

        # --- Update tracker state from JIT results ----------------------
        csm_tracker._set_csm(float(csm_closing))
        lc_tracker._set_balance(float(lc_closing))

        # --- P&L summary ------------------------------------------------
        p_and_l_lc = float(lc_release) - lc_addition

        return GmmStepResult(
            csm_opening                  = csm_opening_val,
            csm_accretion                = float(csm_accretion),
            csm_adjustment_non_financial = float(csm_adj_nf),
            csm_release                  = float(csm_release),
            csm_closing                  = float(csm_closing),
            loss_component_opening       = float(lc_opening_after_addition),
            loss_component_addition      = lc_addition,
            loss_component_release       = float(lc_release),
            loss_component_closing       = float(lc_closing),
            insurance_finance_pl         = finance_pl,
            insurance_finance_oci        = finance_oci,
            lrc                          = float(lrc),
            lic                          = lic,
            p_and_l_csm_release          = float(csm_release),
            p_and_l_loss_component       = p_and_l_lc,
            p_and_l_insurance_finance    = finance_pl,
            bel_current                  = bel_current,
            bel_locked                   = bel_locked,
            risk_adjustment              = risk_adjustment,
        )

    def closing_state(self, cohort_id: str, valuation_date, inception_date) -> Ifrs17State:
        """
        Build the closing Ifrs17State for persistence after the run.

        Called by the run mode orchestrator after the final step() to
        construct the state that will be saved via Ifrs17StateStore.

        Parameters
        ----------
        cohort_id : str
        valuation_date : date
            The reporting date of this run.
        inception_date : date
            From the opening Ifrs17State.
        """
        opening = self._states[cohort_id]
        return Ifrs17State(
            cohort_id                = cohort_id,
            valuation_date           = valuation_date,
            csm_balance              = self._csm_trackers[cohort_id].balance,
            loss_component           = self._lc_trackers[cohort_id].balance,
            remaining_coverage_units = self._providers[cohort_id].units_remaining(
                # Pass a sentinel — providers typically handle end-of-run
                # by returning 0; the run mode can override this value.
                0
            ),
            total_coverage_units     = opening.total_coverage_units,
            locked_in_rate           = opening.locked_in_rate,
            inception_date           = inception_date,
        )
