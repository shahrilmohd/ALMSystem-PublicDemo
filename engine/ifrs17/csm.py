"""
engine/ifrs17/csm.py — CsmTracker

Tracks the Contractual Service Margin (CSM) through a single projection run.

The CSM represents unearned profit. Each period it is:
  1. Accreted at the locked-in discount rate (IFRS 17 para 44)
  2. Adjusted for FCF changes related to future service from non-financial
     assumptions (mortality, inflation) — para 44(c)
  3. Released to P&L in proportion to coverage units consumed — para 45

Financial assumption changes (discount rate) do NOT adjust the CSM.
They flow through insurance finance income/expense (P&L or OCI depending on
the OCI election). That split is handled at the GmmEngine level; CsmTracker
receives only the non-financial adjustment.

If a non-financial FCF increase causes the CSM to go negative (contract group
becomes onerous mid-projection), the CSM is floored at zero and the excess is
reported as `onerous_excess` for GmmEngine to pass to LossComponentTracker.

See DECISIONS.md §28, §34.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CsmStepResult:
    """
    All CSM movement components for a single projection period.

    Fields
    ------
    csm_opening : float
        CSM at the start of the period (before accretion).
    csm_accretion : float
        Interest accretion at the locked-in rate × year_fraction.
    csm_adjustment_non_financial : float
        Adjustment from non-financial FCF changes for future service.
        Negative means FCF improved (CSM increases); positive means deteriorated.
        Already clamped: if the adjustment would make CSM < 0, it is clamped
        and the excess is reported in onerous_excess.
    csm_release : float
        Amount released to P&L (positive value).
    csm_closing : float
        CSM at end of period. Always >= 0.
    onerous_excess : float
        Amount by which a non-financial FCF increase exceeded the available
        CSM (i.e., the portion that turned the group onerous this period).
        Zero for non-onerous groups and non-deteriorating periods.
        GmmEngine passes this to LossComponentTracker.
    """

    csm_opening:                   float
    csm_accretion:                 float
    csm_adjustment_non_financial:  float
    csm_release:                   float
    csm_closing:                   float
    onerous_excess:                float


class CsmTracker:
    """
    Stateful CSM accumulator for one contract group within a single run.

    Instantiated by GmmEngine with the opening CSM from Ifrs17State (which
    was persisted by the prior reporting run). At run end, GmmEngine reads
    self.balance to save the closing state back to Ifrs17StateStore.

    Parameters
    ----------
    opening_csm : float
        CSM at the start of the run (from Ifrs17State.csm_balance).
        Must be >= 0.
    locked_in_rate : float
        Annual discount rate frozen at contract group inception. Used for
        CSM accretion at each period. Must be > -1.
    """

    def __init__(self, opening_csm: float, locked_in_rate: float) -> None:
        if opening_csm < 0.0:
            raise ValueError(
                f"opening_csm must be >= 0, got {opening_csm}"
            )
        if locked_in_rate <= -1.0:
            raise ValueError(
                f"locked_in_rate must be > -1, got {locked_in_rate}"
            )
        self._csm             = opening_csm
        self._locked_in_rate  = locked_in_rate

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def balance(self) -> float:
        """Current CSM balance. Updated after each call to step()."""
        return self._csm

    def _set_csm(self, value: float) -> None:
        """
        Directly update the CSM balance.

        Used exclusively by GmmEngine after the JIT-compiled step
        (_gmm_step_inner) returns the new closing balance.  Not part of
        the public API; callers outside GmmEngine should use step().
        """
        self._csm = float(value)

    def step(
        self,
        units_consumed:           float,
        units_remaining_opening:  float,
        year_fraction:            float,
        fcf_change_non_financial: float = 0.0,
    ) -> CsmStepResult:
        """
        Advance CSM one projection period.

        Parameters
        ----------
        units_consumed : float
            Coverage units consumed this period (from CoverageUnitProvider).
            Must be >= 0.
        units_remaining_opening : float
            Coverage units remaining at the START of this period (before
            consuming units_consumed). Must be > 0.
        year_fraction : float
            Length of this period in years: 1/12 for monthly, 1.0 for annual.
            From ProjectionPeriod.year_fraction (DECISIONS.md §27).
        fcf_change_non_financial : float
            Change in Fulfilment Cash Flows for future service attributable to
            non-financial assumption changes (mortality, inflation).
            Positive = FCF increased = worse for insurer (reduces CSM).
            Negative = FCF decreased = better for insurer (increases CSM).
            Default 0.0 (no assumption change this period).

        Returns
        -------
        CsmStepResult
            Full movement breakdown for this period.

        Raises
        ------
        ValueError
            If units_remaining_opening <= 0 or units_consumed < 0 or
            year_fraction <= 0.
        """
        if units_remaining_opening <= 0.0:
            raise ValueError(
                f"units_remaining_opening must be > 0, "
                f"got {units_remaining_opening}"
            )
        if units_consumed < 0.0:
            raise ValueError(
                f"units_consumed must be >= 0, got {units_consumed}"
            )
        if year_fraction <= 0.0:
            raise ValueError(
                f"year_fraction must be > 0, got {year_fraction}"
            )

        csm_opening = self._csm

        # Step 1: Accretion at locked-in rate, scaled by period length.
        accretion = csm_opening * self._locked_in_rate * year_fraction

        # Step 2: Apply non-financial FCF adjustment.
        # Positive fcf_change_non_financial = FCF worsened = CSM decreases.
        csm_after_accretion   = csm_opening + accretion
        csm_after_adjustment  = csm_after_accretion - fcf_change_non_financial

        # Floor at zero: if the adjustment makes CSM negative the group is
        # (or has become) onerous. Report the excess for LossComponentTracker.
        if csm_after_adjustment < 0.0:
            onerous_excess       = -csm_after_adjustment   # positive amount
            csm_after_adjustment = 0.0
        else:
            onerous_excess = 0.0

        # The adjustment actually applied (may be clamped).
        applied_adjustment = csm_after_accretion - csm_after_adjustment

        # Step 3: Release by coverage unit fraction.
        release_fraction = min(units_consumed / units_remaining_opening, 1.0)
        release          = csm_after_adjustment * release_fraction
        csm_closing      = csm_after_adjustment - release

        # Protect against tiny floating-point negatives.
        csm_closing = max(csm_closing, 0.0)

        self._csm = csm_closing

        return CsmStepResult(
            csm_opening                  = csm_opening,
            csm_accretion                = accretion,
            csm_adjustment_non_financial = -applied_adjustment,   # sign: negative = CSM reduced
            csm_release                  = release,
            csm_closing                  = csm_closing,
            onerous_excess               = onerous_excess,
        )
