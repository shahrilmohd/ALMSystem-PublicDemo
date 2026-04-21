"""
engine/ifrs17/state.py — Ifrs17State

Cross-period rolling IFRS 17 balances for a single contract group.

IFRS 17 is roll-forward accounting: the closing CSM balance from one reporting
period is the opening balance for the next. This dataclass is the unit of state
that persists between runs via Ifrs17StateStore (storage/ifrs17_state_repository.py).

See DECISIONS.md §33 for the full design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Ifrs17State:
    """
    IFRS 17 rolling balances for one contract group at one valuation date.

    All monetary fields are in the same currency unit as the BEL inputs.
    This dataclass is immutable — GmmEngine produces a new Ifrs17State at the
    end of each period; it does not mutate the opening state.

    Attributes
    ----------
    cohort_id : str
        Contract group identifier. Maps to ResultStore cohort_id (DECISIONS.md §17).
    valuation_date : date
        The reporting date this state was saved (closing date of the last run).
    csm_balance : float
        Contractual Service Margin at this valuation date.
        Zero for onerous contract groups (loss component holds the deficit).
    loss_component : float
        Loss component balance. Zero for non-onerous groups.
        CSM and loss_component are mutually exclusive: at most one is non-zero.
    remaining_coverage_units : float
        PV of expected future annuity payments at this valuation date,
        discounted at the locked-in rate. Decreases each period as outgo
        is recognised. See DECISIONS.md §35.
    total_coverage_units : float
        PV of expected future annuity payments at the inception date,
        discounted at the locked-in rate. Fixed at inception; never updated.
        Used as the denominator when computing historical release fractions.
    locked_in_rate : float
        Annual discount rate locked at contract group inception.
        For BPA: post-MA parallel shift rate at transaction date (DECISIONS.md §21).
        Used for CSM accretion and locked-in BEL calculation. Never changes.
    inception_date : date
        Date of first recognition of this contract group.
    """

    cohort_id:                str
    valuation_date:            date
    csm_balance:               float
    loss_component:            float
    remaining_coverage_units:  float
    total_coverage_units:      float
    locked_in_rate:            float
    inception_date:            date

    def __post_init__(self) -> None:
        if self.csm_balance < 0.0:
            raise ValueError(
                f"csm_balance must be >= 0 (got {self.csm_balance}). "
                "Negative CSM is not valid; use loss_component for onerous groups."
            )
        if self.loss_component < 0.0:
            raise ValueError(
                f"loss_component must be >= 0 (got {self.loss_component})."
            )
        if self.csm_balance > 0.0 and self.loss_component > 0.0:
            raise ValueError(
                "csm_balance and loss_component cannot both be non-zero. "
                "A contract group is either profitable (CSM > 0) or onerous "
                "(loss_component > 0), never both."
            )
        if self.remaining_coverage_units < 0.0:
            raise ValueError(
                f"remaining_coverage_units must be >= 0 "
                f"(got {self.remaining_coverage_units})."
            )
        if self.total_coverage_units <= 0.0:
            raise ValueError(
                f"total_coverage_units must be > 0 "
                f"(got {self.total_coverage_units})."
            )
        if self.remaining_coverage_units > self.total_coverage_units + 1e-9:
            raise ValueError(
                f"remaining_coverage_units ({self.remaining_coverage_units}) "
                f"cannot exceed total_coverage_units ({self.total_coverage_units})."
            )
        if self.locked_in_rate <= -1.0:
            raise ValueError(
                f"locked_in_rate must be > -1 (got {self.locked_in_rate})."
            )
        if self.valuation_date < self.inception_date:
            raise ValueError(
                f"valuation_date ({self.valuation_date}) cannot be before "
                f"inception_date ({self.inception_date})."
            )
