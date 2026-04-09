"""
engine/ifrs17/loss_component.py — LossComponentTracker

Tracks the loss component for onerous contract groups.

A contract group is onerous when its Fulfilment Cash Flows (FCF) at inception
are positive (i.e., the contract is a net cost). The loss component equals
FCF_0 and is recognised immediately in P&L. No CSM is established.

If a previously profitable group becomes onerous mid-projection (FCF increases
exceed the available CSM), GmmEngine transfers the excess from CsmTracker to
LossComponentTracker via add_onerous_excess().

The loss component is released to P&L as the onerous contracts generate
cash flows — in proportion to the actual outgo each period relative to the
total remaining expected outgo. Once exhausted, the loss component is zero
and the contract group is no longer onerous.

See DECISIONS.md §28 for the design rationale.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LossComponentStepResult:
    """
    Loss component movements for a single projection period.

    Fields
    ------
    opening : float
        Loss component balance at the start of the period.
    addition : float
        Amount added this period (from onerous_excess passed by CsmTracker).
        Zero for periods where the group is not newly onerous.
    release : float
        Amount released to P&L this period (positive value).
    closing : float
        Loss component balance at end of period. Always >= 0.
    """

    opening:  float
    addition: float
    release:  float
    closing:  float


class LossComponentTracker:
    """
    Stateful loss component accumulator for one contract group within a run.

    Instantiated by GmmEngine with the opening loss component from Ifrs17State.
    For non-onerous groups this is always 0.0 and remains so unless CsmTracker
    reports an onerous_excess during the run.

    Parameters
    ----------
    opening_loss_component : float
        Loss component at the start of the run (from Ifrs17State.loss_component).
        Must be >= 0.
    """

    def __init__(self, opening_loss_component: float = 0.0) -> None:
        if opening_loss_component < 0.0:
            raise ValueError(
                f"opening_loss_component must be >= 0, "
                f"got {opening_loss_component}"
            )
        self._balance = opening_loss_component

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def balance(self) -> float:
        """Current loss component balance. Updated after each step()."""
        return self._balance

    def _set_balance(self, value: float) -> None:
        """
        Directly update the loss component balance.

        Used exclusively by GmmEngine after the JIT-compiled step
        (_gmm_step_inner) returns the new closing balance.  Not part of
        the public API; callers outside GmmEngine should use step().
        """
        self._balance = float(value)

    def add_onerous_excess(self, excess: float) -> None:
        """
        Add an onerous excess amount transferred from CsmTracker.

        Called by GmmEngine when CsmStepResult.onerous_excess > 0. The excess
        is the amount by which a non-financial FCF increase exceeded the
        available CSM, making the group (or keeping it) onerous.

        Parameters
        ----------
        excess : float
            Must be >= 0. Zero calls are a no-op.
        """
        if excess < 0.0:
            raise ValueError(f"excess must be >= 0, got {excess}")
        self._balance += excess

    def step(
        self,
        actual_outgo:          float,
        total_remaining_outgo: float,
    ) -> LossComponentStepResult:
        """
        Release a portion of the loss component proportional to actual outgo.

        The release fraction is: actual_outgo / total_remaining_outgo.
        This mirrors the coverage unit release mechanic used for CSM — the
        loss component shrinks as the onerous contracts pay out their expected
        losses.

        If the loss component balance is zero, the step is a no-op (returns
        all-zero result). This allows GmmEngine to call step() uniformly
        regardless of whether the group is currently onerous.

        Parameters
        ----------
        actual_outgo : float
            Expected cash outgo in the current period. Must be >= 0.
        total_remaining_outgo : float
            Total expected outgo from this period to end of projection.
            Must be > 0.

        Returns
        -------
        LossComponentStepResult

        Raises
        ------
        ValueError
            If actual_outgo < 0 or total_remaining_outgo <= 0.
        """
        if actual_outgo < 0.0:
            raise ValueError(f"actual_outgo must be >= 0, got {actual_outgo}")
        if total_remaining_outgo <= 0.0:
            raise ValueError(
                f"total_remaining_outgo must be > 0, "
                f"got {total_remaining_outgo}"
            )

        opening  = self._balance
        addition = 0.0   # additions are made via add_onerous_excess() before step()

        if opening == 0.0:
            return LossComponentStepResult(
                opening=opening, addition=addition, release=0.0, closing=0.0
            )

        release_fraction = min(actual_outgo / total_remaining_outgo, 1.0)
        release          = opening * release_fraction
        closing          = max(opening - release, 0.0)

        self._balance = closing

        return LossComponentStepResult(
            opening=opening,
            addition=addition,
            release=release,
            closing=closing,
        )
