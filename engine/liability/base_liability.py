"""
Abstract base class for all liability models.

Design rules (from CLAUDE.md):
  - Models receive inputs; they do not fetch them.
  - The time loop lives only in engine/core/top.py — no model advances time internally.
  - Results are never stored inside models. All outputs go to ResultStore.
  - Models are stateless between time steps.

All public methods on BaseLiability subclasses are pure functions:
    input:  model_points DataFrame + assumptions + current timestep
    output: LiabilityCashflows / Decrements / float

The model_points DataFrame uses the unified format produced by
liability_data_loader.py (Step 6 of the build order). Whether it came
from seriatim policies or group model points is irrelevant to the
liability model — it always sees the same column schema.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Result data types
# ---------------------------------------------------------------------------

@dataclass
class LiabilityCashflows:
    """
    Cash flows produced by a liability model for a single projection time step.

    All amounts are totals across all model points (not per-policy).

    Sign convention:
        premiums            — positive = income received by the insurer
        death_claims        — positive = outgo paid by the insurer
        surrender_payments  — positive = outgo paid by the insurer
        maturity_payments   — positive = outgo paid by the insurer
        expenses            — positive = outgo paid by the insurer

    net_outgo is positive when the insurer pays out more than it receives.
    """
    timestep:           int
    premiums:           float
    death_claims:       float
    surrender_payments: float
    maturity_payments:  float
    expenses:           float

    @property
    def net_outgo(self) -> float:
        """
        Net cash outgo for the insurer.

        Positive = net outgo (insurer pays more than it receives).
        Negative = net income (insurer receives more than it pays).
        """
        return (
            self.death_claims
            + self.surrender_payments
            + self.maturity_payments
            + self.expenses
            - self.premiums
        )


@dataclass
class Decrements:
    """
    Policyholder decrements for a single projection time step.

    All counts are totals across all model points.

    Identity: in_force_end = in_force_start - deaths - lapses - maturities
    """
    timestep:       int
    in_force_start: float
    deaths:         float
    lapses:         float
    maturities:     float
    in_force_end:   float


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseLiability(ABC):
    """
    Abstract base class for all liability models.

    Concrete subclasses:
        Conventional  — traditional with-profits / without-profits
        UnitLinked    — unit account, asset share          (Phase 1, later)
        Annuity       — annuity cash flows, RM, CREC       (Phase 1, later)

    All public methods are pure functions. Given the same inputs they
    always return the same outputs. No internal state is mutated.

    Parameters shared across all methods:
        model_points:  pandas DataFrame in the unified liability input schema.
                       Column contract is defined per subclass.
        assumptions:   Product-specific assumption object. Type is defined by
                       each subclass; Any is used here to avoid coupling the
                       ABC to a specific assumptions type.
        timestep:      0-based integer index of the current projection step.
                       Used to label results. Does not represent calendar time.
    """

    @abstractmethod
    def project_cashflows(
        self,
        model_points: pd.DataFrame,
        assumptions: Any,
        timestep: int,
    ) -> LiabilityCashflows:
        """
        Project cash flows for a single time step.

        Args:
            model_points: In-force policies at the START of this time step.
            assumptions:  Product-specific assumption object.
            timestep:     Current projection step index (0-based).

        Returns:
            LiabilityCashflows aggregated across all model points.
        """

    @abstractmethod
    def get_bel(
        self,
        model_points: pd.DataFrame,
        assumptions: Any,
        timestep: int,
    ) -> float:
        """
        Calculate Best Estimate Liability (BEL) at the current time step.

        BEL = present value of future net cash outgo, projected forward
        from the current in-force block for the remaining projection term.

        Args:
            model_points: In-force policies at the current time step.
            assumptions:  Product-specific assumption object.
            timestep:     Current projection step index (used for labelling).

        Returns:
            BEL as a single float (total across all model points).
        """

    @abstractmethod
    def get_reserve(
        self,
        model_points: pd.DataFrame,
        assumptions: Any,
        timestep: int,
    ) -> float:
        """
        Calculate reserve at the current time step.

        Phase 1: reserve = BEL (no risk margin).
        Later phases may add a risk margin above BEL.

        Args:
            model_points: In-force policies at the current time step.
            assumptions:  Product-specific assumption object.
            timestep:     Current projection step index.

        Returns:
            Reserve as a single float.
        """

    @abstractmethod
    def get_decrements(
        self,
        model_points: pd.DataFrame,
        assumptions: Any,
        timestep: int,
    ) -> Decrements:
        """
        Calculate policyholder decrements for this time step.

        Args:
            model_points: In-force policies at the START of this time step.
            assumptions:  Product-specific assumption object.
            timestep:     Current projection step index.

        Returns:
            Decrements for this time step (deaths, lapses, maturities, in-force).
        """
