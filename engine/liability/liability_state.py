"""
LiabilityState structs — pure data containers for vectorised batch processing.

Design (DECISIONS.md §32)
--------------------------
Each struct holds all *mutable* per-step state for one liability class as
plain NumPy arrays of shape ``(n_scenarios, n_groups)``.  They contain no
methods, no DataFrames, and no business logic.

Shape convention
----------------
``(n_scenarios, n_groups)`` — scenario axis first, group axis second.  JAX
``vmap`` maps over axis 0 (scenario), so each vmapped call sees a single
``(n_groups,)`` slice.

Why the scenario dimension matters for Conventional
---------------------------------------------------
For PAR (with-profits) policies, the accrued bonus is driven by the
scenario's investment returns.  The conventional model currently uses a flat
``bonus_rate_yr`` (a Phase 1 simplification), so all scenario rows start
identical.  When the bonus rate is tied to ESG asset returns (Phase 4), the
accrued bonus diverges after the first step — every cashflow component that
depends on the benefit base (death claims, surrender payments, maturity
payments) becomes scenario-specific.  The ``(n_scenarios, n_groups)`` shape
is designed for this: the ``conventional_step`` pure function already accepts
a per-scenario ``bonus_rate_yr``, so the JAX ``vmap`` path is bonus-ready
without any interface change.

NumPy-first
-----------
All fields are plain ``np.ndarray`` here.  The JAX batch path in
``Conventional.batch_step`` converts them to ``jnp.array`` at the boundary
before calling ``vmap``.  This keeps Step 23a-i independent of JAX.

Factory functions
-----------------
Each struct has a corresponding ``*_from_mps()`` factory that converts the
initial model-point DataFrame to the batch array representation.  All
scenarios start from identical initial state — the broadcast is a cheap
memory view; ``.copy()`` materialises it to avoid aliasing bugs.

See also
--------
DECISIONS.md §32                    — vectorisation rationale and migration
engine/liability/conventional_step.py — pure step function for Conventional
engine/liability/conventional.py    — Conventional.batch_step uses these
engine/run_modes/stochastic_run.py  — _execute_vectorised calls batch_step
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# State structs
# ---------------------------------------------------------------------------

class ConventionalState(NamedTuple):
    """
    Mutable per-step state for the Conventional liability model.

    in_force:      lives in force per group.
    accrued_bonus: with-profits accrued bonus per policy (zero for non-PAR).
                   Diverges across scenarios once bonus rates are driven by
                   per-scenario investment returns (Phase 4 / Step 27).
    asset_share:   per-policy asset share (£).  Initialised from
                   ``asset_share_per_policy`` model-point column, falling back
                   to ``sum_assured`` if absent.  Updated each step by
                   crediting the scenario's investment return and netting
                   premium less expenses.  Drives terminal bonus computation
                   in SmoothedBonusStrategy.
    reserve:       per-scenario BEL populated in the backward pass; zero at
                   setup time.
    """
    in_force:      np.ndarray   # (n_scenarios, n_groups)
    accrued_bonus: np.ndarray   # (n_scenarios, n_groups)
    asset_share:   np.ndarray   # (n_scenarios, n_groups)
    reserve:       np.ndarray   # (n_scenarios, n_groups)


class InPaymentState(NamedTuple):
    """
    Mutable per-step state for InPaymentLiability (BPA in-payment cohort).

    in_force:        lives in force (weight × survival probability).
    accrued_pension: LPI-indexed annual pension; starts at pension_pa from the
                     model point, uplifted by the LPI rate each period.
    """
    in_force:        np.ndarray   # (n_scenarios, n_groups)
    accrued_pension: np.ndarray   # (n_scenarios, n_groups)


class DeferredState(NamedTuple):
    """
    Mutable per-step state for DeferredLiability (BPA deferred cohort).

    in_force: lives deferred (weight × survival probability).
    Four decrements (death, TV, ill-health, normal retirement) are applied
    per period; no additional state is needed beyond the count.
    """
    in_force: np.ndarray   # (n_scenarios, n_groups)


class DependantState(NamedTuple):
    """
    Mutable per-step state for DependantLiability (BPA contingent pension).

    member_in_force: surviving member pool — member deaths trigger dependant
                     pension payments, reducing this count over time.
    triggered:       dependant-pension weight already in-payment (member has
                     died and the dependant pension has started).
    """
    member_in_force: np.ndarray   # (n_scenarios, n_groups)
    triggered:       np.ndarray   # (n_scenarios, n_groups)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def conventional_state_from_mps(
    model_points: pd.DataFrame,
    n_scenarios: int,
) -> ConventionalState:
    """
    Build a ConventionalState by broadcasting model-point scalars.

    All scenarios start from the same initial in-force block.  The accrued
    bonus begins identical across scenarios; it diverges once scenario-
    dependent bonus rates are introduced in Phase 4.

    Parameters
    ----------
    model_points : pd.DataFrame
        Conventional model points; must contain ``in_force_count`` and
        ``accrued_bonus_per_policy`` columns.
    n_scenarios : int
        Number of ESG scenarios.  Each scenario row is initially identical.

    Returns
    -------
    ConventionalState with all fields of shape ``(n_scenarios, n_groups)``.
    """
    n_groups = len(model_points)
    base_if = model_points["in_force_count"].to_numpy(dtype=float)
    base_ab = model_points["accrued_bonus_per_policy"].to_numpy(dtype=float)

    if "asset_share_per_policy" in model_points.columns:
        base_as = model_points["asset_share_per_policy"].to_numpy(dtype=float)
    else:
        base_as = model_points["sum_assured"].to_numpy(dtype=float)

    in_force      = np.broadcast_to(base_if[np.newaxis, :], (n_scenarios, n_groups)).copy()
    accrued_bonus = np.broadcast_to(base_ab[np.newaxis, :], (n_scenarios, n_groups)).copy()
    asset_share   = np.broadcast_to(base_as[np.newaxis, :], (n_scenarios, n_groups)).copy()
    reserve       = np.zeros((n_scenarios, n_groups), dtype=float)

    return ConventionalState(
        in_force=in_force,
        accrued_bonus=accrued_bonus,
        asset_share=asset_share,
        reserve=reserve,
    )


def in_payment_state_from_mps(
    model_points: pd.DataFrame,
    n_scenarios: int,
) -> InPaymentState:
    """
    Build an InPaymentState from BPA in-payment model points.

    Parameters
    ----------
    model_points : pd.DataFrame
        Must contain ``in_force_count`` and ``pension_pa`` columns.
    n_scenarios : int

    Returns
    -------
    InPaymentState with all fields of shape ``(n_scenarios, n_groups)``.
    """
    n_groups = len(model_points)
    base_if  = model_points["in_force_count"].to_numpy(dtype=float)
    base_pen = model_points["pension_pa"].to_numpy(dtype=float)

    in_force        = np.broadcast_to(base_if[np.newaxis, :],  (n_scenarios, n_groups)).copy()
    accrued_pension = np.broadcast_to(base_pen[np.newaxis, :], (n_scenarios, n_groups)).copy()

    return InPaymentState(in_force=in_force, accrued_pension=accrued_pension)


def deferred_state_from_mps(
    model_points: pd.DataFrame,
    n_scenarios: int,
) -> DeferredState:
    """
    Build a DeferredState from BPA deferred model points.

    Parameters
    ----------
    model_points : pd.DataFrame
        Must contain ``in_force_count`` column.
    n_scenarios : int

    Returns
    -------
    DeferredState with ``in_force`` of shape ``(n_scenarios, n_groups)``.
    """
    n_groups = len(model_points)
    base_if  = model_points["in_force_count"].to_numpy(dtype=float)
    in_force = np.broadcast_to(base_if[np.newaxis, :], (n_scenarios, n_groups)).copy()
    return DeferredState(in_force=in_force)


def dependant_state_from_mps(
    model_points: pd.DataFrame,
    n_scenarios: int,
) -> DependantState:
    """
    Build a DependantState from BPA dependant model points.

    Parameters
    ----------
    model_points : pd.DataFrame
        Must contain ``weight`` column (member_weight × dependant_proportion).
    n_scenarios : int

    Returns
    -------
    DependantState with both fields of shape ``(n_scenarios, n_groups)``.
    """
    n_groups = len(model_points)
    base_if  = model_points["weight"].to_numpy(dtype=float)

    member_in_force = np.broadcast_to(base_if[np.newaxis, :], (n_scenarios, n_groups)).copy()
    triggered       = np.zeros((n_scenarios, n_groups), dtype=float)

    return DependantState(member_in_force=member_in_force, triggered=triggered)
