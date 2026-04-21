"""
Pure step function for the Conventional liability model.

Design (DECISIONS.md §32, Step 3)
-----------------------------------
This module extracts the mathematical core of ``Conventional._apply_decrements``
and ``Conventional._advance_model_points`` into a standalone pure function
``conventional_step`` that:

  - accepts only arrays and scalars (no DataFrames, no ``self``, no dicts)
  - has no side effects and no internal state
  - is numerically equivalent to the OOP scalar path for a single-group input
  - is JAX-compatible: all operations are element-wise array arithmetic,
    so ``jax.vmap`` can be applied without any code change

Purpose
-------
The scalar OOP interface (``project_cashflows``, ``get_decrements``, etc.) is
the canonical, auditable representation.  ``conventional_step`` is the
performance execution backend for the stochastic vectorised path only.
It must not be used for deterministic or liability-only runs.

Pre-computed static data (``ConventionalStepData``)
----------------------------------------------------
Rate lookups (mortality, lapse, surrender value) use Python dicts in the OOP
path.  Dict lookups are not JAX-traceable, so the batch path requires them to
be materialised as per-group arrays BEFORE calling ``vmap``.

``ConventionalStepData`` holds these pre-computed arrays alongside other
per-group static attributes (sum_assured, annual_premium, etc.).  The factory
``make_step_data`` converts a model-points DataFrame and
``ConventionalAssumptions`` into this struct once per timestep (because
``is_final_month``, ``q_monthly``, and ``w_monthly`` depend on
``policy_duration_mths`` and ``attained_age``, which advance each month).

Per-scenario bonus_rate_yr
--------------------------
The conventional model currently uses a flat ``bonus_rate_yr`` from
``ConventionalAssumptions`` (a Phase 1 simplification).  The step function
accepts ``bonus_rate_yr`` as an explicit per-scenario scalar so that JAX
``vmap`` can map over it (``in_axes=0``) when bonus rates are eventually
driven by ESG investment returns in Phase 4.  For now, callers pass
``np.full(n_scenarios, assumptions.bonus_rate_yr)``.

See also
--------
DECISIONS.md §32                    — vectorisation rationale and migration
engine/liability/conventional.py    — Conventional.batch_step uses these
engine/liability/liability_state.py — ConventionalState consumed here
"""
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd

jnp: Any
try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover — JAX optional at import time
    jnp = np

from engine.liability.conventional import ConventionalAssumptions


# ---------------------------------------------------------------------------
# Pre-computed static data struct
# ---------------------------------------------------------------------------

class ConventionalStepData(NamedTuple):
    """
    Pre-computed per-group static arrays for one conventional projection step.

    All array fields have shape ``(n_groups,)``.  Built once per timestep via
    ``make_step_data``; shared across all scenarios (scenario-independent).

    Fields
    ------
    q_monthly : ndarray
        Monthly mortality probability per group (UDD from annual q_x,
        clamped to [0, 1]).
    w_monthly : ndarray
        Monthly lapse probability per group (UDD from annual w_x,
        clamped to [0, 1]).
    sv_factor : ndarray
        Surrender value factor per group (clamped to [0, 1]).
    is_final_month : ndarray
        Float mask — 1.0 if this is the final projection month for the group
        (remaining_term_mths <= 1), else 0.0.  Maturities occur only here.
    is_par : ndarray
        Float mask — 1.0 for ENDOW_PAR groups, 0.0 otherwise.
        Gates the accrued-bonus update and benefit-base uplift.
    is_term : ndarray
        Float mask — 1.0 for TERM groups, 0.0 otherwise.
        TERM policies have no surrender payment and no maturity benefit.
    sum_assured : ndarray
        Sum assured per policy (£), shape (n_groups,).
    annual_premium : ndarray
        Annual premium per policy (£/yr), shape (n_groups,).
    expense_pct_premium : float
        Expenses as a fraction of annual premium (scalar).
    expense_per_policy : float
        Fixed annual expense per in-force policy (£/yr, scalar).
    """
    q_monthly:           Any   # (n_groups,) np.ndarray or jax.Array
    w_monthly:           Any   # (n_groups,)
    sv_factor:           Any   # (n_groups,)
    is_final_month:      Any   # (n_groups,) float 0/1
    is_par:              Any   # (n_groups,) float 0/1
    is_term:             Any   # (n_groups,) float 0/1
    sum_assured:         Any   # (n_groups,)
    annual_premium:      Any   # (n_groups,)
    expense_pct_premium: float
    expense_per_policy:  float


def make_step_data(
    model_points: pd.DataFrame,
    assumptions: ConventionalAssumptions,
) -> ConventionalStepData:
    """
    Build ``ConventionalStepData`` from a model-points DataFrame.

    Called once per projection step since ``is_final_month``, ``q_monthly``,
    and ``w_monthly`` depend on ``policy_duration_mths`` and ``attained_age``,
    which advance each month.

    Parameters
    ----------
    model_points : pd.DataFrame
        Conventional model points at the current projection step.  Must
        contain all ``Conventional.REQUIRED_COLUMNS``.
    assumptions : ConventionalAssumptions

    Returns
    -------
    ConventionalStepData with ``(n_groups,)`` arrays.
    """
    # Duration in whole years for lapse/SV rate lookups
    duration_yr = model_points["policy_duration_mths"].to_numpy(dtype=int) // 12

    # Materialise dict lookups → per-group arrays, clamped to [0, 1]
    q_yr = np.clip(
        [assumptions.get_mortality_rate_yr(int(a)) for a in model_points["attained_age"]],
        0.0, 1.0,
    )
    w_yr = np.clip(
        [assumptions.get_lapse_rate_yr(int(d)) for d in duration_yr],
        0.0, 1.0,
    )
    sv = np.clip(
        [assumptions.get_surrender_value_factor(int(d)) for d in duration_yr],
        0.0, 1.0,
    )

    # UDD monthly conversion: q_mths = 1 - (1 - q_yr)^(1/12)
    q_monthly = 1.0 - (1.0 - q_yr) ** (1.0 / 12.0)
    w_monthly = 1.0 - (1.0 - w_yr) ** (1.0 / 12.0)

    # Remaining term → is_final_month mask
    remaining = np.clip(
        model_points["policy_term_yr"].to_numpy(dtype=int) * 12
        - model_points["policy_duration_mths"].to_numpy(dtype=int),
        a_min=0, a_max=None,
    )
    is_final_month = (remaining <= 1).astype(float)

    # Policy code masks (float for JAX-compatible arithmetic)
    codes = model_points["policy_code"].to_numpy()
    is_par  = (codes == "ENDOW_PAR").astype(float)
    is_term = (codes == "TERM").astype(float)

    return ConventionalStepData(
        q_monthly=np.asarray(q_monthly, dtype=float),
        w_monthly=np.asarray(w_monthly, dtype=float),
        sv_factor=np.asarray(sv, dtype=float),
        is_final_month=is_final_month,
        is_par=is_par,
        is_term=is_term,
        sum_assured=model_points["sum_assured"].to_numpy(dtype=float),
        annual_premium=model_points["annual_premium"].to_numpy(dtype=float),
        expense_pct_premium=float(assumptions.expense_pct_premium),
        expense_per_policy=float(assumptions.expense_per_policy),
    )


# ---------------------------------------------------------------------------
# Pure step function
# ---------------------------------------------------------------------------

def conventional_step(
    in_force:             Any,
    accrued_bonus:        Any,
    asset_share:          Any,
    bonus_rate_yr:        Any,
    terminal_bonus_rate:  Any,
    earned_return_monthly: Any,
    step_data:            ConventionalStepData,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    """
    Pure single-scenario monthly step for the Conventional liability model.

    Replicates exactly one monthly application of
    ``Conventional._apply_decrements`` followed by
    ``Conventional._advance_model_points``, expressed as element-wise array
    arithmetic with no DataFrames, no dict lookups, and no side effects.

    All operations are element-wise and JAX-traceable.
    ``Conventional.batch_step`` applies ``jax.vmap(conventional_step,
    in_axes=(0, 0, 0, 0, 0, 0, None))`` over the scenario dimension.

    Parameters
    ----------
    in_force : ndarray, shape (n_groups,)
        Lives in force at the START of this month per model-point group.
    accrued_bonus : ndarray, shape (n_groups,)
        Accrued with-profits bonus per policy at the START of this month.
        Zero for non-PAR groups.
    asset_share : ndarray, shape (n_groups,)
        Per-policy asset share (£) at the START of this month.
        Initialised to sum_assured (or asset_share_per_policy if present).
    bonus_rate_yr : float
        Annual bonus rate applied to sum_assured for PAR policies.
        Per-scenario scalar driven by SmoothedBonusStrategy (Step 27).
    terminal_bonus_rate : float
        Terminal bonus loading per scenario, expressed as a fraction of
        the guaranteed benefit (sum_assured + accrued_bonus).  Applied to
        death and maturity payments only (not surrenders — sv_factor already
        embeds discretionary terminal bonus for early exits).
    earned_return_monthly : float
        Scenario's monthly investment return (net of charges).
        Used to credit the asset share each period.
    step_data : ConventionalStepData
        Pre-computed per-group arrays for this projection step.
        Broadcast (not vmapped) — shared across all scenarios.

    Returns
    -------
    new_in_force : ndarray, shape (n_groups,)
        Lives in force after applying all three decrements.
    new_accrued_bonus : ndarray, shape (n_groups,)
        Accrued bonus per policy after monthly PAR bonus accrual.
    new_asset_share : ndarray, shape (n_groups,)
        Asset share after crediting investment return and netting premiums
        less expenses.
    deaths : ndarray, shape (n_groups,)
        Deaths this month per group.
    lapses : ndarray, shape (n_groups,)
        Lapses this month per group.
    maturities : ndarray, shape (n_groups,)
        Maturities this month per group (non-zero only in final month).
    premiums : float
        Total premium income this month across all groups.
    death_claims : float
        Total death benefit paid this month (including terminal bonus).
    surrender_payments : float
        Total surrender value paid this month (no terminal bonus).
    maturity_payments : float
        Total maturity benefit paid this month (including terminal bonus).
    expenses : float
        Total expenses this month.

    Notes
    -----
    Decrement ordering mirrors the OOP implementation:
      1. Deaths    from in_force
      2. Lapses    from survivors after deaths
      3. Maturities from survivors after lapses (final month only)

    Benefit base for PAR policies = sum_assured + accrued_bonus.
    TERM policies have zero surrender and zero maturity benefit.
    Terminal bonus is gated by ``is_par`` so it only applies to PAR groups.
    """
    sd = step_data

    # ------------------------------------------------------------------
    # Decrement cascade
    # ------------------------------------------------------------------
    deaths                 = in_force * sd.q_monthly
    survivors_after_deaths = in_force - deaths
    lapses                 = survivors_after_deaths * sd.w_monthly
    survivors_after_lapses = survivors_after_deaths - lapses
    maturities             = survivors_after_lapses * sd.is_final_month
    new_in_force           = survivors_after_lapses - maturities

    # ------------------------------------------------------------------
    # Benefit base (PAR includes accrued bonus)
    # ------------------------------------------------------------------
    benefit_base = sd.sum_assured + accrued_bonus * sd.is_par

    # ------------------------------------------------------------------
    # Terminal bonus (PAR only; excluded from surrenders — see docstring)
    # ------------------------------------------------------------------
    terminal_benefit = terminal_bonus_rate * benefit_base * sd.is_par

    # ------------------------------------------------------------------
    # Cash flows
    # ------------------------------------------------------------------

    # Premiums: annual / 12, based on opening in-force
    # Use jnp.sum so the result is JAX-traceable when called inside vmap.
    # In the non-JAX path jnp == np and behaviour is identical.
    premiums = jnp.sum(sd.annual_premium / 12.0 * in_force)

    # Death claims (all product types; PAR includes terminal bonus)
    death_claims = jnp.sum((benefit_base + terminal_benefit) * deaths)

    # Surrender payments: TERM pays nothing; no terminal bonus (sv_factor embeds it)
    sv_base            = benefit_base * (1.0 - sd.is_term)
    surrender_payments = jnp.sum(sd.sv_factor * sv_base * lapses)

    # Maturity payments: TERM pays nothing; PAR includes terminal bonus
    maturity_base     = benefit_base * (1.0 - sd.is_term)
    terminal_mat      = terminal_benefit * (1.0 - sd.is_term)
    maturity_payments = jnp.sum((maturity_base + terminal_mat) * maturities)

    # Expenses: (pct_premium × annual_premium + per_policy) / 12
    expenses = jnp.sum(
        (sd.expense_pct_premium * sd.annual_premium / 12.0
         + sd.expense_per_policy / 12.0)
        * in_force
    )

    # ------------------------------------------------------------------
    # Advance accrued bonus (PAR only; per-scenario bonus_rate_yr)
    # ------------------------------------------------------------------
    new_accrued_bonus = accrued_bonus + sd.is_par * bonus_rate_yr * sd.sum_assured / 12.0

    # ------------------------------------------------------------------
    # Asset share update: credit investment return, net premium less expenses
    # ------------------------------------------------------------------
    monthly_premium_net = (
        sd.annual_premium / 12.0
        - (sd.expense_pct_premium * sd.annual_premium + sd.expense_per_policy) / 12.0
    )
    new_asset_share = asset_share * (1.0 + earned_return_monthly) + monthly_premium_net

    return (
        new_in_force,
        new_accrued_bonus,
        new_asset_share,
        deaths,
        lapses,
        maturities,
        premiums,
        death_claims,
        surrender_payments,
        maturity_payments,
        expenses,
    )
