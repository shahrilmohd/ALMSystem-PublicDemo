"""
Conventional life insurance liability model.

Supports seriatim (per-policy) and group model point inputs.
The model receives a unified DataFrame — the input mode is the data loader's concern.

Policy types (policy_code)
--------------------------
ENDOW_NONPAR  Endowment, non-participating.
              Death: sum_assured
              Maturity: sum_assured
              Surrender: sv_factor × sum_assured

ENDOW_PAR     Endowment, participating (with-profits).
              Bonus accrues at bonus_rate_yr × sum_assured per year (flat rate,
              same rate assumed for all remaining years — Phase 1 simplification).
              Death: sum_assured + accrued_bonus_per_policy
              Maturity: sum_assured + accrued_bonus_per_policy
              Surrender: sv_factor × (sum_assured + accrued_bonus_per_policy)

TERM          Term assurance.
              Death: sum_assured
              Maturity: 0  (no payment to survivors at end of term)
              Surrender: 0

Any other policy_code value raises ValueError.

Naming convention
-----------------
_yr    year-based quantity  (e.g. policy_term_yr, q_x_yr, w_x_yr)
_mths  month-based quantity (e.g. policy_duration_mths, remaining_term_mths,
                              q_x_mths, w_x_mths)
No suffix: dimensionless (counts, fractions, identifiers)

Monthly rate conversion (UDD approximation)
--------------------------------------------
    q_x_mths = 1 - (1 - q_x_yr) ^ (1/12)
    w_x_mths = 1 - (1 - w_x_yr) ^ (1/12)

BEL discounting
---------------
BEL = Σ_{s=1}^{remaining_term_mths} (net_outgo_s × rate_curve.discount_factor(s))

discount_factor(s) returns the present value of £1 paid s months from now,
computed from the RiskFreeRateCurve via log-linear interpolation on discount
factors (see engine/curves/rate_curve.py).

Cash flow timing: all items at end of each monthly period.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax
# Float64 is required for actuarial-grade precision (DECISIONS.md §32).
# Must be set before any JAX operation.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd

from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.base_liability import BaseLiability, Decrements, LiabilityCashflows


# ---------------------------------------------------------------------------
# Policy code enum
# ---------------------------------------------------------------------------

class PolicyCode(str, Enum):
    """
    Identifies the insurance product type within the Conventional model.
    Each type has different maturity and surrender payment rules.
    """
    ENDOW_NONPAR = "ENDOW_NONPAR"
    ENDOW_PAR    = "ENDOW_PAR"
    TERM         = "TERM"

    @classmethod
    def valid_values(cls) -> set[str]:
        return {pc.value for pc in cls}


# ---------------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------------

@dataclass
class ConventionalAssumptions:
    """
    Actuarial assumptions for the Conventional liability model.

    All rates are ANNUAL. The model converts them to monthly internally.

    mortality_rates:
        Annual q_x keyed by integer attained age (years).
        E.g., {50: 0.005, 51: 0.006}

    lapse_rates:
        Annual w_d keyed by integer policy duration in whole years.
        E.g., {0: 0.10, 1: 0.08, 2: 0.06}
        Looked up as: lapse_rates[policy_duration_mths // 12]

    expense_pct_premium:
        Expenses as a fraction of annual premium per policy.

    expense_per_policy:
        Fixed expense per in-force policy per year (currency units).

    surrender_value_factors:
        SV paid on lapse as a fraction of base amount, keyed by integer
        duration in whole years.
        Base amount: sum_assured (NONPAR), sum_assured + accrued_bonus (PAR), 0 (TERM).

    rate_curve:
        Risk-free rate curve used for BEL discounting.
        Default: flat 5% curve (RiskFreeRateCurve.flat(0.05)).
        discount_factor(t_months) is called with the projection step (months)
        to obtain the monthly discount factor for that time step.

    bonus_rate_yr:
        Annual bonus rate for ENDOW_PAR policies, as a fraction of sum_assured.
        Flat rate applied uniformly over all remaining years (Phase 1 simplification).
        E.g., 0.03 = 3% of sum_assured per year, accruing monthly.
        Ignored for ENDOW_NONPAR and TERM.

    default_mortality_rate, default_lapse_rate, default_surrender_value_factor:
        Fallback when age or duration is absent from the respective dict.
    """
    mortality_rates:                dict[int, float]
    lapse_rates:                    dict[int, float]
    expense_pct_premium:            float
    expense_per_policy:             float
    surrender_value_factors:        dict[int, float]
    rate_curve:                     RiskFreeRateCurve = field(
                                        default_factory=lambda: RiskFreeRateCurve.flat(0.05)
                                    )
    bonus_rate_yr:                  float = 0.0
    default_mortality_rate:         float = 0.0
    default_lapse_rate:             float = 0.0
    default_surrender_value_factor: float = 0.0

    def get_mortality_rate_yr(self, age: int) -> float:
        """Annual mortality rate q_x for the given attained age (integer years)."""
        return self.mortality_rates.get(age, self.default_mortality_rate)

    def get_lapse_rate_yr(self, duration_yr: int) -> float:
        """Annual lapse rate w_d for the given policy duration in whole years."""
        return self.lapse_rates.get(duration_yr, self.default_lapse_rate)

    def get_surrender_value_factor(self, duration_yr: int) -> float:
        """Surrender value factor for the given policy duration in whole years."""
        return self.surrender_value_factors.get(
            duration_yr, self.default_surrender_value_factor
        )


# ---------------------------------------------------------------------------
# Conventional liability model
# ---------------------------------------------------------------------------

class Conventional(BaseLiability):
    """
    Conventional life insurance liability model.

    Completely stateless — no instance variables are set or mutated.
    All methods are pure functions of (model_points, assumptions, timestep).

    Required model_points columns
    ------------------------------
    group_id                 str    Identifier
    in_force_count           float  Policies in this group (≥ 0)
    sum_assured              float  Benefit per policy (£)
    annual_premium           float  Annual premium per policy (£/yr)
    attained_age             int    Current attained age (whole years)
    policy_code              str    "ENDOW_NONPAR", "ENDOW_PAR", or "TERM"
    policy_term_yr           int    Total policy term (years)
    policy_duration_mths     int    Duration in force (months); increments each step
    accrued_bonus_per_policy float  Bonus accrued to date per policy (£); 0.0 for non-PAR

    Derived (not stored as input, calculated at each step):
    remaining_term_mths = policy_term_yr × 12 − policy_duration_mths
    """

    REQUIRED_COLUMNS: frozenset[str] = frozenset({
        "group_id",
        "in_force_count",
        "sum_assured",
        "annual_premium",
        "attained_age",
        "policy_code",
        "policy_term_yr",
        "policy_duration_mths",
        "accrued_bonus_per_policy",
    })

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def _validate_model_points(self, model_points: pd.DataFrame) -> None:
        """Raise ValueError if any required column is absent or policy_code is invalid."""
        missing = self.REQUIRED_COLUMNS - set(model_points.columns)
        if missing:
            raise ValueError(
                f"model_points is missing required columns: {sorted(missing)}"
            )
        if model_points.empty:
            return
        invalid = set(model_points["policy_code"].unique()) - PolicyCode.valid_values()
        if invalid:
            raise ValueError(
                f"Unknown policy_code values: {sorted(invalid)}. "
                f"Valid values are: {sorted(PolicyCode.valid_values())}"
            )

    # -----------------------------------------------------------------------
    # Core calculation helpers
    # -----------------------------------------------------------------------

    def _apply_decrements(
        self,
        model_points: pd.DataFrame,
        assumptions: ConventionalAssumptions,
    ) -> pd.DataFrame:
        """
        Apply one monthly step of decrements to model_points.

        Does not mutate the input. Returns a new DataFrame with additional
        columns:

            remaining_term_mths          — policy_term_yr×12 − policy_duration_mths
            duration_yr                  — policy_duration_mths // 12 (for rate lookup)
            q_x_yr, q_x_mths            — annual and monthly mortality rates
            w_x_yr, w_x_mths            — annual and monthly lapse rates
            sv_factor                    — surrender value factor
            deaths, lapses, maturities   — decrement counts
            in_force_end                 — in-force count after all decrements

        Monthly rate conversion (UDD):
            q_x_mths = 1 − (1 − q_x_yr) ^ (1/12)
            w_x_mths = 1 − (1 − w_x_yr) ^ (1/12)

        Decrement ordering:
            1. Deaths    from in_force_count
            2. Lapses    from survivors after deaths
            3. Maturities/Expirations from survivors after lapses (final month only)
        """
        mp = model_points.copy()

        # Derived columns
        mp["remaining_term_mths"] = (
            mp["policy_term_yr"] * 12 - mp["policy_duration_mths"]
        ).clip(lower=0)
        mp["duration_yr"] = mp["policy_duration_mths"] // 12

        # Rate lookups (annual)
        mp["q_x_yr"] = mp["attained_age"].map(assumptions.get_mortality_rate_yr)
        mp["w_x_yr"] = mp["duration_yr"].map(assumptions.get_lapse_rate_yr)
        mp["sv_factor"] = mp["duration_yr"].map(assumptions.get_surrender_value_factor)

        # Clamp to [0, 1] — defensive against bad assumption values
        mp["q_x_yr"]   = mp["q_x_yr"].clip(0.0, 1.0)
        mp["w_x_yr"]   = mp["w_x_yr"].clip(0.0, 1.0)
        mp["sv_factor"] = mp["sv_factor"].clip(0.0, 1.0)

        # Convert annual → monthly (UDD approximation)
        mp["q_x_mths"] = 1.0 - (1.0 - mp["q_x_yr"]) ** (1.0 / 12.0)
        mp["w_x_mths"] = 1.0 - (1.0 - mp["w_x_yr"]) ** (1.0 / 12.0)

        # Decrement cascade
        mp["deaths"]           = mp["in_force_count"] * mp["q_x_mths"]
        survivors_after_deaths = mp["in_force_count"] - mp["deaths"]
        mp["lapses"]           = survivors_after_deaths * mp["w_x_mths"]
        survivors_after_lapses = survivors_after_deaths - mp["lapses"]

        # Maturities / term expirations: all remaining survivors in the final month
        is_final_month    = (mp["remaining_term_mths"] <= 1).astype(float)
        mp["maturities"]  = survivors_after_lapses * is_final_month
        mp["in_force_end"] = survivors_after_lapses - mp["maturities"]

        return mp

    def _advance_model_points(
        self,
        model_points: pd.DataFrame,
        assumptions: ConventionalAssumptions,
    ) -> pd.DataFrame:
        """
        Return a new DataFrame representing the in-force block after one monthly step.

        Updates per row:
            in_force_count           ← in_force_end (from _apply_decrements)
            policy_duration_mths     ← policy_duration_mths + 1
            attained_age             ← attained_age + 1 only on policy anniversary
                                        (when policy_duration_mths % 12 == 0 after increment)
            accrued_bonus_per_policy ← += bonus_rate_yr × sum_assured / 12  (PAR only)
        """
        mp     = self._apply_decrements(model_points, assumptions)
        result = model_points.copy()

        result["in_force_count"]       = mp["in_force_end"]
        result["policy_duration_mths"] = result["policy_duration_mths"] + 1

        # Age update: increment on policy anniversary (every 12 months)
        new_duration_mths = result["policy_duration_mths"]
        result["attained_age"] = result["attained_age"] + (
            (new_duration_mths % 12 == 0).astype(int)
        )

        # PAR bonus accrual: bonus_rate_yr × sum_assured / 12 per month
        par_mask = result["policy_code"] == PolicyCode.ENDOW_PAR.value
        result.loc[par_mask, "accrued_bonus_per_policy"] = (
            result.loc[par_mask, "accrued_bonus_per_policy"]
            + assumptions.bonus_rate_yr * result.loc[par_mask, "sum_assured"] / 12.0
        )

        return result

    # -----------------------------------------------------------------------
    # Public interface — BaseLiability implementation
    # -----------------------------------------------------------------------

    def project_cashflows(
        self,
        model_points: pd.DataFrame,
        assumptions: ConventionalAssumptions,
        timestep: int,
    ) -> LiabilityCashflows:
        """
        Project cash flows for a single monthly time step.

        All amounts are totals across all model points.
        Cash flow timing: end of each monthly period.

        Policy-code-specific rules:
            ENDOW_NONPAR: death = SA, maturity = SA, surrender = sv_factor × SA
            ENDOW_PAR:    death = SA + bonus, maturity = SA + bonus,
                          surrender = sv_factor × (SA + bonus)
            TERM:         death = SA, maturity = 0, surrender = 0
        """
        self._validate_model_points(model_points)

        zero = LiabilityCashflows(
            timestep=timestep,
            premiums=0.0, death_claims=0.0,
            surrender_payments=0.0, maturity_payments=0.0, expenses=0.0,
        )

        if model_points.empty or model_points["in_force_count"].sum() <= 0:
            return zero

        mp = self._apply_decrements(model_points, assumptions)

        # --- Premiums (annual premium billed monthly) ---
        premiums = float(
            (mp["annual_premium"] / 12.0 * mp["in_force_count"]).sum()
        )

        # --- Benefit base: effective sum assured including bonus for PAR ---
        benefit_base = mp["sum_assured"].copy()
        par_mask = mp["policy_code"] == PolicyCode.ENDOW_PAR.value
        benefit_base[par_mask] = (
            mp.loc[par_mask, "sum_assured"]
            + mp.loc[par_mask, "accrued_bonus_per_policy"]
        )

        # --- Death claims: all types pay on death ---
        death_claims = float((benefit_base * mp["deaths"]).sum())

        # --- Surrender payments: TERM pays nothing ---
        term_mask   = mp["policy_code"] == PolicyCode.TERM.value
        sv_base     = benefit_base.copy()
        sv_base[term_mask] = 0.0
        surrender_payments = float(
            (mp["sv_factor"] * sv_base * mp["lapses"]).sum()
        )

        # --- Maturity payments: TERM pays nothing ---
        maturity_base = benefit_base.copy()
        maturity_base[term_mask] = 0.0
        maturity_payments = float(
            (maturity_base * mp["maturities"]).sum()
        )

        # --- Expenses (monthly = annual / 12) ---
        expenses = float(
            (
                assumptions.expense_pct_premium * mp["annual_premium"] / 12.0
                * mp["in_force_count"]
                + assumptions.expense_per_policy / 12.0 * mp["in_force_count"]
            ).sum()
        )

        return LiabilityCashflows(
            timestep=timestep,
            premiums=premiums,
            death_claims=death_claims,
            surrender_payments=surrender_payments,
            maturity_payments=maturity_payments,
            expenses=expenses,
        )

    def get_decrements(
        self,
        model_points: pd.DataFrame,
        assumptions: ConventionalAssumptions,
        timestep: int,
    ) -> Decrements:
        """Calculate aggregate decrements for this monthly time step."""
        self._validate_model_points(model_points)

        if model_points.empty:
            return Decrements(
                timestep=timestep,
                in_force_start=0.0, deaths=0.0,
                lapses=0.0, maturities=0.0, in_force_end=0.0,
            )

        mp = self._apply_decrements(model_points, assumptions)

        return Decrements(
            timestep=timestep,
            in_force_start=float(mp["in_force_count"].sum()),
            deaths=float(mp["deaths"].sum()),
            lapses=float(mp["lapses"].sum()),
            maturities=float(mp["maturities"].sum()),
            in_force_end=float(mp["in_force_end"].sum()),
        )

    def get_bel(
        self,
        model_points: pd.DataFrame,
        assumptions: ConventionalAssumptions,
        timestep: int,
    ) -> float:
        """
        Calculate BEL as the present value of future net cash outgo.

        Projects forward from the current model_points state for
        max(remaining_term_mths) monthly steps, applying decrements each step.

        Discount factors are obtained from assumptions.rate_curve for each
        projection step (in months).

        Returns 0.0 if model_points is empty or all policies are expired.
        """
        self._validate_model_points(model_points)

        if model_points.empty or model_points["in_force_count"].sum() <= 0:
            return 0.0

        remaining_term_mths = int(
            (model_points["policy_term_yr"] * 12 - model_points["policy_duration_mths"])
            .clip(lower=0)
            .max()
        )
        if remaining_term_mths <= 0:
            return 0.0

        bel        = 0.0
        current_mp = model_points.copy()

        for step in range(1, remaining_term_mths + 1):
            if current_mp["in_force_count"].sum() <= 0:
                break

            cf   = self.project_cashflows(current_mp, assumptions, timestep=step)
            bel += cf.net_outgo * assumptions.rate_curve.discount_factor(step)

            current_mp = self._advance_model_points(current_mp, assumptions)

        return bel

    def get_reserve(
        self,
        model_points: pd.DataFrame,
        assumptions: ConventionalAssumptions,
        timestep: int,
    ) -> float:
        """
        Calculate reserve at this time step.

        Phase 1: reserve = BEL (no risk margin).
        """
        return self.get_bel(model_points, assumptions, timestep)

    # -----------------------------------------------------------------------
    # Batch interface — JAX vmap override (DECISIONS.md §32, Step 4)
    # -----------------------------------------------------------------------

    def batch_step(
        self,
        states: Any,
        model_points: pd.DataFrame,
        bonus_rates: np.ndarray,
        assumptions: ConventionalAssumptions,
        timestep: int,
        terminal_bonus_rates: np.ndarray | None = None,
        earned_returns_monthly: np.ndarray | None = None,
    ) -> tuple[Any, list[LiabilityCashflows], list[Decrements]]:
        """
        Vectorised batch step using JAX ``vmap`` over the scenario dimension.

        Replaces the default loop in ``BaseLiability.batch_step`` with a
        single compiled JAX call that processes all N scenarios in parallel.
        The scalar interface (``project_cashflows``, ``get_decrements``, etc.)
        is untouched.

        Parameters
        ----------
        states : ConventionalState
            Shape ``(n_scenarios, n_groups)`` for each field.
        model_points : pd.DataFrame
            Static per-group attributes at the current projection step.
        bonus_rates : np.ndarray, shape (n_scenarios,)
            Per-scenario annual reversionary bonus rate (from SmoothedBonusStrategy
            or flat assumptions.bonus_rate_yr when no BonusStrategy is wired).
        assumptions : ConventionalAssumptions
        timestep : int
        terminal_bonus_rates : np.ndarray, shape (n_scenarios,), optional
            Per-scenario terminal bonus loading fraction.  Defaults to zeros
            when not supplied (no BonusStrategy wired).
        earned_returns_monthly : np.ndarray, shape (n_scenarios,), optional
            Monthly investment return per scenario for asset share crediting.
            Defaults to zeros when not supplied.
        timestep : int

        Returns
        -------
        new_states : ConventionalState
            Updated with new ``in_force`` and ``accrued_bonus`` for all
            scenarios.  ``reserve`` is unchanged (populated in backward pass).
        cashflows_list : list[LiabilityCashflows]
        decrements_list : list[Decrements]
        """
        from engine.liability.conventional_step import (
            make_step_data,
            conventional_step,
        )

        n_scenarios = len(bonus_rates)

        # Default optional arrays to zero when no BonusStrategy is wired
        if terminal_bonus_rates is None:
            terminal_bonus_rates = np.zeros(n_scenarios)
        if earned_returns_monthly is None:
            earned_returns_monthly = np.zeros(n_scenarios)

        # Pre-compute per-group static arrays (once for all scenarios)
        step_data = make_step_data(model_points, assumptions)

        # Convert mutable state to JAX arrays
        jax_if  = jnp.array(states.in_force)         # (n_scenarios, n_groups)
        jax_ab  = jnp.array(states.accrued_bonus)    # (n_scenarios, n_groups)
        jax_as  = jnp.array(states.asset_share)      # (n_scenarios, n_groups)
        jax_br  = jnp.array(bonus_rates)             # (n_scenarios,)
        jax_tbr = jnp.array(terminal_bonus_rates)    # (n_scenarios,)
        jax_erm = jnp.array(earned_returns_monthly)  # (n_scenarios,)

        # Build a JAX-compatible version of step_data (arrays only)
        from engine.liability.conventional_step import ConventionalStepData
        jax_sd = ConventionalStepData(
            q_monthly=jnp.array(step_data.q_monthly),
            w_monthly=jnp.array(step_data.w_monthly),
            sv_factor=jnp.array(step_data.sv_factor),
            is_final_month=jnp.array(step_data.is_final_month),
            is_par=jnp.array(step_data.is_par),
            is_term=jnp.array(step_data.is_term),
            sum_assured=jnp.array(step_data.sum_assured),
            annual_premium=jnp.array(step_data.annual_premium),
            expense_pct_premium=step_data.expense_pct_premium,
            expense_per_policy=step_data.expense_per_policy,
        )

        # vmap over (in_force, accrued_bonus, asset_share, bonus_rate_yr,
        #            terminal_bonus_rate, earned_return_monthly); broadcast step_data
        batched = jax.vmap(conventional_step, in_axes=(0, 0, 0, 0, 0, 0, None))
        results: Any = batched(jax_if, jax_ab, jax_as, jax_br, jax_tbr, jax_erm, jax_sd)

        # Unpack 11 outputs
        (
            new_if_jax,             # (n_scenarios, n_groups)
            new_ab_jax,             # (n_scenarios, n_groups)
            new_as_jax,             # (n_scenarios, n_groups)
            deaths_jax,             # (n_scenarios, n_groups)
            lapses_jax,             # (n_scenarios, n_groups)
            maturities_jax,         # (n_scenarios, n_groups)
            premiums_jax,           # (n_scenarios,)
            death_claims_jax,       # (n_scenarios,)
            surrender_payments_jax, # (n_scenarios,)
            maturity_payments_jax,  # (n_scenarios,)
            expenses_jax,           # (n_scenarios,)
        ) = results

        # Convert back to NumPy
        new_if  = np.asarray(new_if_jax)
        new_ab  = np.asarray(new_ab_jax)
        new_as  = np.asarray(new_as_jax)
        deaths_arr     = np.asarray(deaths_jax)
        lapses_arr     = np.asarray(lapses_jax)
        maturities_arr = np.asarray(maturities_jax)

        # Build per-scenario LiabilityCashflows and Decrements lists
        cashflows_list: list[LiabilityCashflows] = []
        decrements_list: list[Decrements] = []

        in_force_start = np.asarray(states.in_force)

        for i in range(n_scenarios):
            cashflows_list.append(LiabilityCashflows(
                timestep=timestep,
                premiums=float(premiums_jax[i]),
                death_claims=float(death_claims_jax[i]),
                surrender_payments=float(surrender_payments_jax[i]),
                maturity_payments=float(maturity_payments_jax[i]),
                expenses=float(expenses_jax[i]),
            ))
            decrements_list.append(Decrements(
                timestep=timestep,
                in_force_start=float(in_force_start[i].sum()),
                deaths=float(deaths_arr[i].sum()),
                lapses=float(lapses_arr[i].sum()),
                maturities=float(maturities_arr[i].sum()),
                in_force_end=float(new_if[i].sum()),
            ))

        new_states = states._replace(
            in_force=new_if,
            accrued_bonus=new_ab,
            asset_share=new_as,
        )
        return new_states, cashflows_list, decrements_list
