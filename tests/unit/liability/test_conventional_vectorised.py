"""
Tests V4–V7: conventional_step pure function and batch_step interface.

V4  conventional_step produces identical cashflows to
    Conventional.project_cashflows for a single-group scalar input.
V5  BaseLiability.batch_step (default loop) with batch=1 matches the
    scalar project_cashflows output.
V6  Conventional.batch_step (JAX vmap) with batch=N matches the default
    loop output for batch=N.
V7  All existing Conventional scalar tests still pass unchanged — the
    batch interface is additive and does not modify the scalar path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.base_liability import Decrements, LiabilityCashflows
from engine.liability.conventional import Conventional, ConventionalAssumptions
from engine.liability.conventional_step import (
    ConventionalStepData,
    conventional_step,
    make_step_data,
)
from engine.liability.liability_state import conventional_state_from_mps


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def assumptions():
    return ConventionalAssumptions(
        mortality_rates={40: 0.002, 50: 0.005, 60: 0.012},
        lapse_rates={0: 0.10, 1: 0.08, 2: 0.05},
        expense_pct_premium=0.02,
        expense_per_policy=50.0,
        surrender_value_factors={0: 0.80, 1: 0.85, 2: 0.90},
        rate_curve=RiskFreeRateCurve.flat(0.05),
        bonus_rate_yr=0.03,
    )


def _make_single_group_mp(
    policy_code: str = "ENDOW_NONPAR",
    in_force: float = 100.0,
    accrued_bonus: float = 0.0,
    attained_age: int = 40,
    policy_duration_mths: int = 12,
    policy_term_yr: int = 10,
) -> pd.DataFrame:
    return pd.DataFrame({
        "group_id":                [f"g_{policy_code}"],
        "in_force_count":          [in_force],
        "sum_assured":             [10_000.0],
        "annual_premium":          [600.0],
        "attained_age":            [attained_age],
        "policy_code":             [policy_code],
        "policy_term_yr":          [policy_term_yr],
        "policy_duration_mths":    [policy_duration_mths],
        "accrued_bonus_per_policy": [accrued_bonus],
    })


def _make_multigroup_mp() -> pd.DataFrame:
    """Three groups covering all three policy codes."""
    return pd.DataFrame({
        "group_id":                ["g_nonpar", "g_par", "g_term"],
        "in_force_count":          [100.0, 80.0, 60.0],
        "sum_assured":             [10_000.0, 15_000.0, 5_000.0],
        "annual_premium":          [600.0, 900.0, 300.0],
        "attained_age":            [40, 50, 60],
        "policy_code":             ["ENDOW_NONPAR", "ENDOW_PAR", "TERM"],
        "policy_term_yr":          [20, 15, 10],
        "policy_duration_mths":    [12, 24, 0],
        "accrued_bonus_per_policy": [0.0, 500.0, 0.0],
    })


# ---------------------------------------------------------------------------
# V4 — Pure step function equals OOP path for scalar input
# ---------------------------------------------------------------------------

class TestConventionalStepEquivalence:
    """V4: conventional_step == Conventional.project_cashflows for one group."""

    @pytest.mark.parametrize("policy_code", ["ENDOW_NONPAR", "ENDOW_PAR", "TERM"])
    def test_cashflows_match_oop(self, policy_code, assumptions):
        accrued = 500.0 if policy_code == "ENDOW_PAR" else 0.0
        mp = _make_single_group_mp(policy_code=policy_code, accrued_bonus=accrued)

        # OOP path
        oop_cfs = Conventional().project_cashflows(mp, assumptions, timestep=0)

        # Pure step path (terminal_bonus_rate=0 → identical to old behaviour)
        sd = make_step_data(mp, assumptions)
        (_, _, _,  # new_if, new_ab, new_as
         _, _, _,  # deaths, lapses, maturities
         premiums, death_claims, surrender_payments, maturity_payments, expenses,
        ) = conventional_step(
            in_force=mp["in_force_count"].to_numpy(),
            accrued_bonus=mp["accrued_bonus_per_policy"].to_numpy(),
            asset_share=mp["sum_assured"].to_numpy(),
            bonus_rate_yr=assumptions.bonus_rate_yr,
            terminal_bonus_rate=0.0,
            earned_return_monthly=0.0,
            step_data=sd,
        )

        assert abs(premiums          - oop_cfs.premiums)          < 1e-9
        assert abs(death_claims      - oop_cfs.death_claims)      < 1e-9
        assert abs(surrender_payments - oop_cfs.surrender_payments) < 1e-9
        assert abs(maturity_payments  - oop_cfs.maturity_payments)  < 1e-9
        assert abs(expenses          - oop_cfs.expenses)           < 1e-9

    def test_decrement_totals_match_oop(self, assumptions):
        mp = _make_single_group_mp()
        oop_dec = Conventional().get_decrements(mp, assumptions, timestep=0)

        sd = make_step_data(mp, assumptions)
        (new_if, _, _,  # new_if, new_ab, new_as
         deaths, lapses, maturities, *_) = conventional_step(
            in_force=mp["in_force_count"].to_numpy(),
            accrued_bonus=mp["accrued_bonus_per_policy"].to_numpy(),
            asset_share=mp["sum_assured"].to_numpy(),
            bonus_rate_yr=assumptions.bonus_rate_yr,
            terminal_bonus_rate=0.0,
            earned_return_monthly=0.0,
            step_data=sd,
        )

        assert abs(deaths.sum()     - oop_dec.deaths)      < 1e-9
        assert abs(lapses.sum()     - oop_dec.lapses)       < 1e-9
        assert abs(maturities.sum() - oop_dec.maturities)   < 1e-9
        assert abs(new_if.sum()     - oop_dec.in_force_end) < 1e-9

    def test_final_month_maturities(self, assumptions):
        """Final month: all survivors become maturities, new_in_force = 0."""
        mp = _make_single_group_mp(
            policy_code="ENDOW_NONPAR",
            policy_term_yr=1,
            policy_duration_mths=11,  # remaining = 1 month → final month
        )
        sd = make_step_data(mp, assumptions)
        (new_if, _, _,  # new_if, new_ab, new_as
         _, _, maturities, *_) = conventional_step(
            in_force=mp["in_force_count"].to_numpy(),
            accrued_bonus=mp["accrued_bonus_per_policy"].to_numpy(),
            asset_share=mp["sum_assured"].to_numpy(),
            bonus_rate_yr=assumptions.bonus_rate_yr,
            terminal_bonus_rate=0.0,
            earned_return_monthly=0.0,
            step_data=sd,
        )
        assert new_if.sum() < 1.0           # effectively zero after maturities
        assert maturities.sum() > 0.0

    def test_par_bonus_accrual(self, assumptions):
        """PAR policy: new_accrued_bonus = old + bonus_rate_yr × SA / 12."""
        mp = _make_single_group_mp(
            policy_code="ENDOW_PAR",
            in_force=100.0,
            accrued_bonus=300.0,
        )
        sd = make_step_data(mp, assumptions)
        (_, new_ab, *_) = conventional_step(
            in_force=mp["in_force_count"].to_numpy(),
            accrued_bonus=mp["accrued_bonus_per_policy"].to_numpy(),
            asset_share=mp["sum_assured"].to_numpy(),
            bonus_rate_yr=assumptions.bonus_rate_yr,
            terminal_bonus_rate=0.0,
            earned_return_monthly=0.0,
            step_data=sd,
        )
        expected_bonus = 300.0 + assumptions.bonus_rate_yr * 10_000.0 / 12.0
        assert abs(new_ab[0] - expected_bonus) < 1e-9

    def test_term_no_sv_no_maturity(self, assumptions):
        mp = _make_single_group_mp(policy_code="TERM")
        sd = make_step_data(mp, assumptions)
        (_, _, _,   # new_if, new_ab, new_as
         _, _, _,   # deaths, lapses, maturities
         _, _, surrender_payments, maturity_payments, _) = conventional_step(
            in_force=mp["in_force_count"].to_numpy(),
            accrued_bonus=mp["accrued_bonus_per_policy"].to_numpy(),
            asset_share=mp["sum_assured"].to_numpy(),
            bonus_rate_yr=assumptions.bonus_rate_yr,
            terminal_bonus_rate=0.0,
            earned_return_monthly=0.0,
            step_data=sd,
        )
        assert surrender_payments == 0.0
        assert maturity_payments  == 0.0

    # ------------------------------------------------------------------
    # Step 27 — asset share and terminal bonus regression / new behaviour
    # ------------------------------------------------------------------

    def test_zero_earned_return_zero_terminal_bonus_identical_cashflows(self, assumptions):
        """earned_return=0, terminal_bonus=0 → cashflows unchanged vs old interface."""
        mp = _make_single_group_mp(policy_code="ENDOW_PAR", accrued_bonus=200.0)
        sd = make_step_data(mp, assumptions)
        (_, _, _,
         _, _, _,
         premiums, dc, sv, mat, exp) = conventional_step(
            in_force=mp["in_force_count"].to_numpy(),
            accrued_bonus=mp["accrued_bonus_per_policy"].to_numpy(),
            asset_share=mp["sum_assured"].to_numpy(),
            bonus_rate_yr=assumptions.bonus_rate_yr,
            terminal_bonus_rate=0.0,
            earned_return_monthly=0.0,
            step_data=sd,
        )
        oop_cfs = Conventional().project_cashflows(mp, assumptions, timestep=0)
        assert abs(premiums - oop_cfs.premiums) < 1e-9
        assert abs(dc  - oop_cfs.death_claims) < 1e-9
        assert abs(sv  - oop_cfs.surrender_payments) < 1e-9
        assert abs(mat - oop_cfs.maturity_payments) < 1e-9

    def test_positive_earned_return_grows_asset_share(self, assumptions):
        mp = _make_single_group_mp()
        sd = make_step_data(mp, assumptions)
        initial_as = mp["sum_assured"].to_numpy()
        (_, _, new_as, *_) = conventional_step(
            in_force=mp["in_force_count"].to_numpy(),
            accrued_bonus=mp["accrued_bonus_per_policy"].to_numpy(),
            asset_share=initial_as,
            bonus_rate_yr=0.0,
            terminal_bonus_rate=0.0,
            earned_return_monthly=0.005,
            step_data=sd,
        )
        assert new_as[0] > initial_as[0]

    def test_terminal_bonus_increases_death_claims_for_par(self, assumptions):
        mp = _make_single_group_mp(policy_code="ENDOW_PAR", accrued_bonus=200.0)
        sd = make_step_data(mp, assumptions)
        common_args = dict(
            in_force=mp["in_force_count"].to_numpy(),
            accrued_bonus=mp["accrued_bonus_per_policy"].to_numpy(),
            asset_share=mp["sum_assured"].to_numpy(),
            bonus_rate_yr=assumptions.bonus_rate_yr,
            earned_return_monthly=0.0,
            step_data=sd,
        )
        (_, _, _, _, _, _, _, dc_no_tb, *_) = conventional_step(
            terminal_bonus_rate=0.0, **common_args
        )
        (_, _, _, _, _, _, _, dc_with_tb, *_) = conventional_step(
            terminal_bonus_rate=0.1, **common_args
        )
        assert dc_with_tb > dc_no_tb

    def test_terminal_bonus_does_not_affect_surrenders(self, assumptions):
        mp = _make_single_group_mp(policy_code="ENDOW_PAR", accrued_bonus=200.0)
        sd = make_step_data(mp, assumptions)
        common_args = dict(
            in_force=mp["in_force_count"].to_numpy(),
            accrued_bonus=mp["accrued_bonus_per_policy"].to_numpy(),
            asset_share=mp["sum_assured"].to_numpy(),
            bonus_rate_yr=assumptions.bonus_rate_yr,
            earned_return_monthly=0.0,
            step_data=sd,
        )
        (_, _, _, _, _, _, _, _, sv_no_tb, *_) = conventional_step(
            terminal_bonus_rate=0.0, **common_args
        )
        (_, _, _, _, _, _, _, _, sv_with_tb, *_) = conventional_step(
            terminal_bonus_rate=0.1, **common_args
        )
        assert abs(sv_with_tb - sv_no_tb) < 1e-12


# ---------------------------------------------------------------------------
# V5 — BaseLiability.batch_step default loop matches scalar path (batch=1)
# ---------------------------------------------------------------------------

class TestBatchStepDefaultLoop:
    """V5: default batch_step with 1 scenario == project_cashflows."""

    def test_cashflows_match_scalar(self, assumptions):
        mp    = _make_multigroup_mp()
        model = Conventional()

        # Scalar path
        oop_cfs = model.project_cashflows(mp, assumptions, timestep=0)

        # Default batch_step with 1 scenario (uses BaseLiability loop)
        states     = conventional_state_from_mps(mp, n_scenarios=1)
        bonus_rates = np.array([assumptions.bonus_rate_yr])

        # Explicitly call the BASE class method to test the default loop
        from engine.liability.base_liability import BaseLiability
        _, cfs_list, _ = BaseLiability.batch_step(
            model, states, mp, bonus_rates, assumptions, timestep=0
        )

        assert len(cfs_list) == 1
        cf = cfs_list[0]
        assert abs(cf.premiums           - oop_cfs.premiums)          < 1e-9
        assert abs(cf.death_claims       - oop_cfs.death_claims)      < 1e-9
        assert abs(cf.surrender_payments - oop_cfs.surrender_payments) < 1e-9
        assert abs(cf.maturity_payments  - oop_cfs.maturity_payments)  < 1e-9
        assert abs(cf.expenses           - oop_cfs.expenses)           < 1e-9

    def test_decrements_match_scalar(self, assumptions):
        mp    = _make_multigroup_mp()
        model = Conventional()
        oop_dec = model.get_decrements(mp, assumptions, timestep=0)

        states     = conventional_state_from_mps(mp, n_scenarios=1)
        bonus_rates = np.array([assumptions.bonus_rate_yr])
        # Explicitly call the BASE class method to test the default loop
        from engine.liability.base_liability import BaseLiability
        _, _, dec_list = BaseLiability.batch_step(
            model, states, mp, bonus_rates, assumptions, timestep=0
        )

        dec = dec_list[0]
        assert abs(dec.deaths      - oop_dec.deaths)      < 1e-9
        assert abs(dec.lapses      - oop_dec.lapses)       < 1e-9
        assert abs(dec.maturities  - oop_dec.maturities)   < 1e-9
        assert abs(dec.in_force_end - oop_dec.in_force_end) < 1e-9

    def test_returns_three_items(self, assumptions):
        mp          = _make_multigroup_mp()
        states      = conventional_state_from_mps(mp, n_scenarios=2)
        bonus_rates = np.full(2, assumptions.bonus_rate_yr)
        result      = Conventional().batch_step(states, mp, bonus_rates, assumptions, 0)
        assert len(result) == 3   # (new_states, cashflows_list, decrements_list)

    def test_cashflows_list_length_equals_n_scenarios(self, assumptions):
        mp          = _make_multigroup_mp()
        n           = 4
        states      = conventional_state_from_mps(mp, n_scenarios=n)
        bonus_rates = np.full(n, assumptions.bonus_rate_yr)
        _, cfs, dec = Conventional().batch_step(states, mp, bonus_rates, assumptions, 0)
        assert len(cfs) == n
        assert len(dec) == n


# ---------------------------------------------------------------------------
# V6 — JAX batch_step matches default loop for batch=N
# ---------------------------------------------------------------------------

class TestConventionalJAXBatchStep:
    """V6: JAX vmap batch_step matches loop batch_step for N>1."""

    N = 8

    def _run_loop(self, mp, assumptions, n):
        """Python loop over conventional_step — the float64 reference path.

        V6 tests that JAX vmap gives the same results as a plain Python loop
        over the pure step function.  We do NOT use BaseLiability.batch_step
        here because its default in_force update uses a proportional
        approximation (different algorithm, not just precision).
        """
        states      = conventional_state_from_mps(mp, n_scenarios=n)
        bonus_rates = np.full(n, assumptions.bonus_rate_yr)
        sd          = make_step_data(mp, assumptions)

        new_if_arr = np.empty_like(states.in_force)
        new_ab_arr = np.empty_like(states.accrued_bonus)
        cashflows_list: list[LiabilityCashflows] = []
        decrements_list: list[Decrements] = []

        for i in range(n):
            (new_if_i, new_ab_i, new_as_i,
             deaths_i, lapses_i, maturities_i,
             premiums_i, dc_i, sv_i, mat_i, exp_i) = conventional_step(
                in_force=states.in_force[i],
                accrued_bonus=states.accrued_bonus[i],
                asset_share=states.asset_share[i],
                bonus_rate_yr=float(bonus_rates[i]),
                terminal_bonus_rate=0.0,
                earned_return_monthly=0.0,
                step_data=sd,
            )
            new_if_arr[i] = new_if_i
            new_ab_arr[i] = new_ab_i
            cashflows_list.append(LiabilityCashflows(
                timestep=0,
                premiums=float(premiums_i),
                death_claims=float(dc_i),
                surrender_payments=float(sv_i),
                maturity_payments=float(mat_i),
                expenses=float(exp_i),
            ))
            decrements_list.append(Decrements(
                timestep=0,
                in_force_start=float(states.in_force[i].sum()),
                deaths=float(deaths_i.sum()),
                lapses=float(lapses_i.sum()),
                maturities=float(maturities_i.sum()),
                in_force_end=float(new_if_i.sum()),
            ))

        new_states = states._replace(in_force=new_if_arr, accrued_bonus=new_ab_arr)
        return new_states, cashflows_list, decrements_list

    def _run_jax(self, mp, assumptions, n):
        """Call Conventional.batch_step (JAX vmap override)."""
        states      = conventional_state_from_mps(mp, n_scenarios=n)
        bonus_rates = np.full(n, assumptions.bonus_rate_yr)
        return Conventional().batch_step(states, mp, bonus_rates, assumptions, timestep=0)

    def test_premiums_match(self, assumptions):
        mp = _make_multigroup_mp()
        _, loop_cfs, _ = self._run_loop(mp, assumptions, self.N)
        _, jax_cfs, _  = self._run_jax(mp, assumptions, self.N)
        for lc, jc in zip(loop_cfs, jax_cfs):
            assert abs(lc.premiums - jc.premiums) < 1e-6

    def test_death_claims_match(self, assumptions):
        mp = _make_multigroup_mp()
        _, loop_cfs, _ = self._run_loop(mp, assumptions, self.N)
        _, jax_cfs, _  = self._run_jax(mp, assumptions, self.N)
        for lc, jc in zip(loop_cfs, jax_cfs):
            assert abs(lc.death_claims - jc.death_claims) < 1e-6

    def test_surrender_payments_match(self, assumptions):
        mp = _make_multigroup_mp()
        _, loop_cfs, _ = self._run_loop(mp, assumptions, self.N)
        _, jax_cfs, _  = self._run_jax(mp, assumptions, self.N)
        for lc, jc in zip(loop_cfs, jax_cfs):
            assert abs(lc.surrender_payments - jc.surrender_payments) < 1e-6

    def test_expenses_match(self, assumptions):
        mp = _make_multigroup_mp()
        _, loop_cfs, _ = self._run_loop(mp, assumptions, self.N)
        _, jax_cfs, _  = self._run_jax(mp, assumptions, self.N)
        for lc, jc in zip(loop_cfs, jax_cfs):
            assert abs(lc.expenses - jc.expenses) < 1e-6

    def test_decrements_match(self, assumptions):
        mp = _make_multigroup_mp()
        _, _, loop_dec = self._run_loop(mp, assumptions, self.N)
        _, _, jax_dec  = self._run_jax(mp, assumptions, self.N)
        for ld, jd in zip(loop_dec, jax_dec):
            assert abs(ld.deaths     - jd.deaths)     < 1e-6
            assert abs(ld.lapses     - jd.lapses)      < 1e-6
            assert abs(ld.maturities - jd.maturities)  < 1e-6

    def test_new_in_force_match(self, assumptions):
        mp = _make_multigroup_mp()
        loop_states, _, _ = self._run_loop(mp, assumptions, self.N)
        jax_states, _, _  = self._run_jax(mp, assumptions, self.N)
        # Both paths call conventional_step with float64 (JAX x64 enabled);
        # results should agree to near machine epsilon.
        np.testing.assert_allclose(
            np.asarray(jax_states.in_force), loop_states.in_force, rtol=1e-10
        )

    def test_diverging_bonus_rates_handled(self, assumptions):
        """Scenarios with different bonus rates produce different accrued bonuses."""
        mp = _make_single_group_mp(policy_code="ENDOW_PAR", accrued_bonus=200.0)
        n  = 4
        states      = conventional_state_from_mps(mp, n_scenarios=n)
        bonus_rates = np.array([0.01, 0.03, 0.05, 0.07])  # different per scenario
        _, _, _ = Conventional().batch_step(states, mp, bonus_rates, assumptions, 0)
        new_states, _, _ = Conventional().batch_step(states, mp, bonus_rates, assumptions, 0)
        # Accrued bonuses should differ across scenarios
        assert new_states.accrued_bonus[0, 0] < new_states.accrued_bonus[3, 0]


# ---------------------------------------------------------------------------
# V7 — Existing scalar interface is unchanged (guard against regression)
# ---------------------------------------------------------------------------

class TestScalarInterfaceUnchanged:
    """V7: all existing scalar methods still work after adding batch_step."""

    def test_project_cashflows_works(self, assumptions):
        mp  = _make_multigroup_mp()
        cfs = Conventional().project_cashflows(mp, assumptions, timestep=0)
        assert cfs.premiums > 0.0

    def test_get_bel_works(self, assumptions):
        mp  = _make_single_group_mp()
        bel = Conventional().get_bel(mp, assumptions, timestep=0)
        assert bel > 0.0

    def test_get_decrements_works(self, assumptions):
        mp  = _make_multigroup_mp()
        dec = Conventional().get_decrements(mp, assumptions, timestep=0)
        assert dec.in_force_start > 0.0
        assert dec.deaths > 0.0

    def test_get_reserve_equals_bel(self, assumptions):
        mp  = _make_single_group_mp()
        bel = Conventional().get_bel(mp, assumptions, timestep=0)
        res = Conventional().get_reserve(mp, assumptions, timestep=0)
        assert abs(bel - res) < 1e-9
