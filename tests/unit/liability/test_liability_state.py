"""
Tests V1–V3: LiabilityState structs and factory functions.

V1  ConventionalState from conventional_state_from_mps: shape (n_sc, n_grp),
    correct initial values (in_force from model points, reserve = 0).
V2  InPaymentState, DeferredState, DependantState factories: shape and initial
    values correct.
V3  All state fields are np.ndarray — no DataFrames, no Python lists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.liability.liability_state import (
    ConventionalState,
    DeferredState,
    DependantState,
    InPaymentState,
    conventional_state_from_mps,
    deferred_state_from_mps,
    dependant_state_from_mps,
    in_payment_state_from_mps,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SCENARIOS = 5
N_GROUPS    = 3


@pytest.fixture()
def conv_mps() -> pd.DataFrame:
    """Minimal conventional model points with 3 groups."""
    return pd.DataFrame({
        "group_id":                ["g1", "g2", "g3"],
        "in_force_count":          [100.0, 50.0, 25.0],
        "sum_assured":             [10_000.0, 20_000.0, 15_000.0],
        "annual_premium":          [500.0, 800.0, 600.0],
        "attained_age":            [40, 50, 60],
        "policy_code":             ["ENDOW_NONPAR", "ENDOW_PAR", "TERM"],
        "policy_term_yr":          [20, 15, 10],
        "policy_duration_mths":    [24, 60, 0],
        "accrued_bonus_per_policy": [0.0, 300.0, 0.0],
    })


@pytest.fixture()
def ip_mps() -> pd.DataFrame:
    """Minimal BPA in-payment model points with 3 groups."""
    return pd.DataFrame({
        "mp_id":          ["p1", "p2", "p3"],
        "sex":            ["M", "F", "M"],
        "age":            [65, 70, 75],
        "in_force_count": [80.0, 40.0, 20.0],
        "pension_pa":     [12_000.0, 8_000.0, 5_000.0],
        "lpi_cap":   [0.05, 0.05, 0.05],
        "lpi_floor": [0.0, 0.0, 0.0],
        "gmp_pa":    [0.0, 0.0, 0.0],
    })


@pytest.fixture()
def def_mps() -> pd.DataFrame:
    """Minimal BPA deferred model points with 2 groups."""
    return pd.DataFrame({
        "mp_id":          ["d1", "d2"],
        "sex":            ["M", "F"],
        "age":            [50, 55],
        "in_force_count": [60.0, 30.0],
        "pension_pa":  [5_000.0, 3_000.0],
        "nra":         [65, 65],
        "lpi_cap":     [0.025, 0.025],
        "lpi_floor":   [0.0, 0.0],
        "gmp_pa":      [0.0, 0.0],
        "tv_factor":   [1.0, 1.0],
        "ill_health_factor": [0.0, 0.0],
    })


@pytest.fixture()
def dep_mps() -> pd.DataFrame:
    """Minimal BPA dependant model points with 2 groups."""
    return pd.DataFrame({
        "mp_id":        ["dep1", "dep2"],
        "sex":          ["F", "M"],
        "age":          [62, 67],
        "weight":       [40.0, 20.0],
        "pension_pa":   [6_000.0, 4_000.0],
        "lpi_cap":      [0.05, 0.05],
        "lpi_floor":    [0.0, 0.0],
        "gmp_pa":       [0.0, 0.0],
        "dependant_proportion": [1.0, 1.0],
    })


# ---------------------------------------------------------------------------
# V1 — ConventionalState
# ---------------------------------------------------------------------------

class TestConventionalStateFactory:
    def test_shape_correct(self, conv_mps):
        state = conventional_state_from_mps(conv_mps, n_scenarios=N_SCENARIOS)
        assert state.in_force.shape      == (N_SCENARIOS, N_GROUPS)
        assert state.accrued_bonus.shape == (N_SCENARIOS, N_GROUPS)
        assert state.asset_share.shape   == (N_SCENARIOS, N_GROUPS)
        assert state.reserve.shape       == (N_SCENARIOS, N_GROUPS)

    def test_in_force_values_match_model_points(self, conv_mps):
        state = conventional_state_from_mps(conv_mps, n_scenarios=N_SCENARIOS)
        expected = conv_mps["in_force_count"].to_numpy()
        for i in range(N_SCENARIOS):
            np.testing.assert_array_equal(state.in_force[i], expected)

    def test_accrued_bonus_values_match_model_points(self, conv_mps):
        state = conventional_state_from_mps(conv_mps, n_scenarios=N_SCENARIOS)
        expected = conv_mps["accrued_bonus_per_policy"].to_numpy()
        for i in range(N_SCENARIOS):
            np.testing.assert_array_equal(state.accrued_bonus[i], expected)

    def test_reserve_initialised_to_zero(self, conv_mps):
        state = conventional_state_from_mps(conv_mps, n_scenarios=N_SCENARIOS)
        np.testing.assert_array_equal(state.reserve, 0.0)

    def test_asset_share_defaults_to_sum_assured(self, conv_mps):
        # No asset_share_per_policy column → falls back to sum_assured
        state = conventional_state_from_mps(conv_mps, n_scenarios=N_SCENARIOS)
        expected = conv_mps["sum_assured"].to_numpy()
        for i in range(N_SCENARIOS):
            np.testing.assert_array_equal(state.asset_share[i], expected)

    def test_asset_share_from_column_when_present(self, conv_mps):
        mps_with_as = conv_mps.copy()
        mps_with_as["asset_share_per_policy"] = [5_000.0, 8_000.0, 3_000.0]
        state = conventional_state_from_mps(mps_with_as, n_scenarios=N_SCENARIOS)
        expected = np.array([5_000.0, 8_000.0, 3_000.0])
        for i in range(N_SCENARIOS):
            np.testing.assert_array_equal(state.asset_share[i], expected)

    def test_asset_share_all_scenarios_identical_at_start(self, conv_mps):
        state = conventional_state_from_mps(conv_mps, n_scenarios=N_SCENARIOS)
        for i in range(1, N_SCENARIOS):
            np.testing.assert_array_equal(state.asset_share[0], state.asset_share[i])

    def test_all_scenario_rows_identical_at_start(self, conv_mps):
        state = conventional_state_from_mps(conv_mps, n_scenarios=N_SCENARIOS)
        for i in range(1, N_SCENARIOS):
            np.testing.assert_array_equal(state.in_force[0], state.in_force[i])
            np.testing.assert_array_equal(state.accrued_bonus[0], state.accrued_bonus[i])

    def test_is_named_tuple(self, conv_mps):
        state = conventional_state_from_mps(conv_mps, n_scenarios=1)
        assert isinstance(state, ConventionalState)

    def test_replace_produces_new_state(self, conv_mps):
        state = conventional_state_from_mps(conv_mps, n_scenarios=N_SCENARIOS)
        new_if = np.zeros_like(state.in_force)
        new_state = state._replace(in_force=new_if)
        assert new_state is not state
        np.testing.assert_array_equal(new_state.in_force, 0.0)
        # original unchanged
        assert state.in_force[0, 0] == 100.0

    def test_single_scenario(self, conv_mps):
        state = conventional_state_from_mps(conv_mps, n_scenarios=1)
        assert state.in_force.shape == (1, N_GROUPS)


# ---------------------------------------------------------------------------
# V2 — BPA state factories
# ---------------------------------------------------------------------------

class TestInPaymentStateFactory:
    def test_shape(self, ip_mps):
        state = in_payment_state_from_mps(ip_mps, n_scenarios=N_SCENARIOS)
        assert state.in_force.shape        == (N_SCENARIOS, 3)
        assert state.accrued_pension.shape == (N_SCENARIOS, 3)

    def test_in_force_from_weight(self, ip_mps):
        state = in_payment_state_from_mps(ip_mps, n_scenarios=N_SCENARIOS)
        expected = ip_mps["in_force_count"].to_numpy()
        for i in range(N_SCENARIOS):
            np.testing.assert_array_equal(state.in_force[i], expected)

    def test_accrued_pension_from_pension_pa(self, ip_mps):
        state = in_payment_state_from_mps(ip_mps, n_scenarios=N_SCENARIOS)
        expected = ip_mps["pension_pa"].to_numpy()
        for i in range(N_SCENARIOS):
            np.testing.assert_array_equal(state.accrued_pension[i], expected)

    def test_is_named_tuple(self, ip_mps):
        state = in_payment_state_from_mps(ip_mps, n_scenarios=1)
        assert isinstance(state, InPaymentState)


class TestDeferredStateFactory:
    def test_shape(self, def_mps):
        state = deferred_state_from_mps(def_mps, n_scenarios=N_SCENARIOS)
        assert state.in_force.shape == (N_SCENARIOS, 2)

    def test_in_force_from_weight(self, def_mps):
        state = deferred_state_from_mps(def_mps, n_scenarios=N_SCENARIOS)
        expected = def_mps["in_force_count"].to_numpy()
        for i in range(N_SCENARIOS):
            np.testing.assert_array_equal(state.in_force[i], expected)

    def test_is_named_tuple(self, def_mps):
        state = deferred_state_from_mps(def_mps, n_scenarios=1)
        assert isinstance(state, DeferredState)


class TestDependantStateFactory:
    def test_shape(self, dep_mps):
        state = dependant_state_from_mps(dep_mps, n_scenarios=N_SCENARIOS)
        assert state.member_in_force.shape == (N_SCENARIOS, 2)
        assert state.triggered.shape       == (N_SCENARIOS, 2)

    def test_member_in_force_from_weight(self, dep_mps):
        state = dependant_state_from_mps(dep_mps, n_scenarios=N_SCENARIOS)
        expected = dep_mps["weight"].to_numpy()
        for i in range(N_SCENARIOS):
            np.testing.assert_array_equal(state.member_in_force[i], expected)

    def test_triggered_initialised_to_zero(self, dep_mps):
        state = dependant_state_from_mps(dep_mps, n_scenarios=N_SCENARIOS)
        np.testing.assert_array_equal(state.triggered, 0.0)

    def test_is_named_tuple(self, dep_mps):
        state = dependant_state_from_mps(dep_mps, n_scenarios=1)
        assert isinstance(state, DependantState)


# ---------------------------------------------------------------------------
# V3 — All fields are np.ndarray
# ---------------------------------------------------------------------------

class TestStateFieldTypes:
    def test_conventional_fields_are_ndarray(self, conv_mps):
        state = conventional_state_from_mps(conv_mps, n_scenarios=2)
        for field_name, field_val in zip(state._fields, state):
            assert isinstance(field_val, np.ndarray), (
                f"ConventionalState.{field_name} is {type(field_val)}, expected np.ndarray"
            )

    def test_in_payment_fields_are_ndarray(self, ip_mps):
        state = in_payment_state_from_mps(ip_mps, n_scenarios=2)
        for field_name, field_val in zip(state._fields, state):
            assert isinstance(field_val, np.ndarray), (
                f"InPaymentState.{field_name} is {type(field_val)}, expected np.ndarray"
            )

    def test_deferred_fields_are_ndarray(self, def_mps):
        state = deferred_state_from_mps(def_mps, n_scenarios=2)
        for field_name, field_val in zip(state._fields, state):
            assert isinstance(field_val, np.ndarray), (
                f"DeferredState.{field_name} is {type(field_val)}, expected np.ndarray"
            )

    def test_dependant_fields_are_ndarray(self, dep_mps):
        state = dependant_state_from_mps(dep_mps, n_scenarios=2)
        for field_name, field_val in zip(state._fields, state):
            assert isinstance(field_val, np.ndarray), (
                f"DependantState.{field_name} is {type(field_val)}, expected np.ndarray"
            )

    def test_fields_are_float64(self, conv_mps):
        state = conventional_state_from_mps(conv_mps, n_scenarios=2)
        for field_name, field_val in zip(state._fields, state):
            assert field_val.dtype == np.float64, (
                f"ConventionalState.{field_name} has dtype {field_val.dtype}"
            )
