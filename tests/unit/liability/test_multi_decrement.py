"""
Unit tests for MultiDecrementLiability (DECISIONS.md §15, bridging step 10a).

What is under test
------------------
MultiDecrementLiability is an abstract class. Tests verify:
  1. It cannot be instantiated directly.
  2. A concrete subclass that implements only get_decrements() but omits the
     inherited BaseLiability abstract methods still raises TypeError.
  3. A fully concrete subclass (all methods implemented) can be instantiated
     and get_decrements() returns a DataFrame with the required columns.
  4. The required columns are present and values are in the valid range.
  5. The class sits in the correct position in the hierarchy — subclass of
     BaseLiability but independent of ConventionalLiability.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.liability.base_liability import BaseLiability, Decrements, LiabilityCashflows
from engine.liability.multi_decrement import MultiDecrementLiability


# ---------------------------------------------------------------------------
# Test-local concrete implementations (module-private by convention)
# ---------------------------------------------------------------------------

class _PartialConcrete(MultiDecrementLiability):
    """Implements get_decrement_rates only — omits all BaseLiability abstract
    methods. Used to verify that partial implementation is still abstract."""

    def get_decrement_rates(self, t: int, model_points: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "mp_id": [1], "q_death": [0.01], "q_retire": [0.0],
            "q_transfer": [0.0], "q_commute": [0.0],
        })
    # project_cashflows, get_bel, get_reserve, get_decrements NOT implemented


class _FullConcrete(MultiDecrementLiability):
    """Implements every abstract method — can be instantiated."""

    def get_decrement_rates(self, t: int, model_points: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "mp_id":      model_points.index.tolist(),
            "q_death":    [0.005] * len(model_points),
            "q_retire":   [0.020] * len(model_points),
            "q_transfer": [0.010] * len(model_points),
            "q_commute":  [0.000] * len(model_points),
        })

    def project_cashflows(self, model_points, assumptions, timestep):
        return LiabilityCashflows(
            timestep=timestep, premiums=0.0, death_claims=0.0,
            surrender_payments=0.0, maturity_payments=0.0, expenses=0.0,
        )

    def get_bel(self, model_points, assumptions, timestep):
        return 0.0

    def get_reserve(self, model_points, assumptions, timestep):
        return 0.0

    def get_decrements(self, model_points, assumptions, timestep):
        return Decrements(
            timestep=timestep, in_force_start=0.0,
            deaths=0.0, lapses=0.0, maturities=0.0, in_force_end=0.0,
        )


# ---------------------------------------------------------------------------
# Class hierarchy
# ---------------------------------------------------------------------------

class TestHierarchy:
    def test_is_subclass_of_base_liability(self):
        assert issubclass(MultiDecrementLiability, BaseLiability)

    def test_multi_decrement_is_abstract(self):
        """MultiDecrementLiability itself cannot be instantiated."""
        with pytest.raises(TypeError):
            MultiDecrementLiability()  # type: ignore[abstract]

    def test_partial_concrete_still_abstract(self):
        """Implementing only get_decrements is not sufficient — all
        BaseLiability abstract methods must also be provided."""
        with pytest.raises(TypeError):
            _PartialConcrete()

    def test_full_concrete_instantiates(self):
        obj = _FullConcrete()
        assert isinstance(obj, MultiDecrementLiability)
        assert isinstance(obj, BaseLiability)


# ---------------------------------------------------------------------------
# get_decrements interface contract
# ---------------------------------------------------------------------------

class TestGetDecrementsContract:

    @pytest.fixture
    def model(self):
        return _FullConcrete()

    @pytest.fixture
    def model_points(self):
        return pd.DataFrame({"policy_id": [101, 102, 103]})

    def test_returns_dataframe(self, model, model_points):
        result = model.get_decrement_rates(t=0, model_points=model_points)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, model, model_points):
        result = model.get_decrement_rates(t=0, model_points=model_points)
        required = {"mp_id", "q_death", "q_retire", "q_transfer", "q_commute"}
        assert required.issubset(set(result.columns))

    def test_row_count_matches_model_points(self, model, model_points):
        result = model.get_decrement_rates(t=0, model_points=model_points)
        assert len(result) == len(model_points)

    def test_probabilities_in_unit_interval(self, model, model_points):
        result = model.get_decrement_rates(t=0, model_points=model_points)
        for col in ["q_death", "q_retire", "q_transfer", "q_commute"]:
            assert (result[col] >= 0.0).all(), f"{col} has negative values"
            assert (result[col] <= 1.0).all(), f"{col} exceeds 1.0"

    def test_total_decrement_does_not_exceed_one(self, model, model_points):
        result = model.get_decrement_rates(t=0, model_points=model_points)
        total = (result["q_death"] + result["q_retire"]
                 + result["q_transfer"] + result["q_commute"])
        assert (total <= 1.0).all()

    def test_timestep_parameter_accepted(self, model):
        """get_decrement_rates must accept any non-negative integer timestep."""
        mp = pd.DataFrame({"policy_id": [1]})
        for t in [0, 1, 11, 120]:
            result = model.get_decrement_rates(t=t, model_points=mp)
            assert len(result) == 1

    def test_model_points_not_mutated(self, model, model_points):
        """get_decrement_rates must not modify the input DataFrame."""
        before = model_points.copy()
        model.get_decrement_rates(t=0, model_points=model_points)
        pd.testing.assert_frame_equal(model_points, before)
