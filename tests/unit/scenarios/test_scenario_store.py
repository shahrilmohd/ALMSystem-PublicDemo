"""
Unit tests for engine/scenarios/scenario_store.py.

Rules under test
----------------
EsgScenario:
  1.  scenario_id and timesteps stored correctly.
  2.  get_timestep(t) returns the correct AssetScenarioPoint.
  3.  get_timestep(t) raises IndexError for out-of-range t.
  4.  n_months returns len(timesteps).

ScenarioStore construction:
  5.  Empty store has count() = 0.
  6.  Single scenario stored and retrievable.
  7.  Multiple scenarios stored; count() correct.
  8.  Duplicate scenario_id raises ValueError.

ScenarioStore retrieval:
  9.  get() returns the correct EsgScenario.
  10. get() raises KeyError for unknown scenario_id.
  11. scenario_ids() returns sorted list of all IDs.
  12. all_scenarios() returns scenarios in ascending ID order.
  13. __len__ matches count().
  14. __contains__ returns True for present ID, False for absent.
"""
from __future__ import annotations

import pytest

from engine.asset.base_asset import AssetScenarioPoint
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.scenarios.scenario_store import EsgScenario, ScenarioStore

FLAT_3PCT = RiskFreeRateCurve.flat(0.03)


def make_point(t: int = 0) -> AssetScenarioPoint:
    return AssetScenarioPoint(timestep=t, rate_curve=FLAT_3PCT, equity_total_return_yr=0.07)


def make_scenario(scenario_id: int, n_months: int = 3) -> EsgScenario:
    return EsgScenario(
        scenario_id=scenario_id,
        timesteps=[make_point(t) for t in range(n_months)],
    )


# ---------------------------------------------------------------------------
# EsgScenario
# ---------------------------------------------------------------------------

class TestEsgScenario:

    def test_fields_stored(self):
        s = make_scenario(scenario_id=42, n_months=5)
        assert s.scenario_id == 42
        assert len(s.timesteps) == 5

    def test_get_timestep_returns_correct_point(self):
        s = make_scenario(scenario_id=1, n_months=3)
        for t in range(3):
            pt = s.get_timestep(t)
            assert pt.timestep == t

    def test_get_timestep_raises_for_negative(self):
        s = make_scenario(1, n_months=3)
        with pytest.raises(IndexError):
            s.get_timestep(-1)

    def test_get_timestep_raises_for_beyond_end(self):
        s = make_scenario(1, n_months=3)
        with pytest.raises(IndexError):
            s.get_timestep(3)

    def test_n_months(self):
        s = make_scenario(1, n_months=12)
        assert s.n_months == 12


# ---------------------------------------------------------------------------
# ScenarioStore construction
# ---------------------------------------------------------------------------

class TestScenarioStoreConstruction:

    def test_empty_store(self):
        store = ScenarioStore([])
        assert store.count() == 0

    def test_single_scenario(self):
        store = ScenarioStore([make_scenario(1)])
        assert store.count() == 1

    def test_multiple_scenarios(self):
        store = ScenarioStore([make_scenario(i) for i in [1, 5, 10]])
        assert store.count() == 3

    def test_duplicate_scenario_id_raises(self):
        with pytest.raises(ValueError, match="duplicate"):
            ScenarioStore([make_scenario(1), make_scenario(1)])


# ---------------------------------------------------------------------------
# ScenarioStore retrieval
# ---------------------------------------------------------------------------

class TestScenarioStoreRetrieval:

    def _make_store(self) -> ScenarioStore:
        return ScenarioStore([make_scenario(i) for i in [3, 1, 5]])

    def test_get_returns_correct_scenario(self):
        store = self._make_store()
        assert store.get(5).scenario_id == 5

    def test_get_raises_for_unknown_id(self):
        store = self._make_store()
        with pytest.raises(KeyError):
            store.get(999)

    def test_scenario_ids_sorted(self):
        store = self._make_store()
        assert store.scenario_ids() == [1, 3, 5]

    def test_all_scenarios_ascending_order(self):
        store = self._make_store()
        ids = [s.scenario_id for s in store.all_scenarios()]
        assert ids == [1, 3, 5]

    def test_len_matches_count(self):
        store = self._make_store()
        assert len(store) == store.count()

    def test_contains_present(self):
        store = self._make_store()
        assert 3 in store

    def test_contains_absent(self):
        store = self._make_store()
        assert 99 not in store
