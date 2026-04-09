"""
ScenarioStore — indexed collection of ESG scenarios for stochastic runs.

Design
------
An EsgScenario is a full time-series path: one AssetScenarioPoint per projection
month. Each point carries a complete RiskFreeRateCurve (the full term structure
at that month under that scenario) and an annual equity total return.

ScenarioStore holds N EsgScenarios, indexed by scenario_id (integer). Indexing
by ID rather than position allows any subset to be retrieved without loading
all preceding scenarios — supporting tail analysis (e.g. "run only scenario 650")
without running scenarios 1–649 first.

See DECISIONS.md §13 for the full rationale.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.asset.base_asset import AssetScenarioPoint


# ---------------------------------------------------------------------------
# EsgScenario
# ---------------------------------------------------------------------------

@dataclass
class EsgScenario:
    """
    One risk-neutral ESG scenario: T AssetScenarioPoints, one per month.

    Parameters
    ----------
    scenario_id : int
        Unique identifier. Does not need to be contiguous or start at 1.
    timesteps : list[AssetScenarioPoint]
        Time-ordered sequence of scenario points. Index 0 = valuation month.
        len(timesteps) must equal the projection term in months.
    """
    scenario_id: int
    timesteps:   list[AssetScenarioPoint]

    def get_timestep(self, t: int) -> AssetScenarioPoint:
        """Return the AssetScenarioPoint for month t. Raises IndexError if out of range."""
        if t < 0 or t >= len(self.timesteps):
            raise IndexError(
                f"EsgScenario {self.scenario_id}: timestep {t} out of range "
                f"[0, {len(self.timesteps) - 1}]."
            )
        return self.timesteps[t]

    @property
    def n_months(self) -> int:
        """Number of monthly timesteps in this scenario."""
        return len(self.timesteps)


# ---------------------------------------------------------------------------
# ScenarioStore
# ---------------------------------------------------------------------------

class ScenarioStore:
    """
    Indexed collection of EsgScenarios.

    Scenarios are keyed by scenario_id. Retrieval by ID is O(1) regardless
    of how many scenarios are stored.

    Parameters
    ----------
    scenarios : list[EsgScenario]
        The scenarios to store. Duplicate scenario_ids raise ValueError.
    """

    def __init__(self, scenarios: list[EsgScenario]) -> None:
        self._store: dict[int, EsgScenario] = {}
        for s in scenarios:
            if s.scenario_id in self._store:
                raise ValueError(
                    f"ScenarioStore: duplicate scenario_id {s.scenario_id}."
                )
            self._store[s.scenario_id] = s

    # -----------------------------------------------------------------------
    # Retrieval
    # -----------------------------------------------------------------------

    def get(self, scenario_id: int) -> EsgScenario:
        """
        Return the EsgScenario for the given scenario_id.

        Raises
        ------
        KeyError
            If scenario_id is not in the store.
        """
        if scenario_id not in self._store:
            raise KeyError(
                f"ScenarioStore: scenario_id {scenario_id} not found. "
                f"Available IDs: {sorted(self._store)[:10]}{'...' if len(self._store) > 10 else ''}."
            )
        return self._store[scenario_id]

    def scenario_ids(self) -> list[int]:
        """Sorted list of all scenario IDs in the store."""
        return sorted(self._store)

    def all_scenarios(self) -> list[EsgScenario]:
        """All scenarios in ascending scenario_id order."""
        return [self._store[sid] for sid in self.scenario_ids()]

    def count(self) -> int:
        """Number of scenarios in the store."""
        return len(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, scenario_id: int) -> bool:
        return scenario_id in self._store
