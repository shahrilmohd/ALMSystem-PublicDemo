"""
ScenarioLoader — builds a ScenarioStore from a CSV file or analytically.

CSV format (DECISIONS.md §13, §16)
------------------------------------
Required columns:
    scenario_id       int   Unique scenario identifier
    timestep          int   Month index (0 = valuation month)
    r_{n}m            float Spot rate at maturity n months
                            Column names encode the maturity:
                            r_1m = 1-month, r_12m = 12-month, r_360m = 30-year.
                            At least one rate column required.
    equity_return_yr  float Annual total equity return (decimal, e.g. 0.07 = 7%)

Optional columns (DECISIONS.md §16 — BPA inflation paths):
    cpi_annual_rate   float Annual CPI rate for this timestep (decimal).
                            If absent, AssetScenarioPoint.cpi_annual_rate = None.
    rpi_annual_rate   float Annual RPI rate for this timestep (decimal).
                            If absent, AssetScenarioPoint.rpi_annual_rate = None.
    Existing ESG files without these columns load without error or modification.

Example header (conventional):
    scenario_id,timestep,r_1m,r_3m,r_6m,r_12m,r_24m,r_60m,r_120m,r_240m,r_360m,equity_return_yr

Example header (BPA with inflation):
    scenario_id,timestep,r_1m,...,r_360m,equity_return_yr,cpi_annual_rate,rpi_annual_rate

Row ordering: rows within a scenario need not be sorted by timestep — the
loader sorts them. Scenarios need not appear in ID order.

Validation performed at load time:
    - All required columns present
    - At least one r_{n}m column present
    - No duplicate (scenario_id, timestep) pairs
    - scenario_ids filter applied after load (unknown IDs silently ignored)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from engine.asset.base_asset import AssetScenarioPoint
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.scenarios.scenario_store import EsgScenario, ScenarioStore


_RATE_COL_RE = re.compile(r"^r_(\d+)m$")
_REQUIRED_BASE_COLS = {"scenario_id", "timestep", "equity_return_yr"}


class ScenarioLoader:
    """
    Builds ScenarioStore objects from CSV files or analytically.

    All methods are class methods — ScenarioLoader is never instantiated.
    """

    # -----------------------------------------------------------------------
    # from_csv — primary production loader
    # -----------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        scenario_ids: Optional[list[int]] = None,
    ) -> ScenarioStore:
        """
        Load ESG scenarios from a CSV file.

        Parameters
        ----------
        path : str | Path
            Path to the ESG CSV file.
        scenario_ids : list[int] | None
            If None, load all scenarios in the file.
            If a list, load only the scenarios whose scenario_id appears in the
            list. IDs in the list that are absent from the file are ignored.

        Returns
        -------
        ScenarioStore
            Indexed by scenario_id. Timesteps sorted ascending within each scenario.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If required columns are missing, no rate columns are found, or
            duplicate (scenario_id, timestep) pairs are detected.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ESG scenario file not found: {path}")

        df = pd.read_csv(path)
        cls._validate_columns(df)

        # Identify rate columns and parse their maturities
        rate_cols, maturities = cls._parse_rate_columns(df)

        # Filter to requested scenario IDs
        if scenario_ids is not None:
            df = df[df["scenario_id"].isin(scenario_ids)].copy()

        # Check for duplicate (scenario_id, timestep)
        dupes = df.duplicated(subset=["scenario_id", "timestep"])
        if dupes.any():
            bad = df.loc[dupes, ["scenario_id", "timestep"]].head(3).to_dict("records")
            raise ValueError(
                f"ESG file contains duplicate (scenario_id, timestep) pairs: {bad}"
            )

        # Build EsgScenario objects
        scenarios: list[EsgScenario] = []
        for scen_id, group in df.groupby("scenario_id", sort=True):
            group = group.sort_values("timestep")
            timesteps: list[AssetScenarioPoint] = []
            for _, row in group.iterrows():
                spot_rates = {
                    m / 12.0: float(row[c])
                    for m, c in zip(maturities, rate_cols)
                }
                curve = RiskFreeRateCurve(spot_rates=spot_rates)
                timesteps.append(AssetScenarioPoint(
                    timestep=int(row["timestep"]),
                    rate_curve=curve,
                    equity_total_return_yr=float(row["equity_return_yr"]),
                    cpi_annual_rate=(
                        float(row["cpi_annual_rate"]) if "cpi_annual_rate" in df.columns else None
                    ),
                    rpi_annual_rate=(
                        float(row["rpi_annual_rate"]) if "rpi_annual_rate" in df.columns else None
                    ),
                ))
            scenarios.append(EsgScenario(scenario_id=int(scen_id), timesteps=timesteps))

        return ScenarioStore(scenarios)

    # -----------------------------------------------------------------------
    # flat — analytical generator for tests and simple sensitivity runs
    # -----------------------------------------------------------------------

    @classmethod
    def flat(
        cls,
        n_scenarios: int,
        rate: float,
        equity_return_yr: float,
        n_months: int,
        first_scenario_id: int = 1,
        cpi_annual_rate: Optional[float] = None,
        rpi_annual_rate: Optional[float] = None,
    ) -> ScenarioStore:
        """
        Generate N identical flat-rate scenarios.

        All scenarios share the same flat RiskFreeRateCurve and equity return
        at every timestep. Useful for unit tests and baseline sensitivity runs
        where scenario variation is not under test.

        Parameters
        ----------
        n_scenarios : int
            Number of scenarios to generate.
        rate : float
            Flat annual spot rate applied at all maturities (e.g. 0.03 = 3%).
        equity_return_yr : float
            Annual total equity return for every month (e.g. 0.07 = 7%).
        n_months : int
            Number of monthly timesteps per scenario (= projection_term_months).
        first_scenario_id : int
            scenario_id assigned to the first generated scenario. Subsequent
            scenarios are numbered first_scenario_id+1, first_scenario_id+2, ...
        cpi_annual_rate : float | None
            Flat annual CPI rate applied at all timesteps (e.g. 0.03 = 3%).
            None (default) for non-BPA runs.
        rpi_annual_rate : float | None
            Flat annual RPI rate applied at all timesteps.
            None (default) for non-BPA runs.

        Returns
        -------
        ScenarioStore
        """
        if n_scenarios < 1:
            raise ValueError(f"n_scenarios must be >= 1, got {n_scenarios}.")
        if n_months < 1:
            raise ValueError(f"n_months must be >= 1, got {n_months}.")

        curve = RiskFreeRateCurve.flat(rate)
        scenarios: list[EsgScenario] = []
        for i in range(n_scenarios):
            scen_id = first_scenario_id + i
            timesteps = [
                AssetScenarioPoint(
                    timestep=t,
                    rate_curve=curve,
                    equity_total_return_yr=equity_return_yr,
                    cpi_annual_rate=cpi_annual_rate,
                    rpi_annual_rate=rpi_annual_rate,
                )
                for t in range(n_months)
            ]
            scenarios.append(EsgScenario(scenario_id=scen_id, timesteps=timesteps))

        return ScenarioStore(scenarios)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @classmethod
    def _validate_columns(cls, df: pd.DataFrame) -> None:
        """Raise ValueError if required base columns or rate columns are missing."""
        missing = _REQUIRED_BASE_COLS - set(df.columns)
        if missing:
            raise ValueError(
                f"ESG CSV is missing required columns: {sorted(missing)}. "
                f"Required: {sorted(_REQUIRED_BASE_COLS)} plus at least one r_{{n}}m column."
            )
        rate_cols = [c for c in df.columns if _RATE_COL_RE.match(c)]
        if not rate_cols:
            raise ValueError(
                "ESG CSV contains no rate columns. "
                "At least one column named r_{n}m (e.g. r_12m) is required."
            )

    @classmethod
    def _parse_rate_columns(
        cls, df: pd.DataFrame
    ) -> tuple[list[str], list[int]]:
        """
        Return (rate_col_names, maturity_months) sorted by maturity ascending.
        """
        pairs: list[tuple[int, str]] = []
        for col in df.columns:
            m = _RATE_COL_RE.match(col)
            if m:
                pairs.append((int(m.group(1)), col))
        pairs.sort()
        maturities = [p[0] for p in pairs]
        rate_cols  = [p[1] for p in pairs]
        return rate_cols, maturities
