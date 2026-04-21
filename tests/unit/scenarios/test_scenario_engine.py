"""
Unit tests for engine/scenarios/scenario_engine.py.

Rules under test
----------------
ScenarioLoader.from_csv:
  1.  Valid CSV produces a ScenarioStore with correct scenario count.
  2.  Rate curve columns (r_{n}m) are parsed and maturities extracted correctly.
  3.  Timesteps within each scenario are sorted ascending by month.
  4.  equity_return_yr is stored on each AssetScenarioPoint.
  5.  scenario_ids filter loads only the requested IDs.
  6.  IDs in the filter absent from the file are silently ignored.
  7.  FileNotFoundError raised for missing file.
  8.  ValueError raised when required column 'scenario_id' is missing.
  9.  ValueError raised when required column 'timestep' is missing.
  10. ValueError raised when required column 'equity_return_yr' is missing.
  11. ValueError raised when no r_{n}m rate columns are present.
  12. ValueError raised for duplicate (scenario_id, timestep) pairs.
  13. CSV with cpi_annual_rate/rpi_annual_rate populates inflation fields.
  14. CSV without inflation columns leaves fields as None (backwards compat).

ScenarioLoader.flat:
  15. Returns ScenarioStore with exactly n_scenarios scenarios.
  16. All scenarios have n_months timesteps.
  17. All AssetScenarioPoints carry the flat rate and equity return.
  18. scenario_ids are sequential from first_scenario_id.
  19. n_scenarios < 1 raises ValueError.
  20. n_months < 1 raises ValueError.
  21. Optional inflation params populate cpi/rpi fields on every timestep.
  22. Default (no inflation params) leaves cpi/rpi fields as None.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from engine.scenarios.scenario_engine import ScenarioLoader


# ---------------------------------------------------------------------------
# Shared CSV fixture
# Two scenarios (IDs 1 and 2), three timesteps each.
# Rate knot points: 1m, 12m, 60m.
# ---------------------------------------------------------------------------

_VALID_CSV = textwrap.dedent("""\
    scenario_id,timestep,r_1m,r_12m,r_60m,equity_return_yr
    1,0,0.0300,0.0350,0.0400,0.07
    1,1,0.0305,0.0355,0.0405,0.071
    1,2,0.0310,0.0360,0.0410,0.072
    2,0,0.0500,0.0550,0.0600,0.05
    2,1,0.0510,0.0560,0.0610,0.051
    2,2,0.0520,0.0570,0.0620,0.052
""")


def write_csv(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "scenarios.csv"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# ScenarioLoader.from_csv
# ---------------------------------------------------------------------------

class TestFromCsv:

    def test_correct_scenario_count(self, tmp_path):
        path = write_csv(tmp_path, _VALID_CSV)
        store = ScenarioLoader.from_csv(path)
        assert store.count() == 2

    def test_rate_columns_parsed(self, tmp_path):
        """Rate curve has knots at 1m, 12m, 60m — all discount factors valid."""
        path = write_csv(tmp_path, _VALID_CSV)
        store = ScenarioLoader.from_csv(path)
        pt = store.get(1).get_timestep(0)
        for maturity in [1, 12, 60]:
            assert 0.0 < pt.rate_curve.discount_factor(maturity) < 1.0

    def test_timesteps_sorted_ascending(self, tmp_path):
        """Rows in reverse order in the file — loader must sort them."""
        reversed_csv = textwrap.dedent("""\
            scenario_id,timestep,r_1m,equity_return_yr
            1,2,0.03,0.07
            1,0,0.03,0.07
            1,1,0.03,0.07
        """)
        path = write_csv(tmp_path, reversed_csv)
        store = ScenarioLoader.from_csv(path)
        ts = [pt.timestep for pt in store.get(1).timesteps]
        assert ts == [0, 1, 2]

    def test_equity_return_stored(self, tmp_path):
        path = write_csv(tmp_path, _VALID_CSV)
        store = ScenarioLoader.from_csv(path)
        pt = store.get(1).get_timestep(0)
        assert pt.equity_total_return_yr == pytest.approx(0.07)

    def test_equity_return_varies_by_timestep(self, tmp_path):
        """Equity return at t=1 differs from t=0 — both stored correctly."""
        path = write_csv(tmp_path, _VALID_CSV)
        store = ScenarioLoader.from_csv(path)
        assert store.get(1).get_timestep(1).equity_total_return_yr == pytest.approx(0.071)

    def test_scenario_ids_filter(self, tmp_path):
        path = write_csv(tmp_path, _VALID_CSV)
        store = ScenarioLoader.from_csv(path, scenario_ids=[1])
        assert store.count() == 1
        assert store.scenario_ids() == [1]

    def test_filter_ignores_absent_ids(self, tmp_path):
        """ID 999 is not in the file — silently ignored, not an error."""
        path = write_csv(tmp_path, _VALID_CSV)
        store = ScenarioLoader.from_csv(path, scenario_ids=[1, 999])
        assert store.count() == 1
        assert 1 in store
        assert 999 not in store

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ScenarioLoader.from_csv(tmp_path / "nonexistent.csv")

    def test_missing_scenario_id_column(self, tmp_path):
        csv = "timestep,r_1m,equity_return_yr\n0,0.03,0.07\n"
        path = write_csv(tmp_path, csv)
        with pytest.raises(ValueError, match="scenario_id"):
            ScenarioLoader.from_csv(path)

    def test_missing_timestep_column(self, tmp_path):
        csv = "scenario_id,r_1m,equity_return_yr\n1,0.03,0.07\n"
        path = write_csv(tmp_path, csv)
        with pytest.raises(ValueError, match="timestep"):
            ScenarioLoader.from_csv(path)

    def test_missing_equity_return_column(self, tmp_path):
        csv = "scenario_id,timestep,r_1m\n1,0,0.03\n"
        path = write_csv(tmp_path, csv)
        with pytest.raises(ValueError, match="equity_return_yr"):
            ScenarioLoader.from_csv(path)

    def test_no_rate_columns(self, tmp_path):
        csv = "scenario_id,timestep,equity_return_yr\n1,0,0.07\n"
        path = write_csv(tmp_path, csv)
        with pytest.raises(ValueError, match="rate"):
            ScenarioLoader.from_csv(path)

    def test_duplicate_scenario_timestep_raises(self, tmp_path):
        csv = textwrap.dedent("""\
            scenario_id,timestep,r_1m,equity_return_yr
            1,0,0.03,0.07
            1,0,0.03,0.07
        """)
        path = write_csv(tmp_path, csv)
        with pytest.raises(ValueError, match="duplicate"):
            ScenarioLoader.from_csv(path)


# ---------------------------------------------------------------------------
# ScenarioLoader.flat
# ---------------------------------------------------------------------------

class TestFlat:

    def test_scenario_count(self):
        store = ScenarioLoader.flat(n_scenarios=5, rate=0.03,
                                    equity_return_yr=0.07, n_months=12)
        assert store.count() == 5

    def test_n_months_per_scenario(self):
        store = ScenarioLoader.flat(n_scenarios=3, rate=0.03,
                                    equity_return_yr=0.07, n_months=24)
        for s in store.all_scenarios():
            assert s.n_months == 24

    def test_rate_curve_applied(self):
        """Flat 3% curve: DF(12) should be in (0, 1)."""
        store = ScenarioLoader.flat(n_scenarios=1, rate=0.03,
                                    equity_return_yr=0.07, n_months=3)
        pt = store.get(1).get_timestep(0)
        df = pt.rate_curve.discount_factor(12)
        assert 0.0 < df < 1.0

    def test_equity_return_stored(self):
        store = ScenarioLoader.flat(n_scenarios=1, rate=0.03,
                                    equity_return_yr=0.08, n_months=3)
        pt = store.get(1).get_timestep(0)
        assert pt.equity_total_return_yr == pytest.approx(0.08)

    def test_scenario_ids_sequential_from_default(self):
        store = ScenarioLoader.flat(n_scenarios=3, rate=0.03,
                                    equity_return_yr=0.07, n_months=3)
        assert store.scenario_ids() == [1, 2, 3]

    def test_scenario_ids_sequential_from_custom_start(self):
        store = ScenarioLoader.flat(n_scenarios=3, rate=0.03,
                                    equity_return_yr=0.07, n_months=3,
                                    first_scenario_id=10)
        assert store.scenario_ids() == [10, 11, 12]

    def test_zero_scenarios_raises(self):
        with pytest.raises(ValueError):
            ScenarioLoader.flat(n_scenarios=0, rate=0.03,
                                equity_return_yr=0.07, n_months=12)

    def test_zero_months_raises(self):
        with pytest.raises(ValueError):
            ScenarioLoader.flat(n_scenarios=1, rate=0.03,
                                equity_return_yr=0.07, n_months=0)

    def test_inflation_fields_populated(self):
        """Passing cpi/rpi to flat() stores them on every AssetScenarioPoint."""
        store = ScenarioLoader.flat(
            n_scenarios=2, rate=0.03, equity_return_yr=0.07, n_months=3,
            cpi_annual_rate=0.025, rpi_annual_rate=0.030,
        )
        for scen in store.all_scenarios():
            for pt in scen.timesteps:
                assert pt.cpi_annual_rate == pytest.approx(0.025)
                assert pt.rpi_annual_rate == pytest.approx(0.030)

    def test_inflation_fields_default_none(self):
        """Omitting cpi/rpi leaves fields as None on every AssetScenarioPoint."""
        store = ScenarioLoader.flat(n_scenarios=1, rate=0.03,
                                    equity_return_yr=0.07, n_months=3)
        for pt in store.get(1).timesteps:
            assert pt.cpi_annual_rate is None
            assert pt.rpi_annual_rate is None


# ---------------------------------------------------------------------------
# ScenarioLoader.from_csv — inflation column tests (§16)
# ---------------------------------------------------------------------------

_INFLATION_CSV = textwrap.dedent("""\
    scenario_id,timestep,r_1m,r_12m,equity_return_yr,cpi_annual_rate,rpi_annual_rate
    1,0,0.03,0.035,0.07,0.025,0.030
    1,1,0.03,0.035,0.07,0.026,0.031
    2,0,0.04,0.045,0.06,0.025,0.030
    2,1,0.04,0.045,0.06,0.027,0.032
""")


class TestFromCsvInflation:

    def test_inflation_columns_populated(self, tmp_path):
        """CSV with inflation columns populates cpi/rpi on each point."""
        path = write_csv(tmp_path, _INFLATION_CSV)
        store = ScenarioLoader.from_csv(path)
        pt = store.get(1).get_timestep(0)
        assert pt.cpi_annual_rate == pytest.approx(0.025)
        assert pt.rpi_annual_rate == pytest.approx(0.030)

    def test_inflation_varies_by_timestep(self, tmp_path):
        """Inflation values differ at t=0 vs t=1 — both stored correctly."""
        path = write_csv(tmp_path, _INFLATION_CSV)
        store = ScenarioLoader.from_csv(path)
        pt1 = store.get(1).get_timestep(1)
        assert pt1.cpi_annual_rate == pytest.approx(0.026)
        assert pt1.rpi_annual_rate == pytest.approx(0.031)

    def test_inflation_varies_by_scenario(self, tmp_path):
        """Inflation differs between scenarios — stored independently."""
        path = write_csv(tmp_path, _INFLATION_CSV)
        store = ScenarioLoader.from_csv(path)
        pt2 = store.get(2).get_timestep(1)
        assert pt2.cpi_annual_rate == pytest.approx(0.027)
        assert pt2.rpi_annual_rate == pytest.approx(0.032)

    def test_no_inflation_columns_gives_none(self, tmp_path):
        """CSV without inflation columns leaves fields as None (backwards compat)."""
        path = write_csv(tmp_path, _VALID_CSV)
        store = ScenarioLoader.from_csv(path)
        pt = store.get(1).get_timestep(0)
        assert pt.cpi_annual_rate is None
        assert pt.rpi_annual_rate is None

    def test_partial_inflation_columns_not_required(self, tmp_path):
        """A CSV with only cpi_annual_rate (no rpi) is valid; rpi stays None."""
        csv = textwrap.dedent("""\
            scenario_id,timestep,r_1m,equity_return_yr,cpi_annual_rate
            1,0,0.03,0.07,0.025
        """)
        path = write_csv(tmp_path, csv)
        store = ScenarioLoader.from_csv(path)
        pt = store.get(1).get_timestep(0)
        assert pt.cpi_annual_rate == pytest.approx(0.025)
        assert pt.rpi_annual_rate is None
