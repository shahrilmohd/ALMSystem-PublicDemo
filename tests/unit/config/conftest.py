"""
Shared fixtures for engine.config unit tests.

Path-dependent validators (fund_config_path, tables_root_dir, file_path)
all check that files/directories exist at construction time. These fixtures
use pytest's tmp_path to create the required stubs so tests can focus on
the rule being tested rather than file I/O.

build_run_config_dict() is a helper that returns a minimal valid
LIABILITY_ONLY run config dict. Individual tests override specific keys
to trigger (or avoid) the rule under test.
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Stub file / directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def assumption_dir(tmp_path: Path) -> Path:
    """Empty directory that satisfies AssumptionTablesConfig.tables_root_dir."""
    d = tmp_path / "assumptions"
    d.mkdir()
    return d


@pytest.fixture
def fund_config_file(tmp_path: Path) -> Path:
    """Stub YAML file that satisfies InputSourcesConfig.fund_config_path."""
    f = tmp_path / "fund_config.yaml"
    f.write_text("placeholder: true\n")
    return f


@pytest.fixture
def asset_file(tmp_path: Path) -> Path:
    """Stub CSV that satisfies InputSourcesConfig.asset_data_path."""
    f = tmp_path / "assets.csv"
    f.write_text("asset_id,value\n")
    return f


@pytest.fixture
def scenario_file(tmp_path: Path) -> Path:
    """Stub CSV that satisfies InputSourcesConfig.scenario_file_path."""
    f = tmp_path / "scenarios.csv"
    f.write_text("scenario_id\n")
    return f


@pytest.fixture
def mp_file(tmp_path: Path) -> Path:
    """Stub CSV that satisfies FileSourceConfig.file_path."""
    f = tmp_path / "model_points.csv"
    f.write_text("policy_id\n1\n")
    return f


# ---------------------------------------------------------------------------
# Minimal valid RunConfig dict builder
# ---------------------------------------------------------------------------

def build_run_config_dict(
    *,
    fund_config_path: Path,
    assumption_dir: Path,
    run_type: str = "liability_only",
    input_mode: str = "group_mp",
    projection_term_years: int = 30,
    projection_timestep: str = "monthly",
    decision_timestep: str = "annual",
    output_timestep: str = "monthly",
    output_horizon_years: int | None = None,
    asset_data_path: Path | None = None,
    scenario_file_path: Path | None = None,
    stochastic: dict | None = None,
) -> dict:
    """
    Return a minimal valid RunConfig dict for LIABILITY_ONLY runs.

    Callers override individual keys to test specific validation rules.
    Model points always use DatabaseSourceConfig (no file existence check).
    """
    d: dict = {
        "run_id": "test-run-001",
        "run_name": "Test Run",
        "run_type": run_type,
        "projection": {
            "valuation_date": "2025-12-31",
            "projection_term_years": projection_term_years,
            "projection_timestep": projection_timestep,
            "decision_timestep": decision_timestep,
        },
        "input_sources": {
            "model_points": {
                "source_type": "database",
                "database": {
                    "connection_string": "sqlite:///test.db",
                    "table_name": "model_points",
                },
            },
            "assumption_tables": {
                "tables_root_dir": str(assumption_dir),
            },
            "fund_config_path": str(fund_config_path),
        },
        "liability": {
            "active_models": ["conventional"],
            "input_mode": input_mode,
        },
        "output": {
            "output_timestep": output_timestep,
        },
    }

    if output_horizon_years is not None:
        d["output"]["output_horizon_years"] = output_horizon_years

    if asset_data_path is not None:
        d["input_sources"]["asset_data_path"] = str(asset_data_path)

    if scenario_file_path is not None:
        d["input_sources"]["scenario_file_path"] = str(scenario_file_path)

    if stochastic is not None:
        d["stochastic"] = stochastic

    return d
