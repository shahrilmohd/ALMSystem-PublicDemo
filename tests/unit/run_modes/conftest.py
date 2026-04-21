"""
Shared fixtures for engine.run_modes unit tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig
from tests.unit.config.conftest import build_run_config_dict


@pytest.fixture
def run_config(tmp_path: Path) -> RunConfig:
    assumption_dir = tmp_path / "assumptions"
    assumption_dir.mkdir()
    fund_config_file = tmp_path / "fund_config.yaml"
    fund_config_file.write_text("placeholder: true\n")
    data = build_run_config_dict(
        fund_config_path=fund_config_file,
        assumption_dir=assumption_dir,
    )
    return RunConfig.from_dict(data)


@pytest.fixture
def fund_config() -> FundConfig:
    return FundConfig.from_dict({
        "fund_id": "FUND_A",
        "fund_name": "Fund A",
        "saa_weights": {"bonds": 0.6, "equities": 0.3, "derivatives": 0.0, "cash": 0.1},
        "crediting_groups": [
            {"group_id": "GRP_A", "group_name": "Group A", "product_codes": ["P1"]},
        ],
    })
