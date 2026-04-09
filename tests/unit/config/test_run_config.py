"""
Unit tests for RunConfig and its sub-configs.

Sub-configs tested here (those with cross-field or path rules):
  - DatabaseSourceConfig: exactly one of table_name / sql_query
  - FileSourceConfig: file_path must exist
  - ModelPointSourceConfig: source_type must match populated sub-config
  - RunConfig (cross-field rules):
      * STOCHASTIC requires stochastic config block
      * STOCHASTIC requires GROUP_MP input mode
      * STOCHASTIC requires scenario_file_path
      * DETERMINISTIC / STOCHASTIC require asset_data_path
      * output_timestep period must be >= projection_timestep period
      * output_horizon_years must be <= projection_term_years

OutputConfig field defaults are also verified here.

Fixtures from conftest.py
--------------------------
assumption_dir      -- empty temp directory
fund_config_file    -- stub YAML file
asset_file          -- stub CSV file
scenario_file       -- stub CSV file
mp_file             -- stub CSV file (for FileSourceConfig tests)
build_run_config_dict() -- helper that builds a valid LIABILITY_ONLY dict
"""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from engine.config.run_config import (
    DatabaseSourceConfig,
    FileSourceConfig,
    AssumptionFileFormat,
    ModelPointSourceConfig,
    ModelPointSourceType,
    OutputConfig,
    OutputTimestep,
    ResultFormat,
    RunConfig,
    RunType,
)
from tests.unit.config.conftest import build_run_config_dict


# ---------------------------------------------------------------------------
# DatabaseSourceConfig
# ---------------------------------------------------------------------------

class TestDatabaseSourceConfig:
    def test_table_name_only_accepted(self):
        cfg = DatabaseSourceConfig(
            connection_string="sqlite:///test.db",
            table_name="model_points",
        )
        assert cfg.table_name == "model_points"
        assert cfg.sql_query is None

    def test_sql_query_only_accepted(self):
        cfg = DatabaseSourceConfig(
            connection_string="sqlite:///test.db",
            sql_query="SELECT * FROM model_points",
        )
        assert cfg.sql_query == "SELECT * FROM model_points"
        assert cfg.table_name is None

    def test_both_table_and_query_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            DatabaseSourceConfig(
                connection_string="sqlite:///test.db",
                table_name="model_points",
                sql_query="SELECT * FROM model_points",
            )
        assert "both" in str(exc_info.value).lower()

    def test_neither_table_nor_query_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            DatabaseSourceConfig(connection_string="sqlite:///test.db")
        assert "one of" in str(exc_info.value).lower()

    def test_default_connection_timeout(self):
        cfg = DatabaseSourceConfig(
            connection_string="sqlite:///test.db",
            table_name="mp",
        )
        assert cfg.connection_timeout_seconds == 30

    def test_custom_connection_timeout(self):
        cfg = DatabaseSourceConfig(
            connection_string="sqlite:///test.db",
            table_name="mp",
            connection_timeout_seconds=60,
        )
        assert cfg.connection_timeout_seconds == 60


# ---------------------------------------------------------------------------
# FileSourceConfig
# ---------------------------------------------------------------------------

class TestFileSourceConfig:
    def test_existing_csv_file_accepted(self, mp_file: Path):
        cfg = FileSourceConfig(file_path=str(mp_file), file_format="csv")
        assert cfg.file_path == mp_file.resolve()

    def test_nonexistent_file_rejected(self, tmp_path: Path):
        with pytest.raises(ValidationError) as exc_info:
            FileSourceConfig(
                file_path=str(tmp_path / "does_not_exist.csv"),
                file_format="csv",
            )
        assert "not found" in str(exc_info.value).lower()

    def test_excel_format_accepted(self, mp_file: Path):
        cfg = FileSourceConfig(file_path=str(mp_file), file_format="excel")
        assert cfg.file_format == AssumptionFileFormat.EXCEL

    def test_sheet_name_defaults_to_none(self, mp_file: Path):
        cfg = FileSourceConfig(file_path=str(mp_file), file_format="csv")
        assert cfg.sheet_name is None


# ---------------------------------------------------------------------------
# ModelPointSourceConfig
# ---------------------------------------------------------------------------

class TestModelPointSourceConfig:
    def test_database_source_accepted(self):
        cfg = ModelPointSourceConfig(
            source_type="database",
            database={
                "connection_string": "sqlite:///test.db",
                "table_name": "mp",
            },
        )
        assert cfg.source_type == ModelPointSourceType.DATABASE
        assert cfg.database is not None
        assert cfg.file is None

    def test_file_source_accepted(self, mp_file: Path):
        cfg = ModelPointSourceConfig(
            source_type="file",
            file={"file_path": str(mp_file), "file_format": "csv"},
        )
        assert cfg.source_type == ModelPointSourceType.FILE
        assert cfg.file is not None
        assert cfg.database is None

    def test_database_type_without_database_block_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ModelPointSourceConfig(source_type="database")
        assert "database" in str(exc_info.value).lower()

    def test_file_type_without_file_block_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ModelPointSourceConfig(source_type="file")
        assert "file" in str(exc_info.value).lower()

    def test_database_type_with_extra_file_block_rejected(self, mp_file: Path):
        with pytest.raises(ValidationError) as exc_info:
            ModelPointSourceConfig(
                source_type="database",
                database={"connection_string": "sqlite:///test.db", "table_name": "mp"},
                file={"file_path": str(mp_file), "file_format": "csv"},
            )
        assert "file" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# OutputConfig defaults
# ---------------------------------------------------------------------------

class TestOutputConfigDefaults:
    def test_all_defaults(self):
        cfg = OutputConfig()
        assert cfg.save_policy_level_results is False
        assert cfg.save_fund_level_results is True
        assert cfg.save_company_level_results is True
        assert cfg.output_timestep == OutputTimestep.MONTHLY
        assert cfg.output_horizon_years is None
        assert cfg.result_format == ResultFormat.CSV
        assert cfg.compress_outputs is False
        assert cfg.output_dir == Path("outputs")


# ---------------------------------------------------------------------------
# RunConfig — happy paths
# ---------------------------------------------------------------------------

class TestRunConfigHappyPath:
    def test_minimal_liability_only_run_accepted(
        self, fund_config_file: Path, assumption_dir: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
        )
        cfg = RunConfig.from_dict(data)
        assert cfg.run_type == RunType.LIABILITY_ONLY
        assert cfg.run_id == "test-run-001"

    def test_output_timestep_coarser_than_projection_accepted(
        self, fund_config_file: Path, assumption_dir: Path
    ):
        # monthly projection → annual output is valid (coarser)
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            projection_timestep="monthly",
            output_timestep="annual",
        )
        cfg = RunConfig.from_dict(data)
        assert cfg.output.output_timestep == OutputTimestep.ANNUAL

    def test_output_timestep_equal_to_projection_accepted(
        self, fund_config_file: Path, assumption_dir: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            projection_timestep="quarterly",
            decision_timestep="annual",
            output_timestep="quarterly",
        )
        cfg = RunConfig.from_dict(data)
        assert cfg.output.output_timestep.value == "quarterly"

    def test_output_horizon_within_term_accepted(
        self, fund_config_file: Path, assumption_dir: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            projection_term_years=100,
            output_horizon_years=30,
        )
        cfg = RunConfig.from_dict(data)
        assert cfg.output.output_horizon_years == 30

    def test_output_horizon_equal_to_term_accepted(
        self, fund_config_file: Path, assumption_dir: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            projection_term_years=30,
            output_horizon_years=30,
        )
        cfg = RunConfig.from_dict(data)
        assert cfg.output.output_horizon_years == 30

    def test_valid_stochastic_run_accepted(
        self,
        fund_config_file: Path,
        assumption_dir: Path,
        asset_file: Path,
        scenario_file: Path,
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            run_type="stochastic",
            input_mode="group_mp",
            asset_data_path=asset_file,
            scenario_file_path=scenario_file,
            stochastic={"num_scenarios": 1000},
        )
        cfg = RunConfig.from_dict(data)
        assert cfg.run_type == RunType.STOCHASTIC
        assert cfg.stochastic.num_scenarios == 1000

    def test_valid_deterministic_run_accepted(
        self, fund_config_file: Path, assumption_dir: Path, asset_file: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            run_type="deterministic",
            asset_data_path=asset_file,
        )
        cfg = RunConfig.from_dict(data)
        assert cfg.run_type == RunType.DETERMINISTIC


# ---------------------------------------------------------------------------
# RunConfig — cross-field validation rejections
# ---------------------------------------------------------------------------

class TestRunConfigCrossFieldValidation:

    # --- output_timestep vs projection_timestep ---

    @pytest.mark.parametrize("projection,output", [
        ("annual",    "monthly"),    # output finer than projection → reject
        ("annual",    "quarterly"),  # output finer than projection → reject
        ("quarterly", "monthly"),    # output finer than projection → reject
    ])
    def test_output_timestep_finer_than_projection_rejected(
        self, fund_config_file: Path, assumption_dir: Path, projection: str, output: str
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            projection_timestep=projection,
            decision_timestep="annual",
            output_timestep=output,
        )
        with pytest.raises(ValidationError) as exc_info:
            RunConfig.from_dict(data)
        assert "output_timestep" in str(exc_info.value)

    # --- output_horizon_years vs projection_term_years ---

    def test_output_horizon_exceeds_term_rejected(
        self, fund_config_file: Path, assumption_dir: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            projection_term_years=30,
            output_horizon_years=50,
        )
        with pytest.raises(ValidationError) as exc_info:
            RunConfig.from_dict(data)
        assert "output_horizon_years" in str(exc_info.value)

    # --- Stochastic rules ---

    def test_stochastic_without_config_block_rejected(
        self, fund_config_file: Path, assumption_dir: Path, asset_file: Path, scenario_file: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            run_type="stochastic",
            input_mode="group_mp",
            asset_data_path=asset_file,
            scenario_file_path=scenario_file,
            # stochastic block intentionally omitted
        )
        with pytest.raises(ValidationError) as exc_info:
            RunConfig.from_dict(data)
        assert "stochastic" in str(exc_info.value).lower()

    def test_stochastic_with_seriatim_input_rejected(
        self, fund_config_file: Path, assumption_dir: Path, asset_file: Path, scenario_file: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            run_type="stochastic",
            input_mode="seriatim",          # not allowed for stochastic
            asset_data_path=asset_file,
            scenario_file_path=scenario_file,
            stochastic={"num_scenarios": 500},
        )
        with pytest.raises(ValidationError) as exc_info:
            RunConfig.from_dict(data)
        assert "group_mp" in str(exc_info.value).lower()

    def test_stochastic_without_scenario_file_rejected(
        self, fund_config_file: Path, assumption_dir: Path, asset_file: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            run_type="stochastic",
            input_mode="group_mp",
            asset_data_path=asset_file,
            # scenario_file_path intentionally omitted
            stochastic={"num_scenarios": 1000},
        )
        with pytest.raises(ValidationError) as exc_info:
            RunConfig.from_dict(data)
        assert "scenario_file_path" in str(exc_info.value)

    # --- Asset data required for non-liability runs ---

    def test_deterministic_without_asset_file_rejected(
        self, fund_config_file: Path, assumption_dir: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            run_type="deterministic",
            # asset_data_path intentionally omitted
        )
        with pytest.raises(ValidationError) as exc_info:
            RunConfig.from_dict(data)
        assert "asset_data_path" in str(exc_info.value)

    def test_stochastic_without_asset_file_rejected(
        self, fund_config_file: Path, assumption_dir: Path, scenario_file: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            run_type="stochastic",
            input_mode="group_mp",
            scenario_file_path=scenario_file,
            stochastic={"num_scenarios": 1000},
            # asset_data_path intentionally omitted
        )
        with pytest.raises(ValidationError) as exc_info:
            RunConfig.from_dict(data)
        assert "asset_data_path" in str(exc_info.value)

    # --- Multiple errors reported together ---

    def test_multiple_stochastic_errors_reported_together(
        self, fund_config_file: Path, assumption_dir: Path
    ):
        # Stochastic run missing: config block, scenario file, and asset file.
        # All three errors should appear in a single ValidationError.
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            run_type="stochastic",
            input_mode="group_mp",
        )
        with pytest.raises(ValidationError) as exc_info:
            RunConfig.from_dict(data)
        error_text = str(exc_info.value)
        assert "stochastic" in error_text.lower()
        assert "scenario_file_path" in error_text
        assert "asset_data_path" in error_text


# ---------------------------------------------------------------------------
# RunConfig — summary()
# ---------------------------------------------------------------------------

class TestRunConfigSummary:
    def test_summary_contains_key_fields(
        self, fund_config_file: Path, assumption_dir: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            projection_term_years=30,
            output_timestep="annual",
            output_horizon_years=10,
        )
        cfg = RunConfig.from_dict(data)
        summary = cfg.summary()
        assert cfg.run_id in summary
        assert cfg.run_name in summary
        assert "liability_only" in summary
        assert "30" in summary          # projection term
        assert "annual" in summary      # output_timestep
        assert "10" in summary          # output_horizon_years

    def test_summary_full_horizon_label(
        self, fund_config_file: Path, assumption_dir: Path
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            projection_term_years=30,
        )
        cfg = RunConfig.from_dict(data)
        assert "full" in cfg.summary()

    def test_summary_stochastic_shows_scenario_count(
        self,
        fund_config_file: Path,
        assumption_dir: Path,
        asset_file: Path,
        scenario_file: Path,
    ):
        data = build_run_config_dict(
            fund_config_path=fund_config_file,
            assumption_dir=assumption_dir,
            run_type="stochastic",
            input_mode="group_mp",
            asset_data_path=asset_file,
            scenario_file_path=scenario_file,
            stochastic={"num_scenarios": 1000},
        )
        cfg = RunConfig.from_dict(data)
        assert "1,000" in cfg.summary()
