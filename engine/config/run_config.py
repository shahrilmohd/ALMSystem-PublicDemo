"""
Master run configuration for the ALM model.

All inputs to a model run are defined and validated here using Pydantic v2.
No run mode, fund, or model class should accept raw dicts or loose keyword
arguments — they must receive a validated config object from this module.

Design notes:
- RunConfig is the single source of truth for one model execution.
- Sub-configs (ProjectionConfig, StochasticConfig, etc.) are nested Pydantic
  models. This keeps validation localised and fields grouped by concern.
- All file paths are stored as pathlib.Path so callers never deal with raw
  strings. Existence checks are done at validation time where appropriate.
- Enums are used for all constrained string fields (run type, timestep, etc.)
  so the AI layer can introspect valid options from the schema directly.
- Currency is stored explicitly even though only one is currently supported.
  This prevents the value from being buried in code and makes it visible to
  the AI layer when it reads the config schema.

Usage:
    config = RunConfig.from_yaml("config_files/run_config.yaml")
    config = RunConfig.from_dict({...})
    config = RunConfig(...)     # direct construction, e.g. in tests

"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Re-export time-dimension types so callers can import everything from here.
from engine.config.projection_config import (  # noqa: F401
    Currency,
    DecisionTimestep,
    ProjectionConfig,
    ProjectionTimestep,
)

# ===========================================================================
# Enumerations
# All constrained string fields use Enum so valid values are discoverable
# programmatically (by the AI layer, by the frontend, by tests).
# ===========================================================================

class RunType(str, Enum):
    """
    The three supported run modes.

    LIABILITY_ONLY:
        Projects liability cash flows only (no asset model).
        Supports both seriatim and group model point input.
        Used for: BEL validation, liability QA, standalone reserve runs.

    DETERMINISTIC:
        Full ALM run over a single economic scenario.
        Supports seriatim liability input.
        Produces: full P&L, asset/liability values, bonus rates.

    STOCHASTIC:
        Full ALM run over N economic scenarios from an ESG file.
        Requires group model point liability input (seriatim not permitted at
        the moment due to runtime and resource constraints).
        Produces: stochastic distribution of results, TVOG.
    """
    LIABILITY_ONLY = "liability_only"
    DETERMINISTIC  = "deterministic"
    STOCHASTIC     = "stochastic"


class LiabilityInputMode(str, Enum):
    """
    Controls whether liability data is loaded as individual policies
    or as pre-grouped model points.

    SERIATIM:
        One row per policy. Used in liability-only and deterministic runs.
        Produces the most granular output but is computationally expensive.
        NOT permitted for stochastic runs at the moment.

    GROUP_MP:
        Compressed representation of the portfolio (1,000–3,000 model points).
        Required for stochastic runs. Also valid for liability-only runs
        when speed matters over granularity.
    """
    SERIATIM = "seriatim"
    GROUP_MP = "group_mp"


class LiabilityModel(str, Enum):
    """
    Liability sub-models available. Multiple models can be active in one run.
    """
    CONVENTIONAL = "conventional"
    UNIT_LINKED  = "unit_linked"
    ANNUITY      = "annuity"


class ModelPointSourceType(str, Enum):
    """
    Controls how model point data is retrieved.

    DATABASE:
        Model points are pulled from a database (SQL Server, PostgreSQL,
        Oracle, SQLite, MySQL). Connection is described by a SQLAlchemy
        connection string — the engine type is encoded in the string itself,
        so no code changes are needed when switching database backends.
        See DatabaseSourceConfig.

    FILE:
        Model points are loaded from a flat file (CSV or Excel).
        Used as a fallback or for testing when no database is available.
        See FileSourceConfig.
    """
    DATABASE = "database"
    FILE     = "file"


class AssumptionFileFormat(str, Enum):
    """
    Supported formats for actuarial assumption table files.
    Assumption tables are always file-based (CSV or Excel).
    """
    CSV   = "csv"
    EXCEL = "excel"


class ResultFormat(str, Enum):
    """
    Output file format for model results.

    CSV:
        Plain text, universally readable, version-control friendly.
        Preferred for small deterministic runs and sharing with stakeholders.

    PARQUET:
        Columnar binary format. Significantly faster to read/write for large
        stochastic outputs where total volume can reach several GB.
        Recommended for stochastic runs.
    """
    CSV     = "csv"
    PARQUET = "parquet"


class OutputTimestep(str, Enum):
    """
    Frequency at which model results are aggregated and written to output files.

    This is independent of projection_timestep (the model's internal computation
    frequency). The model always runs at projection_timestep; this controls how
    results are bucketed before writing.

    MONTHLY:
        One output row per projection month. Largest files. Use when you need
        full granularity for detailed cash flow analysis.
        Example: 30-year monthly projection → 360 output rows per policy/fund.

    QUARTERLY:
        Cash flows summed to quarterly intervals. Moderate file size.
        Example: 30-year monthly projection → 120 output rows per policy/fund.

    ANNUAL:
        Cash flows summed to annual intervals. Smallest files. Use for long
        projections where monthly granularity is not needed.
        Example: 30-year monthly projection → 30 output rows per policy/fund.

    Constraint: output_timestep period must be >= projection_timestep period.
    You cannot output more granularly than the model computes.
    (e.g. monthly projection → monthly/quarterly/annual output all valid.
          annual projection → only annual output is valid.)
    """
    MONTHLY   = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL    = "annual"


# ===========================================================================
# Sub-Configurations
# Each concern has its own Pydantic model. RunConfig composes them.
# ===========================================================================

class DatabaseSourceConfig(BaseModel):
    """
    Connection details for retrieving model points from a database.

    SQLAlchemy connection strings encode the database engine, credentials,
    host, port, and database name in a single URL. This means switching
    from SQL Server to PostgreSQL requires only changing connection_string
    in the YAML — no code changes anywhere in the model.

    Connection string formats by engine:
        SQL Server : "mssql+pyodbc://user:pass@host/dbname?driver=ODBC+Driver+17+for+SQL+Server"
        PostgreSQL : "postgresql+psycopg2://user:pass@host:5432/dbname"
        Oracle     : "oracle+cx_oracle://user:pass@host:1521/sid"
        SQLite     : "sqlite:///path/to/file.db"
        MySQL      : "mysql+pymysql://user:pass@host:3306/dbname"

    table_name:
        Name of the database table holding model point data.
        Format: "schema.table_name" or just "table_name".
        Mutually exclusive with sql_query.

    sql_query:
        Full SQL SELECT statement. Use when you need to filter model points,
        e.g. by fund, product type, or valuation date.
        Mutually exclusive with table_name.
        Example: "SELECT * FROM dbo.ModelPoints WHERE FundID = 'FUND_A'
                  AND ValuationDate = '2025-12-31'"

    connection_timeout_seconds:
        Database connection timeout. Raise this for slow network connections
        to remote databases. Default: 30 seconds.
    """
    connection_string:          str           = Field(
        ...,
        description="SQLAlchemy connection URL. Encodes engine, credentials, host, DB."
    )
    table_name:                 Optional[str] = Field(
        default=None,
        description="DB table name for full table pull. Mutually exclusive with sql_query."
    )
    sql_query:                  Optional[str] = Field(
        default=None,
        description="SQL SELECT statement for filtered extraction. Mutually exclusive with table_name."
    )
    connection_timeout_seconds: int           = Field(
        default=30,
        ge=1,
        le=300,
        description="DB connection timeout in seconds. Default: 30."
    )

    @model_validator(mode="after")
    def exactly_one_data_source(self) -> DatabaseSourceConfig:
        """Enforce that exactly one of table_name or sql_query is provided."""
        has_table = self.table_name is not None
        has_query = self.sql_query is not None
        if has_table and has_query:
            raise ValueError(
                "DatabaseSourceConfig: provide either table_name or sql_query, not both."
            )
        if not has_table and not has_query:
            raise ValueError(
                "DatabaseSourceConfig: one of table_name or sql_query must be provided."
            )
        return self


class FileSourceConfig(BaseModel):
    """
    File-based source for model point data.
    Used as a fallback when no database is available, or for testing.

    file_path:
        Path to the model point file (CSV or Excel).
        Resolved to absolute path at validation time.
        File must exist at config load time.

    file_format:
        Explicit format declaration. Do not rely on file extension alone —
        some systems export CSV files with .txt or .dat extensions.

    sheet_name:
        For Excel files: the sheet containing model point data.
        Ignored for CSV. Default: first sheet (index 0).
    """
    file_path:   Path                  = Field(
        ...,
        description="Path to model point file (CSV or Excel)."
    )
    file_format: AssumptionFileFormat  = Field(
        ...,
        description="File format: csv or excel."
    )
    sheet_name:  Optional[str]         = Field(
        default=None,
        description="Excel sheet name. None = first sheet. Ignored for CSV."
    )

    @field_validator("file_path", mode="before")
    @classmethod
    def resolve_and_check_path(cls, v) -> Path:
        p = Path(v).resolve()
        if not p.exists():
            raise ValueError(
                f"Model point file not found: {p}\n"
                f"Check the path is correct and the file is accessible."
            )
        return p


class ModelPointSourceConfig(BaseModel):
    """
    Unified model point source configuration.
    Wraps either a DatabaseSourceConfig or a FileSourceConfig.

    source_type determines which sub-config is active:
        "database" -> database field must be populated, file must be None.
        "file"     -> file field must be populated, database must be None.

    This design means the liability data loader inspects source_type once
    at startup and delegates to the appropriate loader implementation.
    The rest of the model never knows or cares which source was used.

    column_map:
        Optional mapping from source column names to engine-standard names.
        Use when your input file uses different column names from what the
        engine expects. Keys are raw file column names; values are the
        engine-standard names.
        Example: {"if_count": "in_force_count", "sa": "sum_assured"}
        Columns not listed pass through unchanged.
        When omitted (or empty), column names in the file must exactly match
        the engine-standard names.
    """
    source_type: ModelPointSourceType          = Field(
        ...,
        description="Data source type: 'database' or 'file'."
    )
    database:    Optional[DatabaseSourceConfig] = Field(
        default=None,
        description="Database connection details. Required when source_type='database'."
    )
    file:        Optional[FileSourceConfig]     = Field(
        default=None,
        description="File source details. Required when source_type='file'."
    )
    column_map:  dict[str, str]                = Field(
        default_factory=dict,
        description=(
            "Source-to-engine column name mapping. "
            "Keys are column names in the raw file; values are engine-standard names. "
            "Example: {\"if_count\": \"in_force_count\", \"sa\": \"sum_assured\"}. "
            "Columns not listed pass through unchanged. "
            "Omit or leave empty if file columns already match engine-standard names."
        ),
    )

    @model_validator(mode="after")
    def validate_source_consistency(self) -> ModelPointSourceConfig:
        """Enforce that the correct sub-config is provided for the source_type."""
        errors = []
        if self.source_type == ModelPointSourceType.DATABASE:
            if self.database is None:
                errors.append(
                    "source_type='database' requires a 'database' config block."
                )
            if self.file is not None:
                errors.append(
                    "source_type='database': remove the 'file' block — it is not used."
                )
        elif self.source_type == ModelPointSourceType.FILE:
            if self.file is None:
                errors.append(
                    "source_type='file' requires a 'file' config block."
                )
            if self.database is not None:
                errors.append(
                    "source_type='file': remove the 'database' block — it is not used."
                )
        if errors:
            raise ValueError(
                "ModelPointSourceConfig errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
        return self


class AssumptionTablesConfig(BaseModel):
    """
    Configuration for actuarial assumption tables.

    Assumption tables (mortality rates, lapse rates, expense loadings,
    bonus crediting tables, etc.) are always file-based and organised
    in a single flat folder. The model scans this folder for files
    matching a known registry of expected filenames.

    tables_root_dir:
        Root folder containing all assumption table files.
        All sub-models (conventional, unit-linked, annuity) share this
        folder — files are distinguished by their standardised filenames.
        Must exist at config load time.

    file_format:
        Format of assumption table files. All files in the folder must
        use the same format — mixed formats in one folder are not supported.
        Default: CSV (simplest, version-control friendly).

    expected_tables:
        Registry of filenames the model expects to find in tables_root_dir.
        Populated by the config loader from a master list in the model.
        If a required table is missing, the model raises at startup — not
        mid-run when the missing table is first accessed.
        Key: logical table name (e.g. "mortality_rates").
        Value: expected filename (e.g. "mortality_rates.csv").

    allow_missing_tables:
        When False (default), all expected_tables must be present.
        When True, missing tables are logged as warnings but do not
        prevent the run from starting. Use with caution — a missing
        table will cause a runtime error when it is first accessed.
    """
    tables_root_dir:      Path                 = Field(
        ...,
        description="Root folder containing all assumption table files."
    )
    file_format:          AssumptionFileFormat  = Field(
        default=AssumptionFileFormat.CSV,
        description="Format for all assumption files in folder. Default: csv."
    )
    expected_tables:      dict[str, str]        = Field(
        default_factory=dict,
        description="Registry: logical_name -> filename. Populated by config loader."
    )
    allow_missing_tables: bool                  = Field(
        default=False,
        description="If False, all expected tables must exist. Recommended: False."
    )

    @field_validator("tables_root_dir", mode="before")
    @classmethod
    def resolve_and_check_dir(cls, v) -> Path:
        p = Path(v).resolve()
        if not p.exists():
            raise ValueError(
                f"Assumption tables directory not found: {p}\n"
                f"Create the folder or correct the path."
            )
        if not p.is_dir():
            raise ValueError(
                f"tables_root_dir must be a directory, not a file: {p}"
            )
        return p


class InputSourcesConfig(BaseModel):
    """
    All input data sources for the model run.

    model_points:
        Source config for policyholder / model point data.
        Either database or file-based. See ModelPointSourceConfig.

    assumption_tables:
        Folder-based config for all actuarial assumption tables.
        Flat folder, scanned for known filenames. See AssumptionTablesConfig.

    asset_data_path:
        Path to the asset portfolio input file (CSV or Excel).
        Required for DETERMINISTIC and STOCHASTIC runs.
        Not required for LIABILITY_ONLY runs — validated at RunConfig level.

    scenario_file_path:
        Path to the ESG scenario output file.
        Required for STOCHASTIC runs only.

    fund_config_path:
        Path to the fund configuration YAML.
        Contains: SAA weights, crediting groups, fund membership mappings.
        Required for DETERMINISTIC and STOCHASTIC runs (which need the asset
        model and fund-level coordination).
        Optional for LIABILITY_ONLY runs — the liability projection has no
        asset model and never reads FundConfig at runtime.
        Validated at RunConfig level (see validate_run_type_consistency).
    """
    model_points:       ModelPointSourceConfig  = Field(
        ...,
        description="Model point data source: database or file."
    )
    assumption_tables:  AssumptionTablesConfig  = Field(
        ...,
        description="Assumption tables folder configuration."
    )
    asset_data_path:    Optional[Path]          = Field(
        default=None,
        description="Asset portfolio file (CSV/XLSX). Required for non-liability-only runs."
    )
    scenario_file_path: Optional[Path]          = Field(
        default=None,
        description="ESG scenario file path. Required for stochastic runs."
    )
    fund_config_path:   Optional[Path]          = Field(
        default=None,
        description=(
            "Path to fund configuration YAML file. "
            "Required for DETERMINISTIC and STOCHASTIC runs. "
            "Optional for LIABILITY_ONLY runs."
        ),
    )

    @field_validator("asset_data_path", "scenario_file_path", mode="before")
    @classmethod
    def resolve_optional_path(cls, v) -> Optional[Path]:
        return Path(v).resolve() if v is not None else None

    @field_validator("fund_config_path", mode="before")
    @classmethod
    def resolve_and_check_fund_config(cls, v) -> Optional[Path]:
        if v is None:
            return None
        p = Path(v).resolve()
        if not p.exists():
            raise ValueError(f"Fund config file not found: {p}")
        return p


class LiabilityConfig(BaseModel):
    """
    Control which liability sub-models are active and how they run.

    active_models:
        List of liability sub-models to include in this run.
        At least one must be specified.
        Example: [CONVENTIONAL, UNIT_LINKED]

    input_mode:
        Whether to run on seriatim (per-policy) or group model points.
        Stochastic runs enforce GROUP_MP — this is validated at RunConfig level.

    policy_filter:
        Optional list of fund IDs to restrict the run to a subset of policies.
        When None, all policies in the input file are included.
        Useful for debugging a single fund or validating a specific block.
    """
    active_models: list[LiabilityModel] = Field(
        ...,
        min_length=1,
        description="Liability sub-models to run. At least one required."
    )
    input_mode:    LiabilityInputMode   = Field(
        default=LiabilityInputMode.GROUP_MP,
        description="Seriatim (per-policy) or group model point input."
    )
    policy_filter: Optional[list[str]]  = Field(
        default=None,
        description="Optional list of fund IDs to restrict run scope."
    )

    @field_validator("active_models")
    @classmethod
    def no_duplicate_models(cls, v: list[LiabilityModel]) -> list[LiabilityModel]:
        if len(v) != len(set(v)):
            raise ValueError("active_models contains duplicates.")
        return v


class StochasticConfig(BaseModel):
    """
    Parameters specific to stochastic runs. Ignored for other run types.

    num_scenarios:
        Number of scenarios to draw from the ESG file.
        Must be <= the number of scenarios available in the file.
        The file-level check cannot be done here (file not yet loaded)
        — it is enforced in ScenarioLoader at runtime.
        Typical values: 1,000 – 3,000.

    scenario_seed:
        Optional random seed for reproducibility when sampling scenarios.
        When None, scenarios are drawn in file order (first N scenarios).
        When set, scenarios are sampled randomly with this seed.

    parallel_scenarios:
        Whether to run scenarios in parallel using multiprocessing.
        Set to False for debugging or on machines with limited cores.
        Recommended: True for production runs.

    num_workers:
        Number of parallel worker processes.
        When None, defaults to (CPU count - 1).
        Only relevant when parallel_scenarios is True.
    """
    num_scenarios:      int            = Field(
        ...,
        ge=1,
        le=10_000,
        description="Number of ESG scenarios to run. Typical: 1,000–3,000."
    )
    scenario_seed:      Optional[int]  = Field(
        default=None,
        description="Random seed for scenario sampling. None = file order."
    )
    parallel_scenarios: bool           = Field(
        default=True,
        description="Run scenarios in parallel. Disable for debugging."
    )
    num_workers:        Optional[int]  = Field(
        default=None,
        ge=1,
        description="Parallel worker count. None = CPU count minus one."
    )


class OutputConfig(BaseModel):
    """
    Controls what outputs are produced, at what granularity, and in what format.

    save_policy_level_results:
        Whether to write per-policy (or per-MP) outputs.
        Warning: for seriatim runs with large portfolios, this produces
        very large files. Default False for stochastic, True for others.

    save_fund_level_results:
        Whether to write fund-level aggregated results. Default True.

    save_company_level_results:
        Whether to write company-level aggregated results. Default True.

    output_timestep:
        Frequency at which results are aggregated and written to output files.
        Independent of projection_timestep — the model always computes at
        projection_timestep; this controls how rows are bucketed before writing.
        Must be >= projection_timestep in period length (validated at RunConfig).
        Default: MONTHLY (one row per projection month).
        Example: projection_timestep=monthly, output_timestep=annual →
                 model runs monthly but writes one aggregated row per year.

    output_horizon_years:
        Limit output to the first N years of the projection.
        When None, output covers the full projection_term_years.
        Must be <= projection_term_years (validated at RunConfig).
        Use this to keep output files manageable for long projections where
        only the near-term cash flows are of interest.
        Example: projection_term_years=100, output_horizon_years=30 →
                 model runs 100 years but writes only the first 30 years.

    result_format:
        Output file format. CSV is universally readable; Parquet is
        significantly faster to read/write for large stochastic outputs.

    compress_outputs:
        Whether to gzip output files. Recommended for stochastic runs
        where total output volume can be several GB.

    output_dir:
        Directory where all model outputs will be written.
        Created automatically if it does not exist.
    """
    save_policy_level_results:  bool           = Field(
        default=False,
        description="Write per-policy/MP results. Caution: large files."
    )
    save_fund_level_results:    bool           = Field(
        default=True,
        description="Write fund-level aggregated results."
    )
    save_company_level_results: bool           = Field(
        default=True,
        description="Write company-level aggregated results."
    )
    output_timestep:            OutputTimestep = Field(
        default=OutputTimestep.MONTHLY,
        description=(
            "Output aggregation frequency. Independent of projection_timestep. "
            "Must be >= projection_timestep. Default: monthly."
        )
    )
    output_horizon_years:       Optional[int]  = Field(
        default=None,
        ge=1,
        description=(
            "Limit output to first N years. None = full projection term. "
            "Must be <= projection_term_years."
        )
    )
    result_format:              ResultFormat   = Field(
        default=ResultFormat.CSV,
        description="Output format: 'csv' or 'parquet'."
    )
    compress_outputs:           bool           = Field(
        default=False,
        description="Gzip output files. Recommended for stochastic runs."
    )
    output_dir:                 Path           = Field(
        default=Path("outputs"),
        description="Output directory. Created automatically if absent."
    )


# ===========================================================================
# Master RunConfig
# The single object that fully describes one model execution.
# ===========================================================================

class RunConfig(BaseModel):
    """
    Master configuration for a single ALM model run.

    This is the object passed to the run mode orchestrator. It must be fully
    validated before any model component is instantiated. If RunConfig
    construction succeeds, the run is guaranteed to have all required inputs
    in consistent state.

    Fields:
        run_id:         Unique identifier for this run (UUID string).
                        Set by the API layer when the run is created.
                        Can be set manually for batch/CLI runs.

        run_name:       Human-readable label. Stored with results for
                        identification in the results database and reports.

        run_type:       Determines which run mode class is instantiated.

        projection:     Time dimension config (dates, term, timesteps).

        input_sources:  Data source locations for all model inputs.

        liability:      Liability sub-model selection and input mode.

        stochastic:     Stochastic-specific settings. Must be provided
                        when run_type is STOCHASTIC, ignored otherwise.

        output:         Controls what results are written and in what format,
                        including the output directory.

        notes:          Free-text field for run documentation. Stored with
                        results. Useful for audit trail and team communication.

    Cross-field validation (model_validator):
        - STOCHASTIC runs must provide stochastic config
        - STOCHASTIC runs must use GROUP_MP liability input mode
        - STOCHASTIC runs must provide a scenario_file_path
        - DETERMINISTIC and STOCHASTIC runs must provide asset_data_path
        - SERIATIM input mode is not permitted for STOCHASTIC runs
    """

    run_id:        str                    = Field(
        ...,
        min_length=1,
        description="Unique run identifier. Set by API or caller."
    )
    run_name:      str                    = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable run label for identification in results."
    )
    run_type:      RunType                = Field(
        ...,
        description="Execution mode: liability_only | deterministic | stochastic."
    )
    projection:    ProjectionConfig       = Field(
        ...,
        description="Projection time dimension settings."
    )
    input_sources: InputSourcesConfig     = Field(
        ...,
        description="All model input data sources (model points, assumptions, assets, scenarios)."
    )
    liability:     LiabilityConfig        = Field(
        ...,
        description="Liability sub-model configuration."
    )
    stochastic:    Optional[StochasticConfig] = Field(
        default=None,
        description="Stochastic run parameters. Required when run_type=stochastic."
    )
    output:        OutputConfig           = Field(
        default_factory=OutputConfig,
        description="Output format, content controls, and output directory."
    )
    notes:         Optional[str]          = Field(
        default=None,
        max_length=2000,
        description="Free-text documentation for audit trail."
    )

    # -----------------------------------------------------------------------
    # Cross-field validation
    # Pydantic calls model_validator after all individual field validators
    # have passed. Use this for rules that span multiple fields.
    # -----------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_run_type_consistency(self) -> RunConfig:
        """
        Enforce rules that depend on run_type and multiple other fields.
        These cannot be expressed in individual field validators because
        they require access to more than one field simultaneously.
        """
        errors = []

        # --- Stochastic-specific rules ---
        if self.run_type == RunType.STOCHASTIC:

            # Must have stochastic config block
            if self.stochastic is None:
                errors.append(
                    "run_type='stochastic' requires a 'stochastic' config block "
                    "with num_scenarios specified."
                )

            # Stochastic must use group model points
            if self.liability.input_mode != LiabilityInputMode.GROUP_MP:
                errors.append(
                    "run_type='stochastic' requires liability.input_mode='group_mp'. "
                    "Seriatim input is computationally infeasible for stochastic runs. "
                    f"Received: '{self.liability.input_mode.value}'."
                )

            # Stochastic must have ESG file
            if self.input_sources.scenario_file_path is None:
                errors.append(
                    "run_type='stochastic' requires input_sources.scenario_file_path "
                    "to be specified."
                )

        # --- Asset model and fund config required for non-liability-only runs ---
        if self.run_type in (RunType.DETERMINISTIC, RunType.STOCHASTIC):
            if self.input_sources.asset_data_path is None:
                errors.append(
                    f"run_type='{self.run_type.value}' requires "
                    "input_sources.asset_data_path to be specified."
                )
            if self.input_sources.fund_config_path is None:
                errors.append(
                    f"run_type='{self.run_type.value}' requires "
                    "input_sources.fund_config_path to be specified."
                )

        # --- Stochastic config provided but run type is not stochastic ---
        # Not a hard error — user may be reusing a config template.
        # The stochastic block is silently ignored.

        # --- Output timestep must be >= projection timestep ---
        # You cannot aggregate output more finely than the model computes.
        _period_months: dict[str, int] = {"monthly": 1, "quarterly": 3, "annual": 12}
        proj_months   = _period_months[self.projection.projection_timestep.value]
        output_months = _period_months[self.output.output_timestep.value]
        if output_months < proj_months:
            errors.append(
                f"output.output_timestep ('{self.output.output_timestep.value}', "
                f"{output_months} month(s)) cannot be finer than "
                f"projection.projection_timestep ('{self.projection.projection_timestep.value}', "
                f"{proj_months} month(s)). "
                "You cannot write output more granularly than the model computes."
            )

        # --- Output horizon must not exceed projection term ---
        if self.output.output_horizon_years is not None:
            if self.output.output_horizon_years > self.projection.projection_term_years:
                errors.append(
                    f"output.output_horizon_years ({self.output.output_horizon_years}) "
                    f"exceeds projection.projection_term_years "
                    f"({self.projection.projection_term_years}). "
                    "Cannot write more years of output than are projected."
                )

        if errors:
            # Raise all errors together so the user sees every problem at once,
            # not one at a time (which forces repeated fix-and-retry cycles).
            raise ValueError(
                "RunConfig validation failed with the following errors:\n"
                + "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
            )

        return self

    # -----------------------------------------------------------------------
    # Convenience constructors
    # -----------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """
        Load and validate a RunConfig from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated RunConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML content fails Pydantic validation.
            ImportError: If PyYAML is not installed.

        Example:
            config = RunConfig.from_yaml("config_files/stochastic_run.yaml")
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load configs from YAML files. "
                "Install with: uv add pyyaml"
            ) from e

        yaml_path = Path(path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"RunConfig YAML file not found: {yaml_path}"
            )

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict) -> RunConfig:
        """
        Construct and validate a RunConfig from a plain dictionary.
        Useful for constructing configs programmatically (e.g. from the API
        layer or in tests).

        Args:
            data: Dictionary matching the RunConfig schema.

        Returns:
            Validated RunConfig instance.

        Example:
            config = RunConfig.from_dict({
                "run_id": "run_001",
                "run_name": "Q4 BEL Validation",
                "run_type": "liability_only",
                ...
            })
        """
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """
        Serialise this RunConfig to a YAML file.
        Paths are written as strings (YAML has no native Path type).
        Useful for saving run configs alongside their results for audit.

        Args:
            path: Destination file path.
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to save configs as YAML. "
                "Install with: uv add pyyaml"
            ) from e

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # model_dump() converts Pydantic model to dict.
        # mode="json" ensures dates/enums are serialisable.
        data = self.model_dump(mode="json")

        with open(out_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_json(self, path: str | Path) -> None:
        """
        Serialise this RunConfig to a JSON file.

        Args:
            path: Destination file path.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(self.model_dump_json(indent=2))

    def summary(self) -> str:
        """
        Return a compact human-readable summary of this run config.
        Used in logging and progress panels.
        """
        stoch_info = ""
        if self.run_type == RunType.STOCHASTIC and self.stochastic:
            stoch_info = f" | {self.stochastic.num_scenarios:,} scenarios"

        horizon = (
            f"{self.output.output_horizon_years} years"
            if self.output.output_horizon_years is not None
            else f"{self.projection.projection_term_years} years (full)"
        )

        return (
            f"[{self.run_id}] {self.run_name}\n"
            f"  Type       : {self.run_type.value}{stoch_info}\n"
            f"  Valuation  : {self.projection.valuation_date}\n"
            f"  Term       : {self.projection.projection_term_years} years\n"
            f"  Timesteps  : {self.projection.projection_timestep.value} CFs "
            f"/ {self.projection.decision_timestep.value} decisions\n"
            f"  Currency   : {self.projection.currency.value}\n"
            f"  Liability  : {[m.value for m in self.liability.active_models]} "
            f"({self.liability.input_mode.value})\n"
            f"  Output     : {self.output.output_timestep.value} / {horizon}\n"
            f"  Output dir : {self.output.output_dir}"
        )
