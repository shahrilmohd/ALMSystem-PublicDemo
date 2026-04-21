"""
LiabilityDataLoader — reads and validates model point data for the liability engine.

Purpose
-------
This loader is the bridge between raw input files (CSV, TSV) and the
LiabilityOnlyRun engine.  It does three things:

    1. load()         — reads the file into a raw DataFrame; applies column mapping
    2. validate()     — runs LiabilityValidator on the mapped DataFrame
    3. to_dataframe() — returns a clean, typed copy ready for engine injection

Column mapping
--------------
Source files from different systems use different column names.  For example,
a legacy system might export "if_count" where the engine expects "in_force_count".
The column_map constructor argument handles this:

    loader = LiabilityDataLoader(
        file_path="model_points.csv",
        column_map={"if_count": "in_force_count", "sa": "sum_assured"},
    )

Mapping is applied by _rename_columns() immediately after reading, before any
validation.  Columns not listed in column_map are passed through unchanged.

Supported file formats
----------------------
    .csv    comma-separated
    .tsv    tab-separated

Type coercion
-------------
to_dataframe() casts columns to their required engine types:
    float : in_force_count, sum_assured, annual_premium, accrued_bonus_per_policy
    int   : attained_age, policy_term_yr, policy_duration_mths
    str   : group_id, policy_code (whitespace stripped)

Architecture note
-----------------
This class lives in data/, not engine/.  The engine never imports from data/.
Data flows in one direction only:

    data/loaders → caller → engine/run_modes

Injection happens at the call site (CLI, API, test), not here:

    loader = LiabilityDataLoader(file_path=path)
    loader.load()
    loader.validate()
    mp = loader.to_dataframe()
    run = LiabilityOnlyRun(config=config, model_points=mp, ...)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from data.loaders.base_loader import BaseLoader
from data.validators.liability_validator import LiabilityValidator

# Columns that must be float in the engine
_FLOAT_COLUMNS = (
    "in_force_count",
    "sum_assured",
    "annual_premium",
    "accrued_bonus_per_policy",
)

# Columns that must be int in the engine
_INT_COLUMNS = (
    "attained_age",
    "policy_term_yr",
    "policy_duration_mths",
)


class LiabilityDataLoader(BaseLoader):
    """
    Loads and validates a model point CSV/TSV file for the Conventional model.

    Parameters
    ----------
    file_path : str or Path
        Path to the model point file (.csv or .tsv).
    column_map : dict, optional
        Source-to-engine column name mapping.
        Example: {"if_count": "in_force_count", "sa": "sum_assured"}
        Columns not listed pass through unchanged.

    Usage
    -----
        loader = LiabilityDataLoader("model_points.csv")
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        run = LiabilityOnlyRun(config=config, model_points=df, ...)
    """

    def __init__(
        self,
        file_path: str | Path,
        column_map: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self._file_path  = Path(file_path)
        self._column_map = column_map or {}

    # -----------------------------------------------------------------------
    # Step 1 — load
    # -----------------------------------------------------------------------

    def load(self) -> None:
        """
        Read the model point file and apply the column map.

        After this call, self._raw contains the raw data with engine-standard
        column names (mapping already applied).

        Raises
        ------
        FileNotFoundError
            If file_path does not exist.
        ValueError
            If the file extension is not .csv or .tsv.
        """
        if not self._file_path.exists():
            raise FileNotFoundError(
                f"Model point file not found: {self._file_path}"
            )

        suffix = self._file_path.suffix.lower()
        if suffix == ".csv":
            sep = ","
        elif suffix == ".tsv":
            sep = "\t"
        else:
            raise ValueError(
                f"Unsupported file format '{suffix}'. Use .csv or .tsv."
            )

        raw = pd.read_csv(self._file_path, sep=sep)
        raw = self._rename_columns(raw)
        self._raw = self._strip_strings(raw)

        self._logger.info(
            "Loaded %d rows from %s", len(self._raw), self._file_path
        )

    # -----------------------------------------------------------------------
    # Step 2 — validate
    # -----------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate the mapped model points against all liability rules.

        Delegates to LiabilityValidator.validate().  If any rule fails,
        a ValueError is raised listing every violation found.

        Raises
        ------
        RuntimeError
            If called before load().
        ValueError
            If any validation rule is violated.
        """
        self._require_loaded("validate")
        assert self._raw is not None
        LiabilityValidator.validate(self._raw)
        self._logger.info("Validation passed: %d rows", len(self._raw))

    # -----------------------------------------------------------------------
    # Step 3 — to_dataframe
    # -----------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a clean, typed copy of the validated model point DataFrame.

        Casts each column to the type required by the engine (float, int, str).
        Strips leading/trailing whitespace from string columns.

        Returns
        -------
        pd.DataFrame
            Typed and validated model points, ready for LiabilityOnlyRun.

        Raises
        ------
        RuntimeError
            If called before load().
        """
        self._require_loaded("to_dataframe")
        assert self._raw is not None
        df = self._raw.copy()

        for col in _FLOAT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(float)

        for col in _INT_COLUMNS:
            if col in df.columns:
                df[col] = df[col].astype(int)

        return df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Internal helper — column mapping
    # -----------------------------------------------------------------------

    def _strip_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strip leading/trailing whitespace from all string (object) columns.

        Applied immediately after reading and renaming, so the validator and
        to_dataframe() both receive clean string values.
        """
        str_cols = df.select_dtypes(include=["object", "str"]).columns
        df = df.copy()
        for col in str_cols:
            df[col] = df[col].str.strip()
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the column_map to rename source columns to engine-standard names.

        Only columns listed in column_map are renamed.  All other columns
        are passed through unchanged.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame as read from the file.

        Returns
        -------
        pd.DataFrame
            Same data with columns renamed per column_map.
        """
        if not self._column_map:
            return df
        return df.rename(columns=self._column_map)
