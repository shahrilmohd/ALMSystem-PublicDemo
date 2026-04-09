"""
AssetDataLoader — reads and validates bond/equity portfolio data for the ALM engine.

Purpose
-------
This loader is the bridge between raw input files (CSV, TSV) and the asset
engine.  It does four things:

    1. load()           — reads the file into a raw DataFrame; applies column mapping
    2. validate()       — runs AssetValidator on the mapped DataFrame
    3. to_dataframe()   — returns a clean, typed copy (BaseLoader contract)
    4. to_asset_model() — constructs Bond/Equity objects and returns an AssetModel

Asset types
-----------
An optional ``asset_type`` column controls what object is constructed per row:

    "bond"    → Bond object (default when column is absent)
    "equity"  → Equity object

Mixed portfolios (bonds + equities) are fully supported.

Column layout
-------------
Required for ALL rows:
    asset_id             str    Unique identifier
    initial_book_value   float  Carrying value at valuation date (> 0)
                                Bonds: carrying/purchase cost
                                Equities: initial market value

Required for BOND rows only (column must exist if any bond rows are present):
    face_value           float  Total par value held (>= 0)
    annual_coupon_rate   float  Annual coupon rate as decimal, e.g. 0.05 = 5%
    maturity_month       int    Month index when bond matures (>= 1)
    accounting_basis     str    AC, FVTPL, or FVOCI

Optional for BOND rows (blank/NaN → computed/defaulted):
    eir                  float  Annual EIR; blank → computed from initial_book_value
    calibration_spread   float  z-spread over risk-free curve; blank → 0.0

Optional for EQUITY rows:
    dividend_yield_yr    float  Annual dividend yield; blank → 0.0

Column mapping
--------------
    loader = AssetDataLoader(
        file_path="holdings.csv",
        column_map={"par_value": "face_value", "coupon": "annual_coupon_rate"},
    )

Supported file formats
----------------------
    .csv    comma-separated
    .tsv    tab-separated

Usage
-----
    loader = AssetDataLoader("portfolio.csv")
    loader.load()
    loader.validate()
    asset_model = loader.to_asset_model()
    run = DeterministicRun(config=config, model_points=mp,
                           asset_model=asset_model, ...)

Architecture note
-----------------
This class lives in data/, not engine/.  The engine never imports from data/.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from data.loaders.base_loader import BaseLoader
from data.validators.asset_validator import AssetValidator
from engine.asset.asset_model import AssetModel
from engine.asset.bond import Bond
from engine.asset.equity import Equity

# Columns that must be float for bond rows
_BOND_FLOAT_COLUMNS = (
    "face_value",
    "annual_coupon_rate",
    "initial_book_value",
)

# Columns that must be int for bond rows
_BOND_INT_COLUMNS = ("maturity_month",)

# Optional float columns — cast if present
_OPTIONAL_FLOAT_COLUMNS = ("eir", "calibration_spread", "dividend_yield_yr")


class AssetDataLoader(BaseLoader):
    """
    Loads and validates a bond/equity portfolio CSV/TSV file.

    Parameters
    ----------
    file_path : str or Path
        Path to the portfolio file (.csv or .tsv).
    column_map : dict, optional
        Source-to-engine column name mapping.
        Example: {"par_value": "face_value", "coupon": "annual_coupon_rate"}

    Usage
    -----
        loader = AssetDataLoader("portfolio.csv")
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()
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
        Read the portfolio file and apply the column map.

        Raises
        ------
        FileNotFoundError
            If file_path does not exist.
        ValueError
            If the file extension is not .csv or .tsv.
        """
        if not self._file_path.exists():
            raise FileNotFoundError(
                f"Asset portfolio file not found: {self._file_path}"
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
            "Loaded %d assets from %s", len(self._raw), self._file_path
        )

    # -----------------------------------------------------------------------
    # Step 2 — validate
    # -----------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate the mapped portfolio against all asset rules.

        Raises
        ------
        RuntimeError
            If called before load().
        ValueError
            If any validation rule is violated.
        """
        self._require_loaded("validate")
        AssetValidator.validate(self._raw)
        self._logger.info("Validation passed: %d assets", len(self._raw))

    # -----------------------------------------------------------------------
    # Step 3 — to_dataframe
    # -----------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return a clean, typed copy of the validated portfolio DataFrame.

        Bond columns are cast to float/int; optional columns cast if present.

        Returns
        -------
        pd.DataFrame
            Typed and validated portfolio, one row per asset.

        Raises
        ------
        RuntimeError
            If called before load().
        """
        self._require_loaded("to_dataframe")
        df = self._raw.copy()

        for col in _BOND_FLOAT_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in _BOND_INT_COLUMNS:
            if col in df.columns:
                # Use nullable integer to handle NaN in equity rows gracefully
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in _OPTIONAL_FLOAT_COLUMNS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # initial_book_value is required for all rows
        df["initial_book_value"] = pd.to_numeric(df["initial_book_value"], errors="coerce")

        return df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Step 4 — to_asset_model
    # -----------------------------------------------------------------------

    def to_asset_model(self) -> AssetModel:
        """
        Construct Bond/Equity objects from each row and return an AssetModel.

        Row dispatch
        ------------
        If ``asset_type`` column is present: "bond" → Bond, "equity" → Equity.
        If ``asset_type`` column is absent: all rows treated as Bond.

        EIR (bond rows)
        ---------------
        If ``eir`` is present and non-NaN for a row, it is passed directly to
        Bond (locked at purchase).  Otherwise eir=None is passed and Bond
        computes it from initial_book_value via Bond.calculate_eir().

        calibration_spread (bond rows)
        --------------------------------
        Blank/NaN → 0.0 (risk-free pricing).

        dividend_yield_yr (equity rows)
        ---------------------------------
        Blank/NaN → 0.0 (no dividend income).

        Returns
        -------
        AssetModel
            Ready for injection into a run mode.

        Raises
        ------
        RuntimeError
            If called before load().
        """
        self._require_loaded("to_asset_model")
        df = self.to_dataframe()

        assets = []
        has_asset_type = "asset_type" in df.columns

        for _, row in df.iterrows():
            asset_type = (
                str(row["asset_type"]).strip().lower()
                if has_asset_type
                else "bond"
            )

            if asset_type == "equity":
                assets.append(self._build_equity(row))
            else:
                assets.append(self._build_bond(row))

        asset_model = AssetModel(assets)
        self._logger.info(
            "AssetModel constructed: %d assets (%d bonds, %d equities)",
            len(asset_model),
            sum(1 for a in asset_model if a.asset_class == "bonds"),
            sum(1 for a in asset_model if a.asset_class == "equities"),
        )
        return asset_model

    # -----------------------------------------------------------------------
    # Internal — asset construction
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_bond(row: "pd.Series") -> Bond:
        """Construct a Bond from one DataFrame row."""
        eir: float | None = None
        if "eir" in row.index:
            raw_eir = row["eir"]
            if pd.notna(raw_eir):
                eir = float(raw_eir)

        calibration_spread: float = 0.0
        if "calibration_spread" in row.index:
            raw_cs = row["calibration_spread"]
            if pd.notna(raw_cs):
                calibration_spread = float(raw_cs)

        return Bond(
            asset_id=str(row["asset_id"]),
            face_value=float(row["face_value"]),
            annual_coupon_rate=float(row["annual_coupon_rate"]),
            maturity_month=int(row["maturity_month"]),
            accounting_basis=str(row["accounting_basis"]),
            initial_book_value=float(row["initial_book_value"]),
            eir=eir,
            calibration_spread=calibration_spread,
        )

    @staticmethod
    def _build_equity(row: "pd.Series") -> Equity:
        """Construct an Equity from one DataFrame row."""
        dividend_yield_yr: float = 0.0
        if "dividend_yield_yr" in row.index:
            raw_dy = row["dividend_yield_yr"]
            if pd.notna(raw_dy):
                dividend_yield_yr = float(raw_dy)

        return Equity(
            asset_id=str(row["asset_id"]),
            initial_market_value=float(row["initial_book_value"]),
            dividend_yield_yr=dividend_yield_yr,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _strip_strings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from all string (object) columns."""
        str_cols = df.select_dtypes(include=["object", "str"]).columns
        df = df.copy()
        for col in str_cols:
            df[col] = df[col].str.strip()
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the column_map to rename source columns to engine-standard names."""
        if not self._column_map:
            return df
        return df.rename(columns=self._column_map)
