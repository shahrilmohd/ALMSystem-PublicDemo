"""
AssetValidator — validates bond/equity portfolio DataFrames for the asset data loader.

Validation rules
----------------
All rules are checked in a single pass.  Every violation is collected before
raising, so the error message lists all problems at once.

Column layout
-------------
The portfolio CSV may contain bonds only, equities only, or a mixture.
An optional ``asset_type`` column ("bond" or "equity") controls which
per-row rules apply.  When the column is absent all rows are treated as bonds
(backwards-compatible with files that pre-date equity support).

Columns required for ALL rows:
    asset_id             str    Unique, non-empty identifier
    initial_book_value   float  Carrying value / initial MV at valuation date (> 0)

Column required when any bond rows are present:
    asset_type           str    "bond" or "equity" (optional column, default "bond")
    face_value           float  Total par value held (>= 0)
    annual_coupon_rate   float  Annual coupon rate as decimal ([0, 1])
    maturity_month       int    Month index when bond matures (>= 1)
    accounting_basis     str    Must be one of: AC, FVTPL, FVOCI

Optional columns (validated if present, silently ignored if absent):
    eir                  float  Annual EIR; NaN rows skipped (range (0, 5])
    calibration_spread   float  z-spread over risk-free curve (>= 0)  [bonds only]
    dividend_yield_yr    float  Annual dividend yield [0, 1]           [equities only]

Uniqueness rule:
    asset_id values must be unique across all rows.

Empty DataFrame:
    Raises ValueError — a portfolio with no assets is not a valid input.
"""
from __future__ import annotations

import pandas as pd

_VALID_ACCOUNTING_BASES = frozenset({"AC", "FVTPL", "FVOCI"})
_VALID_ASSET_TYPES      = frozenset({"bond", "equity"})

# Columns required for every row (regardless of asset type)
COMMON_REQUIRED_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "initial_book_value",
)

# Columns required for bond rows (must be present in the file if any bond rows exist)
BOND_REQUIRED_COLUMNS: tuple[str, ...] = (
    "face_value",
    "annual_coupon_rate",
    "maturity_month",
    "accounting_basis",
)

# Kept for backwards compatibility — equivalent to COMMON + BOND when no equity rows
REQUIRED_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "face_value",
    "annual_coupon_rate",
    "maturity_month",
    "accounting_basis",
    "initial_book_value",
)


class AssetValidator:
    """
    Stateless validator for bond/equity portfolio DataFrames.

    All methods are class methods — no instance is needed.  Call:

        AssetValidator.validate(df)

    Raises ValueError listing all violations found, or returns None if valid.
    """

    @classmethod
    def validate(cls, df: pd.DataFrame) -> None:
        """
        Run all validation rules against df.

        Parameters
        ----------
        df : pd.DataFrame
            Portfolio DataFrame, as returned by the loader after column mapping.
            Column names must already match the engine-standard names exactly.

        Raises
        ------
        ValueError
            If any rule is violated.  The message lists every violation found,
            separated by newlines.
        """
        errors: list[str] = []

        # ---- 1. Common required columns ----
        missing_common = [c for c in COMMON_REQUIRED_COLUMNS if c not in df.columns]
        if missing_common:
            raise ValueError(f"Missing required columns: {missing_common}")

        # ---- 2. Empty DataFrame ----
        if df.empty:
            raise ValueError("asset_portfolio is empty — no assets to load.")

        # ---- 3. Determine bond vs equity masks ----
        if "asset_type" in df.columns:
            # Validate asset_type values first
            invalid_types = df.loc[
                ~df["asset_type"].isin(_VALID_ASSET_TYPES), "asset_type"
            ].unique()
            if len(invalid_types) > 0:
                errors.append(
                    f"Invalid asset_type values: {sorted(str(v) for v in invalid_types)}. "
                    f"Allowed: {sorted(_VALID_ASSET_TYPES)}"
                )
            bond_mask   = df["asset_type"] == "bond"
            equity_mask = df["asset_type"] == "equity"
        else:
            # No asset_type column — treat all rows as bonds (backwards compatible)
            bond_mask   = pd.Series([True]  * len(df), index=df.index)
            equity_mask = pd.Series([False] * len(df), index=df.index)

        # ---- 4. Common column validation (all rows) ----
        cls._check_positive(df, "initial_book_value", errors)

        # ---- 5. asset_id uniqueness and non-empty ----
        dup_ids = df["asset_id"][df["asset_id"].duplicated()].unique()
        if len(dup_ids) > 0:
            errors.append(
                f"Duplicate asset_id values: {sorted(str(x) for x in dup_ids)}. "
                "Each asset_id must be unique."
            )
        empty_ids = int((df["asset_id"].astype(str).str.strip() == "").sum())
        if empty_ids:
            errors.append(
                f"asset_id: {empty_ids} empty or whitespace-only value(s). "
                "asset_id must be a non-empty string."
            )

        # ---- 6. Bond-specific column checks ----
        bond_df = df[bond_mask]
        if len(bond_df) > 0:
            # Bond-required columns must be present in the file
            missing_bond = [c for c in BOND_REQUIRED_COLUMNS if c not in df.columns]
            if missing_bond:
                errors.append(
                    f"Missing bond-required columns: {missing_bond}. "
                    "These columns must be present when the portfolio contains bonds."
                )
            else:
                # Validate bond column values
                cls._check_non_negative_subset(bond_df, "face_value", "bond", errors)
                cls._check_range_subset(
                    bond_df, "annual_coupon_rate", 0.0, 1.0, "bond", errors
                )
                cls._check_min_subset(bond_df, "maturity_month", 1, "bond", errors)

                invalid_bases = bond_df.loc[
                    ~bond_df["accounting_basis"].isin(_VALID_ACCOUNTING_BASES),
                    "accounting_basis",
                ].unique()
                if len(invalid_bases) > 0:
                    errors.append(
                        f"Invalid accounting_basis values (bond rows): "
                        f"{sorted(str(v) for v in invalid_bases)}. "
                        f"Allowed: {sorted(_VALID_ACCOUNTING_BASES)}"
                    )

        # ---- 7. Optional: eir (bond rows only) ----
        if "eir" in df.columns and len(bond_df) > 0:
            eir_present = bond_df["eir"].dropna()
            if len(eir_present) > 0:
                bad_eir = int(((eir_present <= 0.0) | (eir_present > 5.0)).sum())
                if bad_eir:
                    errors.append(
                        f"eir: {bad_eir} value(s) outside (0, 5] in bond rows. "
                        "EIR must be a positive annual rate (e.g. 0.069 = 6.9%). "
                        "Leave blank/NaN for rows where EIR should be computed."
                    )

        # ---- 8. Optional: calibration_spread (bond rows only) ----
        if "calibration_spread" in df.columns and len(bond_df) > 0:
            cs = bond_df["calibration_spread"].dropna()
            if len(cs) > 0:
                bad_cs = int((cs < 0.0).sum())
                if bad_cs:
                    errors.append(
                        f"calibration_spread: {bad_cs} negative value(s) in bond rows. "
                        "Calibration spread must be >= 0."
                    )

        # ---- 9. Optional: dividend_yield_yr (equity rows only) ----
        equity_df = df[equity_mask]
        if "dividend_yield_yr" in df.columns and len(equity_df) > 0:
            dy = equity_df["dividend_yield_yr"].dropna()
            if len(dy) > 0:
                bad_dy = int(((dy < 0.0) | (dy > 1.0)).sum())
                if bad_dy:
                    errors.append(
                        f"dividend_yield_yr: {bad_dy} value(s) outside [0, 1] "
                        "in equity rows."
                    )

        if errors:
            raise ValueError("\n".join(errors))

    # -----------------------------------------------------------------------
    # Private range-check helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _check_non_negative(
        df: pd.DataFrame, col: str, errors: list[str]
    ) -> None:
        """Append an error if any value in col is negative."""
        bad = int((df[col] < 0).sum())
        if bad:
            errors.append(
                f"'{col}': {bad} negative value(s). All values must be >= 0."
            )

    @staticmethod
    def _check_positive(
        df: pd.DataFrame, col: str, errors: list[str]
    ) -> None:
        """Append an error if any value in col is <= 0."""
        bad = int((df[col] <= 0).sum())
        if bad:
            errors.append(
                f"'{col}': {bad} non-positive value(s). All values must be > 0."
            )

    @staticmethod
    def _check_non_negative_subset(
        df: pd.DataFrame, col: str, row_type: str, errors: list[str]
    ) -> None:
        bad = int((df[col] < 0).sum())
        if bad:
            errors.append(
                f"'{col}': {bad} negative value(s) in {row_type} rows. "
                "All values must be >= 0."
            )

    @staticmethod
    def _check_range_subset(
        df: pd.DataFrame,
        col: str,
        lo: float,
        hi: float,
        row_type: str,
        errors: list[str],
    ) -> None:
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        if bad:
            errors.append(
                f"'{col}': {bad} value(s) outside [{lo}, {hi}] in {row_type} rows."
            )

    @staticmethod
    def _check_min_subset(
        df: pd.DataFrame, col: str, minimum: int, row_type: str, errors: list[str]
    ) -> None:
        bad = int((df[col] < minimum).sum())
        if bad:
            errors.append(
                f"'{col}': {bad} value(s) below minimum {minimum} in {row_type} rows."
            )

    # -----------------------------------------------------------------------
    # Legacy helpers (kept for any external callers; delegate to new form)
    # -----------------------------------------------------------------------

    @staticmethod
    def _check_range(
        df: pd.DataFrame,
        col: str,
        lo: float,
        hi: float,
        errors: list[str],
    ) -> None:
        bad = int(((df[col] < lo) | (df[col] > hi)).sum())
        if bad:
            errors.append(
                f"'{col}': {bad} value(s) outside [{lo}, {hi}]."
            )

    @staticmethod
    def _check_min(
        df: pd.DataFrame, col: str, minimum: int, errors: list[str]
    ) -> None:
        bad = int((df[col] < minimum).sum())
        if bad:
            errors.append(
                f"'{col}': {bad} value(s) below minimum {minimum}."
            )
