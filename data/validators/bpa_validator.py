"""
BPAValidator — validates BPA model point DataFrames before engine injection.

Four BPA population types are validated independently:

    in_payment  — InPaymentLiability model points
    deferred    — DeferredLiability model points
    dependant   — DependantLiability model points
    enhanced    — EnhancedLiability model points (superset of in_payment)

Each has its own class method.  All rules are collected before raising so
the error message lists every violation at once.

Required columns and rules (DECISIONS.md §44, §45)
---------------------------------------------------
All population types:
    mp_id       : any
    deal_id     : str, non-empty, no surrounding whitespace,
                  must exist in registry when registry is supplied
    sex / member_sex / dependant_sex : "M" or "F"
    age / member_age / dependant_age : float, >= 0
    weight      : float, >= 0

in_payment / enhanced:
    pension_pa  : float, > 0
    lpi_cap     : float, >= 0, <= 1
    lpi_floor   : float, >= 0, <= lpi_cap
    gmp_pa      : float, >= 0
    enhanced only:
    rating_years: float, >= 0

deferred:
    deferred_pension_pa  : float, > 0
    era                  : float, >= 0
    nra                  : float, > era
    revaluation_type     : "CPI", "RPI", or "fixed"
    revaluation_cap      : float, >= 0
    revaluation_floor    : float, >= 0, <= revaluation_cap
    deferment_years      : float, >= 0
    tv_eligible          : int, values in {0, 1}, no nulls

dependant:
    member_sex     : "M" or "F"
    member_age     : float, >= 0
    dependant_sex  : "M" or "F"
    dependant_age  : float, >= 0
    pension_pa     : float, > 0
    lpi_cap        : float, >= 0
    lpi_floor      : float, >= 0, <= lpi_cap

Optional (all population types):
    tranche_id  : str — if the column is present, every row must have a
                  non-null, non-blank value (no partial population).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from engine.liability.bpa.registry import BPADealRegistry

_VALID_SEX         = frozenset({"M", "F"})
_VALID_REVALUATION = frozenset({"CPI", "RPI", "fixed"})
_VALID_TV_ELIGIBLE = frozenset({0, 1})

# Required column sets for each population type
_COMMON_COLS: tuple[str, ...] = ("mp_id", "deal_id")

_IN_PAYMENT_COLS: tuple[str, ...] = _COMMON_COLS + (
    "sex", "age", "in_force_count",
    "pension_pa", "lpi_cap", "lpi_floor", "gmp_pa",
)
_DEFERRED_COLS: tuple[str, ...] = _COMMON_COLS + (
    "sex", "age", "in_force_count",
    "deferred_pension_pa", "era", "nra",
    "revaluation_type", "revaluation_cap", "revaluation_floor",
    "deferment_years", "tv_eligible",
)
_DEPENDANT_COLS: tuple[str, ...] = _COMMON_COLS + (
    "member_sex", "member_age", "dependant_sex", "dependant_age",
    "weight", "pension_pa", "lpi_cap", "lpi_floor",
)
_ENHANCED_EXTRA: tuple[str, ...] = ("rating_years",)


class BPAValidator:
    """
    Stateless validator for BPA model point DataFrames.

    Usage without registry (unit tests / isolated validation):
        BPAValidator.validate_in_payment(df)

    Usage with registry (production — cross-references deal_id values):
        BPAValidator.validate_in_payment(df, registry=registry)
    """

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    @classmethod
    def validate_in_payment(
        cls,
        df: pd.DataFrame,
        registry: "BPADealRegistry | None" = None,
    ) -> None:
        """Validate in-payment pensioner model points."""
        errors: list[str] = []
        cls._check_columns(df, _IN_PAYMENT_COLS, errors)
        if errors:
            raise ValueError("BPA in-payment validation failed:\n" + "\n".join(errors))

        cls._check_deal_id    (df, registry, errors)
        cls._check_non_negative(df, "age",             errors)
        cls._check_non_negative(df, "in_force_count",  errors)
        cls._check_positive    (df, "pension_pa",      errors)
        cls._check_non_negative(df, "gmp_pa",          errors)
        cls._check_sex         (df, "sex",             errors)
        cls._check_lpi         (df, "lpi_cap", "lpi_floor", errors)
        cls._check_tranche_id  (df, errors)

        if errors:
            raise ValueError("BPA in-payment validation failed:\n" + "\n".join(errors))

    @classmethod
    def validate_enhanced(
        cls,
        df: pd.DataFrame,
        registry: "BPADealRegistry | None" = None,
    ) -> None:
        """Validate enhanced (impaired) life model points."""
        errors: list[str] = []
        cls._check_columns(df, _IN_PAYMENT_COLS + _ENHANCED_EXTRA, errors)
        if errors:
            raise ValueError("BPA enhanced validation failed:\n" + "\n".join(errors))

        cls._check_deal_id    (df, registry, errors)
        cls._check_non_negative(df, "age",             errors)
        cls._check_non_negative(df, "in_force_count",  errors)
        cls._check_positive    (df, "pension_pa",      errors)
        cls._check_non_negative(df, "gmp_pa",          errors)
        cls._check_non_negative(df, "rating_years",    errors)
        cls._check_sex         (df, "sex",             errors)
        cls._check_lpi         (df, "lpi_cap", "lpi_floor", errors)
        cls._check_tranche_id  (df, errors)

        if errors:
            raise ValueError("BPA enhanced validation failed:\n" + "\n".join(errors))

    @classmethod
    def validate_deferred(
        cls,
        df: pd.DataFrame,
        registry: "BPADealRegistry | None" = None,
    ) -> None:
        """Validate deferred member model points."""
        errors: list[str] = []
        cls._check_columns(df, _DEFERRED_COLS, errors)
        if errors:
            raise ValueError("BPA deferred validation failed:\n" + "\n".join(errors))

        cls._check_deal_id    (df, registry, errors)
        cls._check_non_negative(df, "age",                 errors)
        cls._check_non_negative(df, "in_force_count",      errors)
        cls._check_positive    (df, "deferred_pension_pa", errors)
        cls._check_non_negative(df, "era",                 errors)
        cls._check_non_negative(df, "deferment_years",     errors)
        cls._check_sex         (df, "sex",                 errors)
        cls._check_tv_eligible (df, errors)
        cls._check_tranche_id  (df, errors)

        # nra must be > era (row by row)
        bad_nra = df["nra"] <= df["era"]
        if bad_nra.any():
            errors.append(f"nra: {int(bad_nra.sum())} row(s) have nra <= era.")

        # revaluation_type values
        bad_rv = ~df["revaluation_type"].isin(_VALID_REVALUATION)
        if bad_rv.any():
            errors.append(
                f"revaluation_type: invalid values "
                f"{sorted(df.loc[bad_rv, 'revaluation_type'].unique())}. "
                f"Allowed: {sorted(_VALID_REVALUATION)}"
            )

        cls._check_non_negative(df, "revaluation_floor", errors)
        cls._check_non_negative(df, "revaluation_cap",   errors)
        bad_cap = df["revaluation_cap"] < df["revaluation_floor"]
        if bad_cap.any():
            errors.append(
                f"revaluation_cap: {int(bad_cap.sum())} row(s) have cap < floor."
            )

        if errors:
            raise ValueError("BPA deferred validation failed:\n" + "\n".join(errors))

    @classmethod
    def validate_dependant(
        cls,
        df: pd.DataFrame,
        registry: "BPADealRegistry | None" = None,
    ) -> None:
        """Validate dependant pension model points."""
        errors: list[str] = []
        cls._check_columns(df, _DEPENDANT_COLS, errors)
        if errors:
            raise ValueError("BPA dependant validation failed:\n" + "\n".join(errors))

        cls._check_deal_id    (df, registry, errors)
        cls._check_non_negative(df, "member_age",    errors)
        cls._check_non_negative(df, "dependant_age", errors)
        cls._check_non_negative(df, "weight",        errors)
        cls._check_positive    (df, "pension_pa",    errors)
        cls._check_sex         (df, "member_sex",    errors)
        cls._check_sex         (df, "dependant_sex", errors)
        cls._check_lpi         (df, "lpi_cap", "lpi_floor", errors)
        cls._check_tranche_id  (df, errors)

        if errors:
            raise ValueError("BPA dependant validation failed:\n" + "\n".join(errors))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_columns(
        df: pd.DataFrame, required: tuple[str, ...], errors: list[str]
    ) -> None:
        missing = [c for c in required if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")

    @staticmethod
    def _check_deal_id(
        df: pd.DataFrame,
        registry: "BPADealRegistry | None",
        errors: list[str],
    ) -> None:
        """deal_id must be non-empty; when a registry is supplied, must exist in it."""
        if "deal_id" not in df.columns:
            return  # already caught by _check_columns

        blank = df["deal_id"].isnull() | (df["deal_id"].astype(str).str.strip() == "")
        if blank.any():
            errors.append(
                f"deal_id: {int(blank.sum())} row(s) have blank or null deal_id."
            )
            return  # no point cross-referencing blanks

        if registry is not None:
            unknown = df["deal_id"].astype(str).str.strip()
            unknown = unknown[~unknown.isin(registry.all_deal_ids())]
            if not unknown.empty:
                errors.append(
                    f"deal_id: {len(unknown)} row(s) have deal_id values not in "
                    f"the registry: {sorted(unknown.unique())}. "
                    f"Registered deals: {registry.all_deal_ids()}"
                )

    @staticmethod
    def _check_tv_eligible(df: pd.DataFrame, errors: list[str]) -> None:
        """tv_eligible must be present with values in {0, 1} and no nulls."""
        if "tv_eligible" not in df.columns:
            return  # already caught by _check_columns

        nulls = df["tv_eligible"].isnull()
        if nulls.any():
            errors.append(
                f"tv_eligible: {int(nulls.sum())} null value(s). Must be 0 or 1."
            )
            return

        bad = ~df["tv_eligible"].isin(_VALID_TV_ELIGIBLE)
        if bad.any():
            errors.append(
                f"tv_eligible: {int(bad.sum())} invalid value(s) "
                f"{sorted(df.loc[bad, 'tv_eligible'].unique())}. Must be 0 or 1."
            )

    @staticmethod
    def _check_tranche_id(df: pd.DataFrame, errors: list[str]) -> None:
        """
        tranche_id is optional — but if the column is present, every row must
        have a non-null, non-blank value (no partial population permitted).
        """
        if "tranche_id" not in df.columns:
            return  # column absent — valid, no tranche granularity for this file

        blank = df["tranche_id"].isnull() | (df["tranche_id"].astype(str).str.strip() == "")
        if blank.any():
            errors.append(
                f"tranche_id: column is present but {int(blank.sum())} row(s) "
                "have blank or null values. Either populate all rows or remove "
                "the column entirely."
            )

    @staticmethod
    def _check_non_negative(
        df: pd.DataFrame, col: str, errors: list[str]
    ) -> None:
        bad = int((df[col] < 0).sum())
        if bad:
            errors.append(f"'{col}': {bad} negative value(s). Must be >= 0.")

    @staticmethod
    def _check_positive(
        df: pd.DataFrame, col: str, errors: list[str]
    ) -> None:
        bad = int((df[col] <= 0).sum())
        if bad:
            errors.append(f"'{col}': {bad} non-positive value(s). Must be > 0.")

    @staticmethod
    def _check_sex(
        df: pd.DataFrame, col: str, errors: list[str]
    ) -> None:
        bad = ~df[col].isin(_VALID_SEX)
        if bad.any():
            errors.append(
                f"'{col}': invalid values "
                f"{sorted(df.loc[bad, col].unique())}. Allowed: {sorted(_VALID_SEX)}"
            )

    @staticmethod
    def _check_lpi(
        df: pd.DataFrame, cap_col: str, floor_col: str, errors: list[str]
    ) -> None:
        bad_cap = df[cap_col] < 0
        if bad_cap.any():
            errors.append(f"'{cap_col}': {int(bad_cap.sum())} negative value(s).")
        bad_floor = df[floor_col] < 0
        if bad_floor.any():
            errors.append(f"'{floor_col}': {int(bad_floor.sum())} negative value(s).")
        bad_order = df[floor_col] > df[cap_col]
        if bad_order.any():
            errors.append(
                f"'{floor_col}' > '{cap_col}': {int(bad_order.sum())} row(s). "
                "Floor must not exceed cap."
            )
