"""
LiabilityValidator — validates model point DataFrames for the Conventional model.

Validation rules
----------------
All rules are checked in a single pass.  Every violation is collected before
raising, so the error message lists all problems at once instead of one at a time.

Required columns (must all be present):
    group_id                 str    Policy group identifier
    in_force_count           float  Number of policies in force (>= 0)
    sum_assured              float  Sum assured per policy (> 0)
    annual_premium           float  Annual premium per policy (>= 0)
    attained_age             int    Attained age in whole years (>= 0)
    policy_code              str    Must be one of: ENDOW_NONPAR, ENDOW_PAR, TERM
    policy_term_yr           int    Policy term in years (>= 1)
    policy_duration_mths     int    Duration in force in months (>= 0)
    accrued_bonus_per_policy float  Accrued bonus per policy (>= 0)

Range rules:
    in_force_count           >= 0
    sum_assured              > 0
    annual_premium           >= 0
    attained_age             >= 0
    policy_term_yr           >= 1
    policy_duration_mths     >= 0
    accrued_bonus_per_policy >= 0

Consistency rules:
    policy_duration_mths < policy_term_yr * 12
        A policy cannot already be at or past its maturity date at valuation.

policy_code rules:
    Must be exactly one of: "ENDOW_NONPAR", "ENDOW_PAR", "TERM"

Empty DataFrame:
    Raises ValueError — a run with no policies is not a valid input.
"""
from __future__ import annotations

import pandas as pd

_VALID_POLICY_CODES = frozenset({"ENDOW_NONPAR", "ENDOW_PAR", "TERM"})

REQUIRED_COLUMNS: tuple[str, ...] = (
    "group_id",
    "in_force_count",
    "sum_assured",
    "annual_premium",
    "attained_age",
    "policy_code",
    "policy_term_yr",
    "policy_duration_mths",
    "accrued_bonus_per_policy",
)


class LiabilityValidator:
    """
    Stateless validator for liability model point DataFrames.

    All methods are class methods — no instance is needed.  Call:

        LiabilityValidator.validate(df)

    Raises ValueError listing all violations found, or returns None if valid.
    """

    @classmethod
    def validate(cls, df: pd.DataFrame) -> None:
        """
        Run all validation rules against df.

        Parameters
        ----------
        df : pd.DataFrame
            Model point DataFrame, as returned by the loader after column
            mapping.  Column names must already match REQUIRED_COLUMNS exactly.

        Raises
        ------
        ValueError
            If any rule is violated.  The message lists every violation found,
            separated by newlines.
        """
        errors: list[str] = []

        # 1. Required columns present
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {missing}")
            # Cannot check values if columns are absent — stop early
            raise ValueError("\n".join(errors))

        # 2. Empty DataFrame
        if df.empty:
            raise ValueError(
                "model_points is empty — no policies to project."
            )

        # 3. Range checks (vectorised across all rows at once)
        cls._check_non_negative(df, "in_force_count", errors)
        cls._check_positive(df, "sum_assured", errors)
        cls._check_non_negative(df, "annual_premium", errors)
        cls._check_non_negative(df, "attained_age", errors)
        cls._check_min(df, "policy_term_yr", 1, errors)
        cls._check_non_negative(df, "policy_duration_mths", errors)
        cls._check_non_negative(df, "accrued_bonus_per_policy", errors)

        # 4. policy_code values
        invalid_codes = df.loc[
            ~df["policy_code"].isin(_VALID_POLICY_CODES), "policy_code"
        ].unique()
        if len(invalid_codes) > 0:
            errors.append(
                f"Invalid policy_code values: {sorted(invalid_codes)}. "
                f"Allowed: {sorted(_VALID_POLICY_CODES)}"
            )

        # 5. Consistency: duration must be strictly less than full term
        term_mths = df["policy_term_yr"] * 12
        expired = df["policy_duration_mths"] >= term_mths
        if expired.any():
            count = int(expired.sum())
            errors.append(
                f"{count} row(s) have policy_duration_mths >= policy_term_yr * 12. "
                "Policies that have already matured cannot be projected."
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
    def _check_min(
        df: pd.DataFrame, col: str, minimum: int, errors: list[str]
    ) -> None:
        """Append an error if any value in col is below minimum."""
        bad = int((df[col] < minimum).sum())
        if bad:
            errors.append(
                f"'{col}': {bad} value(s) below minimum {minimum}."
            )
