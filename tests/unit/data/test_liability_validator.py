"""
Unit tests for LiabilityValidator.

Each test class covers one rule category.  Every test exercises exactly one
rule so failures pinpoint the broken check precisely.

Hand-calculated validation expectations
----------------------------------------
The fixture _valid_df() produces a single-row DataFrame that satisfies every
rule.  Individual tests mutate one field at a time to trigger exactly one
violation.
"""
from __future__ import annotations

import pandas as pd
import pytest

from data.validators.liability_validator import LiabilityValidator, REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Shared fixture builder
# ---------------------------------------------------------------------------

def _valid_row(**overrides) -> pd.DataFrame:
    """Return a one-row DataFrame that passes all validation rules."""
    row = {
        "group_id":                "GRP_A",
        "in_force_count":          100.0,
        "sum_assured":             10_000.0,
        "annual_premium":          1_200.0,
        "attained_age":            50,
        "policy_code":             "ENDOW_NONPAR",
        "policy_term_yr":          5,
        "policy_duration_mths":    36,
        "accrued_bonus_per_policy": 0.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# TestRequiredColumns
# ---------------------------------------------------------------------------

class TestRequiredColumns:
    """Each required column, when missing, must raise ValueError naming it."""

    @pytest.mark.parametrize("col", REQUIRED_COLUMNS)
    def test_missing_column_raises(self, col):
        df = _valid_row()
        df = df.drop(columns=[col])
        with pytest.raises(ValueError, match=col):
            LiabilityValidator.validate(df)

    def test_all_required_columns_present_passes(self):
        LiabilityValidator.validate(_valid_row())  # no exception

    def test_error_lists_all_missing_columns(self):
        df = _valid_row().drop(columns=["sum_assured", "annual_premium"])
        with pytest.raises(ValueError) as exc:
            LiabilityValidator.validate(df)
        msg = str(exc.value)
        assert "sum_assured" in msg
        assert "annual_premium" in msg


# ---------------------------------------------------------------------------
# TestEmptyDataFrame
# ---------------------------------------------------------------------------

class TestEmptyDataFrame:
    """An empty DataFrame (zero rows) is not a valid input."""

    def test_empty_df_raises(self):
        df = _valid_row().iloc[0:0]  # same columns, zero rows
        with pytest.raises(ValueError, match="empty"):
            LiabilityValidator.validate(df)


# ---------------------------------------------------------------------------
# TestRangeChecks
# ---------------------------------------------------------------------------

class TestRangeChecks:
    """Value range rules: each violation is reported by column name."""

    def test_negative_in_force_count_raises(self):
        with pytest.raises(ValueError, match="in_force_count"):
            LiabilityValidator.validate(_valid_row(in_force_count=-1.0))

    def test_zero_in_force_count_passes(self):
        # Zero is allowed — a group with no survivors is valid input
        LiabilityValidator.validate(_valid_row(in_force_count=0.0))

    def test_zero_sum_assured_raises(self):
        with pytest.raises(ValueError, match="sum_assured"):
            LiabilityValidator.validate(_valid_row(sum_assured=0.0))

    def test_negative_sum_assured_raises(self):
        with pytest.raises(ValueError, match="sum_assured"):
            LiabilityValidator.validate(_valid_row(sum_assured=-1.0))

    def test_negative_annual_premium_raises(self):
        with pytest.raises(ValueError, match="annual_premium"):
            LiabilityValidator.validate(_valid_row(annual_premium=-0.01))

    def test_zero_annual_premium_passes(self):
        # Zero premium is valid (e.g. a single-premium policy with prem already paid)
        LiabilityValidator.validate(_valid_row(annual_premium=0.0))

    def test_negative_attained_age_raises(self):
        with pytest.raises(ValueError, match="attained_age"):
            LiabilityValidator.validate(_valid_row(attained_age=-1))

    def test_zero_policy_term_yr_raises(self):
        # A policy must have at least 1 year term
        with pytest.raises(ValueError, match="policy_term_yr"):
            LiabilityValidator.validate(_valid_row(policy_term_yr=0))

    def test_negative_policy_duration_mths_raises(self):
        with pytest.raises(ValueError, match="policy_duration_mths"):
            LiabilityValidator.validate(_valid_row(policy_duration_mths=-1))

    def test_negative_accrued_bonus_raises(self):
        with pytest.raises(ValueError, match="accrued_bonus_per_policy"):
            LiabilityValidator.validate(
                _valid_row(accrued_bonus_per_policy=-100.0)
            )

    def test_multiple_violations_all_reported(self):
        """Both negative in_force_count and zero sum_assured appear in the message."""
        df = _valid_row(in_force_count=-1.0, sum_assured=0.0)
        with pytest.raises(ValueError) as exc:
            LiabilityValidator.validate(df)
        msg = str(exc.value)
        assert "in_force_count" in msg
        assert "sum_assured" in msg


# ---------------------------------------------------------------------------
# TestPolicyCodeRules
# ---------------------------------------------------------------------------

class TestPolicyCodeRules:
    """policy_code must be one of the three valid codes."""

    @pytest.mark.parametrize("code", ["ENDOW_NONPAR", "ENDOW_PAR", "TERM"])
    def test_valid_policy_codes_pass(self, code):
        LiabilityValidator.validate(_valid_row(policy_code=code))

    def test_unknown_policy_code_raises(self):
        with pytest.raises(ValueError, match="policy_code"):
            LiabilityValidator.validate(_valid_row(policy_code="WHOLE_LIFE"))

    def test_invalid_code_message_includes_allowed_codes(self):
        with pytest.raises(ValueError) as exc:
            LiabilityValidator.validate(_valid_row(policy_code="UNKNOWN"))
        msg = str(exc.value)
        assert "ENDOW_NONPAR" in msg or "ENDOW_PAR" in msg or "TERM" in msg

    def test_empty_policy_code_raises(self):
        with pytest.raises(ValueError, match="policy_code"):
            LiabilityValidator.validate(_valid_row(policy_code=""))


# ---------------------------------------------------------------------------
# TestConsistencyRules
# ---------------------------------------------------------------------------

class TestConsistencyRules:
    """
    Consistency rule: policy_duration_mths < policy_term_yr * 12.
    A policy already at or past its maturity date cannot be projected.
    """

    def test_duration_equals_term_mths_raises(self):
        # policy_term_yr=5, policy_duration_mths=60: 60 >= 60 → rejected
        with pytest.raises(ValueError, match="policy_duration_mths"):
            LiabilityValidator.validate(
                _valid_row(policy_term_yr=5, policy_duration_mths=60)
            )

    def test_duration_exceeds_term_mths_raises(self):
        with pytest.raises(ValueError, match="policy_duration_mths"):
            LiabilityValidator.validate(
                _valid_row(policy_term_yr=5, policy_duration_mths=61)
            )

    def test_duration_one_less_than_term_mths_passes(self):
        # policy_term_yr=5, policy_duration_mths=59: 59 < 60 → final month, valid
        LiabilityValidator.validate(
            _valid_row(policy_term_yr=5, policy_duration_mths=59)
        )

    def test_consistency_error_message_mentions_row_count(self):
        df = pd.DataFrame([
            {**_valid_row().iloc[0].to_dict(), "policy_duration_mths": 60},
            {**_valid_row().iloc[0].to_dict(), "policy_duration_mths": 60},
        ])
        with pytest.raises(ValueError) as exc:
            LiabilityValidator.validate(df)
        assert "2" in str(exc.value)


# ---------------------------------------------------------------------------
# TestMultiRowDataFrame
# ---------------------------------------------------------------------------

class TestMultiRowDataFrame:
    """Validator handles DataFrames with many rows correctly."""

    def test_valid_multi_row_passes(self):
        df = pd.DataFrame([
            _valid_row(group_id="GRP_A").iloc[0].to_dict(),
            _valid_row(group_id="GRP_B", policy_code="TERM",
                       policy_duration_mths=0).iloc[0].to_dict(),
            _valid_row(group_id="GRP_C", policy_code="ENDOW_PAR",
                       accrued_bonus_per_policy=500.0).iloc[0].to_dict(),
        ])
        LiabilityValidator.validate(df)  # no exception

    def test_one_bad_row_in_multi_row_raises(self):
        df = pd.DataFrame([
            _valid_row(group_id="GRP_A").iloc[0].to_dict(),
            _valid_row(group_id="GRP_B", sum_assured=-1.0).iloc[0].to_dict(),
        ])
        with pytest.raises(ValueError, match="sum_assured"):
            LiabilityValidator.validate(df)
