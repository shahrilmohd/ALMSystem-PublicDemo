"""
Unit tests for AssetValidator.

Tests cover:
    - Required columns: missing raises ValueError
    - Empty DataFrame raises ValueError
    - face_value: negative rejected
    - annual_coupon_rate: outside [0, 1] rejected
    - maturity_month: < 1 rejected
    - initial_book_value: <= 0 rejected
    - accounting_basis: invalid values rejected
    - asset_id: duplicate values rejected
    - asset_id: empty/whitespace rejected
    - eir (optional): out-of-range (<=0 or >5) rejected; NaN rows skipped
    - calibration_spread (optional): negative rejected; NaN rows skipped
    - Valid portfolio passes without errors
    - All errors collected before raising (not one-at-a-time)
    - Mixed bond/equity portfolios validated correctly
    - asset_type: invalid value rejected
    - dividend_yield_yr (equity): out-of-range rejected
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.validators.asset_validator import AssetValidator, REQUIRED_COLUMNS


# ---------------------------------------------------------------------------
# Shared helper — build a minimal valid DataFrame
# ---------------------------------------------------------------------------

def _valid_df() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "asset_id":           "BOND_001",
            "face_value":         1_000_000.0,
            "annual_coupon_rate": 0.05,
            "maturity_month":     60,
            "accounting_basis":   "AC",
            "initial_book_value": 980_000.0,
        },
        {
            "asset_id":           "BOND_002",
            "face_value":         500_000.0,
            "annual_coupon_rate": 0.03,
            "maturity_month":     36,
            "accounting_basis":   "FVTPL",
            "initial_book_value": 495_000.0,
        },
    ])


# ---------------------------------------------------------------------------
# TestRequiredColumns
# ---------------------------------------------------------------------------

class TestRequiredColumns:

    def test_valid_df_passes(self):
        AssetValidator.validate(_valid_df())  # no raise

    def test_missing_single_column_raises(self):
        df = _valid_df().drop(columns=["face_value"])
        with pytest.raises(ValueError, match="face_value"):
            AssetValidator.validate(df)

    def test_missing_multiple_columns_raises(self):
        df = _valid_df().drop(columns=["face_value", "maturity_month"])
        with pytest.raises(ValueError, match="face_value"):
            AssetValidator.validate(df)

    def test_required_columns_constant_has_six_entries(self):
        assert len(REQUIRED_COLUMNS) == 6

    def test_all_required_columns_named(self):
        expected = {
            "asset_id", "face_value", "annual_coupon_rate",
            "maturity_month", "accounting_basis", "initial_book_value",
        }
        assert set(REQUIRED_COLUMNS) == expected


# ---------------------------------------------------------------------------
# TestEmptyDataFrame
# ---------------------------------------------------------------------------

class TestEmptyDataFrame:

    def test_empty_df_raises(self):
        df = _valid_df().iloc[0:0]  # zero rows, correct columns
        with pytest.raises(ValueError, match="empty"):
            AssetValidator.validate(df)


# ---------------------------------------------------------------------------
# TestNumericRules
# ---------------------------------------------------------------------------

class TestNumericRules:

    def test_face_value_negative_raises(self):
        df = _valid_df()
        df.loc[0, "face_value"] = -1.0
        with pytest.raises(ValueError, match="face_value"):
            AssetValidator.validate(df)

    def test_face_value_zero_passes(self):
        # Zero face value is allowed (fully sold-down position)
        df = _valid_df()
        df.loc[0, "face_value"] = 0.0
        AssetValidator.validate(df)

    def test_annual_coupon_rate_below_zero_raises(self):
        df = _valid_df()
        df.loc[0, "annual_coupon_rate"] = -0.01
        with pytest.raises(ValueError, match="annual_coupon_rate"):
            AssetValidator.validate(df)

    def test_annual_coupon_rate_above_one_raises(self):
        df = _valid_df()
        df.loc[0, "annual_coupon_rate"] = 1.01
        with pytest.raises(ValueError, match="annual_coupon_rate"):
            AssetValidator.validate(df)

    def test_annual_coupon_rate_zero_passes(self):
        df = _valid_df()
        df.loc[0, "annual_coupon_rate"] = 0.0
        AssetValidator.validate(df)

    def test_annual_coupon_rate_one_passes(self):
        df = _valid_df()
        df.loc[0, "annual_coupon_rate"] = 1.0
        AssetValidator.validate(df)

    def test_maturity_month_zero_raises(self):
        df = _valid_df()
        df.loc[0, "maturity_month"] = 0
        with pytest.raises(ValueError, match="maturity_month"):
            AssetValidator.validate(df)

    def test_maturity_month_one_passes(self):
        df = _valid_df()
        df.loc[0, "maturity_month"] = 1
        AssetValidator.validate(df)

    def test_initial_book_value_zero_raises(self):
        df = _valid_df()
        df.loc[0, "initial_book_value"] = 0.0
        with pytest.raises(ValueError, match="initial_book_value"):
            AssetValidator.validate(df)

    def test_initial_book_value_negative_raises(self):
        df = _valid_df()
        df.loc[0, "initial_book_value"] = -100.0
        with pytest.raises(ValueError, match="initial_book_value"):
            AssetValidator.validate(df)


# ---------------------------------------------------------------------------
# TestAccountingBasis
# ---------------------------------------------------------------------------

class TestAccountingBasis:

    def test_invalid_basis_raises(self):
        df = _valid_df()
        df.loc[0, "accounting_basis"] = "HTMM"
        with pytest.raises(ValueError, match="accounting_basis"):
            AssetValidator.validate(df)

    def test_fvoci_passes(self):
        df = _valid_df()
        df.loc[0, "accounting_basis"] = "FVOCI"
        AssetValidator.validate(df)

    def test_lowercase_basis_raises(self):
        df = _valid_df()
        df.loc[0, "accounting_basis"] = "ac"
        with pytest.raises(ValueError, match="accounting_basis"):
            AssetValidator.validate(df)


# ---------------------------------------------------------------------------
# TestAssetIdRules
# ---------------------------------------------------------------------------

class TestAssetIdRules:

    def test_duplicate_asset_id_raises(self):
        df = _valid_df()
        df.loc[1, "asset_id"] = "BOND_001"  # duplicate
        with pytest.raises(ValueError, match="Duplicate asset_id"):
            AssetValidator.validate(df)

    def test_empty_asset_id_raises(self):
        df = _valid_df()
        df.loc[0, "asset_id"] = "   "  # whitespace-only
        with pytest.raises(ValueError, match="asset_id"):
            AssetValidator.validate(df)


# ---------------------------------------------------------------------------
# TestOptionalEir
# ---------------------------------------------------------------------------

class TestOptionalEir:

    def test_eir_column_absent_passes(self):
        # If the column is not in the file at all, no error
        AssetValidator.validate(_valid_df())

    def test_eir_all_nan_passes(self):
        df = _valid_df()
        df["eir"] = np.nan
        AssetValidator.validate(df)

    def test_eir_valid_value_passes(self):
        df = _valid_df()
        df["eir"] = 0.069
        AssetValidator.validate(df)

    def test_eir_zero_raises(self):
        df = _valid_df()
        df["eir"] = 0.0
        with pytest.raises(ValueError, match="eir"):
            AssetValidator.validate(df)

    def test_eir_negative_raises(self):
        df = _valid_df()
        df["eir"] = -0.05
        with pytest.raises(ValueError, match="eir"):
            AssetValidator.validate(df)

    def test_eir_above_five_raises(self):
        df = _valid_df()
        df["eir"] = 5.01
        with pytest.raises(ValueError, match="eir"):
            AssetValidator.validate(df)

    def test_eir_exactly_five_passes(self):
        df = _valid_df()
        df["eir"] = 5.0
        AssetValidator.validate(df)

    def test_eir_partial_nan_valid_row_passes(self):
        # One row has a valid EIR, one row is NaN — both should be fine
        df = _valid_df()
        df["eir"] = [0.05, np.nan]
        AssetValidator.validate(df)

    def test_eir_partial_invalid_row_raises(self):
        df = _valid_df()
        df["eir"] = [0.05, -0.01]
        with pytest.raises(ValueError, match="eir"):
            AssetValidator.validate(df)


# ---------------------------------------------------------------------------
# TestOptionalCalibrationSpread
# ---------------------------------------------------------------------------

class TestOptionalCalibrationSpread:

    def test_calibration_spread_absent_passes(self):
        AssetValidator.validate(_valid_df())

    def test_calibration_spread_zero_passes(self):
        df = _valid_df()
        df["calibration_spread"] = 0.0
        AssetValidator.validate(df)

    def test_calibration_spread_positive_passes(self):
        df = _valid_df()
        df["calibration_spread"] = 0.015
        AssetValidator.validate(df)

    def test_calibration_spread_negative_raises(self):
        df = _valid_df()
        df["calibration_spread"] = -0.001
        with pytest.raises(ValueError, match="calibration_spread"):
            AssetValidator.validate(df)

    def test_calibration_spread_nan_passes(self):
        df = _valid_df()
        df["calibration_spread"] = np.nan
        AssetValidator.validate(df)


# ---------------------------------------------------------------------------
# TestErrorCollection
# ---------------------------------------------------------------------------

class TestErrorCollection:

    def test_multiple_errors_reported_at_once(self):
        df = _valid_df()
        df.loc[0, "face_value"] = -100.0
        df.loc[1, "accounting_basis"] = "JUNK"
        with pytest.raises(ValueError) as exc_info:
            AssetValidator.validate(df)
        msg = str(exc_info.value)
        # Both errors must appear in the message
        assert "face_value" in msg
        assert "accounting_basis" in msg


# ---------------------------------------------------------------------------
# TestMixedPortfolio — asset_type column support
# ---------------------------------------------------------------------------

def _mixed_df() -> pd.DataFrame:
    """DataFrame with 2 bonds and 2 equities."""
    return pd.DataFrame([
        {
            "asset_id":           "BOND_001",
            "asset_type":         "bond",
            "face_value":         1_000_000.0,
            "annual_coupon_rate": 0.05,
            "maturity_month":     60,
            "accounting_basis":   "AC",
            "initial_book_value": 980_000.0,
        },
        {
            "asset_id":           "BOND_002",
            "asset_type":         "bond",
            "face_value":         500_000.0,
            "annual_coupon_rate": 0.03,
            "maturity_month":     36,
            "accounting_basis":   "FVTPL",
            "initial_book_value": 495_000.0,
        },
        {
            "asset_id":           "EQ_001",
            "asset_type":         "equity",
            "face_value":         np.nan,
            "annual_coupon_rate": np.nan,
            "maturity_month":     np.nan,
            "accounting_basis":   np.nan,
            "initial_book_value": 2_000_000.0,
        },
        {
            "asset_id":           "EQ_002",
            "asset_type":         "equity",
            "face_value":         np.nan,
            "annual_coupon_rate": np.nan,
            "maturity_month":     np.nan,
            "accounting_basis":   np.nan,
            "initial_book_value": 1_500_000.0,
        },
    ])


class TestMixedPortfolio:

    def test_mixed_bonds_equities_passes(self):
        AssetValidator.validate(_mixed_df())

    def test_equity_only_portfolio_passes(self):
        df = pd.DataFrame([
            {
                "asset_id":           "EQ_001",
                "asset_type":         "equity",
                "initial_book_value": 5_000_000.0,
            },
        ])
        AssetValidator.validate(df)

    def test_invalid_asset_type_raises(self):
        df = _mixed_df()
        df.loc[2, "asset_type"] = "derivative"
        with pytest.raises(ValueError, match="asset_type"):
            AssetValidator.validate(df)

    def test_bond_required_cols_missing_when_bonds_present_raises(self):
        # Drop a bond-required column from a mixed portfolio
        df = _mixed_df().drop(columns=["face_value"])
        with pytest.raises(ValueError, match="bond-required columns"):
            AssetValidator.validate(df)

    def test_bond_cols_absent_equity_only_passes(self):
        # No bond columns needed when all rows are equities
        df = pd.DataFrame([
            {
                "asset_id":           "EQ_001",
                "asset_type":         "equity",
                "initial_book_value": 5_000_000.0,
            },
            {
                "asset_id":           "EQ_002",
                "asset_type":         "equity",
                "initial_book_value": 3_000_000.0,
            },
        ])
        AssetValidator.validate(df)

    def test_equity_initial_book_value_zero_raises(self):
        df = _mixed_df()
        df.loc[2, "initial_book_value"] = 0.0
        with pytest.raises(ValueError, match="initial_book_value"):
            AssetValidator.validate(df)

    def test_duplicate_id_across_bonds_and_equities_raises(self):
        df = _mixed_df()
        df.loc[2, "asset_id"] = "BOND_001"  # equity row duplicates bond ID
        with pytest.raises(ValueError, match="Duplicate asset_id"):
            AssetValidator.validate(df)

    def test_no_asset_type_column_all_treated_as_bonds(self):
        # Backwards-compatible: no asset_type column → all bonds
        AssetValidator.validate(_valid_df())  # _valid_df has no asset_type column


# ---------------------------------------------------------------------------
# TestDividendYield
# ---------------------------------------------------------------------------

class TestDividendYield:

    def test_dividend_yield_absent_passes(self):
        AssetValidator.validate(_mixed_df())

    def test_dividend_yield_zero_passes(self):
        df = _mixed_df()
        df["dividend_yield_yr"] = 0.0
        AssetValidator.validate(df)

    def test_dividend_yield_valid_passes(self):
        df = _mixed_df()
        df["dividend_yield_yr"] = np.where(df["asset_type"] == "equity", 0.035, np.nan)
        AssetValidator.validate(df)

    def test_dividend_yield_above_one_raises(self):
        df = _mixed_df()
        df["dividend_yield_yr"] = np.where(df["asset_type"] == "equity", 1.5, np.nan)
        with pytest.raises(ValueError, match="dividend_yield_yr"):
            AssetValidator.validate(df)

    def test_dividend_yield_negative_raises(self):
        df = _mixed_df()
        df["dividend_yield_yr"] = np.where(df["asset_type"] == "equity", -0.01, np.nan)
        with pytest.raises(ValueError, match="dividend_yield_yr"):
            AssetValidator.validate(df)


# ---------------------------------------------------------------------------
# TestSampleFile — round-trip load of the actual sample CSV
# ---------------------------------------------------------------------------

class TestSampleFile:

    def test_sample_csv_validates(self, tmp_path):
        """The bundled sample asset model points file must pass validation."""
        import os
        from pathlib import Path
        sample = (
            Path(__file__).resolve()
            .parent.parent.parent.parent  # project root
            / "tests" / "sample_data" / "mp" / "assets" / "asset_model_points.csv"
        )
        if not sample.exists():
            pytest.skip("Sample file not found")
        import pandas as pd
        df = pd.read_csv(sample)
        AssetValidator.validate(df)
