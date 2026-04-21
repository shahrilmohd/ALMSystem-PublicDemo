"""
Unit tests for AssetDataLoader.

Tests cover:
    - load(): file reading, FileNotFoundError, unsupported extension, TSV support
    - _rename_columns(): column_map applied correctly
    - validate(): delegates to AssetValidator (bad data raises)
    - to_dataframe(): type coercion, optional columns, index reset
    - to_asset_model(): Bond construction, EIR fallback, calibration_spread default,
                        correct AssetModel returned, accounting basis preserved
    - to_asset_model(): Equity construction, dividend_yield fallback, asset_class
    - guard: validate/to_dataframe/to_asset_model before load raises RuntimeError
    - End-to-end: load → validate → to_asset_model round-trip (bonds + equities)
    - Sample file round-trip

All tests use tmp_path to write temporary CSV files — no permanent test fixtures
are modified.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.loaders.asset_data_loader import AssetDataLoader
from engine.asset.asset_model import AssetModel
from engine.asset.bond import Bond
from engine.asset.equity import Equity


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_REQUIRED_HEADER = (
    "asset_id,face_value,annual_coupon_rate,maturity_month,"
    "accounting_basis,initial_book_value"
)

_VALID_CSV_CONTENT = (
    f"{_REQUIRED_HEADER}\n"
    "BOND_001,1000000.0,0.05,60,AC,980000.0\n"
    "BOND_002,500000.0,0.03,36,FVTPL,495000.0\n"
)

_VALID_CSV_WITH_OPTIONAL = (
    f"{_REQUIRED_HEADER},eir,calibration_spread\n"
    "BOND_001,1000000.0,0.05,60,AC,980000.0,0.055,0.01\n"
    "BOND_002,500000.0,0.03,36,FVTPL,495000.0,,\n"
)


def _write_csv(
    tmp_path: Path,
    content: str = _VALID_CSV_CONTENT,
    name: str = "bonds.csv",
) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# TestLoad
# ---------------------------------------------------------------------------

class TestLoad:

    def test_load_sets_raw(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = AssetDataLoader(path)
        loader.load()
        assert loader._raw is not None
        assert len(loader._raw) == 2

    def test_load_file_not_found_raises(self, tmp_path):
        loader = AssetDataLoader(tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError, match="missing.csv"):
            loader.load()

    def test_load_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "bonds.xlsx"
        p.write_text("dummy")
        loader = AssetDataLoader(p)
        with pytest.raises(ValueError, match=".xlsx"):
            loader.load()

    def test_load_tsv_file(self, tmp_path):
        content = _VALID_CSV_CONTENT.replace(",", "\t")
        path = _write_csv(tmp_path, content=content, name="bonds.tsv")
        loader = AssetDataLoader(path)
        loader.load()
        assert len(loader._raw) == 2

    def test_column_map_applied_on_load(self, tmp_path):
        # Source file uses "par" instead of "face_value"
        content = _VALID_CSV_CONTENT.replace("face_value", "par")
        path = _write_csv(tmp_path, content=content)
        loader = AssetDataLoader(path, column_map={"par": "face_value"})
        loader.load()
        assert "face_value" in loader._raw.columns
        assert "par" not in loader._raw.columns

    def test_whitespace_stripped_from_string_columns(self, tmp_path):
        content = (
            f"{_REQUIRED_HEADER}\n"
            "  BOND_001  ,1000000.0,0.05,60,  AC  ,980000.0\n"
        )
        path = _write_csv(tmp_path, content=content)
        loader = AssetDataLoader(path)
        loader.load()
        assert loader._raw.loc[0, "asset_id"] == "BOND_001"
        assert loader._raw.loc[0, "accounting_basis"] == "AC"


# ---------------------------------------------------------------------------
# TestValidate
# ---------------------------------------------------------------------------

class TestValidate:

    def test_validate_passes_on_valid_data(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()  # no raise

    def test_validate_before_load_raises_runtime_error(self, tmp_path):
        loader = AssetDataLoader(tmp_path / "bonds.csv")
        with pytest.raises(RuntimeError, match="load()"):
            loader.validate()

    def test_validate_bad_accounting_basis_raises(self, tmp_path):
        content = (
            f"{_REQUIRED_HEADER}\n"
            "BOND_001,1000000.0,0.05,60,BADVALUE,980000.0\n"
        )
        path = _write_csv(tmp_path, content=content)
        loader = AssetDataLoader(path)
        loader.load()
        with pytest.raises(ValueError, match="accounting_basis"):
            loader.validate()

    def test_validate_negative_face_value_raises(self, tmp_path):
        content = (
            f"{_REQUIRED_HEADER}\n"
            "BOND_001,-1000.0,0.05,60,AC,980000.0\n"
        )
        path = _write_csv(tmp_path, content=content)
        loader = AssetDataLoader(path)
        loader.load()
        with pytest.raises(ValueError, match="face_value"):
            loader.validate()

    def test_validate_duplicate_asset_id_raises(self, tmp_path):
        content = (
            f"{_REQUIRED_HEADER}\n"
            "BOND_001,1000000.0,0.05,60,AC,980000.0\n"
            "BOND_001,500000.0,0.03,36,FVTPL,495000.0\n"
        )
        path = _write_csv(tmp_path, content=content)
        loader = AssetDataLoader(path)
        loader.load()
        with pytest.raises(ValueError, match="Duplicate asset_id"):
            loader.validate()


# ---------------------------------------------------------------------------
# TestToDataframe
# ---------------------------------------------------------------------------

class TestToDataframe:

    def test_to_dataframe_before_load_raises(self, tmp_path):
        loader = AssetDataLoader(tmp_path / "bonds.csv")
        with pytest.raises(RuntimeError, match="load()"):
            loader.to_dataframe()

    def test_to_dataframe_returns_correct_types(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        df = loader.to_dataframe()

        assert df["face_value"].dtype == float
        assert df["annual_coupon_rate"].dtype == float
        assert df["initial_book_value"].dtype == float
        assert df["maturity_month"].dtype == int

    def test_to_dataframe_index_reset(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = AssetDataLoader(path)
        loader.load()
        df = loader.to_dataframe()
        assert list(df.index) == [0, 1]

    def test_to_dataframe_optional_eir_cast_to_float(self, tmp_path):
        path = _write_csv(tmp_path, content=_VALID_CSV_WITH_OPTIONAL)
        loader = AssetDataLoader(path)
        loader.load()
        df = loader.to_dataframe()
        # Row 0 has eir=0.055, row 1 has blank (NaN)
        assert df.loc[0, "eir"] == pytest.approx(0.055)
        assert math.isnan(df.loc[1, "eir"])

    def test_to_dataframe_optional_calibration_spread_cast(self, tmp_path):
        path = _write_csv(tmp_path, content=_VALID_CSV_WITH_OPTIONAL)
        loader = AssetDataLoader(path)
        loader.load()
        df = loader.to_dataframe()
        assert df.loc[0, "calibration_spread"] == pytest.approx(0.01)
        assert math.isnan(df.loc[1, "calibration_spread"])

    def test_to_dataframe_does_not_mutate_raw(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = AssetDataLoader(path)
        loader.load()
        raw_before = loader._raw.copy()
        _ = loader.to_dataframe()
        pd.testing.assert_frame_equal(loader._raw, raw_before)


# ---------------------------------------------------------------------------
# TestToAssetModel
# ---------------------------------------------------------------------------

class TestToAssetModel:

    def test_to_asset_model_before_load_raises(self, tmp_path):
        loader = AssetDataLoader(tmp_path / "bonds.csv")
        with pytest.raises(RuntimeError, match="load()"):
            loader.to_asset_model()

    def test_to_asset_model_returns_asset_model(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        result = loader.to_asset_model()
        assert isinstance(result, AssetModel)

    def test_to_asset_model_correct_count(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()
        assert len(asset_model) == 2

    def test_to_asset_model_correct_asset_ids(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()
        assert asset_model.has_asset("BOND_001")
        assert asset_model.has_asset("BOND_002")

    def test_to_asset_model_accounting_basis_preserved(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()
        assert asset_model.get_asset("BOND_001").accounting_basis == "AC"
        assert asset_model.get_asset("BOND_002").accounting_basis == "FVTPL"

    def test_to_asset_model_face_value_preserved(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()
        assert asset_model.get_asset("BOND_001").face_value == pytest.approx(1_000_000.0)

    def test_to_asset_model_eir_supplied_directly(self, tmp_path):
        # When eir is in the file, it must be used exactly (not recomputed)
        path = _write_csv(tmp_path, content=_VALID_CSV_WITH_OPTIONAL)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()
        bond_001 = asset_model.get_asset("BOND_001")
        assert isinstance(bond_001, Bond)
        assert bond_001.eir == pytest.approx(0.055)

    def test_to_asset_model_eir_computed_when_absent(self, tmp_path):
        # When eir is not in the file, Bond computes it; result must be > 0
        path = _write_csv(tmp_path)  # no eir column
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()
        bond_001 = asset_model.get_asset("BOND_001")
        assert bond_001.eir > 0.0

    def test_to_asset_model_eir_computed_when_nan(self, tmp_path):
        # Row 1 has blank eir — Bond must compute it
        path = _write_csv(tmp_path, content=_VALID_CSV_WITH_OPTIONAL)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()
        bond_002 = asset_model.get_asset("BOND_002")
        assert bond_002.eir > 0.0

    def test_to_asset_model_calibration_spread_supplied(self, tmp_path):
        path = _write_csv(tmp_path, content=_VALID_CSV_WITH_OPTIONAL)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()
        bond_001 = asset_model.get_asset("BOND_001")
        assert bond_001.calibration_spread == pytest.approx(0.01)

    def test_to_asset_model_calibration_spread_defaults_to_zero(self, tmp_path):
        # Row 1 has blank calibration_spread — must default to 0.0
        path = _write_csv(tmp_path, content=_VALID_CSV_WITH_OPTIONAL)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()
        bond_002 = asset_model.get_asset("BOND_002")
        assert bond_002.calibration_spread == pytest.approx(0.0)

    def test_to_asset_model_no_spread_column_defaults_to_zero(self, tmp_path):
        path = _write_csv(tmp_path)  # no calibration_spread column
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()
        for bond in asset_model:
            assert bond.calibration_spread == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_round_trip_no_optional_columns(self, tmp_path):
        """load → validate → to_asset_model with required columns only."""
        path = _write_csv(tmp_path)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()

        assert len(asset_model) == 2
        assert asset_model.has_asset("BOND_001")
        assert asset_model.has_asset("BOND_002")

    def test_full_round_trip_with_optional_columns(self, tmp_path):
        """load → validate → to_asset_model with optional eir and spread."""
        path = _write_csv(tmp_path, content=_VALID_CSV_WITH_OPTIONAL)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()

        bond_001 = asset_model.get_asset("BOND_001")
        assert bond_001.eir == pytest.approx(0.055)
        assert bond_001.calibration_spread == pytest.approx(0.01)

    def test_full_round_trip_with_column_map(self, tmp_path):
        """Column mapping is applied transparently."""
        content = _VALID_CSV_CONTENT.replace("face_value", "par_value")
        path = _write_csv(tmp_path, content=content)
        loader = AssetDataLoader(
            path, column_map={"par_value": "face_value"}
        )
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()

        assert len(asset_model) == 2

    def test_single_bond_portfolio(self, tmp_path):
        content = (
            f"{_REQUIRED_HEADER}\n"
            "GOV_UK_5Y,2000000.0,0.0,60,FVOCI,1950000.0\n"
        )
        path = _write_csv(tmp_path, content=content)
        loader = AssetDataLoader(path)
        loader.load()
        loader.validate()
        asset_model = loader.to_asset_model()

        assert len(asset_model) == 1
        bond = asset_model.get_asset("GOV_UK_5Y")
        assert bond.annual_coupon_rate == pytest.approx(0.0)
        assert bond.accounting_basis == "FVOCI"
        assert bond.maturity_month == 60


# ---------------------------------------------------------------------------
# TestEquityLoading
# ---------------------------------------------------------------------------

_MIXED_CSV = (
    "asset_id,asset_type,face_value,annual_coupon_rate,maturity_month,"
    "accounting_basis,initial_book_value,dividend_yield_yr\n"
    "BOND_001,bond,1000000,0.05,60,AC,980000,\n"
    "BOND_002,bond,500000,0.03,36,FVTPL,495000,\n"
    "EQ_001,equity,,,,, 2000000,0.035\n"
    "EQ_002,equity,,,,,1500000,\n"
)


def _write_mixed_csv(tmp_path: Path) -> Path:
    p = tmp_path / "mixed.csv"
    p.write_text(_MIXED_CSV)
    return p


class TestEquityLoading:

    def test_equity_rows_produce_equity_objects(self, tmp_path):
        loader = AssetDataLoader(_write_mixed_csv(tmp_path))
        loader.load()
        loader.validate()
        am = loader.to_asset_model()

        assert isinstance(am.get_asset("EQ_001"), Equity)
        assert isinstance(am.get_asset("EQ_002"), Equity)

    def test_bond_rows_produce_bond_objects(self, tmp_path):
        loader = AssetDataLoader(_write_mixed_csv(tmp_path))
        loader.load()
        loader.validate()
        am = loader.to_asset_model()

        assert isinstance(am.get_asset("BOND_001"), Bond)
        assert isinstance(am.get_asset("BOND_002"), Bond)

    def test_equity_asset_class_is_equities(self, tmp_path):
        loader = AssetDataLoader(_write_mixed_csv(tmp_path))
        loader.load()
        loader.validate()
        am = loader.to_asset_model()
        assert am.get_asset("EQ_001").asset_class == "equities"

    def test_equity_accounting_basis_is_fvtpl(self, tmp_path):
        loader = AssetDataLoader(_write_mixed_csv(tmp_path))
        loader.load()
        loader.validate()
        am = loader.to_asset_model()
        assert am.get_asset("EQ_001").accounting_basis == "FVTPL"

    def test_equity_initial_market_value_from_initial_book_value(self, tmp_path):
        loader = AssetDataLoader(_write_mixed_csv(tmp_path))
        loader.load()
        loader.validate()
        am = loader.to_asset_model()
        from engine.asset.base_asset import AssetScenarioPoint
        from engine.curves.rate_curve import RiskFreeRateCurve
        sc = AssetScenarioPoint(
            timestep=1,
            rate_curve=RiskFreeRateCurve(spot_rates={1.0: 0.04, 30.0: 0.04}),
            equity_total_return_yr=0.07,
        )
        assert am.get_asset("EQ_001").market_value(sc) == pytest.approx(2_000_000.0)

    def test_equity_dividend_yield_supplied(self, tmp_path):
        loader = AssetDataLoader(_write_mixed_csv(tmp_path))
        loader.load()
        loader.validate()
        am = loader.to_asset_model()
        eq = am.get_asset("EQ_001")
        assert eq.dividend_yield_yr == pytest.approx(0.035)

    def test_equity_dividend_yield_defaults_to_zero(self, tmp_path):
        loader = AssetDataLoader(_write_mixed_csv(tmp_path))
        loader.load()
        loader.validate()
        am = loader.to_asset_model()
        eq = am.get_asset("EQ_002")
        assert eq.dividend_yield_yr == pytest.approx(0.0)

    def test_mixed_total_count(self, tmp_path):
        loader = AssetDataLoader(_write_mixed_csv(tmp_path))
        loader.load()
        loader.validate()
        am = loader.to_asset_model()
        assert len(am) == 4

    def test_equity_only_portfolio(self, tmp_path):
        content = (
            "asset_id,asset_type,initial_book_value,dividend_yield_yr\n"
            "EQ_UK,equity,5000000,0.04\n"
            "EQ_US,equity,3000000,0.02\n"
        )
        p = tmp_path / "equities.csv"
        p.write_text(content)
        loader = AssetDataLoader(p)
        loader.load()
        loader.validate()
        am = loader.to_asset_model()
        assert len(am) == 2
        assert all(a.asset_class == "equities" for a in am)


# ---------------------------------------------------------------------------
# TestSampleFileRoundTrip
# ---------------------------------------------------------------------------

class TestSampleFileRoundTrip:

    def test_sample_asset_model_points_loads(self):
        """The bundled sample CSV must load, validate, and build an AssetModel."""
        from pathlib import Path
        sample = (
            Path(__file__).resolve()
            .parent.parent.parent.parent
            / "tests" / "sample_data" / "mp" / "assets" / "asset_model_points.csv"
        )
        if not sample.exists():
            pytest.skip("Sample file not found")

        loader = AssetDataLoader(sample)
        loader.load()
        loader.validate()
        am = loader.to_asset_model()

        # 4 bonds + 2 equities
        assert len(am) == 6
        assert len(am.assets_by_class("bonds")) == 4
        assert len(am.assets_by_class("equities")) == 2

    def test_sample_total_book_value_exceeds_bel(self):
        """Total asset book value must exceed the known opening BEL (~£36.36M)."""
        from pathlib import Path
        sample = (
            Path(__file__).resolve()
            .parent.parent.parent.parent
            / "tests" / "sample_data" / "mp" / "assets" / "asset_model_points.csv"
        )
        if not sample.exists():
            pytest.skip("Sample file not found")

        loader = AssetDataLoader(sample)
        loader.load()
        loader.validate()
        am = loader.to_asset_model()

        opening_bel = 36_361_030.0
        total_bv = sum(a.get_book_value() for a in am)
        assert total_bv > opening_bel, (
            f"Total book value {total_bv:,.0f} does not exceed opening BEL {opening_bel:,.0f}"
        )
