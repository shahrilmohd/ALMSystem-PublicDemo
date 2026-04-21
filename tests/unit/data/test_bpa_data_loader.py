"""
Unit tests for data/loaders/bpa_data_loader.py

Tests cover:
  - BPADataLoader (model point loading) with all four population types
  - BPADataLoader.load_mortality_basis() (CSV → MortalityBasis)
"""
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.loaders.bpa_data_loader import BPADataLoader
from engine.liability.bpa.mortality import MortalityBasis, MIN_TABLE_AGE, TABLE_LENGTH


# ---------------------------------------------------------------------------
# Helpers — CSV fixture factories using tmp_path
# ---------------------------------------------------------------------------

def write_mp_csv(tmp_path: Path, name: str, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def write_mortality_csvs(tmp_path: Path) -> Path:
    """Write four minimal placeholder mortality CSVs for ages 16–120."""
    ages = list(range(MIN_TABLE_AGE, MIN_TABLE_AGE + TABLE_LENGTH))
    qx_values = [0.02] * TABLE_LENGTH
    rate_values = [0.015] * TABLE_LENGTH

    for fname, val_col, values in [
        ("S3PMA.csv", "qx", qx_values),
        ("S3PFA.csv", "qx", qx_values),
        ("CMI_2023_M.csv", "initial_rate", rate_values),
        ("CMI_2023_F.csv", "initial_rate", rate_values),
    ]:
        df = pd.DataFrame({"age": ages, val_col: values})
        df.to_csv(tmp_path / fname, index=False)

    return tmp_path


# ---------------------------------------------------------------------------
# in_payment model points
# ---------------------------------------------------------------------------

_IN_PAYMENT_ROW = {
    "mp_id": "IP001", "deal_id": "AcmePension_2024Q3",
    "sex": "M", "age": 70.0, "in_force_count": 1.0,
    "pension_pa": 12000.0, "lpi_cap": 0.05, "lpi_floor": 0.0, "gmp_pa": 0.0,
}


class TestLoadInPayment:

    def test_load_validate_to_dataframe(self, tmp_path):
        p = write_mp_csv(tmp_path, "ip.csv", [_IN_PAYMENT_ROW])
        loader = BPADataLoader(p, "in_payment")
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert len(df) == 1
        assert df["sex"].iloc[0] == "M"
        assert df["age"].iloc[0] == pytest.approx(70.0)

    def test_float_columns_coerced(self, tmp_path):
        p = write_mp_csv(tmp_path, "ip.csv", [_IN_PAYMENT_ROW])
        loader = BPADataLoader(p, "in_payment")
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert df["pension_pa"].dtype == float
        assert df["lpi_cap"].dtype == float

    def test_validate_before_load_raises(self, tmp_path):
        p = write_mp_csv(tmp_path, "ip.csv", [_IN_PAYMENT_ROW])
        loader = BPADataLoader(p, "in_payment")
        with pytest.raises(RuntimeError, match="load()"):
            loader.validate()

    def test_to_dataframe_before_load_raises(self, tmp_path):
        p = write_mp_csv(tmp_path, "ip.csv", [_IN_PAYMENT_ROW])
        loader = BPADataLoader(p, "in_payment")
        with pytest.raises(RuntimeError, match="load()"):
            loader.to_dataframe()

    def test_file_not_found_raises(self, tmp_path):
        loader = BPADataLoader(tmp_path / "nonexistent.csv", "in_payment")
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "data.xlsx"
        p.write_text("dummy")
        loader = BPADataLoader(p, "in_payment")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load()

    def test_invalid_data_raises_on_validate(self, tmp_path):
        bad_row = dict(_IN_PAYMENT_ROW, pension_pa=0.0)  # zero pension → invalid
        p = write_mp_csv(tmp_path, "ip_bad.csv", [bad_row])
        loader = BPADataLoader(p, "in_payment")
        loader.load()
        with pytest.raises(ValueError, match="pension_pa"):
            loader.validate()

    def test_column_map_applied(self, tmp_path):
        row = dict(_IN_PAYMENT_ROW)
        row["pens"] = row.pop("pension_pa")  # renamed in source file
        p = write_mp_csv(tmp_path, "ip_mapped.csv", [row])
        loader = BPADataLoader(p, "in_payment", column_map={"pens": "pension_pa"})
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert "pension_pa" in df.columns

    def test_whitespace_stripped_from_sex(self, tmp_path):
        row = dict(_IN_PAYMENT_ROW, sex="  M  ")
        p = write_mp_csv(tmp_path, "ip_ws.csv", [row])
        loader = BPADataLoader(p, "in_payment")
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert df["sex"].iloc[0] == "M"

    def test_tsv_loaded(self, tmp_path):
        df = pd.DataFrame([_IN_PAYMENT_ROW])
        p = tmp_path / "ip.tsv"
        df.to_csv(p, sep="\t", index=False)
        loader = BPADataLoader(p, "in_payment")
        loader.load()
        loader.validate()
        assert len(loader.to_dataframe()) == 1


# ---------------------------------------------------------------------------
# deferred model points
# ---------------------------------------------------------------------------

_DEFERRED_ROW = {
    "mp_id": "DEF001", "deal_id": "AcmePension_2024Q3",
    "sex": "M", "age": 55.0, "in_force_count": 1.0,
    "deferred_pension_pa": 5000.0, "era": 55.0, "nra": 65.0,
    "revaluation_type": "CPI", "revaluation_cap": 0.05, "revaluation_floor": 0.0,
    "deferment_years": 10.0, "tv_eligible": 1,
}


class TestLoadDeferred:

    def test_load_validate_to_dataframe(self, tmp_path):
        p = write_mp_csv(tmp_path, "def.csv", [_DEFERRED_ROW])
        loader = BPADataLoader(p, "deferred")
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert len(df) == 1
        assert df["revaluation_type"].iloc[0] == "CPI"

    def test_float_columns_coerced(self, tmp_path):
        p = write_mp_csv(tmp_path, "def.csv", [_DEFERRED_ROW])
        loader = BPADataLoader(p, "deferred")
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert df["deferment_years"].dtype == float


# ---------------------------------------------------------------------------
# enhanced model points
# ---------------------------------------------------------------------------

_ENHANCED_ROW = {
    "mp_id": "EN001", "deal_id": "AcmePension_2024Q3",
    "sex": "F", "age": 65.0, "in_force_count": 1.0,
    "pension_pa": 8000.0, "lpi_cap": 0.05, "lpi_floor": 0.0,
    "gmp_pa": 0.0, "rating_years": 5.0,
}


class TestLoadEnhanced:

    def test_load_validate_to_dataframe(self, tmp_path):
        p = write_mp_csv(tmp_path, "en.csv", [_ENHANCED_ROW])
        loader = BPADataLoader(p, "enhanced")
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert df["rating_years"].dtype == float
        assert df["rating_years"].iloc[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# dependant model points
# ---------------------------------------------------------------------------

_DEPENDANT_ROW = {
    "mp_id": "DEP001", "deal_id": "AcmePension_2024Q3",
    "member_sex": "M", "member_age": 70.0,
    "dependant_sex": "F", "dependant_age": 67.0,
    "weight": 0.6, "pension_pa": 6000.0, "lpi_cap": 0.05, "lpi_floor": 0.0,
}


class TestLoadDependant:

    def test_load_validate_to_dataframe(self, tmp_path):
        p = write_mp_csv(tmp_path, "dep.csv", [_DEPENDANT_ROW])
        loader = BPADataLoader(p, "dependant")
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert len(df) == 1
        assert df["dependant_age"].dtype == float


# ---------------------------------------------------------------------------
# load_mortality_basis
# ---------------------------------------------------------------------------

class TestLoadMortalityBasis:

    def test_returns_mortality_basis(self, tmp_path):
        write_mortality_csvs(tmp_path)
        basis = BPADataLoader.load_mortality_basis(tmp_path)
        assert isinstance(basis, MortalityBasis)

    def test_array_lengths_correct(self, tmp_path):
        write_mortality_csvs(tmp_path)
        basis = BPADataLoader.load_mortality_basis(tmp_path)
        assert len(basis.base_table_male)             == TABLE_LENGTH
        assert len(basis.base_table_female)           == TABLE_LENGTH
        assert len(basis.initial_improvement_male)    == TABLE_LENGTH
        assert len(basis.initial_improvement_female)  == TABLE_LENGTH

    def test_values_loaded_correctly(self, tmp_path):
        write_mortality_csvs(tmp_path)
        basis = BPADataLoader.load_mortality_basis(tmp_path)
        assert basis.base_table_male[0]          == pytest.approx(0.02)
        assert basis.initial_improvement_male[0] == pytest.approx(0.015)

    def test_custom_ltr_applied(self, tmp_path):
        write_mortality_csvs(tmp_path)
        basis = BPADataLoader.load_mortality_basis(tmp_path, ltr=0.005)
        assert basis.ltr == pytest.approx(0.005)

    def test_missing_file_raises(self, tmp_path):
        # Write only 3 of the 4 required files
        write_mortality_csvs(tmp_path)
        (tmp_path / "S3PMA.csv").unlink()
        with pytest.raises(FileNotFoundError):
            BPADataLoader.load_mortality_basis(tmp_path)

    def test_wrong_row_count_raises(self, tmp_path):
        write_mortality_csvs(tmp_path)
        # Overwrite S3PMA.csv with only 10 rows
        df = pd.DataFrame({"age": range(16, 26), "qx": [0.02] * 10})
        df.to_csv(tmp_path / "S3PMA.csv", index=False)
        with pytest.raises(ValueError, match="expected"):
            BPADataLoader.load_mortality_basis(tmp_path)

    def test_missing_value_column_raises(self, tmp_path):
        write_mortality_csvs(tmp_path)
        # Overwrite S3PFA.csv with wrong column name
        ages = list(range(MIN_TABLE_AGE, MIN_TABLE_AGE + TABLE_LENGTH))
        df = pd.DataFrame({"age": ages, "mortality_rate": [0.02] * TABLE_LENGTH})
        df.to_csv(tmp_path / "S3PFA.csv", index=False)
        with pytest.raises(ValueError, match="missing columns"):
            BPADataLoader.load_mortality_basis(tmp_path)

    def test_ages_sorted_correctly(self, tmp_path):
        write_mortality_csvs(tmp_path)
        # Shuffle S3PMA.csv rows — loader should sort by age
        df = pd.read_csv(tmp_path / "S3PMA.csv")
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        df.to_csv(tmp_path / "S3PMA.csv", index=False)
        basis = BPADataLoader.load_mortality_basis(tmp_path)
        assert len(basis.base_table_male) == TABLE_LENGTH
