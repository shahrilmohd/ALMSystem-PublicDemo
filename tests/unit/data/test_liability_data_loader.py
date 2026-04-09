"""
Unit tests for LiabilityDataLoader.

Tests cover:
    - load(): file reading, FileNotFoundError, unsupported extension
    - _rename_columns(): column_map applied correctly
    - validate(): delegates to LiabilityValidator (bad data raises)
    - to_dataframe(): type coercion, whitespace stripping, index reset
    - guard: validate/to_dataframe before load raises RuntimeError
    - TSV format support

All tests use tmp_path to write temporary CSV files — no permanent test fixtures
are modified.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.loaders.liability_data_loader import LiabilityDataLoader


# ---------------------------------------------------------------------------
# Shared helper — write a minimal valid CSV to tmp_path
# ---------------------------------------------------------------------------

_VALID_CSV_CONTENT = (
    "group_id,in_force_count,sum_assured,annual_premium,"
    "attained_age,policy_code,policy_term_yr,policy_duration_mths,"
    "accrued_bonus_per_policy\n"
    "GRP_A,100.0,10000.0,1200.0,50,ENDOW_NONPAR,5,36,0.0\n"
    "GRP_B,50.0,5000.0,600.0,60,TERM,10,36,0.0\n"
)


def _write_csv(tmp_path: Path, content: str = _VALID_CSV_CONTENT,
               name: str = "mps.csv") -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# TestLoad
# ---------------------------------------------------------------------------

class TestLoad:
    """load() reads the file and applies the column map."""

    def test_load_sets_raw(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = LiabilityDataLoader(path)
        loader.load()
        assert loader._raw is not None
        assert len(loader._raw) == 2

    def test_load_file_not_found_raises(self, tmp_path):
        loader = LiabilityDataLoader(tmp_path / "missing.csv")
        with pytest.raises(FileNotFoundError, match="missing.csv"):
            loader.load()

    def test_load_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "mps.xlsx"
        p.write_text("dummy")
        loader = LiabilityDataLoader(p)
        with pytest.raises(ValueError, match=".xlsx"):
            loader.load()

    def test_load_tsv_file(self, tmp_path):
        content = _VALID_CSV_CONTENT.replace(",", "\t")
        path = _write_csv(tmp_path, content=content, name="mps.tsv")
        loader = LiabilityDataLoader(path)
        loader.load()
        assert len(loader._raw) == 2

    def test_column_map_applied_on_load(self, tmp_path):
        # Source file uses "if_count" instead of "in_force_count"
        content = _VALID_CSV_CONTENT.replace(
            "in_force_count", "if_count"
        )
        path = _write_csv(tmp_path, content=content)
        loader = LiabilityDataLoader(
            path, column_map={"if_count": "in_force_count"}
        )
        loader.load()
        assert "in_force_count" in loader._raw.columns
        assert "if_count" not in loader._raw.columns

    def test_unmapped_columns_pass_through(self, tmp_path):
        """Columns not in column_map are left unchanged."""
        path = _write_csv(tmp_path)
        loader = LiabilityDataLoader(
            path, column_map={"nonexistent_col": "something"}
        )
        loader.load()
        assert "group_id" in loader._raw.columns

    def test_no_column_map_leaves_columns_unchanged(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = LiabilityDataLoader(path)
        loader.load()
        assert "in_force_count" in loader._raw.columns


# ---------------------------------------------------------------------------
# TestValidate
# ---------------------------------------------------------------------------

class TestValidate:
    """validate() delegates to LiabilityValidator; bad data raises ValueError."""

    def test_valid_file_passes(self, tmp_path):
        path = _write_csv(tmp_path)
        loader = LiabilityDataLoader(path)
        loader.load()
        loader.validate()  # no exception

    def test_missing_required_column_raises(self, tmp_path):
        # Build a valid CSV then drop the sum_assured column entirely
        import pandas as pd
        df = pd.read_csv(__import__("io").StringIO(_VALID_CSV_CONTENT))
        df = df.drop(columns=["sum_assured"])
        path = tmp_path / "no_sa.csv"
        df.to_csv(path, index=False)
        loader = LiabilityDataLoader(path)
        loader.load()
        with pytest.raises(ValueError, match="sum_assured"):
            loader.validate()

    def test_negative_in_force_count_raises(self, tmp_path):
        bad = _VALID_CSV_CONTENT.replace("100.0,", "-1.0,", 1)
        path = _write_csv(tmp_path, content=bad)
        loader = LiabilityDataLoader(path)
        loader.load()
        with pytest.raises(ValueError, match="in_force_count"):
            loader.validate()

    def test_validate_before_load_raises_runtime_error(self, tmp_path):
        loader = LiabilityDataLoader(tmp_path / "mps.csv")
        with pytest.raises(RuntimeError, match="load"):
            loader.validate()


# ---------------------------------------------------------------------------
# TestToDataframe
# ---------------------------------------------------------------------------

class TestToDataframe:
    """to_dataframe() returns a clean, correctly typed DataFrame."""

    def _loaded_loader(self, tmp_path: Path) -> LiabilityDataLoader:
        path = _write_csv(tmp_path)
        loader = LiabilityDataLoader(path)
        loader.load()
        loader.validate()
        return loader

    def test_returns_dataframe(self, tmp_path):
        df = self._loaded_loader(tmp_path).to_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_row_count_preserved(self, tmp_path):
        df = self._loaded_loader(tmp_path).to_dataframe()
        assert len(df) == 2

    def test_float_columns_are_float(self, tmp_path):
        df = self._loaded_loader(tmp_path).to_dataframe()
        for col in ("in_force_count", "sum_assured", "annual_premium",
                    "accrued_bonus_per_policy"):
            assert df[col].dtype == float, f"{col} should be float"

    def test_int_columns_are_int(self, tmp_path):
        df = self._loaded_loader(tmp_path).to_dataframe()
        for col in ("attained_age", "policy_term_yr", "policy_duration_mths"):
            assert pd.api.types.is_integer_dtype(df[col]), f"{col} should be int"

    def test_whitespace_stripped_from_policy_code(self, tmp_path):
        content = _VALID_CSV_CONTENT.replace("ENDOW_NONPAR", " ENDOW_NONPAR ")
        path = _write_csv(tmp_path, content=content)
        loader = LiabilityDataLoader(path)
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert df["policy_code"].iloc[0] == "ENDOW_NONPAR"

    def test_whitespace_stripped_from_group_id(self, tmp_path):
        content = _VALID_CSV_CONTENT.replace("GRP_A", "  GRP_A  ")
        path = _write_csv(tmp_path, content=content)
        loader = LiabilityDataLoader(path)
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert df["group_id"].iloc[0] == "GRP_A"

    def test_index_is_reset(self, tmp_path):
        df = self._loaded_loader(tmp_path).to_dataframe()
        assert list(df.index) == list(range(len(df)))

    def test_to_dataframe_before_load_raises_runtime_error(self, tmp_path):
        loader = LiabilityDataLoader(tmp_path / "mps.csv")
        with pytest.raises(RuntimeError, match="load"):
            loader.to_dataframe()

    def test_original_raw_not_mutated(self, tmp_path):
        """to_dataframe() returns a copy; self._raw is unchanged."""
        loader = self._loaded_loader(tmp_path)
        raw_id_before = id(loader._raw)
        loader.to_dataframe()
        assert id(loader._raw) == raw_id_before


# ---------------------------------------------------------------------------
# TestColumnMapEndToEnd
# ---------------------------------------------------------------------------

class TestColumnMapEndToEnd:
    """Full load → validate → to_dataframe with a non-standard source file."""

    def test_full_pipeline_with_column_map(self, tmp_path):
        """
        Source file uses legacy names:
            if_count   → in_force_count
            sa         → sum_assured
            ann_prem   → annual_premium

        After mapping, validate and to_dataframe must succeed.
        """
        content = (
            "group_id,if_count,sa,ann_prem,"
            "attained_age,policy_code,policy_term_yr,policy_duration_mths,"
            "accrued_bonus_per_policy\n"
            "GRP_A,100.0,10000.0,1200.0,50,ENDOW_NONPAR,5,36,0.0\n"
        )
        path = tmp_path / "legacy.csv"
        path.write_text(content)

        loader = LiabilityDataLoader(
            path,
            column_map={
                "if_count": "in_force_count",
                "sa":       "sum_assured",
                "ann_prem": "annual_premium",
            },
        )
        loader.load()
        loader.validate()
        df = loader.to_dataframe()

        assert df["in_force_count"].iloc[0] == pytest.approx(100.0)
        assert df["sum_assured"].iloc[0]    == pytest.approx(10_000.0)
        assert df["annual_premium"].iloc[0] == pytest.approx(1_200.0)
