"""
CLI tests for the `optimise` subcommand (main.py).

Tests cover:
  - happy path: feasible result writes output CSV and returns 0
  - infeasible: returns 1, no output file written
  - missing required argument: argparse raises SystemExit(2)
  - missing input file: returns 1 with error message
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from main import _build_parser, cmd_optimise


# ---------------------------------------------------------------------------
# Helpers: write minimal CSV fixtures to tmp_path
# ---------------------------------------------------------------------------

def _write_candidates(path: Path) -> None:
    df = pd.DataFrame({
        "asset_id":                    ["BOND_A", "BOND_B"],
        "face_value":                  [1_000_000.0, 1_000_000.0],
        "annual_coupon_rate":          [0.05, 0.04],
        "maturity_month":              [120, 60],
        "accounting_basis":            ["AC", "AC"],
        "initial_book_value":          [1_081_109.0, 1_000_000.0],
        "calibration_spread":          [0.0, 0.0],
        "rating":                      ["BBB", "BBB"],
        "seniority":                   ["senior_unsecured", "senior_unsecured"],
        "cashflow_type":               ["fixed", "fixed"],
        "currency":                    ["GBP", "GBP"],
        "has_credit_risk_transfer":    [False, False],
        "has_qualifying_currency_swap":[False, False],
        "spread_bps":                  [120.0, 100.0],
        "default_lgd":                 [0.40, 0.40],
    })
    df.to_csv(path, index=False)


def _write_liability_cfs(path: Path) -> None:
    """Liability cashflows from 500k face each of the two bonds."""
    df = pd.DataFrame({
        "t":  [1,      2,      3,      4,      5,       6,      7,      8,      9,      10],
        "cf": [45_000, 45_000, 45_000, 45_000, 545_000, 25_000, 25_000, 25_000, 25_000, 525_000],
    })
    df.to_csv(path, index=False)


def _write_rfr_curve(path: Path) -> None:
    """Flat 4% RFR curve at key tenors."""
    df = pd.DataFrame({
        "maturity_yr": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "spot_rate":   [0.04] * 10,
    })
    df.to_csv(path, index=False)


def _write_fs_table(path: Path) -> None:
    header = "# effective_date: 2024-01-01\n# source_ref: test fixture\n"
    rows = "rating,seniority,tenor_lower,tenor_upper,fs_bps\n"
    rows += "BBB,senior_unsecured,0.0,100.0,80.0\n"
    rows += "A,senior_unsecured,0.0,100.0,50.0\n"
    path.write_text(header + rows)


# ---------------------------------------------------------------------------
# Fixture: build args namespace directly
# ---------------------------------------------------------------------------

def _make_args(
    tmp_path: Path,
    *,
    candidates_path=None,
    liability_cfs_path=None,
    rfr_curve_path=None,
    fs_table_path=None,
    output_path=None,
    bel_target=None,
    duration_tolerance=0.5,
    liability_currency="GBP",
    hp_cap=0.35,
    quiet=True,
):
    import types
    args = types.SimpleNamespace(
        candidates=str(candidates_path or tmp_path / "candidates.csv"),
        liability_cfs=str(liability_cfs_path or tmp_path / "liability_cfs.csv"),
        rfr_curve=str(rfr_curve_path or tmp_path / "rfr_curve.csv"),
        fs_table=str(fs_table_path or tmp_path / "fs_table.csv"),
        output=str(output_path or tmp_path / "output.csv"),
        bel_target=bel_target,
        duration_tolerance=duration_tolerance,
        liability_currency=liability_currency,
        hp_cap=hp_cap,
        quiet=quiet,
    )
    return args


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOptimiseCLIHappyPath:
    def test_returns_zero_on_feasible(self, tmp_path):
        _write_candidates(tmp_path / "candidates.csv")
        _write_liability_cfs(tmp_path / "liability_cfs.csv")
        _write_rfr_curve(tmp_path / "rfr_curve.csv")
        _write_fs_table(tmp_path / "fs_table.csv")

        args = _make_args(tmp_path, duration_tolerance=1.0)
        rc = cmd_optimise(args)
        assert rc == 0

    def test_output_csv_written(self, tmp_path):
        _write_candidates(tmp_path / "candidates.csv")
        _write_liability_cfs(tmp_path / "liability_cfs.csv")
        _write_rfr_curve(tmp_path / "rfr_curve.csv")
        _write_fs_table(tmp_path / "fs_table.csv")

        args = _make_args(tmp_path, duration_tolerance=1.0)
        cmd_optimise(args)

        out = tmp_path / "output.csv"
        assert out.exists()

    def test_output_csv_has_required_columns(self, tmp_path):
        _write_candidates(tmp_path / "candidates.csv")
        _write_liability_cfs(tmp_path / "liability_cfs.csv")
        _write_rfr_curve(tmp_path / "rfr_curve.csv")
        _write_fs_table(tmp_path / "fs_table.csv")

        args = _make_args(tmp_path, duration_tolerance=1.0)
        cmd_optimise(args)

        df = pd.read_csv(tmp_path / "output.csv")
        required = {
            "asset_id", "face_value", "annual_coupon_rate",
            "maturity_month", "accounting_basis", "initial_book_value",
            "calibration_spread",
        }
        assert required.issubset(set(df.columns))

    def test_output_csv_face_values_positive(self, tmp_path):
        _write_candidates(tmp_path / "candidates.csv")
        _write_liability_cfs(tmp_path / "liability_cfs.csv")
        _write_rfr_curve(tmp_path / "rfr_curve.csv")
        _write_fs_table(tmp_path / "fs_table.csv")

        args = _make_args(tmp_path, duration_tolerance=1.0)
        cmd_optimise(args)

        df = pd.read_csv(tmp_path / "output.csv")
        assert (df["face_value"] > 0).all()

    def test_explicit_bel_target_accepted(self, tmp_path):
        """Passing an explicit bel_target should still produce a feasible result."""
        _write_candidates(tmp_path / "candidates.csv")
        _write_liability_cfs(tmp_path / "liability_cfs.csv")
        _write_rfr_curve(tmp_path / "rfr_curve.csv")
        _write_fs_table(tmp_path / "fs_table.csv")

        args = _make_args(tmp_path, duration_tolerance=1.0, bel_target=1_040_600.0)
        rc = cmd_optimise(args)
        assert rc == 0


class TestOptimiseCLIInfeasible:
    def _write_short_candidates(self, path: Path) -> None:
        df = pd.DataFrame({
            "asset_id":                    ["SHORT_A"],
            "face_value":                  [1_000_000.0],
            "annual_coupon_rate":          [0.05],
            "maturity_month":              [24],
            "accounting_basis":            ["AC"],
            "initial_book_value":          [1_000_000.0],
            "calibration_spread":          [0.0],
            "rating":                      ["BBB"],
            "seniority":                   ["senior_unsecured"],
            "cashflow_type":               ["fixed"],
            "currency":                    ["GBP"],
            "has_credit_risk_transfer":    [False],
            "has_qualifying_currency_swap":[False],
            "spread_bps":                  [120.0],
        })
        df.to_csv(path, index=False)

    def test_returns_one_on_infeasible(self, tmp_path):
        self._write_short_candidates(tmp_path / "candidates.csv")
        _write_liability_cfs(tmp_path / "liability_cfs.csv")
        _write_rfr_curve(tmp_path / "rfr_curve.csv")
        _write_fs_table(tmp_path / "fs_table.csv")

        args = _make_args(tmp_path, duration_tolerance=0.1)
        rc = cmd_optimise(args)
        assert rc == 1

    def test_no_output_written_on_infeasible(self, tmp_path):
        self._write_short_candidates(tmp_path / "candidates.csv")
        _write_liability_cfs(tmp_path / "liability_cfs.csv")
        _write_rfr_curve(tmp_path / "rfr_curve.csv")
        _write_fs_table(tmp_path / "fs_table.csv")

        out = tmp_path / "output.csv"
        args = _make_args(tmp_path, output_path=out, duration_tolerance=0.1)
        cmd_optimise(args)
        assert not out.exists()


class TestOptimiseCLIBadInputs:
    def test_missing_candidates_file_returns_one(self, tmp_path):
        _write_liability_cfs(tmp_path / "liability_cfs.csv")
        _write_rfr_curve(tmp_path / "rfr_curve.csv")
        _write_fs_table(tmp_path / "fs_table.csv")

        args = _make_args(
            tmp_path,
            candidates_path=tmp_path / "nonexistent.csv",
        )
        rc = cmd_optimise(args)
        assert rc == 1

    def test_missing_rfr_file_returns_one(self, tmp_path):
        _write_candidates(tmp_path / "candidates.csv")
        _write_liability_cfs(tmp_path / "liability_cfs.csv")
        _write_fs_table(tmp_path / "fs_table.csv")

        args = _make_args(
            tmp_path,
            rfr_curve_path=tmp_path / "nonexistent.csv",
        )
        rc = cmd_optimise(args)
        assert rc == 1

    def test_missing_fs_table_file_returns_one(self, tmp_path):
        _write_candidates(tmp_path / "candidates.csv")
        _write_liability_cfs(tmp_path / "liability_cfs.csv")
        _write_rfr_curve(tmp_path / "rfr_curve.csv")

        args = _make_args(
            tmp_path,
            fs_table_path=tmp_path / "nonexistent.csv",
        )
        rc = cmd_optimise(args)
        assert rc == 1


class TestOptimiseCLIParser:
    def test_parser_has_optimise_subcommand(self):
        parser = _build_parser()
        # argparse stores subparser choices in the action
        subparsers_action = next(
            a for a in parser._actions
            if hasattr(a, "_parser_class")
        )
        assert "optimise" in subparsers_action.choices

    def test_optimise_requires_candidates(self):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([
                "optimise",
                "--liability-cfs", "l.csv",
                "--rfr-curve", "r.csv",
                "--fs-table", "f.csv",
                "--output", "out.csv",
            ])
        assert exc_info.value.code == 2

    def test_optimise_requires_output(self):
        parser = _build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([
                "optimise",
                "--candidates", "c.csv",
                "--liability-cfs", "l.csv",
                "--rfr-curve", "r.csv",
                "--fs-table", "f.csv",
            ])
        assert exc_info.value.code == 2

    def test_optimise_defaults(self):
        parser = _build_parser()
        args = parser.parse_args([
            "optimise",
            "--candidates", "c.csv",
            "--liability-cfs", "l.csv",
            "--rfr-curve", "r.csv",
            "--fs-table", "f.csv",
            "--output", "out.csv",
        ])
        assert args.duration_tolerance == 0.5
        assert args.liability_currency == "GBP"
        assert args.hp_cap == 0.35
        assert args.bel_target is None
