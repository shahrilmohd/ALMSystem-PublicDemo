"""
engine/matching_adjustment/fundamental_spread.py
=================================================
Fundamental Spread (FS) table lookup and per-asset MA contribution.

DECISIONS.md §23
----------------
FS(rating, seniority, tenor) is sourced from the PRA's published fundamental
spread tables (PS10/24). The pre-computed fs_bps column in
pra_fundamental_spread.csv stores the result directly.

Lookup key: (rating, seniority, tenor_lower ≤ tenor < tenor_upper).
Seniority values: "senior_secured", "senior_unsecured", "subordinated".
Post-PS10/24 notching: BBB+, BBB, BBB- are distinct rows.

Governance
----------
FundamentalSpreadTable carries effective_date and source_ref parsed from the
CSV comment header lines. Every MAResult records these fields so calculations
are traceable to a specific PRA publication.

NOTE: get_fs() and FundamentalSpreadCalculator.compute() are not included
in this public demo.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


_VALID_SENIORITIES = frozenset({"senior_secured", "senior_unsecured", "subordinated"})

_REQUIRED_COLUMNS = frozenset(
    {"rating", "seniority", "tenor_lower", "tenor_upper", "fs_bps"}
)


@dataclass(frozen=True)
class FSRow:
    """One row from the FS lookup table."""
    rating: str
    seniority: str
    tenor_lower: float
    tenor_upper: float
    fs_bps: float


class FundamentalSpreadTable:
    """
    In-memory FS lookup table loaded from pra_fundamental_spread.csv.

    Parameters
    ----------
    rows : list[FSRow]
    effective_date : date  From CSV governance header.
    source_ref : str       PRA publication identifier, e.g. "PRA PS10/24".
    """

    def __init__(
        self,
        rows: list[FSRow],
        effective_date: date,
        source_ref: str,
    ) -> None:
        self.effective_date = effective_date
        self.source_ref = source_ref
        self._index: dict[tuple[str, str], list[FSRow]] = {}
        for row in rows:
            key = (row.rating, row.seniority)
            self._index.setdefault(key, []).append(row)
        for key in self._index:
            self._index[key].sort(key=lambda r: r.tenor_lower)

    @classmethod
    def from_csv(cls, path: str | Path) -> "FundamentalSpreadTable":
        """
        Load from pra_fundamental_spread.csv.

        Expects comment header lines:
            # effective_date: YYYY-MM-DD
            # source_ref: <text>
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FS table not found: {path}")

        effective_date: date | None = None
        source_ref: str | None = None
        comment_lines: list[str] = []

        with path.open() as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("#"):
                    comment_lines.append(stripped)
                else:
                    break

        for line in comment_lines:
            body = line.lstrip("#").strip()
            if body.startswith("effective_date:"):
                effective_date = date.fromisoformat(body.split(":", 1)[1].strip())
            elif body.startswith("source_ref:"):
                source_ref = body.split(":", 1)[1].strip()

        if effective_date is None:
            raise ValueError(
                f"FS table {path} is missing '# effective_date: YYYY-MM-DD' header"
            )
        if source_ref is None:
            raise ValueError(
                f"FS table {path} is missing '# source_ref: <text>' header"
            )

        df = pd.read_csv(path, comment="#")
        df.columns = df.columns.str.strip()

        missing = _REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"FS table {path} missing required columns: {sorted(missing)}")

        unknown_seniorities = set(df["seniority"].unique()) - _VALID_SENIORITIES
        if unknown_seniorities:
            raise ValueError(
                f"FS table contains unknown seniority values: {unknown_seniorities}"
            )

        rows = [
            FSRow(
                rating=str(row["rating"]).strip(),
                seniority=str(row["seniority"]).strip(),
                tenor_lower=float(row["tenor_lower"]),
                tenor_upper=float(row["tenor_upper"]),
                fs_bps=float(row["fs_bps"]),
            )
            for _, row in df.iterrows()
        ]

        return cls(rows, effective_date, source_ref)

    def get_fs(self, rating: str, seniority: str, tenor: float) -> float:
        """
        Return the FS in basis points for the given (rating, seniority, tenor).

        NOTE: Implementation not included in this public demo.
        """
        raise NotImplementedError(
            "MA fundamental spread lookup is not included in the public demo."
        )

    def known_ratings(self) -> list[str]:
        """Sorted list of unique rating values in the table."""
        return sorted({k[0] for k in self._index})

    def known_seniorities(self) -> list[str]:
        """Sorted list of unique seniority values in the table."""
        return sorted({k[1] for k in self._index})


class FundamentalSpreadCalculator:
    """
    Compute per-asset FS and MA contribution for a portfolio DataFrame.

    Parameters
    ----------
    table : FundamentalSpreadTable
    """

    def __init__(self, table: FundamentalSpreadTable) -> None:
        self._table = table

    def compute(self, assets_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add fs_bps and ma_contribution_bps columns to assets_df.

        Required columns: asset_id, rating, seniority, tenor_years, spread_bps.

        NOTE: Implementation not included in this public demo.
        """
        raise NotImplementedError(
            "MA fundamental spread computation is not included in the public demo."
        )
