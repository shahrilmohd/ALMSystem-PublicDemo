"""
/assumptions router — read-only access to assumption table snapshots.

Endpoints
---------
GET /assumptions
    List all assumption table names available for a given valuation date.

GET /assumptions/{table_name}
    Return rows from a named assumption table for a given valuation date.

Storage layout
--------------
Assumption tables are stored as CSV files under a root directory configured
via the environment variable ALM_ASSUMPTIONS_ROOT (default: "data/assumptions").
Each valuation date has its own sub-directory:

    <root>/
        2025-12-31/
            mortality_rates.csv
            lapse_rates.csv
            expense_loadings.csv
        2026-03-31/
            mortality_rates.csv
            ...

The valuation_date query parameter selects the sub-directory.  Table names
are the CSV filenames without the .csv extension.

Architectural rule:
    This router does NOT import anything from engine/.
    It reads from the local filesystem — it does not accept user-supplied
    paths; the path is constructed from the server-side root and the
    validated valuation_date and table_name parameters.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.schemas.data_schema import (
    AssumptionTableResponse,
    AssumptionTablesListResponse,
    _MAX_ROWS,
)

router = APIRouter()

# ---------------------------------------------------------------------------
# Server-side root for assumption table snapshots.
# Resolved relative to the project root (parent of api/).
# ---------------------------------------------------------------------------

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "assumptions"
_ASSUMPTIONS_ROOT = Path(os.getenv("ALM_ASSUMPTIONS_ROOT", str(_DEFAULT_ROOT)))

# Validates that valuation_date is YYYY-MM-DD — avoids path traversal.
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Validates table_name contains only safe characters — avoids path traversal.
_TABLE_NAME_RE = re.compile(r"^[\w\-]+$")


def _safe_snapshot_dir(valuation_date: str) -> Path:
    """
    Return the snapshot directory for a valuation date, or raise 422.
    Only YYYY-MM-DD dates are accepted to prevent path traversal.
    """
    if not _DATE_RE.match(valuation_date):
        raise HTTPException(
            status_code=422,
            detail=(
                f"valuation_date must be in YYYY-MM-DD format, "
                f"got {valuation_date!r}."
            ),
        )
    return _ASSUMPTIONS_ROOT / valuation_date


def _safe_table_path(snapshot_dir: Path, table_name: str) -> Path:
    """
    Return the CSV path for a table name, or raise 422 on invalid name.
    Only alphanumeric + hyphen/underscore characters are accepted.
    """
    if not _TABLE_NAME_RE.match(table_name):
        raise HTTPException(
            status_code=422,
            detail=(
                f"table_name must contain only letters, digits, hyphens, and "
                f"underscores, got {table_name!r}."
            ),
        )
    return snapshot_dir / f"{table_name}.csv"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=AssumptionTablesListResponse)
def list_assumption_tables(
    valuation_date: str = Query(
        ...,
        description="Valuation date in YYYY-MM-DD format.",
        examples=["2025-12-31"],
    ),
) -> AssumptionTablesListResponse:
    """
    List all assumption table names available for a valuation date.

    Returns the CSV filenames (without extension) found in the snapshot
    directory for that date.

    Returns **404** if no snapshot exists for the requested date.
    Returns **422** if valuation_date is not in YYYY-MM-DD format.
    """
    snapshot_dir = _safe_snapshot_dir(valuation_date)

    if not snapshot_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No assumption snapshot found for valuation date {valuation_date!r}.",
        )

    tables = sorted(
        p.stem
        for p in snapshot_dir.iterdir()
        if p.suffix.lower() == ".csv" and p.is_file()
    )

    return AssumptionTablesListResponse(
        valuation_date=valuation_date,
        tables=tables,
    )


@router.get("/{table_name}", response_model=AssumptionTableResponse)
def get_assumption_table(
    table_name: str,
    valuation_date: str = Query(
        ...,
        description="Valuation date in YYYY-MM-DD format.",
        examples=["2025-12-31"],
    ),
) -> AssumptionTableResponse:
    """
    Return rows from a named assumption table for a valuation date.

    Returns up to 500 rows; truncated is True if the file had more.

    Returns **404** if the snapshot or the table does not exist.
    Returns **422** if valuation_date or table_name are invalid.
    Returns **503** if the CSV file cannot be parsed.
    """
    snapshot_dir = _safe_snapshot_dir(valuation_date)
    if not snapshot_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No assumption snapshot found for valuation date {valuation_date!r}.",
        )

    csv_path = _safe_table_path(snapshot_dir, table_name)
    if not csv_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Table {table_name!r} not found in snapshot for "
                f"{valuation_date!r}."
            ),
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"Failed to read assumption table {table_name!r}: {exc}",
        ) from exc

    total_rows = len(df)
    truncated = total_rows > _MAX_ROWS
    df = df.head(_MAX_ROWS)

    return AssumptionTableResponse(
        table_name=table_name,
        valuation_date=valuation_date,
        row_count=len(df),
        truncated=truncated,
        columns=list(df.columns),
        data=df.where(pd.notna(df), other=None).to_dict(orient="records"),
    )
