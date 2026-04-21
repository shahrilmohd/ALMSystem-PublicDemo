"""
MCP Server for comparing actuarial assumption tables between two folders.

Usage (via uv):
    uv run basis_compare_server.py <original_folder> <new_folder>

The two folder paths are passed as command-line arguments so they can be
configured at startup without touching the source code.
"""

import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from basis_compare_core import (
    compare_table_sets,
    compare_table,
    table_diff_to_md,
    generate_full_report,
)

if len(sys.argv) < 3:
    print(
        "Usage: basis_compare_server.py <original_folder> <new_folder>",
        file=sys.stderr,
    )
    sys.exit(1)

ORIGINAL_FOLDER = sys.argv[1]
NEW_FOLDER = sys.argv[2]

mcp = FastMCP("basis_compare_server")


# ── Tool 1: Table-level changes ───────────────────────────────────────────────

@mcp.tool()
async def list_table_changes() -> dict:
    """
    Identify which tables have been added, removed, or are present in both folders.

    Returns three lists:
      - new_tables: tables in the new folder but not the original
      - missing_tables: tables in the original but not the new folder
      - common_tables: tables present in both (candidates for further comparison)
    """
    return compare_table_sets(ORIGINAL_FOLDER, NEW_FOLDER)


# ── Tool 2: New rows ──────────────────────────────────────────────────────────

@mcp.tool()
async def list_new_rows(table_name: str) -> dict:
    """
    List rows that exist in the new table but are absent from the original.
    The key column is auto-detected (first non-numeric column).

    Args:
        table_name: Table name without .csv extension (e.g. 'mortality_rates')
    """
    result = compare_table(table_name, ORIGINAL_FOLDER, NEW_FOLDER)
    return {
        "table_name": table_name,
        "key_column": result["key_column"],
        "new_row_keys": result["added_rows"],
        "new_rows_data": result["added_rows_data"],
    }


# ── Tool 3: Missing rows ──────────────────────────────────────────────────────

@mcp.tool()
async def list_missing_rows(table_name: str) -> dict:
    """
    List rows that exist in the original table but are absent from the new one.

    Args:
        table_name: Table name without .csv extension
    """
    result = compare_table(table_name, ORIGINAL_FOLDER, NEW_FOLDER)
    return {
        "table_name": table_name,
        "key_column": result["key_column"],
        "missing_row_keys": result["removed_rows"],
        "missing_rows_data": result["removed_rows_data"],
    }


# ── Tool 4: New columns ───────────────────────────────────────────────────────

@mcp.tool()
async def list_new_columns(table_name: str) -> dict:
    """
    List columns added in the new table that were not present in the original.

    Args:
        table_name: Table name without .csv extension
    """
    result = compare_table(table_name, ORIGINAL_FOLDER, NEW_FOLDER)
    return {
        "table_name": table_name,
        "added_columns": result["added_columns"],
        "schema_restructured": result["schema_restructured"],
    }


# ── Tool 5: Missing columns ───────────────────────────────────────────────────

@mcp.tool()
async def list_missing_columns(table_name: str) -> dict:
    """
    List columns that existed in the original table but are missing from the new one.

    Args:
        table_name: Table name without .csv extension
    """
    result = compare_table(table_name, ORIGINAL_FOLDER, NEW_FOLDER)
    return {
        "table_name": table_name,
        "removed_columns": result["removed_columns"],
        "schema_restructured": result["schema_restructured"],
    }


# ── Tool 6: Value changes ─────────────────────────────────────────────────────

@mcp.tool()
async def list_value_changes(table_name: str) -> dict:
    """
    List all rows where one or more values changed between the original and new table.
    Each changed cell shows the old value, new value, and percentage change (for numeric fields).
    Numeric values are compared with floating-point tolerance so formatting differences
    (e.g. 0.1200 vs 0.12) are not reported as changes.

    Args:
        table_name: Table name without .csv extension
    """
    result = compare_table(table_name, ORIGINAL_FOLDER, NEW_FOLDER)
    return {
        "table_name": table_name,
        "key_column": result["key_column"],
        "schema_restructured": result["schema_restructured"],
        "value_changes": result["value_changes"],
        "rows_affected": len(result["value_changes"]),
    }


# ── Tool 7: Schema restructure detection ──────────────────────────────────────

@mcp.tool()
async def detect_schema_restructure(table_name: str) -> dict:
    """
    Detect whether a table's column structure has changed so significantly
    that it constitutes a schema restructure rather than incremental changes.
    A restructure is flagged when fewer than 50% of columns are shared.

    Args:
        table_name: Table name without .csv extension
    """
    result = compare_table(table_name, ORIGINAL_FOLDER, NEW_FOLDER)
    return {
        "table_name": table_name,
        "schema_restructured": result["schema_restructured"],
        "original_columns": result["original_columns"],
        "new_columns": result["new_table_columns"],
        "added_columns": result["added_columns"],
        "removed_columns": result["removed_columns"],
    }


# ── Tool 8: Single-table full comparison (markdown) ───────────────────────────

@mcp.tool()
async def compare_single_table(table_name: str) -> str:
    """
    Run a complete comparison of one table and return the results as a
    formatted markdown section. Covers schema restructures, added/removed
    columns, added/removed rows, and value changes with % deltas.

    Args:
        table_name: Table name without .csv extension
    """
    result = compare_table(table_name, ORIGINAL_FOLDER, NEW_FOLDER)
    return table_diff_to_md(result)


# ── Tool 9: Full report (markdown) ────────────────────────────────────────────

@mcp.tool()
async def generate_comparison_report() -> str:
    """
    Generate a complete markdown comparison report across all tables in both folders.

    The report includes:
      - Executive summary (table counts, change counts)
      - New and missing tables
      - Per-table analysis: schema restructures, added/removed columns,
        added/removed rows, value changes with percentage deltas
    """
    return generate_full_report(ORIGINAL_FOLDER, NEW_FOLDER)


if __name__ == "__main__":
    mcp.run(transport="stdio")
