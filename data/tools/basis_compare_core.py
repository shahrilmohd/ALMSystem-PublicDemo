"""
Core comparison logic for actuarial assumption tables.
No MCP dependencies — can be imported and tested independently.
"""

import csv
from pathlib import Path
from datetime import datetime
from typing import Optional

# Two numeric values within this tolerance are treated as identical.
NUMERIC_TOLERANCE = 1e-9

# If fewer than this fraction of columns are shared, it's a schema restructure.
SCHEMA_RESTRUCTURE_THRESHOLD = 0.5


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(filepath: str) -> tuple[list[str], list[dict]]:
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return headers, rows


# ── Value utilities ───────────────────────────────────────────────────────────

def is_numeric(value: str) -> bool:
    try:
        float(str(value).strip())
        return True
    except (ValueError, AttributeError):
        return False


def normalize_value(value: str) -> str:
    """Normalise for comparison: strip whitespace, lowercase booleans."""
    v = str(value).strip()
    if v.lower() in ("true", "false"):
        return v.lower()
    return v


def values_equal(v1: str, v2: str) -> bool:
    """True if values are semantically equal (numeric tolerance + bool normalisation)."""
    n1 = normalize_value(v1)
    n2 = normalize_value(v2)
    if n1 == n2:
        return True
    if is_numeric(n1) and is_numeric(n2):
        return abs(float(n1) - float(n2)) < NUMERIC_TOLERANCE
    return False


def pct_change(old_val: str, new_val: str) -> Optional[float]:
    """Return percentage change if both values are numeric, else None."""
    if is_numeric(old_val) and is_numeric(new_val):
        old = float(old_val)
        new = float(new_val)
        if old == 0:
            return None
        return (new - old) / abs(old) * 100
    return None


# ── Key column detection ──────────────────────────────────────────────────────

def detect_key_column(headers: list[str], rows: list[dict]) -> str:
    """
    Auto-detect the natural key column.
    Strategy: pick the first column whose values are not all numeric,
    which indicates an identifier rather than a data field.
    Falls back to the first column if everything is numeric.
    """
    if not headers:
        return ""
    for col in headers:
        values = [row.get(col, "") for row in rows]
        if any(not is_numeric(v) for v in values if v):
            return col
    return headers[0]


# ── Folder / table helpers ────────────────────────────────────────────────────

def get_table_names(folder: str) -> set[str]:
    return {f.stem for f in Path(folder).glob("*.csv")}


def get_table_path(folder: str, name: str) -> str:
    return str(Path(folder) / f"{name}.csv")


# ── Table-set comparison ──────────────────────────────────────────────────────

def compare_table_sets(original_folder: str, new_folder: str) -> dict:
    """Identify tables added, removed, or present in both folders."""
    orig = get_table_names(original_folder)
    new = get_table_names(new_folder)
    return {
        "new_tables": sorted(new - orig),
        "missing_tables": sorted(orig - new),
        "common_tables": sorted(orig & new),
    }


# ── Single-table comparison ───────────────────────────────────────────────────

def is_schema_restructure(orig_headers: list[str], new_headers: list[str]) -> bool:
    orig_set = set(orig_headers)
    new_set = set(new_headers)
    if not orig_set or not new_set:
        return False
    overlap = orig_set & new_set
    max_cols = max(len(orig_set), len(new_set))
    return (len(overlap) / max_cols) < SCHEMA_RESTRUCTURE_THRESHOLD


def compare_table(table_name: str, original_folder: str, new_folder: str) -> dict:
    """
    Full comparison of a single table present in both folders.
    Returns a structured dict used by both the MCP tools and markdown generators.
    """
    orig_headers, orig_rows = load_csv(get_table_path(original_folder, table_name))
    new_headers, new_rows = load_csv(get_table_path(new_folder, table_name))

    restructured = is_schema_restructure(orig_headers, new_headers)

    orig_col_set = set(orig_headers)
    new_col_set = set(new_headers)
    added_columns = sorted(new_col_set - orig_col_set)
    removed_columns = sorted(orig_col_set - new_col_set)
    # Preserve original column order for common columns
    common_columns = [c for c in orig_headers if c in new_col_set]

    key_col = (
        detect_key_column(orig_headers, orig_rows)
        if orig_rows
        else (orig_headers[0] if orig_headers else "")
    )

    orig_keyed = {normalize_value(row.get(key_col, "")): row for row in orig_rows}
    new_keyed = {normalize_value(row.get(key_col, "")): row for row in new_rows}

    added_row_keys = sorted(set(new_keyed) - set(orig_keyed))
    removed_row_keys = sorted(set(orig_keyed) - set(new_keyed))
    common_keys = sorted(set(orig_keyed) & set(new_keyed))

    added_rows_data = [new_keyed[k] for k in added_row_keys]
    removed_rows_data = [orig_keyed[k] for k in removed_row_keys]

    value_changes: list[dict] = []
    if not restructured:
        for key in common_keys:
            orig_row = orig_keyed[key]
            new_row = new_keyed[key]
            row_changes = []
            for col in common_columns:
                if col == key_col:
                    continue
                orig_val = orig_row.get(col, "")
                new_val = new_row.get(col, "")
                if not values_equal(orig_val, new_val):
                    change: dict = {
                        "column": col,
                        "original_value": orig_val,
                        "new_value": new_val,
                    }
                    pct = pct_change(orig_val, new_val)
                    if pct is not None:
                        change["pct_change"] = round(pct, 4)
                    row_changes.append(change)
            if row_changes:
                value_changes.append({"key": key, "key_column": key_col, "changes": row_changes})

    total_changes = (
        len(added_columns)
        + len(removed_columns)
        + len(added_row_keys)
        + len(removed_row_keys)
        + len(value_changes)
        + (1 if restructured else 0)
    )

    return {
        "table_name": table_name,
        "key_column": key_col,
        "schema_restructured": restructured,
        "original_columns": orig_headers,
        "new_table_columns": new_headers,
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "common_columns": common_columns,
        "added_rows": added_row_keys,
        "added_rows_data": added_rows_data,
        "removed_rows": removed_row_keys,
        "removed_rows_data": removed_rows_data,
        "value_changes": value_changes,
        "total_changes": total_changes,
    }


# ── Markdown helpers ──────────────────────────────────────────────────────────

def _md_table(headers: list[str], rows: list[dict]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


def _fmt_pct(pct: Optional[float]) -> str:
    if pct is None:
        return "—"
    return f"{pct:+.2f}%"


def table_diff_to_md(result: dict) -> str:
    """Render a single table comparison result as a markdown section."""
    lines: list[str] = []
    name = result["table_name"]

    lines += [f"### `{name}`", ""]

    if result["schema_restructured"]:
        orig_cols = ", ".join(f"`{c}`" for c in result["original_columns"])
        new_cols = ", ".join(f"`{c}`" for c in result["new_table_columns"])
        lines += [
            "> ⚠️ **Schema Restructure Detected** — column structure has changed significantly.",
            f"> - **Original columns:** {orig_cols}",
            f"> - **New columns:** {new_cols}",
            "",
        ]

    lines += [f"**Key column:** `{result['key_column']}`", ""]

    lines += [
        "| Category | Count |",
        "|---|---|",
        f"| Added columns | {len(result['added_columns'])} |",
        f"| Removed columns | {len(result['removed_columns'])} |",
        f"| New rows | {len(result['added_rows'])} |",
        f"| Missing rows | {len(result['removed_rows'])} |",
        f"| Rows with value changes | {len(result['value_changes'])} |",
        "",
    ]

    if result["schema_restructured"]:
        lines += ["_Value-level comparison skipped due to schema restructure._", ""]
        return "\n".join(lines)

    if result["added_columns"]:
        lines += ["#### Added Columns"]
        lines += [f"- `{c}`" for c in result["added_columns"]]
        lines.append("")

    if result["removed_columns"]:
        lines += ["#### Removed Columns"]
        lines += [f"- `{c}`" for c in result["removed_columns"]]
        lines.append("")

    if result["added_rows_data"]:
        lines += [
            "#### New Rows",
            _md_table(result["new_table_columns"], result["added_rows_data"]),
            "",
        ]

    if result["removed_rows_data"]:
        lines += [
            "#### Missing Rows",
            _md_table(result["original_columns"], result["removed_rows_data"]),
            "",
        ]

    if result["value_changes"]:
        key_col = result["key_column"]
        lines += [
            "#### Value Changes",
            f"| {key_col} | Column | Original Value | New Value | % Change |",
            "|---|---|---|---|---|",
        ]
        for row_change in result["value_changes"]:
            for col_change in row_change["changes"]:
                lines.append(
                    f"| {row_change['key']} "
                    f"| {col_change['column']} "
                    f"| {col_change['original_value']} "
                    f"| {col_change['new_value']} "
                    f"| {_fmt_pct(col_change.get('pct_change'))} |"
                )
        lines.append("")

    if result["total_changes"] == 0:
        lines += ["_No changes detected._", ""]

    return "\n".join(lines)


# ── Full report ───────────────────────────────────────────────────────────────

def generate_full_report(original_folder: str, new_folder: str) -> str:
    """Generate a complete markdown comparison report across all tables."""
    orig_name = Path(original_folder).name
    new_name = Path(new_folder).name
    table_sets = compare_table_sets(original_folder, new_folder)

    common_results = [
        compare_table(t, original_folder, new_folder)
        for t in table_sets["common_tables"]
    ]
    changed = [r for r in common_results if r["total_changes"] > 0]
    unchanged = [r for r in common_results if r["total_changes"] == 0]

    total_orig = len(table_sets["missing_tables"]) + len(table_sets["common_tables"])
    total_new = len(table_sets["new_tables"]) + len(table_sets["common_tables"])

    lines: list[str] = [
        "# Assumption Table Comparison Report",
        "",
        "| | |",
        "|---|---|",
        f"| **Original** | `{orig_name}` |",
        f"| **New** | `{new_name}` |",
        f"| **Generated** | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Category | Original | New |",
        "|---|---|---|",
        f"| Total tables | {total_orig} | {total_new} |",
        f"| New tables | — | {len(table_sets['new_tables'])} |",
        f"| Missing tables | {len(table_sets['missing_tables'])} | — |",
        f"| Common — unchanged | {len(unchanged)} | {len(unchanged)} |",
        f"| Common — changed | {len(changed)} | {len(changed)} |",
        "",
    ]

    # New / missing table lists
    if table_sets["new_tables"]:
        lines += ["**New tables:** " + ", ".join(f"`{t}`" for t in table_sets["new_tables"]), ""]
    if table_sets["missing_tables"]:
        lines += ["**Missing tables:** " + ", ".join(f"`{t}`" for t in table_sets["missing_tables"]), ""]
    if unchanged:
        lines += ["**Unchanged tables:** " + ", ".join(f"`{r['table_name']}`" for r in unchanged), ""]

    # Changed-tables-at-a-glance breakdown
    if changed:
        lines += [
            "### Changed Tables at a Glance",
            "",
            "| Table | Added Cols | Removed Cols | New Rows | Missing Rows | Value Changes | Schema |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in changed:
            schema_flag = "⚠️ Restructured" if r["schema_restructured"] else "—"
            lines.append(
                f"| `{r['table_name']}` "
                f"| {len(r['added_columns'])} "
                f"| {len(r['removed_columns'])} "
                f"| {len(r['added_rows'])} "
                f"| {len(r['removed_rows'])} "
                f"| {len(r['value_changes'])} "
                f"| {schema_flag} |"
            )
        lines.append("")

    lines += ["---", "", "## Per-Table Analysis", ""]

    for result in changed:
        lines.append(table_diff_to_md(result))
        lines += ["---", ""]

    return "\n".join(lines)
