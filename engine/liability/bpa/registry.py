"""
engine/liability/bpa/registry.py — BPA deal registry and cohort_id derivation.

BPADealRegistry is a thin lookup table keyed by deal_id.  It is loaded once
per BPA run from a CSV file (bpa_deals.csv) and passed to BPAValidator and
the BPA run mode orchestrator.

See DECISIONS.md §44, §45 for the full rationale.

Design constraints
------------------
- Registry is immutable after construction (frozen dataclass entries).
- Models never read files directly (CLAUDE.md rule 2) — BPADealRegistry.from_csv()
  is the boundary; everything beyond it receives the populated registry object.
- deal_id must never change after a deal goes live: it is the root from which
  the IFRS 17 cohort_id is derived, and changing it orphans historical state records.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd


# Valid values for deal_type (DECISIONS.md §45)
DealType = Literal["buyout", "buyin"]

# Valid population type suffixes for cohort_id derivation
PopulationType = Literal["pensioner", "deferred", "dependant", "enhanced"]


@dataclass(frozen=True)
class BPADealMetadata:
    """
    Immutable metadata for a single BPA deal.

    Attributes
    ----------
    deal_id : str
        Primary key.  Convention: {SchemeName}_{Year}Q{Quarter}.
        Must be globally unique and stable for the life of the deal.
    deal_type : str
        "buyout" — legal liability transfers to insurer; insurer pays members directly.
        "buyin"  — scheme retains legal liability; insurer pays the scheme.
    inception_date : date
        Date of initial recognition (IFRS 17 para 25).  Fixed at transaction date.
    deal_name : str
        Human-readable label used in reports.
    ma_eligible : bool
        Provisional flag indicating the deal is submitted for MA assessment (Step 19).
        True does not guarantee MA — the full eligibility test runs at Step 19.
    """

    deal_id:        str
    deal_type:      DealType
    inception_date: date
    deal_name:      str
    ma_eligible:    bool


class BPADealRegistry:
    """
    Lookup table of BPA deal metadata, keyed by deal_id.

    Loaded once per run from bpa_deals.csv.  Passed to BPAValidator (for
    cross-referencing deal_id values in model point files) and to the BPA
    run mode orchestrator (for deriving cohort_id and fetching inception_date).

    Parameters
    ----------
    deals : list[BPADealMetadata]
        All known deals.  deal_id values must be unique.

    Raises
    ------
    ValueError
        If any two entries share the same deal_id.
    """

    def __init__(self, deals: list[BPADealMetadata]) -> None:
        seen: dict[str, BPADealMetadata] = {}
        for d in deals:
            if d.deal_id in seen:
                raise ValueError(
                    f"Duplicate deal_id '{d.deal_id}' in BPADealRegistry."
                )
            seen[d.deal_id] = d
        self._deals: dict[str, BPADealMetadata] = seen

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, deal_id: str) -> BPADealMetadata:
        """
        Return metadata for the given deal_id.

        Raises
        ------
        KeyError
            If deal_id is not registered.
        """
        try:
            return self._deals[deal_id]
        except KeyError:
            raise KeyError(
                f"Unknown deal_id '{deal_id}'. "
                f"Registered deals: {sorted(self._deals)}"
            )

    def all_deal_ids(self) -> list[str]:
        """Return all registered deal_ids in sorted order."""
        return sorted(self._deals)

    def __contains__(self, deal_id: str) -> bool:
        return deal_id in self._deals

    def __len__(self) -> int:
        return len(self._deals)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: str | Path) -> "BPADealRegistry":
        """
        Load from a bpa_deals.csv file.

        Required columns
        ----------------
        deal_id, deal_type, inception_date, deal_name, ma_eligible

        inception_date must be ISO 8601 (YYYY-MM-DD).
        deal_type must be "buyout" or "buyin" (case-sensitive).
        ma_eligible must be "true"/"false" or "1"/"0" (case-insensitive).

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If required columns are missing, deal_type is invalid, dates cannot
            be parsed, or any deal_id is duplicated.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BPA deal registry not found: {path}")

        df = pd.read_csv(path)

        _REQUIRED = {"deal_id", "deal_type", "inception_date", "deal_name", "ma_eligible"}
        missing = _REQUIRED - set(df.columns)
        if missing:
            raise ValueError(
                f"bpa_deals.csv missing required columns: {sorted(missing)}"
            )

        deals: list[BPADealMetadata] = []
        errors: list[str] = []

        for idx, row in df.iterrows():
            deal_id   = str(row["deal_id"]).strip()
            deal_type = str(row["deal_type"]).strip()
            name      = str(row["deal_name"]).strip()

            # Validate deal_type
            if deal_type not in ("buyout", "buyin"):
                errors.append(
                    f"Row {idx}: deal_type '{deal_type}' is invalid. "
                    "Must be 'buyout' or 'buyin'."
                )
                continue

            # Parse inception_date
            try:
                inception = date.fromisoformat(str(row["inception_date"]).strip())
            except ValueError:
                errors.append(
                    f"Row {idx}: inception_date '{row['inception_date']}' "
                    "cannot be parsed. Use YYYY-MM-DD."
                )
                continue

            # Parse ma_eligible
            ma_raw = str(row["ma_eligible"]).strip().lower()
            if ma_raw in ("true", "1"):
                ma_eligible = True
            elif ma_raw in ("false", "0"):
                ma_eligible = False
            else:
                errors.append(
                    f"Row {idx}: ma_eligible '{row['ma_eligible']}' is invalid. "
                    "Use true/false or 1/0."
                )
                continue

            deals.append(BPADealMetadata(
                deal_id        = deal_id,
                deal_type      = deal_type,  # type: ignore[arg-type]
                inception_date = inception,
                deal_name      = name,
                ma_eligible    = ma_eligible,
            ))

        if errors:
            raise ValueError(
                "bpa_deals.csv parse errors:\n" + "\n".join(errors)
            )

        return cls(deals)


# ---------------------------------------------------------------------------
# cohort_id derivation (DECISIONS.md §44)
# ---------------------------------------------------------------------------

def make_cohort_id(deal_id: str, population_type: str) -> str:
    """
    Derive the IFRS 17 cohort_id for a contract group.

    Parameters
    ----------
    deal_id : str
        From BPADealRegistry, e.g. "AcmePension_2024Q3".
    population_type : str
        One of "pensioner", "deferred", "dependant", "enhanced".

    Returns
    -------
    str
        e.g. "AcmePension_2024Q3_pensioner"

    Notes
    -----
    This string is the primary key in Ifrs17State and Ifrs17MovementRecord.
    It must never change after a deal goes live.
    """
    return f"{deal_id}_{population_type}"
