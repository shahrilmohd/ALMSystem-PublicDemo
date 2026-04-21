"""
data/tools/bpa_mp_compressor.py
================================
BPA Model Point Compressor — seriatim → group compression.

Compresses individual (seriatim) BPA model point records into grouped model
points by aggregating over age bands and sex.  The module is the authoritative
implementation of the grouping logic used whenever group model points need to
be derived from seriatim data.

Design rationale (DECISIONS.md §54, §55)
-----------------------------------------
Seriatim is always the source of truth.  Group model points are derived from
seriatim by partitioning policies into age/sex cohorts and aggregating.

The compression rate (1 − n_groups/n_seriatim) measures how many records are
eliminated.  A rate of ≥ 90% is the production target for in-payment and
dependant populations.  Smaller populations (enhanced, some deferred) will
naturally have lower rates due to limited record counts.

Grouping keys per population type
----------------------------------
+--------------+----------------------------------------------+
| Population   | Grouping keys                                |
+--------------+----------------------------------------------+
| in_payment   | sex + age_band                               |
| deferred     | sex + age_band + revaluation_type            |
| dependant    | member_sex + member_age_band + dependant_sex |
| enhanced     | sex + age_band                               |
+--------------+----------------------------------------------+

Aggregation rules
-----------------
weight            — sum (total number of lives / expected dependants)
age               — weighted mean (w = weight)
pension_pa        — weighted mean
deferred_pension_pa — weighted mean
gmp_pa            — weighted mean
lpi_cap           — weighted mean
lpi_floor         — weighted mean
rating_years      — weighted mean
deferment_years   — weighted mean
era / nra         — weighted mean
revaluation_cap   — weighted mean
revaluation_floor — weighted mean
member_age        — weighted mean
dependant_age     — weighted mean
revaluation_type  — modal value within each group
tv_eligible       — 1 if any member in the group is TV-eligible, else 0

Usage
-----
    from data.tools.bpa_mp_compressor import BPAMPCompressor, CompressionReport

    compressor = BPAMPCompressor(age_band_width=5)

    # Compress one population type at a time
    group_ip, report = compressor.compress_in_payment(
        seriatim_df_in_payment,
        deal_id="AcmePension_2024Q1",
        deal_abbrev="ACM",
    )

    # Or compress the full combined seriatim file at once
    group_df, reports = compressor.compress_all(
        full_seriatim_df,         # includes population_type column
        deal_abbrev_map={
            "AcmePension_2024Q1":  "ACM",
            "BetaScheme_2024Q3":   "BET",
        },
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CompressionReport
# ---------------------------------------------------------------------------

@dataclass
class CompressionReport:
    """
    Statistics describing the quality of one compression pass.

    Attributes
    ----------
    population_type : str
        One of "in_payment", "deferred", "dependant", "enhanced".
    deal_id : str
        The deal this report applies to.
    n_seriatim : int
        Number of individual records in the input.
    n_groups : int
        Number of group model points produced.
    compression_rate : float
        (n_seriatim − n_groups) / n_seriatim.  1.0 = all records merged into
        one group; 0.0 = no compression.
    weight_preserved : bool
        True if the sum of group weights equals the sum of seriatim weights
        to within floating-point tolerance (1e-6 relative).
    total_weight_seriatim : float
        Sum of weights across all seriatim records.
    total_weight_group : float
        Sum of weights across all group records.
    total_pension_seriatim : float
        Sum of (pension_pa × weight) across seriatim records.
        None when population_type is "deferred" (uses deferred_pension_pa).
    total_pension_group : float
        Sum of (pension_pa × weight) across group records.
        None for deferred.
    pension_error_pct : float
        Percentage difference between group and seriatim total pension.
        None for deferred.
    mean_age_delta : float
        |weighted mean age (group) − weighted mean age (seriatim)|.
    """
    population_type:       str
    deal_id:               str
    n_seriatim:            int
    n_groups:              int
    compression_rate:      float
    weight_preserved:      bool
    total_weight_seriatim: float
    total_weight_group:    float
    total_pension_seriatim: Optional[float]
    total_pension_group:   Optional[float]
    pension_error_pct:     Optional[float]
    mean_age_delta:        float

    def __str__(self) -> str:
        lines = [
            f"  {self.population_type} / {self.deal_id}:",
            f"    Records   : {self.n_seriatim} seriatim → {self.n_groups} groups"
            f" ({self.compression_rate:.1%} compression)",
            f"    Weights   : seriatim={self.total_weight_seriatim:.2f}"
            f"  group={self.total_weight_group:.2f}"
            f"  preserved={self.weight_preserved}",
            f"    Age delta : {self.mean_age_delta:.3f} years",
        ]
        if self.pension_error_pct is not None:
            lines.append(
                f"    Pension Δ : {self.pension_error_pct:+.4f}% "
                f"(seriatim={self.total_pension_seriatim:,.0f}"
                f"  group={self.total_pension_group:,.0f})"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# BPAMPCompressor
# ---------------------------------------------------------------------------

class BPAMPCompressor:
    """
    Compresses BPA seriatim model points into group model points.

    Parameters
    ----------
    age_band_width : int
        Width of each age band in years.  Default: 5.
        Smaller values → more groups, less compression.
        Larger values → fewer groups, more compression but coarser age granularity.
    """

    def __init__(self, age_band_width: int = 5) -> None:
        if age_band_width < 1:
            raise ValueError(f"age_band_width must be >= 1, got {age_band_width}")
        self.age_band_width = age_band_width

    # ------------------------------------------------------------------
    # Public: compress individual population types
    # ------------------------------------------------------------------

    def compress_in_payment(
        self,
        df: pd.DataFrame,
        deal_id: str,
        deal_abbrev: str,
    ) -> tuple[pd.DataFrame, CompressionReport]:
        """
        Compress in-payment seriatim records to group model points.

        Grouping keys: sex + age_band.
        All records must have population_type == "in_payment" (or the column
        may be absent if already pre-filtered).

        Returns
        -------
        group_df : pd.DataFrame
            Group model point DataFrame with population_type, mp_id, deal_id,
            weight, sex, age, pension_pa, lpi_cap, lpi_floor, gmp_pa columns.
        report : CompressionReport
        """
        df = self._pre_filter(df, "in_payment")
        df = df.copy()
        df["_band"] = self._age_band(df["age"])

        rows = []
        for (sex, band), grp in df.groupby(["sex", "_band"], sort=True):
            w = grp["in_force_count"].values
            rows.append({
                "population_type": "in_payment",
                "mp_id":           f"GIP_{deal_abbrev}_{sex}{band}",
                "deal_id":         deal_id,
                "in_force_count":  float(w.sum()),
                "sex":             sex,
                "age":             float(np.average(grp["age"].values, weights=w)),
                "pension_pa":      float(np.average(grp["pension_pa"].values, weights=w)),
                "lpi_cap":         float(np.average(grp["lpi_cap"].values, weights=w)),
                "lpi_floor":       float(np.average(grp["lpi_floor"].values, weights=w)),
                "gmp_pa":          float(np.average(grp["gmp_pa"].values, weights=w)),
            })

        group_df = pd.DataFrame(rows)
        report   = self._report(
            "in_payment", deal_id, df, group_df,
            age_col="age", pension_col="pension_pa", weight_col="in_force_count",
        )
        return group_df, report

    def compress_deferred(
        self,
        df: pd.DataFrame,
        deal_id: str,
        deal_abbrev: str,
    ) -> tuple[pd.DataFrame, CompressionReport]:
        """
        Compress deferred seriatim records to group model points.

        Grouping keys: sex + age_band + revaluation_type.
        revaluation_type is kept as a grouping key because it materially
        affects cashflow revaluation during deferment.
        """
        df = self._pre_filter(df, "deferred")
        df = df.copy()
        df["_band"] = self._age_band(df["age"])

        rows = []
        for (sex, band, rev_type), grp in df.groupby(
            ["sex", "_band", "revaluation_type"], sort=True
        ):
            w = grp["in_force_count"].values
            rows.append({
                "population_type":    "deferred",
                "mp_id":              f"GD_{deal_abbrev}_{sex}{band}_{rev_type[:3]}",
                "deal_id":            deal_id,
                "in_force_count":     float(w.sum()),
                "sex":                sex,
                "age":                float(np.average(grp["age"].values, weights=w)),
                "lpi_cap":            float(np.average(grp["lpi_cap"].values, weights=w)),
                "lpi_floor":          float(np.average(grp["lpi_floor"].values, weights=w)),
                "deferred_pension_pa": float(
                    np.average(grp["deferred_pension_pa"].values, weights=w)
                ),
                "era":                float(np.average(grp["era"].values, weights=w)),
                "nra":                float(np.average(grp["nra"].values, weights=w)),
                "revaluation_type":   rev_type,
                "revaluation_cap":    float(
                    np.average(grp["revaluation_cap"].values, weights=w)
                ),
                "revaluation_floor":  float(
                    np.average(grp["revaluation_floor"].values, weights=w)
                ),
                "deferment_years":    float(
                    np.average(grp["deferment_years"].values, weights=w)
                ),
                "tv_eligible":        int(grp["tv_eligible"].max()),  # 1 if any eligible
            })

        group_df = pd.DataFrame(rows)
        report   = self._report(
            "deferred", deal_id, df, group_df,
            age_col="age", pension_col=None, weight_col="in_force_count",
        )
        return group_df, report

    def compress_dependant(
        self,
        df: pd.DataFrame,
        deal_id: str,
        deal_abbrev: str,
    ) -> tuple[pd.DataFrame, CompressionReport]:
        """
        Compress dependant seriatim records to group model points.

        Grouping keys: member_sex + member_age_band + dependant_sex.
        """
        df = self._pre_filter(df, "dependant")
        df = df.copy()
        df["_mband"] = self._age_band(df["member_age"])

        rows = []
        for (msex, mband, dsex), grp in df.groupby(
            ["member_sex", "_mband", "dependant_sex"], sort=True
        ):
            w = grp["weight"].values
            rows.append({
                "population_type": "dependant",
                "mp_id":           f"GDep_{deal_abbrev}_{msex}{mband}_{dsex}",
                "deal_id":         deal_id,
                "weight":          float(w.sum()),
                "pension_pa":      float(np.average(grp["pension_pa"].values, weights=w)),
                "lpi_cap":         float(np.average(grp["lpi_cap"].values, weights=w)),
                "lpi_floor":       float(np.average(grp["lpi_floor"].values, weights=w)),
                "member_sex":      msex,
                "member_age":      float(np.average(grp["member_age"].values, weights=w)),
                "dependant_sex":   dsex,
                "dependant_age":   float(
                    np.average(grp["dependant_age"].values, weights=w)
                ),
            })

        group_df = pd.DataFrame(rows)
        report   = self._report(
            "dependant", deal_id, df, group_df,
            age_col="member_age", pension_col="pension_pa",
        )
        return group_df, report

    def compress_enhanced(
        self,
        df: pd.DataFrame,
        deal_id: str,
        deal_abbrev: str,
    ) -> tuple[pd.DataFrame, CompressionReport]:
        """
        Compress enhanced (impaired-life) seriatim records to group model points.

        Grouping keys: sex + age_band.  rating_years is averaged within each band.
        """
        df = self._pre_filter(df, "enhanced")
        df = df.copy()
        df["_band"] = self._age_band(df["age"])

        rows = []
        for (sex, band), grp in df.groupby(["sex", "_band"], sort=True):
            w = grp["in_force_count"].values
            rows.append({
                "population_type": "enhanced",
                "mp_id":           f"GE_{deal_abbrev}_{sex}{band}",
                "deal_id":         deal_id,
                "in_force_count":  float(w.sum()),
                "sex":             sex,
                "age":             float(np.average(grp["age"].values, weights=w)),
                "pension_pa":      float(np.average(grp["pension_pa"].values, weights=w)),
                "lpi_cap":         float(np.average(grp["lpi_cap"].values, weights=w)),
                "lpi_floor":       float(np.average(grp["lpi_floor"].values, weights=w)),
                "gmp_pa":          float(np.average(grp["gmp_pa"].values, weights=w)),
                "rating_years":    float(
                    np.average(grp["rating_years"].values, weights=w)
                ),
            })

        group_df = pd.DataFrame(rows)
        report   = self._report(
            "enhanced", deal_id, df, group_df,
            age_col="age", pension_col="pension_pa", weight_col="in_force_count",
        )
        return group_df, report

    # ------------------------------------------------------------------
    # Public: compress_all — convenience entry point
    # ------------------------------------------------------------------

    def compress_all(
        self,
        seriatim_df: pd.DataFrame,
        deal_abbrev_map: dict[str, str],
    ) -> tuple[pd.DataFrame, list[CompressionReport]]:
        """
        Compress all population types in a combined seriatim DataFrame.

        Parameters
        ----------
        seriatim_df : pd.DataFrame
            Full BPA model point file, including a ``population_type`` column
            with values in {"in_payment", "deferred", "dependant", "enhanced"}.
        deal_abbrev_map : dict[str, str]
            Maps deal_id → short abbreviation used in generated mp_ids.
            E.g. {"AcmePension_2024Q1": "ACM", "BetaScheme_2024Q3": "BET"}.

        Returns
        -------
        group_df : pd.DataFrame
            Combined group model point DataFrame (all population types, all deals).
            Columns include population_type plus all type-specific fields; columns
            not relevant to a given population type are absent (the file is in the
            same wide-with-NaN format as the seriatim input).
        reports : list[CompressionReport]
            One report per (population_type × deal_id) combination processed.
        """
        _COMPRESS = {
            "in_payment": self.compress_in_payment,
            "deferred":   self.compress_deferred,
            "dependant":  self.compress_dependant,
            "enhanced":   self.compress_enhanced,
        }
        _ORDER = ["in_payment", "deferred", "dependant", "enhanced"]

        parts: list[pd.DataFrame] = []
        reports: list[CompressionReport] = []

        for pop_type in _ORDER:
            subset = seriatim_df[seriatim_df["population_type"] == pop_type]
            if subset.empty:
                logger.debug("No %s records — skipped.", pop_type)
                continue

            compress_fn = _COMPRESS[pop_type]

            for deal_id in sorted(subset["deal_id"].unique()):
                deal_subset = subset[subset["deal_id"] == deal_id]
                abbrev      = deal_abbrev_map.get(deal_id, deal_id[:3].upper())
                group_part, report = compress_fn(deal_subset, deal_id, abbrev)
                parts.append(group_part)
                reports.append(report)
                logger.info("%s", report)

        if not parts:
            return pd.DataFrame(), reports

        # Merge all parts into a single wide DataFrame
        # (missing columns for each type will be NaN)
        group_df = pd.concat(parts, ignore_index=True, sort=False)
        return group_df, reports

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _age_band(self, ages: pd.Series) -> pd.Series:
        """Return the lower bound of the age band for each age."""
        w = self.age_band_width
        return (ages // w * w).astype(int)

    @staticmethod
    def _pre_filter(df: pd.DataFrame, population_type: str) -> pd.DataFrame:
        """
        Filter to the requested population type if the column is present.
        Drops the population_type column before returning so it does not
        interfere with groupby operations.
        """
        if "population_type" in df.columns:
            df = df[df["population_type"] == population_type].copy()
            df = df.drop(columns=["population_type"])
        return df

    def _report(
        self,
        pop_type:    str,
        deal_id:     str,
        seriatim_df: pd.DataFrame,
        group_df:    pd.DataFrame,
        age_col:     str,
        pension_col: Optional[str],
        weight_col:  str = "weight",   # "in_force_count" for ip/deferred/enhanced; "weight" for dependant
    ) -> CompressionReport:
        """Compute a CompressionReport comparing seriatim and group DataFrames."""
        n_s = len(seriatim_df)
        n_g = len(group_df)
        rate = (n_s - n_g) / n_s if n_s > 0 else 0.0

        w_s = float(seriatim_df[weight_col].sum())
        w_g = float(group_df[weight_col].sum())
        weight_ok = abs(w_g - w_s) <= 1e-6 * max(abs(w_s), 1.0)

        # Weighted mean age
        if n_s > 0 and age_col in seriatim_df.columns:
            w_arr   = seriatim_df[weight_col].values
            age_s   = float(np.average(seriatim_df[age_col].values, weights=w_arr))
        else:
            age_s = 0.0
        if n_g > 0 and age_col in group_df.columns:
            w_arr2  = group_df[weight_col].values
            age_g   = float(np.average(group_df[age_col].values, weights=w_arr2))
        else:
            age_g = 0.0
        age_delta = abs(age_g - age_s)

        # Pension totals
        total_pen_s: Optional[float] = None
        total_pen_g: Optional[float] = None
        pen_err:     Optional[float] = None
        if pension_col and pension_col in seriatim_df.columns:
            total_pen_s = float((seriatim_df[pension_col] * seriatim_df[weight_col]).sum())
            total_pen_g = float((group_df[pension_col] * group_df[weight_col]).sum())
            if total_pen_s > 0:
                pen_err = (total_pen_g - total_pen_s) / total_pen_s * 100.0

        return CompressionReport(
            population_type       = pop_type,
            deal_id               = deal_id,
            n_seriatim            = n_s,
            n_groups              = n_g,
            compression_rate      = rate,
            weight_preserved      = weight_ok,
            total_weight_seriatim = w_s,
            total_weight_group    = w_g,
            total_pension_seriatim= total_pen_s,
            total_pension_group   = total_pen_g,
            pension_error_pct     = pen_err,
            mean_age_delta        = age_delta,
        )
