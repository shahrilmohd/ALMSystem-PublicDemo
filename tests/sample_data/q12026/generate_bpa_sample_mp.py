"""
tests/sample_data/q42025/generate_bpa_sample_mp.py
====================================================
Generate BPA seriatim and group model point files for Q4 2025 sample data.

Seriatim is the authoritative source (DECISIONS.md §54).
Group model points are derived by passing seriatim through BPAMPCompressor
with a 5-year age band and targeting ≥ 90% compression for the in-payment
and dependant populations.

Re-run this script whenever the sample seriatim data needs to be refreshed.
The numpy seed is fixed at 42 so output is fully reproducible.

Output files
------------
mp/seriatim/bpa_model_points_seriatim.csv  — individual policies (weight = 1)
mp/group_mp/bpa_model_points_group_mp.csv  — compressed groups

Scale calibration
-----------------
Two deals:
  AcmePension_2024Q1  (inception 2024-01-15, £30M face asset portfolio target)
  BetaScheme_2024Q3   (inception 2024-07-01, £25M face asset portfolio target)

Total in-payment lives: ~209 (AcmePension ~115, BetaScheme ~94)
Total annual in-payment pension: ~£2.9M → pre-MA BEL target ~£34M
Combined with deferred/dependant/enhanced: total pre-MA BEL ~£45–50M
This matches the £55M face-value asset portfolio (book value ~£55.5M).

Usage
-----
    uv run python tests/sample_data/q42025/generate_bpa_sample_mp.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from data.tools.bpa_mp_compressor import BPAMPCompressor  # noqa: E402

rng = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUT_DIR       = Path(__file__).parent
SERIATIM_PATH = OUT_DIR / "mp" / "seriatim"  / "bpa_model_points_seriatim.csv"
GROUP_PATH    = OUT_DIR / "mp" / "group_mp"  / "bpa_model_points_group_mp.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEAL_ACM = "AcmePension_2024Q1"
DEAL_BET = "BetaScheme_2024Q3"
ABBREV   = {DEAL_ACM: "ACM", DEAL_BET: "BET"}


# ---------------------------------------------------------------------------
# Helper: random ages with given mean and spread within a min/max range
# ---------------------------------------------------------------------------

def _ages(n: int, lo: float, hi: float, *, decimals: int = 1) -> np.ndarray:
    """Uniform random ages in [lo, hi], rounded to 1 decimal place."""
    return np.round(rng.uniform(lo, hi, n), decimals)


def _pensions(
    n: int,
    mean: float,
    cv: float = 0.20,
    lo: float | None = None,
    hi: float | None = None,
) -> np.ndarray:
    """
    Log-normal annual pensions centred at ``mean`` with coefficient of
    variation ``cv``.  Clipped to [lo, hi] when supplied.
    Rounded to nearest £100.
    """
    sigma = np.sqrt(np.log(1 + cv**2))
    mu    = np.log(mean) - 0.5 * sigma**2
    vals  = rng.lognormal(mu, sigma, n)
    if lo is not None:
        vals = np.clip(vals, lo, None)
    if hi is not None:
        vals = np.clip(vals, None, hi)
    return np.round(vals / 100) * 100   # round to nearest £100


def _gmps(pension_pa: np.ndarray, gmp_fraction: float) -> np.ndarray:
    """GMP component: a fixed fraction of the pension, rounded to nearest £50."""
    raw = pension_pa * gmp_fraction
    return np.round(raw / 50) * 50


# ---------------------------------------------------------------------------
# In-payment generator
# ---------------------------------------------------------------------------

def _in_payment_block(
    deal_id: str,
    prefix:  str,
    records: list[dict],
    sex:     str,
    ages_lo: float, ages_hi: float,
    pension_mean: float,
    lpi_cap: float,
    gmp_fraction: float,
    n: int,
    start_idx: int,
) -> int:
    """Append ``n`` in-payment records to ``records``. Returns next start_idx."""
    ages    = _ages(n, ages_lo, ages_hi)
    pensions = _pensions(n, pension_mean, cv=0.22, lo=5000.0)
    gmps    = _gmps(pensions, gmp_fraction) if gmp_fraction > 0 else np.zeros(n)
    for i in range(n):
        idx = start_idx + i
        records.append({
            "population_type": "in_payment",
            "mp_id":           f"IP_{prefix}_{idx:04d}",
            "deal_id":         deal_id,
            "in_force_count":  1.0,
            "sex":             sex,
            "age":             float(ages[i]),
            "pension_pa":      float(pensions[i]),
            "lpi_cap":         lpi_cap,
            "lpi_floor":       0.0,
            "gmp_pa":          float(gmps[i]),
        })
    return start_idx + n


def build_in_payment() -> pd.DataFrame:
    records: list[dict] = []

    # ── AcmePension_2024Q1 ───────────────────────────────────────────────────
    # Target: ~115 lives, ~£1.5M total annual pension
    # Males: 60 policies across three age bands
    idx = 1
    idx = _in_payment_block(DEAL_ACM, "ACM", records, "M", 60.0, 67.5,
                             pension_mean=13_500, lpi_cap=0.05, gmp_fraction=0.20,
                             n=18, start_idx=idx)
    idx = _in_payment_block(DEAL_ACM, "ACM", records, "M", 68.0, 75.5,
                             pension_mean=18_500, lpi_cap=0.05, gmp_fraction=0.15,
                             n=22, start_idx=idx)
    idx = _in_payment_block(DEAL_ACM, "ACM", records, "M", 76.0, 86.0,
                             pension_mean=24_000, lpi_cap=0.03, gmp_fraction=0.10,
                             n=14, start_idx=idx)
    # Females: 55 policies across three age bands
    idx = _in_payment_block(DEAL_ACM, "ACM", records, "F", 61.0, 68.5,
                             pension_mean=10_500, lpi_cap=0.05, gmp_fraction=0.0,
                             n=20, start_idx=idx)
    idx = _in_payment_block(DEAL_ACM, "ACM", records, "F", 69.0, 76.5,
                             pension_mean=9_800, lpi_cap=0.05, gmp_fraction=0.0,
                             n=25, start_idx=idx)
    _in_payment_block(DEAL_ACM, "ACM", records, "F", 77.0, 85.0,
                      pension_mean=8_800, lpi_cap=0.025, gmp_fraction=0.0,
                      n=16, start_idx=idx)

    # ── BetaScheme_2024Q3 ────────────────────────────────────────────────────
    # Target: ~94 lives, ~£1.4M total annual pension
    # Males: 50 policies
    idx = 1
    idx = _in_payment_block(DEAL_BET, "BET", records, "M", 62.0, 69.5,
                             pension_mean=15_500, lpi_cap=0.05, gmp_fraction=0.22,
                             n=15, start_idx=idx)
    idx = _in_payment_block(DEAL_BET, "BET", records, "M", 70.0, 77.5,
                             pension_mean=20_000, lpi_cap=0.05, gmp_fraction=0.16,
                             n=19, start_idx=idx)
    idx = _in_payment_block(DEAL_BET, "BET", records, "M", 78.0, 87.0,
                             pension_mean=27_000, lpi_cap=0.03, gmp_fraction=0.08,
                             n=12, start_idx=idx)
    # Females: 44 policies
    idx = _in_payment_block(DEAL_BET, "BET", records, "F", 63.0, 70.5,
                             pension_mean=11_000, lpi_cap=0.05, gmp_fraction=0.0,
                             n=17, start_idx=idx)
    idx = _in_payment_block(DEAL_BET, "BET", records, "F", 71.0, 77.5,
                             pension_mean=10_800, lpi_cap=0.05, gmp_fraction=0.0,
                             n=21, start_idx=idx)
    _in_payment_block(DEAL_BET, "BET", records, "F", 78.0, 83.0,
                      pension_mean=9_200, lpi_cap=0.025, gmp_fraction=0.0,
                      n=10, start_idx=idx)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Deferred generator
# ---------------------------------------------------------------------------

def _deferred_block(
    deal_id:           str,
    prefix:            str,
    records:           list[dict],
    sex:               str,
    ages_lo: float, ages_hi: float,
    def_pension_mean:  float,
    lpi_cap:           float,
    revaluation_type:  str,
    deferment_lo: float, deferment_hi: float,
    tv_eligible:       int,
    era:               float,
    nra:               float,
    n:                 int,
    start_idx:         int,
) -> int:
    ages        = _ages(n, ages_lo, ages_hi)
    def_pensions = _pensions(n, def_pension_mean, cv=0.25, lo=4000.0)
    deferments  = np.round(rng.uniform(deferment_lo, deferment_hi, n), 1)
    for i in range(n):
        idx = start_idx + i
        records.append({
            "population_type":    "deferred",
            "mp_id":              f"D_{prefix}_{idx:04d}",
            "deal_id":            deal_id,
            "in_force_count":     1.0,
            "sex":                sex,
            "age":                float(ages[i]),
            "lpi_cap":            lpi_cap,
            "lpi_floor":          0.0,
            "deferred_pension_pa": float(def_pensions[i]),
            "era":                era,
            "nra":                nra,
            "revaluation_type":   revaluation_type,
            "revaluation_cap":    lpi_cap,
            "revaluation_floor":  0.0,
            "deferment_years":    float(deferments[i]),
            "tv_eligible":        tv_eligible,
        })
    return start_idx + n


def build_deferred() -> pd.DataFrame:
    records: list[dict] = []

    # ── AcmePension_2024Q1 ───────────────────────────────────────────────────
    # Target: ~26 lives
    idx = 1
    idx = _deferred_block(DEAL_ACM, "ACM", records, "M", 47.0, 55.0,
                           9_000, 0.05, "CPI", 10.0, 15.0, 1, 60.0, 65.0,
                           n=8, start_idx=idx)
    idx = _deferred_block(DEAL_ACM, "ACM", records, "M", 55.0, 62.0,
                           11_500, 0.05, "RPI", 15.0, 20.0, 0, 60.0, 65.0,
                           n=6, start_idx=idx)
    idx = _deferred_block(DEAL_ACM, "ACM", records, "F", 43.0, 52.0,
                           7_000, 0.025, "CPI", 6.0, 10.0, 1, 60.0, 65.0,
                           n=7, start_idx=idx)
    _deferred_block(DEAL_ACM, "ACM", records, "F", 52.0, 60.0,
                    9_200, 0.05, "CPI", 13.0, 18.0, 1, 60.0, 65.0,
                    n=5, start_idx=idx)

    # ── BetaScheme_2024Q3 ────────────────────────────────────────────────────
    # Target: ~18 lives
    idx = 1
    idx = _deferred_block(DEAL_BET, "BET", records, "M", 47.0, 55.0,
                           10_500, 0.05, "CPI", 8.0, 13.0, 1, 60.0, 65.0,
                           n=7, start_idx=idx)
    idx = _deferred_block(DEAL_BET, "BET", records, "M", 55.0, 60.0,
                           12_000, 0.05, "CPI", 14.0, 18.0, 0, 60.0, 65.0,
                           n=5, start_idx=idx)
    _deferred_block(DEAL_BET, "BET", records, "F", 44.0, 55.0,
                    7_800, 0.025, "fixed", 4.0, 8.0, 1, 60.0, 65.0,
                    n=6, start_idx=idx)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Dependant generator
# ---------------------------------------------------------------------------

def build_dependant() -> pd.DataFrame:
    """
    Generate dependant records.

    Each record = one member-dependant pair.
    weight = dependant proportion (probability that this member has a
    qualifying dependant).  In seriatim data this is policy-level; values
    are typically 0.65–0.90.
    """
    records: list[dict] = []

    def _dep_block(
        deal_id: str, prefix: str,
        msex: str, m_age_lo: float, m_age_hi: float,
        dsex: str, dep_age_offset_lo: float, dep_age_offset_hi: float,
        pension_mean: float, lpi_cap: float,
        dep_prop_lo: float, dep_prop_hi: float,
        n: int, start_idx: int,
    ) -> int:
        m_ages  = _ages(n, m_age_lo, m_age_hi)
        offsets = np.round(rng.uniform(dep_age_offset_lo, dep_age_offset_hi, n), 1)
        d_ages  = np.clip(np.round(m_ages + offsets, 1), 45.0, 95.0)
        pensions = _pensions(n, pension_mean, cv=0.25, lo=2_000.0)
        props   = np.round(rng.uniform(dep_prop_lo, dep_prop_hi, n), 2)
        for i in range(n):
            idx = start_idx + i
            records.append({
                "population_type": "dependant",
                "mp_id":           f"Dep_{prefix}_{idx:04d}",
                "deal_id":         deal_id,
                "weight":          float(props[i]),
                "pension_pa":      float(pensions[i]),
                "lpi_cap":         lpi_cap,
                "lpi_floor":       0.0,
                "member_sex":      msex,
                "member_age":      float(m_ages[i]),
                "dependant_sex":   dsex,
                "dependant_age":   float(d_ages[i]),
            })
        return start_idx + n

    # ── AcmePension (~80 records, weight sum ~60) ────────────────────────────
    idx = 1
    # M member → F dependant (most common)
    idx = _dep_block(DEAL_ACM, "ACM", "M", 62.0, 70.0, "F", -8.0, -4.0,
                     5_500, 0.05, 0.80, 0.90, n=22, start_idx=idx)
    idx = _dep_block(DEAL_ACM, "ACM", "M", 70.0, 78.0, "F", -8.0, -3.0,
                     6_200, 0.05, 0.78, 0.88, n=25, start_idx=idx)
    idx = _dep_block(DEAL_ACM, "ACM", "M", 78.0, 86.0, "F", -7.0, -2.0,
                     9_600, 0.03, 0.70, 0.82, n=18, start_idx=idx)
    # F member → M dependant
    _dep_block(DEAL_ACM, "ACM", "F", 68.0, 78.0, "M", 3.0, 8.0,
               4_900, 0.05, 0.65, 0.78, n=15, start_idx=idx)

    # ── BetaScheme (~68 records, weight sum ~51) ─────────────────────────────
    idx = 1
    idx = _dep_block(DEAL_BET, "BET", "M", 63.0, 71.0, "F", -8.0, -4.0,
                     7_750, 0.05, 0.80, 0.90, n=20, start_idx=idx)
    idx = _dep_block(DEAL_BET, "BET", "M", 71.0, 79.0, "F", -7.0, -3.0,
                     10_000, 0.04, 0.78, 0.88, n=22, start_idx=idx)
    idx = _dep_block(DEAL_BET, "BET", "M", 79.0, 86.0, "F", -6.0, -2.0,
                     13_500, 0.03, 0.68, 0.80, n=14, start_idx=idx)
    _dep_block(DEAL_BET, "BET", "F", 70.0, 78.0, "M", 3.0, 8.0,
               5_400, 0.05, 0.65, 0.78, n=12, start_idx=idx)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Enhanced generator
# ---------------------------------------------------------------------------

def build_enhanced() -> pd.DataFrame:
    records: list[dict] = []

    def _enh_block(
        deal_id: str, prefix: str,
        sex: str, ages_lo: float, ages_hi: float,
        pension_mean: float, lpi_cap: float,
        gmp_fraction: float,
        rating_lo: float, rating_hi: float,
        n: int, start_idx: int,
    ) -> int:
        ages      = _ages(n, ages_lo, ages_hi)
        pensions  = _pensions(n, pension_mean, cv=0.20, lo=8_000.0)
        gmps      = _gmps(pensions, gmp_fraction) if gmp_fraction > 0 else np.zeros(n)
        rtgs      = np.round(rng.uniform(rating_lo, rating_hi, n), 1)
        for i in range(n):
            idx = start_idx + i
            records.append({
                "population_type": "enhanced",
                "mp_id":           f"E_{prefix}_{idx:04d}",
                "deal_id":         deal_id,
                "in_force_count":  1.0,
                "sex":             sex,
                "age":             float(ages[i]),
                "pension_pa":      float(pensions[i]),
                "lpi_cap":         lpi_cap,
                "lpi_floor":       0.0,
                "gmp_pa":          float(gmps[i]),
                "rating_years":    float(rtgs[i]),
            })
        return start_idx + n

    # ── AcmePension (~12 lives) ───────────────────────────────────────────────
    idx = 1
    idx = _enh_block(DEAL_ACM, "ACM", "M", 63.0, 70.0, 16_500, 0.05,
                     0.15, 4.0, 7.0, n=5, start_idx=idx)
    idx = _enh_block(DEAL_ACM, "ACM", "M", 70.0, 76.0, 19_000, 0.03,
                     0.18, 6.0, 9.0, n=4, start_idx=idx)
    _enh_block(DEAL_ACM, "ACM", "F", 66.0, 73.0, 12_000, 0.05,
               0.0, 3.0, 6.0, n=3, start_idx=idx)

    # ── BetaScheme (~7 lives) ─────────────────────────────────────────────────
    idx = 1
    idx = _enh_block(DEAL_BET, "BET", "M", 65.0, 72.0, 17_500, 0.05,
                     0.17, 5.0, 8.0, n=4, start_idx=idx)
    _enh_block(DEAL_BET, "BET", "F", 71.0, 77.0, 13_500, 0.03,
               0.0, 5.0, 7.0, n=3, start_idx=idx)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Combine and write seriatim
# ---------------------------------------------------------------------------

def build_seriatim() -> pd.DataFrame:
    parts = [
        build_in_payment(),
        build_deferred(),
        build_dependant(),
        build_enhanced(),
    ]
    df = pd.concat(parts, ignore_index=True, sort=False)

    # Define canonical column order (NaN for non-applicable columns)
    # in_force_count: integer head count for ip/deferred/enhanced
    # weight: fractional dependant proportion (dependant population only)
    COLS = [
        "population_type", "mp_id", "deal_id", "in_force_count", "weight",
        "sex", "age", "pension_pa", "lpi_cap", "lpi_floor", "gmp_pa",
        "rating_years",
        "deferred_pension_pa", "era", "nra",
        "revaluation_type", "revaluation_cap", "revaluation_floor",
        "deferment_years", "tv_eligible",
        "member_sex", "member_age", "dependant_sex", "dependant_age",
    ]
    for col in COLS:
        if col not in df.columns:
            df[col] = float("nan")
    return df[COLS]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("BPA sample model point generator")
    print(f"  seed       : 42")
    print(f"  age bands  : 5 years")
    print("=" * 60)

    # 1. Build seriatim
    print("\n[1/3] Building seriatim model points ...")
    seriatim = build_seriatim()
    for pop in ["in_payment", "deferred", "dependant", "enhanced"]:
        sub = seriatim[seriatim["population_type"] == pop]
        print(f"      {pop:<12s}: {len(sub):>4d} records")
    print(f"      {'TOTAL':<12s}: {len(seriatim):>4d} records")

    # 2. Compress to group MPs
    print("\n[2/3] Compressing to group model points ...")
    compressor = BPAMPCompressor(age_band_width=5)
    group_df, reports = compressor.compress_all(seriatim, deal_abbrev_map=ABBREV)
    print()
    for r in reports:
        print(str(r).encode("ascii", errors="replace").decode("ascii"))
    print(f"\n      {len(seriatim)} seriatim -> {len(group_df)} group records")
    overall_rate = (len(seriatim) - len(group_df)) / len(seriatim)
    print(f"      Overall compression: {overall_rate:.1%}")

    # 3. Write files
    print("\n[3/3] Writing CSV files ...")
    SERIATIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    GROUP_PATH.parent.mkdir(parents=True, exist_ok=True)

    seriatim.to_csv(SERIATIM_PATH, index=False, float_format="%.4f")
    print(f"      Seriatim -> {SERIATIM_PATH.relative_to(SERIATIM_PATH.parents[5])}")

    group_df.to_csv(GROUP_PATH, index=False, float_format="%.4f")
    print(f"      Group    -> {GROUP_PATH.relative_to(GROUP_PATH.parents[5])}")

    print("\nDone.")


if __name__ == "__main__":
    main()
