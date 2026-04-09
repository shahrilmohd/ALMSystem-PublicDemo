"""
Generate sample model point files for the ALM system.

Produces two files:
    tests/sample_data/mp/seriatim/model_points_seriatim.csv
        1,000 rows — one row per policy (in_force_count = 1.0 for each).

    tests/sample_data/mp/group_mp/model_points_group_mp.csv
        Grouped from the seriatim file.
        Groups are formed by: policy_code + attained_age + policy_term_yr
        + duration_yr (= policy_duration_mths // 12).
        Per-policy values (sum_assured, annual_premium, accrued_bonus_per_policy)
        are averaged within each group; in_force_count is summed.

Product mix
-----------
    ENDOW_NONPAR   350 policies
    ENDOW_PAR      150 policies
    TERM           500 policies

Run with:
    uv run python tests/sample_data/generate_sample_mp.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED       = 42
N_ENDOW_NONPAR = 350
N_ENDOW_PAR    = 150
N_TERM         = 500

SERIATIM_PATH  = Path(__file__).parent / "mp/seriatim/model_points_seriatim.csv"
GROUP_MP_PATH  = Path(__file__).parent / "mp/group_mp/model_points_group_mp.csv"


def _generate_policies(
    rng: np.random.Generator,
    n: int,
    policy_code: str,
    id_start: int,
) -> list[dict]:
    """
    Generate n individual policy rows for the given policy_code.

    Assumptions on random ranges
    ----------------------------
    attained_age        : 30 – 55 (uniform over discrete values)
    policy_term_yr      : one of {20, 25, 30}
    policy_duration_mths: 0 to min(policy_term_yr*12 − 1, 180)
                          — maximum 15 years in-force to keep the portfolio young
    sum_assured         :
        ENDOW_NONPAR/PAR  £50,000 – £300,000
        TERM              £100,000 – £600,000
    annual_premium      :
        ENDOW_NONPAR/PAR  2.5% – 5.0% of sum_assured
        TERM              0.3% – 1.2% of sum_assured
    accrued_bonus_per_policy:
        ENDOW_PAR   bonus_rate × sum_assured × duration_years
                    bonus_rate drawn from 1.5% – 4.0%
        all others  0.0
    """
    rows: list[dict] = []
    terms = np.array([20, 25, 30])

    for i in range(n):
        term_yr   = int(rng.choice(terms))
        max_dur_m = min(term_yr * 12 - 1, 180)
        dur_mths  = int(rng.integers(0, max_dur_m + 1))
        age       = int(rng.integers(30, 56))          # 30 – 55 inclusive

        if policy_code in ("ENDOW_NONPAR", "ENDOW_PAR"):
            sa   = round(float(rng.uniform(50_000,  300_000)), 2)
            prem = round(float(sa * rng.uniform(0.025, 0.050)), 2)
        else:                                           # TERM
            sa   = round(float(rng.uniform(100_000, 600_000)), 2)
            prem = round(float(sa * rng.uniform(0.003, 0.012)), 2)

        if policy_code == "ENDOW_PAR":
            dur_yr  = dur_mths // 12
            b_rate  = rng.uniform(0.015, 0.040)
            accrued = round(float(b_rate * sa * dur_yr), 2)
        else:
            accrued = 0.0

        rows.append({
            "group_id":                 f"POL_{id_start + i:05d}",
            "in_force_count":           1.0,
            "sum_assured":              sa,
            "annual_premium":           prem,
            "attained_age":             age,
            "policy_code":              policy_code,
            "policy_term_yr":           term_yr,
            "policy_duration_mths":     dur_mths,
            "accrued_bonus_per_policy": accrued,
        })

    return rows


def generate_seriatim(rng: np.random.Generator) -> pd.DataFrame:
    rows  = _generate_policies(rng, N_ENDOW_NONPAR, "ENDOW_NONPAR", id_start=1)
    rows += _generate_policies(rng, N_ENDOW_PAR,    "ENDOW_PAR",    id_start=N_ENDOW_NONPAR + 1)
    rows += _generate_policies(rng, N_TERM,         "TERM",         id_start=N_ENDOW_NONPAR + N_ENDOW_PAR + 1)
    return pd.DataFrame(rows)


def generate_group_mp(df_seriatim: pd.DataFrame) -> pd.DataFrame:
    """
    Group seriatim model points into representative model points using banding.

    Banding
    -------
    Policies are banded before grouping to produce a manageable number of
    model points with meaningful group sizes — the standard actuarial approach.

    Age band     : floor attained_age to nearest 5 (30, 35, 40, 45, 50, 55)
                   Representative attained_age = band start + 2  (midpoint)
                   e.g. age band 30 -> representative age 32

    Duration band: floor duration in whole years to nearest 5 (0, 5, 10, 15)
                   Representative policy_duration_mths = band_start_yr x 12 + 6
                   i.e. mid-point of the 5-year band in months
                   e.g. duration band 0 -> representative 6 months

    policy_term_yr is already discrete {20, 25, 30} — no banding needed.

    Grouping keys  : policy_code, age_band, policy_term_yr, duration_band
    in_force_count : sum   (total policies in the cell)
    sum_assured    : mean  (representative per-policy value)
    annual_premium : mean  (representative per-policy value)
    accrued_bonus  : mean  (representative per-policy value)

    Consistency check
    -----------------
    The representative duration_mths must be < policy_term_yr * 12.
    Any cell where the mid-point duration meets or exceeds term is capped at
    (policy_term_yr * 12 - 6) months — one month before the final period.
    This preserves the validator constraint without discarding data.
    """
    df = df_seriatim.copy()
    df["duration_yr"]   = df["policy_duration_mths"] // 12

    # --- Apply bands ---
    df["age_band"]      = (df["attained_age"] // 5) * 5          # 30,35,40,45,50,55
    df["duration_band"] = (df["duration_yr"]  // 5) * 5          # 0,5,10,15

    grouped = (
        df.groupby(
            ["policy_code", "age_band", "policy_term_yr", "duration_band"],
            as_index=False,
            sort=True,
        )
        .agg(
            in_force_count           =("in_force_count",           "sum"),
            sum_assured              =("sum_assured",               "mean"),
            annual_premium           =("annual_premium",            "mean"),
            accrued_bonus_per_policy =("accrued_bonus_per_policy",  "mean"),
        )
    )

    # --- Representative values ---
    grouped["attained_age"]          = grouped["age_band"] + 2    # band midpoint
    grouped["policy_duration_mths"]  = grouped["duration_band"] * 12 + 6  # mid-band months

    # Cap duration_mths so it stays strictly below policy_term_yr * 12
    max_dur = grouped["policy_term_yr"] * 12 - 6
    grouped["policy_duration_mths"]  = grouped["policy_duration_mths"].clip(upper=max_dur)
    grouped["policy_duration_mths"]  = grouped["policy_duration_mths"].astype(int)

    # --- Round monetary values ---
    grouped["sum_assured"]              = grouped["sum_assured"].round(2)
    grouped["annual_premium"]           = grouped["annual_premium"].round(2)
    grouped["accrued_bonus_per_policy"] = grouped["accrued_bonus_per_policy"].round(2)
    grouped["in_force_count"]           = grouped["in_force_count"].astype(float)

    # --- Readable group_id ---
    grouped["group_id"] = grouped.apply(
        lambda r: (
            f"GRP_{r['policy_code'][:4]}_"
            f"A{int(r['age_band']):02d}_"
            f"T{int(r['policy_term_yr']):02d}_"
            f"D{int(r['duration_band']):02d}"
        ),
        axis=1,
    )

    cols = [
        "group_id", "in_force_count", "sum_assured", "annual_premium",
        "attained_age", "policy_code", "policy_term_yr",
        "policy_duration_mths", "accrued_bonus_per_policy",
    ]
    return grouped[cols].reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    print("Generating seriatim model points…")
    df_seriatim = generate_seriatim(rng)
    SERIATIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_seriatim.to_csv(SERIATIM_PATH, index=False)
    print(f"  Saved {len(df_seriatim):,} rows to {SERIATIM_PATH}")

    print("Generating group model points…")
    df_group_mp = generate_group_mp(df_seriatim)
    GROUP_MP_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_group_mp.to_csv(GROUP_MP_PATH, index=False)
    print(f"  Saved {len(df_group_mp):,} rows to {GROUP_MP_PATH}")

    # Summary
    print("\nProduct mix (seriatim):")
    print(df_seriatim["policy_code"].value_counts().to_string())
    print("\nGroup MP — groups per product:")
    print(df_group_mp["policy_code"].value_counts().to_string())
    print(f"\nGroup MP — in_force_count range: "
          f"{df_group_mp['in_force_count'].min():.0f} – "
          f"{df_group_mp['in_force_count'].max():.0f}")
    print(f"Group MP — mean group size: "
          f"{df_group_mp['in_force_count'].mean():.2f}")


if __name__ == "__main__":
    main()
