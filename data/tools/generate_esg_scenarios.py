"""
data/tools/generate_esg_scenarios.py
=====================================
Generate an ESG scenario CSV for stochastic testing (Option 2).

Model
-----
Yield curve:
    A level-shift Vasicek model.  Each scenario draws an initial shock from
    N(0, initial_shock_std), then that level factor mean-reverts with speed
    kappa and volatility sigma_level over time.  All maturities shift by the
    same amount (parallel shift) — adequate for testing bonus/TVOG divergence
    across scenarios without a full 3-factor term-structure model.

    Rate(maturity, t) = max(base_rate(maturity) + theta(t), rate_floor)

    Vasicek step (monthly dt = 1/12):
        theta(t+1) = theta(t) + kappa*(theta_bar - theta(t))*dt
                     + sigma_level*sqrt(dt)*z_r

Equity:
    GBM lognormal.  Each monthly timestep draws an annual total return:
        log_return_yr = (mu - 0.5*sigma^2) + sigma*z_eq
        equity_return_yr = exp(log_return_yr) - 1

    Consumed by equity.py as: monthly_factor = (1 + equity_return_yr)^(1/12)

Output columns
--------------
    scenario_id, timestep,
    r_1m, r_12m, r_24m, r_60m, r_120m, r_240m, r_360m,
    equity_return_yr

Base curve
----------
UK-style upward-sloping risk-free spot curve sourced from the Q4 2025
sample rate_curve.csv.  Monthly tenors below 1 year use the 1-year rate
(flat extrapolation — no sub-1yr data in the sample file).

Usage (from project root)
--------------------------
    uv run python data/tools/generate_esg_scenarios.py

    uv run python data/tools/generate_esg_scenarios.py \\
        --scenarios 200 --months 300 \\
        --out tests/sample_data/q42025/esg/esg_scenarios_200_s99.csv \\
        --seed 99

    # To regenerate both valuation-date files in one command:
    uv run python data/tools/generate_esg_scenarios.py --seed 42 \\
        --out tests/sample_data/q42025/esg/esg_scenarios_100_s42.csv

    uv run python data/tools/generate_esg_scenarios.py --seed 42 \\
        --out tests/sample_data/q12026/esg/esg_scenarios_100_s42.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Base UK risk-free spot curve (annual spot rates)
# Tenors match the column names: r_{n}m where n = maturity in months.
# Source: tests/sample_data/q42025/tables/liability/rate_curve.csv
# ---------------------------------------------------------------------------
_BASE_CURVE: dict[str, float] = {
    "r_1m":   0.03540,   # sub-1yr — flat extrapolation from 1yr
    "r_12m":  0.03540,   # 1 year
    "r_24m":  0.03492,   # 2 years
    "r_60m":  0.03665,   # 5 years
    "r_120m": 0.04045,   # 10 years
    "r_240m": 0.04536,   # 20 years
    "r_360m": 0.04589,   # 30 years
}

_RATE_COLS  = list(_BASE_CURVE.keys())
_BASE_RATES = np.array(list(_BASE_CURVE.values()), dtype=float)  # (7,)

_RATE_FLOOR = 0.005   # 50 bps nominal floor — prevents negative rates

# ---------------------------------------------------------------------------
# Default model parameters
# ---------------------------------------------------------------------------
_KAPPA             = 0.10    # Vasicek mean-reversion speed (per year)
_THETA_BAR         = 0.00    # long-run mean of the level shift factor
_SIGMA_LEVEL       = 0.005   # level-factor annual vol (~50 bps)
_INITIAL_SHOCK_STD = 0.010   # cross-sectional std at t=0 (~100 bps)
_MU_EQUITY         = 0.07    # annual equity drift (7%)
_SIGMA_EQUITY      = 0.15    # annual equity vol (15%)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate_scenarios(
    n_scenarios: int,
    n_months: int,
    seed: int,
    kappa: float            = _KAPPA,
    theta_bar: float        = _THETA_BAR,
    sigma_level: float      = _SIGMA_LEVEL,
    initial_shock_std: float = _INITIAL_SHOCK_STD,
    mu_equity: float        = _MU_EQUITY,
    sigma_equity: float     = _SIGMA_EQUITY,
) -> list[dict]:
    """
    Generate ESG scenario rows as a list of dicts (ready for csv.DictWriter).

    Parameters
    ----------
    n_scenarios : int
        Number of independent ESG scenarios.
    n_months : int
        Number of monthly timesteps per scenario (= projection_term_months).
    seed : int
        NumPy random seed for full reproducibility.

    Returns
    -------
    list[dict]
        One dict per (scenario, timestep) row.
        Keys: scenario_id, timestep, r_1m…r_360m, equity_return_yr.
    """
    rng = np.random.default_rng(seed)
    dt  = 1.0 / 12.0   # 1 month in years

    rows: list[dict] = []

    for scen_id in range(1, n_scenarios + 1):
        # ------------------------------------------------------------------
        # Initialise scenario-specific starting state
        # ------------------------------------------------------------------
        theta = rng.normal(0.0, initial_shock_std)   # level shock at t=0

        for t in range(n_months):
            # ---- yield curve ------------------------------------------
            shifted = _BASE_RATES + theta
            shifted = np.maximum(shifted, _RATE_FLOOR)

            # ---- equity return (GBM, annualised) -----------------------
            z_eq = rng.standard_normal()
            log_return_yr   = (mu_equity - 0.5 * sigma_equity ** 2) + sigma_equity * z_eq
            equity_return_yr = float(np.exp(log_return_yr) - 1.0)

            # ---- assemble row ------------------------------------------
            row: dict = {
                "scenario_id":      scen_id,
                "timestep":         t,
                "equity_return_yr": round(equity_return_yr, 6),
            }
            for col, rate in zip(_RATE_COLS, shifted):
                row[col] = round(float(rate), 6)

            rows.append(row)

            # ---- advance level factor (Vasicek) ------------------------
            z_r   = rng.standard_normal()
            theta += (
                kappa * (theta_bar - theta) * dt
                + sigma_level * np.sqrt(dt) * z_r
            )

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate ESG scenario CSV for stochastic testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scenarios", type=int, default=100,
        help="Number of independent ESG scenarios.",
    )
    p.add_argument(
        "--months", type=int, default=240,
        help="Number of monthly timesteps per scenario (= projection_term_months).",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="NumPy random seed for reproducibility.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("tests/sample_data/q42025/esg/esg_scenarios_100_s42.csv"),
        help="Output CSV path.  Parent directories are created if absent.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    total_rows = args.scenarios * args.months
    print(
        f"Generating {args.scenarios} scenarios × {args.months} months "
        f"= {total_rows:,} rows  (seed={args.seed}) ..."
    )

    rows = generate_scenarios(args.scenarios, args.months, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["scenario_id", "timestep"] + _RATE_COLS + ["equity_return_yr"]

    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {total_rows:,} rows to {args.out}")
    _print_summary(rows, args.scenarios)


def _print_summary(rows: list[dict], n_scenarios: int) -> None:
    """Print per-scenario statistics at t=0 as a quick sanity check."""
    t0 = [r for r in rows if r["timestep"] == 0]
    r12_vals  = [r["r_12m"]          for r in t0]
    eq_vals   = [r["equity_return_yr"] for r in t0]
    print(
        f"\nSanity check — t=0 across {n_scenarios} scenarios:\n"
        f"  r_12m  : mean={np.mean(r12_vals):.4f}  "
        f"min={np.min(r12_vals):.4f}  max={np.max(r12_vals):.4f}\n"
        f"  equity : mean={np.mean(eq_vals):.4f}  "
        f"min={np.min(eq_vals):.4f}  max={np.max(eq_vals):.4f}"
    )


if __name__ == "__main__":
    main()
