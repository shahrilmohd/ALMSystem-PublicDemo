"""
Rate curve validation chart generator.

Produces rate_curve_comparison.png in the same directory as this script.

Run with:
    uv run python document/validation/plot_rate_curve.py

The chart shows four panels for the reference curve defined below:
    Panel 1 — Discount factors (FF vs SW), with input knot points marked.
    Panel 2 — Implied annual spot rates (FF vs SW), with input rates marked.
    Panel 3 — Monthly discount factors from 0 to 360 months (BEL view).
    Panel 4 — Continuously-compounded forward rates (FF vs SW).

Reference curve
---------------
    Spot rates: {1yr: 3.0%, 5yr: 3.5%, 10yr: 4.0%, 20yr: 4.2%}
    Smith-Wilson: UFR = 4.2%, alpha = 0.1
    Horizon shown: 0 to 60 years (720 months).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

# Allow importing from the project root regardless of cwd.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")           # headless — no GUI required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from engine.curves.rate_curve import ExtrapolationMethod, RiskFreeRateCurve

# ---------------------------------------------------------------------------
# Reference curve definition
# ---------------------------------------------------------------------------

SPOT_RATES = {1.0: 0.030, 5.0: 0.035, 10.0: 0.040, 20.0: 0.042}
UFR        = 0.042
ALPHA      = 0.1

ff_curve = RiskFreeRateCurve(
    spot_rates=SPOT_RATES,
    extrapolation=ExtrapolationMethod.FLAT_FORWARD,
)
sw_curve = RiskFreeRateCurve(
    spot_rates=SPOT_RATES,
    extrapolation=ExtrapolationMethod.SMITH_WILSON,
    ufr=UFR,
    alpha=ALPHA,
)

# ---------------------------------------------------------------------------
# Evaluation grids
# ---------------------------------------------------------------------------

# Annual grid: 0 to 60 years (fine resolution)
t_yr  = np.linspace(0.001, 60.0, 3_000)           # avoid t=0 for log calculations
t_mths_annual = t_yr * 12.0

# Monthly grid: 0 to 360 months (BEL view)
t_mths_bel = np.arange(0, 361, 1, dtype=float)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dfs(curve: RiskFreeRateCurve, t_months_arr: np.ndarray) -> np.ndarray:
    """Vectorised discount factors from a RiskFreeRateCurve."""
    return np.array([curve.discount_factor(float(t)) for t in t_months_arr])


def implied_spot(df_arr: np.ndarray, t_yr_arr: np.ndarray) -> np.ndarray:
    """Implied annual spot rate: DF^(-1/t) - 1."""
    return df_arr ** (-1.0 / t_yr_arr) - 1.0


def fwd_rate_cc(curve: RiskFreeRateCurve, t_yr_arr: np.ndarray, dt: float = 1e-4
                ) -> np.ndarray:
    """
    Continuously-compounded instantaneous forward rate estimated numerically:
        f(t) ≈ -d(log DF)/dt  using a small forward difference.
    """
    log_df_t    = np.log(dfs(curve, t_yr_arr * 12.0))
    log_df_tdt  = np.log(dfs(curve, (t_yr_arr + dt) * 12.0))
    return -(log_df_tdt - log_df_t) / dt


# ---------------------------------------------------------------------------
# Compute series
# ---------------------------------------------------------------------------

# Panel 1 — Discount factors (annual grid)
df_ff = dfs(ff_curve, t_mths_annual)
df_sw = dfs(sw_curve, t_mths_annual)

# Knot values
knot_t_yr = np.array(sorted(SPOT_RATES.keys()), dtype=float)
knot_df   = np.array([(1 + SPOT_RATES[t]) ** -t for t in knot_t_yr])

# Panel 2 — Implied annual spot rates
spot_ff = implied_spot(df_ff, t_yr)
spot_sw = implied_spot(df_sw, t_yr)
knot_spot = np.array(list(SPOT_RATES.values()))

# Panel 3 — Monthly DFs (BEL view)
df_ff_mthly = dfs(ff_curve, t_mths_bel)
df_sw_mthly = dfs(sw_curve, t_mths_bel)

# Panel 4 — Forward rates
fwd_ff = fwd_rate_cc(ff_curve, t_yr)
fwd_sw = fwd_rate_cc(sw_curve, t_yr)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

COLOUR_FF    = "#1f77b4"   # blue
COLOUR_SW    = "#d62728"   # red
COLOUR_KNOT  = "#2ca02c"   # green
COLOUR_UFR   = "#9467bd"   # purple
T_MAX        = max(SPOT_RATES.keys())

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Risk-Free Rate Curve Validation\n"
    f"Spot rates: {{{', '.join(f'{int(t)}yr: {r*100:.1f}%' for t, r in SPOT_RATES.items())}}}  |  "
    f"UFR: {UFR*100:.1f}%  |  α: {ALPHA}",
    fontsize=12, fontweight="bold"
)

# ---- Panel 1: Discount factors ----
ax = axes[0, 0]
ax.plot(t_yr, df_ff, color=COLOUR_FF, linewidth=1.6, label="Flat-Forward")
ax.plot(t_yr, df_sw, color=COLOUR_SW, linewidth=1.6, linestyle="--", label="Smith-Wilson")
ax.scatter(knot_t_yr, knot_df, color=COLOUR_KNOT, zorder=5, s=60, label="Input knots")
ax.axvline(T_MAX, color="grey", linestyle=":", linewidth=0.8, label=f"T_max = {int(T_MAX)}yr")
ax.set_title("Discount Factors  DF(t)", fontsize=11)
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("DF(t)")
ax.set_xlim(0, 60)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Annotate knot DFs
for t, df_v in zip(knot_t_yr, knot_df):
    ax.annotate(f"{df_v:.4f}", xy=(t, df_v), xytext=(t + 0.5, df_v + 0.02),
                fontsize=7, color=COLOUR_KNOT)

# ---- Panel 2: Implied spot rates ----
ax = axes[0, 1]
ax.plot(t_yr, spot_ff * 100, color=COLOUR_FF, linewidth=1.6, label="Flat-Forward")
ax.plot(t_yr, spot_sw * 100, color=COLOUR_SW, linewidth=1.6, linestyle="--", label="Smith-Wilson")
ax.scatter(knot_t_yr, knot_spot * 100, color=COLOUR_KNOT, zorder=5, s=60, label="Input rates")
ax.axhline(UFR * 100, color=COLOUR_UFR, linestyle=":", linewidth=1.2, label=f"UFR = {UFR*100:.1f}%")
ax.axvline(T_MAX, color="grey", linestyle=":", linewidth=0.8)
ax.set_title("Implied Annual Spot Rates", fontsize=11)
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Spot rate (%)")
ax.set_xlim(0, 60)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ---- Panel 3: Monthly DFs — BEL view ----
ax = axes[1, 0]
ax.plot(t_mths_bel, df_ff_mthly, color=COLOUR_FF, linewidth=1.4, label="Flat-Forward")
ax.plot(t_mths_bel, df_sw_mthly, color=COLOUR_SW, linewidth=1.4, linestyle="--", label="Smith-Wilson")
ax.axvline(T_MAX * 12, color="grey", linestyle=":", linewidth=0.8,
           label=f"T_max = {int(T_MAX * 12)}m")
ax.set_title("Monthly Discount Factors  v(t)  [BEL view]", fontsize=11)
ax.set_xlabel("Months from valuation date")
ax.set_ylabel("v(t)")
ax.set_xlim(0, 360)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Mark representative monthly values
for t_mth in [12, 60, 120, 240, 360]:
    df_v = ff_curve.discount_factor(float(t_mth))
    ax.annotate(f"v({t_mth})={df_v:.4f}", xy=(t_mth, df_v),
                xytext=(t_mth + 5, df_v + 0.02), fontsize=7,
                color=COLOUR_FF, arrowprops=dict(arrowstyle="-", color=COLOUR_FF, lw=0.5))

# ---- Panel 4: Forward rates ----
ax = axes[1, 1]
ax.plot(t_yr, fwd_ff * 100, color=COLOUR_FF, linewidth=1.6, label="Flat-Forward")
ax.plot(t_yr, fwd_sw * 100, color=COLOUR_SW, linewidth=1.6, linestyle="--", label="Smith-Wilson")
ax.axhline(UFR * 100, color=COLOUR_UFR, linestyle=":", linewidth=1.2, label=f"UFR = {UFR*100:.1f}%")
ax.axvline(T_MAX, color="grey", linestyle=":", linewidth=0.8, label=f"T_max = {int(T_MAX)}yr")
ax.set_title("Continuously-Compounded Instantaneous Forward Rates", fontsize=11)
ax.set_xlabel("Maturity (years)")
ax.set_ylabel("Forward rate (%)")
ax.set_xlim(0, 60)
ax.set_ylim(2.0, 6.0)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

output_path = Path(__file__).parent / "rate_curve_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Chart saved to: {output_path}")
