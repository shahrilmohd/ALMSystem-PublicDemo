"""
engine/scr/scr_result.py
========================
SCRResult — frozen dataclass holding all SII standard formula stress outputs
for the three BPA-material SCR sub-modules built in Step 22.

Design (DECISIONS.md §51)
--------------------------
SCR is a point-in-time calculation at the valuation date. All three engines
(spread, interest, longevity) consume base-run outputs and return their results
to SCRCalculator, which assembles this single immutable object.

Own funds sign convention
--------------------------
All own_funds_change fields follow the accounting sign: a loss in own funds is
negative, a gain is positive.

  own_funds_change = ΔASSET_MV − ΔBEL
                   = (stressed_asset_mv − base_asset_mv)
                   − (stressed_bel − base_bel)

A spread widening that hurts assets more than the MA offset compensates produces
a negative own_funds_change. SCR headline numbers are always non-negative.

BSCR aggregation
-----------------
This dataclass intentionally does NOT contain a BSCR field. Full correlation-matrix
aggregation across all SII sub-modules is deferred to Step 26 (Phase 4), when the
complete sub-module set (lapse, expense, operational, counterparty, currency) is
built. Aggregating only three sub-modules would produce a materially understated
BSCR and could mislead users.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SCRResult:
    """
    Immutable output of SCRCalculator.compute().

    Spread stress fields
    --------------------
    scr_spread_up_own_funds_change : float
        Change in own funds under spread widening shock.
        Negative = own funds loss; positive = gain.
    scr_spread_down_own_funds_change : float
        Change in own funds under spread tightening shock.
        Typically positive (gain) for a bond portfolio.
    scr_spread : float
        Capital requirement = max(-scr_spread_up_own_funds_change, 0).
        Always >= 0.
    spread_up_asset_mv_change : float
        Total ΔMVS across all bonds under widening shock.
    spread_up_bel_change : float
        ΔBEL (post-MA) under widening shock. Negative when BEL falls
        due to the MA offset increasing the discount rate.
    spread_down_asset_mv_change : float
        Total ΔMVS across all bonds under tightening shock.
    spread_down_bel_change : float
        ΔBEL (post-MA) under tightening shock.

    Interest rate stress fields
    ---------------------------
    scr_interest_up_own_funds_change : float
        Change in own funds under rate up shock.
    scr_interest_down_own_funds_change : float
        Change in own funds under rate down shock.
    scr_interest : float
        Capital requirement = max(-up_change, -down_change, 0).
        Always >= 0.
    rate_up_asset_mv_change : float
        Total ΔMVS under rate up shock.
    rate_up_bel_change : float
        ΔBEL under rate up shock (BEL falls when rates rise).
    rate_down_asset_mv_change : float
        Total ΔMVS under rate down shock.
    rate_down_bel_change : float
        ΔBEL under rate down shock (BEL rises when rates fall).

    Longevity stress fields
    -----------------------
    scr_longevity : float
        Capital requirement = stressed_bel_t0 - base_bel_post_ma.
        Always >= 0 for a portfolio with positive expected outgo.
    longevity_stressed_bel_series : tuple[float, ...]
        Per-period post-MA BEL under 20% mortality improvement.
        One value per projection period (same length as the base BEL series).
        Tuple (not list) for frozen-dataclass compatibility.
        Feeds CostOfCapitalRA.compute() for the IFRS 17 RA (DECISIONS.md §49).

    Governance / input record
    -------------------------
    base_asset_mv : float
        Total asset market value at valuation date (sum across all bonds).
    base_bel_post_ma : float
        Base BEL (post-MA) at t=0 before any stress.
    spread_up_bps : float
        Spread widening shock applied (basis points).
    spread_down_bps : float
        Spread tightening shock applied (basis points).
    rate_up_bps : float
        Rate curve up-shift applied (basis points).
    rate_down_bps : float
        Rate curve down-shift applied (basis points).
    longevity_mortality_stress_factor : float
        Mortality improvement fraction applied (0.20 = 20%).
    """

    # --- Spread stress ---
    scr_spread_up_own_funds_change: float
    scr_spread_down_own_funds_change: float
    scr_spread: float
    spread_up_asset_mv_change: float
    spread_up_bel_change: float
    spread_down_asset_mv_change: float
    spread_down_bel_change: float

    # --- Interest rate stress ---
    scr_interest_up_own_funds_change: float
    scr_interest_down_own_funds_change: float
    scr_interest: float
    rate_up_asset_mv_change: float
    rate_up_bel_change: float
    rate_down_asset_mv_change: float
    rate_down_bel_change: float

    # --- Longevity stress ---
    scr_longevity: float
    longevity_stressed_bel_series: tuple[float, ...]

    # --- Governance ---
    base_asset_mv: float
    base_bel_post_ma: float
    spread_up_bps: float
    spread_down_bps: float
    rate_up_bps: float
    rate_down_bps: float
    longevity_mortality_stress_factor: float
