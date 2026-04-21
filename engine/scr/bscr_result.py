"""
engine/scr/bscr_result.py
==========================
BSCRResult — composite output of the full SII BSCR computation.

Design (DECISIONS.md §63)
--------------------------
``BSCRResult`` is the top-level output of ``BSCRCalculator.compute()``. It
carries every sub-module SCR, the aggregated module SCRs, the full BSCR,
operational risk, the total SCR, and the Risk Margin — plus the complete
``SCRStressAssumptions`` snapshot used in the computation for a full audit trail.

``SCRResult`` and ``SCRCalculator`` (Step 22) are kept unchanged for backward
compatibility. ``BSCRResult`` is the new composite for Step 26+.
"""
from __future__ import annotations

from dataclasses import dataclass

from engine.scr.scr_assumptions import SCRStressAssumptions


@dataclass(frozen=True)
class BSCRResult:
    """
    Full SII standard formula BSCR output.

    Sub-module SCRs
    ---------------
    scr_spread : float      Spread stress (Step 22).
    scr_interest : float    Interest rate stress (Step 22).
    scr_longevity : float   Longevity stress (Step 22). BPA: non-zero; conventional: 0.0.
    longevity_stressed_bel_series : tuple[float, ...]
        Per-period stressed BEL series from LongevityStressEngine.
        Forwarded to CostOfCapitalRA for the IFRS 17 RA calculation.
    scr_mortality : float   Mortality stress. Conventional: non-zero; BPA: 0.0.
    scr_lapse : float       Lapse stress (Step 26).
    scr_expense : float     Expense stress (Step 26).
    scr_currency : float    Currency stress (Step 26).
    scr_counterparty : float  Counterparty default (Step 26).

    Aggregated module SCRs
    ----------------------
    scr_market : float
        Market sub-SCR: quadratic form of [interest, spread, currency] using
        market_corr from SCRStressAssumptions.
    scr_life : float
        Life sub-SCR: quadratic form of [mortality, longevity, lapse, expense]
        using life_corr (4×4) from SCRStressAssumptions.
    bscr : float
        Basic BSCR: quadratic form of [market, life, counterparty] using
        module_corr from SCRStressAssumptions. Operational risk excluded
        (SII DR Art 107(3)).
    scr_operational : float
        Operational risk SCR (added linearly to BSCR per Art 107(3)).
    scr_total : float
        Total SCR = bscr + scr_operational.

    Risk Margin
    -----------
    risk_margin : float
        Solvency II Risk Margin via Cost of Capital projection.

    Governance
    ----------
    base_asset_mv : float   Market value of assets at valuation date.
    base_bel_post_ma : float  Base BEL at t=0 after MA adjustment.
    assumptions : SCRStressAssumptions
        Complete snapshot of every parameter used in this computation.
        Provides a full audit trail for review and sensitivity testing.
    """

    # ---- Step-22 sub-module SCRs ----
    scr_spread: float
    scr_interest: float
    scr_longevity: float
    longevity_stressed_bel_series: tuple[float, ...]

    # ---- Step-26 sub-module SCRs ----
    scr_mortality: float
    scr_lapse: float
    scr_expense: float
    scr_currency: float
    scr_counterparty: float

    # ---- Aggregated module SCRs ----
    scr_market: float
    scr_life: float
    bscr: float
    scr_operational: float
    scr_total: float

    # ---- Risk Margin ----
    risk_margin: float

    # ---- Governance ----
    base_asset_mv: float
    base_bel_post_ma: float
    assumptions: SCRStressAssumptions
