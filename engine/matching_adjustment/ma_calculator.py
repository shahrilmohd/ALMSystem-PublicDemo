"""
engine/matching_adjustment/ma_calculator.py
============================================
MACalculator — orchestrates eligibility, FS lookup, and MA benefit computation.

Computation sequence (DECISIONS.md §21–23):
1. Conditions 1–3 eligibility filter (incl. swap passthrough for condition 2)
2. Highly-predictable cap — trim ascending ma_contribution_bps
3. Cashflow matching test (condition 4) — net_of_default guard enforced
4. FS + per-asset MA contributions on eligible subset
5. PV cashflow-weighted MA benefit = Σ w(i) × ma_contribution(i)
6. Populate MAResult with governance metadata from FundamentalSpreadTable

MACalculator is a pure library: DataFrames in, MAResult out.
No knowledge of DeterministicRun, ResultStore, or the projection loop.

NOTE: MACalculator.compute() is not included in this public demo.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from engine.matching_adjustment.eligibility import EligibilityChecker
from engine.matching_adjustment.fundamental_spread import (
    FundamentalSpreadCalculator,
    FundamentalSpreadTable,
)


@dataclass
class MAResult:
    """
    Output of MACalculator.compute().

    Attributes
    ----------
    ma_benefit_bps : float
        Portfolio-level MA benefit in basis points, PV cashflow-weighted.
    eligible_asset_ids : list[str]
    cashflow_test_passes : bool
    failing_periods : list[int]  Empty when cashflow_test_passes is True.
    per_asset_contributions : pd.DataFrame
        Columns: asset_id, rating, seniority, spread_bps, fs_bps,
                 ma_contribution_bps, weight, weighted_contribution_bps.
    fs_table_effective_date : date
    fs_table_source_ref : str
    swap_passthrough_assets : list[str]
    """
    ma_benefit_bps: float
    eligible_asset_ids: list[str]
    cashflow_test_passes: bool
    failing_periods: list[int]
    per_asset_contributions: pd.DataFrame
    fs_table_effective_date: date
    fs_table_source_ref: str
    swap_passthrough_assets: list[str] = field(default_factory=list)


class MACalculator:
    """
    Compute the MA benefit for a BPA asset portfolio.

    Parameters
    ----------
    fs_table : FundamentalSpreadTable
    ma_highly_predictable_cap : float  Default 0.35 per PRA PS10/24.
    """

    def __init__(
        self,
        fs_table: FundamentalSpreadTable,
        ma_highly_predictable_cap: float = 0.35,
    ) -> None:
        self._fs_table = fs_table
        self._fs_calc = FundamentalSpreadCalculator(fs_table)
        self._eligibility = EligibilityChecker(ma_highly_predictable_cap)

    def compute(
        self,
        assets_df: pd.DataFrame,
        asset_cfs: pd.DataFrame,
        liability_cfs: pd.DataFrame,
        liability_currency: str,
        net_of_default: bool = True,
    ) -> MAResult:
        """
        Run the full MA computation sequence and return MAResult.

        Parameters
        ----------
        assets_df : pd.DataFrame
            Required columns: asset_id, cashflow_type, currency,
            has_credit_risk_transfer, has_qualifying_currency_swap,
            rating, seniority, tenor_years, spread_bps.
        asset_cfs : pd.DataFrame  Columns: t, asset_id, cf.
        liability_cfs : pd.DataFrame  Columns: t, cf.
        liability_currency : str
        net_of_default : bool  Must be True.

        NOTE: Implementation not included in this public demo.
        """
        raise NotImplementedError(
            "MA computation logic is not included in the public demo."
        )
