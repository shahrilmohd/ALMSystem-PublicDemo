"""
engine/matching_adjustment/eligibility.py
==========================================
MA eligibility assessment: four conditions from DECISIONS.md §22.

Condition 1 — Fixed or highly predictable cashflows
    cashflow_type must be "fixed" or "highly_predictable".

Condition 2 — Currency match
    Asset currency must match liability_currency, OR
    has_qualifying_currency_swap must be True (actuary-asserted pass-through).

Condition 3 — No credit risk transfer features
    has_credit_risk_transfer must be False.

Condition 4 — Cashflow matching test
    Cumulative asset CFs (net of default) ≥ cumulative liability CFs at every
    annual time point.

Highly-predictable cap (PS10/24)
    After conditions 1–3, if highly_predictable assets exceed
    ma_highly_predictable_cap (default 35%) of total portfolio PV,
    lowest-ma_contribution_bps assets are removed first.

NOTE: Computation logic is not included in this public demo.
"""
from __future__ import annotations

import pandas as pd


class EligibilityChecker:
    """
    Assess MA eligibility for an asset portfolio.

    Parameters
    ----------
    ma_highly_predictable_cap : float
        Maximum portfolio weight for highly_predictable assets. Default 0.35.
    """

    def __init__(self, ma_highly_predictable_cap: float = 0.35) -> None:
        if not 0.0 <= ma_highly_predictable_cap <= 1.0:
            raise ValueError(
                f"ma_highly_predictable_cap must be in [0, 1], "
                f"got {ma_highly_predictable_cap}"
            )
        self.ma_highly_predictable_cap = ma_highly_predictable_cap

    def check_conditions_1_to_3(
        self,
        assets_df: pd.DataFrame,
        liability_currency: str,
    ) -> pd.DataFrame:
        """
        Assess conditions 1–3 for each asset.

        Required input columns: asset_id, cashflow_type, currency,
        has_credit_risk_transfer, has_qualifying_currency_swap.

        Added output columns: eligible_c1c2c3, fail_reason, swap_passthrough.

        NOTE: Implementation not included in this public demo.
        """
        raise NotImplementedError(
            "MA eligibility logic is not included in the public demo."
        )

    def apply_highly_predictable_cap(
        self,
        eligible_assets: pd.DataFrame,
        asset_pv: "pd.Series",
        ma_contributions: "pd.Series",
    ) -> pd.DataFrame:
        """
        Trim highly_predictable assets so their PV share ≤ ma_highly_predictable_cap.

        Lowest-contributing assets removed first to preserve maximum MA benefit.

        NOTE: Implementation not included in this public demo.
        """
        raise NotImplementedError(
            "MA eligibility logic is not included in the public demo."
        )

    def check_cashflow_matching(
        self,
        asset_cfs: pd.DataFrame,
        liability_cfs: pd.DataFrame,
        net_of_default: bool,
    ) -> tuple[bool, list[int]]:
        """
        Annual cumulative cashflow matching test (condition 4).

        net_of_default must be True; raises ValueError if False.
        Returns (passes, failing_periods).

        NOTE: Implementation not included in this public demo.
        """
        raise NotImplementedError(
            "MA eligibility logic is not included in the public demo."
        )
