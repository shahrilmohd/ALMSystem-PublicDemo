"""
Asset portfolio model — holds all asset instances for one fund.

AssetModel owns a list of BaseAsset holdings and provides fund-level
aggregation methods used by fund.py at each time step.

Design rules
------------
- AssetModel never reads files.  Assets are injected at construction.
- AssetModel does not own the cash balance.  Cash is held at Fund level
  and passed in to methods that need it for portfolio-level aggregates.
- The time loop lives in engine/core/top.py.  AssetModel.step_time()
  is called by Fund at Step 8e of the per-time-step sequence.
- Results are never stored in AssetModel.  All outputs go to ResultStore
  (written by Fund after calling AssetModel methods).

Asset classes and accounting bases
-----------------------------------
market_value_by_class(scenario) → {asset_class: float}
    Aggregated by asset_class label ("bonds", "equities", etc.).

market_value_by_basis(scenario) → {"AC": float, "FVTPL": float, "FVOCI": float}
    Aggregated by accounting_basis.  Used by Fund for balance sheet and
    company-level aggregation (AC and FVTPL totals must be kept separate
    per DECISIONS.md Section 1 and ALM_Architecture.md Section 4).
"""
from __future__ import annotations

import logging
from typing import Iterator

from engine.asset.base_asset import AssetCashflows, AssetScenarioPoint, BaseAsset

logger = logging.getLogger(__name__)


class AssetModel:
    """
    Portfolio of BaseAsset holdings for one segregated fund.

    Parameters
    ----------
    assets : list[BaseAsset]
        Initial list of assets.  May be empty (assets added later via
        add_asset).  Duplicate asset_id values are not permitted.
    """

    def __init__(self, assets: list[BaseAsset] | None = None) -> None:
        self._assets: dict[str, BaseAsset] = {}
        for asset in (assets or []):
            self.add_asset(asset)

    # -----------------------------------------------------------------------
    # Portfolio management
    # -----------------------------------------------------------------------

    def add_asset(self, asset: BaseAsset) -> None:
        """
        Add an asset to the portfolio.

        Raises:
            ValueError: If an asset with the same asset_id already exists.
        """
        if asset.asset_id in self._assets:
            raise ValueError(
                f"Asset '{asset.asset_id}' already exists in the portfolio. "
                "Use remove_asset() first if you intend to replace it."
            )
        self._assets[asset.asset_id] = asset

    def remove_asset(self, asset_id: str) -> BaseAsset:
        """
        Remove and return an asset by its asset_id.

        Raises:
            KeyError: If no asset with that id exists.
        """
        if asset_id not in self._assets:
            raise KeyError(f"Asset '{asset_id}' not found in portfolio.")
        return self._assets.pop(asset_id)

    def get_asset(self, asset_id: str) -> BaseAsset:
        """Return the asset with the given id.  Raises KeyError if absent."""
        if asset_id not in self._assets:
            raise KeyError(f"Asset '{asset_id}' not found in portfolio.")
        return self._assets[asset_id]

    def has_asset(self, asset_id: str) -> bool:
        """Return True if an asset with this id exists in the portfolio."""
        return asset_id in self._assets

    def __iter__(self) -> Iterator[BaseAsset]:
        """Iterate over all assets."""
        return iter(self._assets.values())

    def __len__(self) -> int:
        """Number of assets in the portfolio."""
        return len(self._assets)

    # -----------------------------------------------------------------------
    # Aggregation — market values
    # -----------------------------------------------------------------------

    def total_market_value(self, scenario: AssetScenarioPoint) -> float:
        """Sum of market values across all assets."""
        return sum(a.market_value(scenario) for a in self._assets.values())

    def total_book_value(self) -> float:
        """Sum of book values across all assets."""
        return sum(a.get_book_value() for a in self._assets.values())

    def market_value_by_class(self, scenario: AssetScenarioPoint) -> dict[str, float]:
        """
        Market value aggregated by asset_class.

        Returns:
            Dict keyed by asset_class label (e.g. "bonds", "equities").
            Only classes with at least one holding are included.
        """
        result: dict[str, float] = {}
        for asset in self._assets.values():
            cls = asset.asset_class
            result[cls] = result.get(cls, 0.0) + asset.market_value(scenario)
        return result

    def market_value_by_basis(self, scenario: AssetScenarioPoint) -> dict[str, float]:
        """
        Market value aggregated by accounting_basis.

        Returns:
            Dict with keys "AC", "FVTPL", "FVOCI".
            All three keys are always present (value = 0.0 if no holdings).
        """
        result = {"AC": 0.0, "FVTPL": 0.0, "FVOCI": 0.0}
        for asset in self._assets.values():
            result[asset.accounting_basis] += asset.market_value(scenario)
        return result

    def book_value_by_basis(self) -> dict[str, float]:
        """
        Book value aggregated by accounting_basis.

        Returns:
            Dict with keys "AC", "FVTPL", "FVOCI".
        """
        result = {"AC": 0.0, "FVTPL": 0.0, "FVOCI": 0.0}
        for asset in self._assets.values():
            result[asset.accounting_basis] += asset.get_book_value()
        return result

    # -----------------------------------------------------------------------
    # Cash flow collection
    # -----------------------------------------------------------------------

    def collect_cashflows(self, scenario: AssetScenarioPoint) -> AssetCashflows:
        """
        Aggregate income cash flows from all assets for this period.

        Corresponds to Step 2 of the Fund per-time-step sequence
        (ALM_Architecture.md Section 6.2).

        Returns:
            AssetCashflows with timestep = scenario.timestep and aggregated
            coupon_income, dividend_income, maturity_proceeds.
        """
        total_coupon    = 0.0
        total_dividend  = 0.0
        total_maturity  = 0.0
        matured_ids: list[str] = []

        for asset in self._assets.values():
            cf = asset.project_cashflows(scenario)
            total_coupon   += cf.coupon_income
            total_dividend += cf.dividend_income
            total_maturity += cf.maturity_proceeds

            # Flag bonds that mature this period for removal after cashflow collection.
            if cf.maturity_proceeds > 0.0:
                matured_ids.append(asset.asset_id)

        # Remove matured bonds from the portfolio.
        for aid in matured_ids:
            self._assets.pop(aid)
            logger.debug("Bond '%s' matured and removed from portfolio.", aid)

        return AssetCashflows(
            timestep=scenario.timestep,
            coupon_income=total_coupon,
            dividend_income=total_dividend,
            maturity_proceeds=total_maturity,
        )

    # -----------------------------------------------------------------------
    # P&L aggregation
    # -----------------------------------------------------------------------

    def aggregate_pnl(self) -> dict:
        """
        Aggregate get_pnl_components() across all assets.

        Must be called AFTER step_time() so each asset's P&L reflects the
        completed period.

        Returns:
            Dict with summed keys: eir_income, coupon_received, dividend_income,
            unrealised_gl, realised_gl.  oci_reserve is NOT summed here —
            use oci_reserve_by_asset() for FVOCI-specific reporting.
        """
        totals = {
            "eir_income":      0.0,
            "coupon_received": 0.0,
            "dividend_income": 0.0,
            "unrealised_gl":   0.0,
            "realised_gl":     0.0,
        }
        for asset in self._assets.values():
            pnl = asset.get_pnl_components()
            for key in totals:
                totals[key] += pnl.get(key, 0.0)
        return totals

    def pnl_by_basis(self) -> dict[str, dict]:
        """
        P&L aggregated separately by accounting_basis.

        Required for regulatory reporting and attribution (AC and FVTPL
        portfolios must never be aggregated before per-basis calculation
        is complete — DECISIONS.md Section 1).

        Returns:
            Dict keyed by "AC", "FVTPL", "FVOCI", each containing the
            summed P&L components for that basis.
        """
        result: dict[str, dict] = {
            "AC":    {"eir_income": 0.0, "coupon_received": 0.0, "dividend_income": 0.0,
                      "unrealised_gl": 0.0, "realised_gl": 0.0, "oci_reserve": 0.0},
            "FVTPL": {"eir_income": 0.0, "coupon_received": 0.0, "dividend_income": 0.0,
                      "unrealised_gl": 0.0, "realised_gl": 0.0, "oci_reserve": 0.0},
            "FVOCI": {"eir_income": 0.0, "coupon_received": 0.0, "dividend_income": 0.0,
                      "unrealised_gl": 0.0, "realised_gl": 0.0, "oci_reserve": 0.0},
        }
        for asset in self._assets.values():
            basis = asset.accounting_basis
            pnl   = asset.get_pnl_components()
            for key in result[basis]:
                result[basis][key] += pnl.get(key, 0.0)
        return result

    # -----------------------------------------------------------------------
    # Time stepping
    # -----------------------------------------------------------------------

    def step_time(self, scenario: AssetScenarioPoint) -> None:
        """
        Advance all assets one projection period.

        Calls step_time(scenario) on every asset in the portfolio.
        Corresponds to Step 8e of the Fund calculation sequence.

        Args:
            scenario: Economic conditions for the period just completed.
        """
        for asset in self._assets.values():
            asset.step_time(scenario)

    # -----------------------------------------------------------------------
    # Default allowance aggregation
    # -----------------------------------------------------------------------

    def total_default_allowance(self, lgd_rate: float = 0.40) -> float:
        """
        Sum of expected credit losses across all assets for this period.

        Args:
            lgd_rate: Loss Given Default assumption (e.g. 0.40).

        Returns:
            Total default allowance in currency units.
        """
        return sum(
            a.get_default_allowance(lgd_rate) for a in self._assets.values()
        )

    # -----------------------------------------------------------------------
    # Convenience
    # -----------------------------------------------------------------------

    def asset_ids(self) -> list[str]:
        """Return list of all asset_id values in the portfolio."""
        return list(self._assets.keys())

    def assets_by_class(self, asset_class: str) -> list[BaseAsset]:
        """Return all assets with the given asset_class label."""
        return [a for a in self._assets.values() if a.asset_class == asset_class]

    def assets_by_basis(self, basis: str) -> list[BaseAsset]:
        """Return all assets with the given accounting_basis."""
        return [a for a in self._assets.values() if a.accounting_basis == basis]
