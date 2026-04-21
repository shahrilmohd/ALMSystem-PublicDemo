"""
Equity asset model.

An Equity instance represents a single equity holding (e.g. a listed stock
or equity index fund).  _market_value tracks the current total market value
of the holding.

Market value evolution
----------------------
Each period the total return is decomposed into:
    price_return_yr   = equity_total_return_yr − dividend_yield_yr
    monthly_factor    = (1 + price_return_yr)^(1/12)
    new_market_value  = old_market_value × monthly_factor

The price return is applied by step_time() (called once per period).
Dividends are collected as income via project_cashflows() and do NOT reduce
the market value — total return already excludes reinvested income.

Accounting basis
----------------
Equities are always "FVTPL": market value through P&L.
    Balance sheet: _market_value (reset each period).
    P&L:           dividend income + MV movement (price appreciation).

EIR, calibration spread, and default allowance are not applicable
to equities; all three return 0.0.

Rebalancing
-----------
rebalance(target_value, scenario) directly sets _market_value to target_value.
Trade amount = target_value − current_market_value.
Realised G/L ≈ 0 for FVTPL (MV movements already in P&L via step_time).
"""
from __future__ import annotations

from engine.asset.base_asset import AssetCashflows, AssetScenarioPoint, BaseAsset


class Equity(BaseAsset):
    """
    Equity holding with dividend yield and price appreciation.

    Parameters
    ----------
    asset_id : str
        Unique identifier for this holding.
    initial_market_value : float
        Market value at the valuation date (> 0).
    dividend_yield_yr : float
        Annual dividend yield as a decimal, e.g. 0.03 = 3%.
        Must be in [0, 1].  Applied as income each period; does not
        reduce the tracked market value (total return convention).
    """

    def __init__(
        self,
        asset_id: str,
        initial_market_value: float,
        dividend_yield_yr: float = 0.0,
    ) -> None:
        if initial_market_value <= 0.0:
            raise ValueError(
                f"initial_market_value must be > 0, got {initial_market_value}"
            )
        if not (0.0 <= dividend_yield_yr <= 1.0):
            raise ValueError(
                f"dividend_yield_yr must be in [0, 1], got {dividend_yield_yr}"
            )

        self._asset_id            = asset_id
        self._market_value: float = initial_market_value
        self.dividend_yield_yr    = dividend_yield_yr

        # _realised_gl_this_period: accumulated by rebalance(), incorporated
        # and reset by step_time().
        self._realised_gl_this_period: float = 0.0

        # _last_pnl: P&L components from the most recently completed period.
        self._last_pnl: dict = {
            "eir_income":      0.0,
            "coupon_received": 0.0,
            "dividend_income": 0.0,
            "unrealised_gl":   0.0,
            "realised_gl":     0.0,
            "oci_reserve":     0.0,
        }

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def asset_id(self) -> str:
        return self._asset_id

    @property
    def asset_class(self) -> str:
        return "equities"

    @property
    def accounting_basis(self) -> str:
        """Equities are always FVTPL."""
        return "FVTPL"

    # -----------------------------------------------------------------------
    # BaseAsset implementation — valuation
    # -----------------------------------------------------------------------

    def market_value(self, scenario: AssetScenarioPoint) -> float:
        """Returns the internally tracked market value.  Read-only."""
        return self._market_value

    def get_book_value(self) -> float:
        """Book value = market value for FVTPL equities."""
        return self._market_value

    # -----------------------------------------------------------------------
    # BaseAsset implementation — cash flows
    # -----------------------------------------------------------------------

    def project_cashflows(self, scenario: AssetScenarioPoint) -> AssetCashflows:
        """
        Dividend income for this period.

        dividend_income = _market_value × (dividend_yield_yr / 12)

        Dividends are income to the fund; they do not reduce _market_value
        (total return convention).
        """
        dividend = self._market_value * self.dividend_yield_yr / 12.0
        return AssetCashflows(
            timestep=scenario.timestep,
            dividend_income=dividend,
        )

    # -----------------------------------------------------------------------
    # BaseAsset implementation — P&L and spread
    # -----------------------------------------------------------------------

    def get_pnl_components(self) -> dict:
        """P&L decomposition from the most recently completed period."""
        return dict(self._last_pnl)

    def get_calibration_spread(self) -> float:
        """Equities have no calibration spread.  Returns 0.0."""
        return 0.0

    def get_default_allowance(self, lgd_rate: float = 0.40) -> float:
        """Equities have no credit default allowance.  Returns 0.0."""
        return 0.0

    # -----------------------------------------------------------------------
    # BaseAsset implementation — duration
    # -----------------------------------------------------------------------

    def get_duration(self, scenario: AssetScenarioPoint) -> float:
        """Equities have no interest-rate duration.  Returns 0.0."""
        return 0.0

    # -----------------------------------------------------------------------
    # BaseAsset implementation — time stepping
    # -----------------------------------------------------------------------

    def step_time(self, scenario: AssetScenarioPoint) -> None:
        """
        Advance the equity holding one projection period.

        Applies price appreciation (total return minus dividend yield) to
        _market_value and stores P&L components in _last_pnl.

        price_return_yr  = equity_total_return_yr − dividend_yield_yr
        monthly_factor   = (1 + price_return_yr)^(1/12)
        new_market_value = old_market_value × monthly_factor

        Dividend income for this period is computed separately in
        project_cashflows(); it is stored here as part of the P&L record.
        """
        old_mv         = self._market_value
        dividend       = old_mv * self.dividend_yield_yr / 12.0
        price_ret_yr   = scenario.equity_total_return_yr - self.dividend_yield_yr
        monthly_factor = (1.0 + price_ret_yr) ** (1.0 / 12.0)
        new_mv         = old_mv * monthly_factor
        mv_movement    = new_mv - old_mv                  # price appreciation

        self._market_value = new_mv

        self._last_pnl = {
            "eir_income":      0.0,
            "coupon_received": 0.0,
            "dividend_income": dividend,
            "unrealised_gl":   mv_movement,               # FVTPL: in P&L
            "realised_gl":     self._realised_gl_this_period,
            "oci_reserve":     0.0,
        }
        self._realised_gl_this_period = 0.0

    def apply_return(self, scenario: AssetScenarioPoint) -> None:
        """Delegates to step_time() for backwards compatibility."""
        self.step_time(scenario)

    # -----------------------------------------------------------------------
    # BaseAsset implementation — rebalancing
    # -----------------------------------------------------------------------

    def rebalance(self, target_value: float, scenario: AssetScenarioPoint) -> float:
        """
        Set _market_value to target_value.

        For FVTPL equities, realised G/L ≈ 0 (MV movements are already
        recognised in P&L each period via step_time).

        Args:
            target_value : Desired market value (≥ 0).
            scenario     : Current economic conditions (unused for equity
                           since MV is tracked state, not computed from curve).

        Returns:
            Trade amount: positive = buy, negative = sell.
        """
        trade              = target_value - self._market_value
        self._market_value = target_value
        # FVTPL: no realised G/L to accumulate.
        return trade
