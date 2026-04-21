"""
Abstract base class and shared data types for all asset models.

Design rules (from CLAUDE.md / ALM_Architecture.md):
  - Models receive inputs; they do not fetch them.
  - The time loop lives only in engine/core/top.py.
  - Results are never stored inside models — all outputs go to ResultStore.
  - Strategies are always injected, never hardcoded.

AssetScenarioPoint
------------------
Carries the economic conditions needed by asset models for a single timestep.
The run-mode orchestrator constructs one AssetScenarioPoint per timestep and
passes it to the AssetModel.  Asset models never fetch data themselves.

In a deterministic run the orchestrator may use a yield curve that shifts
over time (e.g. a forward-curve evolution) or a fixed curve — the asset
models do not know or care which.  In a stochastic run the ScenarioEngine
provides a different curve and equity return for each scenario path/timestep.

BaseAsset — interface contract
------------------------------
Read-only methods (project_cashflows, market_value, get_duration,
get_book_value, get_pnl_components, get_calibration_spread,
get_default_allowance):
    Do not mutate internal state (except get_pnl_components which reads
    state accumulated by step_time / rebalance).

    Bond.market_value() is a pure function of face_value and the injected
    rate curve — there is no tracked price state on Bond.

    Equity.market_value() reads the internally tracked _market_value
    without modifying it.  _market_value is updated only by step_time()
    and rebalance().

Mutating methods:
    rebalance(target_value, scenario) — adjusts holding size, returns trade
                                        amount (cash impact).
    step_time(scenario)               — advances one period:
                                        Bond-AC: amortises book value via EIR.
                                        Bond-FVTPL/FVOCI: resets BV to MV.
                                        Equity: advances market value by
                                        price return.

Accounting basis (DECISIONS.md Section 1)
------------------------------------------
Every asset carries an accounting_basis: "AC", "FVTPL", or "FVOCI".
This designation is fixed at inception and never changes.
All book-value, P&L, and OCI calculations branch on accounting_basis
at the individual asset level.  Aggregation happens only after per-asset
calculations are complete.

P&L components returned by get_pnl_components()
------------------------------------------------
eir_income      : EIR interest income (AC/FVOCI only; 0 for FVTPL).
coupon_received : Cash coupon paid this period (bonds).
dividend_income : Cash dividend received this period (equities).
unrealised_gl   : MV − BV this period.
                  AC:    disclosed only — does NOT enter P&L.
                  FVTPL: recognised in P&L immediately.
                  FVOCI: recognised in OCI reserve, not P&L.
realised_gl     : Gain/loss crystallised by a sale or maturity this period.
                  AC bonds: MV − amortised BV at point of sale.
                  FVTPL bonds: ≈ 0 (MV movements already through P&L).
                  FVOCI bonds: MV − amortised cost; OCI reserve recycled.
oci_reserve     : Cumulative OCI reserve balance (FVOCI only; 0 otherwise).

Asset class labels (must match fund_config.AssetClassWeights field names):
    "bonds", "equities", "derivatives", "cash"
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from engine.curves.rate_curve import RiskFreeRateCurve


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------

@dataclass
class AssetCashflows:
    """
    Income cash flows produced by an asset for one projection timestep.

    Sign convention — all positive = cash received by the fund:
        coupon_income      — bond coupon payments
        dividend_income    — equity dividend payments
        maturity_proceeds  — par redemption at bond maturity

    total_income aggregates all three items.  The fund uses this to
    compute the net cash position: total_income − liability net_outgo.
    """
    timestep:          int
    coupon_income:     float = 0.0
    dividend_income:   float = 0.0
    maturity_proceeds: float = 0.0

    @property
    def total_income(self) -> float:
        return self.coupon_income + self.dividend_income + self.maturity_proceeds


@dataclass
class AssetScenarioPoint:
    """
    Economic conditions for a single projection timestep.

    Constructed by the run-mode orchestrator and passed to all asset models
    at each step.  Asset models never build or fetch this themselves.

    Parameters
    ----------
    timestep : int
        0-based month index of the current projection step.

    rate_curve : RiskFreeRateCurve
        Term structure of risk-free rates at this timestep.
        Used by Bond for market-value discounting and duration.
        The orchestrator controls how the curve evolves over time
        (e.g. forward-rate evolution, scenario path); asset models are
        agnostic to this choice.

    equity_total_return_yr : float
        Annualised total return for equities during this period.
        Equity splits it into price appreciation and dividend yield
        components internally (price_return = total_return - dividend_yield).

    dt : float
        Length of this projection period in years.
        Default 1/12 (monthly) — preserves all existing non-BPA behaviour.
        BPARun sets this to period.year_fraction from ProjectionCalendar,
        giving BPA liability classes the correct hybrid period length without
        any hardcoded 1/12 (DECISIONS.md §27).
    """
    timestep:               int
    rate_curve:             RiskFreeRateCurve
    equity_total_return_yr: float
    # Optional inflation paths — None when not supplied (non-BPA runs).
    # Populated by ScenarioLoader when cpi_annual_rate / rpi_annual_rate columns
    # are present in the ESG CSV (DECISIONS.md §16).
    cpi_annual_rate:        float | None = None
    rpi_annual_rate:        float | None = None
    # Hybrid timestep period length — default 1/12 (backward compatible).
    # BPARun sets this from ProjectionCalendar.period.year_fraction (DECISIONS.md §27).
    dt:                     float = 1.0 / 12.0


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseAsset(ABC):
    """
    Abstract base class for all asset models.

    Concrete subclasses:
        Bond        — fixed-coupon bullet bond (AC / FVTPL / FVOCI)
        Equity      — equity holding with dividend yield + price appreciation
        Derivative  — Phase 1 stub; full implementation in Phase 2

    Each instance represents a single holding.  E.g. a Bond instance holds
    a specific notional amount of a particular bond — face_value is the
    total par value owned, not a per-unit quantity.

    Architectural contract
    ----------------------
    project_cashflows(), market_value(), get_duration(),
    get_book_value(), get_pnl_components(), get_calibration_spread(),
    get_default_allowance()
        Read-only.  Do not mutate internal state (get_pnl_components reads
        state set by step_time / rebalance, but does not mutate it).

    rebalance(target_value, scenario)
        Scales the holding to reach target_value at current market price.
        Mutates face_value (Bond) or _market_value (Equity) and book_value.
        Returns the TRADE AMOUNT (cash impact): positive = buy, negative = sell.
        Side effect: updates _realised_gl_this_period for use in
        get_pnl_components() after the next step_time() call.

    step_time(scenario)
        Advances one period.  For AC bonds: amortises book value via EIR.
        For FVTPL/FVOCI bonds: resets book value to market value.
        For equities: applies price return.
        Computes and stores P&L components for get_pnl_components().
    """

    # --- Identity ---

    @property
    @abstractmethod
    def asset_id(self) -> str:
        """Unique identifier for this asset holding."""

    @property
    @abstractmethod
    def asset_class(self) -> str:
        """
        Asset class label: 'bonds', 'equities', 'derivatives', or 'cash'.
        Used by InvestmentStrategy to match holdings to SAA target weights.
        """

    @property
    @abstractmethod
    def accounting_basis(self) -> str:
        """
        Accounting designation: 'AC', 'FVTPL', or 'FVOCI'.

        Designated at asset inception and immutable for the projection life.
        All book-value, P&L, and OCI calculations branch on this field at
        the individual asset level (DECISIONS.md Section 1).
        """

    # --- Valuation ---

    @abstractmethod
    def market_value(self, scenario: AssetScenarioPoint) -> float:
        """
        Current market value of this holding.

        Read-only — does not mutate internal state.

        Bond: computes PV of remaining cash flows discounted using
        scenario.rate_curve + calibration_spread.
        Equity: returns the internally tracked _market_value.

        Args:
            scenario: Economic conditions for this timestep.

        Returns:
            Market value in currency units, ≥ 0.
        """

    @abstractmethod
    def get_book_value(self) -> float:
        """
        Current carrying (book) value on the balance sheet.

        AC:    amortised cost — walks from purchase price to par via EIR.
        FVTPL: equals market_value() (reset each period by step_time).
        FVOCI: equals market_value() (reset each period by step_time).

        Read-only — does not mutate internal state.

        Returns:
            Book value in currency units.
        """

    # --- Cash flows ---

    @abstractmethod
    def project_cashflows(self, scenario: AssetScenarioPoint) -> AssetCashflows:
        """
        Income cash flows (coupon, dividend, maturity proceeds) this period.

        Read-only.

        Args:
            scenario: Economic conditions for this timestep.

        Returns:
            AssetCashflows with timestep = scenario.timestep.
        """

    # --- P&L decomposition ---

    @abstractmethod
    def get_pnl_components(self) -> dict:
        """
        P&L decomposition from the most recently completed period.

        Returns a dict with keys:
            eir_income      (float) — EIR interest income.
                                      AC/FVOCI: old_book_value × monthly_EIR.
                                      FVTPL: 0.
            coupon_received (float) — Cash coupon received (bonds).
            dividend_income (float) — Cash dividend received (equities).
            unrealised_gl   (float) — MV movement this period.
                                      AC: disclosed, NOT in P&L.
                                      FVTPL: recognised in P&L.
                                      FVOCI: goes to OCI reserve.
            realised_gl     (float) — Gain/loss on any sale or maturity.
            oci_reserve     (float) — Cumulative OCI reserve (FVOCI only).

        Must be called AFTER step_time() to reflect the completed period.
        Returns all zeros if step_time() has not yet been called.

        Read-only.
        """

    # --- Spread and credit ---

    @abstractmethod
    def get_calibration_spread(self) -> float:
        """
        Calibration spread locked at the valuation date.

        The parallel z-spread over the risk-free curve that equates
        PV(cash flows) to the observed market value at the valuation date
        (DECISIONS.md Section 3).  Zero for equities and derivatives.

        Read-only.
        """

    @abstractmethod
    def get_default_allowance(self, lgd_rate: float = 0.40) -> float:
        """
        Expected credit loss on this period's projected cash flows.

        Derived from calibration_spread and an assumed LGD rate
        (DECISIONS.md Section 6).  Zero for equities and derivatives.

        Args:
            lgd_rate: Loss Given Default assumption (e.g. 0.40 = 40%).

        Read-only.
        """

    # --- Duration ---

    @abstractmethod
    def get_duration(self, scenario: AssetScenarioPoint) -> float:
        """
        Macaulay duration in years:  D = Σ [t_yr × CF_t × DF(t)] / MV

        Read-only.

        Returns 0.0 for equities and derivatives where duration is not
        applicable.

        Args:
            scenario: Economic conditions for this timestep.

        Returns:
            Duration in years, ≥ 0.
        """

    # --- Time stepping ---

    @abstractmethod
    def step_time(self, scenario: AssetScenarioPoint) -> None:
        """
        Advance this asset one projection period.

        Bond-AC:
            new_book_value = old_book_value × (1 + monthly_EIR) − monthly_coupon
            Stores eir_income, unrealised_gl in _last_pnl.

        Bond-FVTPL:
            new_book_value = new_market_value
            Stores coupon_received + MV movement as P&L.

        Bond-FVOCI:
            new_book_value = new_market_value  (balance sheet)
            Shadow amortised cost amortised via EIR  (P&L income)
            OCI reserve += MV movement

        Equity:
            _market_value × (1 + price_return_this_period)
            price_return = total_return − dividend_yield

        All accumulated realised_gl from rebalance() calls during this
        period is incorporated and then reset to 0.

        Must be called once per projection timestep, at the end of the
        period (Step 8e in the Fund calculation sequence).

        Args:
            scenario: Economic conditions for this timestep (needed by
                      FVTPL/FVOCI bonds to reprice MV, and by Equity).
        """

    # --- Portfolio management ---

    @abstractmethod
    def rebalance(self, target_value: float, scenario: AssetScenarioPoint) -> float:
        """
        Scale holding size to achieve target market value.

        For Bond: scales face_value proportionally.
        For Equity: scales _market_value directly.

        Computes and stores realised G/L (to be included in get_pnl_components
        after the next step_time call):
            AC bonds sold:   realised_gl = (MV − amortised_BV) × sold_fraction
            FVTPL bonds sold: realised_gl ≈ 0
            FVOCI bonds sold: realised_gl = (MV − amortised_cost) × sold_fraction
                              OCI reserve recycled proportionally.

        AC bonds may NOT be sold for routine rebalancing — this constraint
        is enforced by InvestmentStrategy, not by this method.  If called
        on an AC bond, it executes the trade and records the realised G/L.

        Args:
            target_value: Desired market value after the trade (≥ 0).
            scenario:     Current economic conditions.

        Returns:
            Trade amount in currency units (cash impact).
            Positive = buy (holding increased).
            Negative = sell (holding reduced).
        """
