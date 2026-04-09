"""
Fixed-coupon bullet bond asset model.

A Bond instance represents a single holding: a specific notional amount of
one bullet bond.  face_value is the total par value held (e.g. £1,000,000).

Accounting basis (DECISIONS.md Section 1 and ALM_Architecture.md Section 8)
----------------------------------------------------------------------------
Every bond carries an immutable accounting_basis: "AC", "FVTPL", or "FVOCI".

AC (Amortised Cost)
    Balance sheet: amortised book value (walks from purchase price to par).
    P&L:           EIR income only (coupon + discount/premium unwind).
    Unrealised G/L: disclosed separately; does NOT enter P&L.

FVTPL (Fair Value through P&L)
    Balance sheet: market value (reset each period).
    P&L:           coupon income + MV movement (recognised immediately).

FVOCI (Fair Value through Other Comprehensive Income)
    Balance sheet: market value.
    P&L:           EIR income only (computed from shadow amortised cost).
    OCI reserve:   cumulative MV movements (not through P&L; recycled on sale).

Effective Interest Rate (DECISIONS.md Section 2)
-------------------------------------------------
EIR = the annual yield-to-maturity at the purchase date, calculated from the
bond's own cash flows and initial book value (purchase price):

    Solve annual_eir such that:
        Σ_{m=1}^{M} CF_m / (1 + monthly_eir)^m = initial_book_value
    where monthly_eir = (1 + annual_eir)^(1/12) − 1

EIR is locked at purchase and never recalculated.

Each period for AC:
    new_bv = old_bv × (1 + monthly_eir) − monthly_coupon
    eir_income = old_bv × monthly_eir  (total interest income for the period)

EIR for FVTPL bonds is stored but not used for book value (BV = MV always).
EIR for FVOCI bonds drives the P&L income; book value tracks MV separately.

Calibration spread (DECISIONS.md Section 3)
--------------------------------------------
The calibration_spread is the z-spread over the risk-free curve that equates
PV(cash flows) to the observed market value at the valuation date.  It is
locked at valuation and used for all subsequent MV projections.

    Market value: MV = Σ_{h=1}^{remaining} CF_h × rf_DF(h) × exp(−cs × h/12)

Call Bond.calibrate_spread() to solve cs from an observed market value.

Cash flow timing
----------------
All cash flows occur at the END of each monthly period.
    - Months 1 through remaining−1 : monthly coupon only
    - Month remaining               : monthly coupon + par redemption
      (where remaining = maturity_month − current_timestep)

The bond is alive while remaining ≥ 1 (timestep < maturity_month).
Once timestep ≥ maturity_month the bond has redeemed.

Rebalancing
-----------
rebalance(target_value, scenario) scales face_value proportionally so that
market_value(scenario) equals target_value.  calibration_spread is NOT
recalibrated during rebalancing (same bond, different notional).

For AC bonds: realised_gl = (MV − amortised_BV) × sold_fraction.
    The AC no-sale constraint is enforced by InvestmentStrategy, not here.
For FVTPL bonds: realised_gl ≈ 0 (MV ≈ BV, movements already in P&L).
For FVOCI bonds: realised_gl = (MV − shadow_amortised_cost) × sold_fraction;
    OCI reserve recycled proportionally.

Returns trade amount (positive = buy, negative = sell).
"""
from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.optimize import brentq

from engine.asset.base_asset import AssetCashflows, AssetScenarioPoint, BaseAsset


_ZERO_PNL: dict = {
    "eir_income":      0.0,
    "coupon_received": 0.0,
    "dividend_income": 0.0,
    "unrealised_gl":   0.0,
    "realised_gl":     0.0,
    "oci_reserve":     0.0,
}


class Bond(BaseAsset):
    """
    Fixed-coupon bullet bond with full accounting basis support.

    Parameters
    ----------
    asset_id : str
        Unique identifier for this holding.
    face_value : float
        Total par value held (≥ 0).
    annual_coupon_rate : float
        Annual coupon rate as a decimal, e.g. 0.05 = 5%.  Must be in [0, 1].
    maturity_month : int
        Month index (1-based from projection start) when the bond matures.
        Must be ≥ 1.
    accounting_basis : str
        "AC", "FVTPL", or "FVOCI".  Immutable for the projection life.
    initial_book_value : float
        Carrying value at the valuation date (> 0).
        AC:    purchase cost; used to initialise the EIR amortisation.
        FVTPL/FVOCI: market value at valuation date.
    eir : float, optional
        Annual effective interest rate (e.g. 0.069 = 6.9%).
        If not provided it is computed from initial_book_value and the
        bond's cash flows via Bond.calculate_eir().  Always required for
        AC and FVOCI accounting bases; stored but unused for FVTPL.
    calibration_spread : float
        z-spread over the risk-free curve (default 0.0).
        Calibrated via Bond.calibrate_spread() when constructing from
        observed market data.  Locked for the projection life.
    """

    def __init__(
        self,
        asset_id: str,
        face_value: float,
        annual_coupon_rate: float,
        maturity_month: int,
        accounting_basis: Literal["AC", "FVTPL", "FVOCI"],
        initial_book_value: float,
        eir: float | None = None,
        calibration_spread: float = 0.0,
    ) -> None:
        if face_value < 0:
            raise ValueError(f"face_value must be ≥ 0, got {face_value}")
        if not (0.0 <= annual_coupon_rate <= 1.0):
            raise ValueError(
                f"annual_coupon_rate must be in [0, 1], got {annual_coupon_rate}"
            )
        if maturity_month < 1:
            raise ValueError(f"maturity_month must be ≥ 1, got {maturity_month}")
        if accounting_basis not in ("AC", "FVTPL", "FVOCI"):
            raise ValueError(
                f"accounting_basis must be 'AC', 'FVTPL', or 'FVOCI', "
                f"got {accounting_basis!r}"
            )
        if initial_book_value <= 0:
            raise ValueError(
                f"initial_book_value must be > 0, got {initial_book_value}"
            )
        if calibration_spread < 0.0:
            raise ValueError(
                f"calibration_spread must be ≥ 0, got {calibration_spread}"
            )

        self._asset_id          = asset_id
        self.face_value         = face_value
        self.annual_coupon_rate = annual_coupon_rate
        self.maturity_month     = maturity_month
        self._accounting_basis  = accounting_basis
        self.calibration_spread = calibration_spread

        # ---- EIR ----
        # Existing bonds: eir is supplied directly from the asset data file.
        # New bonds purchased during projection: eir is None and must be
        # computed.  For those, the caller should use Bond.calculate_eir()
        # with a scenario to get a curve-consistent EIR, then pass it here.
        # As a fallback (no scenario available), we derive EIR from
        # initial_book_value using flat-rate discounting.
        if eir is not None:
            self._eir = float(eir)
        else:
            self._eir = Bond.calculate_eir(
                face_value=face_value,
                annual_coupon_rate=annual_coupon_rate,
                maturity_month=maturity_month,
                initial_book_value=initial_book_value,
            )
        self._monthly_eir: float = (1.0 + self._eir) ** (1.0 / 12.0) - 1.0

        # ---- Book value state ----
        # _book_value: balance sheet carrying value.
        #   AC:    amortised cost (initially = initial_book_value)
        #   FVTPL: market value  (initially = initial_book_value, reset each period)
        #   FVOCI: market value  (initially = initial_book_value, reset each period)
        self._book_value: float = initial_book_value

        # _amortised_cost: shadow EIR-amortised value used for FVOCI P&L.
        #   AC:    same as _book_value (they track together)
        #   FVTPL: not used but stored for completeness
        #   FVOCI: separate from _book_value; advances via EIR each period
        self._amortised_cost: float = initial_book_value

        # _oci_reserve: cumulative OCI reserve (FVOCI only).
        self._oci_reserve: float = 0.0

        # _realised_gl_this_period: accumulates from rebalance() calls,
        # incorporated and reset by step_time().
        self._realised_gl_this_period: float = 0.0

        # _last_pnl: P&L components from the most recently completed period.
        self._last_pnl: dict = dict(_ZERO_PNL)

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def asset_id(self) -> str:
        return self._asset_id

    @property
    def asset_class(self) -> str:
        return "bonds"

    @property
    def accounting_basis(self) -> str:
        return self._accounting_basis

    @property
    def eir(self) -> float:
        """Annual effective interest rate (locked at purchase)."""
        return self._eir

    @property
    def oci_reserve(self) -> float:
        """Cumulative OCI reserve (FVOCI only; 0 for AC and FVTPL)."""
        return self._oci_reserve

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _remaining(self, timestep: int) -> int:
        """Number of cash-flow periods remaining at the start of timestep."""
        return max(0, self.maturity_month - timestep)

    def _monthly_coupon(self) -> float:
        return self.face_value * self.annual_coupon_rate / 12.0

    def _df(self, scenario: AssetScenarioPoint, h_months: int) -> float:
        """
        Discount factor at h months, adjusted for calibration spread.

        DF_adj(h) = rate_curve.discount_factor(h) × exp(−cs × h/12)
        """
        rf_df = scenario.rate_curve.discount_factor(h_months)
        if self.calibration_spread == 0.0:
            return rf_df
        return rf_df * float(np.exp(-self.calibration_spread * h_months / 12.0))

    @staticmethod
    def _pv_at_spread(
        face_value: float,
        annual_coupon_rate: float,
        remaining: int,
        scenario: AssetScenarioPoint,
        spread: float,
    ) -> float:
        """PV of bond cash flows discounted at risk-free curve + spread."""
        monthly_coupon = face_value * annual_coupon_rate / 12.0
        pv = 0.0
        for h in range(1, remaining + 1):
            cf = monthly_coupon + (face_value if h == remaining else 0.0)
            df = scenario.rate_curve.discount_factor(h) * math.exp(-spread * h / 12.0)
            pv += cf * df
        return pv

    # -----------------------------------------------------------------------
    # Spread calibration
    # -----------------------------------------------------------------------

    @classmethod
    def calibrate_spread(
        cls,
        face_value: float,
        annual_coupon_rate: float,
        maturity_month: int,
        observed_market_value: float,
        scenario: AssetScenarioPoint,
        spread_lo: float = 0.0,
        spread_hi: float = 0.50,
        tol: float = 1e-8,
    ) -> float:
        """
        Calibrate the z-spread that equates PV(cash flows) to the observed
        market value at the valuation date (DECISIONS.md Section 3).

        Solves for cs in:
            Σ_{h=1}^{remaining} CF_h × rf_DF(h) × exp(−cs × h/12)
                = observed_market_value

        This is the z-spread (parallel shift to the continuously-compounded
        risk-free discount curve) implied by the bond's market price.

        The asset data loader calls this when constructing Bond instances from
        input files.  Pass the returned spread as calibration_spread to the
        Bond constructor.

        Args:
            face_value            : Total par value of the bond.
            annual_coupon_rate    : Annual coupon rate, e.g. 0.05.
            maturity_month        : Month when the bond matures (≥ 1).
            observed_market_value : Market price to calibrate against (> 0).
            scenario              : Economic conditions at the valuation date
                                    (typically timestep 0).
            spread_lo             : Lower bound for the solver search (default 0%).
            spread_hi             : Upper bound for the solver search (default 50%).
            tol                   : Solver convergence tolerance (default 1e-8).

        Returns:
            Calibrated z-spread cs ≥ 0.

        Raises:
            ValueError : If observed_market_value ≤ 0, bond is already matured,
                         or the implied spread lies outside [spread_lo, spread_hi].
        """
        if observed_market_value <= 0.0:
            raise ValueError(
                f"observed_market_value must be > 0, got {observed_market_value}"
            )
        remaining = max(0, maturity_month - scenario.timestep)
        if remaining == 0:
            raise ValueError(
                "Bond has already matured at the calibration timestep "
                f"(maturity_month={maturity_month}, timestep={scenario.timestep})."
            )

        def objective(cs: float) -> float:
            return (
                cls._pv_at_spread(
                    face_value, annual_coupon_rate, remaining, scenario, cs
                )
                - observed_market_value
            )

        f_lo = objective(spread_lo)
        f_hi = objective(spread_hi)
        if f_lo * f_hi > 0:
            pv_lo = f_lo + observed_market_value
            pv_hi = f_hi + observed_market_value
            raise ValueError(
                f"Cannot calibrate spread: the implied z-spread for the observed "
                f"market value ({observed_market_value:.2f}) lies outside the "
                f"search range [{spread_lo:.4f}, {spread_hi:.4f}].\n"
                f"  PV at spread={spread_lo:.4f}: {pv_lo:.2f}\n"
                f"  PV at spread={spread_hi:.4f}: {pv_hi:.2f}\n"
                "Widen spread_lo / spread_hi if the bond trades at an extreme price."
            )

        return float(brentq(objective, spread_lo, spread_hi, xtol=tol))

    # -----------------------------------------------------------------------
    # EIR calculation
    # -----------------------------------------------------------------------

    @staticmethod
    def calculate_eir(
        face_value: float,
        annual_coupon_rate: float,
        maturity_month: int,
        initial_book_value: float | None = None,
        scenario: AssetScenarioPoint | None = None,
        calibration_spread: float = 0.0,
        tol: float = 1e-10,
    ) -> float:
        """
        Calculate the annual Effective Interest Rate (DECISIONS.md Section 2).

        EIR is the flat annual yield-to-maturity at the purchase price, locked
        at inception.  It drives AC and FVOCI book value amortisation each period:

            new_bv = old_bv × (1 + monthly_eir) − monthly_coupon
            monthly_eir = (1 + annual_eir)^(1/12) − 1

        Two call patterns
        -----------------
        EXISTING bonds (loaded from asset data file):
            Pass ``eir`` directly to the Bond constructor — the EIR was locked
            at original purchase and is recorded in the input data.
            Do NOT call this method for existing bonds.

        NEW bonds purchased during the projection:
            Call this method.  Supply a ``scenario`` at the purchase timestep
            so the risk-free yield curve is used to compute the initial book
            value (= fair market value at purchase).  The method uses
            ``RiskFreeRateCurve.discount_factor()`` — the same machinery as
            ``Bond.market_value()`` — to compute:

                initial_book_value = Σ_{h=1}^{M} CF_h
                                     × rf_DF(h) × exp(−calibration_spread × h/12)

            It then solves for the flat YTM from that curve-priced book value.
            The EIR is locked from the moment of purchase and never recalculated.

        The resulting EIR is always expressed as a flat annual rate, not a
        z-spread.  This keeps income recognition constant period-to-period
        (IFRS 9 requirement) regardless of how the risk-free curve evolves.

        Relationship to calibration_spread
        ------------------------------------
        The calibration_spread is the z-spread that prices the bond at its
        observed market value (DECISIONS.md Section 3).  They are distinct:
          - calibration_spread drives MARKET VALUE repricing each period.
          - EIR (flat rate) drives BOOK VALUE amortisation.
        For a bond bought at fair value, EIR ≈ blended yield of risk-free
        curve + calibration_spread, but they are not numerically identical.

        Args:
            face_value          : Total par value.
            annual_coupon_rate  : Annual coupon rate as a decimal.
            maturity_month      : Number of monthly periods until maturity (≥ 1).
            initial_book_value  : Carrying value at purchase (> 0).
                                  Supply this for existing bonds whose book
                                  value is known directly.  If None, the method
                                  computes it from ``scenario``.
            scenario            : Economic conditions at the purchase timestep.
                                  Required when ``initial_book_value`` is None.
                                  ``scenario.rate_curve`` (RiskFreeRateCurve)
                                  provides the discount factors used to compute
                                  the initial book value from the yield curve.
            calibration_spread  : z-spread used when deriving initial_book_value
                                  from the curve (default 0.0).  Ignored when
                                  ``initial_book_value`` is supplied directly.
            tol                 : Brent solver convergence tolerance (default 1e-10).

        Returns:
            Annual EIR as a decimal (e.g. 0.069 = 6.9%).

        Raises:
            ValueError: If neither ``initial_book_value`` nor ``scenario`` is
                        given; if computed or supplied book value ≤ 0; or if
                        the solver cannot bracket a solution.
        """
        # --- Step 1: resolve initial book value ---
        if initial_book_value is None:
            if scenario is None:
                raise ValueError(
                    "calculate_eir() requires either initial_book_value or "
                    "a scenario (AssetScenarioPoint) to compute the initial "
                    "book value from the risk-free yield curve."
                )
            remaining = max(0, maturity_month - scenario.timestep)
            if remaining == 0:
                raise ValueError(
                    "Bond has already matured at the given scenario timestep "
                    f"(maturity_month={maturity_month}, "
                    f"timestep={scenario.timestep})."
                )
            # Compute initial book value using the risk-free curve from the
            # scenario — the same formula used by Bond.market_value().
            initial_book_value = Bond._pv_at_spread(
                face_value=face_value,
                annual_coupon_rate=annual_coupon_rate,
                remaining=remaining,
                scenario=scenario,
                spread=calibration_spread,
            )

        if initial_book_value <= 0.0:
            raise ValueError(
                f"initial_book_value must be > 0, got {initial_book_value:.6f}"
            )
        if maturity_month < 1:
            raise ValueError(f"maturity_month must be ≥ 1, got {maturity_month}")

        # --- Step 2: solve flat monthly EIR via brentq ---
        # Find monthly_eir such that PV(cash flows, flat rate) = initial_book_value.
        # Flat-rate discounting gives a constant income rate, satisfying IFRS 9.
        monthly_coupon = face_value * annual_coupon_rate / 12.0

        def pv_at_monthly_rate(mr: float) -> float:
            pv = 0.0
            for m in range(1, maturity_month + 1):
                cf = monthly_coupon + (face_value if m == maturity_month else 0.0)
                pv += cf / (1.0 + mr) ** m
            return pv

        def objective(mr: float) -> float:
            return pv_at_monthly_rate(mr) - initial_book_value

        # Annual range [−50%, +500%] covers all realistic bonds.
        lo, hi = -0.0567, 0.16          # monthly equivalents of −50% / ~+400% pa
        f_lo, f_hi = objective(lo), objective(hi)

        if f_lo * f_hi > 0:             # extend for deeply discounted/premium bonds
            lo, hi = -0.08, 0.30
            f_lo, f_hi = objective(lo), objective(hi)

        if f_lo * f_hi > 0:
            raise ValueError(
                "Cannot compute EIR: solution lies outside the search range. "
                f"initial_book_value={initial_book_value:.4f}, "
                f"face_value={face_value:.4f}, "
                f"annual_coupon_rate={annual_coupon_rate:.4f}, "
                f"maturity_month={maturity_month}."
            )

        monthly_eir = float(brentq(objective, lo, hi, xtol=tol))
        return (1.0 + monthly_eir) ** 12 - 1.0

    # -----------------------------------------------------------------------
    # BaseAsset implementation — valuation
    # -----------------------------------------------------------------------

    def market_value(self, scenario: AssetScenarioPoint) -> float:
        """
        Present value of remaining cash flows.

        MV = Σ_{h=1}^{remaining} CF_h × DF_adj(h, calibration_spread)

        Returns 0.0 when face_value = 0 or the bond has matured.
        """
        t         = scenario.timestep
        remaining = self._remaining(t)

        if self.face_value <= 0.0 or remaining == 0:
            return 0.0

        coupon = self._monthly_coupon()
        mv     = 0.0
        for h in range(1, remaining + 1):
            cf  = coupon + (self.face_value if h == remaining else 0.0)
            mv += cf * self._df(scenario, h)

        return mv

    def get_book_value(self) -> float:
        """
        Current carrying value on the balance sheet.

        AC:    amortised cost (advanced each period by step_time via EIR).
        FVTPL: market value  (reset to MV each period by step_time).
        FVOCI: market value  (reset to MV each period by step_time).
        """
        return self._book_value

    # -----------------------------------------------------------------------
    # BaseAsset implementation — cash flows
    # -----------------------------------------------------------------------

    def project_cashflows(self, scenario: AssetScenarioPoint) -> AssetCashflows:
        """
        Income cash flows for this period.

        Coupon income each active period.  Maturity proceeds (= face_value)
        on the final period (remaining == 1).  Zero once matured.
        """
        t         = scenario.timestep
        remaining = self._remaining(t)

        if self.face_value <= 0.0 or remaining == 0:
            return AssetCashflows(timestep=t)

        coupon  = self._monthly_coupon()
        is_last = (remaining == 1)

        return AssetCashflows(
            timestep=t,
            coupon_income=coupon,
            maturity_proceeds=self.face_value if is_last else 0.0,
        )

    # -----------------------------------------------------------------------
    # BaseAsset implementation — P&L and spread
    # -----------------------------------------------------------------------

    def get_pnl_components(self) -> dict:
        """
        P&L decomposition from the most recently completed period.

        Returns a copy of _last_pnl (set by step_time).
        Keys: eir_income, coupon_received, dividend_income, unrealised_gl,
              realised_gl, oci_reserve.
        """
        return dict(self._last_pnl)

    def get_calibration_spread(self) -> float:
        """The locked calibration z-spread (DECISIONS.md Section 3)."""
        return self.calibration_spread

    def get_default_allowance(self, lgd_rate: float = 0.40) -> float:
        """
        Expected credit loss on this period's cash flows.

        Simplified Phase 1 approach (DECISIONS.md Section 6):
            annual_PD = calibration_spread / lgd_rate  (spread ≈ PD × LGD)
            monthly_PD = 1 − (1 − annual_PD)^(1/12)
            default_allowance = monthly_coupon × monthly_PD × lgd_rate
                              ≈ monthly_coupon × calibration_spread / 12

        Under a credit spread stress, the stressed calibration_spread
        produces a larger default allowance automatically (DECISIONS.md §6).

        Returns 0.0 when the bond has matured or calibration_spread is zero.
        """
        if self.calibration_spread <= 0.0 or self.face_value <= 0.0:
            return 0.0
        if lgd_rate <= 0.0:
            return 0.0

        monthly_coupon = self._monthly_coupon()
        annual_pd      = min(1.0, self.calibration_spread / lgd_rate)
        monthly_pd     = 1.0 - (1.0 - annual_pd) ** (1.0 / 12.0)
        return monthly_coupon * monthly_pd * lgd_rate

    # -----------------------------------------------------------------------
    # BaseAsset implementation — duration
    # -----------------------------------------------------------------------

    def get_duration(self, scenario: AssetScenarioPoint) -> float:
        """
        Macaulay duration in years.

        D = Σ_{h=1}^{remaining} [(h/12) × CF_h × DF_adj(h)] / MV

        Returns 0.0 when face_value = 0 or the bond has matured.
        """
        t         = scenario.timestep
        remaining = self._remaining(t)

        if self.face_value <= 0.0 or remaining == 0:
            return 0.0

        mv = self.market_value(scenario)
        if mv <= 0.0:
            return 0.0

        coupon        = self._monthly_coupon()
        weighted_time = 0.0
        for h in range(1, remaining + 1):
            cf             = coupon + (self.face_value if h == remaining else 0.0)
            weighted_time += (h / 12.0) * cf * self._df(scenario, h)

        return weighted_time / mv

    # -----------------------------------------------------------------------
    # BaseAsset implementation — time stepping
    # -----------------------------------------------------------------------

    def step_time(self, scenario: AssetScenarioPoint) -> None:
        """
        Advance the bond one projection period.

        Computes P&L components for the period just completed and stores them
        in _last_pnl.  Incorporates any realised_gl accumulated from
        rebalance() calls this period, then resets _realised_gl_this_period.

        AC:
            eir_income = old_book_value × monthly_EIR
            new_book_value = old_book_value × (1 + monthly_EIR) − monthly_coupon
            unrealised_gl  = new_market_value − new_book_value  (disclosed only)
            _amortised_cost advances in lockstep with _book_value.

        FVTPL:
            unrealised_gl  = new_market_value − old_book_value  (in P&L)
            new_book_value = new_market_value

        FVOCI:
            eir_income = old_amortised_cost × monthly_EIR  (in P&L)
            new_amortised_cost = old × (1 + monthly_EIR) − monthly_coupon
            unrealised_gl = new_market_value − old_book_value  (to OCI reserve)
            new_book_value = new_market_value
            oci_reserve  += unrealised_gl
        """
        t         = scenario.timestep
        remaining = self._remaining(t)

        if self.face_value <= 0.0 or remaining == 0:
            # Bond matured or fully sold — preserve any realised G/L from this
            # period (e.g. from a forced sell that zeroed the position) before
            # clearing P&L state.
            self._last_pnl = {**dict(_ZERO_PNL), "realised_gl": self._realised_gl_this_period}
            self._realised_gl_this_period = 0.0
            return

        monthly_coupon   = self._monthly_coupon()
        new_mv           = self.market_value(scenario)
        old_bv           = self._book_value
        realised_gl      = self._realised_gl_this_period

        if self._accounting_basis == "AC":
            eir_income  = old_bv * self._monthly_eir
            new_bv      = old_bv * (1.0 + self._monthly_eir) - monthly_coupon
            unrealised  = new_mv - new_bv          # disclosed only, not in P&L

            self._book_value       = new_bv
            self._amortised_cost   = new_bv        # same for AC

            self._last_pnl = {
                "eir_income":      eir_income,
                "coupon_received": monthly_coupon,
                "dividend_income": 0.0,
                "unrealised_gl":   unrealised,
                "realised_gl":     realised_gl,
                "oci_reserve":     0.0,
            }

        elif self._accounting_basis == "FVTPL":
            mv_movement = new_mv - old_bv          # recognised in P&L

            self._book_value     = new_mv
            self._amortised_cost = new_mv          # keep in sync (not used for P&L)

            self._last_pnl = {
                "eir_income":      0.0,
                "coupon_received": monthly_coupon,
                "dividend_income": 0.0,
                "unrealised_gl":   mv_movement,
                "realised_gl":     realised_gl,
                "oci_reserve":     0.0,
            }

        else:  # FVOCI
            old_ac      = self._amortised_cost
            eir_income  = old_ac * self._monthly_eir
            new_ac      = old_ac * (1.0 + self._monthly_eir) - monthly_coupon
            mv_movement = new_mv - old_bv          # goes to OCI reserve

            self._amortised_cost = new_ac
            self._book_value     = new_mv           # balance sheet = MV
            self._oci_reserve   += mv_movement

            self._last_pnl = {
                "eir_income":      eir_income,
                "coupon_received": monthly_coupon,
                "dividend_income": 0.0,
                "unrealised_gl":   mv_movement,
                "realised_gl":     realised_gl,
                "oci_reserve":     self._oci_reserve,
            }

        # Reset accumulated realised G/L for next period.
        self._realised_gl_this_period = 0.0

    # -----------------------------------------------------------------------
    # BaseAsset implementation — rebalancing
    # -----------------------------------------------------------------------

    def rebalance(self, target_value: float, scenario: AssetScenarioPoint) -> float:
        """
        Scale face_value so that market_value(scenario) ≈ target_value.

        Proportional scaling: new_face_value = face_value × (target / current_mv).
        calibration_spread is NOT recalibrated — same bond, different notional.

        Updates _book_value and _amortised_cost proportionally.
        Accumulates realised G/L (per accounting basis) in
        _realised_gl_this_period for collection by step_time().

        Returns 0.0 if the bond has matured or current market value is zero.

        AC bonds: the AC no-sale constraint is enforced by InvestmentStrategy,
        not by this method.  Calling rebalance on an AC bond correctly
        records the realised G/L = (MV − amortised_BV) × sold_fraction.

        Args:
            target_value : Desired market value (≥ 0).
            scenario     : Current economic conditions.

        Returns:
            Trade amount: positive = buy, negative = sell.
        """
        current_mv = self.market_value(scenario)
        if current_mv <= 0.0:
            return 0.0

        trade = target_value - current_mv
        scale = target_value / current_mv

        # Compute realised G/L on the sold portion (scale < 1 means partial sell).
        if scale < 1.0:
            sold_fraction = 1.0 - scale
            if self._accounting_basis == "AC":
                # Crystallise gain/loss: MV portion − book value portion
                realised_gl = (current_mv - self._book_value) * sold_fraction
            elif self._accounting_basis == "FVOCI":
                # Crystallise gain/loss from amortised cost; OCI recycled
                realised_gl = (current_mv - self._amortised_cost) * sold_fraction
                self._oci_reserve *= scale   # recycle OCI for sold portion
            else:  # FVTPL
                # BV = MV always → realised_gl ≈ 0
                realised_gl = 0.0
            self._realised_gl_this_period += realised_gl
        # No realised G/L on a buy (scale ≥ 1).

        # Scale all size-dependent state proportionally.
        self.face_value       *= scale
        self._book_value      *= scale
        self._amortised_cost  *= scale

        return trade
