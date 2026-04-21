"""
Risk-free rate curve: spot-rate input, log-linear interpolation on discount
factors, flat-forward and Smith–Wilson extrapolation.

Design
------
Input:
    Annual spot rates at discrete maturities (in years), typically loaded from
    assumption tables (CSV) and injected into the model — rates are never read
    directly from files inside this module.

Discount factors (annual compounding):
    DF(T) = 1 / (1 + r_T) ^ T

Interpolation — log-linear on discount factors:
    For T1 ≤ t ≤ T2:
        log DF(t) = log DF(T1) + (t − T1)/(T2 − T1) × (log DF(T2) − log DF(T1))

Below-T_min extrapolation (linear from DF = 1.0 at t = 0):
    log DF(t) = (t / T_min) × log DF(T_min)
    This is equivalent to applying the same continuously-compounded rate as the
    first segment, down to zero.

Beyond-T_max extrapolation (two methods):

    FLAT_FORWARD
        The continuously-compounded forward rate of the last observable segment
        is extended flat beyond T_max.

        f_last = −Δ log DF / ΔT  (last knot-to-knot segment)
        DF(t)  = DF(T_max) × exp(−f_last × (t − T_max))

    SMITH_WILSON  (EIOPA-style convergence to the Ultimate Forward Rate)
        Forward rate decays exponentially from f_last to ω (the UFR expressed
        as a continuously-compounded intensity):

            ω = log(1 + UFR)          e.g. UFR = 4.2% → ω = 0.041124

        f(s) = ω + (f_last − ω) × exp(−alpha × (s − T_max))

        Integrating to get the log discount factor:
        log DF(t) = log DF(T_max)
                  − ω × Δt
                  − (f_last − ω)/alpha × (1 − exp(−alpha × Δt))
        where Δt = t − T_max.

        Note: this is a simplified approximation of EIOPA's full Smith-Wilson
        kernel (EIOPA-BoS-24-533 §9.7). The full method fits a Wilson kernel
        to the entire par-swap cash-flow matrix. In practice, insurance
        companies load EIOPA's published RFR vector (1Y–150Y) directly as
        spot rate inputs, making the extrapolation method immaterial for
        maturities within the published range.

Public interface
----------------
RiskFreeRateCurve.discount_factor(t_months) → float
    Discount factor v(t) for t months from the valuation date.

RiskFreeRateCurve.flat(rate_yr, **kwargs) → RiskFreeRateCurve
    Convenience constructor for a constant-rate curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# ---------------------------------------------------------------------------
# Extrapolation method
# ---------------------------------------------------------------------------

class ExtrapolationMethod(str, Enum):
    """Method used to extend the curve beyond the last observed maturity."""
    FLAT_FORWARD = "FLAT_FORWARD"
    SMITH_WILSON = "SMITH_WILSON"


# ---------------------------------------------------------------------------
# Rate curve
# ---------------------------------------------------------------------------

@dataclass
class RiskFreeRateCurve:
    """
    Risk-free rate curve built from annual spot rates at discrete maturities.

    Parameters
    ----------
    spot_rates : dict[float, float]
        Annual spot rates keyed by maturity in years.
        E.g. {1.0: 0.03, 5.0: 0.035, 10.0: 0.04, 20.0: 0.045}
        All maturities must be strictly positive; all rates must be non-negative.

    extrapolation : ExtrapolationMethod
        Method for extending the curve beyond the last observed maturity.
        Default: FLAT_FORWARD.

    ufr : float
        Ultimate Forward Rate (annual, e.g. 0.042 for 4.2%) used only by
        SMITH_WILSON extrapolation. Internally converted to a continuously-
        compounded intensity ω = log(1 + ufr) so that the extrapolation is
        self-consistent with the continuously-compounded f_last.
        Typical EIOPA value: 0.042 (4.2% — subject to annual revision).

    alpha : float
        Speed-of-convergence parameter used only by SMITH_WILSON extrapolation.
        Higher alpha = faster convergence to UFR.

    Notes
    -----
    The three numpy arrays (_t_yr, _df, _log_df) are derived in __post_init__
    and are not dataclass fields — they do not appear in __eq__ or __repr__.
    """

    spot_rates:    dict[float, float]
    extrapolation: ExtrapolationMethod = ExtrapolationMethod.FLAT_FORWARD
    ufr:           float               = 0.042
    alpha:         float               = 0.1

    def __post_init__(self) -> None:
        if not self.spot_rates:
            raise ValueError("spot_rates must not be empty.")

        sorted_keys = sorted(self.spot_rates.keys())

        if any(t <= 0.0 for t in sorted_keys):
            raise ValueError(
                f"All maturities must be strictly positive. Got: {sorted_keys}"
            )
        if any(r < 0.0 for r in self.spot_rates.values()):
            raise ValueError("All spot rates must be non-negative.")

        self._t_yr:   np.ndarray = np.array(sorted_keys, dtype=float)
        rates:        np.ndarray = np.array(
            [self.spot_rates[t] for t in sorted_keys], dtype=float
        )
        self._df:     np.ndarray = (1.0 + rates) ** (-self._t_yr)
        self._log_df: np.ndarray = np.log(self._df)

    # -----------------------------------------------------------------------
    # Public interface
    # -----------------------------------------------------------------------

    def discount_factor(self, t_months: float) -> float:
        """
        Discount factor v(t) for t months from the valuation date.

        v(0) = 1.0 by definition.
        v(t) = exp(log_df(t / 12))  for t > 0.

        Parameters
        ----------
        t_months : float
            Time in months (non-negative).

        Returns
        -------
        float
            Discount factor in (0, 1].
        """
        if t_months <= 0.0:
            return 1.0
        return float(np.exp(self._log_df_at(t_months / 12.0)))

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _log_df_at(self, t_yr: float) -> float:
        """Log discount factor at t_yr years (internal, year-based)."""
        if t_yr <= self._t_yr[0]:
            # Linear interpolation from (0, 0) to (T_min, log DF(T_min))
            return (t_yr / self._t_yr[0]) * self._log_df[0]

        if t_yr >= self._t_yr[-1]:
            return self._extrapolate(t_yr)

        # Log-linear interpolation between knots
        idx = int(np.searchsorted(self._t_yr, t_yr)) - 1
        t1, t2  = self._t_yr[idx], self._t_yr[idx + 1]
        ld1, ld2 = self._log_df[idx], self._log_df[idx + 1]
        return ld1 + (t_yr - t1) / (t2 - t1) * (ld2 - ld1)

    def _last_forward_rate(self) -> float:
        """
        Continuously-compounded forward rate of the last observable segment.

        For a single knot: uses the spot-rate-to-zero interpretation
        (the same rate that prices the only observed DF).
        """
        if len(self._t_yr) >= 2:
            T1, T2   = self._t_yr[-2], self._t_yr[-1]
            ld1, ld2 = self._log_df[-2], self._log_df[-1]
            return (ld1 - ld2) / (T2 - T1)
        else:
            # Single knot: f = -log(DF(T)) / T  (flat continuously-compounded)
            return -self._log_df[0] / self._t_yr[0]

    def _extrapolate(self, t_yr: float) -> float:
        """Log discount factor beyond the last observed maturity."""
        T_max  = self._t_yr[-1]
        ld_max = self._log_df[-1]
        f_last = self._last_forward_rate()
        dt     = t_yr - T_max

        if self.extrapolation == ExtrapolationMethod.FLAT_FORWARD:
            # f(t) = f_last for all t > T_max
            return ld_max - f_last * dt

        elif self.extrapolation == ExtrapolationMethod.SMITH_WILSON:
            # Convert annual UFR to continuously-compounded intensity ω.
            # f_last is already a c.c. rate, so the asymptote must also be
            # expressed in c.c. terms for the integral to be self-consistent.
            omega = np.log(1.0 + self.ufr)
            # f(s) = ω + (f_last − ω) × exp(−alpha × (s − T_max))
            # integral_{T_max}^{t} f(s) ds:
            integral = (
                omega * dt
                + (f_last - omega) / self.alpha
                * (1.0 - np.exp(-self.alpha * dt))
            )
            return ld_max - integral

        else:
            raise ValueError(
                f"Unknown extrapolation method: {self.extrapolation!r}"
            )

    # -----------------------------------------------------------------------
    # Convenience constructor
    # -----------------------------------------------------------------------

    @classmethod
    def flat(cls, rate_yr: float, **kwargs) -> RiskFreeRateCurve:
        """
        Create a flat-rate curve where the same annual spot rate applies to
        all maturities.

        A single knot at T = 1 year is sufficient. Below T = 1 yr the
        below-T_min extrapolation produces the correct DF; beyond T = 1 yr
        the flat-forward extrapolation (default) extends at the same rate.

        Parameters
        ----------
        rate_yr : float
            Annual spot rate (e.g. 0.05 for 5%).
        **kwargs
            Passed to the constructor (e.g. extrapolation=SMITH_WILSON).

        Returns
        -------
        RiskFreeRateCurve
        """
        return cls(spot_rates={1.0: rate_yr}, **kwargs)
