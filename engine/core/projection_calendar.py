"""
engine/core/projection_calendar.py — ProjectionPeriod and ProjectionCalendar

Hybrid timestep design for BPA projections: monthly for the first N years,
annual thereafter. See DECISIONS.md §27 for full design rationale.

This module is a pure data/calculation utility. It has no dependencies on any
other engine module and may be imported freely.

Usage
-----
    cal = ProjectionCalendar(projection_years=60, monthly_years=10)
    for period in cal.periods:
        dt = period.year_fraction   # 1/12 or 1.0
        t  = period.time_in_years   # cumulative from t=0

    df = cal.discount_factor(period_index=120, annual_rate=0.05)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectionPeriod:
    """
    A single period in a hybrid-timestep projection.

    Attributes
    ----------
    period_index : int
        0-based sequential index across the full projection.
    year_fraction : float
        Length of this period in years: 1/12 for monthly, 1.0 for annual.
    is_monthly : bool
        True for monthly periods (0 to monthly_years×12 − 1), False for annual.
    time_in_years : float
        Cumulative elapsed time from t=0 at the START of this period.
    """

    period_index:  int
    year_fraction: float
    is_monthly:    bool
    time_in_years: float


class ProjectionCalendar:
    """
    Generates the ordered sequence of ProjectionPeriod objects for a BPA projection.

    Monthly periods: indices 0 to (monthly_years × 12 − 1) inclusive.
    Annual periods:  indices monthly_years × 12 to end of projection.
    Total periods:   monthly_years × 12 + (projection_years − monthly_years)

    Parameters
    ----------
    projection_years : int
        Total projection horizon in years. Must be > 0 and >= monthly_years.
    monthly_years : int
        Number of years projected at monthly granularity. Default: 10.

    See DECISIONS.md §27.
    """

    def __init__(self, projection_years: int, monthly_years: int = 10) -> None:
        if projection_years <= 0:
            raise ValueError(
                f"projection_years must be positive, got {projection_years}"
            )
        if monthly_years < 0:
            raise ValueError(
                f"monthly_years must be non-negative, got {monthly_years}"
            )
        if monthly_years > projection_years:
            raise ValueError(
                f"monthly_years ({monthly_years}) cannot exceed "
                f"projection_years ({projection_years})"
            )
        self._projection_years = projection_years
        self._monthly_years    = monthly_years
        self._periods: list[ProjectionPeriod] = self._build()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build(self) -> list[ProjectionPeriod]:
        periods: list[ProjectionPeriod] = []
        n_monthly  = self._monthly_years * 12
        dt_monthly = 1.0 / 12.0
        n_annual   = self._projection_years - self._monthly_years

        # Monthly block — use integer division to avoid float accumulation.
        # i / 12.0 is exact when i is a multiple of 12 (e.g. i=120 → 10.0).
        for i in range(n_monthly):
            periods.append(ProjectionPeriod(
                period_index  = i,
                year_fraction = dt_monthly,
                is_monthly    = True,
                time_in_years = i / 12.0,
            ))

        # Annual block — time is an exact integer number of years.
        for j in range(n_annual):
            periods.append(ProjectionPeriod(
                period_index  = n_monthly + j,
                year_fraction = 1.0,
                is_monthly    = False,
                time_in_years = float(self._monthly_years + j),
            ))

        return periods

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def periods(self) -> list[ProjectionPeriod]:
        """Full ordered list of ProjectionPeriod objects."""
        return self._periods

    @property
    def n_periods(self) -> int:
        """Total number of projection periods."""
        return len(self._periods)

    def time_at(self, period_index: int) -> float:
        """
        Cumulative elapsed time in years at the START of period_index.

        period_index == n_periods is valid and returns the total projection
        horizon (the time at the END of the final period).

        Parameters
        ----------
        period_index : int
            Must be in [0, n_periods].

        Raises
        ------
        IndexError
            If period_index < 0 or > n_periods.
        """
        if period_index < 0 or period_index > self.n_periods:
            raise IndexError(
                f"period_index {period_index} out of range "
                f"[0, {self.n_periods}]"
            )
        if period_index == self.n_periods:
            return float(self._projection_years)
        return self._periods[period_index].time_in_years

    def discount_factor(self, period_index: int, annual_rate: float) -> float:
        """
        Discount factor from t=0 to the START of period_index.

        DF = (1 + annual_rate)^(−time_in_years)

        period_index=0 always returns 1.0 (no time has elapsed).
        period_index=n_periods returns the end-of-projection discount factor.

        Parameters
        ----------
        period_index : int
            Target period. Must be in [0, n_periods].
        annual_rate : float
            Annual discount rate as a decimal (e.g. 0.05 for 5%).

        Raises
        ------
        IndexError
            If period_index is out of range.
        ValueError
            If annual_rate < -1.
        """
        if annual_rate <= -1.0:
            raise ValueError(
                f"annual_rate must be > -1, got {annual_rate}"
            )
        t = self.time_at(period_index)
        if t == 0.0 or annual_rate == 0.0:
            return 1.0
        return (1.0 + annual_rate) ** (-t)
