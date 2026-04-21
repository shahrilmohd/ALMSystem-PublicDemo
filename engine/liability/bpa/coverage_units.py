"""
engine/liability/bpa/coverage_units.py — BPACoverageUnitProvider

Concrete CoverageUnitProvider for BPA contract groups.

Coverage units for BPA are defined as the present value of expected future
annuity payments discounted at the locked-in rate (DECISIONS.md §35).  This
is a decreasing stock measure:

    units_consumed[t]  = outgo[t] × (1 + r)^(-t_end[t])
    units_remaining[t] = Σ_{s≥t} units_consumed[s]   (reverse cumulative sum)

All PVs are absolute from the inception date (t=0), not conditional on
surviving to period t.  The CSM release fraction at period t is:

    release_fraction(t) = units_consumed(t) / units_remaining(t)

See DECISIONS.md §35 for the full rationale on why PV-based coverage units
are preferred over undiscounted outgo for long-duration BPA contracts.

This class is instantiated once per contract group (cohort_id) per BPA run
by BPARun._run_gmm_engine(), after the backward BEL pass has produced the
full per-period outgo series.
"""
from __future__ import annotations


class BPACoverageUnitProvider:
    """
    CoverageUnitProvider implementation for BPA contract groups.

    All coverage unit values are pre-computed at construction from the
    projection-period outgo series.  GmmEngine calls units_consumed(t) and
    units_remaining(t) at each step without re-projecting cashflows.

    Parameters
    ----------
    period_outgos : list[float]
        Expected net outgo in each projection period, in the same monetary
        unit as the BEL inputs.  One entry per calendar period (monthly or
        annual), as produced by _BPACompositeLiability._cohort_net_outgos.
        Values are typically >= 0; negative values are accepted but unusual.
    locked_in_rate : float
        Annual discount rate at contract group inception (post-MA).
        Must be > -1.  Used to compute absolute PV weights from t=0.
    period_end_times_years : list[float]
        Time from the run start (t=0) to the END of each period, in years.
        Must have the same length as period_outgos.
        For the hybrid BPA calendar: first N_monthly entries are multiples
        of 1/12, subsequent entries are integer years.

    Raises
    ------
    ValueError
        If period_outgos and period_end_times_years have different lengths,
        or if locked_in_rate <= -1.
    """

    def __init__(
        self,
        period_outgos:          list[float],
        locked_in_rate:         float,
        period_end_times_years: list[float],
    ) -> None:
        if len(period_outgos) != len(period_end_times_years):
            raise ValueError(
                f"period_outgos length ({len(period_outgos)}) must equal "
                f"period_end_times_years length ({len(period_end_times_years)})."
            )
        if locked_in_rate <= -1.0:
            raise ValueError(
                f"locked_in_rate must be > -1, got {locked_in_rate}."
            )

        n = len(period_outgos)

        # Absolute PV at locked-in rate for each period's outgo.
        # units_consumed[t] = outgo[t] × (1 + r)^(-t_end[t])
        uc: list[float] = [
            outgo * (1.0 + locked_in_rate) ** (-t_end)
            for outgo, t_end in zip(period_outgos, period_end_times_years)
        ]

        # Reverse-cumulative sum: units_remaining[t] = Σ_{s≥t} uc[s]
        # Length n+1; last element is always 0.0 (no outgo beyond final period).
        ur: list[float] = [0.0] * (n + 1)
        for i in range(n - 1, -1, -1):
            ur[i] = ur[i + 1] + uc[i]

        self._units_consumed: list[float] = uc
        self._units_remaining: list[float] = ur   # length n+1

    # ------------------------------------------------------------------
    # CoverageUnitProvider protocol
    # ------------------------------------------------------------------

    def units_consumed(self, t: int) -> float:
        """
        PV at t=0 of expected outgo in period t, at the locked-in rate.

        Returns
        -------
        float
            Coverage units consumed in period t.  0.0 for t outside [0, N-1].
        """
        if t < 0 or t >= len(self._units_consumed):
            return 0.0
        return self._units_consumed[t]

    def units_remaining(self, t: int) -> float:
        """
        PV at t=0 of all expected future outgo from period t onwards.

        This is the opening remaining balance at the START of period t
        (before consuming period t's units), consistent with the release
        fraction denominator in GmmEngine (DECISIONS.md §35).

        Returns
        -------
        float
            Remaining coverage units at start of period t.
            units_remaining(0) == total_coverage_units.
            units_remaining(N) == 0.0.
            0.0 for t > N.
        """
        if t < 0 or t > len(self._units_consumed):
            return 0.0
        return self._units_remaining[t]

    # ------------------------------------------------------------------
    # Convenience property
    # ------------------------------------------------------------------

    @property
    def total_coverage_units(self) -> float:
        """
        Total coverage units at inception: units_remaining(0).

        Used to initialise Ifrs17State.total_coverage_units when creating
        the inception state for a new contract group.
        """
        return self._units_remaining[0]
