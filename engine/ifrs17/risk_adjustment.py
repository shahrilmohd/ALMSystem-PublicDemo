"""
engine/ifrs17/risk_adjustment.py — CostOfCapitalRA

Computes the IFRS 17 Risk Adjustment (RA) for non-financial risk using the
Cost of Capital (CoC) method.

The RA represents compensation required for bearing uncertainty in the amount
and timing of cashflows from non-financial risks. For BPA the dominant
non-financial risk is longevity risk.

Formula (DECISIONS.md §28, §24):
    RA = Σ [ CoC_rate × SCR(t) × DF(t) ]   for t = 1 to N

where:
    CoC_rate = 6% p.a. (consistent with Solvency II Risk Margin; IFRS 17
               does not mandate a rate but 6% is standard practice)
    SCR(t)   = projected Solvency Capital Requirement at future time t
               (product-specific; injected by the run mode)
    DF(t)    = discount factor from t=0 to time t at the current discount rate

The SCR projection is product-specific and supplied by the run mode. This
class only performs the discounting and summation.
"""
from __future__ import annotations


_DEFAULT_COC_RATE = 0.06


class CostOfCapitalRA:
    """
    Risk Adjustment calculator using the Cost of Capital method.

    Parameters
    ----------
    coc_rate : float
        Cost of capital rate as a decimal. Default: 0.06 (6% p.a.).
        Must be > 0.
    """

    def __init__(self, coc_rate: float = _DEFAULT_COC_RATE) -> None:
        if coc_rate <= 0.0:
            raise ValueError(f"coc_rate must be > 0, got {coc_rate}")
        self._coc_rate = coc_rate

    @property
    def coc_rate(self) -> float:
        return self._coc_rate

    def compute(
        self,
        future_scr:       list[float],
        discount_factors: list[float],
    ) -> float:
        """
        Compute the Risk Adjustment from a sequence of future SCR values.

        RA = CoC_rate × Σ [ SCR(t) × DF(t) ]

        Parameters
        ----------
        future_scr : list[float]
            SCR at each future projection period. Must be non-empty.
            All values must be >= 0.
        discount_factors : list[float]
            Discount factor from t=0 to each corresponding SCR period.
            Must be the same length as future_scr.
            All values must be in (0, 1].

        Returns
        -------
        float
            Risk Adjustment. Always >= 0.

        Raises
        ------
        ValueError
            If inputs are empty, different lengths, contain negative SCR
            values, or discount factors outside (0, 1].
        """
        if not future_scr:
            raise ValueError("future_scr must not be empty.")
        if len(future_scr) != len(discount_factors):
            raise ValueError(
                f"future_scr length ({len(future_scr)}) must equal "
                f"discount_factors length ({len(discount_factors)})."
            )
        for i, scr in enumerate(future_scr):
            if scr < 0.0:
                raise ValueError(
                    f"future_scr[{i}] must be >= 0, got {scr}"
                )
        for i, df in enumerate(discount_factors):
            if not (0.0 < df <= 1.0):
                raise ValueError(
                    f"discount_factors[{i}] must be in (0, 1], got {df}"
                )

        pv_scr = sum(scr * df for scr, df in zip(future_scr, discount_factors))
        return self._coc_rate * pv_scr
