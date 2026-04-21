"""
tests/unit/ifrs17/test_ifrs17_risk_adjustment.py

Numerical tests for CostOfCapitalRA.

All expected values are computed by hand and commented inline.
"""
import pytest

from engine.ifrs17.risk_adjustment import CostOfCapitalRA


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ra(coc_rate: float = 0.06) -> CostOfCapitalRA:
    return CostOfCapitalRA(coc_rate=coc_rate)


# ---------------------------------------------------------------------------
# Canonical 5-period test case from the plan
#
#   SCR  = [100, 80, 60, 40, 20]
#   rate = 5% p.a.,  CoC = 6%
#
#   DF   = [1/1.05¹, 1/1.05², 1/1.05³, 1/1.05⁴, 1/1.05⁵]
#        = [0.952381, 0.907029, 0.863838, 0.822702, 0.783526]
#
#   PV_SCR = 100×0.952381 + 80×0.907029 + 60×0.863838
#          + 40×0.822702  + 20×0.783526
#          = 95.2381 + 72.5623 + 51.8303 + 32.9081 + 15.6705
#          = 268.2093
#
#   RA = 0.06 × 268.2093 = 16.0926 (rounded)
# ---------------------------------------------------------------------------

class TestCanonicalFivePeriod:
    @pytest.fixture
    def inputs(self):
        scr = [100.0, 80.0, 60.0, 40.0, 20.0]
        dfs = [1 / (1.05 ** t) for t in range(1, 6)]
        return scr, dfs

    def test_ra_value(self, inputs):
        scr, dfs = inputs
        ra = make_ra(coc_rate=0.06)
        result = ra.compute(scr, dfs)
        expected = 0.06 * sum(s * d for s, d in zip(scr, dfs))
        assert result == pytest.approx(expected, rel=1e-10)

    def test_ra_approximately_16_09(self, inputs):
        scr, dfs = inputs
        result = make_ra().compute(scr, dfs)
        assert result == pytest.approx(16.0926, rel=1e-4)

    def test_ra_non_negative(self, inputs):
        scr, dfs = inputs
        result = make_ra().compute(scr, dfs)
        assert result >= 0.0

    def test_higher_coc_scales_linearly(self, inputs):
        scr, dfs = inputs
        ra6  = make_ra(coc_rate=0.06).compute(scr, dfs)
        ra12 = make_ra(coc_rate=0.12).compute(scr, dfs)
        assert ra12 == pytest.approx(2.0 * ra6, rel=1e-10)


# ---------------------------------------------------------------------------
# Single-period contract
#
#   SCR = [200], DF = [1/1.04], CoC = 6%
#   RA  = 0.06 × 200 × (1/1.04) = 0.06 × 192.3077 = 11.5385
# ---------------------------------------------------------------------------

class TestSinglePeriod:
    def test_single_scr_value(self):
        ra = make_ra(coc_rate=0.06)
        result = ra.compute([200.0], [1 / 1.04])
        expected = 0.06 * 200.0 / 1.04
        assert result == pytest.approx(expected, rel=1e-10)


# ---------------------------------------------------------------------------
# Zero SCR — RA should be zero
# ---------------------------------------------------------------------------

class TestZeroScr:
    def test_zero_scr_gives_zero_ra(self):
        ra = make_ra()
        result = ra.compute([0.0, 0.0, 0.0], [0.9, 0.8, 0.7])
        assert result == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# coc_rate property
# ---------------------------------------------------------------------------

class TestCocRateProperty:
    def test_default_coc_rate_is_six_percent(self):
        ra = CostOfCapitalRA()
        assert ra.coc_rate == 0.06

    def test_custom_coc_rate_stored(self):
        ra = CostOfCapitalRA(coc_rate=0.08)
        assert ra.coc_rate == 0.08


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_zero_coc_rate_raises(self):
        with pytest.raises(ValueError, match="coc_rate"):
            CostOfCapitalRA(coc_rate=0.0)

    def test_negative_coc_rate_raises(self):
        with pytest.raises(ValueError, match="coc_rate"):
            CostOfCapitalRA(coc_rate=-0.01)

    def test_empty_scr_raises(self):
        ra = make_ra()
        with pytest.raises(ValueError, match="empty"):
            ra.compute([], [])

    def test_mismatched_lengths_raise(self):
        ra = make_ra()
        with pytest.raises(ValueError, match="length"):
            ra.compute([100.0, 80.0], [0.95])

    def test_negative_scr_raises(self):
        ra = make_ra()
        with pytest.raises(ValueError, match="future_scr"):
            ra.compute([-10.0], [0.95])

    def test_zero_discount_factor_raises(self):
        ra = make_ra()
        with pytest.raises(ValueError, match="discount_factors"):
            ra.compute([100.0], [0.0])

    def test_discount_factor_above_one_raises(self):
        ra = make_ra()
        with pytest.raises(ValueError, match="discount_factors"):
            ra.compute([100.0], [1.01])
