"""
Tests for BonusStrategy hierarchy (Step 27).

B1  declare_reversionary: returns clipped smoothed returns
B2  declare_reversionary: value within corridor passes through unchanged
B3  compute_terminal_bonus_rate: asset_share == guaranteed → rate == 0
B4  compute_terminal_bonus_rate: asset_share > guaranteed → positive rate
B5  compute_terminal_bonus_rate: zero guaranteed_benefit groups contribute 0
B6  declare_reversionary: shape (1,) and (100,) both work
B7  SmoothedBonusStrategy is a BonusStrategy instance
B8  terminal bonus rate scales with terminal_bonus_fraction
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.strategy.bonus_strategy import BonusStrategy, SmoothedBonusStrategy


@pytest.fixture()
def strategy() -> SmoothedBonusStrategy:
    return SmoothedBonusStrategy(
        smoothing_alpha=0.3,
        min_reversionary=0.0,
        max_reversionary=0.05,
        terminal_bonus_fraction=0.5,
    )


# ---------------------------------------------------------------------------
# B1 — clipping behaviour
# ---------------------------------------------------------------------------

class TestDeclareReversionary:
    def test_below_floor_clipped_to_min(self, strategy):
        rates = strategy.declare_reversionary(np.array([-0.10, -0.01]))
        np.testing.assert_array_equal(rates, [0.0, 0.0])

    def test_above_cap_clipped_to_max(self, strategy):
        rates = strategy.declare_reversionary(np.array([0.10, 0.20]))
        np.testing.assert_array_equal(rates, [0.05, 0.05])

    def test_within_corridor_unchanged(self, strategy):
        smoothed = np.array([0.02, 0.03, 0.04])
        rates = strategy.declare_reversionary(smoothed)
        np.testing.assert_array_almost_equal(rates, smoothed)

    def test_shape_1(self, strategy):
        rates = strategy.declare_reversionary(np.array([0.03]))
        assert rates.shape == (1,)

    def test_shape_100(self, strategy):
        rates = strategy.declare_reversionary(np.full(100, 0.03))
        assert rates.shape == (100,)

    def test_output_dtype_float(self, strategy):
        rates = strategy.declare_reversionary(np.array([0.02]))
        assert rates.dtype.kind == "f"


# ---------------------------------------------------------------------------
# B3–B5 — terminal bonus rate
# ---------------------------------------------------------------------------

class TestComputeTerminalBonusRate:
    def test_asset_share_equals_guarantee_returns_zero(self, strategy):
        # surplus = 0 everywhere → rate = 0
        n_sc, n_grp = 4, 3
        guaranteed = np.array([10_000.0, 20_000.0, 15_000.0])
        asset_shares = np.broadcast_to(guaranteed, (n_sc, n_grp)).copy()
        rates = strategy.compute_terminal_bonus_rate(asset_shares, guaranteed)
        np.testing.assert_array_equal(rates, 0.0)

    def test_positive_surplus_gives_positive_rate(self, strategy):
        guaranteed = np.array([10_000.0])
        asset_shares = np.array([[12_000.0]])  # surplus = 2000
        rates = strategy.compute_terminal_bonus_rate(asset_shares, guaranteed)
        assert rates[0] > 0.0

    def test_zero_guaranteed_contributes_zero(self, strategy):
        # Group 0 has zero guaranteed — should not divide by zero
        guaranteed = np.array([0.0, 10_000.0])
        asset_shares = np.array([[5_000.0, 12_000.0]])
        rates = strategy.compute_terminal_bonus_rate(asset_shares, guaranteed)
        assert np.isfinite(rates[0])
        assert rates[0] >= 0.0

    def test_rate_scales_with_fraction(self):
        s_half = SmoothedBonusStrategy(0.3, 0.0, 0.05, terminal_bonus_fraction=0.5)
        s_full = SmoothedBonusStrategy(0.3, 0.0, 0.05, terminal_bonus_fraction=1.0)
        guaranteed = np.array([10_000.0])
        asset_shares = np.array([[12_000.0]])
        r_half = s_half.compute_terminal_bonus_rate(asset_shares, guaranteed)
        r_full = s_full.compute_terminal_bonus_rate(asset_shares, guaranteed)
        np.testing.assert_almost_equal(r_full[0], 2.0 * r_half[0])

    def test_output_shape_is_n_scenarios(self, strategy):
        n_sc, n_grp = 10, 5
        guaranteed = np.ones(n_grp) * 10_000.0
        asset_shares = np.ones((n_sc, n_grp)) * 11_000.0
        rates = strategy.compute_terminal_bonus_rate(asset_shares, guaranteed)
        assert rates.shape == (n_sc,)

    def test_negative_surplus_clamped_to_zero(self, strategy):
        # asset_share below guaranteed → no negative terminal bonus
        guaranteed = np.array([10_000.0])
        asset_shares = np.array([[8_000.0]])
        rates = strategy.compute_terminal_bonus_rate(asset_shares, guaranteed)
        assert rates[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# B7 — type hierarchy
# ---------------------------------------------------------------------------

class TestTypeHierarchy:
    def test_is_bonus_strategy_instance(self, strategy):
        assert isinstance(strategy, BonusStrategy)

    def test_name_property(self, strategy):
        assert strategy.name == "SmoothedBonusStrategy"
