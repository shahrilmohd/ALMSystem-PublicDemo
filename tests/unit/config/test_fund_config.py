"""
Unit tests for FundConfig, AssetClassWeights, and CreditingGroup.

Rules under test
----------------
AssetClassWeights:
  1. All weights default to 0.0.
  2. Weights must sum to 1.0 (within 1e-6 tolerance).
  3. Each individual weight is constrained to [0, 1].

CreditingGroup:
  4. product_codes must not contain duplicates within a single group.
  5. product_codes must have at least one entry.

FundConfig:
  6. Default rebalancing_tolerance is 0.05.
  7. crediting_groups must have at least one entry.
  8. crediting_groups must not contain duplicate group_id values.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from engine.config.fund_config import AssetClassWeights, CreditingGroup, FundConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_crediting_group(group_id: str = "GRP_A", product_codes: list[str] | None = None) -> dict:
    return {
        "group_id": group_id,
        "group_name": f"Group {group_id}",
        "product_codes": product_codes or ["PROD_01"],
    }


def make_fund_config(crediting_groups: list[dict] | None = None, **overrides) -> dict:
    base = {
        "fund_id": "FUND_A",
        "fund_name": "Fund A",
        "saa_weights": {"bonds": 0.6, "equities": 0.3, "derivatives": 0.0, "cash": 0.1},
        "crediting_groups": [make_crediting_group()] if crediting_groups is None else crediting_groups,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# AssetClassWeights
# ---------------------------------------------------------------------------

class TestAssetClassWeights:
    def test_valid_weights_accepted(self):
        w = AssetClassWeights(bonds=0.6, equities=0.3, derivatives=0.0, cash=0.1)
        assert w.bonds == 0.6

    def test_all_defaults_are_zero(self):
        # Default weights all zero — does NOT sum to 1, so direct construction
        # of an all-zero AssetClassWeights should fail the sum validator.
        with pytest.raises(ValidationError):
            AssetClassWeights()

    def test_single_asset_class_all_in_one(self):
        w = AssetClassWeights(bonds=1.0)
        assert w.bonds == 1.0
        assert w.cash == 0.0

    def test_weights_not_summing_to_one_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            AssetClassWeights(bonds=0.5, equities=0.3, cash=0.1)  # sums to 0.9
        assert "sum" in str(exc_info.value).lower()

    def test_weights_slightly_over_one_rejected(self):
        with pytest.raises(ValidationError):
            AssetClassWeights(bonds=0.6, equities=0.3, cash=0.11)  # sums to 1.01

    def test_negative_weight_rejected(self):
        with pytest.raises(ValidationError):
            AssetClassWeights(bonds=-0.1, equities=0.6, cash=0.5)

    def test_weight_over_one_rejected(self):
        with pytest.raises(ValidationError):
            AssetClassWeights(bonds=1.1, equities=0.0, cash=-0.1)

    def test_floating_point_tolerance_accepted(self):
        # Sum is 1.0 + 5e-7 — within the 1e-6 tolerance, should be accepted.
        w = AssetClassWeights(
            bonds=0.333334,
            equities=0.333333,
            cash=0.333333,
        )
        assert abs(w.bonds + w.equities + w.cash - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# CreditingGroup
# ---------------------------------------------------------------------------

class TestCreditingGroup:
    def test_valid_group_accepted(self):
        g = CreditingGroup(group_id="GRP_A", group_name="Group A", product_codes=["P1", "P2"])
        assert g.group_id == "GRP_A"
        assert len(g.product_codes) == 2

    def test_single_product_code_accepted(self):
        g = CreditingGroup(group_id="GRP_A", group_name="Group A", product_codes=["P1"])
        assert g.product_codes == ["P1"]

    def test_duplicate_product_codes_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            CreditingGroup(
                group_id="GRP_A",
                group_name="Group A",
                product_codes=["P1", "P1"],
            )
        assert "duplicate" in str(exc_info.value).lower()

    def test_empty_product_codes_rejected(self):
        with pytest.raises(ValidationError):
            CreditingGroup(group_id="GRP_A", group_name="Group A", product_codes=[])


# ---------------------------------------------------------------------------
# FundConfig
# ---------------------------------------------------------------------------

class TestFundConfig:
    def test_valid_fund_config_accepted(self):
        cfg = FundConfig.from_dict(make_fund_config())
        assert cfg.fund_id == "FUND_A"
        assert cfg.fund_name == "Fund A"
        assert len(cfg.crediting_groups) == 1

    def test_default_rebalancing_tolerance(self):
        cfg = FundConfig.from_dict(make_fund_config())
        assert cfg.rebalancing_tolerance == 0.05

    def test_custom_rebalancing_tolerance(self):
        cfg = FundConfig.from_dict(make_fund_config(rebalancing_tolerance=0.0))
        assert cfg.rebalancing_tolerance == 0.0

    def test_multiple_crediting_groups_accepted(self):
        groups = [
            make_crediting_group("GRP_A", ["PROD_01", "PROD_02"]),
            make_crediting_group("GRP_B", ["PROD_03"]),
        ]
        cfg = FundConfig.from_dict(make_fund_config(crediting_groups=groups))
        assert len(cfg.crediting_groups) == 2

    def test_empty_crediting_groups_rejected(self):
        with pytest.raises(ValidationError):
            FundConfig.from_dict(make_fund_config(crediting_groups=[]))

    def test_duplicate_group_ids_rejected(self):
        groups = [
            make_crediting_group("GRP_A", ["PROD_01"]),
            make_crediting_group("GRP_A", ["PROD_02"]),  # same group_id
        ]
        with pytest.raises(ValidationError) as exc_info:
            FundConfig.from_dict(make_fund_config(crediting_groups=groups))
        assert "duplicate" in str(exc_info.value).lower()

    def test_saa_weights_validated(self):
        data = make_fund_config()
        data["saa_weights"] = {"bonds": 0.5, "equities": 0.3, "cash": 0.1}  # sums to 0.9
        with pytest.raises(ValidationError):
            FundConfig.from_dict(data)
