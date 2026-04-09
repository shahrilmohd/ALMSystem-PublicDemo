"""
Fund-level configuration for the ALM model.

Each fund has its own Strategic Asset Allocation (SAA), crediting groups,
and rebalancing tolerance. This config is loaded from a separate YAML file
(referenced by InputSourcesConfig.fund_config_path in RunConfig) and injected
into Fund at construction.

Usage:
    fund_cfg = FundConfig.from_yaml("config_files/fund_config.yaml")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# SAA weights
# ---------------------------------------------------------------------------

class AssetClassWeights(BaseModel):
    """
    Strategic Asset Allocation weights for one fund.

    Each weight is a proportion [0, 1] of total fund assets allocated to that
    asset class. All weights must sum to 1.0 (within floating-point tolerance).

    bonds:       Fixed income: government and corporate bonds.
    equities:    Listed equity (domestic and international).
    derivatives: Interest rate derivatives, options, hedging instruments.
    cash:        Cash and money market instruments.
    """
    bonds:       float = Field(default=0.0, ge=0.0, le=1.0,
                               description="Bond allocation weight [0, 1].")
    equities:    float = Field(default=0.0, ge=0.0, le=1.0,
                               description="Equity allocation weight [0, 1].")
    derivatives: float = Field(default=0.0, ge=0.0, le=1.0,
                               description="Derivatives allocation weight [0, 1].")
    cash:        float = Field(default=0.0, ge=0.0, le=1.0,
                               description="Cash allocation weight [0, 1].")

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> AssetClassWeights:
        total = self.bonds + self.equities + self.derivatives + self.cash
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"SAA weights must sum to 1.0, got {total:.8f}. "
                f"(bonds={self.bonds}, equities={self.equities}, "
                f"derivatives={self.derivatives}, cash={self.cash})"
            )
        return self


# ---------------------------------------------------------------------------
# Crediting groups
# ---------------------------------------------------------------------------

class CreditingGroup(BaseModel):
    """
    A crediting group is a set of products that share the same bonus crediting
    rate for a given period. The bonus strategy assigns one rate per group per
    decision timestep.

    group_id:
        Unique identifier for this group. Used as key in crediting rate tables.

    group_name:
        Human-readable label. Stored in results for reporting.

    product_codes:
        List of product codes whose policies belong to this crediting group.
        Product codes are matched against the product_code column in the
        model point DataFrame.
    """
    group_id:      str       = Field(..., min_length=1,
                                     description="Unique crediting group ID.")
    group_name:    str       = Field(..., min_length=1,
                                     description="Human-readable group label.")
    product_codes: list[str] = Field(..., min_length=1,
                                     description="Product codes assigned to this group.")

    @field_validator("product_codes")
    @classmethod
    def no_duplicate_product_codes(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError("product_codes contains duplicates within a crediting group.")
        return v


# ---------------------------------------------------------------------------
# FundConfig
# ---------------------------------------------------------------------------

class FundConfig(BaseModel):
    """
    Full configuration for one segregated fund.

    fund_id:
        Unique identifier for this fund. Must match the fund_id column in
        model point and asset data files.

    fund_name:
        Human-readable fund label. Stored in results and reports.

    saa_weights:
        Strategic Asset Allocation target weights. The InvestmentStrategy uses
        these to compute buy/sell orders at each decision timestep.

    crediting_groups:
        List of crediting groups defined for this fund. The BonusStrategy
        assigns one crediting rate per group per decision period. At least one
        group must be defined.

    rebalancing_tolerance:
        Percentage band around SAA target weights within which no rebalancing
        trade is executed. Prevents excessive transaction costs from minor
        drift. Default: 5% (0.05). Set to 0.0 to force exact rebalancing.
    """
    fund_id:               str                  = Field(
        ..., min_length=1,
        description="Unique fund identifier. Must match data files."
    )
    fund_name:             str                  = Field(
        ..., min_length=1,
        description="Human-readable fund label."
    )
    saa_weights:           AssetClassWeights    = Field(
        ...,
        description="SAA target weights for all asset classes."
    )
    crediting_groups:      list[CreditingGroup] = Field(
        ..., min_length=1,
        description="Crediting groups for bonus strategy. At least one required."
    )
    rebalancing_tolerance: float                = Field(
        default=0.05, ge=0.0, le=1.0,
        description="Band around SAA weights that suppresses rebalancing trades. Default: 5%."
    )

    @field_validator("crediting_groups")
    @classmethod
    def no_duplicate_group_ids(cls, v: list[CreditingGroup]) -> list[CreditingGroup]:
        ids = [g.group_id for g in v]
        if len(ids) != len(set(ids)):
            raise ValueError("crediting_groups contains duplicate group_id values.")
        return v

    # -----------------------------------------------------------------------
    # Convenience constructors
    # -----------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> FundConfig:
        """
        Load and validate a FundConfig from a YAML file.

        Args:
            path: Path to the fund config YAML file.

        Returns:
            Validated FundConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML content fails Pydantic validation.
        """
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load fund configs from YAML files. "
                "Install with: uv add pyyaml"
            ) from e

        yaml_path = Path(path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"FundConfig YAML file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict) -> FundConfig:
        """Construct and validate a FundConfig from a plain dictionary."""
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise this FundConfig to a YAML file."""
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to save fund configs as YAML. "
                "Install with: uv add pyyaml"
            ) from e

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        with open(out_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
