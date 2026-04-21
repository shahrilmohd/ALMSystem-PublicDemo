"""
Quick standalone BPA validation run.

Runs the BPA engine against the newly generated group model points
and prints the cohort BEL summary. Used to verify scale is in the
expected £40-50M pre-MA range with the £55M face-value asset portfolio.

Usage:
    uv run python tests/sample_data/q42025/run_bpa_validation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from data.loaders.asset_data_loader import AssetDataLoader
from data.loaders.bpa_data_loader import BPADataLoader
from engine.config.run_config import RunConfig, RunType
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.bpa.assumptions import BPAAssumptions
from engine.liability.bpa.mortality import TABLE_LENGTH
from engine.liability.bpa.registry import BPADealRegistry
from engine.matching_adjustment.fundamental_spread import FundamentalSpreadTable
from engine.run_modes.bpa_run import BPARun
from engine.run_modes.base_run import RunStatus

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAMPLE = Path(__file__).parent
TABLES = SAMPLE / "tables"
MP_DIR = SAMPLE / "mp"

GROUP_MP_PATH   = MP_DIR / "group_mp" / "bpa_model_points_group_mp.csv"
ASSET_MP_PATH   = MP_DIR / "assets"  / "bpa_asset_model_points.csv"
ASSET_CF_PATH   = MP_DIR / "assets"  / "bpa_asset_cashflows.csv"
DEALS_PATH      = TABLES / "liability" / "bpa_deals.csv"
MORTALITY_DIR   = TABLES / "liability"
FS_PATH         = TABLES / "assets"   / "pra_fundamental_spread.csv"
RATE_CURVE_PATH = TABLES / "liability" / "rate_curve.csv"
FUND_CFG_PATH   = TABLES / "assets"   / "bpa_fund_config.yaml"


def build_run_config() -> RunConfig:
    """Build a minimal RunConfig for BPA validation."""
    return RunConfig.model_validate({
        "run_id":   "validation_run_001",
        "run_name": "BPA Group MP Validation",
        "run_type": "bpa",
        "liability": {
            "active_models": ["conventional"],
        },
        "projection": {
            "valuation_date":         "2025-12-31",
            "projection_term_years":  40,
        },
        "bpa": {
            "monthly_years": 10,
        },
        # Use a longer horizon so the full tail of the liability is captured
        "input_sources": {
            "asset_data_path": str(ASSET_MP_PATH),
            "assumption_tables": {
                "tables_root_dir": str(TABLES / "liability"),
            },
            "bpa_inputs": {
                "bpa_model_points_path":   str(GROUP_MP_PATH),
                "bpa_deals_path":          str(DEALS_PATH),
                "mortality_dir":           str(MORTALITY_DIR),
                "fundamental_spread_path": str(FS_PATH),
                "asset_cashflows_path":    str(ASSET_CF_PATH),
            },
        },
    })


def _split_bpa_mps(df: pd.DataFrame, population_type: str, registry: BPADealRegistry) -> dict:
    subset = df[df["population_type"] == population_type].copy()
    result: dict[str, pd.DataFrame] = {}
    for deal_id in registry.all_deal_ids():
        deal_rows = subset[subset["deal_id"] == deal_id].copy()
        if deal_rows.empty:
            continue
        # Drop all-NaN columns so each population type gets only its columns.
        deal_rows = deal_rows.dropna(axis=1, how="all")
        result[deal_id] = deal_rows.reset_index(drop=True)
    return result


def main() -> None:
    print("=" * 60)
    print("BPA Group MP Validation Run")
    print("=" * 60)

    run_config = build_run_config()

    # 1. Deal registry
    registry = BPADealRegistry.from_csv(DEALS_PATH)

    # 2. Mortality basis
    mortality = BPADataLoader.load_mortality_basis(MORTALITY_DIR)

    # 3. BPA model points — split by population_type
    mp_df = pd.read_csv(GROUP_MP_PATH)
    in_payment_mps = _split_bpa_mps(mp_df, "in_payment",  registry)
    deferred_mps   = _split_bpa_mps(mp_df, "deferred",    registry)
    dependant_mps  = _split_bpa_mps(mp_df, "dependant",   registry)
    enhanced_mps   = _split_bpa_mps(mp_df, "enhanced",    registry)

    print(f"  in_payment cohorts  : {len(in_payment_mps)}")
    print(f"  deferred cohorts    : {len(deferred_mps)}")
    print(f"  dependant cohorts   : {len(dependant_mps)}")
    print(f"  enhanced cohorts    : {len(enhanced_mps)}")

    # 4. Asset model
    asset_loader = AssetDataLoader(ASSET_MP_PATH)
    asset_loader.load()
    asset_loader.validate()
    asset_model = asset_loader.to_asset_model()
    print(f"  Assets loaded       : ok")

    # 5. Asset cashflows
    asset_cfs = pd.read_csv(ASSET_CF_PATH)
    asset_cfs["t"]  = asset_cfs["t"].astype(int)
    asset_cfs["cf"] = asset_cfs["cf"].astype(float)

    # 6. Asset metadata for MA eligibility
    raw_assets = pd.read_csv(ASSET_MP_PATH)
    _MA_COLS = [
        "asset_id", "cashflow_type", "currency",
        "has_credit_risk_transfer", "has_qualifying_currency_swap",
        "rating", "seniority", "tenor_years", "spread_bps",
    ]
    assets_df = raw_assets[_MA_COLS].copy()
    for bool_col in ("has_credit_risk_transfer", "has_qualifying_currency_swap"):
        assets_df[bool_col] = assets_df[bool_col].map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes")
        )

    # 7. Fundamental spread table
    fs_table = FundamentalSpreadTable.from_csv(FS_PATH)

    # 8. Rate curve + assumptions
    curve_df   = pd.read_csv(RATE_CURVE_PATH)
    spot_rates = dict(zip(curve_df["maturity_yr"].astype(float),
                          curve_df["spot_rate"].astype(float)))
    rate_curve = RiskFreeRateCurve(spot_rates=spot_rates)

    ill_health = np.zeros(TABLE_LENGTH, dtype=float)
    assumptions = BPAAssumptions(
        mortality        = mortality,
        valuation_year   = run_config.projection.valuation_date.year,
        discount_curve   = rate_curve,
        inflation_rate   = 0.025,
        rpi_rate         = 0.030,
        tv_rate          = 0.02,
        ill_health_rates = ill_health,
        expense_pa       = 150.0,
    )

    # 9. BPARun
    print("\nRunning BPA projection (12y, 10y monthly) ...")
    from engine.config.fund_config import FundConfig
    fund_config = FundConfig.from_yaml(FUND_CFG_PATH)

    run = BPARun(
        config            = run_config,
        fund_config       = fund_config,
        in_payment_mps    = in_payment_mps,
        deferred_mps      = deferred_mps,
        dependant_mps     = dependant_mps,
        enhanced_mps      = enhanced_mps,
        assumptions       = assumptions,
        asset_model       = asset_model,
        assets_df         = assets_df,
        asset_cfs         = asset_cfs,
        fs_table          = fs_table,
        deal_registry     = registry,
        projection_years  = run_config.projection.projection_term_years,  # 40 years
        monthly_years     = 10,
    )
    run.run()

    if run.result.status == RunStatus.FAILED:
        print(f"FAILED: {run.result.error}")
        sys.exit(1)

    # 10. Extract cohort summary at t=0
    store = run._store
    all_results = store.as_dataframe()

    # Filter to t=0 and scenario 0, select per-cohort BEL columns
    t0 = all_results[
        (all_results["timestep"] == 0) & (all_results["scenario_id"] == 0)
    ][["cohort_id", "bel_pre_ma", "bel_post_ma", "in_force_start"]].copy()
    t0.columns = ["cohort_id", "bel_pre_ma_t0", "bel_post_ma_t0", "in_force_lives_t0"]

    print("\n=== COHORT BEL SUMMARY (t=0) ===")
    print(t0.to_string(index=False))

    total_pre  = t0["bel_pre_ma_t0"].sum()
    total_post = t0["bel_post_ma_t0"].sum()
    ma_benefit = total_pre - total_post
    ma_pct     = ma_benefit / total_pre * 100 if total_pre > 0 else 0.0

    print(f"\n  Total pre-MA  BEL : GBP {total_pre:>12,.0f}")
    print(f"  Total post-MA BEL : GBP {total_post:>12,.0f}")
    print(f"  MA benefit        : GBP {ma_benefit:>12,.0f}  ({ma_pct:.2f}%)")

    # Asset face value for comparison
    face_total = raw_assets["face_value"].sum()
    print(f"  Asset face value  : GBP {face_total:>12,.0f}")
    surplus = face_total - total_post
    print(f"  Notional surplus  : GBP {surplus:>12,.0f}  (face - post-MA BEL)")

    print("\nDone.")


if __name__ == "__main__":
    main()
