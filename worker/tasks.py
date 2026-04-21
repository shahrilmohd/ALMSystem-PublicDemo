"""
RQ task functions for ALM projection jobs.

This is the ONLY file that imports from engine/.
The API layer never imports engine/ directly — it only enqueues a run_id
and this module does the rest.

Why a single function?
----------------------
All three run types (LIABILITY_ONLY, DETERMINISTIC, STOCHASTIC) are handled
by one function — run_alm_job().  The function reads the run_type from the
stored RunConfig and branches accordingly.  A separate function per run type
would require the API to know which function to enqueue, coupling the API to
execution details it should not care about.

Execution sequence inside run_alm_job()
----------------------------------------
1.  Mark the RunRecord as RUNNING.
2.  Deserialise the RunConfig from the JSON stored in the RunRecord.
3.  Load all required data (model points, assets, scenarios).
4.  Construct the appropriate run mode object.
5.  Call run_mode.run() — this executes the full projection.
6.  Persist all ResultStore rows to the DB via ResultRepository.
7.  Mark the RunRecord as COMPLETED.

On any exception at any step:
    - Mark the RunRecord as FAILED with the error message.
    - Re-raise so RQ marks the job as failed in Redis too.

Progress reporting
------------------
A progress_callback is wired from worker.progress.report into the run mode
constructor.  As the engine advances through scenarios / timesteps it calls
report(fraction, message), which writes into the RQ job's metadata in Redis.
The frontend polls this via GET /runs/{run_id} (Phase 2 desktop app reads
progress from the job meta).
"""
from __future__ import annotations

import json
import logging
import os
import traceback as _traceback
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig, RunType
from engine.liability.conventional import ConventionalAssumptions
from engine.run_modes.base_run import RunStatus
from storage.db import create_all_tables, get_engine, get_session_factory
from storage.result_repository import ResultRepository
from storage.run_repository import RunRepository
from worker.progress import report

if TYPE_CHECKING:
    from engine.liability.bpa.registry import BPADealRegistry

logger = logging.getLogger(__name__)

_DB_URL = os.getenv("ALM_DB_URL", "sqlite:///alm.db")


def run_alm_job(run_id: str) -> None:
    """
    Execute a queued ALM projection run end-to-end.

    Called by the RQ worker process.  The run_id must already exist as a
    PENDING RunRecord in the database — POST /runs creates it before
    enqueueing this job.

    Parameters
    ----------
    run_id : str
        UUID identifying the RunRecord whose config_json will be executed.

    Raises
    ------
    KeyError
        If no RunRecord exists for run_id.
    Exception
        Any error raised by the engine during projection.  The RunRecord is
        marked FAILED before the exception is re-raised.
    """
    engine  = get_engine(_DB_URL)
    create_all_tables(engine)
    Session = get_session_factory(engine)

    with Session() as session:
        run_repo    = RunRepository(session)
        result_repo = ResultRepository(session)

        # ------------------------------------------------------------------
        # 1. Mark RUNNING
        # ------------------------------------------------------------------
        report(0.0, "Starting")
        run_repo.update_status(
            run_id,
            "RUNNING",
            started_at=datetime.now(timezone.utc),
        )
        session.commit()

        try:
            # --------------------------------------------------------------
            # 2. Deserialise RunConfig from stored JSON
            # --------------------------------------------------------------
            record     = run_repo.get(run_id)
            # Override run_id with the DB record's run_id — the GUI generates
            # its own UUID when building the config, but the API assigns a new
            # one when saving the RunRecord.  The DB run_id is authoritative.
            config_dict = json.loads(record.config_json)
            config_dict["run_id"] = run_id
            run_config = RunConfig.model_validate(config_dict)

            # Update n_timesteps now that we have the parsed config
            n_timesteps = run_config.projection.projection_term_years * 12
            run_repo.update_status(run_id, "RUNNING", )
            record.n_timesteps = n_timesteps
            session.commit()

            report(0.05, "Config loaded")

            # --------------------------------------------------------------
            # 3. Load FundConfig
            #    Required for DETERMINISTIC and STOCHASTIC; None for LIABILITY_ONLY.
            # --------------------------------------------------------------
            fund_config = (
                FundConfig.from_yaml(run_config.input_sources.fund_config_path)
                if run_config.input_sources.fund_config_path is not None
                else None
            )

            # --------------------------------------------------------------
            # 4. Load model points (conventional runs only)
            # BPA runs omit model_points from the config and load their own
            # combined MP file (discriminated by population_type) inside _run_bpa().
            # --------------------------------------------------------------
            model_points = None
            if run_config.input_sources.model_points is not None:
                report(0.10, "Loading model points")
                from data.loaders.liability_data_loader import LiabilityDataLoader
                assert run_config.input_sources.model_points.file is not None
                mp_file    = run_config.input_sources.model_points.file.file_path
                column_map = run_config.input_sources.model_points.column_map
                loader     = LiabilityDataLoader(mp_file, column_map=column_map)
                loader.load()
                loader.validate()
                model_points = loader.to_dataframe()

            # --------------------------------------------------------------
            # 5. Construct run mode and execute
            # --------------------------------------------------------------
            run_type = run_config.run_type

            if run_type == RunType.LIABILITY_ONLY:
                assert fund_config is not None
                store = _run_liability_only(
                    run_id, run_config, fund_config, model_points
                )

            elif run_type == RunType.DETERMINISTIC:
                assert fund_config is not None
                store = _run_deterministic(
                    run_id, run_config, fund_config, model_points
                )

            elif run_type == RunType.STOCHASTIC:
                assert fund_config is not None
                assert run_config.stochastic is not None
                record.n_scenarios = run_config.stochastic.num_scenarios
                session.commit()
                store = _run_stochastic(
                    run_id, run_config, fund_config, model_points
                )

            elif run_type == RunType.BPA:
                assert fund_config is not None
                store = _run_bpa(run_id, run_config, fund_config)

            else:
                raise ValueError(f"Unsupported run_type: {run_type!r}")

            # --------------------------------------------------------------
            # 6. Persist results
            # --------------------------------------------------------------
            report(0.95, "Saving results")
            result_repo.save_all(run_id, store)
            session.commit()

            # --------------------------------------------------------------
            # 7. Mark COMPLETED
            # --------------------------------------------------------------
            completed_at = datetime.now(timezone.utc)
            started_at   = run_repo.get(run_id).started_at
            # SQLite returns naive datetimes; normalise to UTC before subtracting.
            if started_at is not None and started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            duration     = (
                (completed_at - started_at).total_seconds()
                if started_at else None
            )
            run_repo.update_status(
                run_id,
                "COMPLETED",
                completed_at=completed_at,
                duration_seconds=duration,
            )
            session.commit()
            report(1.0, "Complete")

        except Exception as exc:
            # Mark FAILED, store full traceback for the debug agent, then re-raise.
            error_msg = _traceback.format_exc() or (type(exc).__name__)
            try:
                run_repo.update_status(
                    run_id,
                    "FAILED",
                    completed_at=datetime.now(timezone.utc),
                    error_message=error_msg,
                )
                session.commit()
            except Exception:
                pass  # Don't mask the original exception
            logger.exception("run_alm_job failed for run_id=%r", run_id)
            raise


# ---------------------------------------------------------------------------
# Private helpers — one per run type
# Each returns a completed ResultStore.
# ---------------------------------------------------------------------------

def _build_assumptions(run_config: RunConfig) -> ConventionalAssumptions:
    """Load assumption tables from the configured folder and build ConventionalAssumptions."""
    import pandas as pd
    from engine.curves.rate_curve import RiskFreeRateCurve
    from engine.liability.conventional import ConventionalAssumptions

    root = run_config.input_sources.assumption_tables.tables_root_dir

    # Mortality rates: attained_age -> q_x
    mort_df = pd.read_csv(root / "mortality_rates.csv")
    mortality_rates = dict(zip(mort_df["attained_age"].astype(int), mort_df["q_x"].astype(float)))

    # Lapse rates: duration_yr -> w_x
    lapse_df = pd.read_csv(root / "lapse_rates.csv")
    lapse_rates = dict(zip(lapse_df["duration_yr"].astype(int), lapse_df["w_x"].astype(float)))

    # Surrender value factors: duration_yr -> sv_factor
    sv_df = pd.read_csv(root / "surrender_value_factors.csv")
    surrender_value_factors = dict(zip(sv_df["duration_yr"].astype(int), sv_df["sv_factor"].astype(float)))

    # Scalar assumptions (expense rates, bonus rate, defaults)
    scalar_df = pd.read_csv(root / "scalar_assumptions.csv", index_col="assumption")
    def _s(name: str, default: float = 0.0) -> float:
        return float(scalar_df.loc[name, "value"]) if name in scalar_df.index else default

    # Rate curve: maturity_yr -> spot_rate
    curve_df = pd.read_csv(root / "rate_curve.csv")
    spot_rates = dict(zip(curve_df["maturity_yr"].astype(float), curve_df["spot_rate"].astype(float)))
    rate_curve = RiskFreeRateCurve(spot_rates=spot_rates)

    return ConventionalAssumptions(
        mortality_rates=mortality_rates,
        lapse_rates=lapse_rates,
        expense_pct_premium=_s("expense_pct_premium"),
        expense_per_policy=_s("expense_per_policy"),
        surrender_value_factors=surrender_value_factors,
        rate_curve=rate_curve,
        bonus_rate_yr=_s("bonus_rate_yr"),
        default_mortality_rate=_s("default_mortality_rate"),
        default_lapse_rate=_s("default_lapse_rate"),
        default_surrender_value_factor=_s("default_sv_factor"),
    )


def _run_liability_only(
    run_id:       str,
    run_config:   RunConfig,
    fund_config:  FundConfig,
    model_points: pd.DataFrame,
):
    from engine.run_modes.liability_only_run import LiabilityOnlyRun

    def _progress(fraction: float, message: str) -> None:
        report(0.15 + fraction * 0.75, message)

    assumptions = _build_assumptions(run_config)
    run = LiabilityOnlyRun(
        config=run_config,
        fund_config=fund_config,
        model_points=model_points,
        assumptions=assumptions,
        progress_callback=_progress,
    )
    run.run()
    if run.result.status == RunStatus.FAILED:
        raise RuntimeError(run.result.error or "LiabilityOnlyRun failed")
    return run._store


def _run_deterministic(
    run_id:       str,
    run_config:   RunConfig,
    fund_config:  FundConfig,
    model_points: pd.DataFrame,
):
    from data.loaders.asset_data_loader import AssetDataLoader
    from engine.run_modes.deterministic_run import DeterministicRun
    from engine.strategy.investment_strategy import InvestmentStrategy

    def _progress(fraction: float, message: str) -> None:
        report(0.15 + fraction * 0.75, message)

    assumptions  = _build_assumptions(run_config)
    assert run_config.input_sources.asset_data_path is not None
    asset_loader = AssetDataLoader(run_config.input_sources.asset_data_path)
    asset_loader.load()
    asset_loader.validate()
    asset_model  = asset_loader.to_asset_model()
    inv_strategy = InvestmentStrategy(
        saa_weights=fund_config.saa_weights,
        rebalancing_tolerance=fund_config.rebalancing_tolerance,
    )

    run = DeterministicRun(
        config=run_config,
        fund_config=fund_config,
        model_points=model_points,
        assumptions=assumptions,
        asset_model=asset_model,
        investment_strategy=inv_strategy,
        progress_callback=_progress,
    )
    run.run()
    if run.result.status == RunStatus.FAILED:
        raise RuntimeError(run.result.error or "DeterministicRun failed")
    return run._store


def _run_stochastic(
    run_id:       str,
    run_config:   RunConfig,
    fund_config:  FundConfig,
    model_points: pd.DataFrame,
):
    from data.loaders.asset_data_loader import AssetDataLoader
    from engine.run_modes.stochastic_run import StochasticRun
    from engine.scenarios.scenario_engine import ScenarioLoader
    from engine.strategy.investment_strategy import InvestmentStrategy

    def _progress(fraction: float, message: str) -> None:
        report(0.15 + fraction * 0.75, message)

    assert run_config.input_sources.asset_data_path is not None
    assert run_config.stochastic is not None
    assumptions  = _build_assumptions(run_config)
    asset_loader = AssetDataLoader(run_config.input_sources.asset_data_path)
    asset_loader.load()
    asset_loader.validate()
    asset_model  = asset_loader.to_asset_model()
    inv_strategy = InvestmentStrategy(
        saa_weights=fund_config.saa_weights,
        rebalancing_tolerance=fund_config.rebalancing_tolerance,
    )
    assert run_config.input_sources.scenario_file_path is not None
    n = run_config.stochastic.num_scenarios
    scenario_store = ScenarioLoader.from_csv(
        run_config.input_sources.scenario_file_path,
        scenario_ids=list(range(n)) if n is not None else None,
    )

    run = StochasticRun(
        config=run_config,
        fund_config=fund_config,
        model_points=model_points,
        assumptions=assumptions,
        asset_model=asset_model,
        investment_strategy=inv_strategy,
        scenario_store=scenario_store,
        progress_callback=_progress,
    )
    run.run()
    if run.result.status == RunStatus.FAILED:
        raise RuntimeError(run.result.error or "StochasticRun failed")
    return run._store


def _run_bpa(
    run_id:      str,
    run_config:  RunConfig,
    fund_config: FundConfig,
):
    """
    Execute a BPA MA-portfolio run end-to-end.

    Loading sequence
    ----------------
    1. Deal registry        — BPADealRegistry.from_csv(bpa_deals_path)
    2. Mortality basis      — BPADataLoader.load_mortality_basis(mortality_dir)
    3. BPA model points     — load combined CSV, split by population_type, validate
    4. Asset portfolio      — AssetDataLoader → AssetModel
    5. Asset cashflows      — pd.read_csv(asset_cashflows_path)
    6. Assets metadata      — extract MA columns from bpa_asset_model_points CSV
    7. Fundamental spread   — FundamentalSpreadTable.from_csv(fundamental_spread_path)
    8. BPA assumptions      — BPAAssumptions.default(mortality) with CPI/RPI from config
    9. BPARun               — construct and run
    """
    from data.loaders.asset_data_loader import AssetDataLoader
    from data.loaders.bpa_data_loader import BPADataLoader
    from engine.curves.rate_curve import RiskFreeRateCurve
    from engine.liability.bpa.assumptions import BPAAssumptions
    from engine.liability.bpa.registry import BPADealRegistry
    from engine.matching_adjustment.fundamental_spread import FundamentalSpreadTable
    from engine.run_modes.bpa_run import BPARun

    bpa = run_config.input_sources.bpa_inputs
    assert bpa is not None, "BPA run requires bpa_inputs in run config"

    def _progress(fraction: float, message: str) -> None:
        report(0.10 + fraction * 0.85, message)

    # 1. Deal registry
    _progress(0.00, "Loading deal registry")
    registry = BPADealRegistry.from_csv(bpa.bpa_deals_path)

    # 2. Mortality basis
    _progress(0.05, "Loading mortality basis")
    mortality = BPADataLoader.load_mortality_basis(bpa.mortality_dir)

    # 3. BPA model points — split combined CSV by population_type
    _progress(0.10, "Loading BPA model points")
    mp_df = pd.read_csv(bpa.bpa_model_points_path)
    in_payment_mps = _split_bpa_mps(mp_df, "in_payment",  registry)
    deferred_mps   = _split_bpa_mps(mp_df, "deferred",    registry)
    dependant_mps  = _split_bpa_mps(mp_df, "dependant",   registry)
    enhanced_mps   = _split_bpa_mps(mp_df, "enhanced",    registry)

    # 4. Asset portfolio
    _progress(0.20, "Loading asset portfolio")
    assert run_config.input_sources.asset_data_path is not None
    asset_loader = AssetDataLoader(run_config.input_sources.asset_data_path)
    asset_loader.load()
    asset_loader.validate()
    asset_model = asset_loader.to_asset_model()

    # 5. Asset cashflows
    _progress(0.25, "Loading asset cashflows")
    asset_cfs = pd.read_csv(bpa.asset_cashflows_path)
    asset_cfs["t"]  = asset_cfs["t"].astype(int)
    asset_cfs["cf"] = asset_cfs["cf"].astype(float)

    # 6. Assets metadata for MACalculator (MA eligibility columns)
    _progress(0.28, "Preparing asset metadata")
    raw_assets = pd.read_csv(run_config.input_sources.asset_data_path)
    _MA_COLS = [
        "asset_id", "cashflow_type", "currency",
        "has_credit_risk_transfer", "has_qualifying_currency_swap",
        "rating", "seniority", "tenor_years", "spread_bps",
    ]
    missing_ma = set(_MA_COLS) - set(raw_assets.columns)
    if missing_ma:
        raise ValueError(
            f"BPA asset file is missing MA eligibility columns: {sorted(missing_ma)}. "
            "Ensure the file is the BPA-specific asset CSV, not the conventional one."
        )
    assets_df = raw_assets[_MA_COLS].copy()
    # Coerce boolean columns — they are stored as strings in CSV
    for bool_col in ("has_credit_risk_transfer", "has_qualifying_currency_swap"):
        assets_df[bool_col] = assets_df[bool_col].map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes")
        )

    # 7. Fundamental spread table
    _progress(0.30, "Loading fundamental spread table")
    fs_table = FundamentalSpreadTable.from_csv(bpa.fundamental_spread_path)

    # 8. BPA assumptions — use default() with the loaded mortality basis.
    #    The rate curve is loaded from the standard assumption_tables folder
    #    (rate_curve.csv) so the same curve drives both BPA and conventional runs.
    _progress(0.35, "Building BPA assumptions")
    root = run_config.input_sources.assumption_tables.tables_root_dir
    curve_df   = pd.read_csv(root / "rate_curve.csv")
    spot_rates = dict(zip(curve_df["maturity_yr"].astype(float),
                          curve_df["spot_rate"].astype(float)))
    rate_curve = RiskFreeRateCurve(spot_rates=spot_rates)

    import numpy as np
    from engine.liability.bpa.mortality import TABLE_LENGTH
    ill_health = np.zeros(TABLE_LENGTH, dtype=float)
    assumptions = BPAAssumptions(
        mortality      = mortality,
        valuation_year = run_config.projection.valuation_date.year,
        discount_curve = rate_curve,
        inflation_rate = 0.025,   # 2.5% CPI best estimate
        rpi_rate       = 0.030,   # 3.0% RPI best estimate
        tv_rate        = 0.02,    # 2% p.a. TV election rate for deferreds
        ill_health_rates = ill_health,
        expense_pa     = 150.0,
    )

    # 9. Construct and run BPARun
    _progress(0.40, "Starting BPA projection")
    run = BPARun(
        config           = run_config,
        fund_config      = fund_config,
        in_payment_mps   = in_payment_mps,
        deferred_mps     = deferred_mps,
        dependant_mps    = dependant_mps,
        enhanced_mps     = enhanced_mps,
        assumptions      = assumptions,
        asset_model      = asset_model,
        assets_df        = assets_df,
        asset_cfs        = asset_cfs,
        fs_table         = fs_table,
        deal_registry    = registry,
        projection_years = run_config.projection.projection_term_years,
        monthly_years    = min(
            run_config.bpa.monthly_years if run_config.bpa is not None else 10,
            run_config.projection.projection_term_years,
        ),
        progress_callback = _progress,
    )
    run.run()
    if run.result.status == RunStatus.FAILED:
        raise RuntimeError(run.result.error or "BPARun failed")
    return run._store


def _split_bpa_mps(
    df:              pd.DataFrame,
    population_type: str,
    registry:        "BPADealRegistry",
) -> dict:
    """
    Filter the combined BPA model point DataFrame to one population type and
    split by deal_id into the dict format expected by BPARun.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are deal_ids; values are DataFrames with only the columns
        relevant to that population type (NaN-only columns dropped).
    """
    from data.loaders.bpa_data_loader import BPADataLoader
    from data.validators.bpa_validator import BPAValidator

    subset = df[df["population_type"] == population_type].copy()
    if subset.empty:
        return {}

    # Drop columns that are entirely NaN for this population type
    subset = subset.dropna(axis=1, how="all")

    # Type coercion — mirrors BPADataLoader.to_dataframe()
    _FLOAT_COLS = (
        "age", "in_force_count", "weight",   # weight kept for dependant population
        "pension_pa", "lpi_cap", "lpi_floor", "gmp_pa",
        "deferred_pension_pa", "era", "nra", "revaluation_cap",
        "revaluation_floor", "deferment_years", "rating_years",
        "member_age", "dependant_age",
    )
    for col in _FLOAT_COLS:
        if col in subset.columns:
            subset[col] = subset[col].astype(float)
    if "tv_eligible" in subset.columns:
        subset["tv_eligible"] = subset["tv_eligible"].astype(int)

    # Strip whitespace from string columns
    str_cols = subset.select_dtypes(include=["object"]).columns
    for col in str_cols:
        subset[col] = subset[col].str.strip()

    # Validate using the appropriate BPAValidator method
    validator_fn = {
        "in_payment": BPAValidator.validate_in_payment,
        "deferred":   BPAValidator.validate_deferred,
        "dependant":  BPAValidator.validate_dependant,
        "enhanced":   BPAValidator.validate_enhanced,
    }[population_type]
    validator_fn(subset, registry=registry)

    # Split by deal_id
    result: dict = {}
    for deal_id, group in subset.groupby("deal_id"):
        result[deal_id] = group.drop(columns=["population_type"]).reset_index(drop=True)
    return result
