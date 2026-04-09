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
from datetime import datetime, timezone

import pandas as pd

from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig, RunType
from engine.liability.conventional import ConventionalAssumptions
from engine.run_modes.base_run import RunStatus
from storage.db import create_all_tables, get_engine, get_session_factory
from storage.result_repository import ResultRepository
from storage.run_repository import RunRepository
from worker.progress import report

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
            # 4. Load model points
            # --------------------------------------------------------------
            report(0.10, "Loading model points")
            from data.loaders.liability_data_loader import LiabilityDataLoader
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
                store = _run_liability_only(
                    run_id, run_config, fund_config, model_points
                )

            elif run_type == RunType.DETERMINISTIC:
                store = _run_deterministic(
                    run_id, run_config, fund_config, model_points
                )

            elif run_type == RunType.STOCHASTIC:
                record.n_scenarios = run_config.stochastic.num_scenarios
                session.commit()
                store = _run_stochastic(
                    run_id, run_config, fund_config, model_points
                )

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
            # Mark FAILED, store first line of error, then re-raise.
            error_msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
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

    assumptions  = _build_assumptions(run_config)
    asset_loader = AssetDataLoader(run_config.input_sources.asset_data_path)
    asset_loader.load()
    asset_loader.validate()
    asset_model  = asset_loader.to_asset_model()
    inv_strategy = InvestmentStrategy(
        saa_weights=fund_config.saa_weights,
        rebalancing_tolerance=fund_config.rebalancing_tolerance,
    )
    scenario_store = ScenarioLoader.from_csv(
        run_config.input_sources.scenario_file_path,
        n_scenarios=run_config.stochastic.num_scenarios,
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
