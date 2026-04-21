"""
ALM System — CLI entry point.

Runs projections directly against the engine, bypassing the API and worker
queue.  Useful for local testing, batch scripting, and CI validation.
No Redis, no database, no server required.

Commands
--------
  run       Execute a projection from a YAML config file.
  validate  Validate a YAML config without running the projection.
  schema    Print the RunConfig JSON schema to stdout.
  optimise  Select the optimal BPA asset portfolio for a new deal (Step 25).

Examples
--------
  uv run python main.py validate config_files/bpa_run.yaml
  uv run python main.py run config_files/bpa_run.yaml
  uv run python main.py run config_files/bpa_run.yaml --output-dir /tmp/results
  uv run python main.py schema | python -m json.tool
  uv run python main.py optimise \\
      --candidates candidate_bonds.csv \\
      --liability-cfs liability_annual_cfs.csv \\
      --rfr-curve rfr_curve.csv \\
      --fs-table pra_fundamental_spread.csv \\
      --output portfolio_output.csv

Notes
-----
- Results are written to the output_dir specified in the YAML config.
  Use --output-dir to override it without editing the file.
- The run_id is taken from the YAML config.  Use --run-id to override it
  (useful when running the same config multiple times).
- A copy of the resolved config is saved as run_config.json alongside the
  results for audit purposes.
- The optimise command is a pre-deal pricing tool: it takes pre-computed
  liability cashflows (from a prior LiabilityOnlyRun) and a candidate bond
  universe, and returns the optimal MA portfolio CSV for use as input to a
  BPARun.  See DECISIONS.md §62 for the full pricing workflow.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------

def _make_progress(quiet: bool):
    """Return a (fraction, message) -> None callback that prints to stdout."""
    def _report(fraction: float, message: str) -> None:
        if quiet:
            return
        bar_len = 30
        filled  = int(bar_len * max(0.0, min(1.0, fraction)))
        bar     = "#" * filled + "-" * (bar_len - filled)
        pct     = int(fraction * 100)
        print(f"\r  [{bar}] {pct:3d}%  {message:<40}", end="", flush=True)
        if fraction >= 1.0:
            print()
    return _report


# ---------------------------------------------------------------------------
# Result writing
# ---------------------------------------------------------------------------

def _write_results(store, run_config, output_dir: Path, quiet: bool) -> None:
    """Write ResultStore to disk as CSV or Parquet, plus the resolved config JSON."""
    from engine.config.run_config import ResultFormat

    output_dir.mkdir(parents=True, exist_ok=True)
    df = store.as_dataframe()

    compress = run_config.output.compress_outputs
    if run_config.output.result_format == ResultFormat.PARQUET:
        out_path = output_dir / "results.parquet"
        df.to_parquet(
            out_path,
            index=False,
            compression="gzip" if compress else None,
        )
    else:
        if compress:
            out_path = output_dir / "results.csv.gz"
            df.to_csv(out_path, index=False, compression="gzip")
        else:
            out_path = output_dir / "results.csv"
            df.to_csv(out_path, index=False)

    config_path = output_dir / "run_config.json"
    run_config.to_json(config_path)

    if not quiet:
        print(f"  Output dir : {output_dir}")
        print(f"  Results    : {out_path.name}  ({len(df):,} rows)")
        print(f"  Config     : {config_path.name}")


# ---------------------------------------------------------------------------
# Assumption loader  (mirrors worker/tasks.py _build_assumptions)
# ---------------------------------------------------------------------------

def _build_assumptions(run_config):
    """Load assumption tables and return a ConventionalAssumptions instance."""
    import pandas as pd
    from engine.curves.rate_curve import RiskFreeRateCurve
    from engine.liability.conventional import ConventionalAssumptions

    root = run_config.input_sources.assumption_tables.tables_root_dir

    mort_df          = pd.read_csv(root / "mortality_rates.csv")
    mortality_rates  = dict(zip(mort_df["attained_age"].astype(int), mort_df["q_x"].astype(float)))

    lapse_df         = pd.read_csv(root / "lapse_rates.csv")
    lapse_rates      = dict(zip(lapse_df["duration_yr"].astype(int), lapse_df["w_x"].astype(float)))

    sv_df                  = pd.read_csv(root / "surrender_value_factors.csv")
    surrender_value_factors = dict(zip(sv_df["duration_yr"].astype(int), sv_df["sv_factor"].astype(float)))

    scalar_df = pd.read_csv(root / "scalar_assumptions.csv", index_col="assumption")
    def _s(name: str, default: float = 0.0) -> float:
        return float(scalar_df.loc[name, "value"]) if name in scalar_df.index else default

    curve_df   = pd.read_csv(root / "rate_curve.csv")
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


# ---------------------------------------------------------------------------
# BPA model point splitter  (mirrors worker/tasks.py _split_bpa_mps)
# ---------------------------------------------------------------------------

def _split_bpa_mps(df, population_type: str, registry) -> dict:
    """
    Filter the combined BPA model point DataFrame to one population type
    and split by deal_id.  Returns dict[deal_id, DataFrame].
    """
    from data.loaders.bpa_data_loader import BPADataLoader  # noqa: F401
    from data.validators.bpa_validator import BPAValidator

    subset = df[df["population_type"] == population_type].copy()
    if subset.empty:
        return {}

    subset = subset.dropna(axis=1, how="all")

    _FLOAT_COLS = (
        "age", "in_force_count", "weight",
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

    str_cols = subset.select_dtypes(include=["object"]).columns
    for col in str_cols:
        subset[col] = subset[col].str.strip()

    validator_fn = {
        "in_payment": BPAValidator.validate_in_payment,
        "deferred":   BPAValidator.validate_deferred,
        "dependant":  BPAValidator.validate_dependant,
        "enhanced":   BPAValidator.validate_enhanced,
    }[population_type]
    validator_fn(subset, registry=registry)

    result: dict = {}
    for deal_id, group in subset.groupby("deal_id"):
        result[deal_id] = group.drop(columns=["population_type"]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Per-run-type dispatchers
# ---------------------------------------------------------------------------

def _run_liability_only(run_config, fund_config, model_points, progress):
    from engine.run_modes.base_run import RunStatus
    from engine.run_modes.liability_only_run import LiabilityOnlyRun

    assumptions = _build_assumptions(run_config)
    run = LiabilityOnlyRun(
        config=run_config,
        fund_config=fund_config,
        model_points=model_points,
        assumptions=assumptions,
        progress_callback=lambda f, m: progress(0.10 + f * 0.85, m),
    )
    run.run()
    if run.result.status == RunStatus.FAILED:
        raise RuntimeError(run.result.error or "LiabilityOnlyRun failed")
    return run._store


def _run_deterministic(run_config, fund_config, model_points, progress):
    from data.loaders.asset_data_loader import AssetDataLoader
    from engine.run_modes.base_run import RunStatus
    from engine.run_modes.deterministic_run import DeterministicRun
    from engine.strategy.investment_strategy import InvestmentStrategy

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
        progress_callback=lambda f, m: progress(0.10 + f * 0.85, m),
    )
    run.run()
    if run.result.status == RunStatus.FAILED:
        raise RuntimeError(run.result.error or "DeterministicRun failed")
    return run._store


def _run_stochastic(run_config, fund_config, model_points, progress):
    from data.loaders.asset_data_loader import AssetDataLoader
    from engine.run_modes.base_run import RunStatus
    from engine.run_modes.stochastic_run import StochasticRun
    from engine.scenarios.scenario_engine import ScenarioLoader
    from engine.strategy.investment_strategy import InvestmentStrategy

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
        progress_callback=lambda f, m: progress(0.10 + f * 0.85, m),
    )
    run.run()
    if run.result.status == RunStatus.FAILED:
        raise RuntimeError(run.result.error or "StochasticRun failed")
    return run._store


def _run_bpa(run_config, fund_config, progress):
    import numpy as np
    import pandas as pd
    from data.loaders.asset_data_loader import AssetDataLoader
    from data.loaders.bpa_data_loader import BPADataLoader
    from engine.curves.rate_curve import RiskFreeRateCurve
    from engine.liability.bpa.assumptions import BPAAssumptions
    from engine.liability.bpa.mortality import TABLE_LENGTH
    from engine.liability.bpa.registry import BPADealRegistry
    from engine.matching_adjustment.fundamental_spread import FundamentalSpreadTable
    from engine.run_modes.base_run import RunStatus
    from engine.run_modes.bpa_run import BPARun

    bpa = run_config.input_sources.bpa_inputs

    progress(0.02, "Loading deal registry")
    registry = BPADealRegistry.from_csv(bpa.bpa_deals_path)

    progress(0.05, "Loading mortality basis")
    mortality = BPADataLoader.load_mortality_basis(bpa.mortality_dir)

    progress(0.10, "Loading BPA model points")
    mp_df          = pd.read_csv(bpa.bpa_model_points_path)
    in_payment_mps = _split_bpa_mps(mp_df, "in_payment", registry)
    deferred_mps   = _split_bpa_mps(mp_df, "deferred",   registry)
    dependant_mps  = _split_bpa_mps(mp_df, "dependant",  registry)
    enhanced_mps   = _split_bpa_mps(mp_df, "enhanced",   registry)

    progress(0.20, "Loading asset portfolio")
    asset_loader = AssetDataLoader(run_config.input_sources.asset_data_path)
    asset_loader.load()
    asset_loader.validate()
    asset_model = asset_loader.to_asset_model()

    progress(0.25, "Loading asset cashflows")
    asset_cfs       = pd.read_csv(bpa.asset_cashflows_path)
    asset_cfs["t"]  = asset_cfs["t"].astype(int)
    asset_cfs["cf"] = asset_cfs["cf"].astype(float)

    progress(0.28, "Preparing asset metadata")
    raw_assets = pd.read_csv(run_config.input_sources.asset_data_path)
    _MA_COLS   = [
        "asset_id", "cashflow_type", "currency",
        "has_credit_risk_transfer", "has_qualifying_currency_swap",
        "rating", "seniority", "tenor_years", "spread_bps",
    ]
    missing_ma = set(_MA_COLS) - set(raw_assets.columns)
    if missing_ma:
        raise ValueError(
            f"BPA asset file is missing MA eligibility columns: {sorted(missing_ma)}"
        )
    assets_df = raw_assets[_MA_COLS].copy()
    for bool_col in ("has_credit_risk_transfer", "has_qualifying_currency_swap"):
        assets_df[bool_col] = assets_df[bool_col].map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes")
        )

    progress(0.30, "Loading fundamental spread table")
    fs_table = FundamentalSpreadTable.from_csv(bpa.fundamental_spread_path)

    progress(0.35, "Building BPA assumptions")
    root       = run_config.input_sources.assumption_tables.tables_root_dir
    curve_df   = pd.read_csv(root / "rate_curve.csv")
    spot_rates = dict(zip(curve_df["maturity_yr"].astype(float), curve_df["spot_rate"].astype(float)))
    rate_curve = RiskFreeRateCurve(spot_rates=spot_rates)
    ill_health = np.zeros(TABLE_LENGTH, dtype=float)

    assumptions = BPAAssumptions(
        mortality=mortality,
        valuation_year=run_config.projection.valuation_date.year,
        discount_curve=rate_curve,
        inflation_rate=0.025,
        rpi_rate=0.030,
        tv_rate=0.02,
        ill_health_rates=ill_health,
        expense_pa=150.0,
    )

    progress(0.40, "Starting BPA projection")
    run = BPARun(
        config=run_config,
        fund_config=fund_config,
        in_payment_mps=in_payment_mps,
        deferred_mps=deferred_mps,
        dependant_mps=dependant_mps,
        enhanced_mps=enhanced_mps,
        assumptions=assumptions,
        asset_model=asset_model,
        assets_df=assets_df,
        asset_cfs=asset_cfs,
        fs_table=fs_table,
        deal_registry=registry,
        projection_years=run_config.projection.projection_term_years,
        monthly_years=min(
            run_config.bpa.monthly_years if run_config.bpa is not None else 10,
            run_config.projection.projection_term_years,
        ),
        progress_callback=lambda f, m: progress(0.40 + f * 0.55, m),
    )
    run.run()
    if run.result.status == RunStatus.FAILED:
        raise RuntimeError(run.result.error or "BPARun failed")
    return run._store


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------

def cmd_optimise(args) -> int:
    """
    Select the optimal BPA MA asset portfolio for a new deal.

    Pricing workflow (DECISIONS.md §62):
      Step B → LiabilityOnlyRun produces liability_annual_cfs.csv + BEL
      Step C → this command selects the optimal portfolio
      Step D → use output CSV as asset input to BPARun
    """
    import pandas as pd
    from engine.curves.rate_curve import RiskFreeRateCurve
    from engine.matching_adjustment.fundamental_spread import FundamentalSpreadTable
    from engine.portfolio_optimiser.candidate_loader import CandidateLoader
    from engine.portfolio_optimiser.liability_profile import LiabilityProfile
    from engine.portfolio_optimiser.optimiser import PortfolioOptimiser
    from engine.portfolio_optimiser.output_writer import write_asset_csv

    # ------------------------------------------------------------------
    # 1. Load inputs
    # ------------------------------------------------------------------
    candidates_path   = Path(args.candidates)
    liability_cfs_path = Path(args.liability_cfs)
    rfr_curve_path    = Path(args.rfr_curve)
    fs_table_path     = Path(args.fs_table)
    output_path       = Path(args.output)

    for p in (candidates_path, liability_cfs_path, rfr_curve_path, fs_table_path):
        if not p.exists():
            print(f"Error: input file not found: {p}", file=sys.stderr)
            return 1

    try:
        candidates = CandidateLoader.from_csv(candidates_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: invalid candidate universe:\n  {exc}", file=sys.stderr)
        return 1

    try:
        liab_df = pd.read_csv(liability_cfs_path)
    except Exception as exc:
        print(f"Error: could not read liability cashflows:\n  {exc}", file=sys.stderr)
        return 1

    try:
        curve_df   = pd.read_csv(rfr_curve_path)
        spot_rates = dict(
            zip(curve_df["maturity_yr"].astype(float), curve_df["spot_rate"].astype(float))
        )
        rfr_curve = RiskFreeRateCurve(spot_rates=spot_rates)
    except Exception as exc:
        print(f"Error: could not load RFR curve:\n  {exc}", file=sys.stderr)
        return 1

    try:
        fs_table = FundamentalSpreadTable.from_csv(fs_table_path)
    except Exception as exc:
        print(f"Error: could not load fundamental spread table:\n  {exc}", file=sys.stderr)
        return 1

    try:
        liability_profile = LiabilityProfile.from_cashflows(liab_df, rfr_curve)
    except ValueError as exc:
        print(f"Error: invalid liability cashflows:\n  {exc}", file=sys.stderr)
        return 1

    bel_target: float | None = float(args.bel_target) if args.bel_target is not None else None

    # ------------------------------------------------------------------
    # 2. Run optimiser
    # ------------------------------------------------------------------
    opt = PortfolioOptimiser(
        fs_table=fs_table,
        rfr_curve=rfr_curve,
        liability_currency=args.liability_currency,
        duration_tolerance_years=float(args.duration_tolerance),
        ma_highly_predictable_cap=float(args.hp_cap),
    )

    try:
        result = opt.optimise(candidates, liability_profile, bel_target=bel_target)
    except Exception as exc:
        print(f"Error: optimisation failed unexpectedly:\n  {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 3. Print summary
    # ------------------------------------------------------------------
    if result.feasible:
        n_selected = len(result.selected_portfolio)
        n_total    = len(candidates)
        ma         = result.ma_result.ma_benefit_bps
        cf_status  = "PASS" if result.ma_result.cashflow_test_passes else "FAIL"
        bel_cov    = result.bel_covered
        bel_tgt    = bel_target if bel_target is not None else liability_profile.bel
        dur_port   = result.pre_ma_duration_years
        dur_liab   = result.liability_duration_years
        dur_gap    = result.duration_gap_years

        print("Optimisation result : FEASIBLE")
        print(f"MA benefit          : {ma:.1f} bps")
        print(
            f"Portfolio duration  : {dur_port:.2f} yr  "
            f"(liability: {dur_liab:.2f} yr, gap: {dur_gap:+.2f} yr)"
        )
        print(f"BEL covered         : {bel_cov:,.0f}  (target: {bel_tgt:,.0f})")
        print(f"Cashflow match      : {cf_status}")
        print(f"Selected bonds      : {n_selected} of {n_total} candidates")

        try:
            write_asset_csv(result, output_path)
            print(f"Output written to   : {output_path}")
        except Exception as exc:
            print(f"Error: could not write output CSV:\n  {exc}", file=sys.stderr)
            return 1

        return 0

    else:
        print("Optimisation result : INFEASIBLE")
        print(f"Reason              : {result.infeasibility_reason}")
        print()
        print(
            "Suggestions:\n"
            "  - Add longer-maturity bonds to cover the full liability term\n"
            "  - Widen --duration-tolerance\n"
            "  - Increase the candidate universe size\n"
            "  - Check --liability-currency matches the bond currencies"
        )
        return 1


def cmd_run(args) -> int:
    """Execute a projection from a YAML config and write results to disk."""
    from engine.config.run_config import RunConfig, RunType
    from engine.config.fund_config import FundConfig
    import pandas as pd

    quiet    = args.quiet
    progress = _make_progress(quiet)

    # ------------------------------------------------------------------
    # 1. Load and validate config
    # ------------------------------------------------------------------
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    if not quiet:
        print(f"ALM System CLI — run")
        print(f"  Config : {config_path}")

    try:
        run_config = RunConfig.from_yaml(config_path)
    except Exception as exc:
        print(f"\nError: config validation failed:\n  {exc}", file=sys.stderr)
        return 1

    # Apply overrides
    if args.run_id:
        run_config = run_config.model_copy(update={"run_id": args.run_id})
    if args.output_dir:
        updated_output = run_config.output.model_copy(
            update={"output_dir": Path(args.output_dir)}
        )
        run_config = run_config.model_copy(update={"output": updated_output})

    output_dir = Path(run_config.output.output_dir)

    if not quiet:
        print(f"  Run ID : {run_config.run_id}")
        print(f"  Type   : {run_config.run_type.value}")
        print(f"  Output : {output_dir}")
        print()

    # ------------------------------------------------------------------
    # 2. Load FundConfig
    # ------------------------------------------------------------------
    fund_config = None
    if run_config.input_sources.fund_config_path is not None:
        progress(0.01, "Loading fund config")
        fund_config = FundConfig.from_yaml(run_config.input_sources.fund_config_path)

    # ------------------------------------------------------------------
    # 3. Load model points (conventional runs)
    # ------------------------------------------------------------------
    model_points = None
    if run_config.input_sources.model_points is not None:
        progress(0.05, "Loading model points")
        from data.loaders.liability_data_loader import LiabilityDataLoader
        assert run_config.input_sources.model_points.file is not None
        mp_file    = run_config.input_sources.model_points.file.file_path
        column_map = run_config.input_sources.model_points.column_map
        loader     = LiabilityDataLoader(mp_file, column_map=column_map)
        loader.load()
        loader.validate()
        model_points = loader.to_dataframe()

    # ------------------------------------------------------------------
    # 4. Dispatch to run mode
    # ------------------------------------------------------------------
    run_type = run_config.run_type
    try:
        if run_type == RunType.LIABILITY_ONLY:
            store = _run_liability_only(run_config, fund_config, model_points, progress)
        elif run_type == RunType.DETERMINISTIC:
            store = _run_deterministic(run_config, fund_config, model_points, progress)
        elif run_type == RunType.STOCHASTIC:
            store = _run_stochastic(run_config, fund_config, model_points, progress)
        elif run_type == RunType.BPA:
            store = _run_bpa(run_config, fund_config, progress)
        else:
            print(f"Error: unsupported run_type: {run_type!r}", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"\nError: projection failed:\n  {exc}", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # 5. Write results
    # ------------------------------------------------------------------
    progress(0.97, "Writing results")
    try:
        _write_results(store, run_config, output_dir, quiet)
    except Exception as exc:
        print(f"\nError: failed to write results:\n  {exc}", file=sys.stderr)
        return 1

    progress(1.0, "Done")
    if not quiet:
        print()
        print("Run complete.")
    return 0


def cmd_validate(args) -> int:
    """Validate a YAML config and print a summary without running."""
    from engine.config.run_config import RunConfig

    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        run_config = RunConfig.from_yaml(config_path)
    except Exception as exc:
        print(f"INVALID — {config_path}\n\n{exc}", file=sys.stderr)
        return 1

    print(f"VALID — {config_path}")
    print()
    print(run_config.summary())
    return 0


def cmd_schema(args) -> int:
    """Print the RunConfig JSON schema to stdout."""
    from engine.config.run_config import RunConfig

    schema = RunConfig.model_json_schema()
    print(json.dumps(schema, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "ALM System CLI — run projections directly against the engine "
            "without requiring the API server or worker queue."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python main.py validate config_files/bpa_run.yaml\n"
            "  uv run python main.py run config_files/bpa_run.yaml\n"
            "  uv run python main.py run config_files/bpa_run.yaml --output-dir /tmp/out\n"
            "  uv run python main.py schema | python -m json.tool\n"
        ),
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    # -- run ----------------------------------------------------------------
    p_run = sub.add_parser(
        "run",
        help="Execute a projection from a YAML config.",
        description=(
            "Load a YAML RunConfig, run the projection directly against the engine, "
            "and write results to the output directory specified in the config."
        ),
    )
    p_run.add_argument(
        "config_file",
        metavar="CONFIG",
        help="Path to the YAML run config file.",
    )
    p_run.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Override the output_dir from the config (results written here).",
    )
    p_run.add_argument(
        "--run-id",
        metavar="ID",
        default=None,
        help="Override the run_id from the config.",
    )
    p_run.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    p_run.set_defaults(func=cmd_run)

    # -- validate -----------------------------------------------------------
    p_val = sub.add_parser(
        "validate",
        help="Validate a YAML config without running.",
        description="Load and validate a YAML RunConfig, printing a summary if valid.",
    )
    p_val.add_argument(
        "config_file",
        metavar="CONFIG",
        help="Path to the YAML run config file.",
    )
    p_val.set_defaults(func=cmd_validate)

    # -- schema -------------------------------------------------------------
    p_sch = sub.add_parser(
        "schema",
        help="Print the RunConfig JSON schema.",
        description=(
            "Print the full JSON schema for RunConfig to stdout. "
            "Pipe to 'python -m json.tool' for pretty-printing."
        ),
    )
    p_sch.set_defaults(func=cmd_schema)

    # -- optimise -----------------------------------------------------------
    p_opt = sub.add_parser(
        "optimise",
        help="Select the optimal BPA asset portfolio for a new deal.",
        description=(
            "Pre-deal pricing tool: given a candidate bond universe and pre-computed "
            "liability cashflows, selects the MA-optimal portfolio that passes the PRA "
            "cashflow matching test and meets a duration target.  See DECISIONS.md §62."
        ),
    )
    p_opt.add_argument(
        "--candidates",
        metavar="CSV",
        required=True,
        help="Candidate bond universe CSV (columns: asset_id, face_value, "
             "annual_coupon_rate, maturity_month, accounting_basis, "
             "initial_book_value, calibration_spread, rating, seniority, "
             "cashflow_type, currency, has_credit_risk_transfer, "
             "has_qualifying_currency_swap, spread_bps).",
    )
    p_opt.add_argument(
        "--liability-cfs",
        metavar="CSV",
        required=True,
        help="Liability annual cashflows CSV (columns: t (int year), cf (float)).",
    )
    p_opt.add_argument(
        "--rfr-curve",
        metavar="CSV",
        required=True,
        help="Risk-free rate curve CSV (columns: tenor, rate).",
    )
    p_opt.add_argument(
        "--fs-table",
        metavar="CSV",
        required=True,
        help="PRA Fundamental Spread table CSV "
             "(columns: rating, seniority, tenor_lower, tenor_upper, fs_bps).",
    )
    p_opt.add_argument(
        "--output",
        metavar="CSV",
        required=True,
        help="Output path for the selected portfolio CSV.",
    )
    p_opt.add_argument(
        "--bel-target",
        metavar="FLOAT",
        type=float,
        default=None,
        help="BEL target (£).  If omitted, computed from the liability cashflows "
             "discounted at the RFR curve.",
    )
    p_opt.add_argument(
        "--duration-tolerance",
        metavar="YEARS",
        type=float,
        default=0.5,
        help="Maximum allowed |asset duration − liability duration| in years "
             "(default: 0.5).",
    )
    p_opt.add_argument(
        "--liability-currency",
        metavar="CCY",
        default="GBP",
        help="Liability currency for MA eligibility filtering (default: GBP).",
    )
    p_opt.add_argument(
        "--hp-cap",
        metavar="FRAC",
        type=float,
        default=0.35,
        help="Maximum fraction of portfolio PV in highly-predictable assets "
             "(default: 0.35).",
    )
    p_opt.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    p_opt.set_defaults(func=cmd_optimise)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
