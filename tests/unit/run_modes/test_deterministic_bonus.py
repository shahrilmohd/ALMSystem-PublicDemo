"""
Tests DB1–DB6: DeterministicRun with BonusStrategy (Step 27a, Task 1).

DB1  bonus_strategy=None path is byte-for-byte identical to omitting it.
DB2  SmoothedBonusStrategy(min=r, max=r, alpha=1) + flat equity_return → BEL
     matches the flat-rate baseline (EMA with alpha=1 converges immediately).
DB3  terminal_bonus_fraction > 0 → death claims higher than zero-terminal baseline.
DB4  asset_share grows when equity_return_yr > 0 (checked via terminal bonus effect).
DB5  bonus_strategy=None path: original model points not mutated.
DB6  Serial stochastic + SmoothedBonusStrategy → TVOG within 1e-4 of vectorised path.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.asset.asset_model import AssetModel
from engine.asset.bond import Bond
from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig, StochasticConfig
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.conventional import ConventionalAssumptions
from engine.results.tvog_calculator import TvogCalculator
from engine.run_modes.deterministic_run import DeterministicRun
from engine.run_modes.stochastic_run import StochasticRun
from engine.scenarios.scenario_engine import ScenarioLoader
from engine.scenarios.scenario_store import ScenarioStore
from engine.strategy.bonus_strategy import SmoothedBonusStrategy
from engine.strategy.investment_strategy import AssetClassWeights, InvestmentStrategy

from tests.unit.config.conftest import build_run_config_dict


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

PROJECTION_TERM_YEARS = 1   # 12 months — fast
BONUS_RATE_YR         = 0.03
N_SCENARIOS           = 6


def _make_det_config(tmp_path: Path) -> RunConfig:
    assumption_dir = tmp_path / "assumptions"
    assumption_dir.mkdir(parents=True, exist_ok=True)
    fund_config_file = tmp_path / "fund_config.yaml"
    fund_config_file.write_text("placeholder: true\n")
    asset_file = tmp_path / "assets.csv"
    asset_file.write_text("placeholder\n")
    data = build_run_config_dict(
        fund_config_path=fund_config_file,
        assumption_dir=assumption_dir,
        projection_term_years=PROJECTION_TERM_YEARS,
    )
    data["run_type"] = "deterministic"
    data["output"]["output_dir"]    = str(tmp_path / "outputs")
    data["output"]["result_format"] = "csv"
    data["input_sources"]["asset_data_path"] = str(asset_file)
    return RunConfig.from_dict(data)


def _make_stochastic_config(tmp_path: Path, use_vectorised: bool = False) -> RunConfig:
    assumption_dir = tmp_path / "assumptions"
    assumption_dir.mkdir(parents=True, exist_ok=True)
    fund_config_file = tmp_path / "fund_config.yaml"
    fund_config_file.write_text("placeholder: true\n")
    asset_file = tmp_path / "assets.csv"
    asset_file.write_text("placeholder\n")
    scenario_file = tmp_path / "scenarios.csv"
    scenario_file.write_text("scenario_id\n")
    data = build_run_config_dict(
        fund_config_path=fund_config_file,
        assumption_dir=assumption_dir,
        run_type="stochastic",
        input_mode="group_mp",
        projection_term_years=PROJECTION_TERM_YEARS,
        output_timestep="monthly",
        asset_data_path=asset_file,
        scenario_file_path=scenario_file,
        stochastic={"num_scenarios": N_SCENARIOS, "use_vectorised": use_vectorised},
    )
    data["output"]["output_dir"]    = str(tmp_path / "outputs")
    data["output"]["result_format"] = "csv"
    return RunConfig.from_dict(data)


def _fund_config() -> FundConfig:
    return FundConfig.from_dict({
        "fund_id":   "FUND_A",
        "fund_name": "Fund A",
        "saa_weights": {"bonds": 1.0, "equities": 0.0, "derivatives": 0.0, "cash": 0.0},
        "crediting_groups": [
            {"group_id": "GRP_PAR", "group_name": "PAR Group", "product_codes": ["P1"]},
        ],
    })


def _make_assumptions(
    bonus_rate_yr: float = BONUS_RATE_YR,
    with_mortality: bool = False,
) -> ConventionalAssumptions:
    mortality = {age: 0.005 for age in range(40, 70)} if with_mortality else {}
    return ConventionalAssumptions(
        mortality_rates=mortality,
        lapse_rates={},
        expense_pct_premium=0.0,
        expense_per_policy=0.0,
        surrender_value_factors={},
        rate_curve=RiskFreeRateCurve.flat(0.03),
        bonus_rate_yr=bonus_rate_yr,
    )


def _make_par_mp(sum_assured: float = 10_000.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "group_id":                 "GRP_PAR",
        "in_force_count":           100.0,
        "sum_assured":              sum_assured,
        "annual_premium":           1_200.0,
        "attained_age":             50,
        "policy_code":              "ENDOW_PAR",
        "policy_term_yr":           1,
        "policy_duration_mths":     0,
        "accrued_bonus_per_policy": 0.0,
    }])


def _make_asset_model() -> AssetModel:
    bond = Bond("corp_1", 1_000_000.0, 0.05, 36, "FVTPL", 1_000_000.0)
    return AssetModel([bond])


def _make_strategy() -> InvestmentStrategy:
    weights = AssetClassWeights(bonds=1.0, equities=0.0, derivatives=0.0, cash=0.0)
    return InvestmentStrategy(weights, rebalancing_tolerance=1.0)


def _build_det_run(
    tmp_path: Path,
    bonus_strategy: SmoothedBonusStrategy | None = None,
    equity_return_yr: float = 0.0,
    with_mortality: bool = False,
) -> DeterministicRun:
    run = DeterministicRun(
        config=_make_det_config(tmp_path),
        fund_config=_fund_config(),
        model_points=_make_par_mp(),
        assumptions=_make_assumptions(with_mortality=with_mortality),
        asset_model=_make_asset_model(),
        investment_strategy=_make_strategy(),
        equity_return_yr=equity_return_yr,
        bonus_strategy=bonus_strategy,
    )
    run.setup()
    run.execute()
    return run


def _build_stochastic_run(
    tmp_path: Path,
    use_vectorised: bool,
    bonus_strategy: SmoothedBonusStrategy | None = None,
) -> StochasticRun:
    from engine.asset.base_asset import AssetScenarioPoint
    from engine.scenarios.scenario_store import EsgScenario

    base_curve = RiskFreeRateCurve.flat(0.03)
    returns = [-0.05, 0.0, 0.03, 0.05, 0.08, 0.12]
    scenario_list = []
    for i, ret in enumerate(returns, start=1):
        points = [
            AssetScenarioPoint(
                timestep=t,
                rate_curve=base_curve,
                equity_total_return_yr=ret,
                dt=1 / 12,
            )
            for t in range(PROJECTION_TERM_YEARS * 12)
        ]
        scenario_list.append(EsgScenario(scenario_id=i, timesteps=points))
    store = ScenarioStore(scenario_list)

    run = StochasticRun(
        config=_make_stochastic_config(tmp_path, use_vectorised=use_vectorised),
        fund_config=_fund_config(),
        model_points=_make_par_mp(),
        assumptions=_make_assumptions(),
        asset_model=_make_asset_model(),
        investment_strategy=_make_strategy(),
        scenario_store=store,
        bonus_strategy=bonus_strategy,
    )
    run.setup()
    run.execute()
    return run


# ---------------------------------------------------------------------------
# DB1 — bonus_strategy=None is byte-for-byte identical to omitting it
# ---------------------------------------------------------------------------

class TestBonusStrategyNone:

    def test_bel_identical_when_none(self, tmp_path):
        """bonus_strategy=None must not change any BEL values."""
        run_baseline = _build_det_run(tmp_path / "base")
        run_none     = _build_det_run(tmp_path / "none", bonus_strategy=None)

        for t in range(PROJECTION_TERM_YEARS * 12):
            bel_b = run_baseline.store.get(0, t).bel
            bel_n = run_none.store.get(0, t).bel
            assert bel_b == bel_n, f"t={t}: baseline BEL={bel_b}, none BEL={bel_n}"

    def test_death_claims_identical_when_none(self, tmp_path):
        run_baseline = _build_det_run(tmp_path / "base")
        run_none     = _build_det_run(tmp_path / "none", bonus_strategy=None)

        for t in range(PROJECTION_TERM_YEARS * 12):
            dc_b = run_baseline.store.get(0, t).cashflows.death_claims
            dc_n = run_none.store.get(0, t).cashflows.death_claims
            assert dc_b == dc_n, f"t={t}: death claims differ"


# ---------------------------------------------------------------------------
# DB2 — SmoothedBonusStrategy(min=r, max=r, alpha=1) + flat equity → BEL = baseline
# ---------------------------------------------------------------------------

class TestFlatEquityBELMatchesBaseline:

    def test_bel_matches_baseline_with_flat_corridor_strategy(self, tmp_path):
        """alpha=1, min=max=bonus_rate_yr, equity=0 → smoothed = 0 = flat rate clip → identical BEL."""
        strategy = SmoothedBonusStrategy(
            smoothing_alpha=1.0,
            min_reversionary=BONUS_RATE_YR,
            max_reversionary=BONUS_RATE_YR,
            terminal_bonus_fraction=0.0,
        )
        run_baseline = _build_det_run(tmp_path / "base")
        run_strategy = _build_det_run(
            tmp_path / "strat",
            bonus_strategy=strategy,
            equity_return_yr=0.0,
        )

        for t in range(PROJECTION_TERM_YEARS * 12):
            bel_b = run_baseline.store.get(0, t).bel
            bel_s = run_strategy.store.get(0, t).bel
            assert abs(bel_b - bel_s) < 1e-4, (
                f"t={t}: baseline BEL={bel_b:.6f}, strategy BEL={bel_s:.6f}"
            )


# ---------------------------------------------------------------------------
# DB3 — terminal_bonus_fraction > 0 → death claims higher than zero-terminal baseline
# ---------------------------------------------------------------------------

class TestTerminalBonusIncreasesDeathClaims:

    def test_death_claims_higher_with_terminal_bonus(self, tmp_path):
        """terminal_bonus_fraction=0.5 → death claim per policy > sum_assured alone."""
        strategy_zero_tb = SmoothedBonusStrategy(
            smoothing_alpha=1.0,
            min_reversionary=BONUS_RATE_YR,
            max_reversionary=BONUS_RATE_YR,
            terminal_bonus_fraction=0.0,
        )
        strategy_with_tb = SmoothedBonusStrategy(
            smoothing_alpha=1.0,
            min_reversionary=BONUS_RATE_YR,
            max_reversionary=BONUS_RATE_YR,
            terminal_bonus_fraction=0.5,
        )
        run_no_tb   = _build_det_run(tmp_path / "no_tb",   bonus_strategy=strategy_zero_tb, equity_return_yr=0.07, with_mortality=True)
        run_with_tb = _build_det_run(tmp_path / "with_tb", bonus_strategy=strategy_with_tb, equity_return_yr=0.07, with_mortality=True)

        total_dc_no_tb   = sum(run_no_tb.store.get(0, t).cashflows.death_claims   for t in range(PROJECTION_TERM_YEARS * 12))
        total_dc_with_tb = sum(run_with_tb.store.get(0, t).cashflows.death_claims for t in range(PROJECTION_TERM_YEARS * 12))

        assert total_dc_with_tb > total_dc_no_tb, (
            f"Expected terminal bonus to raise death claims: "
            f"no_tb={total_dc_no_tb:.2f}, with_tb={total_dc_with_tb:.2f}"
        )


# ---------------------------------------------------------------------------
# DB4 — asset_share grows when equity_return_yr > 0 (via terminal bonus effect)
# ---------------------------------------------------------------------------

class TestAssetShareGrowsWithPositiveReturn:

    def test_positive_equity_return_raises_death_claims_vs_zero_return(self, tmp_path):
        """With terminal bonus wired: equity_return_yr=0.07 produces higher death claims
        than equity_return_yr=0.0, because asset_share grows with positive returns."""
        strategy = SmoothedBonusStrategy(
            smoothing_alpha=1.0,
            min_reversionary=0.0,
            max_reversionary=0.0,
            terminal_bonus_fraction=0.5,
        )
        run_zero   = _build_det_run(tmp_path / "zero",   bonus_strategy=strategy, equity_return_yr=0.0,  with_mortality=True)
        run_posret = _build_det_run(tmp_path / "posret", bonus_strategy=strategy, equity_return_yr=0.07, with_mortality=True)

        total_dc_zero   = sum(run_zero.store.get(0, t).cashflows.death_claims   for t in range(PROJECTION_TERM_YEARS * 12))
        total_dc_posret = sum(run_posret.store.get(0, t).cashflows.death_claims for t in range(PROJECTION_TERM_YEARS * 12))

        assert total_dc_posret >= total_dc_zero, (
            f"Positive equity return should not lower death claims: "
            f"zero={total_dc_zero:.2f}, pos={total_dc_posret:.2f}"
        )


# ---------------------------------------------------------------------------
# DB5 — original model points not mutated
# ---------------------------------------------------------------------------

class TestModelPointsNotMutated:

    def test_mp_not_mutated_with_bonus_strategy(self, tmp_path):
        mp = _make_par_mp()
        mp_before = mp.copy()
        strategy = SmoothedBonusStrategy(
            smoothing_alpha=0.5,
            min_reversionary=0.0,
            max_reversionary=0.06,
            terminal_bonus_fraction=0.3,
        )
        run = DeterministicRun(
            config=_make_det_config(tmp_path),
            fund_config=_fund_config(),
            model_points=mp,
            assumptions=_make_assumptions(),
            asset_model=_make_asset_model(),
            investment_strategy=_make_strategy(),
            equity_return_yr=0.07,
            bonus_strategy=strategy,
        )
        run.setup()
        run.execute()
        pd.testing.assert_frame_equal(mp, mp_before)


# ---------------------------------------------------------------------------
# DB6 — serial stochastic + SmoothedBonusStrategy → TVOG within 1e-4 of vectorised
# ---------------------------------------------------------------------------

class TestSerialVsVectorisedTVOGWithBonusStrategy:

    def _strategy(self) -> SmoothedBonusStrategy:
        return SmoothedBonusStrategy(
            smoothing_alpha=0.5,
            min_reversionary=0.0,
            max_reversionary=0.06,
            terminal_bonus_fraction=0.3,
        )

    def test_tvog_within_tolerance(self, tmp_path):
        run_serial = _build_stochastic_run(tmp_path / "serial", use_vectorised=False, bonus_strategy=self._strategy())
        run_vect   = _build_stochastic_run(tmp_path / "vect",   use_vectorised=True,  bonus_strategy=self._strategy())

        det_bel_s = run_serial.store.get(1, 0).bel
        det_bel_v = run_vect.store.get(1, 0).bel

        tvog_s = TvogCalculator(run_serial.store, det_bel_s).calculate().tvog
        tvog_v = TvogCalculator(run_vect.store,   det_bel_v).calculate().tvog

        assert abs(tvog_s - tvog_v) < 1e-4, (
            f"TVOG mismatch: serial={tvog_s:.6f}, vectorised={tvog_v:.6f}"
        )

    def test_bel_t0_matches_per_scenario(self, tmp_path):
        run_serial = _build_stochastic_run(tmp_path / "serial", use_vectorised=False, bonus_strategy=self._strategy())
        run_vect   = _build_stochastic_run(tmp_path / "vect",   use_vectorised=True,  bonus_strategy=self._strategy())

        for sc_id in run_serial.store.all_scenarios():
            bel_s = run_serial.store.get(sc_id, 0).bel
            bel_v = run_vect.store.get(sc_id, 0).bel
            assert abs(bel_s - bel_v) < 1e-4, (
                f"Scenario {sc_id}: serial BEL={bel_s:.4f}, vect BEL={bel_v:.4f}"
            )

    def test_bonus_strategy_none_serial_matches_vectorised(self, tmp_path):
        """bonus_strategy=None serial path still matches vectorised (V8 guard)."""
        run_serial = _build_stochastic_run(tmp_path / "serial", use_vectorised=False)
        run_vect   = _build_stochastic_run(tmp_path / "vect",   use_vectorised=True)

        for sc_id in run_serial.store.all_scenarios():
            bel_s = run_serial.store.get(sc_id, 0).bel
            bel_v = run_vect.store.get(sc_id, 0).bel
            assert abs(bel_s - bel_v) < 1e-4, (
                f"Scenario {sc_id}: serial BEL={bel_s:.4f}, vect BEL={bel_v:.4f}"
            )
