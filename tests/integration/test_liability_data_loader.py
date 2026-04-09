"""
Integration tests for LiabilityDataLoader → LiabilityOnlyRun.

These tests verify the full pipeline:

    CSV file → LiabilityDataLoader → LiabilityOnlyRun → ResultStore

They confirm that data loaded from a file produces the same numerically
correct results as data built directly in memory (as tested in Step 5).

Test scenario
-------------
Uses tests/fixtures/sample_group_mps.csv which contains three groups:

    GRP_A: 100 ENDOW_NONPAR, SA=10,000, prem=1,200, term=1yr, dur=0mths
    GRP_B:  50 TERM,          SA= 5,000, prem=  600, term=10yr, dur=36mths
    GRP_C: 200 ENDOW_PAR,     SA=20,000, prem=2,400, term=20yr, dur=120mths

With zero assumptions and zero discount rate:

GRP_A (1-year term, starting at month 0):
    Months 0–10: premium income = 100 × 1,200/12 = 10,000/month
    Month 11:    maturity_payments = 100 × 10,000 = 1,000,000

GRP_B (10-year TERM, 36 months elapsed, 84 remaining):
    Monthly premium = 50 × 600/12 = 2,500
    No maturity payments (TERM policy — no maturity benefit)
    Death claims = 0 (zero mortality assumption)

GRP_C (20-year PAR, 120 months elapsed, 120 remaining):
    Monthly premium = 200 × 2,400/12 = 40,000
    Maturity in month 119 (relative to projection start)

Combined month 0 totals (hand-calculated):
    premiums = 10,000 + 2,500 + 40,000 = 52,500
    maturity_payments = 0 (no group is in its final month at t=0)
    net_outgo = 0 − 52,500 = −52,500
"""
from __future__ import annotations

from pathlib import Path

import pytest

from data.loaders.liability_data_loader import LiabilityDataLoader
from engine.config.fund_config import FundConfig
from engine.config.run_config import RunConfig
from engine.curves.rate_curve import RiskFreeRateCurve
from engine.liability.conventional import ConventionalAssumptions
from engine.run_modes.liability_only_run import LiabilityOnlyRun
from tests.unit.config.conftest import build_run_config_dict

# Path to the shared fixture file
_FIXTURE_CSV = Path(__file__).parent.parent / "fixtures" / "sample_group_mps.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(tmp_path: Path, projection_term_years: int = 1) -> RunConfig:
    assumption_dir   = tmp_path / "assumptions"
    assumption_dir.mkdir(exist_ok=True)
    fund_config_file = tmp_path / "fund_config.yaml"
    fund_config_file.write_text("placeholder: true\n")
    data = build_run_config_dict(
        fund_config_path=fund_config_file,
        assumption_dir=assumption_dir,
        projection_term_years=projection_term_years,
    )
    data["output"]["output_dir"] = str(tmp_path / "outputs")
    return RunConfig.from_dict(data)


def _make_fund_config() -> FundConfig:
    return FundConfig.from_dict({
        "fund_id":   "FUND_A",
        "fund_name": "Fund A",
        "saa_weights": {
            "bonds": 0.6, "equities": 0.3, "derivatives": 0.0, "cash": 0.1
        },
        "crediting_groups": [
            {"group_id": "GRP_A", "group_name": "Group A", "product_codes": ["P1"]},
        ],
    })


def _zero_assumptions() -> ConventionalAssumptions:
    return ConventionalAssumptions(
        mortality_rates={},
        lapse_rates={},
        expense_pct_premium=0.0,
        expense_per_policy=0.0,
        surrender_value_factors={},
        rate_curve=RiskFreeRateCurve.flat(0.0),
    )


# ---------------------------------------------------------------------------
# TestLoaderPipeline
# ---------------------------------------------------------------------------

class TestLoaderPipeline:
    """Verify that the loader produces the expected DataFrame."""

    def test_fixture_file_exists(self):
        assert _FIXTURE_CSV.exists(), f"Fixture not found: {_FIXTURE_CSV}"

    def test_load_validate_returns_three_rows(self):
        loader = LiabilityDataLoader(_FIXTURE_CSV)
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert len(df) == 3

    def test_group_ids_loaded_correctly(self):
        loader = LiabilityDataLoader(_FIXTURE_CSV)
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert set(df["group_id"]) == {"GRP_A", "GRP_B", "GRP_C"}

    def test_float_types_after_load(self):
        loader = LiabilityDataLoader(_FIXTURE_CSV)
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        assert df["in_force_count"].dtype == float
        assert df["sum_assured"].dtype    == float

    def test_int_types_after_load(self):
        loader = LiabilityDataLoader(_FIXTURE_CSV)
        loader.load()
        loader.validate()
        df = loader.to_dataframe()
        import pandas as pd
        assert pd.api.types.is_integer_dtype(df["policy_term_yr"])
        assert pd.api.types.is_integer_dtype(df["policy_duration_mths"])


# ---------------------------------------------------------------------------
# TestLoaderToEngineEndToEnd
# ---------------------------------------------------------------------------

class TestLoaderToEngineEndToEnd:
    """
    Load the fixture CSV, inject it into LiabilityOnlyRun, and verify
    that the engine produces numerically correct results.
    """

    @pytest.fixture
    def run(self, tmp_path) -> LiabilityOnlyRun:
        loader = LiabilityDataLoader(_FIXTURE_CSV)
        loader.load()
        loader.validate()
        mp = loader.to_dataframe()

        config      = _make_config(tmp_path, projection_term_years=1)
        fund_config = _make_fund_config()
        assumptions = _zero_assumptions()

        r = LiabilityOnlyRun(
            config=config,
            fund_config=fund_config,
            model_points=mp,
            assumptions=assumptions,
        )
        r.run()
        return r

    def test_result_count_is_12(self, run):
        """1-year projection → 12 monthly results."""
        assert run.store.result_count() == 12

    def test_premiums_at_t0_match_hand_calculation(self, run):
        """
        Month 0 premiums (zero decrements):
            GRP_A: 100 × 1200/12 = 10,000
            GRP_B:  50 ×  600/12 =  2,500
            GRP_C: 200 × 2400/12 = 40,000
            Total                = 52,500
        """
        r = run.store.get(scenario_id=0, timestep=0)
        assert r.cashflows.premiums == pytest.approx(52_500.0, rel=1e-6)

    def test_no_maturity_payments_at_t0(self, run):
        """
        At t=0: GRP_A has 11 months remaining, GRP_B/C have many months.
        No group is in its final month → maturity_payments = 0.
        """
        r = run.store.get(scenario_id=0, timestep=0)
        assert r.cashflows.maturity_payments == pytest.approx(0.0, abs=1e-9)

    def test_net_outgo_negative_at_t0(self, run):
        """
        With no outgo and positive premium income, net_outgo is negative
        (convention: net_outgo = outgo − premiums → −52,500 at t=0).
        """
        r = run.store.get(scenario_id=0, timestep=0)
        assert r.cashflows.net_outgo == pytest.approx(-52_500.0, rel=1e-6)

    def test_grp_a_matures_in_final_month(self, run):
        """
        GRP_A: 1-year term starting from duration=0.
        At t=11, remaining_term=1 → all 100 policies mature.
        maturity_payments includes GRP_A's 100 × 10,000 = 1,000,000.
        GRP_B and GRP_C do not mature (far from term end).
        """
        r = run.store.get(scenario_id=0, timestep=11)
        assert r.cashflows.maturity_payments == pytest.approx(1_000_000.0, rel=1e-6)

    def test_loaded_results_match_in_memory_results(self, tmp_path):
        """
        Build the same model points in memory (no file I/O) and confirm the
        BEL at t=0 matches the file-loaded run exactly.
        This proves the loader does not alter the values.
        """
        import pandas as pd_mem

        mp_memory = pd_mem.DataFrame([
            {
                "group_id": "GRP_A", "in_force_count": 100.0,
                "sum_assured": 10_000.0, "annual_premium": 1_200.0,
                "attained_age": 50, "policy_code": "ENDOW_NONPAR",
                "policy_term_yr": 1, "policy_duration_mths": 0,
                "accrued_bonus_per_policy": 0.0,
            },
            {
                "group_id": "GRP_B", "in_force_count": 50.0,
                "sum_assured": 5_000.0, "annual_premium": 600.0,
                "attained_age": 60, "policy_code": "TERM",
                "policy_term_yr": 10, "policy_duration_mths": 36,
                "accrued_bonus_per_policy": 0.0,
            },
            {
                "group_id": "GRP_C", "in_force_count": 200.0,
                "sum_assured": 20_000.0, "annual_premium": 2_400.0,
                "attained_age": 45, "policy_code": "ENDOW_PAR",
                "policy_term_yr": 20, "policy_duration_mths": 120,
                "accrued_bonus_per_policy": 500.0,
            },
        ])

        config      = _make_config(tmp_path, projection_term_years=1)
        fund_config = _make_fund_config()

        run_memory = LiabilityOnlyRun(
            config=config,
            fund_config=fund_config,
            model_points=mp_memory,
            assumptions=_zero_assumptions(),
        )
        run_memory.run()

        loader = LiabilityDataLoader(_FIXTURE_CSV)
        loader.load()
        loader.validate()
        mp_file = loader.to_dataframe()

        (tmp_path / "run2").mkdir(exist_ok=True)
        config2 = _make_config(tmp_path / "run2", projection_term_years=1)
        run_file = LiabilityOnlyRun(
            config=config2,
            fund_config=fund_config,
            model_points=mp_file,
            assumptions=_zero_assumptions(),
        )
        run_file.run()

        bel_memory = run_memory.store.get(0, 0).bel
        bel_file   = run_file.store.get(0, 0).bel
        assert bel_memory == pytest.approx(bel_file, rel=1e-6)
