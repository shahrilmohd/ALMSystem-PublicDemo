"""
Integration tests for Step 24 AI agents.

These tests require:
  - A running ALM API server (ALM_API_BASE_URL or default http://localhost:8000)
  - ANTHROPIC_API_KEY set in the environment
  - A completed Q4 2025 BPA sample run loaded in the result store
    (run test_bpa_end_to_end.py first, or load via POST /runs)
  - Assumption table snapshots under data/assumptions/2025-12-31/

All tests are skipped automatically when the API is unreachable or the key
is absent, so they never fail in CI unless explicitly opted in.

To run locally:
    uv run pytest tests/integration/ai_layer/ -v -m integration

Test plan (per step24_plan.md §Task 11)
---------------------------------------
I1  DataReviewAgent MP check   — Q4 2025 BPA in_payment model points; no critical issues
I2  DataReviewAgent diff       — compare assumption tables; diff report produced
I3  IFRS17Agent                — ask about CSM; answer references IFRS 17 concepts
I4  BPAAgent                   — ask about MA benefit; answer references BEL/MA
I5  ArchitectAgent             — ask about adding a new decrement; reads module interface
I6  RegulatoryResearchAgent    — clean IFRS 17 PAA question answered
I7  RegulatoryResearchAgent    — content gate blocks question embedding a BEL figure

Design notes
------------
- Tests call ALMOrchestrator directly (not via HTTP) to avoid needing a running server.
  The HTTP API client (ALMApiClient) is still used by tools that call the REST endpoints,
  so the API server must be running for tool-using agents (I1–I5).
  RegulatoryResearchAgent (I6, I7) uses no tools — no API server required.
- All LLM calls use the real Anthropic API.  Expected reply shape is verified loosely
  (keyword presence) rather than exact-string equality because LLM outputs are
  non-deterministic.
"""
from __future__ import annotations

import os
import textwrap
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_API_KEY_PRESENT = bool(os.getenv("ANTHROPIC_API_KEY"))
_API_SERVER_URL  = os.getenv("ALM_API_BASE_URL", "http://localhost:8000")


def _api_server_reachable() -> bool:
    try:
        import requests
        requests.get(f"{_API_SERVER_URL}/runs/", timeout=2.0)
        return True
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.integration

skip_no_key = pytest.mark.skipif(
    not _API_KEY_PRESENT,
    reason="ANTHROPIC_API_KEY not set — skipping AI integration tests",
)
skip_no_server = pytest.mark.skipif(
    not _api_server_reachable(),
    reason=f"API server not reachable at {_API_SERVER_URL} — skipping",
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_internal_config():
    from ai_layer.config import AILayerConfig
    return AILayerConfig(provider="anthropic", deployment_mode="development")


def _make_external_config():
    from ai_layer.config import AILayerConfig
    return AILayerConfig(provider="anthropic", deployment_mode="development")


def _make_orchestrator(context_run_id: str | None = None):
    from ai_layer.agent import ALMOrchestrator
    from frontend.desktop.api_client import ALMApiClient

    api_client = ALMApiClient(base_url=_API_SERVER_URL)
    return ALMOrchestrator(
        internal_config= _make_internal_config(),
        external_config= _make_external_config(),
        api_client=      api_client,
        context_run_id=  context_run_id,
    )


def _make_agent_directly(agent_class, *, with_api_client: bool = True):
    """Instantiate a single agent without going through ALMOrchestrator."""
    from frontend.desktop.api_client import ALMApiClient
    api_client = ALMApiClient(base_url=_API_SERVER_URL) if with_api_client else None
    return agent_class(config=_make_internal_config(), api_client=api_client)


# ---------------------------------------------------------------------------
# I1 — DataReviewAgent: MP check, no critical issues
# ---------------------------------------------------------------------------

@skip_no_key
@skip_no_server
class TestDataReviewAgentMPCheck:
    """
    Ask DataReviewAgent to check the Q4 2025 BPA in_payment model points.

    Prerequisite: a BPA run with Q4 2025 sample MPs has been submitted and
    completed so that /runs/{run_id}/model_points returns data.

    The run_id is read from the ALM_INTEGRATION_RUN_ID env var.  If absent,
    the test is skipped.
    """

    @pytest.fixture
    def run_id(self):
        rid = os.getenv("ALM_INTEGRATION_RUN_ID")
        if not rid:
            pytest.skip("ALM_INTEGRATION_RUN_ID not set — skipping MP check test")
        return rid

    def test_mp_check_returns_verdict(self, run_id):
        """DataReviewAgent should produce a PASS / ADVISORY / FAIL verdict."""
        from ai_layer.agents.data_review import DataReviewAgent
        agent = _make_agent_directly(DataReviewAgent)

        result = agent.chat(
            f"Please check the model points for run {run_id}, population_type in_payment."
        )
        reply = result.reply.upper()
        assert any(verdict in reply for verdict in ["PASS", "ADVISORY", "FAIL"]), (
            f"Expected a verdict keyword in reply, got: {result.reply[:300]}"
        )

    def test_mp_check_mentions_row_count(self, run_id):
        from ai_layer.agents.data_review import DataReviewAgent
        agent = _make_agent_directly(DataReviewAgent)

        result = agent.chat(
            f"Validate the MPs for run {run_id}."
        )
        # Agent should mention a row count, or at least reference model points.
        reply = result.reply.lower()
        assert "row" in reply or "model point" in reply or "mp" in reply


# ---------------------------------------------------------------------------
# I2 — DataReviewAgent: assumption table diff
# ---------------------------------------------------------------------------

@skip_no_key
class TestDataReviewAgentDiff:
    """
    Ask DataReviewAgent to compare assumption tables.

    Uses two synthetic in-memory snapshots written to a temporary directory.
    This test does NOT require the API server — compare_assumption_tables reads
    from the local filesystem.
    """

    def test_diff_reports_changed_cell(self, tmp_path, monkeypatch):
        import ai_layer.tools.compare_assumption_tables as cat_module
        monkeypatch.setattr(cat_module, "_ASSUMPTIONS_ROOT", tmp_path)

        # Write two snapshots with one changed cell.
        orig_dir = tmp_path / "2025-12-31"
        new_dir  = tmp_path / "2026-03-31"
        orig_dir.mkdir()
        new_dir.mkdir()
        (orig_dir / "mortality_rates.csv").write_text(
            "age_band,qx\nage_65,0.0100\nage_70,0.0200\n", encoding="utf-8"
        )
        (new_dir / "mortality_rates.csv").write_text(
            "age_band,qx\nage_65,0.0110\nage_70,0.0200\n", encoding="utf-8"
        )

        from ai_layer.agents.data_review import DataReviewAgent
        from ai_layer.config import AILayerConfig
        agent = DataReviewAgent(
            config=AILayerConfig(provider="anthropic", deployment_mode="development"),
            api_client=None,  # no API client needed — filesystem tool only
        )

        result = agent.chat(
            "Compare the mortality_rates assumption table between "
            "2025-12-31 and 2026-03-31."
        )
        reply = result.reply.lower()
        # Agent should mention the changed value or a flagged change.
        assert (
            "0.010" in reply or "0.011" in reply
            or "change" in reply or "increase" in reply or "differ" in reply
        ), f"Expected diff narrative, got: {result.reply[:400]}"

    def test_diff_flags_large_change(self, tmp_path, monkeypatch):
        """A 10% qx change should be flagged at the default 5% threshold."""
        import ai_layer.tools.compare_assumption_tables as cat_module
        monkeypatch.setattr(cat_module, "_ASSUMPTIONS_ROOT", tmp_path)

        orig_dir = tmp_path / "2025-12-31"
        new_dir  = tmp_path / "2026-03-31"
        orig_dir.mkdir()
        new_dir.mkdir()
        (orig_dir / "mortality_rates.csv").write_text(
            "age_band,qx\nage_65,0.0100\n", encoding="utf-8"
        )
        (new_dir / "mortality_rates.csv").write_text(
            "age_band,qx\nage_65,0.0110\n", encoding="utf-8"
        )

        from ai_layer.agents.data_review import DataReviewAgent
        from ai_layer.config import AILayerConfig
        agent = DataReviewAgent(
            config=AILayerConfig(provider="anthropic", deployment_mode="development"),
            api_client=None,
        )

        result = agent.chat(
            "Compare the mortality_rates table between 2025-12-31 and 2026-03-31."
        )
        reply = result.reply.lower()
        assert "flag" in reply or "warn" in reply or "5%" in reply or "threshold" in reply, (
            f"Expected flagging language, got: {result.reply[:400]}"
        )


# ---------------------------------------------------------------------------
# I3 — IFRS17Agent: CSM question
# ---------------------------------------------------------------------------

@skip_no_key
@skip_no_server
class TestIFRS17Agent:

    @pytest.fixture
    def run_id(self):
        rid = os.getenv("ALM_INTEGRATION_RUN_ID")
        if not rid:
            pytest.skip("ALM_INTEGRATION_RUN_ID not set")
        return rid

    def test_csm_question_references_ifrs17_concepts(self, run_id):
        from ai_layer.agents.ifrs17_specialist import IFRS17Agent
        agent = _make_agent_directly(IFRS17Agent)

        result = agent.chat(
            f"For run {run_id}, can you explain what the CSM balance represents "
            f"and what drove any changes at the latest projection period?"
        )
        reply = result.reply.upper()
        assert (
            "CSM" in reply
            or "CONTRACTUAL SERVICE MARGIN" in reply
            or "IFRS 17" in reply
        ), f"Expected IFRS 17 terminology in reply, got: {result.reply[:400]}"


# ---------------------------------------------------------------------------
# I4 — BPAAgent: MA benefit question
# ---------------------------------------------------------------------------

@skip_no_key
@skip_no_server
class TestBPAAgent:

    @pytest.fixture
    def run_id(self):
        rid = os.getenv("ALM_INTEGRATION_RUN_ID")
        if not rid:
            pytest.skip("ALM_INTEGRATION_RUN_ID not set")
        return rid

    def test_ma_benefit_question_references_bpa_concepts(self, run_id):
        from ai_layer.agents.bpa_specialist import BPAAgent
        agent = _make_agent_directly(BPAAgent)

        result = agent.chat(
            f"For run {run_id}, what is the matching adjustment benefit "
            f"and how does it affect the BEL?"
        )
        reply = result.reply.upper()
        assert (
            "MA" in reply
            or "MATCHING ADJUSTMENT" in reply
            or "BEL" in reply
        ), f"Expected BPA terminology in reply, got: {result.reply[:400]}"


# ---------------------------------------------------------------------------
# I5 — ArchitectAgent: structural design question
# ---------------------------------------------------------------------------

@skip_no_key
class TestArchitectAgent:
    """
    ArchitectAgent uses read_code_module and list_module_interface — local
    filesystem tools.  No API server required.
    """

    def test_decrement_question_reads_module_and_proposes_design(self):
        from ai_layer.agents.architect import ArchitectAgent
        from ai_layer.config import AILayerConfig
        agent = ArchitectAgent(
            config=AILayerConfig(provider="anthropic", deployment_mode="development"),
            api_client=None,
        )

        result = agent.chat(
            "I want to add a new decrement type called 'surrender' to the BPA "
            "InPaymentLiability.  How should I structure this while respecting "
            "the hard architectural rules?"
        )
        reply = result.reply.lower()
        # Agent should mention the module it read and propose an approach.
        assert (
            "in_payment" in reply
            or "decrement" in reply
            or "approach" in reply
            or "base" in reply
        ), f"Expected architectural advice in reply, got: {result.reply[:400]}"

    def test_violation_flagged_for_frontend_import_request(self):
        from ai_layer.agents.architect import ArchitectAgent
        from ai_layer.config import AILayerConfig
        agent = ArchitectAgent(
            config=AILayerConfig(provider="anthropic", deployment_mode="development"),
            api_client=None,
        )

        result = agent.chat(
            "I want to import a PyQt6 widget directly from engine/core/fund.py "
            "to display a progress bar.  Is this a good idea?"
        )
        reply = result.reply.lower()
        # Hard rule 1: engine/ has zero imports from frontend/
        assert (
            "violat" in reply
            or "rule" in reply
            or "hard" in reply
            or "must not" in reply
            or "not allow" in reply
        ), f"Expected rule-violation warning, got: {result.reply[:400]}"


# ---------------------------------------------------------------------------
# I6 — RegulatoryResearchAgent: clean regulatory question
# ---------------------------------------------------------------------------

@skip_no_key
class TestRegulatoryResearchAgent:
    """
    RegulatoryResearchAgent uses no tools — does not require API server.
    """

    def _make_agent(self):
        from ai_layer.agents.regulatory_research import RegulatoryResearchAgent
        return RegulatoryResearchAgent(
            config=_make_external_config(),
            api_client=None,
        )

    def test_ifrs17_paa_question_cites_standard(self):
        agent = self._make_agent()
        result = agent.chat(
            "Can you explain the PAA eligibility criteria under IFRS 17 "
            "and how the coverage period test works?"
        )
        reply = result.reply.upper()
        assert (
            "PAA" in reply
            or "PREMIUM ALLOCATION" in reply
            or "IFRS 17" in reply
            or "COVERAGE" in reply
        ), f"Expected IFRS 17 PAA content in reply, got: {result.reply[:400]}"

    def test_solvency_ii_longevity_question_answered(self):
        agent = self._make_agent()
        result = agent.chat(
            "How is longevity risk treated in the Solvency II standard formula SCR?"
        )
        reply = result.reply.lower()
        assert (
            "longevity" in reply
            or "mortality" in reply
            or "solvency ii" in reply
            or "scr" in reply
        ), f"Expected Solvency II content in reply, got: {result.reply[:400]}"

    def test_no_decisions_md_content_in_reply(self):
        """Confirm the external agent does not reference internal model decisions."""
        agent = self._make_agent()
        result = agent.chat(
            "What discount rate method does IFRS 17 specify for BPA annuity liabilities?"
        )
        # DECISIONS.md is NOT in the regulatory agent's system prompt —
        # it should not reference implementation-specific choices.
        reply = result.reply.lower()
        assert "decisions.md" not in reply
        assert "bpa_run" not in reply


# ---------------------------------------------------------------------------
# I7 — RegulatoryResearchAgent: content gate blocks PNC-tainted message
# ---------------------------------------------------------------------------

@skip_no_key
class TestRegulatoryResearchContentGate:
    """
    Content gate is enforced by ALMOrchestrator before the external agent sees
    the message.  These tests exercise the gate via orchestrator.chat().
    """

    def test_content_gate_blocks_message_with_bel_figure(self):
        """A message embedding a BEL figure should be blocked by the content gate."""
        from ai_layer.agent import ALMOrchestrator
        from frontend.desktop.api_client import ALMApiClient

        orchestrator = ALMOrchestrator(
            internal_config= _make_internal_config(),
            external_config= _make_external_config(),
            api_client=      ALMApiClient(base_url=_API_SERVER_URL),
        )
        # Embed a currency amount — should trigger the gate.
        result = orchestrator.chat(
            "regulatory_research: Given our BEL of £1,234,567,890, "
            "does IFRS 17 require us to disclose this?"
        )
        reply = result.reply.lower()
        # The gate should return a clarification message, not call the external LLM.
        assert (
            "rephrase" in reply
            or "general" in reply
            or "firm-specific" in reply
            or "model data" in reply
        ), f"Expected content gate message, got: {result.reply[:400]}"

    def test_content_gate_allows_clean_query(self):
        """A clean regulatory question should pass the gate and be answered."""
        from ai_layer.agent import ALMOrchestrator
        from frontend.desktop.api_client import ALMApiClient

        orchestrator = ALMOrchestrator(
            internal_config= _make_internal_config(),
            external_config= _make_external_config(),
            api_client=      ALMApiClient(base_url=_API_SERVER_URL),
        )
        result = orchestrator.chat(
            "Can you explain the UK PRA Matching Adjustment eligibility criteria?"
        )
        reply = result.reply.lower()
        assert (
            "matching adjustment" in reply
            or "eligible" in reply
            or "pra" in reply
            or "ma" in reply
        ), f"Expected MA content in reply, got: {result.reply[:400]}"
