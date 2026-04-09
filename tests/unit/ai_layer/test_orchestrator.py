"""
Unit tests for ALMOrchestrator routing and response assembly.

All LLM calls and agent .chat() calls are mocked — no real API calls made.

Coverage
--------
Routing
    - "analyst" route → RunAnalystAgent called
    - "advisor" route without pending_submit → ConfigAdvisorAgent called; no reviewer
    - "advisor" route with pending_submit → ReviewerAgent.review() called automatically
    - "modelling" route → ModellingAgent called; reply and agent_used correct
    - "unknown" route → fallback reply, no agent called
    - Phase 3 topic → short-circuit reply, no LLM router call

Reviewer pattern
    - reviewer verdict "approved" → pending_submit forwarded to UI
    - reviewer verdict "rejected" → pending_submit NOT forwarded; reply explains rejection
    - reviewer failure → degraded to "needs_revision" verdict; pending_submit still forwarded

OrchestratorResponse
    - agent_used set correctly for each route
    - tool_calls aggregated from agent response
    - session_id / context_run_id plumbing

History
    - history grows across turns
    - reset_history() clears history

ModellingAgent routing
    - "modelling" route calls ModellingAgent.chat()
    - analyst and advisor not called on modelling route
    - reply propagated; agent_used == "modelling"
    - tool_calls from ModellingAgent propagated
    - context_run_id prepended for modelling route
    - history updated after modelling turn
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from ai_layer.agent import ALMOrchestrator
from ai_layer.agents.base_agent import AgentResponse
from ai_layer.config import AILayerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dev_config() -> AILayerConfig:
    return AILayerConfig(provider="anthropic", deployment_mode="development")


def _make_orchestrator(
    router_returns: str = "analyst",
    analyst_reply: str = "Here are the results.",
    advisor_reply: str = "I updated the config.",
    modelling_reply: str = "Here is how BEL is calculated.",
    advisor_pending: dict | None = None,
    reviewer_verdict: dict | None = None,
    api_client=None,
) -> tuple[ALMOrchestrator, MagicMock, MagicMock, MagicMock, MagicMock]:
    """
    Build an ALMOrchestrator with mocked agents and router.

    Returns (orchestrator, mock_analyst, mock_advisor, mock_reviewer, mock_modelling).
    """
    config = _dev_config()

    analyst_response = AgentResponse(
        reply=analyst_reply,
        tool_calls=[{"name": "get_run_results", "input": {"run_id": "r1"}}],
    )
    advisor_response = AgentResponse(
        reply=advisor_reply,
        tool_calls=[{"name": "get_run_config", "input": {"run_id": "r1"}}],
        pending_submit=advisor_pending,
    )
    modelling_response = AgentResponse(
        reply=modelling_reply,
        tool_calls=[{"name": "get_run_config", "input": {"run_id": "r1"}}],
    )
    default_verdict = reviewer_verdict or {
        "verdict": "approved",
        "summary": "Config looks consistent.",
        "issues": [],
        "suggestions": [],
    }

    mock_analyst   = MagicMock()
    mock_advisor   = MagicMock()
    mock_reviewer  = MagicMock()
    mock_modelling = MagicMock()
    mock_analyst.chat.return_value    = analyst_response
    mock_advisor.chat.return_value    = advisor_response
    mock_reviewer.review.return_value = default_verdict
    mock_modelling.chat.return_value  = modelling_response

    with patch("ai_layer.agent.RunAnalystAgent",   return_value=mock_analyst), \
         patch("ai_layer.agent.ConfigAdvisorAgent", return_value=mock_advisor), \
         patch("ai_layer.agent.ReviewerAgent",      return_value=mock_reviewer), \
         patch("ai_layer.agent.ModellingAgent",     return_value=mock_modelling), \
         patch("ai_layer.agent.IFRS17Agent",        return_value=MagicMock()), \
         patch("ai_layer.agent.SolvencyIIAgent",    return_value=MagicMock()), \
         patch("ai_layer.agent.BPAAgent",           return_value=MagicMock()):

        orchestrator = ALMOrchestrator(config=config, api_client=api_client)

    # Patch the router so we control routing deterministically.
    orchestrator._call_router = MagicMock(return_value=router_returns)

    return orchestrator, mock_analyst, mock_advisor, mock_reviewer, mock_modelling


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

class TestRouting:
    def test_analyst_route_calls_analyst_agent(self):
        orch, analyst, advisor, _, _ = _make_orchestrator(router_returns="analyst")
        resp = orch.chat("Why is BEL high?")
        analyst.chat.assert_called_once()
        advisor.chat.assert_not_called()
        assert resp.agent_used == "analyst"

    def test_advisor_route_calls_advisor_agent(self):
        orch, analyst, advisor, _, _ = _make_orchestrator(router_returns="advisor")
        resp = orch.chat("Increase mortality by 5%.")
        advisor.chat.assert_called_once()
        analyst.chat.assert_not_called()
        assert resp.agent_used == "advisor"

    def test_unknown_route_returns_fallback(self):
        orch, analyst, advisor, _, _ = _make_orchestrator(router_returns="unknown")
        resp = orch.chat("Hello there!")
        analyst.chat.assert_not_called()
        advisor.chat.assert_not_called()
        assert resp.agent_used == "orchestrator"
        assert "clarify" in resp.reply.lower() or "help" in resp.reply.lower()

    def test_phase3_topic_short_circuits_without_router_call(self):
        orch, _, _, _, _ = _make_orchestrator()
        resp = orch.chat("Can you explain the CSM movement?")
        orch._call_router.assert_not_called()
        assert resp.agent_used == "orchestrator"
        assert "Phase 3" in resp.reply or "not yet available" in resp.reply.lower()

    def test_phase3_topic_bpa(self):
        orch, _, _, _, _ = _make_orchestrator()
        resp = orch.chat("What is the BPA matching adjustment?")
        orch._call_router.assert_not_called()

    def test_phase3_topic_ifrs17(self):
        orch, _, _, _, _ = _make_orchestrator()
        resp = orch.chat("Explain the IFRS 17 liability movement.")
        orch._call_router.assert_not_called()


# ---------------------------------------------------------------------------
# Reviewer pattern
# ---------------------------------------------------------------------------

class TestReviewerPattern:
    _PENDING = {"config_json": '{"run_type":"STOCHASTIC"}'}

    def test_no_pending_submit_skips_reviewer(self):
        orch, _, advisor, reviewer, _ = _make_orchestrator(
            router_returns="advisor",
            advisor_pending=None,
        )
        orch.chat("Explain the config.")
        reviewer.review.assert_not_called()

    def test_pending_submit_triggers_reviewer(self):
        orch, _, _, reviewer, _ = _make_orchestrator(
            router_returns="advisor",
            advisor_pending=self._PENDING,
        )
        orch.chat("Increase mortality by 5%.")
        reviewer.review.assert_called_once()

    def test_approved_verdict_forwarded_in_pending_submit(self):
        orch, _, _, _, _ = _make_orchestrator(
            router_returns="advisor",
            advisor_pending=self._PENDING,
            reviewer_verdict={"verdict": "approved", "summary": "OK", "issues": [], "suggestions": []},
        )
        resp = orch.chat("Increase mortality by 5%.")
        assert resp.pending_submit is not None
        assert resp.pending_submit["config_json"] == self._PENDING["config_json"]

    def test_rejected_verdict_suppresses_pending_submit(self):
        orch, _, _, _, _ = _make_orchestrator(
            router_returns="advisor",
            advisor_pending=self._PENDING,
            reviewer_verdict={"verdict": "rejected", "summary": "Invalid term.", "issues": ["term=0"], "suggestions": []},
        )
        resp = orch.chat("Set projection term to 0.")
        assert resp.pending_submit is None
        assert "rejected" in resp.reply.lower()

    def test_needs_revision_verdict_forwarded(self):
        orch, _, _, _, _ = _make_orchestrator(
            router_returns="advisor",
            advisor_pending=self._PENDING,
            reviewer_verdict={"verdict": "needs_revision", "summary": "Minor issue.", "issues": ["check term"], "suggestions": []},
        )
        resp = orch.chat("Tweak the config.")
        # needs_revision is not "rejected" → still forwarded
        assert resp.pending_submit is not None

    def test_reviewer_failure_uses_needs_revision_fallback(self):
        orch, _, _, reviewer, _ = _make_orchestrator(
            router_returns="advisor",
            advisor_pending=self._PENDING,
        )
        reviewer.review.side_effect = Exception("LLM timeout")
        resp = orch.chat("Increase mortality by 5%.")
        # Should degrade gracefully, not raise
        assert resp.reviewer_verdict["verdict"] == "needs_revision"
        assert resp.pending_submit is not None   # still forwarded (not rejected)


# ---------------------------------------------------------------------------
# OrchestratorResponse fields
# ---------------------------------------------------------------------------

class TestResponseFields:
    def test_tool_calls_aggregated(self):
        orch, _, _, _, _ = _make_orchestrator(router_returns="analyst")
        resp = orch.chat("Explain BEL.")
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["name"] == "get_run_results"

    def test_reviewer_verdict_in_response(self):
        pending = {"config_json": '{"run_type":"DETERMINISTIC"}'}
        verdict = {"verdict": "approved", "summary": "Fine.", "issues": [], "suggestions": []}
        orch, _, _, _, _ = _make_orchestrator(
            router_returns="advisor",
            advisor_pending=pending,
            reviewer_verdict=verdict,
        )
        resp = orch.chat("Re-run with new mortality.")
        assert resp.reviewer_verdict is not None
        assert resp.reviewer_verdict["verdict"] == "approved"

    def test_reply_contains_reviewer_section(self):
        pending = {"config_json": '{"run_type":"DETERMINISTIC"}'}
        orch, _, _, _, _ = _make_orchestrator(
            router_returns="advisor",
            advisor_pending=pending,
        )
        resp = orch.chat("Re-run with new mortality.")
        assert "Reviewer verdict" in resp.reply


# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------

class TestHistory:
    def test_history_grows_across_turns(self):
        orch, _, _, _, _ = _make_orchestrator(router_returns="analyst")
        orch.chat("First question.")
        orch.chat("Second question.")
        # 2 turns × 2 messages (user + assistant) = 4
        assert len(orch._history) == 4

    def test_reset_history_clears_all(self):
        orch, _, _, _, _ = _make_orchestrator(router_returns="analyst")
        orch.chat("A question.")
        orch.reset_history()
        assert orch._history == []

    def test_context_run_id_prepended_to_message(self):
        orch, analyst, _, _, _ = _make_orchestrator(router_returns="analyst")
        orch.context_run_id = "run-abc"
        orch.chat("Explain this run.")
        call_args = analyst.chat.call_args[0][0]   # messages list
        last_user_msg = call_args[-1]["content"]
        assert "run-abc" in last_user_msg


# ---------------------------------------------------------------------------
# Modelling route
# ---------------------------------------------------------------------------

class TestModellingRoute:
    def test_modelling_route_calls_modelling_agent(self):
        orch, analyst, advisor, _, modelling = _make_orchestrator(
            router_returns="modelling"
        )
        orch.chat("How is BEL calculated?")
        modelling.chat.assert_called_once()
        analyst.chat.assert_not_called()
        advisor.chat.assert_not_called()

    def test_modelling_route_agent_used_is_modelling(self):
        orch, _, _, _, _ = _make_orchestrator(router_returns="modelling")
        resp = orch.chat("Explain the stochastic loop.")
        assert resp.agent_used == "modelling"

    def test_modelling_route_reply_propagated(self):
        orch, _, _, _, _ = _make_orchestrator(
            router_returns="modelling",
            modelling_reply="BEL is computed in two passes.",
        )
        resp = orch.chat("How is BEL calculated?")
        assert resp.reply == "BEL is computed in two passes."

    def test_modelling_route_tool_calls_propagated(self):
        orch, _, _, _, _ = _make_orchestrator(router_returns="modelling")
        resp = orch.chat("Walk me through Fund.run().")
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["name"] == "get_run_config"

    def test_modelling_route_no_pending_submit(self):
        orch, _, _, _, _ = _make_orchestrator(router_returns="modelling")
        resp = orch.chat("How does bond accounting work?")
        assert resp.pending_submit is None

    def test_modelling_route_context_run_id_prepended(self):
        orch, _, _, _, modelling = _make_orchestrator(router_returns="modelling")
        orch.context_run_id = "run-xyz"
        orch.chat("Explain this run's BEL.")
        call_args = modelling.chat.call_args[0][0]
        last_user_msg = call_args[-1]["content"]
        assert "run-xyz" in last_user_msg

    def test_modelling_route_history_updated(self):
        orch, _, _, _, _ = _make_orchestrator(router_returns="modelling")
        orch.chat("How is TVOG derived?")
        assert len(orch._history) == 2
        assert orch._history[0]["role"] == "user"
        assert orch._history[1]["role"] == "assistant"
