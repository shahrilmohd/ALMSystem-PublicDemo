"""
ALMOrchestrator — routes user messages to the correct specialist agent.

Routing
-------
A lightweight classification call (one LLM round, no tools, no history) classifies
each user message into one of three categories:

  "analyst"  — user wants to understand existing results (BEL, TVOG, cash flows)
  "advisor"  — user wants to change assumptions, modify a config, or re-run
  "unknown"  — anything else (greetings, out-of-scope questions)

No special keywords are required from the user.  The LLM infers intent from natural
language.  The only hard-coded short-circuit is for Phase 3 topics (IFRS 17, SCR,
BPA, etc.) — these return a "not available yet" message without an LLM call.

Reviewer pattern
----------------
When ConfigAdvisorAgent proposes a new RunConfig (pending_submit is set), the
orchestrator automatically passes it to ReviewerAgent before returning to the UI.
The combined reply includes the advisor explanation + reviewer verdict.
pending_submit is only forwarded to the UI if the reviewer did not reject the config.

Session history
---------------
Conversation history is maintained internally across turns in Anthropic message
format.  The desktop UI passes a plain string; the orchestrator handles wrapping.
Call reset_history() to start a new session.

Phase 3 agents
--------------
IFRS17Agent, SolvencyIIAgent, and BPAAgent are instantiated as stubs but raise
NotImplementedError.  The orchestrator catches this and returns a polite message.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ai_layer.agents.base_agent import AgentResponse
from ai_layer.agents.bpa_specialist import BPAAgent
from ai_layer.agents.config_advisor import ConfigAdvisorAgent
from ai_layer.agents.ifrs17_specialist import IFRS17Agent
from ai_layer.agents.modelling_agent import ModellingAgent
from ai_layer.agents.reviewer import ReviewerAgent
from ai_layer.agents.run_analyst import RunAnalystAgent
from ai_layer.agents.solvency2_specialist import SolvencyIIAgent
from ai_layer.config import AILayerConfig

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Router classifier prompt
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = """\
You are a routing classifier for an actuarial ALM (Asset and Liability Model) assistant.

Classify the user message into exactly ONE of these four categories:

  analyst   — the user wants to understand or explain existing projection results.
              Examples: explaining BEL, TVOG, cash flow movements, scenario spread,
              why a number is high or low, what drove a result.

  advisor   — the user wants to change assumptions, modify a RunConfig, or submit
              a new projection run with different settings.
              Examples: increase mortality rates, change projection term, re-run
              with a different ESG scenario file, update lapse rates.

  modelling — the user wants to understand how the model works internally:
              calculation mechanics, code logic, formulas, class/method behaviour,
              or the actuarial rationale behind a design decision.
              Examples: "how is BEL calculated?", "how does the stochastic loop work?",
              "what does Fund.run() do?", "how are AC and FVTPL bonds treated
              differently?", "walk me through the TVOG derivation".

  unknown   — anything else: greetings, out-of-scope questions, clarifications
              about the system, or ambiguous requests.

Reply with ONLY the single lowercase word: analyst, advisor, modelling, or unknown.
No punctuation. No explanation. No other text.
"""

# Topics that belong to Phase 3 — short-circuit before making an LLM call.
_PHASE3_TOPICS = (
    "ifrs17", "ifrs 17", "ifrs-17",
    "solvency ii", "solvency2", "solvency 2",
    "scr stress", "capital adequacy",
    "bpa ", "bulk purchase annuity",
    "matching adjustment", "fundamental spread",
    "csm", "contractual service margin",
    "lrc", "lic", "loss component",
    "risk adjustment",
)


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResponse:
    """Structured response returned to the desktop UI after each turn."""

    reply: str
    """Final text to display in the chat window."""

    agent_used: str
    """Which agent produced the reply: 'analyst', 'advisor', 'orchestrator'."""

    pending_submit: dict[str, Any] | None = None
    """
    Set when ConfigAdvisorAgent proposed a run AND ReviewerAgent did not reject it.

    Contains:
        config_json : str  — the proposed RunConfig as a JSON string
        reviewer    : dict — {verdict, summary, issues, suggestions}

    The desktop UI must show a confirmation dialog with this data.
    submit_run must only be called after the actuary explicitly approves.
    """

    reviewer_verdict: dict[str, Any] | None = None
    """Full ReviewerAgent verdict, present whenever a config was reviewed this turn."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """All tool calls made across all agents this turn (for audit logging)."""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ALMOrchestrator:
    """
    Routes user messages to specialist agents and manages session history.

    Parameters
    ----------
    config : AILayerConfig
        Provider, model, API key, and deployment mode.
    api_client : Any | None
        ALMApiClient instance injected into all agents.  Pass None in unit tests.
    context_run_id : str | None
        The run currently selected in the desktop UI.  Prepended to each user
        message so agents know which run to reference by default.
    """

    def __init__(
        self,
        config: AILayerConfig,
        api_client: Any = None,
        context_run_id: str | None = None,
    ) -> None:
        config.assert_production_safe()

        self._config = config
        self._api_client = api_client
        self.context_run_id = context_run_id

        # Conversation history in Anthropic message format, shared across turns.
        self._history: list[dict[str, Any]] = []

        # Phase 2 agents — model is resolved automatically via each agent's
        # _AGENT_TYPE class variable and AILayerConfig.model_for().
        self._analyst   = RunAnalystAgent(config, api_client)
        self._advisor   = ConfigAdvisorAgent(config, api_client)
        self._reviewer  = ReviewerAgent(config, api_client)
        self._modelling = ModellingAgent(config, api_client)

        # Phase 3 stubs — instantiated but raise NotImplementedError on use.
        self._ifrs17    = IFRS17Agent(config, api_client)
        self._solvency2 = SolvencyIIAgent(config, api_client)
        self._bpa       = BPAAgent(config, api_client)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> OrchestratorResponse:
        """
        Process one user turn and return a structured response.

        Conversation history is maintained internally — the caller does not
        need to pass history between turns.

        Parameters
        ----------
        user_message : str
            The actuary's message as plain text.

        Returns
        -------
        OrchestratorResponse
        """
        # Short-circuit Phase 3 topics without an LLM call.
        if self._is_phase3_topic(user_message):
            reply = (
                "Analysis for IFRS 17, Solvency II, SCR, and BPA is not yet available. "
                "These capabilities will be enabled in Phase 3 once the corresponding "
                "engine modules are validated (Steps 16–22)."
            )
            self._append_history(user_message, reply)
            return OrchestratorResponse(reply=reply, agent_used="orchestrator")

        # Prepend context run_id if set so all agents know which run is in focus.
        if self.context_run_id:
            full_message = f"[Context run: {self.context_run_id}]\n\n{user_message}"
        else:
            full_message = user_message

        # Classify intent.
        route = self._call_router(full_message)
        log.debug("Route=%r  message=%.80r", route, user_message)

        if route == "advisor":
            return self._run_advisor_flow(user_message, full_message)

        if route == "analyst":
            return self._run_analyst_flow(user_message, full_message)

        if route == "modelling":
            return self._run_modelling_flow(user_message, full_message)

        # "unknown" — friendly fallback.
        reply = (
            "I can help you understand ALM projection results or propose changes to "
            "run configurations. Could you clarify what you would like to do?"
        )
        self._append_history(user_message, reply)
        return OrchestratorResponse(reply=reply, agent_used="orchestrator")

    def reset_history(self) -> None:
        """Clear the conversation history and start a fresh session."""
        self._history.clear()

    # ------------------------------------------------------------------
    # Router — single cheap LLM call, no tools, no history
    # ------------------------------------------------------------------

    def _call_router(self, message: str) -> str:
        """
        Classify the user's message as 'analyst', 'advisor', or 'unknown'.

        Makes one short LLM call (max_tokens=10, temperature=0) using the
        router model (defaults to Haiku — one-word classification needs no
        expert reasoning).  Returns 'analyst' on any error so the user
        always gets a response.
        """
        try:
            if self._config.provider == "anthropic":
                import anthropic  # type: ignore[import-untyped]
                client = anthropic.Anthropic(api_key=self._config.get_api_key())
                response = client.messages.create(
                    model=self._config.model_for("router"),
                    max_tokens=10,
                    temperature=0.0,
                    system=_ROUTER_SYSTEM,
                    messages=[{"role": "user", "content": message}],
                )
                raw = response.content[0].text.strip().lower()

            else:
                import openai  # type: ignore[import-untyped]
                client = openai.OpenAI(
                    base_url=self._config.base_url,
                    api_key=self._config.get_api_key(),
                )
                response = client.chat.completions.create(
                    model=self._config.model_for("router"),
                    max_tokens=10,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": _ROUTER_SYSTEM},
                        {"role": "user",   "content": message},
                    ],
                )
                raw = (response.choices[0].message.content or "").strip().lower()

            # Exact match first, then prefix match for any extra tokens.
            if raw in ("analyst", "advisor", "modelling", "unknown"):
                return raw
            for label in ("analyst", "advisor", "modelling", "unknown"):
                if raw.startswith(label):
                    return label
            log.warning("Router returned unexpected value %r — defaulting to analyst", raw)
            return "analyst"

        except Exception as exc:
            log.warning("Router call failed (%s) — defaulting to analyst", exc)
            return "analyst"

    # ------------------------------------------------------------------
    # Agent flows
    # ------------------------------------------------------------------

    def _run_analyst_flow(
        self, raw_message: str, full_message: str
    ) -> OrchestratorResponse:
        """Run RunAnalystAgent and return its response."""
        messages = self._history + [{"role": "user", "content": full_message}]

        try:
            response: AgentResponse = self._analyst.chat(messages)
        except NotImplementedError as exc:
            return OrchestratorResponse(reply=str(exc), agent_used="orchestrator")

        self._append_history(raw_message, response.reply)
        return OrchestratorResponse(
            reply=response.reply,
            agent_used="analyst",
            tool_calls=response.tool_calls,
        )

    def _run_advisor_flow(
        self, raw_message: str, full_message: str
    ) -> OrchestratorResponse:
        """
        Run ConfigAdvisorAgent, then automatically run ReviewerAgent if a
        config proposal is returned.  Apply the reviewer pattern before
        passing anything to the UI.
        """
        messages = self._history + [{"role": "user", "content": full_message}]

        try:
            advisor_resp: AgentResponse = self._advisor.chat(messages)
        except NotImplementedError as exc:
            return OrchestratorResponse(reply=str(exc), agent_used="orchestrator")

        all_tool_calls = list(advisor_resp.tool_calls)

        # Advisor answered a question without proposing a new run.
        if advisor_resp.pending_submit is None:
            self._append_history(raw_message, advisor_resp.reply)
            return OrchestratorResponse(
                reply=advisor_resp.reply,
                agent_used="advisor",
                tool_calls=all_tool_calls,
            )

        # Advisor proposed a config — run it through ReviewerAgent.
        config_json = advisor_resp.pending_submit.get("config_json", "")

        try:
            verdict = self._reviewer.review(
                config_json,
                reference_run_id=self.context_run_id,
            )
        except Exception as exc:
            log.warning("ReviewerAgent failed: %s", exc)
            verdict = {
                "verdict":     "needs_revision",
                "summary":     f"Reviewer encountered an error: {exc}",
                "issues":      [str(exc)],
                "suggestions": [],
            }

        # Append reviewer commentary to the reply.
        verdict_label  = verdict.get("verdict", "unknown")
        reviewer_reply = self._format_reviewer_verdict(verdict)
        combined_reply = advisor_resp.reply + reviewer_reply

        # Only expose pending_submit to the UI if the reviewer did not outright reject.
        pending: dict[str, Any] | None = None
        if verdict_label != "rejected":
            pending = {
                "config_json": config_json,
                "reviewer":    verdict,
            }

        self._append_history(raw_message, combined_reply)
        return OrchestratorResponse(
            reply=combined_reply,
            agent_used="advisor",
            pending_submit=pending,
            reviewer_verdict=verdict,
            tool_calls=all_tool_calls,
        )

    def _run_modelling_flow(
        self, raw_message: str, full_message: str
    ) -> OrchestratorResponse:
        """Run ModellingAgent and return its response."""
        messages = self._history + [{"role": "user", "content": full_message}]

        try:
            response: AgentResponse = self._modelling.chat(messages)
        except NotImplementedError as exc:
            return OrchestratorResponse(reply=str(exc), agent_used="orchestrator")

        self._append_history(raw_message, response.reply)
        return OrchestratorResponse(
            reply=response.reply,
            agent_used="modelling",
            tool_calls=response.tool_calls,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_phase3_topic(message: str) -> bool:
        lower = message.lower()
        return any(topic in lower for topic in _PHASE3_TOPICS)

    @staticmethod
    def _format_reviewer_verdict(verdict: dict[str, Any]) -> str:
        """Format the reviewer verdict as a markdown section appended to the reply."""
        label    = verdict.get("verdict", "unknown")
        summary  = verdict.get("summary", "")
        issues   = verdict.get("issues", [])
        suggestions = verdict.get("suggestions", [])

        lines = [f"\n\n---\n**Reviewer verdict: {label}**\n{summary}"]

        if issues:
            lines.append("\n**Issues:**")
            lines.extend(f"- {i}" for i in issues)

        if suggestions:
            lines.append("\n**Suggestions:**")
            lines.extend(f"- {s}" for s in suggestions)

        return "\n".join(lines)

    def _append_history(self, user_message: str, assistant_reply: str) -> None:
        self._history.append({"role": "user",      "content": user_message})
        self._history.append({"role": "assistant", "content": assistant_reply})
