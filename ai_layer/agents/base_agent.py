"""
base_agent — abstract base class for all ALM AI specialist agents.

Dual-SDK design
---------------
- config.provider == "anthropic"         → anthropic.Anthropic client
- config.provider == "openai_compatible" → openai.OpenAI(base_url=...) client

Subclasses implement:
  system_prompt   : str       — agent-specific instructions + injected knowledge
  tool_schemas    : list[dict] — Anthropic-format tool descriptors
  _execute_tool   : (name, inputs) → dict — call the appropriate tool function

The chat() method runs the tool loop (up to MAX_TOOL_ROUNDS iterations) and
returns an AgentResponse.

submit_run interception
-----------------------
When the model calls "submit_run", BaseAgent stores the proposed config in
AgentResponse.pending_submit and does NOT execute it.  The desktop UI must
show a confirmation dialog and only call submit_run() explicitly on approval.
See DECISIONS.md §30.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ai_layer.config import AILayerConfig

MAX_TOOL_ROUNDS = 5


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """Structured result returned by BaseAgent.chat()."""

    reply: str
    """Final text reply from the model."""

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    """All tool calls made during this turn (for logging / audit)."""

    pending_submit: dict[str, Any] | None = None
    """
    Set when the model requested submit_run but human approval is required.
    Contains {"config_json": <str>} from the model's tool input.
    The desktop UI must present this for approval before submitting.
    """


# ---------------------------------------------------------------------------
# Abstract base agent
# ---------------------------------------------------------------------------

class BaseAgent(ABC):
    """
    Abstract base for RunAnalystAgent, ConfigAdvisorAgent, ReviewerAgent, etc.

    Parameters
    ----------
    config : AILayerConfig
        Provider, model, API key, deployment mode.
    api_client : ALMApiClient | None
        HTTP client for tool calls.  May be None in test scenarios where only
        read-only or no tools are needed.

    Model resolution
    ----------------
    Each subclass declares a class variable ``_AGENT_TYPE`` (e.g. "analyst",
    "advisor", "reviewer", "modelling").  BaseAgent calls
    ``config.model_for(_AGENT_TYPE)`` so the tier assignment in AILayerConfig
    is applied automatically — no need to pass a model at call sites.
    Subclasses with no ``_AGENT_TYPE`` fall back to ``config.model``.
    """

    _AGENT_TYPE: str = ""   # subclasses set this to their agent key

    def __init__(self, config: AILayerConfig, api_client: Any = None) -> None:
        self._config = config
        self._model = config.model_for(self._AGENT_TYPE) if self._AGENT_TYPE else config.model
        self._api_client = api_client
        self._llm = self._build_llm_client()

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Agent-specific system prompt, including any injected knowledge."""

    @property
    @abstractmethod
    def tool_schemas(self) -> list[dict[str, Any]]:
        """
        Tool descriptors in Anthropic format:
            {"name": ..., "description": ..., "input_schema": {...}}
        BaseAgent converts to OpenAI format automatically when needed.
        """

    @abstractmethod
    def _execute_tool(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a tool call and return a JSON-serialisable result dict.
        Must NOT execute "submit_run" — BaseAgent intercepts that before calling here.
        """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, messages: list[dict[str, Any]]) -> AgentResponse:
        """
        Run the agentic tool loop for a single user turn.

        Parameters
        ----------
        messages : list[dict]
            Conversation history in Anthropic message format:
            [{"role": "user"|"assistant", "content": <str|list>}, ...]
            The system prompt is injected separately — do not include it here.

        Returns
        -------
        AgentResponse
            Final reply, tool calls made, and any pending submit_run.
        """
        all_tool_calls: list[dict[str, Any]] = []
        pending_submit: dict[str, Any] | None = None
        working_messages = list(messages)

        for _round in range(MAX_TOOL_ROUNDS):
            try:
                text, tool_uses = self._call_model(working_messages)
            except Exception as exc:  # noqa: BLE001
                return AgentResponse(
                    reply=f"I encountered an error calling the AI model: {exc}",
                    tool_calls=all_tool_calls,
                    pending_submit=pending_submit,
                )

            if not tool_uses:
                # Model is done — no more tool calls.
                return AgentResponse(
                    reply=text or "",
                    tool_calls=all_tool_calls,
                    pending_submit=pending_submit,
                )

            # ---- Process tool calls ----------------------------------------
            tool_results: list[dict[str, Any]] = []

            for tu in tool_uses:
                tool_name   = tu["name"]
                tool_input  = tu["input"]
                tool_use_id = tu.get("id", "")

                all_tool_calls.append({"name": tool_name, "input": tool_input})

                # Intercept submit_run — never auto-execute.
                if tool_name == "submit_run":
                    pending_submit = dict(tool_input)
                    result_content = {
                        "status": "pending_approval",
                        "message": (
                            "The proposed run configuration has been flagged for "
                            "human review.  It will not be submitted until the "
                            "actuary explicitly approves it in the UI."
                        ),
                    }
                else:
                    try:
                        result_content = self._execute_tool(tool_name, tool_input)
                    except Exception as exc:  # noqa: BLE001
                        result_content = {"error": f"Tool execution failed: {exc}"}

                tool_results.append({
                    "tool_use_id": tool_use_id,
                    "content":     json.dumps(result_content),
                })

            # Append assistant turn + tool results to history.
            working_messages = self._append_tool_round(
                working_messages, text, tool_uses, tool_results
            )

        # Max rounds reached — return whatever text the model last produced.
        last_text = self._extract_last_text(working_messages)
        return AgentResponse(
            reply=last_text,
            tool_calls=all_tool_calls,
            pending_submit=pending_submit,
        )

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    def _build_llm_client(self) -> Any:
        """Instantiate the appropriate SDK client."""
        if self._config.provider == "anthropic":
            import anthropic  # type: ignore[import-untyped]
            return anthropic.Anthropic(api_key=self._config.get_api_key())

        if self._config.provider == "openai_compatible":
            import openai  # type: ignore[import-untyped]
            return openai.OpenAI(
                base_url=self._config.base_url,
                api_key=self._config.get_api_key(),
            )

        raise ValueError(f"Unknown provider: {self._config.provider!r}")

    def _call_model(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Call the model and return (text_reply, tool_uses).
        tool_uses is a list of {"id", "name", "input"} dicts (may be empty).
        """
        if self._config.provider == "anthropic":
            return self._call_anthropic(messages)
        return self._call_openai(messages)

    # ------------------------------------------------------------------
    # Anthropic path
    # ------------------------------------------------------------------

    def _call_anthropic(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        # Wrap the system prompt in a content block so Anthropic caches it.
        # After the first call the cached tokens cost ~10% of normal input
        # tokens and do not count against the input token rate limit.
        system_block = [
            {
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        # Also cache the tool definitions — add a cache breakpoint after the
        # last tool so both system prompt + tools are cached together.
        tools = list(self.tool_schemas)
        if tools:
            last = dict(tools[-1])
            last["cache_control"] = {"type": "ephemeral"}
            tools = [*tools[:-1], last]

        response = self._llm.messages.create(
            model=self._model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            system=system_block,
            tools=tools,
            messages=messages,
        )
        return self._extract_anthropic(response)

    @staticmethod
    def _extract_anthropic(
        response: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Parse Anthropic response into (text, tool_uses)."""
        text = ""
        tool_uses: list[dict[str, Any]] = []
        for block in response.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_uses.append({
                    "id":    block.id,
                    "name":  block.name,
                    "input": block.input,
                })
        return text, tool_uses

    def _append_tool_round_anthropic(
        self,
        messages: list[dict[str, Any]],
        text: str,
        tool_uses: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Append assistant + tool_result turns in Anthropic format."""
        assistant_content: list[dict[str, Any]] = []
        if text:
            assistant_content.append({"type": "text", "text": text})
        for tu in tool_uses:
            assistant_content.append({
                "type":  "tool_use",
                "id":    tu["id"],
                "name":  tu["name"],
                "input": tu["input"],
            })

        tool_result_content: list[dict[str, Any]] = [
            {
                "type":        "tool_result",
                "tool_use_id": tr["tool_use_id"],
                "content":     tr["content"],
            }
            for tr in tool_results
        ]

        return messages + [
            {"role": "assistant", "content": assistant_content},
            {"role": "user",      "content": tool_result_content},
        ]

    # ------------------------------------------------------------------
    # OpenAI-compatible path
    # ------------------------------------------------------------------

    def _call_openai(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        oai_tools = [self._anthropic_to_openai_tool(t) for t in self.tool_schemas]
        oai_messages = [{"role": "system", "content": self.system_prompt}] + messages

        response = self._llm.chat.completions.create(
            model=self._model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            tools=oai_tools,
            messages=oai_messages,
        )
        return self._extract_openai(response)

    @staticmethod
    def _extract_openai(
        response: Any,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Parse OpenAI response into (text, tool_uses)."""
        message = response.choices[0].message
        text = message.content or ""
        tool_uses: list[dict[str, Any]] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_uses.append({
                    "id":    tc.id,
                    "name":  tc.function.name,
                    "input": json.loads(tc.function.arguments),
                })
        return text, tool_uses

    def _append_tool_round_openai(
        self,
        messages: list[dict[str, Any]],
        text: str,
        tool_uses: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Append assistant + tool_result turns in OpenAI format."""
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": text or ""}
        if tool_uses:
            assistant_msg["tool_calls"] = [
                {
                    "id":       tu["id"],
                    "type":     "function",
                    "function": {
                        "name":      tu["name"],
                        "arguments": json.dumps(tu["input"]),
                    },
                }
                for tu in tool_uses
            ]

        result_msgs = [
            {
                "role":         "tool",
                "tool_call_id": tr["tool_use_id"],
                "content":      tr["content"],
            }
            for tr in tool_results
        ]
        return messages + [assistant_msg] + result_msgs

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _append_tool_round(
        self,
        messages: list[dict[str, Any]],
        text: str,
        tool_uses: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._config.provider == "anthropic":
            return self._append_tool_round_anthropic(
                messages, text, tool_uses, tool_results
            )
        return self._append_tool_round_openai(
            messages, text, tool_uses, tool_results
        )

    @staticmethod
    def _anthropic_to_openai_tool(schema: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic tool schema to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name":        schema["name"],
                "description": schema.get("description", ""),
                "parameters":  schema.get("input_schema", {"type": "object", "properties": {}}),
            },
        }

    @staticmethod
    def _extract_last_text(messages: list[dict[str, Any]]) -> str:
        """Return the last assistant text from the message list (best-effort)."""
        for msg in reversed(messages):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        return block.get("text", "")
        return ""
