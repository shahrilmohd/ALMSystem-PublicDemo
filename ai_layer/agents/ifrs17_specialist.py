"""
IFRS17Agent — Phase 3 stub.

Explains CSM, RA, LRC, LIC, and loss component for IFRS 17 GMM runs.
Not implemented until Phase 3 Step 23 (after IFRS 17 engine is validated).
"""
from __future__ import annotations

from typing import Any

from ai_layer.agents.base_agent import BaseAgent
from ai_layer.config import AILayerConfig


class IFRS17Agent(BaseAgent):
    """Phase 3 stub — raises NotImplementedError."""

    def __init__(self, config: AILayerConfig, api_client: Any = None) -> None:
        # Do not call super().__init__() — we don't want to build an LLM client yet.
        _ = config
        _ = api_client

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError("IFRS17Agent is not implemented until Phase 3 Step 23.")

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        raise NotImplementedError("IFRS17Agent is not implemented until Phase 3 Step 23.")

    def _execute_tool(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("IFRS17Agent is not implemented until Phase 3 Step 23.")

    def chat(self, messages: list[dict[str, Any]]) -> Any:
        raise NotImplementedError(
            "IFRS17Agent is a Phase 3 component.  "
            "It must not be used until the IFRS 17 engine (Steps 16–22) is validated."
        )
