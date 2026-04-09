"""
SolvencyIIAgent — Phase 3 stub.

Handles SCR stress scenarios, MA offset calculations, and capital adequacy analysis.
Not implemented until Phase 3 Step 23 (after BPA engine is validated).
"""
from __future__ import annotations

from typing import Any

from ai_layer.agents.base_agent import BaseAgent
from ai_layer.config import AILayerConfig


class SolvencyIIAgent(BaseAgent):
    """Phase 3 stub — raises NotImplementedError."""

    def __init__(self, config: AILayerConfig, api_client: Any = None) -> None:
        _ = config
        _ = api_client

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError("SolvencyIIAgent is not implemented until Phase 3 Step 23.")

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        raise NotImplementedError("SolvencyIIAgent is not implemented until Phase 3 Step 23.")

    def _execute_tool(self, name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("SolvencyIIAgent is not implemented until Phase 3 Step 23.")

    def chat(self, messages: list[dict[str, Any]]) -> Any:
        raise NotImplementedError(
            "SolvencyIIAgent is a Phase 3 component.  "
            "It must not be used until the BPA engine (Steps 17–22) is validated."
        )
