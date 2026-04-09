"""
ai_layer — Multi-agent AI assistant for the ALM system.

Phase 2 scope: conventional segregated fund products only.
Phase 3 will add IFRS17Agent, SolvencyIIAgent, BPAAgent.

Entry point:  ALMOrchestrator  (ai_layer/agent.py)
Config:       AILayerConfig    (ai_layer/config.py)

PNC note: see DECISIONS.md §30 for data governance requirements.
In production, use AILayerConfig.for_production() with an on-premise endpoint.
"""
from ai_layer.agent import ALMOrchestrator, OrchestratorResponse
from ai_layer.config import AILayerConfig

__all__ = ["AILayerConfig", "ALMOrchestrator", "OrchestratorResponse"]
