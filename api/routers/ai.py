"""
/ai router — conversational AI assistant endpoint.

Endpoints
---------
POST /ai/chat
    Accept a user message, route it to the correct specialist agent via
    ALMOrchestrator, and return the reply plus any pending run proposal.

DELETE /ai/sessions/{session_id}
    Clear the conversation history for a session (new conversation).

Session management
------------------
Sessions are stored in memory (module-level dict) keyed by session_id (UUID).
Each session holds one ALMOrchestrator instance with its conversation history.

Configuration split
-------------------
Secret (from .env — never sent over the wire):
  ANTHROPIC_API_KEY        Anthropic API key.
  LLM_API_KEY              Key for an openai_compatible in-house model (if used).

User settings (sent in the request body from the GUI settings panel):
  provider         "anthropic" or "openai_compatible"
  model            Model ID (e.g. "claude-opus-4-6" or an in-house model name)
  base_url         Required for openai_compatible — the endpoint URL
  deployment_mode  "development" or "production"

The GUI settings are applied when a NEW session is created (first turn, no
session_id).  Subsequent turns reuse the existing session's settings.

Architectural rules
-------------------
- This router does NOT import from engine/.
- The endpoint never calls POST /runs automatically.  Any run submission is
  returned in pending_submit and requires explicit actuary approval in the UI.
"""
from __future__ import annotations

import logging
import os
import uuid

from fastapi import APIRouter, HTTPException

from ai_layer.agent import ALMOrchestrator, OrchestratorResponse
from ai_layer.config import AILayerConfig
from api.schemas.ai_schema import (
    AIChatRequest,
    AIChatResponse,
    PendingSubmitSchema,
    ReviewerVerdictSchema,
)
from frontend.desktop.api_client import ALMApiClient

log = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory session store  {session_id: ALMOrchestrator}
# ---------------------------------------------------------------------------

_sessions: dict[str, ALMOrchestrator] = {}

# ---------------------------------------------------------------------------
# API base URL for the ALMApiClient used by tool calls.
# This is the only value still read from the environment — it is a server-side
# routing detail, not a user preference.
# ---------------------------------------------------------------------------

_API_BASE_URL = os.getenv("ALM_API_BASE_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_config_from_request(body: AIChatRequest) -> AILayerConfig:
    """
    Build AILayerConfig using GUI settings from the request body.

    The API key is resolved server-side from the environment (.env file).
    It is never sent over the wire.

    The key env var name is determined by the provider:
      "anthropic"         → ANTHROPIC_API_KEY
      "openai_compatible" → LLM_API_KEY
    (These defaults match AILayerConfig; the user can override via the GUI
    if their in-house model uses a different env var name.)
    """
    key_env_var = "ANTHROPIC_API_KEY" if body.provider == "anthropic" else "LLM_API_KEY"

    if not os.getenv(key_env_var):
        raise HTTPException(
            status_code=503,
            detail=(
                f"AI Assistant is unavailable: environment variable {key_env_var!r} "
                f"is not set. Add it to your .env file and restart the server."
            ),
        )

    try:
        return AILayerConfig(
            provider=        body.provider,
            model=           body.model,
            api_key_env_var= key_env_var,
            base_url=        body.base_url or None,
            deployment_mode= body.deployment_mode,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _get_or_create_session(body: AIChatRequest) -> tuple[str, ALMOrchestrator]:
    """
    Return (session_id, orchestrator) for this request.

    - If session_id is present and known: resume that session.
    - Otherwise: create a new session using the GUI settings in the request body.
    - Always update context_run_id (the user may switch runs between turns).

    Raises HTTPException 503/422 if the config is invalid or the key is missing.
    """
    if body.session_id and body.session_id in _sessions:
        orchestrator = _sessions[body.session_id]
        orchestrator.context_run_id = body.context_run_id
        return body.session_id, orchestrator

    # New session — build config from request body settings.
    config     = _build_config_from_request(body)
    session_id = str(uuid.uuid4())
    api_client = ALMApiClient(base_url=_API_BASE_URL)

    try:
        orchestrator = ALMOrchestrator(
            config=         config,
            api_client=     api_client,
            context_run_id= body.context_run_id,
        )
    except RuntimeError as exc:
        # assert_production_safe() fired — PNC enforcement.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    _sessions[session_id] = orchestrator
    log.debug(
        "Created AI session %s  provider=%s  model=%s  mode=%s",
        session_id, body.provider, body.model, body.deployment_mode,
    )
    return session_id, orchestrator


def _build_response(
    session_id: str,
    result: OrchestratorResponse,
) -> AIChatResponse:
    """Convert OrchestratorResponse → AIChatResponse (Pydantic schema)."""
    pending = None
    if result.pending_submit:
        raw = result.pending_submit.get("reviewer", {})
        pending = PendingSubmitSchema(
            config_json=result.pending_submit.get("config_json", ""),
            reviewer=ReviewerVerdictSchema(
                verdict=     raw.get("verdict",     "unknown"),
                summary=     raw.get("summary",     ""),
                issues=      raw.get("issues",      []),
                suggestions= raw.get("suggestions", []),
            ),
        )

    return AIChatResponse(
        reply=            result.reply,
        session_id=       session_id,
        agent_used=       result.agent_used,
        pending_submit=   pending,
        reviewer_verdict= result.reviewer_verdict,
        tool_calls=       result.tool_calls,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=AIChatResponse)
def ai_chat(body: AIChatRequest) -> AIChatResponse:
    """
    Send a message to the AI assistant and receive a reply.

    **First turn:** omit `session_id` — the server creates a new session using
    the `provider`, `model`, `base_url`, and `deployment_mode` from the request
    (set by the GUI settings panel).  The server returns a `session_id`; store
    it and pass it back on all subsequent turns.

    **API key:** never sent in the request.  The server reads `ANTHROPIC_API_KEY`
    (or `LLM_API_KEY` for openai_compatible) from the `.env` file.

    **Context run:** set `context_run_id` to the run currently selected in the
    desktop UI.  Agents will use it as the default run when you ask questions
    like "explain these results" without specifying a run ID.

    **Pending submit:** when the AI proposes a new run, `pending_submit` is set.
    The desktop UI must show the proposed config and reviewer verdict, and only
    call `POST /runs` after the actuary explicitly approves.  This endpoint
    never submits runs automatically.

    Returns **503** if the required API key env var is absent or the provider
    config is invalid.
    """
    sid, orchestrator = _get_or_create_session(body)

    try:
        result = orchestrator.chat(body.message)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("AI chat error in session %s", sid)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred in the AI assistant.",
        ) from exc

    return _build_response(sid, result)


@router.delete("/sessions/{session_id}", status_code=204)
def clear_session(session_id: str) -> None:
    """
    Clear the conversation history for a session.

    Call this when the user clicks "New conversation" in the desktop UI.
    Returns 204 whether or not the session existed.
    """
    _sessions.pop(session_id, None)
    log.debug("Cleared AI session %s", session_id)
