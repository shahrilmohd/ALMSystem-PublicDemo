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
# Key resolution for openai_compatible providers
# ---------------------------------------------------------------------------
# Maps a substring of base_url → candidate env var names (priority order).
# The first env var in the list that is non-empty in the environment is used.
# Falls through to the generic fallbacks if nothing matches.

_URL_KEY_CANDIDATES: list[tuple[str, list[str]]] = [
    ("openai.com",      ["OPENAI_API_KEY",   "LLM_API_KEY"]),
    ("googleapis.com",  ["GOOGLE_API_KEY",   "LLM_API_KEY"]),
    ("deepseek.com",    ["DEEPSEEK_API_KEY", "LLM_API_KEY"]),
    ("azure",           ["AZURE_OPENAI_API_KEY", "LLM_API_KEY"]),
    ("anthropic.com",   ["ANTHROPIC_API_KEY","LLM_API_KEY"]),
]

# Private / localhost endpoints (Ollama, vLLM, …) don't need a real key.
# The OpenAI SDK requires a non-empty string, so we fall back to "none".
_LOCAL_URL_PREFIXES = ("localhost", "127.0.0.1", "192.168.", "10.", "172.")


def _resolve_openai_compat_key(base_url: str | None) -> str:
    """
    Return the name of the env var that holds the API key for this endpoint.

    Tries each candidate for the matching URL pattern; returns the first one
    whose env var is set.  For localhost endpoints where no key is needed,
    also accepts a missing env var (the agent layer will use "none").
    """
    is_local = base_url and any(p in base_url.lower() for p in _LOCAL_URL_PREFIXES)

    if base_url:
        url_lower = base_url.lower()
        for fragment, candidates in _URL_KEY_CANDIDATES:
            if fragment in url_lower:
                for var in candidates:
                    if os.getenv(var):
                        return var
                # Pattern matched but no candidate is set — return first anyway
                # so the error message names the right variable.
                return candidates[0]

    # No pattern matched — try generic fallbacks in order.
    for var in ("LLM_API_KEY", "OPENAI_API_KEY"):
        if os.getenv(var):
            return var

    # Localhost: no key required; caller will supply a dummy.
    if is_local:
        return "LLM_API_KEY"

    return "LLM_API_KEY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_config_from_request(body: AIChatRequest) -> AILayerConfig:
    """
    Build AILayerConfig using GUI settings from the request body.

    The API key is resolved server-side from the environment (.env file).
    It is never sent over the wire.

    For openai_compatible providers the key env var is inferred from the
    base_url (e.g. api.openai.com → OPENAI_API_KEY, deepseek.com →
    DEEPSEEK_API_KEY).  Falls back to LLM_API_KEY if no pattern matches.
    Local/Ollama endpoints work without a real key (dummy "none" is used).
    """
    if body.provider == "anthropic":
        key_env_var = "ANTHROPIC_API_KEY"
    else:
        key_env_var = _resolve_openai_compat_key(body.base_url)

    is_local = body.base_url and any(
        p in body.base_url.lower() for p in _LOCAL_URL_PREFIXES
    )
    key_value = os.getenv(key_env_var)

    if not key_value:
        if is_local:
            # Local models (Ollama, vLLM) don't need a real key.
            # Set a dummy so the OpenAI SDK is satisfied.
            os.environ[key_env_var] = "none"
        else:
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


def _build_external_config_from_request(body: AIChatRequest) -> AILayerConfig | None:
    """
    Build the optional external AILayerConfig for RegulatoryResearchAgent.

    Returns None if external_provider is not set — regulatory research routing
    will be unavailable for that session.

    The external LLM is intentionally kept separate from the internal LLM so that
    the regulatory research context never contains firm-specific data.
    """
    if not body.external_provider:
        return None

    ext_key_env_var = (
        "ANTHROPIC_API_KEY"
        if body.external_provider == "anthropic"
        else _resolve_openai_compat_key(body.external_base_url)
    )
    if not os.getenv(ext_key_env_var):
        # Non-fatal — external config is optional; log a warning and return None.
        log.warning(
            "external_provider=%r requested but env var %r not set; "
            "regulatory_research routing unavailable for this session.",
            body.external_provider, ext_key_env_var,
        )
        return None

    try:
        return AILayerConfig(
            provider=        body.external_provider,
            model=           body.external_model or "claude-sonnet-4-6",
            api_key_env_var= ext_key_env_var,
            base_url=        body.external_base_url or None,
            deployment_mode= body.deployment_mode,
        )
    except (ValueError, RuntimeError) as exc:
        log.warning("Failed to build external config: %s", exc)
        return None


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

    # New session — build internal + optional external configs from request body.
    config          = _build_config_from_request(body)
    external_config = _build_external_config_from_request(body)
    session_id      = str(uuid.uuid4())
    api_client      = ALMApiClient(base_url=_API_BASE_URL)

    try:
        orchestrator = ALMOrchestrator(
            internal_config= config,
            external_config= external_config,
            api_client=      api_client,
            context_run_id=  body.context_run_id,
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
        agent_trace=      result.agent_trace,
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
