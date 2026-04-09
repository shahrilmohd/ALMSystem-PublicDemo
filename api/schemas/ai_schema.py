"""
Pydantic schemas for the /ai endpoints.

Request schema:  AIChatRequest   — what the caller sends to POST /ai/chat.
Response schema: AIChatResponse  — what POST /ai/chat returns.

Design notes
------------
- session_id is optional on the first turn; the server creates a new session and
  returns the ID.  The client stores it and passes it back on subsequent turns so
  the server can reload the correct conversation history.
- model, provider, base_url, deployment_mode are GUI settings passed by the desktop
  app.  They are only used when creating a new session (first turn); subsequent
  turns use the settings stored in the existing session.
- The API key is never passed over the wire — it is read from .env on the server.
- pending_submit carries the proposed RunConfig + reviewer verdict when the AI
  wants to submit a run.  The desktop UI must show this to the actuary and call
  POST /runs only after explicit approval.  The API never auto-submits.
- tool_calls is included for audit logging.  The UI may ignore it.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AIChatRequest(BaseModel):
    """Body for POST /ai/chat."""

    message: str = Field(
        ...,
        description="The actuary's message as plain text.",
    )
    session_id: Optional[str] = Field(
        None,
        description=(
            "Session ID returned by a previous POST /ai/chat call. "
            "Omit to start a new conversation session."
        ),
    )
    context_run_id: Optional[str] = Field(
        None,
        description=(
            "UUID of the run currently selected in the UI. "
            "Prepended to each message so agents know which run to reference by default."
        ),
    )

    # --- GUI-controlled settings (used only when creating a new session) ---

    provider: str = Field(
        "anthropic",
        description="AI provider: 'anthropic' or 'openai_compatible'. Set in the GUI settings panel.",
    )
    model: str = Field(
        "claude-opus-4-6",
        description="Model ID accepted by the provider. Set in the GUI settings panel.",
    )
    base_url: Optional[str] = Field(
        None,
        description=(
            "Base URL for openai_compatible provider "
            "(e.g. 'https://your-llm.internal/v1'). Set in the GUI settings panel."
        ),
    )
    deployment_mode: str = Field(
        "development",
        description=(
            "'development' allows external APIs (Anthropic). "
            "'production' requires an on-premise openai_compatible endpoint. "
            "Set in the GUI settings panel."
        ),
    )


class ReviewerVerdictSchema(BaseModel):
    """Reviewer verdict embedded inside PendingSubmitSchema."""

    verdict:     str       = Field(..., description="'approved', 'needs_revision', or 'rejected'.")
    summary:     str       = Field(..., description="One-sentence summary of the finding.")
    issues:      list[str] = Field(default_factory=list, description="List of consistency issues found.")
    suggestions: list[str] = Field(default_factory=list, description="Optional suggestions for improvement.")


class PendingSubmitSchema(BaseModel):
    """
    Proposed run configuration awaiting actuary approval.

    The desktop UI must display config_json and the reviewer verdict in a
    confirmation dialog.  submit_run (POST /runs) must only be called after
    the actuary explicitly approves.  See DECISIONS.md §30.
    """

    config_json: str                   = Field(..., description="Proposed RunConfig serialised as a JSON string.")
    reviewer:    ReviewerVerdictSchema = Field(..., description="ReviewerAgent verdict for this proposal.")


class AIChatResponse(BaseModel):
    """Response from POST /ai/chat."""

    reply: str = Field(
        ...,
        description="Final text to display in the chat window.",
    )
    session_id: str = Field(
        ...,
        description="Session ID — store this and pass it back on the next turn to maintain conversation history.",
    )
    agent_used: str = Field(
        ...,
        description="Which agent produced the reply: 'analyst', 'advisor', or 'orchestrator'.",
    )
    pending_submit: Optional[PendingSubmitSchema] = Field(
        None,
        description=(
            "Set when the AI proposes a new run configuration. "
            "Show a confirmation dialog — never call POST /runs without actuary approval."
        ),
    )
    reviewer_verdict: Optional[dict[str, Any]] = Field(
        None,
        description="Full reviewer verdict if a config was reviewed this turn.",
    )
    tool_calls: list[dict[str, Any]] = Field(
        default_factory=list,
        description="All tool calls made this turn — for audit logging.",
    )
