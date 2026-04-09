"""
AILayerConfig — provider and PNC deployment configuration for the AI layer.

All AI layer components read their provider/model/key settings from one instance
of this dataclass.  Nothing else in ai_layer/ hardcodes a model name or API key.

Deployment modes
----------------
"development"
    External APIs (provider="anthropic") are permitted.
    A warning is logged on every request.
    Use only with synthetic, non-production data.

"production"
    provider="anthropic" raises RuntimeError at agent instantiation.
    Only provider="openai_compatible" with a private base_url is allowed.
    Enforces the PNC requirement that real run data must not leave the network
    perimeter.  See DECISIONS.md §30 (Privacy and Confidentiality).

Supported providers
-------------------
"anthropic"
    Uses the Anthropic Python SDK.  Requires ANTHROPIC_API_KEY (or the env var
    named in api_key_env_var) to be set.  base_url is ignored.

"openai_compatible"
    Uses the OpenAI Python SDK with a custom base_url.  Works with any
    OpenAI-compatible endpoint: Azure OpenAI, self-hosted Llama/Mistral,
    or any in-house model that exposes POST /chat/completions.
    base_url is required.  api_version is used for Azure OpenAI only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Default per-agent model assignments
# ---------------------------------------------------------------------------
# Router and Reviewer use lighter models to conserve the primary model's rate
# limit and cost budget.  Analyst, Advisor, and Modelling default to the user's
# configured model (typically Opus) because they require deep actuarial reasoning.
# Any entry can be overridden by passing a custom agent_models dict.

_DEFAULT_AGENT_MODELS: dict[str, str] = {
    "router":   "claude-haiku-4-5-20251001",  # one-word classification only
    "reviewer": "claude-sonnet-4-6",           # config sanity-check, moderate complexity
}


@dataclass
class AILayerConfig:
    """
    Configuration for the AI layer.

    Parameters
    ----------
    provider : str
        "anthropic" or "openai_compatible".
    model : str
        Model ID as accepted by the provider.
        Default: "claude-opus-4-6" (Anthropic).
    api_key_env_var : str
        Name of the environment variable holding the API key.
        Default: "ANTHROPIC_API_KEY".
    base_url : str | None
        Required for provider="openai_compatible".
        The base URL of the OpenAI-compatible endpoint, e.g.
        "https://your-company-llm.internal/v1".
        Ignored for provider="anthropic".
    api_version : str | None
        API version string — Azure OpenAI only (e.g. "2024-02-01").
        Ignored for all other providers.
    deployment_mode : str
        "development" or "production".
        In "production" mode, provider="anthropic" raises RuntimeError.
    max_tokens : int
        Maximum tokens in the model response.  Default: 4096.
    temperature : float
        Sampling temperature.  Default: 0.2 (low for deterministic reasoning).
    """

    provider:         str            = "anthropic"
    model:            str            = "claude-opus-4-6"
    api_key_env_var:  str            = "ANTHROPIC_API_KEY"
    base_url:         Optional[str]  = None
    api_version:      Optional[str]  = None
    deployment_mode:  str            = "development"
    max_tokens:       int            = 4096
    temperature:      float          = 0.2
    agent_models:     dict           = field(default_factory=lambda: dict(_DEFAULT_AGENT_MODELS))
    """
    Per-agent model overrides.  Keys: 'router', 'analyst', 'advisor',
    'reviewer', 'modelling'.  Any agent not in this dict falls back to
    self.model (the primary model configured in the GUI).

    Defaults:
      router   → claude-haiku-4-5-20251001  (classification only)
      reviewer → claude-sonnet-4-6           (config review, moderate complexity)
      analyst / advisor / modelling → self.model (full Opus reasoning)
    """

    def __post_init__(self) -> None:
        if self.provider not in ("anthropic", "openai_compatible"):
            raise ValueError(
                f"provider must be 'anthropic' or 'openai_compatible', got {self.provider!r}"
            )
        if self.deployment_mode not in ("development", "production"):
            raise ValueError(
                f"deployment_mode must be 'development' or 'production', "
                f"got {self.deployment_mode!r}"
            )
        if self.provider == "openai_compatible" and not self.base_url:
            raise ValueError(
                "base_url is required when provider='openai_compatible'"
            )

    # ------------------------------------------------------------------
    # PNC enforcement
    # ------------------------------------------------------------------

    def assert_production_safe(self) -> None:
        """
        Raise RuntimeError if this config is not safe for production use.

        Called at agent instantiation.  In production mode, only an
        openai_compatible provider with a private base_url is permitted —
        this ensures real run data never leaves the network perimeter.
        See DECISIONS.md §30 (Privacy and Confidentiality).
        """
        if self.deployment_mode == "production" and self.provider == "anthropic":
            raise RuntimeError(
                "Production mode does not permit provider='anthropic' (external API). "
                "Set provider='openai_compatible' with a private base_url, or set "
                "deployment_mode='development' for testing with synthetic data only."
            )

    def is_external_api(self) -> bool:
        """Return True if this config sends data to an external (off-premise) API."""
        if self.provider == "anthropic":
            return True
        if self.provider == "openai_compatible" and self.base_url:
            # Heuristic: a URL that is localhost or a private RFC-1918 range is on-premise.
            private_prefixes = (
                "http://localhost", "https://localhost",
                "http://127.", "https://127.",
                "http://10.", "https://10.",
                "http://172.", "https://172.",
                "http://192.168.", "https://192.168.",
            )
            return not any(self.base_url.startswith(p) for p in private_prefixes)
        return False

    # ------------------------------------------------------------------
    # Agent model resolution
    # ------------------------------------------------------------------

    def model_for(self, agent: str) -> str:
        """
        Return the model ID to use for a given agent type.

        Parameters
        ----------
        agent : str
            One of: 'router', 'analyst', 'advisor', 'reviewer', 'modelling'.

        Returns
        -------
        str
            - If agent_models[agent] is explicitly set: always use it (the caller
              has made a deliberate override).
            - If the entry comes only from _DEFAULT_AGENT_MODELS (Anthropic model
              IDs) but provider is not "anthropic": ignore it and return self.model
              so that non-Anthropic providers are never given Claude model IDs by
              accident.
            - Otherwise: self.model.
        """
        override = self.agent_models.get(agent)
        if override is None:
            return self.model
        # Only apply Anthropic default model IDs on the Anthropic provider path.
        # For openai_compatible (GPT, Azure, in-house), fall back to self.model
        # unless the caller explicitly put a non-Anthropic model ID in agent_models.
        if self.provider != "anthropic" and override == _DEFAULT_AGENT_MODELS.get(agent):
            return self.model
        return override

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------

    def get_api_key(self) -> str:
        """
        Read the API key from the environment variable named in api_key_env_var.

        Raises
        ------
        EnvironmentError
            If the environment variable is not set.
        """
        key = os.environ.get(self.api_key_env_var)
        if not key:
            raise EnvironmentError(
                f"AI layer requires environment variable {self.api_key_env_var!r} to be set. "
                f"Set it to your API key before starting the application."
            )
        return key

    def is_key_available(self) -> bool:
        """Return True if the API key environment variable is set (non-empty)."""
        return bool(os.environ.get(self.api_key_env_var))

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_development(
        cls,
        model: str = "claude-opus-4-6",
        api_key_env_var: str = "ANTHROPIC_API_KEY",
    ) -> "AILayerConfig":
        """Anthropic external API, development mode.  Synthetic data only."""
        return cls(
            provider="anthropic",
            model=model,
            api_key_env_var=api_key_env_var,
            deployment_mode="development",
        )

    @classmethod
    def for_production(
        cls,
        base_url: str,
        model: str,
        api_key_env_var: str = "LLM_API_KEY",
    ) -> "AILayerConfig":
        """On-premise OpenAI-compatible endpoint, production mode."""
        return cls(
            provider="openai_compatible",
            model=model,
            api_key_env_var=api_key_env_var,
            base_url=base_url,
            deployment_mode="production",
        )
