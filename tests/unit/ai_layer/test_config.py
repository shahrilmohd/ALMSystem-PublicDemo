"""
Unit tests for AILayerConfig.

Coverage
--------
- Default values
- provider validation (reject unknown values)
- deployment_mode validation
- base_url required when provider=openai_compatible
- assert_production_safe: anthropic + production → RuntimeError
- assert_production_safe: openai_compatible + production → OK
- is_external_api: anthropic → True; private base_url → False; public base_url → True
- get_api_key: reads from env var; raises EnvironmentError when absent
- is_key_available: True/False based on env var
- for_development / for_production convenience constructors
"""
from __future__ import annotations

import os

import pytest

from ai_layer.config import AILayerConfig


class TestDefaults:
    def test_default_provider(self):
        cfg = AILayerConfig()
        assert cfg.provider == "anthropic"

    def test_default_model(self):
        cfg = AILayerConfig()
        assert cfg.model == "claude-opus-4-6"

    def test_default_deployment_mode(self):
        cfg = AILayerConfig()
        assert cfg.deployment_mode == "development"

    def test_default_max_tokens(self):
        cfg = AILayerConfig()
        assert cfg.max_tokens == 4096

    def test_default_temperature(self):
        cfg = AILayerConfig()
        assert cfg.temperature == 0.2


class TestValidation:
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="provider"):
            AILayerConfig(provider="grok")

    def test_unknown_deployment_mode_raises(self):
        with pytest.raises(ValueError, match="deployment_mode"):
            AILayerConfig(deployment_mode="staging")

    def test_openai_compatible_without_base_url_raises(self):
        with pytest.raises(ValueError, match="base_url"):
            AILayerConfig(provider="openai_compatible")

    def test_openai_compatible_with_base_url_ok(self):
        cfg = AILayerConfig(
            provider="openai_compatible",
            base_url="http://localhost:11434/v1",
        )
        assert cfg.provider == "openai_compatible"


class TestProductionSafety:
    def test_anthropic_development_is_safe(self):
        cfg = AILayerConfig(provider="anthropic", deployment_mode="development")
        cfg.assert_production_safe()   # must not raise

    def test_anthropic_production_raises(self):
        cfg = AILayerConfig(provider="anthropic", deployment_mode="production")
        with pytest.raises(RuntimeError, match="(?i)production"):
            cfg.assert_production_safe()

    def test_openai_compatible_production_is_safe(self):
        cfg = AILayerConfig(
            provider="openai_compatible",
            base_url="http://10.0.0.5/v1",
            deployment_mode="production",
        )
        cfg.assert_production_safe()   # must not raise


class TestIsExternalApi:
    def test_anthropic_is_external(self):
        assert AILayerConfig(provider="anthropic").is_external_api() is True

    def test_localhost_is_not_external(self):
        cfg = AILayerConfig(provider="openai_compatible", base_url="http://localhost:11434/v1")
        assert cfg.is_external_api() is False

    def test_private_ip_is_not_external(self):
        cfg = AILayerConfig(provider="openai_compatible", base_url="http://192.168.1.10/v1")
        assert cfg.is_external_api() is False

    def test_public_url_is_external(self):
        cfg = AILayerConfig(provider="openai_compatible", base_url="https://api.openai.com/v1")
        assert cfg.is_external_api() is True


class TestGetApiKey:
    def test_reads_key_from_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        cfg = AILayerConfig()
        assert cfg.get_api_key() == "sk-test-123"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = AILayerConfig()
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            cfg.get_api_key()

    def test_custom_key_env_var(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "my-local-key")
        cfg = AILayerConfig(
            provider="openai_compatible",
            base_url="http://localhost/v1",
            api_key_env_var="LLM_API_KEY",
        )
        assert cfg.get_api_key() == "my-local-key"


class TestIsKeyAvailable:
    def test_returns_true_when_set(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        assert AILayerConfig().is_key_available() is True

    def test_returns_false_when_absent(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert AILayerConfig().is_key_available() is False


class TestConvenienceConstructors:
    def test_for_development_sets_anthropic(self):
        cfg = AILayerConfig.for_development()
        assert cfg.provider == "anthropic"
        assert cfg.deployment_mode == "development"

    def test_for_development_custom_model(self):
        cfg = AILayerConfig.for_development(model="claude-haiku-4-5-20251001")
        assert cfg.model == "claude-haiku-4-5-20251001"

    def test_for_production_sets_openai_compatible(self):
        cfg = AILayerConfig.for_production(
            base_url="http://10.0.0.5/v1",
            model="llama3",
        )
        assert cfg.provider == "openai_compatible"
        assert cfg.deployment_mode == "production"
        assert cfg.base_url == "http://10.0.0.5/v1"
        assert cfg.model == "llama3"
