"""Tests for the openai-compatible embedding provider.

Covers:
- Provider is registered and resolvable by name
- OpenAIEmbeddingFunction accepts no API key when base_url is set (no-key fallback)
- openai-compatible factory raises a clear error when model_name is missing
- openai provider behaviour is unchanged (still requires API key without base_url)
- Environment variable wiring (OPENAI_COMPAT_* vars)
"""

import pytest

pytest.importorskip("chromadb")

from unittest.mock import MagicMock, patch

from zotero_mcp.embeddings.registry import PROVIDERS, merge_env_config, resolve_provider

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_openai_compatible_is_registered():
    assert "openai-compatible" in PROVIDERS
    assert PROVIDERS["openai-compatible"].name == "openai-compatible"


def test_openai_compatible_resolves_to_itself():
    spec, defaults, overrides = resolve_provider("openai-compatible")
    assert spec is PROVIDERS["openai-compatible"]
    assert defaults == {}
    assert overrides == {}


def test_openai_compatible_does_not_require_api_key():
    assert PROVIDERS["openai-compatible"].env.requires_api_key is False


def test_openai_compatible_has_no_default_model():
    """No sensible default: users must specify the model for self-hosted backends."""
    assert PROVIDERS["openai-compatible"].default_model is None


# ---------------------------------------------------------------------------
# OpenAIEmbeddingFunction: no-key fallback
# ---------------------------------------------------------------------------


def _make_ef(api_key=None, base_url=None):
    """Construct an OpenAIEmbeddingFunction with a mocked openai client."""
    from zotero_mcp.embeddings.providers.openai import OpenAIEmbeddingFunction

    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        ef = OpenAIEmbeddingFunction(
            model_name="test-model",
            api_key=api_key,
            base_url=base_url,
        )
    return ef


def test_no_key_fallback_when_base_url_is_set(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ef = _make_ef(base_url="http://localhost:8000/v1")
    assert ef.api_key == "no-key"


def test_explicit_key_wins_over_fallback(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ef = _make_ef(api_key="sk-real", base_url="http://localhost:8000/v1")
    assert ef.api_key == "sk-real"


def test_env_key_wins_over_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    ef = _make_ef(base_url="http://localhost:8000/v1")
    assert ef.api_key == "sk-from-env"


def test_raises_without_key_and_without_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    from zotero_mcp.embeddings.providers.openai import OpenAIEmbeddingFunction

    with pytest.raises(ValueError, match="API key is required"):
        OpenAIEmbeddingFunction(model_name="test-model")


# ---------------------------------------------------------------------------
# Factory: clear error when model_name is missing
# ---------------------------------------------------------------------------


def test_factory_raises_without_model_name():
    factory = PROVIDERS["openai-compatible"].ef_factory
    with pytest.raises(ValueError, match="model_name"):
        factory({})


def test_factory_raises_with_empty_model_name():
    factory = PROVIDERS["openai-compatible"].ef_factory
    with pytest.raises(ValueError, match="model_name"):
        factory({"model_name": ""})


def test_factory_builds_with_model_and_base_url(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    factory = PROVIDERS["openai-compatible"].ef_factory
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        ef = factory({"model_name": "intfloat/e5-large", "base_url": "http://localhost:8000/v1"})
    assert ef.model_name == "intfloat/e5-large"
    assert ef.api_key == "no-key"


def test_factory_uses_conservative_default_batch_size(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    factory = PROVIDERS["openai-compatible"].ef_factory
    with patch("openai.OpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        ef = factory({"model_name": "any-model", "base_url": "http://localhost:8000/v1"})
    assert ef.request_batch_size == 32


# ---------------------------------------------------------------------------
# Environment variable wiring
# ---------------------------------------------------------------------------


def test_merge_env_config_reads_compat_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "http://vllm.local/v1")
    monkeypatch.setenv("OPENAI_COMPAT_EMBEDDING_MODEL", "my-model")
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)

    config = merge_env_config("openai-compatible", {})
    assert config["base_url"] == "http://vllm.local/v1"
    assert config["model_name"] == "my-model"


def test_merge_env_config_explicit_config_wins_over_env(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_EMBEDDING_MODEL", "env-model")

    config = merge_env_config("openai-compatible", {"model_name": "explicit-model"})
    assert config["model_name"] == "explicit-model"
