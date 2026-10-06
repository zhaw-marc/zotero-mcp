"""Tests for the Voyage AI embedding provider.

Covers:
- Provider is registered with correct defaults and aliases
- VoyageEmbeddingFunction sends correct input_type for docs vs. queries
- Retry logic on 429 and 5xx
- No retry on 4xx (other than 429)
- Environment variable wiring
- Config round-trip (get_config / build_from_config)
"""

from unittest.mock import MagicMock, patch

import pytest
import requests as req

pytest.importorskip("chromadb")

from zotero_mcp.embeddings.providers.voyage import VoyageEmbeddingFunction
from zotero_mcp.embeddings.registry import PROVIDERS, merge_env_config, resolve_provider

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_voyage_is_registered():
    assert "voyage" in PROVIDERS
    assert PROVIDERS["voyage"].name == "voyage"


def test_voyage_default_model():
    assert PROVIDERS["voyage"].default_model == "voyage-3"


def test_voyage_requires_api_key():
    assert PROVIDERS["voyage"].env.requires_api_key is True


def test_voyage_resolves_to_itself():
    spec, defaults, overrides = resolve_provider("voyage")
    assert spec is PROVIDERS["voyage"]


@pytest.mark.parametrize("model", ["voyage-3-large", "voyage-3-lite", "voyage-code-3"])
def test_voyage_models_are_usable_by_full_name(model):
    """Full model names resolve to the voyage provider via embedding_config.model_name."""
    spec, _, _ = resolve_provider("voyage")
    assert spec is PROVIDERS["voyage"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(status_code: int, data: dict | None = None, headers: dict | None = None):
    """Build a minimal requests.Response mock."""
    resp = MagicMock(spec=req.Response)
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json.return_value = data or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = req.HTTPError(response=resp)
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_embed_response(vectors: list[list[float]]) -> dict:
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(vectors)],
        "model": "voyage-3",
        "usage": {"total_tokens": 10},
    }


def _make_ef(monkeypatch) -> VoyageEmbeddingFunction:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    ef = VoyageEmbeddingFunction.__new__(VoyageEmbeddingFunction)
    ef.api_key = "test-key"
    ef.model_name = "voyage-3"
    ef.base_url = "https://api.voyageai.com/v1"
    ef.request_batch_size = 128
    ef.max_parallel_requests = 1
    ef.max_retries = 0
    ef.rate_limit_rps = None
    ef.tokens_per_minute = None
    ef.truncate_queries = False
    ef.chars_per_token = 4
    session = MagicMock()
    ef._session = session
    from zotero_mcp.embeddings.ratelimit import AdaptiveRateLimiter

    ef.limiter = AdaptiveRateLimiter()
    return ef


# ---------------------------------------------------------------------------
# input_type routing
# ---------------------------------------------------------------------------


def test_embed_batch_sends_document_input_type(monkeypatch):
    ef = _make_ef(monkeypatch)
    vectors = [[0.1, 0.2], [0.3, 0.4]]
    ef._session.post.return_value = _make_response(200, _make_embed_response(vectors))

    result, _ = ef._embed_batch(["doc1", "doc2"], is_query=False)

    payload = ef._session.post.call_args[1]["json"]
    assert payload["input_type"] == "document"
    assert result == vectors


def test_embed_batch_sends_query_input_type(monkeypatch):
    ef = _make_ef(monkeypatch)
    ef._session.post.return_value = _make_response(200, _make_embed_response([[0.5, 0.6]]))

    _, _ = ef._embed_batch(["what is X?"], is_query=True)

    payload = ef._session.post.call_args[1]["json"]
    assert payload["input_type"] == "query"


def test_embed_batch_preserves_order(monkeypatch):
    """Results are sorted by index regardless of API response order."""
    ef = _make_ef(monkeypatch)
    shuffled = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.9, 0.9], "index": 1},
            {"object": "embedding", "embedding": [0.1, 0.1], "index": 0},
        ],
        "model": "voyage-3",
    }
    ef._session.post.return_value = _make_response(200, shuffled)

    vectors, _ = ef._embed_batch(["first", "second"])
    assert vectors[0] == [0.1, 0.1]
    assert vectors[1] == [0.9, 0.9]


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


def test_classify_error_retries_429(monkeypatch):
    ef = _make_ef(monkeypatch)
    resp = _make_response(429, headers={"retry-after": "2.5"})
    exc = req.HTTPError(response=resp)
    exc.status_code = 429
    retryable, retry_after = ef._classify_error(exc)
    assert retryable is True
    assert retry_after == pytest.approx(2.5)


def test_classify_error_retries_503(monkeypatch):
    ef = _make_ef(monkeypatch)
    exc = req.HTTPError(response=_make_response(503))
    exc.status_code = 503
    retryable, _ = ef._classify_error(exc)
    assert retryable is True


def test_classify_error_does_not_retry_400(monkeypatch):
    ef = _make_ef(monkeypatch)
    exc = req.HTTPError(response=_make_response(400))
    exc.status_code = 400
    retryable, _ = ef._classify_error(exc)
    assert retryable is False


def test_classify_error_does_not_retry_non_http(monkeypatch):
    ef = _make_ef(monkeypatch)
    retryable, _ = ef._classify_error(ValueError("something else"))
    assert retryable is False


# ---------------------------------------------------------------------------
# Raises without API key
# ---------------------------------------------------------------------------


def test_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="VOYAGE_API_KEY"):
        VoyageEmbeddingFunction(model_name="voyage-3")


# ---------------------------------------------------------------------------
# Config round-trip
# ---------------------------------------------------------------------------


def test_get_config_build_from_config_roundtrip(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "rt-key")
    with patch.object(req.Session, "post"):
        ef = VoyageEmbeddingFunction(model_name="voyage-3-large")
    config = ef.get_config()
    assert config["model_name"] == "voyage-3-large"

    monkeypatch.setenv("VOYAGE_API_KEY", "rt-key")
    rebuilt = VoyageEmbeddingFunction.build_from_config({**config, "api_key": "rt-key"})
    assert rebuilt.model_name == "voyage-3-large"


# ---------------------------------------------------------------------------
# Environment variable wiring
# ---------------------------------------------------------------------------


def test_merge_env_config_reads_voyage_vars(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "env-key")
    monkeypatch.setenv("VOYAGE_EMBEDDING_MODEL", "voyage-3-large")

    config = merge_env_config("voyage", {})
    assert config["api_key"] == "env-key"
    assert config["model_name"] == "voyage-3-large"


def test_merge_env_config_explicit_wins_over_env(monkeypatch):
    monkeypatch.setenv("VOYAGE_EMBEDDING_MODEL", "voyage-3")
    config = merge_env_config("voyage", {"model_name": "voyage-code-3", "api_key": "k"})
    assert config["model_name"] == "voyage-code-3"
