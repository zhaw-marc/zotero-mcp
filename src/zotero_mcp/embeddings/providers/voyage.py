"""Voyage AI embedding function, backed by the requests library.

No extra SDK dependency: Voyage's REST API is simple enough that requests
(already a hard dependency) covers it cleanly. The `input_type` parameter
("document" for indexing, "query" for search) is the key quality lever —
it shifts the embedding space for asymmetric retrieval and is the primary
reason to use a dedicated Voyage provider instead of the openai-compatible
one pointing at Voyage's OpenAI-compatible endpoint.
"""

import os
from typing import Any

import requests
from chromadb.utils.embedding_functions import register_embedding_function

from zotero_mcp.embeddings.base import RemoteEmbeddingFunction

_VOYAGE_BASE_URL = "https://api.voyageai.com/v1"


@register_embedding_function
class VoyageEmbeddingFunction(RemoteEmbeddingFunction):
    """Voyage AI embedding function for ChromaDB.

    Registered under the name "voyage" so ChromaDB can rebuild it from a
    persisted collection's config.

    Uses `input_type="document"` when indexing and `input_type="query"` at
    search time, matching Voyage's asymmetric retrieval optimisation.

    Recommended models for academic text:
    - ``voyage-3-large``: highest quality (1024 dims, 32k context)
    - ``voyage-3``: balanced quality/cost (1024 dims) — default
    - ``voyage-3-lite``: fastest/cheapest (512 dims)
    - ``voyage-code-3``: optimised for code and technical documents
    """

    # Voyage allows up to 128 inputs per request.
    DEFAULT_REQUEST_BATCH_SIZE = 128
    default_request_batch_size = DEFAULT_REQUEST_BATCH_SIZE

    # Free tier: 1 000 000 TPM. Leave 5 % headroom; Voyage reports its
    # remaining tokens in x-ratelimit-remaining-tokens, so the adaptive
    # limiter corrects upward on paid tiers automatically.
    DEFAULT_TOKENS_PER_MINUTE = 950_000.0
    default_tokens_per_minute = DEFAULT_TOKENS_PER_MINUTE

    # 2 concurrent requests is conservative for the free tier (300 RPM).
    # Users on higher tiers can raise this via embedding_config.max_parallel_requests.
    DEFAULT_MAX_PARALLEL_REQUESTS = 2
    max_parallel_requests_default = DEFAULT_MAX_PARALLEL_REQUESTS

    # voyage-3* models support up to 32 000 tokens.
    max_input_tokens = 32_000

    # embed_query_text goes through the same truncation+prepare path as
    # documents; no separate truncation needed before _prepare_query.
    truncate_queries = False

    def __init__(
        self,
        model_name: str = "voyage-3",
        api_key: str | None = None,
        base_url: str | None = None,
        request_batch_size: int | None = None,
        rate_limit_rps: float | None = None,
        max_parallel_requests: int | None = None,
        max_retries: int | None = None,
        tokens_per_minute: float | None = None,
    ):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Voyage API key is required. Set VOYAGE_API_KEY or pass api_key. Get one at https://www.voyageai.com/."
            )
        self._init_common(
            model_name=model_name,
            base_url=base_url or os.getenv("VOYAGE_BASE_URL") or _VOYAGE_BASE_URL,
            request_batch_size=request_batch_size,
            rate_limit_rps=rate_limit_rps,
            max_parallel_requests=max_parallel_requests,
            max_retries=max_retries,
            tokens_per_minute=tokens_per_minute,
        )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    @staticmethod
    def name() -> str:
        return "voyage"

    def get_config(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            **self._common_config(),
        }

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "VoyageEmbeddingFunction":
        return VoyageEmbeddingFunction(
            model_name=config.get("model_name", "voyage-3"),
            api_key=config.get("api_key"),
            base_url=config.get("base_url"),
            request_batch_size=config.get("request_batch_size"),
            rate_limit_rps=config.get("rate_limit_rps"),
            max_parallel_requests=config.get("max_parallel_requests"),
            max_retries=config.get("max_retries"),
            tokens_per_minute=config.get("tokens_per_minute"),
        )

    def _embed_batch(self, texts: list[str], is_query: bool = False) -> tuple[list[list[float]], Any]:
        """POST to /v1/embeddings with the appropriate input_type."""
        payload = {
            "model": self.model_name,
            "input": texts,
            "input_type": "query" if is_query else "document",
        }
        url = (self.base_url or _VOYAGE_BASE_URL).rstrip("/") + "/embeddings"
        response = self._session.post(url, json=payload, timeout=120)

        if response.status_code == 429:
            exc = requests.HTTPError(response=response)
            exc.status_code = 429  # type: ignore[attr-defined]
            raise exc
        if response.status_code >= 500:
            exc = requests.HTTPError(response=response)
            exc.status_code = response.status_code  # type: ignore[attr-defined]
            raise exc
        response.raise_for_status()

        data = response.json()
        vectors = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        return vectors, response.headers

    def _classify_error(self, exc: Exception) -> tuple[bool, float | None]:
        """Retry 429s and 5xx; extract Retry-After when present."""
        if not isinstance(exc, requests.HTTPError):
            return False, None
        status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
        if status == 429 or (isinstance(status, int) and status >= 500):
            retry_after = self._parse_retry_after(exc)
            return True, retry_after
        return False, None

    @staticmethod
    def _parse_retry_after(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        value = headers.get("retry-after") or headers.get("Retry-After")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
