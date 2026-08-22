"""OpenAI-compatible provider.

Covers any endpoint speaking the OpenAI wire format — OpenCode, OpenRouter,
Together, Groq, a local llama.cpp or Ollama server. Only ``base_url``, ``model``
and the key change, which is why this single adapter is worth more than three
provider-specific ones.

Configured via environment:

    OPENCODE_API_KEY / OPENAI_API_KEY   the bearer token
    OPENCODE_BASE_URL / OPENAI_BASE_URL endpoint root, default OpenAI's
    OPENCODE_CHAT_MODEL                 model id for generation
    OPENCODE_EMBED_MODEL                model id for embeddings
"""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

import httpx
import numpy as np

from verity.llm.base import (
    EMBED_DOCUMENT,
    EmbeddingResult,
    GenerationResult,
    ProviderError,
    Usage,
    approx_tokens,
    estimate_cost_inr,
    l2_normalize,
)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
TIMEOUT_SECONDS = 60.0


def _resolve(explicit: str | None, *names: str, default: str | None = None) -> str | None:
    if explicit:
        return explicit
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


class OpenAICompatibleClient:
    """Shared transport for the embedder and generator below."""

    name = "opencode"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._key = _resolve(api_key, "OPENCODE_API_KEY", "OPENAI_API_KEY")
        self.base_url = (
            _resolve(base_url, "OPENCODE_BASE_URL", "OPENAI_BASE_URL", default=DEFAULT_BASE_URL)
            or DEFAULT_BASE_URL
        ).rstrip("/")

    def _require_key(self) -> str:
        if not self._key:
            raise ProviderError(
                self.name,
                "no API key found; set OPENCODE_API_KEY in .env",
                retryable=False,
            )
        return self._key

    def post(self, path: str, payload: dict) -> dict:
        key = self._require_key()
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
                response = client.post(
                    url, headers={"Authorization": f"Bearer {key}"}, json=payload
                )
        except httpx.HTTPError as exc:
            raise ProviderError(self.name, f"transport error: {exc}") from exc

        if response.status_code >= 400:
            retryable = response.status_code == 429 or response.status_code >= 500
            raise ProviderError(
                self.name,
                f"HTTP {response.status_code}: {response.text[:300]}",
                retryable=retryable,
            )
        return response.json()


class OpenAICompatibleEmbedder(OpenAICompatibleClient):
    def __init__(
        self,
        model: str | None = None,
        dimension: int = 768,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url)
        self.model = _resolve(model, "OPENCODE_EMBED_MODEL", default="text-embedding-3-small") or ""
        self.dimension = dimension

    def embed(self, texts: Sequence[str], kind: str = EMBED_DOCUMENT) -> EmbeddingResult:
        texts = list(texts)
        if not texts:
            raise ProviderError(self.name, "no texts supplied", retryable=False)

        started = time.perf_counter()
        data = self.post("embeddings", {"model": self.model, "input": texts})
        elapsed = (time.perf_counter() - started) * 1000

        items = sorted(data.get("data") or [], key=lambda d: d.get("index", 0))
        if len(items) != len(texts):
            raise ProviderError(
                self.name, f"expected {len(texts)} embeddings, received {len(items)}"
            )
        matrix = l2_normalize(
            np.asarray([item["embedding"] for item in items], dtype=np.float32)
        )

        usage = data.get("usage") or {}
        tokens = int(usage.get("prompt_tokens") or sum(approx_tokens(t) for t in texts))
        return EmbeddingResult(
            vectors=matrix,
            usage=Usage(
                provider=self.name,
                model=self.model,
                input_tokens=tokens,
                latency_ms=elapsed,
                estimated_cost_inr=estimate_cost_inr(self.model, tokens, 0),
            ),
        )


class OpenAICompatibleGenerator(OpenAICompatibleClient):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url)
        self.model = _resolve(model, "OPENCODE_CHAT_MODEL", default="gpt-4o-mini") or ""

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> GenerationResult:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        data = self.post(
            "chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        elapsed = (time.perf_counter() - started) * 1000

        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(self.name, "no choices returned")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""

        usage = data.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or approx_tokens(prompt))
        output_tokens = int(usage.get("completion_tokens") or approx_tokens(text))

        return GenerationResult(
            text=text,
            finish_reason=choices[0].get("finish_reason", "stop"),
            usage=Usage(
                provider=self.name,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=elapsed,
                estimated_cost_inr=estimate_cost_inr(
                    self.model, input_tokens, output_tokens
                ),
            ),
        )
