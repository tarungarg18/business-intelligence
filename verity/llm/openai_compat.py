"""OpenAI-compatible provider.

Covers any endpoint speaking the OpenAI wire format — OpenRouter, OpenCode,
Together, Groq, a local llama.cpp or Ollama server. Only ``base_url``, ``model``
and the key change, which is why this single adapter is worth more than three
provider-specific ones.

Configured via environment, checked in this order:

    OPENROUTER_API_KEY / OPENCODE_API_KEY / OPENAI_API_KEY       bearer token
    OPENROUTER_BASE_URL / OPENCODE_BASE_URL / OPENAI_BASE_URL    endpoint root
    OPENROUTER_CHAT_MODEL / OPENCODE_CHAT_MODEL                  generation model
    OPENROUTER_EMBED_MODEL / OPENCODE_EMBED_MODEL                embedding model

Setting ``OPENROUTER_API_KEY`` alone is enough: the base URL defaults to
OpenRouter's when that key is the one present.

**OpenRouter does not serve embeddings** — it is a chat-completions gateway.
:func:`supports_embeddings` reports that, so the registry wires it as a
generation fallback only rather than adding a provider guaranteed to fail on
every embed call.
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
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"
TIMEOUT_SECONDS = 60.0

KEY_VARS = ("OPENROUTER_API_KEY", "OPENCODE_API_KEY", "OPENAI_API_KEY")
BASE_VARS = ("OPENROUTER_BASE_URL", "OPENCODE_BASE_URL", "OPENAI_BASE_URL")
HF_KEY_VARS = ("HF_TOKEN", "HUGGINGFACE_API_KEY", "HUGGINGFACE_TOKEN")
HF_MODEL_VARS = ("HF_CHAT_MODEL", "HUGGINGFACE_CHAT_MODEL")

# Endpoints that only implement chat completions. Listing them keeps the
# registry from wiring an embedding provider that cannot succeed.
CHAT_ONLY_HOSTS = ("openrouter.ai", "router.huggingface.co")


def _resolve(explicit: str | None, *names: str, default: str | None = None) -> str | None:
    if explicit:
        return explicit
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _default_base_for_key() -> str:
    """Infer the endpoint from which key is configured.

    An OpenRouter key against OpenAI's base URL is a guaranteed 401, which is
    exactly the confusing failure this avoids.
    """
    if os.getenv("OPENROUTER_API_KEY"):
        return OPENROUTER_BASE_URL
    return DEFAULT_BASE_URL


def supports_embeddings(base_url: str | None = None) -> bool:
    url = base_url or _resolve(None, *BASE_VARS, default=_default_base_for_key()) or ""
    return not any(host in url for host in CHAT_ONLY_HOSTS)


class OpenAICompatibleClient:
    """Shared transport for the embedder and generator below."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._key = _resolve(api_key, *KEY_VARS)
        self.base_url = (
            _resolve(base_url, *BASE_VARS, default=_default_base_for_key())
            or DEFAULT_BASE_URL
        ).rstrip("/")
        # Name the provider after the endpoint actually in use, so telemetry
        # and audit entries say "openrouter" rather than a stale label.
        if "openrouter.ai" in self.base_url:
            self.name = "openrouter"
        elif "router.huggingface.co" in self.base_url:
            self.name = "huggingface"
        else:
            self.name = "openai-compatible"

    def _require_key(self) -> str:
        if not self._key:
            raise ProviderError(
                self.name,
                f"no API key found; set one of {', '.join(KEY_VARS)} in .env",
                retryable=False,
            )
        return self._key

    def post(self, path: str, payload: dict) -> dict:
        key = self._require_key()
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {"Authorization": f"Bearer {key}"}
        if "openrouter.ai" in self.base_url:
            # OpenRouter attributes traffic by these; optional but polite.
            headers["HTTP-Referer"] = "https://github.com/tarungarg18/business-intelligence"
            headers["X-Title"] = "Verity KPI Engine"
        try:
            with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
                response = client.post(url, headers=headers, json=payload)
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
        self.model = (
            _resolve(
                model,
                "OPENROUTER_EMBED_MODEL",
                "OPENCODE_EMBED_MODEL",
                default="text-embedding-3-small",
            )
            or ""
        )
        self.dimension = dimension
        if not supports_embeddings(self.base_url):
            raise ProviderError(
                self.name,
                f"{self.base_url} serves chat completions only and has no "
                f"embeddings endpoint",
                retryable=False,
            )

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
        default_model = (
            "google/gemma-4-31b-it:free"
            if "openrouter.ai" in self.base_url
            else "gpt-4o-mini"
        )
        self.model = (
            _resolve(
                model,
                "OPENROUTER_CHAT_MODEL",
                "OPENCODE_CHAT_MODEL",
                default=default_model,
            )
            or ""
        )

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


class HuggingFaceGenerator(OpenAICompatibleGenerator):
    """Hugging Face Inference Providers via its OpenAI-compatible chat router."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str = HUGGINGFACE_BASE_URL,
    ) -> None:
        key = _resolve(api_key, *HF_KEY_VARS)
        super().__init__(
            model=_resolve(
                model,
                *HF_MODEL_VARS,
                default="meta-llama/Llama-3.1-8B-Instruct:fastest",
            ),
            api_key=key,
            base_url=base_url,
        )
