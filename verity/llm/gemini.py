"""Google Gemini provider over plain REST.

Deliberately no SDK. The two endpoints needed here are simple, and avoiding
``google-genai`` keeps the dependency footprint at zero — which matters on a
machine where a 2 GB torch install was the reason we are here at all.
"""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

import httpx
import numpy as np

from verity.llm.base import (
    EMBED_DOCUMENT,
    EMBED_QUERY,
    EmbeddingResult,
    GenerationResult,
    ProviderError,
    Usage,
    approx_tokens,
    estimate_cost_inr,
    l2_normalize,
)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

DEFAULT_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
DEFAULT_DIMENSION = 768

# Gemini's asymmetric embedding task types. Telling the model which side of the
# pair it is embedding measurably improves retrieval over a single shared mode.
TASK_TYPES = {
    EMBED_DOCUMENT: "RETRIEVAL_DOCUMENT",
    EMBED_QUERY: "RETRIEVAL_QUERY",
}

TIMEOUT_SECONDS = 45.0


def _api_key(explicit: str | None = None) -> str:
    key = explicit or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ProviderError(
            "gemini",
            "no API key found; set GEMINI_API_KEY in .env",
            retryable=False,
        )
    return key


class GeminiEmbedder:
    name = "gemini"

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        api_key: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self._key = api_key

    def embed(self, texts: Sequence[str], kind: str = EMBED_DOCUMENT) -> EmbeddingResult:
        texts = list(texts)
        if not texts:
            raise ProviderError(self.name, "no texts supplied", retryable=False)

        key = _api_key(self._key)
        task_type = TASK_TYPES.get(kind, TASK_TYPES[EMBED_DOCUMENT])
        started = time.perf_counter()
        vectors: list[list[float]] = []

        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                payload = {
                    "requests": [
                        {
                            "model": f"models/{self.model}",
                            "content": {"parts": [{"text": text}]},
                            "taskType": task_type,
                            "outputDimensionality": self.dimension,
                        }
                        for text in batch
                    ]
                }
                data = _post(
                    client,
                    f"{API_ROOT}/models/{self.model}:batchEmbedContents",
                    key,
                    payload,
                    self.name,
                )
                embeddings = data.get("embeddings") or []
                if len(embeddings) != len(batch):
                    raise ProviderError(
                        self.name,
                        f"expected {len(batch)} embeddings, received {len(embeddings)}",
                    )
                vectors.extend(e.get("values", []) for e in embeddings)

        elapsed = (time.perf_counter() - started) * 1000
        matrix = l2_normalize(np.asarray(vectors, dtype=np.float32))
        tokens = sum(approx_tokens(t) for t in texts)

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


class GeminiGenerator:
    name = "gemini"

    def __init__(self, model: str = DEFAULT_CHAT_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self._key = api_key

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> GenerationResult:
        key = _api_key(self._key)
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        started = time.perf_counter()
        with httpx.Client(timeout=TIMEOUT_SECONDS) as client:
            data = _post(
                client,
                f"{API_ROOT}/models/{self.model}:generateContent",
                key,
                payload,
                self.name,
            )
        elapsed = (time.perf_counter() - started) * 1000

        candidates = data.get("candidates") or []
        if not candidates:
            feedback = data.get("promptFeedback", {})
            raise ProviderError(
                self.name, f"no candidates returned (promptFeedback={feedback})"
            )
        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(part.get("text", "") for part in parts)

        usage_meta = data.get("usageMetadata") or {}
        input_tokens = int(usage_meta.get("promptTokenCount") or approx_tokens(prompt))
        output_tokens = int(usage_meta.get("candidatesTokenCount") or approx_tokens(text))

        return GenerationResult(
            text=text,
            finish_reason=candidate.get("finishReason", "STOP"),
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


def _post(
    client: httpx.Client, url: str, key: str, payload: dict, provider: str
) -> dict:
    try:
        response = client.post(url, headers={"x-goog-api-key": key}, json=payload)
    except httpx.HTTPError as exc:
        raise ProviderError(provider, f"transport error: {exc}") from exc

    if response.status_code >= 400:
        # 4xx other than rate-limiting is a request problem: retrying the same
        # call against the same provider will fail identically, so do not let
        # the fallback chain waste a second provider on it.
        retryable = response.status_code == 429 or response.status_code >= 500
        detail = response.text[:300]
        raise ProviderError(
            provider, f"HTTP {response.status_code}: {detail}", retryable=retryable
        )
    return response.json()
