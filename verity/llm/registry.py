"""Provider chain: Gemini primary, OpenCode fallback, offline floor.

Two behaviours worth stating plainly.

**Fallback is not retry.** A provider that returns 401 or 400 will return it
again; only rate limits, timeouts and 5xx are worth passing down the chain.
Retrying a malformed request against a second provider burns quota to produce
the same failure twice.

**Embedding caches are keyed by provider and model.** Vectors from different
models are not comparable — mixing them silently produces a similarity space
where nothing means anything. The cache key includes the model id so a
provider switch misses cleanly instead of returning incompatible vectors.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np

from verity.llm.base import (
    EMBED_DOCUMENT,
    Embedder,
    EmbeddingResult,
    GenerationResult,
    ProviderError,
    TextGenerator,
    Usage,
)

DEFAULT_CACHE_DIR = Path("data/cache/embeddings")


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Minimal .env loader. Values already in the environment win."""
    path = Path(path)
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


class EmbedderChain:
    """Tries each embedder in order, recording which one answered."""

    name = "chain"

    def __init__(self, providers: Sequence[Embedder]) -> None:
        if not providers:
            raise ValueError("an embedder chain needs at least one provider")
        self._providers = list(providers)

    @property
    def primary(self) -> Embedder:
        return self._providers[0]

    @property
    def model(self) -> str:
        return self.primary.model

    @property
    def dimension(self) -> int:
        return self.primary.dimension

    def embed(self, texts: Sequence[str], kind: str = EMBED_DOCUMENT) -> EmbeddingResult:
        failures: list[str] = []
        for index, provider in enumerate(self._providers):
            try:
                result = provider.embed(texts, kind=kind)
            except ProviderError as exc:
                failures.append(str(exc))
                if not exc.retryable and index == 0:
                    # Configuration problem rather than an outage. Keep going —
                    # a missing key for the primary is exactly when the offline
                    # floor should take over — but remember why.
                    continue
                continue
            except Exception as exc:  # noqa: BLE001 - a provider must not break the chain
                failures.append(f"{provider.name}: unexpected {type(exc).__name__}: {exc}")
                continue

            if index > 0:
                result = EmbeddingResult(
                    vectors=result.vectors,
                    usage=Usage(
                        **{
                            **result.usage.__dict__,
                            "fallback_from": self._providers[0].name,
                        }
                    ),
                )
            return result

        raise ProviderError(
            "chain", "every embedding provider failed: " + " | ".join(failures), retryable=False
        )


class GeneratorChain:
    """Tries each generator in order, recording which one answered."""

    name = "chain"

    def __init__(self, providers: Sequence[TextGenerator]) -> None:
        if not providers:
            raise ValueError("a generator chain needs at least one provider")
        self._providers = list(providers)

    @property
    def model(self) -> str:
        return self._providers[0].model

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> GenerationResult:
        failures: list[str] = []
        for index, provider in enumerate(self._providers):
            try:
                result = provider.generate(
                    prompt, system=system, max_tokens=max_tokens, temperature=temperature
                )
            except ProviderError as exc:
                failures.append(str(exc))
                continue
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{provider.name}: unexpected {type(exc).__name__}: {exc}")
                continue

            if index > 0:
                result = GenerationResult(
                    text=result.text,
                    finish_reason=result.finish_reason,
                    usage=Usage(
                        **{
                            **result.usage.__dict__,
                            "fallback_from": self._providers[0].name,
                        }
                    ),
                )
            return result

        raise ProviderError(
            "chain", "every generation provider failed: " + " | ".join(failures), retryable=False
        )


class CachedEmbedder:
    """Disk cache in front of any embedder.

    The corpus is embedded once and reused thereafter, so a demo runs offline
    even when the primary provider is an API. Cache keys include the provider
    and model id: vectors from different models occupy different spaces and
    must never be mixed.
    """

    name = "cached"

    def __init__(self, inner: Embedder, cache_dir: str | Path = DEFAULT_CACHE_DIR) -> None:
        self._inner = inner
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def _key(self, text: str, kind: str) -> str:
        digest = hashlib.sha256(
            "\x00".join([self._inner.name, self._inner.model, kind, text]).encode("utf-8")
        ).hexdigest()
        return digest

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.npy"

    def embed(self, texts: Sequence[str], kind: str = EMBED_DOCUMENT) -> EmbeddingResult:
        texts = list(texts)
        keys = [self._key(t, kind) for t in texts]

        cached: dict[int, np.ndarray] = {}
        missing: list[int] = []
        for index, key in enumerate(keys):
            path = self._path(key)
            if path.exists():
                cached[index] = np.load(path)
            else:
                missing.append(index)

        usage = Usage(provider=f"{self._inner.name}+cache", model=self.model, cache_hit=True)
        if missing:
            fresh = self._inner.embed([texts[i] for i in missing], kind=kind)
            for slot, index in enumerate(missing):
                vector = fresh.vectors[slot]
                cached[index] = vector
                np.save(self._path(keys[index]), vector)
            usage = Usage(
                **{
                    **fresh.usage.__dict__,
                    "cache_hit": len(missing) < len(texts),
                }
            )

        matrix = np.vstack([cached[i] for i in range(len(texts))]).astype(np.float32)
        return EmbeddingResult(vectors=matrix, usage=usage)

    def clear(self) -> int:
        removed = 0
        for path in self._dir.glob("*.npy"):
            path.unlink()
            removed += 1
        return removed


def build_embedder(
    *, offline_only: bool = False, cache_dir: str | Path = DEFAULT_CACHE_DIR
) -> tuple[Embedder, list[str]]:
    """Assemble the embedding chain from whatever credentials are present.

    Returns the embedder and a human-readable description of the chain, so the
    UI can state which provider is actually answering rather than implying the
    best one always is.
    """
    from verity.llm.gemini import GeminiEmbedder
    from verity.llm.offline import LsaEmbedder
    from verity.llm.openai_compat import (
        KEY_VARS,
        OpenAICompatibleEmbedder,
        supports_embeddings,
    )

    load_dotenv()
    providers: list[Embedder] = []
    described: list[str] = []

    if not offline_only:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            providers.append(GeminiEmbedder())
            described.append("gemini/gemini-embedding-001 (primary)")

        # Only wire the OpenAI-compatible endpoint if it actually serves
        # embeddings. OpenRouter is chat-completions only, and adding it here
        # would guarantee a failed call before every fallback.
        if any(os.getenv(v) for v in KEY_VARS):
            if supports_embeddings():
                providers.append(OpenAICompatibleEmbedder())
                described.append("openai-compatible (fallback)")
            else:
                described.append("openrouter has no embeddings endpoint - skipped")

    if not providers:
        described.append("offline LSA (no embedding provider available)")
        return LsaEmbedder(), described

    return CachedEmbedder(EmbedderChain(providers), cache_dir=cache_dir), described


def build_generator(*, offline_only: bool = False) -> tuple[TextGenerator, list[str]]:
    """Assemble the generation chain from whatever credentials are present."""
    from verity.llm.gemini import GeminiGenerator
    from verity.llm.offline import TemplateGenerator
    from verity.llm.openai_compat import KEY_VARS, OpenAICompatibleGenerator

    load_dotenv()
    providers: list[TextGenerator] = []
    described: list[str] = []

    if not offline_only:
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            providers.append(GeminiGenerator())
            described.append("gemini/gemini-2.5-flash (primary)")
        if any(os.getenv(v) for v in KEY_VARS):
            generator = OpenAICompatibleGenerator()
            providers.append(generator)
            described.append(f"{generator.name}/{generator.model} (fallback)")

    # The template generator always terminates the chain: a demo must degrade
    # to something honest rather than raising when every API is unreachable.
    providers.append(TemplateGenerator())
    described.append("offline template (floor)")
    return GeneratorChain(providers), described
