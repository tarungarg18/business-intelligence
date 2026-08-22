"""Provider-agnostic interfaces for embeddings and text generation.

Two rules this package exists to enforce:

  1. **Nothing in the system imports a provider directly.** Swapping Gemini for
     an OpenAI-compatible endpoint, or for the offline embedder, is a config
     change and nothing else.

  2. **Every call is metered.** Each result carries provider, model, tokens,
     latency and estimated cost, so the Cost Governor reports measurements
     rather than estimates. A call that is not metered cannot be governed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

# Task hint for providers that embed queries and documents differently.
# Asymmetric models score noticeably better when told which side they are on.
EMBED_DOCUMENT = "document"
EMBED_QUERY = "query"


@dataclass(frozen=True)
class Usage:
    """Telemetry for a single provider call."""

    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_inr: float = 0.0
    cache_hit: bool = False
    fallback_from: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def render(self) -> str:
        bits = [
            f"{self.provider}/{self.model}",
            f"{self.total_tokens} tok",
            f"{self.latency_ms:.0f} ms",
            f"Rs {self.estimated_cost_inr:.4f}",
        ]
        if self.cache_hit:
            bits.append("cached")
        if self.fallback_from:
            bits.append(f"fell back from {self.fallback_from}")
        return " | ".join(bits)


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray
    usage: Usage

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 else 0


@dataclass(frozen=True)
class GenerationResult:
    text: str
    usage: Usage
    finish_reason: str = "stop"


class ProviderError(RuntimeError):
    """Raised when a provider call fails and the chain should try the next one."""

    def __init__(self, provider: str, message: str, retryable: bool = True) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.retryable = retryable


@runtime_checkable
class Embedder(Protocol):
    name: str
    model: str
    dimension: int

    def embed(self, texts: Sequence[str], kind: str = EMBED_DOCUMENT) -> EmbeddingResult:
        ...


@runtime_checkable
class TextGenerator(Protocol):
    name: str
    model: str

    def generate(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> GenerationResult:
        ...


# --- cost model -----------------------------------------------------------

# Indicative rates in INR per million tokens. Deliberately a lookup table
# rather than hardcoded arithmetic: rates change, and the Cost Governor panel
# must be able to show what it charged against.
USD_TO_INR = 84.0

RATE_CARD: dict[str, tuple[float, float]] = {
    # model-substring: (input USD per 1M, output USD per 1M)
    "gemini-embedding": (0.15, 0.0),
    "text-embedding": (0.15, 0.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}

DEFAULT_RATE = (0.20, 0.80)


def estimate_cost_inr(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate call cost in INR.

    Approximate by design — the point is a comparable number across routing
    tiers, not an invoice. The Cost Governor reports it as an estimate.
    """
    rate = DEFAULT_RATE
    for key, value in RATE_CARD.items():
        if key in model:
            rate = value
            break
    usd = (input_tokens / 1e6) * rate[0] + (output_tokens / 1e6) * rate[1]
    return usd * USD_TO_INR


def approx_tokens(text: str) -> int:
    """Rough token count for providers that do not report usage.

    Roughly four characters per token for English prose. Used only when a
    provider returns no usage block, and flagged as approximate wherever shown.
    """
    return max(1, len(text) // 4)


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Unit-normalise rows so cosine similarity reduces to a dot product."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
