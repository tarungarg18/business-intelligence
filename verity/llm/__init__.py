"""Provider-agnostic LLM and embedding layer.

Nothing outside this package imports a provider directly. Chain order is
Gemini -> OpenCode -> offline, assembled from whatever credentials exist.
"""

from verity.llm.base import (
    EMBED_DOCUMENT,
    EMBED_QUERY,
    Embedder,
    EmbeddingResult,
    GenerationResult,
    ProviderError,
    TextGenerator,
    Usage,
    approx_tokens,
    estimate_cost_inr,
    l2_normalize,
)
from verity.llm.offline import LsaEmbedder, TemplateGenerator
from verity.llm.registry import (
    CachedEmbedder,
    EmbedderChain,
    GeneratorChain,
    build_embedder,
    build_generator,
    load_dotenv,
)

__all__ = [
    "EMBED_DOCUMENT",
    "EMBED_QUERY",
    "Embedder",
    "EmbeddingResult",
    "GenerationResult",
    "ProviderError",
    "TextGenerator",
    "Usage",
    "approx_tokens",
    "estimate_cost_inr",
    "l2_normalize",
    "LsaEmbedder",
    "TemplateGenerator",
    "CachedEmbedder",
    "EmbedderChain",
    "GeneratorChain",
    "build_embedder",
    "build_generator",
    "load_dotenv",
]
