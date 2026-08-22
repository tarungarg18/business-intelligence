"""Offline providers: no network, no API key, no model download.

These exist so the whole system — tests, CI, and a demo on conference wifi —
can run with nothing installed beyond scikit-learn.

The embedder is **LSA** (TF-IDF followed by truncated SVD), and it is labelled
as such everywhere rather than dressed up as a neural embedding. LSA is a real
distributional method: it places documents in a dense space where cosine
similarity reflects shared latent topics. It is genuinely weaker than a modern
sentence encoder on paraphrase, and the honest framing is that it is the floor
the system degrades to, not the target.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from verity.llm.base import (
    EMBED_DOCUMENT,
    EmbeddingResult,
    GenerationResult,
    ProviderError,
    Usage,
    approx_tokens,
    l2_normalize,
)

DEFAULT_DIMENSION = 256
RANDOM_STATE = 20260822


class LsaEmbedder:
    """TF-IDF + truncated SVD. Deterministic, offline, corpus-fitted.

    Unlike an API embedder this must be fitted before use, because the latent
    space is defined by the corpus. :meth:`fit` is called once during index
    build; queries are then projected into that fitted space.
    """

    name = "offline"

    def __init__(self, dimension: int = DEFAULT_DIMENSION) -> None:
        self.model = f"lsa-tfidf-{dimension}d"
        self.dimension = dimension
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None

    @property
    def is_fitted(self) -> bool:
        return self._vectorizer is not None and self._svd is not None

    def fit(self, corpus: Sequence[str]) -> "LsaEmbedder":
        if not corpus:
            raise ProviderError(self.name, "cannot fit on an empty corpus", retryable=False)

        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        matrix = self._vectorizer.fit_transform(corpus)

        # SVD cannot produce more components than the smaller matrix dimension.
        # Small corpora legitimately yield a lower-dimensional space; report the
        # real dimension rather than zero-padding to look bigger.
        components = min(self.dimension, matrix.shape[0] - 1, matrix.shape[1] - 1)
        components = max(components, 2)
        self._svd = TruncatedSVD(n_components=components, random_state=RANDOM_STATE)
        self._svd.fit(matrix)
        self.dimension = components
        self.model = f"lsa-tfidf-{components}d"
        return self

    def embed(self, texts: Sequence[str], kind: str = EMBED_DOCUMENT) -> EmbeddingResult:
        if not self.is_fitted:
            raise ProviderError(
                self.name,
                "embedder must be fitted on the corpus before use; call fit()",
                retryable=False,
            )
        started = time.perf_counter()
        matrix = self._vectorizer.transform(list(texts))  # type: ignore[union-attr]
        vectors = l2_normalize(self._svd.transform(matrix))  # type: ignore[union-attr]
        elapsed = (time.perf_counter() - started) * 1000

        return EmbeddingResult(
            vectors=vectors,
            usage=Usage(
                provider=self.name,
                model=self.model,
                input_tokens=sum(approx_tokens(t) for t in texts),
                latency_ms=elapsed,
                estimated_cost_inr=0.0,  # local compute
            ),
        )

    @property
    def explained_variance(self) -> float:
        if self._svd is None:
            return 0.0
        return float(self._svd.explained_variance_ratio_.sum())


class TemplateGenerator:
    """Deterministic stand-in for a language model.

    Emits a structured summary of the Evidence Pack it is given instead of
    prose. Tests assert on narrative *structure* and on the citation guardrail;
    neither needs a real model, and using one would make tests slow, costly and
    non-deterministic.
    """

    name = "offline"
    model = "template-v1"

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> GenerationResult:
        started = time.perf_counter()
        # Echo any evidence IDs present so the citation validator has something
        # real to check, rather than passing trivially on empty output.
        import re

        ids = re.findall(r"\b[EP]\d{3,4}\b", prompt)
        cited = ", ".join(dict.fromkeys(ids)) or "no evidence supplied"
        text = (
            "[offline template generator - no language model was called]\n"
            f"Evidence considered: {cited}."
        )
        elapsed = (time.perf_counter() - started) * 1000
        return GenerationResult(
            text=text,
            usage=Usage(
                provider=self.name,
                model=self.model,
                input_tokens=approx_tokens(prompt),
                output_tokens=approx_tokens(text),
                latency_ms=elapsed,
                estimated_cost_inr=0.0,
            ),
        )
