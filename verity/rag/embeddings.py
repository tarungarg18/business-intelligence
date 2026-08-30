"""Text embeddings for semantic retrieval, with an offline fallback.

Sentence-Transformers (``all-MiniLM-L6-v2``) is used when it is installed. It is
an optional dependency, however: it pulls in a large PyTorch stack, and the
project's contract is that the demo must run on a laptop with nothing beyond the
core requirements. When the package is absent, this module degrades to a
deterministic hashing vectoriser built on scikit-learn — genuinely weaker on
paraphrase, but real distributional similarity that keeps the pipeline offline
and reproducible instead of crashing at import time.
"""

from __future__ import annotations

import numpy as np

try:  # optional heavy dependency
    from sentence_transformers import SentenceTransformer

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure means "use the fallback"
    SentenceTransformer = None  # type: ignore[assignment]
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

from sklearn.feature_extraction.text import HashingVectorizer

FALLBACK_DIMENSION = 512


class EmbeddingModel:
    """Convert text into embeddings and compare semantic similarity.

    Prefers a sentence-transformer; falls back to a stateless hashing vectoriser
    when that package is unavailable so the rest of the system never has to know
    which backend answered.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = FALLBACK_DIMENSION):
        self.model_name = model_name
        self._encoder = None
        self._fallback: HashingVectorizer | None = None

        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self._encoder = SentenceTransformer(model_name)
            except Exception:  # noqa: BLE001 - offline / download failure -> fallback
                self._encoder = None

        if self._encoder is None:
            self.backend = "hashing-fallback"
            # Character n-grams rather than words: they capture morphological
            # overlap ("product"/"products", "unavailable"/"unavailability")
            # that word tokens miss, which is what makes the offline floor rank
            # semantically related text sensibly without a neural model.
            self._fallback = HashingVectorizer(
                n_features=max(dimension, 4096),
                alternate_sign=False,
                norm="l2",
                analyzer="char_wb",
                ngram_range=(3, 5),
            )
        else:
            self.backend = f"sentence-transformers/{model_name}"

    def embed_text(self, text: str) -> np.ndarray:
        """Convert one piece of text into an embedding vector."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Convert multiple pieces of text into embedding vectors."""
        texts = list(texts)
        if self._encoder is not None:
            return np.asarray(
                self._encoder.encode(texts, convert_to_numpy=True), dtype=np.float32
            )
        matrix = self._fallback.transform(texts)  # type: ignore[union-attr]
        return matrix.toarray().astype(np.float32)

    def similarity(self, text1: str, text2: str) -> float:
        """Return semantic similarity between two texts (higher = closer)."""
        embedding1 = self.embed_text(text1)
        embedding2 = self.embed_text(text2)

        denominator = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        if denominator == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / denominator)
