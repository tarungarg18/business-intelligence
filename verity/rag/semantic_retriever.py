from dataclasses import dataclass

import numpy as np

from verity.rag.embeddings import EmbeddingModel


@dataclass
class SemanticResult:
    index: int
    text: str
    score: float


class SemanticRetriever:
    def __init__(self, model: EmbeddingModel | None = None):
        self.model = model or EmbeddingModel()

        # Stores document embeddings in memory
        self.document_cache: dict[str, np.ndarray] = {}

    def _get_document_embeddings(
        self,
        documents: list[str],
    ) -> np.ndarray:

        missing_documents = [
            document
            for document in documents
            if document not in self.document_cache
        ]

        # Generate embeddings only for documents
        # that are not already cached
        if missing_documents:
            new_embeddings = self.model.embed_texts(
                missing_documents
            )

            for document, embedding in zip(
                missing_documents,
                new_embeddings,
            ):
                self.document_cache[document] = embedding

        return np.array([
            self.document_cache[document]
            for document in documents
        ])

    def search(
        self,
        query: str,
        documents: list[str],
        top_k: int = 5,
    ) -> list[SemanticResult]:

        if not documents:
            return []

        # Query embedding is needed once per search
        query_embedding = self.model.embed_text(query)

        # Document embeddings are reused from cache
        document_embeddings = self._get_document_embeddings(
            documents
        )

        results = []

        for index, document_embedding in enumerate(
            document_embeddings
        ):

            denominator = (
                np.linalg.norm(query_embedding)
                * np.linalg.norm(document_embedding)
            )

            if denominator == 0:
                score = 0.0
            else:
                score = float(
                    np.dot(
                        query_embedding,
                        document_embedding,
                    )
                    / denominator
                )

            results.append(
                SemanticResult(
                    index=index,
                    text=documents[index],
                    score=score,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results[:top_k]