"""Persistent vector store over ChromaDB, optional by design.

ChromaDB is an optional dependency. When it is not installed the store becomes a
no-op: ``add_documents`` does nothing and ``search`` returns an empty list, so
the Evidence Engine transparently falls back to its in-process semantic
retriever. This keeps the offline demo working without provisioning a vector
database while still using Chroma when it is available.
"""

from __future__ import annotations

from typing import Any

try:  # optional dependency
    import chromadb

    _CHROMADB_AVAILABLE = True
except Exception:  # noqa: BLE001 - any import failure disables the store
    chromadb = None  # type: ignore[assignment]
    _CHROMADB_AVAILABLE = False

from verity.rag.embeddings import EmbeddingModel


class ChromaStore:
    def __init__(
        self,
        collection_name: str = "verity_evidence",
        persist_directory: str = ".verity_chroma",
        model: EmbeddingModel | None = None,
    ):
        self.available = _CHROMADB_AVAILABLE
        if not self.available:
            self.model = None
            self.client = None
            self.collection = None
            return

        self.model = model or EmbeddingModel()
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        if not self.available or not documents:
            return
        embeddings = self.model.embed_texts(documents)
        self.collection.upsert(
            documents=documents,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas,
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.available or self.collection.count() == 0:
            return []

        query_embedding = self.model.embed_text(query)
        top_k = min(top_k, self.collection.count())

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for index, document_id in enumerate(results["ids"][0]):
            output.append(
                {
                    "id": document_id,
                    "text": results["documents"][0][index],
                    "metadata": results["metadatas"][0][index],
                    "distance": results["distances"][0][index],
                }
            )
        return output
