from typing import Any

import chromadb

from verity.rag.embeddings import EmbeddingModel


class ChromaStore:
    def __init__(
        self,
        collection_name: str = "verity_evidence",
        persist_directory: str = ".verity_chroma",
        model: EmbeddingModel | None = None,
    ):
        self.model = model or EmbeddingModel()

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "cosine",
            },
        )

    def add_documents(
        self,
        documents: list[str],
        ids: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:

        embeddings = self.model.embed_texts(documents)

        self.collection.upsert(
            documents=documents,
            embeddings=embeddings.tolist(),
            ids=ids,
            metadatas=metadatas,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        if self.collection.count() == 0:
            return []

        query_embedding = self.model.embed_text(query)

        top_k = min(top_k, self.collection.count())

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
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