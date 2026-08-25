import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Handles converting text into numerical embeddings
    and comparing semantic similarity.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str):
        """Convert one piece of text into an embedding vector."""
        return self.model.encode(
            text,
            convert_to_numpy=True
        )

    def embed_texts(self, texts: list[str]):
        """Convert multiple pieces of text into embedding vectors."""
        return self.model.encode(
            texts,
            convert_to_numpy=True
        )

    def similarity(self, text1: str, text2: str) -> float:
        """
        Return semantic similarity between two texts.
        Higher score means more similar meaning.
        """
        embedding1 = self.embed_text(text1)
        embedding2 = self.embed_text(text2)

        denominator = (
            np.linalg.norm(embedding1)
            * np.linalg.norm(embedding2)
        )

        if denominator == 0:
            return 0.0

        return float(
            np.dot(embedding1, embedding2) / denominator
        )