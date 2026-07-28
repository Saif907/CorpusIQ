import logging
from typing import List
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)


class BGEMbeddingEngine:
    """Local 768-dimensional embedding generator using BAAI/bge-base-en-v1.5 (Pure CPU Execution)."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        logger.info(f"Loading local embedding model '{model_name}' on CPU...")
        self.model_name = model_name
        self.embedding_model = TextEmbedding(
            model_name=model_name,
            providers=["CPUExecutionProvider"]
        )
        self.vector_size = 768

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates 768-dim vector embeddings for a list of text strings."""
        if not texts:
            return []
        
        embeddings_generator = self.embedding_model.embed(texts, batch_size=256)
        return [embedding.tolist() for embedding in embeddings_generator]

    def embed_query(self, query: str) -> List[float]:
        """Generates 768-dim vector embedding for a single query string."""
        return self.embed_texts([query])[0]