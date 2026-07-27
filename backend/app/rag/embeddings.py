import logging 
from typing import List 
from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

class BGEMbeddingEngine:

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        logger.info(f"Loading local embedding model: {model_name}")
        self.model_name = model_name
        self.embedding_model = TextEmbedding(model_name= model_name)
        self.vector_size = 768

    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        embeddings_generator = self.embedding_model.embed(texts)
        return [embedding.tolist() for embedding in embeddings_generator]

    
    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]