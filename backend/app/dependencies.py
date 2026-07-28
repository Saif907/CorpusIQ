from fastapi import Request
from app.rag.pipeline import RAGPipeline


def get_rag_pipeline(request: Request) -> RAGPipeline:
    """FastAPI dependency that retrieves the RAGPipeline singleton from app state.

    The pipeline is initialized once during app lifespan startup (main.py)
    and shared across all requests to avoid reloading heavy ML models.
    """
    return request.app.state.rag_pipeline
