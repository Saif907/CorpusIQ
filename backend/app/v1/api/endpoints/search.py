import logging
from fastapi import APIRouter, Depends, HTTPException

from app.rag.pipeline import RAGPipeline
from app.dependencies import get_rag_pipeline
from app.schemas.api_schema import SearchRequest, SearchResponse, CollectionStatsResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="RAG Search",
    description="Executes the full hybrid RAG pipeline: BM25+Dense retrieval → RRF fusion → Cross-Encoder reranking → Thread expansion → LLM synthesis with citations.",
)
def search(
    request: SearchRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> SearchResponse:
    """Synchronous endpoint — FastAPI runs this in a threadpool automatically
    since the underlying pipeline is CPU-bound (embeddings, reranking).
    """
    try:
        logger.info(f"API search request: query='{request.query}', candidate_k={request.candidate_k}")

        response = pipeline.query(
            query_text=request.query,
            candidate_k=request.candidate_k,
            rerank_top_n=request.rerank_top_n,
            enable_thread_expansion=request.enable_thread_expansion,
            from_address_filter=request.from_address_filter,
            thread_id_filter=request.thread_id_filter,
            sender_domain_filter=request.sender_domain_filter,
        )

        return SearchResponse(
            query=response.query,
            answer=response.answer,
            sources=response.sources,
            total_sources=len(response.sources),
        )

    except Exception as e:
        logger.error(f"Search endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search pipeline error: {str(e)}")


@router.get(
    "/collection/stats",
    response_model=CollectionStatsResponse,
    summary="Collection Statistics",
    description="Returns Qdrant vector collection metadata (point count, vector count, status).",
)
def collection_stats(
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> CollectionStatsResponse:
    """Fetches live stats from the underlying Qdrant collection."""
    try:
        collection_info = pipeline.vector_store.client.get_collection(
            collection_name=pipeline.vector_store.collection_name
        )

        return CollectionStatsResponse(
            collection_name=pipeline.vector_store.collection_name,
            points_count=collection_info.points_count or 0,
            vectors_count=collection_info.vectors_count or 0,
            status=str(collection_info.status),
        )

    except Exception as e:
        logger.error(f"Collection stats failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch collection stats: {str(e)}")
