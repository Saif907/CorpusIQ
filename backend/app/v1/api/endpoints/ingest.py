import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException

from app import settings
from app.rag.pipeline import RAGPipeline
from app.dependencies import get_rag_pipeline
from app.schemas.api_schema import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest Dataset",
    description="Streams a JSON email dataset through the ingestion pipeline, chunks it, and indexes into Qdrant with dual-vector embeddings.",
)
def ingest_dataset(
    request: IngestRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> IngestResponse:
    """Synchronous endpoint — ingestion is CPU-bound (embedding generation).
    FastAPI auto-runs sync handlers in a threadpool.
    """
    try:
        # Resolve dataset file path
        if request.file_path:
            target_path = Path(request.file_path)
        else:
            target_path = Path(settings.DATA_DIR) / settings.THREADED_EMAILS_FILE

        if not target_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Dataset file not found: {target_path}",
            )

        logger.info(
            f"API ingest request: file='{target_path}', "
            f"batch_size={request.batch_size}, max_records={request.max_records}"
        )

        metrics = pipeline.index_dataset(
            file_path=str(target_path),
            batch_size=request.batch_size,
            max_records=request.max_records,
        )

        return IngestResponse(
            status="completed",
            total_processed=metrics.total_processed,
            total_batches=metrics.total_batches,
            elapsed_seconds=metrics.elapsed_seconds,
            throughput_msg_per_sec=metrics.throughput_msg_per_sec,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline error: {str(e)}")
