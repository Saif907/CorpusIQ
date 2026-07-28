from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """API request body for the RAG search endpoint."""

    query: str = Field(..., min_length=1, max_length=1000, description="Natural language question or search query")
    candidate_k: int = Field(default=20, ge=1, le=100, description="Number of hybrid retrieval candidates before reranking")
    rerank_top_n: int = Field(default=5, ge=1, le=50, description="Number of top results after cross-encoder reranking")
    enable_thread_expansion: bool = Field(default=True, description="Inject parent thread context for replies")
    from_address_filter: Optional[str] = Field(default=None, description="Filter results by sender email address")
    thread_id_filter: Optional[str] = Field(default=None, description="Filter results by specific thread ID")
    sender_domain_filter: Optional[str] = Field(default=None, description="Filter results by sender domain (e.g. enron.com)")


class SearchResponse(BaseModel):
    """API response for the RAG search endpoint."""

    query: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    total_sources: int = 0


class IngestRequest(BaseModel):
    """API request body for the ingestion endpoint."""

    file_path: Optional[str] = Field(default=None, description="Path to dataset JSON file. Uses default Enron dataset if not provided.")
    batch_size: int = Field(default=500, ge=1, le=5000, description="Number of emails per ingestion batch")
    max_records: Optional[int] = Field(default=None, ge=1, description="Maximum emails to ingest. Processes all if not set.")


class IngestResponse(BaseModel):
    """API response for the ingestion endpoint."""

    status: str
    total_processed: int
    total_batches: int
    elapsed_seconds: float
    throughput_msg_per_sec: float


class CollectionStatsResponse(BaseModel):
    """API response for collection statistics."""

    collection_name: str
    points_count: int
    vectors_count: int
    status: str


class HealthResponse(BaseModel):
    """API response for health check."""

    status: str = "healthy"
    service: str = "CorpusIQ"
    version: str = "0.1.0"
