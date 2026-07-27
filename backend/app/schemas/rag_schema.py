from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class RAGChunk(BaseModel):
  
    chunk_id: str = Field(..., description="Unique chunk ID (e.g., hash or msg_id_chunk_0)")
    page_content: str = Field(..., description="Header-formatted text block embedded into vector DB")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Rich payload metadata for Qdrant filtering")


class RAGQueryRequest(BaseModel):
   
    query: str = Field(..., description="User question or search query")
    top_k: int = Field(default=5, description="Number of vector results to retrieve")
    from_address_filter: Optional[str] = Field(default=None, description="Filter by sender email")
    thread_id_filter: Optional[str] = Field(default=None, description="Filter by specific thread ID")
    sender_domain_filter: Optional[str] = Field(default=None, description="Filter by sender domain (e.g. enron.com)")


class RAGSearchResult(BaseModel):
  
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    chunk_id: str
    page_content: str
    metadata: Dict[str, Any]


class RAGQueryResponse(BaseModel):
   
    query: str
    answer: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
