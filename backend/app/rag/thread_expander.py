import logging 
from typing import List, Set 
from app.schemas.rag_schema import RAGQueryRequest,RAGSearchResult
from app.rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class ThreadExpander:

    def __init__(self, vector_store: QdrantVectorStore):
        self.vector_store = vector_store

    def expand_threads(self, top_chunks: List[RAGSearchResult]) -> List[RAGSearchResult]:

        if not top_chunks:
            return []

        expanded_chunks = []

        seen_chunk_ids = set()

        for chunk in top_chunks:
            thread_id = chunk.metadata.get("thread_id")
            thread_position = chunk.metadata.get("thread_position", 0)
            # If the hit is a reply (position > 0), fetch the root email (position == 0) first
            if thread_position > 0 and thread_id:
                root_request = RAGQueryRequest(
                    query="",  # Empty query; metadata filter retrieves the root thread point
                    top_k=1,
                    thread_id_filter=thread_id
                )
                
                root_hits = self.vector_store.search(root_request)
                if root_hits:
                    root_chunk = root_hits[0]
                    if root_chunk.chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(root_chunk.chunk_id)
                        expanded_chunks.append(root_chunk)
         
            if chunk.chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk.chunk_id)
                expanded_chunks.append(chunk)
        logger.info(f"Thread Expander expanded {len(top_chunks)} chunks to {len(expanded_chunks)} thread-aware chunks.")
        return expanded_chunks