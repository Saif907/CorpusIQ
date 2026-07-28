import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.schemas.email import EmailMessage
from app.schemas.rag_schema import RAGChunk, RAGQueryRequest, RAGSearchResult, RAGQueryResponse
from app.ingestion.pipeline import IngestionPipeline
from app.rag.chunker import EmailChunker
from app.rag.vector_store import QdrantVectorStore
from app.rag.reranker import CrossEncoderReranker
from app.rag.thread_expander import ThreadExpander
from app.rag.synthesizer import RAGSynthesizer

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Enterprise 7-Stage Hybrid RAG Pipeline with Cross-Encoder Reranking & Thread Expansion."""

    def __init__(
        self, 
        vector_store: Optional[QdrantVectorStore] = None,
        chunker: Optional[EmailChunker] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        thread_expander: Optional[ThreadExpander] = None,
        synthesizer: Optional[RAGSynthesizer] = None
    ):
        self.vector_store = vector_store or QdrantVectorStore()
        self.chunker = chunker or EmailChunker()
        self.reranker = reranker or CrossEncoderReranker()
        self.thread_expander = thread_expander or ThreadExpander(vector_store=self.vector_store)
        self.synthesizer = synthesizer or RAGSynthesizer()

    def process_and_index_batch(self, batch: List[EmailMessage]):
        """Callback handler: Chunks and indexes a batch of EmailMessage objects into Qdrant."""
        if not batch:
            return

        all_chunks: List[RAGChunk] = []
        for msg in batch:
            chunks = self.chunker.process_message(msg)
            all_chunks.extend(chunks)

        logger.info(f"Chunker generated {len(all_chunks)} RAG chunks from {len(batch)} emails.")
        self.vector_store.ingest_chunks(all_chunks)

    def index_dataset(self, file_path: str, batch_size: int = 500, max_records: Optional[int] = None):
        """Streams dataset file via Stage 1 Ingestion and indexes it into Qdrant."""
        logger.info(f"Starting end-to-end RAG indexing for file: {file_path}")
        ingest_pipeline = IngestionPipeline(batch_size=batch_size)
        metrics = ingest_pipeline.run(
            file_path=file_path, 
            batch_callback=self.process_and_index_batch,
            max_records=max_records
        )
        logger.info(f"Indexing complete. Indexed {metrics.total_processed} messages into Qdrant.")
        return metrics

    def query(
        self, 
        query_text: str, 
        candidate_k: int = 20, 
        rerank_top_n: int = 5,
        enable_thread_expansion: bool = True,
        from_address_filter: Optional[str] = None,
        thread_id_filter: Optional[str] = None,
        sender_domain_filter: Optional[str] = None
    ) -> RAGQueryResponse:
        """Executes the full 7-stage retrieval & synthesis pipeline:
        
        1. BM25 + Dense Hybrid Vector Search (Qdrant RRF) -> Candidate_K
        2. Cross-Encoder Reranking (BAAI/bge-reranker-base) -> Rerank_Top_N
        3. Thread Expansion (Parent-Child context window)
        4. LLM Synthesis with Source Citations
        """
        logger.info(f"Executing 7-stage RAG query: '{query_text}'")

        # Step 1: Hybrid BM25 + Dense Vector Search via Qdrant RRF
        request = RAGQueryRequest(
            query=query_text,
            top_k=candidate_k,
            from_address_filter=from_address_filter,
            thread_id_filter=thread_id_filter,
            sender_domain_filter=sender_domain_filter
        )
        candidate_hits = self.vector_store.search(request)
        logger.info(f"Qdrant RRF search retrieved {len(candidate_hits)} candidate chunks.")

        if not candidate_hits:
            return RAGQueryResponse(query=query_text, answer="No relevant emails found.", sources=[])

        # Step 2: Cross-Encoder Reranking
        reranked_hits = self.reranker.rerank(
            query=query_text, 
            candidates=candidate_hits, 
            top_n=rerank_top_n
        )

        # Step 3: Thread Context Expansion (Parent-Child)
        if enable_thread_expansion:
            final_context_chunks = self.thread_expander.expand_threads(reranked_hits)
        else:
            final_context_chunks = reranked_hits

        # Step 4: LLM Synthesis & Citation Formatting
        response = self.synthesizer.synthesize_answer(
            query=query_text, 
            chunks=final_context_chunks
        )
        
        return response
