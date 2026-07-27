import logging
from typing import List
from fastembed.rerank.cross_encoder import TextCrossEncoder
from app.schemas.rag_schema import RAGSearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Production-grade local Cross-Encoder Reranker using FastEmbed v0.8.0 TextCrossEncoder."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        logger.info(f"Loading FastEmbed TextCrossEncoder model: {model_name}")
        self.model_name = model_name
        self.reranker = TextCrossEncoder(model_name=model_name)

    def rerank(
        self, 
        query: str, 
        candidates: List[RAGSearchResult], 
        top_n: int = 5
    ) -> List[RAGSearchResult]:
        """Re-scores candidate hits using joint Query-Document attention.
        
        Args:
            query: User search question.
            candidates: Candidate RAGSearchResult objects retrieved from Qdrant.
            top_n: Number of highest-precision results to return.
            
        Returns:
            List of RAGSearchResult objects sorted by Cross-Encoder relevance score.
        """
        if not candidates:
            return []

        # Extract text blocks for Cross-Encoder joint scoring
        documents = [c.page_content for c in candidates]
        
        # Execute TextCrossEncoder reranking (yields list of float scores)
        scores = list(self.reranker.rerank(query=query, documents=documents))
        
        # Pair candidates with their Cross-Encoder score
        scored_candidates: List[RAGSearchResult] = []
        for candidate, score in zip(candidates, scores):
            scored_candidates.append(
                RAGSearchResult(
                    score=round(float(score), 4),
                    chunk_id=candidate.chunk_id,
                    page_content=candidate.page_content,
                    metadata=candidate.metadata
                )
            )

        # Sort descending by Cross-Encoder precision score
        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        reranked_results = scored_candidates[:top_n]

        logger.info(
            f"Reranked {len(candidates)} candidates down to Top {len(reranked_results)} chunks via TextCrossEncoder."
        )
        return reranked_results
