import logging
from typing import List
from fastembed.rerank.cross_encoder import TextCrossEncoder
from app.schemas.rag_schema import RAGSearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Production-grade local Cross-Encoder Reranker using FastEmbed TextCrossEncoder on CPU."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        logger.info(f"Loading FastEmbed TextCrossEncoder model '{model_name}' on CPU...")
        self.model_name = model_name
        self.reranker = TextCrossEncoder(
            model_name=model_name,
            providers=["CPUExecutionProvider"]
        )

    def rerank(
        self, 
        query: str, 
        candidates: List[RAGSearchResult], 
        top_n: int = 5
    ) -> List[RAGSearchResult]:
       
        if not candidates:
            return []

        
        documents = [c.page_content for c in candidates]
        
        
        scores = list(self.reranker.rerank(query=query, documents=documents))
        
        
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

        scored_candidates.sort(key=lambda x: x.score, reverse=True)
        reranked_results = scored_candidates[:top_n]

        logger.info(
            f"Reranked {len(candidates)} candidates down to Top {len(reranked_results)} chunks via TextCrossEncoder."
        )
        return reranked_results
