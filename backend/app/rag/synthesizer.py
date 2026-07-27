import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

from app import settings
from app.schemas.rag_schema import RAGSearchResult, RAGQueryResponse

logger = logging.getLogger(__name__)


class RAGSynthesizer:
    """Formats retrieved context and generates grounded AI answers using the OpenAI Python SDK."""

    def __init__(self, model_name: Optional[str] = None):
  
        if settings.GROQ_API_KEY:
            
            self.client = OpenAI(
                api_key=settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )
            self.model_name = model_name or settings.GROQ_MODEL or "llama-3.3-70b-versatile"
        elif settings.OPENROUTER_API_KEY:
            self.client = OpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1"
            )
            self.model_name = model_name or settings.OPENROUTER_MODEL or "openai/gpt-4o-mini"
        else:
            self.client = None
            self.model_name = model_name or "gpt-4o-mini"

    def build_context_text(self, chunks: List[RAGSearchResult]) -> str:
        """Formats retrieved RAG chunks into a clean, structured context string."""
        if not chunks:
            return "No relevant email documents were retrieved."

        context_blocks = []
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk.metadata
            header = (
                f"[DOCUMENT {idx}]\n"
                f"Thread ID: {meta.get('thread_id', 'N/A')}\n"
                f"Subject: {meta.get('subject', 'N/A')}\n"
                f"From: {meta.get('from_address', 'N/A')} | To: {meta.get('to_address', 'N/A')}\n"
                f"Date: {meta.get('date_str', 'N/A')}\n"
                f"Cross-Encoder Score: {chunk.score}\n"
            )
            block = f"{header}\nContent:\n{chunk.page_content}"
            context_blocks.append(block)

        return "\n\n" + ("=" * 50) + "\n\n".join(context_blocks) + "\n\n" + ("=" * 50)

    def extract_sources(self, chunks: List[RAGSearchResult]) -> List[Dict[str, Any]]:
        """Extracts citation metadata dicts for search response attribution."""
        sources = []
        for chunk in chunks:
            sources.append({
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "thread_id": chunk.metadata.get("thread_id"),
                "subject": chunk.metadata.get("subject"),
                "from_address": chunk.metadata.get("from_address"),
                "date_str": chunk.metadata.get("date_str")
            })
        return sources

    def synthesize_answer(
        self, 
        query: str, 
        chunks: List[RAGSearchResult]
    ) -> RAGQueryResponse:
       
        context_text = self.build_context_text(chunks)
        sources = self.extract_sources(chunks)

        system_prompt = (
            "You are CorpusIQ, an enterprise email intelligence AI assistant.\n"
            "Answer the user's question using ONLY the provided email context documents below.\n"
            "Always cite the Thread ID and Sender when referencing facts.\n"
            "If the answer cannot be determined from the context, state clearly that the information is not available in the emails."
        )

        user_prompt = f"User Question: {query}\n\nEmail Context Documents:\n{context_text}"

        answer_text = ""
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2
                )
                answer_text = response.choices[0].message.content
            except Exception as err:
                logger.warning(f"OpenAI Client completion error: {err}")

        # Fallback response if no API key is configured or call fails
        if not answer_text:
            answer_text = (
                f"Retrieved {len(chunks)} relevant email chunks from CorpusIQ index.\n"
                f"Top Context Document: Subject: '{chunks[0].metadata.get('subject')}' from {chunks[0].metadata.get('from_address')}.\n"
                f"(Configure OPENAI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY in .env for full LLM synthesis)."
            ) if chunks else "No relevant documents found."

        return RAGQueryResponse(
            query=query,
            answer=answer_text,
            sources=sources
        )
