import uuid
import logging
from typing import List
from app.schemas.email import EmailMessage
from app.schemas.rag_schema import RAGChunk

logger = logging.getLogger(__name__)


class EmailChunker:

    def __init__(
        self,
        max_chunk_chars : int = 1500,
        overlap_chars: int = 150
    ):

        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars

    
    def extract_domain(self, email_address: str) -> str:

        if "@" in email_address:
            return email_address.split("@")[-1].strip().lower()
        return "unknown"

    
    def format_header_text(self, msg: EmailMessage) -> str:

        return (
            f"Subject: {msg.subject}\n"
            f"From: {msg.from_address} | To: {msg.to_address}\n"
            f"Date: {msg.date_str}\n\n"
            f"{msg.body}"
        )

    def process_message(self, msg: EmailMessage) -> List[RAGChunk]:

        full_text = self.format_header_text(msg)
        sender_domain = self.extract_domain(msg.from_address)
        recipient_domain = self.extract_domain(msg.to_address)
        is_thread_root = (msg.thread_position == 0) or (msg.in_reply_to == "")

        base_metadata = {
            "message_id": msg.message_id,
            "thread_id": msg.thread_id,
            "in_reply_to": msg.in_reply_to,
            "is_thread_root": is_thread_root,
            "from_address": msg.from_address,
            "to_address": msg.to_address,
            "sender_domain": sender_domain,
            "recipient_name": recipient_domain,
            "subject": msg.subject,
            "date_str": msg.date_str,
            "filename": msg.filename
        }


        if len(full_text) <= self.max_chunk_chars:
            chunk_id = f"{msg.message_id}_chunk_0"
            chunk_metadata = {**base_metadata, "chunk_index": 0, "total_chunks": 1}
            return [
                RAGChunk(
                    chunk_id=chunk_id,
                    page_content=full_text,
                    metadata=chunk_metadata
                )
            ]
        

        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(full_text):
            end = start + self.max_chunk_chars
            chunk_text = full_text[start:end]
            chunk_id = f"{msg.message_id}_chunk_{chunk_idx}"
            
            chunk_metadata = {**base_metadata, "chunk_index": chunk_idx}
            chunks.append(
                RAGChunk(
                    chunk_id=chunk_id,
                    page_content=chunk_text,
                    metadata=chunk_metadata
                )
            )
            start += (self.max_chunk_chars - self.overlap_chars)
            chunk_idx += 1

        for c in chunks:
            c.metadata["total_chunks"] = len(chunks)
        return chunks