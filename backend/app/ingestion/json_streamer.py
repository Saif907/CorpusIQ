import ijson
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator
from app.schemas.email import EmailThread, EmailMessage
from app.ingestion.base import BaseIngestor

logger = logging.getLogger(__name__)


class JsonStreamer():

    def stream_messages(self, file_path: str) -> Generator[EmailMessage, None, None]:
        
        filepath = Path(file_path)

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found at: {filepath}")

        with open(filepath, "rb") as f:
            for thread_id, raw_messages in ijson.kvitems(f, ""):
                if not isinstance(raw_messages, list):
                    logger.warning(
                        f"Skipping key '{thread_id}' expected list of msg but got {type(raw_messages)}"
                    )
                    continue

                for msg_dict in raw_messages:
                    try:
                        msg = EmailMessage.model_validate(msg_dict)
                        yield msg
                    except Exception as val_err:
                        logger.warning(
                            f"Validation failed for thread id: {thread_id}, error: {val_err}"
                        )

    def stream_batches(
        self,
        file_path: str,
        batch_size: int = 100
    ) -> Generator[List[EmailMessage], None, None]:
        
        batch = []

        for msg in self.stream_messages(file_path):
            batch.append(msg)
            if len(batch) >= batch_size:
                yield batch 
                batch = []
            
        if batch:  
            yield batch


