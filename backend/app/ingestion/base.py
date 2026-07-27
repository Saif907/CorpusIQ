from abc import ABC, abstractmethod
from typing import Generator, List
from app.schemas.email import EmailMessage, EmailThread


class BaseIngestor(ABC):

    @abstractmethod
    def stream_messages(self, file_path: str) -> Generator[EmailThread, None, None]:
        pass 


    @abstractmethod
    def stream_batches(
        self, 
        file_path: str, 
        batch_size: int = 500
        ) -> Generator[List[EmailThread], None, None]:

        pass
