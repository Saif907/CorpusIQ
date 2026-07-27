import time 
from pydantic import BaseModel, Field 
from typing import Optional


class IngestionMetrics(BaseModel):
    
    total_batches: int = 0
    total_processed: int = 0
    valid_count: int = 0
    skipped_count: int = 0
    start_time: float = Field(default_factory=time.time)
    end_time: Optional[float] = None
    elapsed_seconds: float = 0.0
    throughput_msg_per_sec: float = 0.0


