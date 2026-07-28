import time 
import logging 
from pathlib import Path 
from typing import Dict, Any, List, Optional
from app import settings
from app.ingestion.json_streamer import JsonStreamer
from app.schemas.email import EmailMessage
from app.schemas.ingestion_schema import IngestionMetrics

logger = logging.getLogger(__name__)


class IngestionPipeline:

    def __init__(self, batch_size: int = None):
        self.streamer = JsonStreamer()
        self.batch_size = batch_size or settings.INGESTION_BATCH_SIZE
        self.file_path = Path(settings.DATA_DIR) / settings.THREADED_EMAILS_FILE

    def run(self, file_path=None, batch_callback=None, max_records: Optional[int] = None) -> IngestionMetrics:
        metrics = IngestionMetrics()

        target_file_path = Path(file_path) if file_path else self.file_path

        logger.info(f"Starting ingestion pipeline for file: {target_file_path}")
        start_time = time.perf_counter()

        try:
            for batch in self.streamer.stream_batches(
                file_path=target_file_path, batch_size=self.batch_size
            ):
                metrics.total_processed += len(batch)
                metrics.valid_count += len(batch)
                metrics.total_batches += 1

                if batch_callback:
                    batch_callback(batch)

                if max_records and metrics.total_processed >= max_records:
                    logger.info(f"Reached max_records limit of {max_records}. Halting ingestion stream.")
                    break

            metrics.end_time = time.time()
            metrics.elapsed_seconds = round(time.perf_counter() - start_time, 2)
            if metrics.elapsed_seconds > 0:
                metrics.throughput_msg_per_sec = round(
                    metrics.total_processed / metrics.elapsed_seconds, 2
                )

        except Exception as e:
            logger.error(f"Ingestion pipeline failed: {e}")
            raise e

        return metrics
