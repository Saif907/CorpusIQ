import sys
import logging
from pathlib import Path

# Add backend directory to python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app import settings
from app.rag.pipeline import RAGPipeline

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("index_full_dataset")


def main():
    # Resolve absolute path to the main Enron email dataset
    dataset_file_path = Path(settings.DATA_DIR) / settings.THREADED_EMAILS_FILE

    logger.info(f"Resolving full dataset file path: {dataset_file_path}")

    if not dataset_file_path.exists():
        logger.error(f"Error: Dataset file not found at '{dataset_file_path}'. Please check file path.")
        return

    # Initialize RAG Pipeline
    pipeline = RAGPipeline()

    logger.info(f"Starting memory-bounded streaming indexing into Qdrant (Batch Size: {settings.INGESTION_BATCH_SIZE})...")
    metrics = pipeline.index_dataset(
        file_path=str(dataset_file_path),
        batch_size=settings.INGESTION_BATCH_SIZE
    )

    print("\n" + "=" * 50)
    print("      FULL DATASET INDEXING METRICS")
    print("=" * 50)
    print(metrics.model_dump_json(indent=2))
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
