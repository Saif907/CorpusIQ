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

    # Index 10,000 emails with batch size 500 for fast CPU indexing
    target_limit = 10000
    batch_size = 500

    logger.info(f"Starting multi-core CPU indexing into Qdrant (Limit: {target_limit} emails, Batch Size: {batch_size})...")
    metrics = pipeline.index_dataset(
        file_path=str(dataset_file_path),
        batch_size=batch_size,
        max_records=target_limit
    )

    print("\n" + "=" * 50)
    print("      DATASET INDEXING METRICS")
    print("=" * 50)
    print(metrics.model_dump_json(indent=2))
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
