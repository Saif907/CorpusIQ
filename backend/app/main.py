import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logger import setup_logging
from app.rag.pipeline import RAGPipeline
from app.v1.api import api_router

# Initialize structured logging before anything else
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Startup: Initializes the RAG pipeline once (loads embedding models,
    cross-encoder reranker, and Qdrant connection). These are heavy ML models
    that must be loaded exactly once and shared across all requests.

    Shutdown: Cleanup and log graceful termination.
    """
    logger.info("Starting CorpusIQ — initializing RAG pipeline and ML models...")
    app.state.rag_pipeline = RAGPipeline()
    logger.info("RAG pipeline initialized. All models loaded and ready.")
    yield
    logger.info("Shutting down CorpusIQ.")


app = FastAPI(
    title="CorpusIQ",
    description=(
        "AI-powered enterprise email intelligence API. "
        "Hybrid RAG with cross-encoder reranking, thread-aware context expansion, "
        "and multi-provider LLM synthesis with source citations."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permissive for development, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount versioned API routes at /api/v1
app.include_router(api_router, prefix=settings.API_PREFIX)