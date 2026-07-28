from fastapi import APIRouter
from app.v1.api.endpoints import health, search, ingest

api_router = APIRouter()
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(search.router, tags=["Search"])
api_router.include_router(ingest.router, tags=["Ingestion"])
