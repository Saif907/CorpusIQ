import logging
from fastapi import APIRouter

from app.schemas.api_schema import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns service health status. Use for uptime monitoring and load balancer probes.",
)
async def health_check() -> HealthResponse:
    """Lightweight async health probe — no heavy computation."""
    return HealthResponse()
