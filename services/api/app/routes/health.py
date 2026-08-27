"""
Health Check and System Diagnostics.
"""

from fastapi import APIRouter
import time
from ..services.navigation_service import navigation_service

router = APIRouter(prefix="/api/health", tags=["Health"])

START_TIME = time.time()


@router.get("")
async def health_check():
    """System health check and operational metrics."""
    uptime_sec = round(time.time() - START_TIME, 2)
    return {
        "status": "healthy",
        "system": "AGASTYA AI Dead Reckoning Engine",
        "uptime_seconds": uptime_sec,
        "active_scenario": navigation_service.scenario_name,
        "is_simulating": navigation_service.is_running,
        "connected_ws_clients": len(navigation_service.connected_websockets),
        "total_distance_m": round(navigation_service.fusion_ai.total_distance_travelled, 1)
    }
