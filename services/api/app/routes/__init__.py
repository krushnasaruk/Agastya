from .navigation import router as navigation_router
from .telemetry import router as telemetry_router
from .health import router as health_router
from .simulation import router as simulation_router

__all__ = [
    "navigation_router",
    "telemetry_router",
    "health_router",
    "simulation_router",
]
