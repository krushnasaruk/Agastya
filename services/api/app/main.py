"""
FastAPI Application Entrypoint for AGASTYA AI Dead Reckoning Engine.
"""

import sys
import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure path resolution
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
NAV_ENGINE_DIR = os.path.join(BASE_DIR, "services", "navigation-engine")
ML_DIR = os.path.join(BASE_DIR, "services", "ml")
for p in [BASE_DIR, NAV_ENGINE_DIR, ML_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from services.api.app.routes import (
    navigation_router,
    telemetry_router,
    health_router,
    simulation_router
)
from services.api.app.services.navigation_service import navigation_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Start background 100Hz simulation loop
    loop_task = asyncio.create_task(navigation_service.simulation_loop())
    print("[AGASTYA API] Application started. Simulation loop active.")
    yield
    # Shutdown: Cancel loop
    loop_task.cancel()
    try:
        await loop_task
    except asyncio.CancelledError:
        pass
    print("[AGASTYA API] Application shutdown.")


app = FastAPI(
    title="AGASTYA AI Dead Reckoning API",
    description="Asynchronous Neural-Inertial Navigation, ES-EKF Sensor Fusion, and Telemetry Engine.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(navigation_router)
app.include_router(telemetry_router)
app.include_router(health_router)
app.include_router(simulation_router)


@app.get("/")
async def root():
    return {
        "name": "AGASTYA AI Dead Reckoning System",
        "docs": "/docs",
        "ws_telemetry": "/ws/telemetry",
        "health": "/api/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.api.app.main:app", host="0.0.0.0", port=8000, reload=True)
