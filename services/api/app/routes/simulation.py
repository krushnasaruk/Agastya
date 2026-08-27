"""
Simulation Control Endpoints.
"""

from fastapi import APIRouter, HTTPException
import os
import json
from typing import List
from ..services.navigation_service import navigation_service, BASE_DIR
from ..schemas.navigation import ScenarioInfoSchema, SimulationSpeedRequest

router = APIRouter(prefix="/api/simulation", tags=["Simulation"])


@router.get("/scenarios", response_model=List[ScenarioInfoSchema])
async def list_scenarios():
    """List all available trajectory scenarios."""
    scenarios_dir = os.path.join(BASE_DIR, "simulation", "scenarios")
    results = []
    
    if os.path.exists(scenarios_dir):
        for f in os.listdir(scenarios_dir):
            if f.endswith(".json"):
                sc_id = f.replace(".json", "")
                fpath = os.path.join(scenarios_dir, f)
                try:
                    with open(fpath, "r") as fp:
                        data = json.load(fp)
                        results.append(ScenarioInfoSchema(
                            id=sc_id,
                            name=data.get("name", sc_id),
                            description=data.get("description", ""),
                            duration_sec=data.get("duration_sec", 45.0),
                            trajectory_type=data.get("trajectory_type", "figure_8"),
                            speed_mps=data.get("speed_mps", 20.0)
                        ))
                except Exception:
                    pass
    return results


@router.post("/scenario/{name}")
async def select_scenario(name: str):
    """Load and execute scenario."""
    scenarios_dir = os.path.join(BASE_DIR, "simulation", "scenarios")
    target_file = os.path.join(scenarios_dir, f"{name}.json")
    if not os.path.exists(target_file):
        raise HTTPException(status_code=404, detail=f"Scenario '{name}' not found")

    navigation_service.load_scenario(name)
    return {"status": "success", "active_scenario": name}


@router.post("/start")
async def start_simulation():
    """Resume / start simulation execution."""
    navigation_service.is_running = True
    return {"status": "success", "is_running": True}


@router.post("/pause")
async def pause_simulation():
    """Pause simulation clock."""
    navigation_service.is_running = False
    return {"status": "success", "is_running": False}


@router.post("/reset")
async def reset_simulation():
    """Reset simulation clock to t=0."""
    navigation_service.reset()
    return {"status": "success", "message": "Simulation reset to t=0"}


@router.post("/speed")
async def set_speed(req: SimulationSpeedRequest):
    """Set simulation playback speed multiplier."""
    navigation_service.set_playback_speed(req.speed_multiplier)
    return {"status": "success", "speed_multiplier": req.speed_multiplier}
