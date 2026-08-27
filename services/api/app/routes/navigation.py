"""
Navigation REST Endpoints.
"""

from fastapi import APIRouter, HTTPException
from ..services.navigation_service import navigation_service
from ..schemas.navigation import NavigationStateSchema, ModeChangeRequest, FaultInjectionRequest

router = APIRouter(prefix="/api/navigation", tags=["Navigation"])


@router.get("/state", response_model=dict)
async def get_navigation_state():
    """Retrieve current fused navigation state and covariance."""
    return navigation_service.fusion_ai.fused_state.to_dict()


@router.post("/reset")
async def reset_navigation():
    """Reset filter, integrator, and error buffers."""
    navigation_service.reset()
    return {"status": "success", "message": "Navigation engine state reset"}


@router.post("/mode")
async def set_navigation_mode(req: ModeChangeRequest):
    """Set operating navigation mode."""
    allowed = ["ai_enhanced_ekf", "classical_ekf", "pure_dr", "ai_only"]
    if req.mode not in allowed:
        raise HTTPException(status_code=400, detail=f"Mode must be one of {allowed}")
    navigation_service.set_mode(req.mode)
    return {"status": "success", "mode": req.mode}


@router.post("/inject-fault")
async def inject_fault(req: FaultInjectionRequest):
    """Inject runtime sensor fault."""
    navigation_service.inject_fault(
        fault_type=req.fault_type,
        value=req.value if req.value is not None else 1.0,
        duration_sec=req.duration_sec if req.duration_sec is not None else 10.0
    )
    return {"status": "success", "fault": req.fault_type}
