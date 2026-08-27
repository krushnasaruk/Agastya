"""
Telemetry Endpoints and WebSocket Streaming.
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
from ..services.navigation_service import navigation_service

router = APIRouter(tags=["Telemetry"])


@router.get("/api/telemetry/history", response_model=List[Dict[str, Any]])
async def get_telemetry_history():
    """Get recent sliding buffer of telemetry packets."""
    return list(navigation_service.telemetry_history)


@router.websocket("/ws/telemetry")
async def websocket_telemetry_stream(websocket: WebSocket):
    """
    High-frequency 50Hz telemetry stream for real-time Avionics HUD and Map visualizers.
    """
    await websocket.accept()
    navigation_service.connected_websockets.add(websocket)
    print(f"[WebSocket] Client connected. Total active: {len(navigation_service.connected_websockets)}")

    try:
        while True:
            # Keep socket alive and handle incoming client commands (e.g. ping or mode changes)
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "set_mode":
                    navigation_service.set_mode(msg.get("mode", "ai_enhanced_ekf"))
                elif msg.get("action") == "inject_fault":
                    navigation_service.inject_fault(
                        msg.get("fault_type"),
                        msg.get("value", 1.0)
                    )
            except Exception:
                pass
    except WebSocketDisconnect:
        navigation_service.connected_websockets.discard(websocket)
        print(f"[WebSocket] Client disconnected. Total active: {len(navigation_service.connected_websockets)}")
    except Exception as e:
        navigation_service.connected_websockets.discard(websocket)
        print(f"[WebSocket] Socket error: {e}")
