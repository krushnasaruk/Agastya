"""
Pydantic Schemas for Navigation State, Telemetry Frames, and API Requests.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class EulerAngles(BaseModel):
    roll: float
    pitch: float
    yaw: float


class NavigationStateSchema(BaseModel):
    timestamp: float
    position: List[float] = Field(..., description="[North, East, Down] in meters")
    velocity: List[float] = Field(..., description="[v_N, v_E, v_D] in m/s")
    quaternion: List[float] = Field(..., description="[qw, qx, qy, qz]")
    euler: EulerAngles
    accel_bias: List[float]
    gyro_bias: List[float]
    cov_diag: List[float]
    mode: str
    gnss_valid: bool


class GroundTruthStateSchema(BaseModel):
    timestamp: float
    position: List[float]
    velocity: List[float]
    euler: EulerAngles


class IMUSchema(BaseModel):
    timestamp: float
    accel: List[float]
    gyro: List[float]
    temperature: float
    is_valid: bool


class GNSSSchema(BaseModel):
    timestamp: float
    position: List[float]
    velocity: List[float]
    fix_type: int
    satellites_in_view: int
    hdop: float
    vdop: float
    cov_diag: List[float]
    is_valid: bool


class VOSchema(BaseModel):
    timestamp: float
    velocity_body: List[float]
    confidence: float
    inlier_count: int
    is_valid: bool


class TelemetryMetrics(BaseModel):
    ate_rmse: float
    max_error: float
    drift_percentage: float
    total_distance: float
    ai_confidence: float = 0.95


class TelemetryFrameSchema(BaseModel):
    timestamp: float
    mode: str
    ground_truth: GroundTruthStateSchema
    estimated: NavigationStateSchema
    pure_dr: Dict[str, Any]
    classical_ekf: Dict[str, Any]
    imu: IMUSchema
    gnss: Optional[GNSSSchema] = None
    vo: Optional[VOSchema] = None
    metrics: TelemetryMetrics
    scenario_progress: float = 0.0


class ModeChangeRequest(BaseModel):
    mode: str = Field(..., description="Operating mode: ai_enhanced_ekf, classical_ekf, pure_dr, ai_only")


class FaultInjectionRequest(BaseModel):
    fault_type: str = Field(..., description="gps_jamming, accel_bias_jump, gyro_bias_jump, vo_dropout")
    value: Optional[float] = 1.0
    duration_sec: Optional[float] = 10.0


class ScenarioInfoSchema(BaseModel):
    id: str
    name: str
    description: str
    duration_sec: float
    trajectory_type: str
    speed_mps: float


class SimulationSpeedRequest(BaseModel):
    speed_multiplier: float = Field(1.0, ge=0.1, le=20.0)
