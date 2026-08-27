"""
Strongly Typed Sensor and Telemetry Data Packets for Real-Time Pipeline.
Provides strict validation, serialization, and array conversions for high-rate avionics.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass(frozen=True)
class IMUPacket:
    """Synchronized 6-DOF IMU sensor packet."""
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    temperature_c: float = 25.0
    sequence_id: int = 0

    def __post_init__(self):
        if self.timestamp < 0.0:
            raise ValueError(f"Timestamp must be non-negative, got {self.timestamp}")

    @property
    def accel(self) -> np.ndarray:
        return np.array([self.accel_x, self.accel_y, self.accel_z], dtype=np.float64)

    @property
    def gyro(self) -> np.ndarray:
        return np.array([self.gyro_x, self.gyro_y, self.gyro_z], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "accel": [self.accel_x, self.accel_y, self.accel_z],
            "gyro": [self.gyro_x, self.gyro_y, self.gyro_z],
            "temperature_c": float(self.temperature_c),
            "sequence_id": int(self.sequence_id)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IMUPacket":
        acc = data.get("accel", [0.0, 0.0, 0.0])
        gyr = data.get("gyro", [0.0, 0.0, 0.0])
        return cls(
            timestamp=float(data.get("timestamp", 0.0)),
            accel_x=float(acc[0]),
            accel_y=float(acc[1]),
            accel_z=float(acc[2]),
            gyro_x=float(gyr[0]),
            gyro_y=float(gyr[1]),
            gyro_z=float(gyr[2]),
            temperature_c=float(data.get("temperature_c", 25.0)),
            sequence_id=int(data.get("sequence_id", 0))
        )


@dataclass(frozen=True)
class GNSSPacket:
    """GNSS Position, Velocity, and Integrity packet."""
    timestamp: float
    pos_north: float
    pos_east: float
    pos_down: float
    vel_north: float
    vel_east: float
    vel_down: float
    hdop: float = 1.0
    satellites_visible: int = 10
    is_valid: bool = True
    sequence_id: int = 0

    @property
    def position_ned(self) -> np.ndarray:
        return np.array([self.pos_north, self.pos_east, self.pos_down], dtype=np.float64)

    @property
    def velocity_ned(self) -> np.ndarray:
        return np.array([self.vel_north, self.vel_east, self.vel_down], dtype=np.float64)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "position_ned": [self.pos_north, self.pos_east, self.pos_down],
            "velocity_ned": [self.vel_north, self.vel_east, self.vel_down],
            "hdop": float(self.hdop),
            "satellites_visible": int(self.satellites_visible),
            "is_valid": bool(self.is_valid),
            "sequence_id": int(self.sequence_id)
        }


@dataclass(frozen=True)
class VisualOdometryPacket:
    """Visual Odometry relative translation and rotation packet."""
    timestamp: float
    delta_pos: List[float]
    delta_yaw: float
    confidence: float = 1.0
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "delta_pos": list(self.delta_pos),
            "delta_yaw": float(self.delta_yaw),
            "confidence": float(self.confidence),
            "is_valid": bool(self.is_valid)
        }


@dataclass
class TelemetryFramePacket:
    """Unified broadcast telemetry frame combining estimated state, sensors, and metrics."""
    timestamp: float
    mode: str
    ground_truth_pos: List[float]
    estimated_pos: List[float]
    estimated_vel: List[float]
    euler_angles: Dict[str, float]
    covariance_diag: List[float]
    gnss_valid: bool
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "mode": str(self.mode),
            "ground_truth": {
                "pos": [round(x, 4) for x in self.ground_truth_pos]
            },
            "estimated": {
                "pos": [round(x, 4) for x in self.estimated_pos],
                "vel": [round(x, 4) for x in self.estimated_vel],
                "attitude": self.euler_angles,
                "cov_diag": [round(x, 6) for x in self.covariance_diag]
            },
            "gnss_valid": bool(self.gnss_valid),
            "metrics": self.metrics
        }
