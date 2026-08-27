"""
Physical Navigation State Definitions for Project AGASTYA (Objective 3).
Provides 2D planar ground vehicle navigation state containers and angle wrapping utilities
in the Local East-North-Up (ENU) metric coordinate frame.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
import numpy as np


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap angle in radians to [-pi, pi]."""
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def wrap_to_2pi(angle_rad: float) -> float:
    """Wrap angle in radians to [0, 2*pi)."""
    return float(angle_rad % (2.0 * np.pi))


@dataclass
class PlanarNavigationState:
    """
    2D Planar Ground Vehicle Navigation State in Local East-North-Up (ENU) Frame.
    
    Coordinate Convention:
      - East Position (p_east_m): Meters along Local East axis (+E positive East)
      - North Position (p_north_m): Meters along Local North axis (+N positive North)
      - Heading (heading_rad): Radians clockwise from Geographic North [0, 2*pi)
      - Forward Speed (forward_speed_ms): Linear velocity along vehicle body +X axis (m/s)
      - Timestamp (time_sec): Current sequence timestamp in seconds
    """
    time_sec: float = 0.0
    p_east_m: float = 0.0
    p_north_m: float = 0.0
    heading_rad: float = 0.0
    forward_speed_ms: float = 0.0
    yaw_rate_rads: float = 0.0
    accel_longitudinal_ms2: float = 0.0
    is_stationary: bool = False
    quality_status: str = "VALID"

    def clone(self) -> "PlanarNavigationState":
        return PlanarNavigationState(
            time_sec=self.time_sec,
            p_east_m=self.p_east_m,
            p_north_m=self.p_north_m,
            heading_rad=self.heading_rad,
            forward_speed_ms=self.forward_speed_ms,
            yaw_rate_rads=self.yaw_rate_rads,
            accel_longitudinal_ms2=self.accel_longitudinal_ms2,
            is_stationary=self.is_stationary,
            quality_status=self.quality_status
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_sec": round(self.time_sec, 4),
            "p_east_m": round(self.p_east_m, 4),
            "p_north_m": round(self.p_north_m, 4),
            "heading_rad": round(self.heading_rad, 5),
            "heading_deg": round(float(np.degrees(self.heading_rad)), 2),
            "forward_speed_ms": round(self.forward_speed_ms, 3),
            "forward_speed_kmh": round(self.forward_speed_ms * 3.6, 2),
            "yaw_rate_rads": round(self.yaw_rate_rads, 5),
            "is_stationary": self.is_stationary,
            "quality_status": self.quality_status
        }


@dataclass
class DeadReckoningTrajectory:
    """
    Complete propagated dead-reckoning trajectory container.
    """
    timestamps_sec: np.ndarray
    dt_array_sec: np.ndarray
    p_east_m: np.ndarray
    p_north_m: np.ndarray
    heading_rad: np.ndarray
    forward_speed_ms: np.ndarray
    yaw_rate_rads: np.ndarray
    baseline_name: str
    total_distance_m: float = 0.0

    def to_dataframe(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame({
            "time_sec": self.timestamps_sec,
            "dt_sec": self.dt_array_sec,
            "estimated_p_east_m": self.p_east_m,
            "estimated_p_north_m": self.p_north_m,
            "estimated_heading_rad": self.heading_rad,
            "estimated_heading_deg": np.degrees(self.heading_rad),
            "estimated_speed_ms": self.forward_speed_ms,
            "yaw_rate_rads": self.yaw_rate_rads
        })
