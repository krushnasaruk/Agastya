"""
Safety Guard Module for Project AGASTYA (Objective 5).
Enforces physical bounding gates on AI residual predictions and guarantees
deterministic fallback to the classical physics engine during invalid sensor states or low confidence.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional
import numpy as np


@dataclass
class GuardedResidual:
    delta_velocity_ms: float
    delta_yaw_rate_rads: float
    is_clamped: bool
    is_fallback: bool
    status: str


class SafetyGuard:
    """
    Physical bounds and sensor health guard for AI residual navigation corrections.
    """
    def __init__(
        self,
        max_velocity_bound_ms: float = 3.0,     # Max plausible tire slip speed (~10.8 km/h)
        max_yaw_bound_rads: float = 0.50,       # Max plausible cornering yaw rate correction (~28.6 deg/s)
        max_velocity_variance: float = 1.00,
        max_yaw_variance: float = 0.25
    ):
        self.max_v = max_velocity_bound_ms
        self.max_yaw = max_yaw_bound_rads
        self.max_var_v = max_velocity_variance
        self.max_var_yaw = max_yaw_variance

    def clamp_velocity(self, raw_delta_v: float) -> Tuple[float, bool]:
        if abs(raw_delta_v) > self.max_v:
            return float(np.clip(raw_delta_v, -self.max_v, self.max_v)), True
        return float(raw_delta_v), False

    def clamp_yaw_rate(self, raw_delta_yaw: float) -> Tuple[float, bool]:
        if abs(raw_delta_yaw) > self.max_yaw:
            return float(np.clip(raw_delta_yaw, -self.max_yaw, self.max_yaw)), True
        return float(raw_delta_yaw), False

    def sanitize(
        self,
        raw_delta_v: float,
        raw_delta_yaw: float,
        is_sensor_valid: bool = True,
        is_stationary: bool = False,
        var_v: float = 0.0,
        var_yaw: float = 0.0
    ) -> GuardedResidual:
        # 1. Sensor Health or Stationary Check -> Fallback
        if not is_sensor_valid or is_stationary:
            return GuardedResidual(
                delta_velocity_ms=0.0,
                delta_yaw_rate_rads=0.0,
                is_clamped=False,
                is_fallback=True,
                status="FALLBACK_SENSOR_DEGRADED_OR_STATIONARY"
            )

        # 2. Uncertainty Check -> Fallback
        if var_v > self.max_var_v or var_yaw > self.max_var_yaw:
            return GuardedResidual(
                delta_velocity_ms=0.0,
                delta_yaw_rate_rads=0.0,
                is_clamped=False,
                is_fallback=True,
                status="FALLBACK_LOW_CONFIDENCE"
            )

        # 3. Physical Bounds Clamping
        clean_v, clamped_v = self.clamp_velocity(raw_delta_v)
        clean_yaw, clamped_yaw = self.clamp_yaw_rate(raw_delta_yaw)
        is_clamped = clamped_v or clamped_yaw

        return GuardedResidual(
            delta_velocity_ms=clean_v,
            delta_yaw_rate_rads=clean_yaw,
            is_clamped=is_clamped,
            is_fallback=False,
            status="CLAMPED" if is_clamped else "APPLIED"
        )
