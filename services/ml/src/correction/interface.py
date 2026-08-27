"""
AI Correction Interface & Safety Fallback Architecture for Project AGASTYA (Objective 4).
Defines strict input/output schemas, uncertainty representation, physical bounding gates,
and graceful degradation to the deterministic classical physics engine.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


@dataclass
class AICorrectionInput:
    time_sec: float
    dt_sec: float
    feature_window: np.ndarray        # Shape [W, D] causal feature tensor
    classical_forward_speed_ms: float
    classical_yaw_rate_rads: float
    is_sensor_valid: bool
    is_stationary: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_sec": self.time_sec,
            "dt_sec": self.dt_sec,
            "classical_forward_speed_ms": self.classical_forward_speed_ms,
            "classical_yaw_rate_rads": self.classical_yaw_rate_rads,
            "is_sensor_valid": self.is_sensor_valid,
            "is_stationary": self.is_stationary,
            "window_shape": list(self.feature_window.shape)
        }


@dataclass
class AICorrectionOutput:
    delta_velocity_ms: float                 # Predicted forward speed correction
    delta_yaw_rate_rads: float              # Predicted yaw rate correction
    velocity_uncertainty_variance: float    # sigma_v^2 confidence estimate
    yaw_uncertainty_variance: float         # sigma_omega^2 confidence estimate
    correction_applied: bool
    status: str                              # 'APPLIED', 'CLAMPED', 'FALLBACK_LOW_CONFIDENCE', 'FALLBACK_SENSOR_DEGRADED'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AICorrectionSafetyGuard:
    """
    Guards navigation state against AI hallucination, unphysical spikes, or low-confidence outputs.
    Enforces graceful, deterministic fallback to the classical physics engine.
    """
    def __init__(
        self,
        max_velocity_correction_ms: float = 3.0,
        max_yaw_rate_correction_rads: float = 0.50,
        max_acceptable_velocity_variance: float = 1.00,
        max_acceptable_yaw_variance: float = 0.25
    ):
        self.max_velocity_correction_ms = max_velocity_correction_ms
        self.max_yaw_rate_correction_rads = max_yaw_rate_correction_rads
        self.max_acceptable_velocity_variance = max_acceptable_velocity_variance
        self.max_acceptable_yaw_variance = max_acceptable_yaw_variance

    def sanitize_correction(
        self,
        raw_delta_v: float,
        raw_delta_yaw: float,
        var_v: float,
        var_yaw: float,
        is_sensor_valid: bool = True,
        is_stationary: bool = False
    ) -> AICorrectionOutput:
        """
        Sanitize and bounds-check candidate AI corrections.
        """
        # 1. Sensor Degradation or Stationary Fallback
        if not is_sensor_valid or is_stationary:
            return AICorrectionOutput(
                delta_velocity_ms=0.0,
                delta_yaw_rate_rads=0.0,
                velocity_uncertainty_variance=float(var_v),
                yaw_uncertainty_variance=float(var_yaw),
                correction_applied=False,
                status="FALLBACK_SENSOR_DEGRADED_OR_STATIONARY"
            )

        # 2. High Uncertainty / Low Confidence Fallback
        if var_v > self.max_acceptable_velocity_variance or var_yaw > self.max_acceptable_yaw_variance:
            return AICorrectionOutput(
                delta_velocity_ms=0.0,
                delta_yaw_rate_rads=0.0,
                velocity_uncertainty_variance=float(var_v),
                yaw_uncertainty_variance=float(var_yaw),
                correction_applied=False,
                status="FALLBACK_LOW_CONFIDENCE"
            )

        # 3. Physical Bound Clamping
        clamped = False
        clean_dv = raw_delta_v
        if abs(raw_delta_v) > self.max_velocity_correction_ms:
            clean_dv = float(np.clip(raw_delta_v, -self.max_velocity_correction_ms, self.max_velocity_correction_ms))
            clamped = True

        clean_dyaw = raw_delta_yaw
        if abs(raw_delta_yaw) > self.max_yaw_rate_correction_rads:
            clean_dyaw = float(np.clip(raw_delta_yaw, -self.max_yaw_rate_correction_rads, self.max_yaw_rate_correction_rads))
            clamped = True

        return AICorrectionOutput(
            delta_velocity_ms=clean_dv,
            delta_yaw_rate_rads=clean_dyaw,
            velocity_uncertainty_variance=float(var_v),
            yaw_uncertainty_variance=float(var_yaw),
            correction_applied=True,
            status="CLAMPED" if clamped else "APPLIED"
        )
