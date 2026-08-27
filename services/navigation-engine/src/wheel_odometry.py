"""
Wheel Odometry & Kinematic Velocity Estimation Module for Project AGASTYA.
Computes forward speed, differential kinematic yaw rate, and stationary states
from 4-wheel speed measurements with causal outlier and dropout resilience.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional, List
import numpy as np


@dataclass
class WheelSpeedEstimate:
    forward_speed_ms: float
    kinematic_yaw_rate_rads: float
    is_stationary: bool
    slip_detected: bool
    used_channels: List[str]
    status: str = "OK"


class WheelOdometryEstimator:
    """
    Classical Wheel Odometry Estimator for 2-track ground vehicles.
    
    Parameters:
      - track_width_m: Rear track width in meters [PROVISIONAL: Ford Fiesta Mk7 = 1.47 m]
      - zero_speed_threshold_ms: Velocity threshold below which vehicle is stationary [PROVISIONAL = 0.08 m/s]
      - max_plausible_speed_ms: Velocity upper bound [PROVISIONAL = 70.0 m/s]
      - slip_diff_threshold_ms: Wheel speed discrepancy threshold for slip [PROVISIONAL = 2.5 m/s]
    """
    def __init__(
        self,
        track_width_m: float = 1.47,
        zero_speed_threshold_ms: float = 0.08,
        max_plausible_speed_ms: float = 70.0,
        slip_diff_threshold_ms: float = 2.5
    ):
        self.track_width_m = track_width_m
        self.zero_speed_threshold_ms = zero_speed_threshold_ms
        self.max_plausible_speed_ms = max_plausible_speed_ms
        self.slip_diff_threshold_ms = slip_diff_threshold_ms

    def get_parameter_metadata(self) -> Dict[str, Any]:
        return {
            "track_width_m": {
                "value": self.track_width_m,
                "unit": "meters",
                "status": "PROVISIONAL / CONFIGURATION REQUIRED",
                "description": "Nominal Ford Fiesta rear track width for differential odometry"
            },
            "zero_speed_threshold_ms": {
                "value": self.zero_speed_threshold_ms,
                "unit": "m/s",
                "status": "PROVISIONAL",
                "description": "Stationary detection threshold for Zero-Velocity Updates (ZUPT)"
            },
            "slip_diff_threshold_ms": {
                "value": self.slip_diff_threshold_ms,
                "unit": "m/s",
                "status": "PROVISIONAL",
                "description": "Discrepancy threshold to detect tire slip / spinning"
            }
        }

    def estimate_speed(
        self,
        v_fl: Optional[float] = None,
        v_fr: Optional[float] = None,
        v_rl: Optional[float] = None,
        v_rr: Optional[float] = None
    ) -> WheelSpeedEstimate:
        """
        Estimate forward speed and kinematic yaw rate from available wheel speeds.
        Causally handles missing, NaN, or anomalous wheel channels.
        """
        valid_rear = []
        valid_front = []
        used = []

        # 1. Validate Rear Wheels (Unsteered - Preferred for Odometry)
        if v_rl is not None and not np.isnan(v_rl) and 0.0 <= v_rl <= self.max_plausible_speed_ms:
            valid_rear.append(("RL", float(v_rl)))
        if v_rr is not None and not np.isnan(v_rr) and 0.0 <= v_rr <= self.max_plausible_speed_ms:
            valid_rear.append(("RR", float(v_rr)))

        # 2. Validate Front Wheels (Steered - Fallback)
        if v_fl is not None and not np.isnan(v_fl) and 0.0 <= v_fl <= self.max_plausible_speed_ms:
            valid_front.append(("FL", float(v_fl)))
        if v_fr is not None and not np.isnan(v_fr) and 0.0 <= v_fr <= self.max_plausible_speed_ms:
            valid_front.append(("FR", float(v_fr)))

        # Kinematic yaw rate from rear wheels: omega_z = (v_RR - v_RL) / W_track
        kinematic_yaw = 0.0
        slip = False

        if len(valid_rear) == 2:
            v_rl_val = valid_rear[0][1]
            v_rr_val = valid_rear[1][1]
            fwd_speed = 0.5 * (v_rl_val + v_rr_val)
            kinematic_yaw = (v_rr_val - v_rl_val) / max(self.track_width_m, 0.1)
            used = ["RL", "RR"]

            # Check wheel discrepancy slip
            if abs(v_rr_val - v_rl_val) > self.slip_diff_threshold_ms and fwd_speed > 2.0:
                slip = True

        elif len(valid_rear) == 1:
            # Single rear wheel available
            fwd_speed = valid_rear[0][1]
            used = [valid_rear[0][0]]
        elif len(valid_front) >= 1:
            # Fallback to front wheels
            fwd_speed = float(np.mean([x[1] for x in valid_front]))
            used = [x[0] for x in valid_front]
        else:
            # Complete wheel dropout
            fwd_speed = 0.0
            used = []

        # Zero-velocity check
        is_stat = fwd_speed < self.zero_speed_threshold_ms
        if is_stat:
            fwd_speed = 0.0
            kinematic_yaw = 0.0

        return WheelSpeedEstimate(
            forward_speed_ms=float(fwd_speed),
            kinematic_yaw_rate_rads=float(kinematic_yaw),
            is_stationary=is_stat,
            slip_detected=slip,
            used_channels=used,
            status="OK" if len(used) > 0 else "DROPOUT"
        )
