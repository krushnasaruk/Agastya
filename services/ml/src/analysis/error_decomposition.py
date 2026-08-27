"""
Physical Error Decomposition Module for Project AGASTYA (Objective 4).
Analyzes root-cause physical contributors to classical dead-reckoning errors:
tire rolling-radius variation, wheel slip, gyro bias, and cornering dynamics.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


@dataclass
class PhysicalErrorBreakdown:
    estimated_tire_scale_factor_error_pct: float    # [OBSERVED: mean(v_ref / v_wheel) - 1.0]
    slip_event_count: int                           # [OBSERVED: Count of epochs where slip threshold exceeded]
    slip_mean_velocity_discrepancy_ms: float        # [OBSERVED: Average speed discrepancy during slip]
    gyro_estimated_bias_rads: float                 # [OBSERVED: Mean stationary yaw rate offset]
    cornering_velocity_error_correlation: float     # [CORRELATED: Correlation between lateral accel/curvature and velocity error]
    timing_jitter_std_ms: float                     # [OBSERVED: Sampling interval standard deviation]
    findings_summary: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PhysicalErrorDecomposer:
    """
    Decomposes observed dead-reckoning residual errors into underlying physical sources.
    """
    @classmethod
    def decompose(
        cls,
        time_sec: np.ndarray,
        dt_sec: np.ndarray,
        v_wheel_rear_ms: np.ndarray,
        yaw_rate_can_rads: np.ndarray,
        accel_x_ms2: np.ndarray,
        v_ref_ms: np.ndarray,
        v_classical_ms: np.ndarray,
        heading_classical_rad: np.ndarray,
        heading_ref_rad: np.ndarray
    ) -> PhysicalErrorBreakdown:
        """
        Compute physical error breakdown metrics.
        """
        valid_moving = (v_wheel_rear_ms > 1.0) & (~np.isnan(v_ref_ms)) & (v_ref_ms > 1.0)
        
        # 1. Tire Rolling-Radius Scale Factor Error
        if np.sum(valid_moving) > 10:
            scale_ratios = v_ref_ms[valid_moving] / v_wheel_rear_ms[valid_moving]
            mean_scale_err_pct = float((np.mean(scale_ratios) - 1.0) * 100.0)
        else:
            mean_scale_err_pct = 0.0

        # 2. Wheel Slip Analysis
        v_diff = np.abs(v_ref_ms - v_classical_ms)
        slip_mask = (v_diff > 0.5) & valid_moving
        slip_count = int(np.sum(slip_mask))
        slip_mean_disc = float(np.mean(v_diff[slip_mask])) if slip_count > 0 else 0.0

        # 3. Stationary Gyro Bias
        stat_mask = v_wheel_rear_ms < 0.08
        if np.sum(stat_mask) > 5:
            gyro_bias = float(np.mean(yaw_rate_can_rads[stat_mask]))
        else:
            gyro_bias = 0.0

        # 4. Cornering vs Velocity Error Correlation
        curvature = np.abs(yaw_rate_can_rads) / np.maximum(v_classical_ms, 0.1)
        if np.sum(valid_moving) > 10 and np.std(curvature[valid_moving]) > 1e-6 and np.std(v_diff[valid_moving]) > 1e-6:
            r_corner = float(np.corrcoef(curvature[valid_moving], v_diff[valid_moving])[0, 1])
        else:
            r_corner = 0.0

        # 5. Timing Jitter
        jitter_std_ms = float(np.std(dt_sec) * 1000.0)

        findings = [
            f"Tire Rolling Radius Variation: Scale factor offset is {mean_scale_err_pct:+.3f}% [OBSERVED]",
            f"Transient Wheel Slip: {slip_count} epochs observed with mean velocity discrepancy of {slip_mean_disc:.3f} m/s [OBSERVED]",
            f"Chassis Gyroscope Bias: Estimated stationary drift is {gyro_bias:+.5f} rad/s [OBSERVED]",
            f"Cornering Dynamic Coupling: Curvature correlation with speed error is r = {r_corner:+.3f} [CORRELATED]",
            f"Sampling Loop Jitter: Timestep standard deviation is {jitter_std_ms:.2f} ms [OBSERVED]"
        ]

        return PhysicalErrorBreakdown(
            estimated_tire_scale_factor_error_pct=round(mean_scale_err_pct, 4),
            slip_event_count=slip_count,
            slip_mean_velocity_discrepancy_ms=round(slip_mean_disc, 4),
            gyro_estimated_bias_rads=round(gyro_bias, 6),
            cornering_velocity_error_correlation=round(r_corner, 4),
            timing_jitter_std_ms=round(jitter_std_ms, 2),
            findings_summary=findings
        )
