"""
Numerical Stability Monitor for Objective 8.
Tracks and reports NaNs, Infs, state explosions, and heading angle wrapping anomalies.
"""

from typing import Dict, Any, List
import numpy as np


class NumericalStabilityMonitor:
    """
    Supervises state boundaries and numerical sanity during long-duration runs.
    """
    def __init__(
        self,
        max_speed_ms: float = 70.0,
        max_position_drift_m: float = 1e6,
        max_pos_bound_m: Optional[float] = None,
        max_speed_bound_ms: Optional[float] = None
    ):
        self.max_speed_ms = max_speed_bound_ms if max_speed_bound_ms is not None else max_speed_ms
        self.max_position_drift_m = max_pos_bound_m if max_pos_bound_m is not None else max_position_drift_m

        self.nan_count = 0
        self.inf_count = 0
        self.speed_explosions = 0
        self.position_explosions = 0
        self.heading_wrapping_errors = 0
        self.total_checked_states = 0

    def record_violation(self, reason: str = "Unknown") -> None:
        """Records arbitrary numerical violation."""
        self.nan_count += 1

    def check_state(
        self,
        p_east_m: float,
        p_north_m: float,
        heading_rad: float,
        forward_speed_ms: float
    ) -> bool:
        """
        Returns True if state is numerically healthy, False if any anomaly is detected.
        """
        self.total_checked_states += 1
        is_healthy = True

        # Check NaN
        if np.isnan(p_east_m) or np.isnan(p_north_m) or np.isnan(heading_rad) or np.isnan(forward_speed_ms):
            self.nan_count += 1
            is_healthy = False

        # Check Inf
        if np.isinf(p_east_m) or np.isinf(p_north_m) or np.isinf(heading_rad) or np.isinf(forward_speed_ms):
            self.inf_count += 1
            is_healthy = False

        # Check speed explosion
        if not np.isnan(forward_speed_ms) and (abs(forward_speed_ms) > self.max_speed_ms):
            self.speed_explosions += 1
            is_healthy = False

        # Check position explosion
        if not np.isnan(p_east_m) and not np.isnan(p_north_m):
            dist_sq = p_east_m**2 + p_north_m**2
            if dist_sq > (self.max_position_drift_m ** 2):
                self.position_explosions += 1
                is_healthy = False

        # Check heading bounds [-pi, 2pi]
        if not np.isnan(heading_rad) and (heading_rad < -np.pi or heading_rad > 2 * np.pi + 0.1):
            self.heading_wrapping_errors += 1
            is_healthy = False

        return is_healthy

    def get_summary(self) -> Dict[str, Any]:
        has_anomalies = bool(
            self.nan_count > 0 or
            self.inf_count > 0 or
            self.speed_explosions > 0 or
            self.position_explosions > 0 or
            self.heading_wrapping_errors > 0
        )
        return {
            "total_checked_states": self.total_checked_states,
            "nan_occurrences": self.nan_count,
            "inf_occurrences": self.inf_count,
            "speed_explosions": self.speed_explosions,
            "position_explosions": self.position_explosions,
            "heading_wrapping_errors": self.heading_wrapping_errors,
            "is_numerically_stable": not has_anomalies,
            "status": "PASS" if not has_anomalies else "NUMERICAL_INSTABILITY_DETECTED"
        }
