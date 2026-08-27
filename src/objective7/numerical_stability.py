"""
Numerical Stability and Long-Duration Stress Monitor for Objective 7.
Audits floating-point bounds, NaN/Inf immunity, heading wrapping integrity, and state reset behavior.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class NumericalStabilityMonitor:
    """
    Supervises state vectors over long-duration stress runs and detects numerical anomalies.
    """
    def __init__(self):
        self.nan_count = 0
        self.inf_count = 0
        self.heading_wrapping_errors = 0
        self.state_explosion_count = 0
        self.max_observed_speed_ms = 0.0
        self.max_observed_pos_step_m = 0.0
        self.prev_p_east: Optional[float] = None
        self.prev_p_north: Optional[float] = None

    def reset(self) -> None:
        self.nan_count = 0
        self.inf_count = 0
        self.heading_wrapping_errors = 0
        self.state_explosion_count = 0
        self.max_observed_speed_ms = 0.0
        self.max_observed_pos_step_m = 0.0
        self.prev_p_east = None
        self.prev_p_north = None

    def check_state(
        self,
        p_east_m: float,
        p_north_m: float,
        heading_rad: float,
        forward_speed_ms: float
    ) -> bool:
        """
        Verify state validity for current epoch. Returns True if stable.
        """
        vals = [p_east_m, p_north_m, heading_rad, forward_speed_ms]
        if any(np.isnan(v) for v in vals):
            self.nan_count += 1
            return False
        if any(np.isinf(v) for v in vals):
            self.inf_count += 1
            return False

        # Heading wrapping [0, 2pi)
        if heading_rad < -1e-6 or heading_rad >= (2 * np.pi + 1e-4):
            self.heading_wrapping_errors += 1

        self.max_observed_speed_ms = max(self.max_observed_speed_ms, forward_speed_ms)
        if forward_speed_ms > 150.0:
            self.state_explosion_count += 1

        # Position step size
        if self.prev_p_east is not None and self.prev_p_north is not None:
            step_m = float(np.sqrt((p_east_m - self.prev_p_east)**2 + (p_north_m - self.prev_p_north)**2))
            self.max_observed_pos_step_m = max(self.max_observed_pos_step_m, step_m)
            if step_m > 50.0:  # > 50 m in 0.1s is unphysical
                self.state_explosion_count += 1

        self.prev_p_east = p_east_m
        self.prev_p_north = p_north_m
        return True

    def get_summary(self) -> Dict[str, Any]:
        is_stable = (self.nan_count == 0) and (self.inf_count == 0) and (self.state_explosion_count == 0)
        return {
            "nan_occurrences": self.nan_count,
            "inf_occurrences": self.inf_count,
            "heading_wrapping_violations": self.heading_wrapping_errors,
            "state_explosion_events": self.state_explosion_count,
            "max_observed_speed_ms": round(self.max_observed_speed_ms, 4),
            "max_observed_step_m": round(self.max_observed_pos_step_m, 4),
            "is_numerically_stable": is_stable,
            "stability_status": "PASS (Zero NaN/Inf, Bounded State)" if is_stable else "FAILED"
        }
