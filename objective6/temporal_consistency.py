"""
Causal Temporal Consistency Monitor for Objective 6.
Evaluates step-to-step jump stability of residual predictions without any future lookahead.
"""

from typing import Dict, Any, Optional
import numpy as np


class TemporalConsistencyMonitor:
    """
    Tracks causal prediction stability between step k and step k-1.
    Rejects sudden non-physical high-frequency fluctuations.
    """
    def __init__(
        self,
        max_velocity_jump_ms: float = 0.60,
        max_yaw_jump_rads: float = 0.25,
        ema_alpha: float = 0.30
    ):
        self.max_v_jump = max_velocity_jump_ms
        self.max_w_jump = max_yaw_jump_rads
        self.ema_alpha = ema_alpha

        self.prev_delta_v: Optional[float] = None
        self.prev_delta_w: Optional[float] = None
        self.smoothed_delta_v: float = 0.0
        self.smoothed_delta_w: float = 0.0
        self.step_count: int = 0

    def reset(self) -> None:
        """Reset internal causal state at the start of a sequence."""
        self.prev_delta_v = None
        self.prev_delta_w = None
        self.smoothed_delta_v = 0.0
        self.smoothed_delta_w = 0.0
        self.step_count = 0

    def evaluate_step(
        self,
        current_delta_v: float,
        current_delta_w: float = 0.0
    ) -> Dict[str, Any]:
        """
        Evaluate temporal jump against previous step.
        """
        if np.isnan(current_delta_v) or np.isinf(current_delta_v):
            return {
                "is_consistent": False,
                "velocity_jump_ms": 999.0,
                "yaw_jump_rads": 999.0,
                "reason": "NAN_OR_INF_INPUT"
            }

        # First step is trivially consistent
        if self.prev_delta_v is None:
            self.prev_delta_v = current_delta_v
            self.prev_delta_w = current_delta_w
            self.smoothed_delta_v = current_delta_v
            self.smoothed_delta_w = current_delta_w
            self.step_count = 1
            return {
                "is_consistent": True,
                "velocity_jump_ms": 0.0,
                "yaw_jump_rads": 0.0,
                "reason": "INITIAL_STEP"
            }

        v_jump = float(abs(current_delta_v - self.prev_delta_v))
        w_jump = float(abs(current_delta_w - self.prev_delta_w))

        is_v_ok = (v_jump <= self.max_v_jump)
        is_w_ok = (w_jump <= self.max_w_jump)
        is_consistent = is_v_ok and is_w_ok

        # Update previous causal state
        self.prev_delta_v = current_delta_v
        self.prev_delta_w = current_delta_w
        self.smoothed_delta_v = self.ema_alpha * current_delta_v + (1.0 - self.ema_alpha) * self.smoothed_delta_v
        self.smoothed_delta_w = self.ema_alpha * current_delta_w + (1.0 - self.ema_alpha) * self.smoothed_delta_w
        self.step_count += 1

        reason = "CONSISTENT"
        if not is_v_ok:
            reason = "VELOCITY_JUMP_EXCEEDED"
        elif not is_w_ok:
            reason = "YAW_JUMP_EXCEEDED"

        return {
            "is_consistent": is_consistent,
            "velocity_jump_ms": round(v_jump, 5),
            "yaw_jump_rads": round(w_jump, 5),
            "smoothed_delta_v": round(self.smoothed_delta_v, 5),
            "smoothed_delta_w": round(self.smoothed_delta_w, 5),
            "reason": reason
        }
